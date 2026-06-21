"""Window-spec generator for OVER clauses.

Role: builds the contents of an OVER (...) clause attached to a
windowed function call. Called from `gen_expr`'s window branch
(`kind == "window"`) and consumed by the printer to render the
spec inline OR — after `hoist_named_windows` — as a `WINDOW name AS
(...)` reference. Windows are only valid in SELECT-list and ORDER BY
positions; `gen_select` enforces this with the `allow_window` flag,
which is False everywhere else.

One public entry point: `gen_window_spec(ctx)` builds a WindowSpec
with optional PARTITION BY, ORDER BY, and frame clause.

Window specs are intentionally simple:

  * PARTITION BY contents are column refs only — picked from
    `ctx.scope.visible_columns()`. Arbitrary expressions are
    syntactically valid but rare in real SQL; complex window
    specs add visual clutter without exercising new generator
    paths.

  * ORDER BY in the spec is also column refs only, with random
    ASC/DESC.

  * Frame clauses are ROWS-only with BETWEEN bounds. Subset chosen
    for two reasons: ROWS doesn't require ORDER BY (RANGE/GROUPS
    do, for offset bounds), and ROWS-with-integer-offsets is
    universally valid regardless of the ORDER BY column types.
    EXCLUDE clauses (CURRENT ROW / GROUP / TIES / NO OTHERS) are
    deferred — small additional complexity, low marginal value.

The probability gates (`p_partition_by`, `p_order_by_in_window`,
`p_window_frame`) fire independently — all can be False, producing
the empty `OVER ()` form which PG accepts (entire result set as
one partition).
"""
from __future__ import annotations

from dataclasses import replace

from ..ast import (
    BinaryOp, Cast, ColumnRef, Expr, FrameBound, FrameClause, FuncCall,
    Literal, NamedWindow, OrderByItem, Select, UnaryOp, WindowRef,
    WindowSpec,
)
from ..context import GenContext
from ..types import FLOAT8, INT4, INT8, NUMERIC, PgType


# Frame bound choices, partitioned by their valid position in the
# BETWEEN range. PG requires the start to come "before" the end in
# this implicit ordering: UNBOUNDED PRECEDING < N PRECEDING < CURRENT
# ROW < N FOLLOWING < UNBOUNDED FOLLOWING. By restricting `_START_KINDS`
# and `_END_KINDS` to non-overlapping subsets of valid choices, every
# (start, end) combo we generate is automatically valid — no need for
# a post-pick ordering check.
_START_KINDS: tuple[str, ...] = (
    "unbounded_preceding", "preceding", "current_row",
)
_END_KINDS: tuple[str, ...] = (
    "current_row", "following", "unbounded_following",
)

# Small integer offsets for `N PRECEDING` / `N FOLLOWING`. Kept small
# so generated frames look realistic (centered moving averages with
# huge offsets are rare in practice).
_OFFSETS: tuple[int, ...] = (1, 2, 3, 5, 10)

# EXCLUDE clause body choices. PG accepts these four; "NO OTHERS" is
# the default (semantically same as omitting the clause), but we
# generate it sometimes for grammar coverage.
_EXCLUDE_KINDS: tuple[str, ...] = (
    "CURRENT ROW", "GROUP", "TIES", "NO OTHERS",
)

# Probability of attaching an EXCLUDE clause to a generated frame.
# Most real frames omit EXCLUDE entirely; biased moderate-low so
# the path gets exercised without dominating output.
_P_FRAME_EXCLUDE: float = 0.25

# Numeric types eligible to back a RANGE-with-offset frame. PG allows
# any type with `+`/`-` against the offset's type, but our generator
# only produces integer offsets via _OFFSETS, so these are the safe
# ORDER BY column types. Adding INTERVAL+TIMESTAMPTZ would require
# generating INTERVAL offsets — deferred.
_RANGE_OFFSET_NUMERIC_TYPES: frozenset[PgType] = frozenset({INT4, INT8, NUMERIC, FLOAT8})

# Frame-bound subsets without offset kinds. Used when picking RANGE
# without offsets (universally valid) vs RANGE with offsets (needs
# numeric ORDER BY).
_START_KINDS_NO_OFFSET: tuple[str, ...] = (
    "unbounded_preceding", "current_row",
)
_END_KINDS_NO_OFFSET: tuple[str, ...] = (
    "current_row", "unbounded_following",
)


def gen_window_spec(ctx: GenContext) -> WindowSpec:
    """Generate a WindowSpec — partition_by + order_by + optional
    frame clause.

    All three sections are independent dice rolls; any combination
    can be empty, yielding `OVER ()` (a single partition over the
    entire result set, which PG accepts and is occasionally
    semantically useful for `count(*) OVER ()`-style "total row
    count alongside each row" patterns).

    Column refs are drawn from the current scope's visible columns
    — same pool as ordinary expression generation. If the scope
    has no visible columns (a degenerate case during testing),
    we fall back to empty `OVER ()` for safety.
    """
    cfg = ctx.config
    rng = ctx.rng

    visible = ctx.scope.visible_columns()
    if not visible:
        # Defensive: real production calls always have visible
        # columns by the time we're generating SELECT-list items,
        # but tests building GenContext directly might not.
        return WindowSpec()

    partition_by: tuple[ColumnRef, ...] = ()
    if rng.random() < cfg.p_partition_by:
        n = min(
            rng.randint(1, cfg.max_partition_by_items),
            len(visible),
        )
        bindings = rng.sample(visible, n)
        partition_by = tuple(
            ColumnRef(b.type, b.table_alias, b.column)
            for b in bindings
        )

    order_by: tuple[OrderByItem, ...] = ()
    if rng.random() < cfg.p_order_by_in_window:
        n = min(
            rng.randint(1, cfg.max_order_by_in_window_items),
            len(visible),
        )
        bindings = rng.sample(visible, n)
        order_by = tuple(
            OrderByItem(
                expr=ColumnRef(b.type, b.table_alias, b.column),
                direction=rng.choice(("ASC", "DESC")),
            )
            for b in bindings
        )

    frame = (
        _gen_frame(ctx, order_by)
        if rng.random() < cfg.p_window_frame
        else None
    )

    return WindowSpec(
        partition_by=partition_by, order_by=order_by, frame=frame,
    )


# CONSTRAINT: frame syntax depends on the presence and type of the
# parent spec's ORDER BY. This isn't optional — PG rejects mismatched
# combos at parse-analysis. The (unit, allow_offset_bounds) option
# list built inside `_gen_frame` is the explicit encoding of those
# rules; expanding it later (INTERVAL offsets for TIMESTAMPTZ, etc.)
# means also expanding `_RANGE_OFFSET_NUMERIC_TYPES` consistently.
def _gen_frame(
    ctx: GenContext,
    order_by: tuple[OrderByItem, ...],
) -> FrameClause:
    """Build a frame clause whose unit and bounds are guaranteed
    valid given the parent WindowSpec's `order_by`.

    PG's frame-validity rules (verified empirically against PG 17):

      * **ROWS**: any bounds, any ORDER BY situation. Universal.
      * **RANGE without offset bounds**: any ORDER BY (including
        none). Just identifies peer rows by ORDER BY value.
      * **RANGE with offset bounds**: needs EXACTLY ONE ORDER BY
        column whose type supports `+`/`-` with the offset type.
        Our offsets are integers, so that one column must be
        numeric (int4/int8/numeric/float8). Multi-column ORDER BY
        is invalid even when the first column is numeric.
      * **GROUPS**: requires ORDER BY (any type, any number of
        columns). Bounds with integer offsets always OK.

    Strategy: build a list of valid `(unit, allow_offset_bounds)`
    options based on the spec's order_by, then pick uniformly from
    that list. Bounds are then drawn from offset-allowed or
    no-offset-only kind pools to match.

    EXCLUDE clause appended at probability _P_FRAME_EXCLUDE; choice
    universal across all four bodies.
    """
    rng = ctx.rng

    # Build the option list: (unit, allow_offset_bounds).
    has_order = bool(order_by)
    first_order_type = order_by[0].expr.pg_type if has_order else None
    # RANGE with offsets needs SINGLE numeric ORDER BY column.
    # GROUPS with offsets only needs ORDER BY (any column count, any
    # type); the offset is a peer-group count, not an arithmetic delta.
    range_offset_ok = (
        has_order
        and len(order_by) == 1
        and first_order_type in _RANGE_OFFSET_NUMERIC_TYPES
    )

    options: list[tuple[str, bool]] = [
        ("ROWS", True),                 # always valid
        ("RANGE", False),               # always valid (no offsets)
    ]
    if has_order:
        options.append(("GROUPS", True))  # GROUPS requires ORDER BY
    if range_offset_ok:
        options.append(("RANGE", True))

    unit, allow_offset = rng.choice(options)
    start_kinds = _START_KINDS if allow_offset else _START_KINDS_NO_OFFSET
    end_kinds = _END_KINDS if allow_offset else _END_KINDS_NO_OFFSET
    start = _gen_bound(rng, kinds=start_kinds)
    end = _gen_bound(rng, kinds=end_kinds)
    exclude = (
        rng.choice(_EXCLUDE_KINDS)
        if rng.random() < _P_FRAME_EXCLUDE
        else None
    )
    return FrameClause(unit=unit, start=start, end=end, exclude=exclude)


def _gen_bound(rng, *, kinds: tuple[str, ...]) -> FrameBound:
    """Construct a FrameBound with a kind drawn from `kinds`. Adds an
    integer-literal offset for the preceding/following kinds (which
    require one) and leaves it None for the unbounded/current_row
    kinds (which forbid one). The post-init invariant on FrameBound
    enforces this match — passing a wrong combo would error at
    construction, surfacing the bug immediately."""
    kind = rng.choice(kinds)
    if kind in ("preceding", "following"):
        return FrameBound(kind=kind, offset=Literal(INT4, rng.choice(_OFFSETS)))
    return FrameBound(kind=kind)


# ===========================================================================
# Named-window hoisting (post-pass deduplication)
# ===========================================================================
#
# When the SELECT-list (or HAVING) generation produces multiple windowed
# aggregates with structurally-identical WindowSpecs, hoisting the
# common spec into a WINDOW clause and replacing inline OVER (...) with
# OVER name produces the canonical PG idiom for spec deduplication.
# Generators don't try to produce duplicates intentionally; this catches
# the natural ones that emerge from random window-spec generation
# (more likely with small scopes and biased PARTITION BY weights).
#
# We deliberately don't recurse into Subquery/Exists/InSubquery bodies —
# named windows are scoped to their own SELECT, and inner SELECTs run
# their own hoisting independently.

def _collect_window_specs(expr: Expr) -> list[WindowSpec]:
    """Recursively gather every WindowSpec appearing on a FuncCall.over
    within `expr`. Returns specs in left-to-right traversal order so
    later equality grouping is deterministic.

    Stops at subquery boundaries — the inner SELECT's own hoist pass
    handles those independently."""
    out: list[WindowSpec] = []

    def walk(e: Expr) -> None:
        if isinstance(e, FuncCall):
            if isinstance(e.over, WindowSpec):
                out.append(e.over)
            for a in e.args:
                walk(a)
            if e.filter_ is not None:
                walk(e.filter_)
        elif isinstance(e, BinaryOp):
            walk(e.left)
            walk(e.right)
        elif isinstance(e, UnaryOp):
            walk(e.operand)
        elif isinstance(e, Cast):
            walk(e.expr)
        # Subquery/Exists/InSubquery/Literal/ColumnRef: no descent.

    walk(expr)
    return out


def _replace_window_specs(
    expr: Expr,
    mapping: dict[WindowSpec, str],
) -> Expr:
    """Rebuild `expr` with each WindowSpec found in `mapping` replaced
    by a WindowRef pointing at the mapped name. Same boundary rules
    as _collect_window_specs (no subquery descent)."""
    if isinstance(expr, FuncCall):
        # Widen the local annotation explicitly: expr.over is
        # `Optional[WindowSpec | WindowRef]`, and we may swap a spec
        # for a ref. Without this, mypy infers the narrower type from
        # the initial assignment and rejects the WindowRef branch.
        new_over: WindowSpec | WindowRef | None = expr.over
        if isinstance(expr.over, WindowSpec) and expr.over in mapping:
            new_over = WindowRef(mapping[expr.over])
        new_args = tuple(_replace_window_specs(a, mapping) for a in expr.args)
        new_filter = (
            _replace_window_specs(expr.filter_, mapping)
            if expr.filter_ is not None else None
        )
        return replace(expr, args=new_args, over=new_over, filter_=new_filter)
    if isinstance(expr, BinaryOp):
        return replace(
            expr,
            left=_replace_window_specs(expr.left, mapping),
            right=_replace_window_specs(expr.right, mapping),
        )
    if isinstance(expr, UnaryOp):
        return replace(expr, operand=_replace_window_specs(expr.operand, mapping))
    if isinstance(expr, Cast):
        return replace(expr, expr=_replace_window_specs(expr.expr, mapping))
    return expr


# INVARIANT: this pass is idempotent — running it twice on the same
# Select produces the same Select (after the first pass the surviving
# WindowSpecs each appear once inline at most, so the spec_counts >= 2
# gate never fires the second time). Useful for tests that compose
# multiple post-passes; nothing currently exercises this property but
# it's a free correctness guarantee.
def hoist_named_windows(s: Select) -> Select:
    """Post-pass: dedupe structurally-identical WindowSpecs within a
    Select by hoisting them to a WINDOW clause and replacing inline
    OVER (...) with OVER name.

    Only specs appearing 2+ times in the SELECT-level expressions
    (targets and HAVING — WHERE/ORDER BY can't host windows) get
    hoisted. Single-use specs stay inline; hoisting them would just
    add noise without exercising the dedup mechanic.

    Operates on the SELECT's structure only — does not descend into
    Subquery/Exists/InSubquery bodies. Those have their own hoisting
    via their own gen_select pass.
    """
    # WindowSpec must be a frozen dataclass for this to work — equality
    # and hashing are what cause structurally-identical specs (different
    # objects, same partition_by/order_by/frame) to collapse into the
    # same dict key. If WindowSpec ever loses `frozen=True` or grows a
    # field that breaks structural equality, hoisting silently stops
    # deduplicating.
    spec_counts: dict[WindowSpec, int] = {}
    spec_order: list[WindowSpec] = []

    def tally(spec: WindowSpec) -> None:
        if spec not in spec_counts:
            spec_order.append(spec)
            spec_counts[spec] = 0
        spec_counts[spec] += 1

    for t in s.targets:
        for spec in _collect_window_specs(t.expr):
            tally(spec)
    if s.having is not None:
        for spec in _collect_window_specs(s.having):
            tally(spec)

    # Hoist only specs with multiple occurrences. Names assigned in
    # first-seen order so the WINDOW clause's name list is
    # deterministic given the same RNG state.
    to_hoist = [spec for spec in spec_order if spec_counts[spec] >= 2]
    if not to_hoist:
        return s

    name_for: dict[WindowSpec, str] = {
        spec: f"w{i + 1}" for i, spec in enumerate(to_hoist)
    }
    named_windows = tuple(
        NamedWindow(name=name_for[spec], spec=spec) for spec in to_hoist
    )

    new_targets = tuple(
        replace(t, expr=_replace_window_specs(t.expr, name_for))
        for t in s.targets
    )
    new_having = (
        _replace_window_specs(s.having, name_for)
        if s.having is not None else None
    )

    return replace(
        s,
        targets=new_targets,
        having=new_having,
        windows=named_windows,
    )


__all__ = ["gen_window_spec", "hoist_named_windows"]
