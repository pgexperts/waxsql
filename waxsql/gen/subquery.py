"""Subquery generators.

Role: covers the four SQL surfaces where a SELECT appears as a value
or table source inside another query. Each public entry point hands
back an AST node that gen_expr or gen_select can splice directly into
the surrounding tree.

SCOPE HANDLING is the subtle bit. Correlated subqueries need the
outer scope visible while the inner is being built (so inner
expressions can pick outer columns); derived tables produce a fresh
scope whose *outer* names are projected back as `alias.cN` and whose
*internals* are NOT visible to siblings. `descend_subquery` on the
context is the chokepoint that decides which: passing
correlated=True keeps the outer chain reachable from the child scope;
correlated=False severs it.

Four public entry points. The first three return a complete `Expr`
for use in `gen/expr.py`'s candidate dispatch; the fourth returns a
FromItem for `gen/select.py`'s FROM-clause builder:

  * `gen_scalar_subquery(ctx, target_type, *, correlated)` — returns
    a `Subquery(target_type, inner)`. Inner has a single target of
    `target_type` and `LIMIT 1` for runtime safety.

  * `gen_exists_subquery(ctx, *, correlated)` — returns
    `Exists(BOOL, inner, negated)`. Inner has the canonical
    `SELECT 1 FROM ...` shape.

  * `gen_in_subquery(ctx, *, correlated)` — returns
    `InSubquery(BOOL, lhs, inner, negated)`. The LHS is generated
    BEFORE descending (so it references the outer scope, not the
    inner FROM tables); the inner produces a single column of the
    LHS's type.

  * `gen_derived_table(ctx, alias, *, lateral)` — returns a
    `DerivedTable` FromItem (`[LATERAL ](SELECT ...) AS alias`).
    Inner has 1..N targets aliased c1..cN; with `lateral=True`,
    the same forced-correlation predicate gets injected so LATERAL
    actually exercises its capability.

All four share `_build_subquery_select`, which handles the recurring
pattern: descend, build FROM, build target(s), optionally WHERE,
force correlation if requested.

The "force correlation" path is what gives the `correlated=True`
flag teeth — without it, the inner WHERE might happen to pick an
outer column or might not, and the test suite's
"correlated-references-outer" invariant would be flaky. With it,
every correlated subquery has at least one outer-column reference
in its WHERE.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

from ..ast import (
    BinaryOp, ColumnRef, DerivedTable, Exists, Expr, InSubquery, Literal,
    Select, SelectTarget, Subquery,
)
from ..config import FEATURE_WHERE
from ..context import GenContext
from ..scope import Scope
from ..types import BOOL, INT4, INT8, NUMERIC, PgType, TEXT, TIMESTAMPTZ
from .expr import gen_expr, gen_literal


# Probability of negating an EXISTS / IN subquery. Kept low: most
# real SQL uses the affirmative form; NOT EXISTS / NOT IN appear
# as anti-joins or set differences but less commonly.
_P_NEGATE_EXISTS = 0.3
_P_NEGATE_IN = 0.3

# Type pool for the IN subquery's comparison. Chosen from types where
# `=` is in the catalog so the inner SELECT's column type aligns with
# what the outer LHS expression can produce. Excludes JSONB / arrays
# since the catalog has no `=` op for those.
_IN_COMPARISON_TYPES: tuple[PgType, ...] = (
    INT4, INT8, NUMERIC, TEXT, TIMESTAMPTZ,
)


# ===========================================================================
# Public entry points
# ===========================================================================

def gen_scalar_subquery(
    ctx: GenContext,
    target_type: PgType,
    *,
    correlated: bool,
) -> Subquery:
    """Generate a scalar subquery `(SELECT col FROM ...)` of
    `target_type`. The inner SELECT has exactly one target, and
    `LIMIT 1` to be runtime-safe even though we don't actually run
    the queries (PG accepts multi-row scalar subqueries at parse
    time but errors at runtime if more than one row comes back)."""
    inner = _build_subquery_select(
        ctx,
        correlated=correlated,
        target_type=target_type,
        with_limit_1=True,
    )
    return Subquery(target_type, inner)


def gen_exists_subquery(
    ctx: GenContext,
    *,
    correlated: bool,
) -> Exists:
    """Generate `[NOT ]EXISTS (SELECT 1 FROM ...)`. The constant `1`
    is the canonical EXISTS body — PG ignores the SELECT list at
    runtime, so we don't waste generator effort building elaborate
    expressions there."""
    inner = _build_subquery_select(
        ctx,
        correlated=correlated,
        target_expr=Literal(INT4, 1),
        with_limit_1=False,
    )
    negated = ctx.rng.random() < _P_NEGATE_EXISTS
    return Exists(BOOL, inner, negated=negated)


def gen_derived_table(
    ctx: GenContext,
    alias: str,
    *,
    lateral: bool,
) -> DerivedTable:
    """Generate a derived-table FromItem `[LATERAL ](SELECT ...) AS alias`.

    The inner SELECT has 1..N targets, each with explicit `cN` alias
    so the outer query can reference `<alias>.cN` deterministically
    regardless of what expressions the targets carry. Column count
    drawn from the config knob `max_derived_table_columns` (capped
    by determinism — same draw on same RNG state always picks the
    same count). The single-column case remains the most common
    output by virtue of the small max.

    With `lateral=True`, the inner SELECT can reference the outer
    scope's preceding-sibling aliases. The forced-correlation
    predicate inside `_build_subquery_select` (already used by
    correlated scalar/EXISTS/IN subqueries in milestone 3) guarantees
    at least one outer-column reference appears in the inner WHERE
    — turning "LATERAL capability" into "LATERAL actually exercises
    the capability."

    No LIMIT 1: derived tables are virtual TABLES (multi-row by
    nature), unlike scalar subqueries that need LIMIT 1 for PG's
    single-row runtime constraint.
    """
    rng = ctx.rng
    cfg = ctx.config
    n_cols = rng.randint(1, cfg.max_derived_table_columns)
    target_types = tuple(
        rng.choice(_IN_COMPARISON_TYPES) for _ in range(n_cols)
    )
    inner = _build_subquery_select(
        ctx,
        correlated=lateral,
        target_types=target_types,
        with_limit_1=False,
    )
    # Re-wrap each target with an explicit `cN` alias for stable
    # outer-side column resolution.
    inner_aliased = replace(
        inner,
        targets=tuple(
            SelectTarget(expr=t.expr, alias=f"c{i + 1}")
            for i, t in enumerate(inner.targets)
        ),
    )
    return DerivedTable(inner_aliased, alias, lateral=lateral)


def gen_in_subquery(
    ctx: GenContext,
    *,
    correlated: bool,
) -> InSubquery:
    """Generate `<lhs> [NOT ]IN (SELECT col FROM ...)`. The LHS is
    generated against the OUTER context so it references outer
    columns, not the inner FROM tables. The inner produces a single
    column whose type matches the LHS."""
    # The InSubquery node returns BOOL to the caller, but the LHS and
    # inner column share a different type — the equality-comparable
    # type chosen here. Caller's "target_type=BOOL" gate (in expr.py)
    # is what makes this candidate visible; this local `target_type`
    # is the comparison-element type, not the outer-expression type.
    target_type = ctx.rng.choice(_IN_COMPARISON_TYPES)
    # CRITICAL: build LHS before descending. After descend_subquery
    # the ctx.scope is the inner scope, and LHS column refs would
    # come from inner tables — semantically wrong (LHS is part of
    # the outer query's expression, not the inner subquery).
    # ORDERING DEPENDENCY: this line MUST stay above the
    # _build_subquery_select call below; reordering would silently
    # produce SQL that parses but means something different.
    lhs = gen_expr(ctx, target_type)
    inner = _build_subquery_select(
        ctx,
        correlated=correlated,
        target_type=target_type,
        with_limit_1=False,
    )
    negated = ctx.rng.random() < _P_NEGATE_IN
    return InSubquery(BOOL, lhs, inner, negated=negated)


# ===========================================================================
# Shared subquery body builder
# ===========================================================================

def _build_subquery_select(
    ctx: GenContext,
    *,
    correlated: bool,
    target_type: Optional[PgType] = None,
    target_types: Optional[tuple[PgType, ...]] = None,
    target_expr: Optional[Expr] = None,
    with_limit_1: bool = False,
) -> Select:
    """Construct a SELECT for use as a subquery body.

    Caller provides EXACTLY ONE target spec:
      * `target_type` (singular) — generate a single target of that
        type; convenience for the common single-column case.
      * `target_types` (tuple) — generate len(target_types) targets,
        one of each type. Used by multi-column derived tables / CTEs.
      * `target_expr` — a pre-built single target expression
        (used by EXISTS's constant `SELECT 1`).

    The descent flow:
      1. Capture outer scope (needed by the correlation forcer).
      2. Descend via ctx.descend_subquery — fresh expression depth,
         fresh aggregate flags, child scope (correlated or not).
      3. Build FROM clause; the child scope acquires the inner tables.
      4. Build the target list (generated or supplied).
      5. Optionally generate WHERE (with allow_aggregates=False —
         WHERE forbids aggregates, just like in the outer query).
      6. If correlated, AND-inject an outer-referencing predicate
         into WHERE.
      7. Add LIMIT 1 if requested.
    """
    # Lazy import to break the cycle: gen/expr.py imports the public
    # entry points above; importing _gen_from_clause at module top
    # would close the import loop. Lazy import is the standard Python
    # answer and incurs only a sys.modules dict lookup after first call.
    from .select import _gen_from_clause

    # Capture outer scope BEFORE descending — `descend_subquery`
    # replaces ctx.scope with the child scope, so this is the last
    # chance to keep a handle on the outer bindings. The correlation
    # forcer below needs that handle to pick an outer column for the
    # left-hand side of its injected predicate.
    outer_scope = ctx.scope
    child_ctx = ctx.descend_subquery(correlated=correlated)
    cfg = child_ctx.config
    rng = child_ctx.rng

    from_clause = _gen_from_clause(child_ctx)

    # Resolve the three input forms into a single `target_exprs`
    # tuple. Exactly one of (target_expr, target_type, target_types)
    # must be set; combinations would be ambiguous.
    set_count = sum(
        1 for x in (target_expr, target_type, target_types) if x is not None
    )
    if set_count != 1:
        raise ValueError(
            "_build_subquery_select needs exactly one of target_expr, "
            f"target_type, target_types (got {set_count})"
        )

    if target_expr is not None:
        target_exprs: tuple[Expr, ...] = (target_expr,)
    else:
        # Subquery targets generated with allow_aggregates=False to
        # prevent the "mixed aggregate + non-aggregate-column-ref in
        # the same target expression" bug class. Without GROUP BY,
        # `col - max(other)` triggers implicit-grouping inference;
        # the un-grouped col then errors at PARSE time (42803).
        # Pure-aggregate subqueries lose direct generability here in
        # exchange for PARSE-correctness; can be added back via a
        # dedicated path if needed.
        target_ctx = replace(child_ctx, allow_aggregates=False)
        if target_types is not None:
            types_tuple: tuple[PgType, ...] = target_types
        else:
            # Validated above: exactly-one-of (expr, type, types).
            # When we reach this branch, target_expr is None and
            # target_types is None, so target_type MUST be non-None.
            assert target_type is not None
            types_tuple = (target_type,)
        target_exprs = tuple(gen_expr(target_ctx, t) for t in types_tuple)

    where: Optional[Expr] = None
    if FEATURE_WHERE in cfg.feature_flags and rng.random() < cfg.p_where:
        where_ctx = replace(child_ctx, allow_aggregates=False)
        where = gen_expr(where_ctx, BOOL)

    if correlated:
        where = _force_correlation_predicate(child_ctx, outer_scope, where)

    return Select(
        targets=tuple(SelectTarget(expr=e) for e in target_exprs),
        from_=from_clause,
        where=where,
        limit=Literal(INT4, 1) if with_limit_1 else None,
    )


# ===========================================================================
# Correlation enforcement
# ===========================================================================

def _force_correlation_predicate(
    child_ctx: GenContext,
    outer_scope: Scope,
    existing_where: Optional[Expr],
) -> Optional[Expr]:
    """Inject `outer_col = X` into the inner WHERE, AND-combined with
    any existing WHERE. Guarantees the subquery references the outer
    scope at least once — the test suite's "correlated subqueries
    actually correlate" invariant relies on this.

    Picks an outer column whose type has a usable `=` operator (filters
    out JSONB / arrays etc. that don't have catalog-registered equality).
    The right-hand side is preferentially another column of the same
    type from the inner scope; failing that, a literal of that type
    (still satisfies "references outer" because the outer column is
    on the left).
    """
    rng = child_ctx.rng

    # Outer bindings whose types have a usable `=` in our catalog.
    bool_ops = child_ctx.catalog.binary_ops_returning(BOOL)
    # Set is fine for membership tests; we never iterate it.
    eq_types = {
        o.left for o in bool_ops
        if o.symbol == "=" and o.left == o.right
    }
    outer_candidates = [
        b for b in outer_scope.visible_columns()
        if b.type in eq_types
    ]
    if not outer_candidates:
        # Pathological — outer has no comparable columns. Skip
        # correlation rather than emit invalid SQL.
        return existing_where

    outer = rng.choice(outer_candidates)

    # RHS: prefer an inner column of the same type; fall back to a
    # literal. Either way, the outer column reference on the LEFT is
    # what makes this a correlated subquery.
    #
    # Why filter by `inner_aliases` rather than just calling
    # visible_columns()? In a correlated child scope, visible_columns()
    # returns BOTH inner bindings AND outer bindings (correlation lets
    # outer columns leak in). If we picked an OUTER column for the RHS,
    # the predicate would compare two outer columns — semantically a
    # constant from the inner subquery's POV, which PG might pull out
    # of the subquery as a constant filter, defeating correlation.
    # Restricting to inner aliases keeps the LEFT/RIGHT asymmetry that
    # makes this a real correlated reference.
    inner_aliases = {a for a, _ in child_ctx.scope.aliased_tables()}
    inner_candidates = [
        b for b in child_ctx.scope.visible_columns()
        if b.table_alias in inner_aliases and b.type == outer.type
    ]
    if inner_candidates:
        inner = rng.choice(inner_candidates)
        rhs: Expr = ColumnRef(inner.type, inner.table_alias, inner.column)
    else:
        rhs = gen_literal(rng, outer.type)

    correlation = BinaryOp(
        BOOL, "=",
        ColumnRef(outer.type, outer.table_alias, outer.column),
        rhs,
    )

    if existing_where is None:
        return correlation
    # AND-combine: outer-correlation predicate first, existing where
    # second. Order is just for readability — both are evaluated.
    return BinaryOp(BOOL, "AND", correlation, existing_where)


__all__ = [
    "gen_scalar_subquery",
    "gen_exists_subquery",
    "gen_in_subquery",
    "gen_derived_table",
]
