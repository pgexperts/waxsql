"""SELECT-statement generator.

Role: the integration layer between schema (tables, columns, FKs),
scope (visible bindings), and `gen_expr` (typed expression
production). Every other generator in `gen/` that emits a SELECT —
subquery bodies, CTE bodies, set-op arms — ultimately routes through
`gen_select` or shares its FROM-clause helpers, so the JOIN/scope
rules here are the canonical SELECT-shape definition for the project.

The integration layer: picks tables for the FROM clause, populates the
scope, decides comma-FROM vs explicit JOIN, biases ON conditions
toward FK relationships, generates SELECT-list expressions via
`gen_expr`, and optionally adds WHERE / ORDER BY / LIMIT according
to the complexity dial.

Notable design choices:

  * The caller owns the scope. `gen_select` does NOT create its own
    scope — it adds tables to whatever scope is on `ctx`. For
    top-level use, the caller passes an empty Scope; for the future
    subquery case, the caller pushes a child scope first. Keeping
    scope ownership outside `gen_select` keeps the generator
    composable.

  * JOIN tree is left-deep, with each new table added to scope
    BEFORE its ON condition is generated. This matches PostgreSQL's
    rule that the ON of `tA JOIN tB` can only see tA and tB (not
    later tables), so the generator's notion of "visible" tracks PG's.

  * ON-condition FK biasing looks at FKs in either direction (right
    table → existing tables, OR existing tables → right table). When
    the schema's FK density is non-zero, this dominates over the
    random-BOOL fallback, producing joins that read like real ones.

  * Two top-level shapes — non-aggregating and aggregating (with
    GROUP BY / HAVING) — are picked between by `gen_select` based
    on the FEATURE_AGGREGATE flag and `p_aggregate_query` config.
    Splitting on the shape keeps each path's invariants local: the
    non-agg path forces `allow_aggregates=False` so aggregates can't
    sneak in via WHERE or SELECT items; the agg path constructs
    every SELECT-list non-aggregate as a verbatim GROUP BY entry so
    PG's "must appear in GROUP BY" rule is satisfied by construction.
"""
from __future__ import annotations

import random
from dataclasses import replace

from typing import Optional

from ..ast import (
    BinaryOp, ColumnRef, CteDef, CteRef, Expr, FromItem,
    FuncCall, GroupingSet, JoinExpr, Literal, OrderByItem, Select,
    SelectTarget, TableRef,
)
from ..catalog import FuncKind
from ..config import (
    FEATURE_AGGREGATE, FEATURE_CTE, FEATURE_DERIVED_TABLE, FEATURE_GROUPING_SET,
    FEATURE_HAVING, FEATURE_INNER_JOIN, FEATURE_LATERAL, FEATURE_LEFT_JOIN,
    FEATURE_LIMIT, FEATURE_ORDER_BY, FEATURE_RECURSIVE_CTE, FEATURE_WHERE,
)
from ..context import GenContext
from ..schema import Table
from ..types import (
    BOOL, INT4, INT8, NUMERIC, PgType, TEXT, TIMESTAMPTZ,
)
from .cte import gen_cte_def, gen_recursive_cte_def
from .expr import (
    ORDERED_SET_AGGREGATES, coerce_to_param_type, gen_expr,
    gen_filter_predicate, gen_literal, gen_ordered_set_agg,
    should_emit_count_star, should_emit_filter,
)
from .subquery import gen_derived_table
from .window import hoist_named_windows


# Target-type weights for SELECT-list expressions. Biased toward the
# common types so output reads like normal SQL; rare types (UUID,
# JSONB, INTERVAL, FLOAT8) are excluded here because their literals
# and operators tend to produce visually noisy output, and they're
# still reachable via column refs when the schema includes them.
_SELECT_TYPE_WEIGHTS: tuple[tuple[PgType, float], ...] = (
    (INT4, 2.0),
    (INT8, 2.0),
    (NUMERIC, 1.0),
    (TEXT, 3.0),
    (BOOL, 1.5),
    (TIMESTAMPTZ, 1.0),
)

# Limit values to draw from. Small set keeps output predictable;
# fuzzing rare LIMIT-edge-cases (LIMIT 0, LIMIT NULL, LIMIT ALL) is
# its own future concern.
_LIMIT_VALUES: tuple[int, ...] = (1, 5, 10, 25, 50, 100)


# ===========================================================================
# Public entry point
# ===========================================================================

def gen_select(ctx: GenContext) -> Select:
    """Generate a SELECT statement.

    `ctx.scope` is mutated: tables chosen for the FROM clause are
    added to it. The caller is expected to pass a scope appropriate
    for the use case — empty for a top-level query, or a child scope
    pushed via `ctx.scope.push_subquery(...)` for a future subquery.

    Other parts of `ctx` are read-only.

    Two code paths:

      * **Aggregate query** (with GROUP BY) — chosen with probability
        `p_aggregate_query` when FEATURE_AGGREGATE is unlocked.
      * **Non-aggregate query** — the default shape (no GROUP BY,
        no HAVING); chosen otherwise. The non-agg path explicitly
        disables aggregates on the gen_expr context so they cannot
        sneak in via SELECT items or WHERE.

    A WITH clause may be generated AHEAD of either path (when
    FEATURE_CTE is unlocked AND ctx.allow_with). Each CTE is
    registered in ctx.scope before generating the next, so later
    CTEs can reference earlier ones — and the main query body
    (after the WITH) sees all of them.
    """
    # ORDERING DEPENDENCY: WITH clause comes first — its CTEs must be
    # in `ctx.scope` before _gen_from_clause runs so a base-vs-CTE
    # decision in _make_from_item can find them. Reversing this would
    # silently downgrade every "FROM cte_name" pick to "FROM base_table".
    with_ctes = _gen_with_clause(ctx)

    is_agg = (
        FEATURE_AGGREGATE in ctx.config.feature_flags
        and ctx.rng.random() < ctx.config.p_aggregate_query
    )
    if is_agg:
        s = _gen_aggregate_select(ctx, with_ctes=with_ctes)
    else:
        # Non-aggregating path. Forcing allow_aggregates=False prevents
        # gen_expr from sneaking aggregates into SELECT-list items or
        # WHERE — without GROUP BY they'd be either parse errors or
        # implicit-single-group queries (which we deliberately don't
        # generate from this path).
        s = _gen_non_aggregate_select(
            replace(ctx, allow_aggregates=False),
            with_ctes=with_ctes,
        )
    # Post-pass: hoist any duplicate WindowSpecs into a WINDOW clause.
    # No-op when there are no duplicates (the common case); cheap
    # to run unconditionally.
    return hoist_named_windows(s)


def _gen_with_clause(ctx: GenContext) -> tuple[CteDef, ...]:
    """Generate zero or more CTE definitions for the WITH clause.

    Returns an empty tuple when:
      * FEATURE_CTE is locked (c < 7), OR
      * ctx.allow_with is False (we're inside a subquery), OR
      * no subquery-depth budget remains (each CTE body costs 1), OR
      * the per-call probability roll fails.

    Each generated CTE is registered in ctx.scope BEFORE the next
    is generated, so cte2 can reference cte1 (the natural CTE-to-CTE
    visibility). The same registration makes them visible to the
    main query body that runs after this returns.
    """
    cfg = ctx.config
    rng = ctx.rng

    if (FEATURE_CTE not in cfg.feature_flags
            or not ctx.allow_with
            or ctx.at_subquery_leaf()
            or rng.random() >= cfg.p_with_clause):
        return ()

    n_ctes = rng.randint(1, cfg.max_ctes_per_with)
    defs: list[CteDef] = []
    for _ in range(n_ctes):
        name = f"cte{ctx.cte_counter.take()}"
        # Recursive vs plain CTE — gated on FEATURE_RECURSIVE_CTE
        # AND a probability roll.
        recursive = (
            FEATURE_RECURSIVE_CTE in cfg.feature_flags
            and rng.random() < cfg.p_recursive_when_cte
        )
        if recursive:
            # gen_recursive_cte_def registers the CTE name internally
            # (the recursive arm needs it visible during generation).
            # Don't re-register here.
            cte_def, _columns = gen_recursive_cte_def(ctx, name)
        else:
            cte_def, columns = gen_cte_def(ctx, name)
            # Register in OUTER scope so subsequent CTEs (and the main
            # body) can resolve `name` via lookup_cte / has_visible_ctes.
            ctx.scope.add_cte(name, columns)
        defs.append(cte_def)
    return tuple(defs)


def _gen_non_aggregate_select(
    ctx: GenContext,
    *,
    with_ctes: tuple[CteDef, ...] = (),
) -> Select:
    """The milestone-1 SELECT shape: targets / FROM / WHERE / ORDER BY
    / LIMIT, no GROUP BY, no HAVING. May be prefixed with a WITH
    clause (passed in by gen_select)."""
    cfg = ctx.config
    rng = ctx.rng
    flags = cfg.feature_flags

    from_ = _gen_from_clause(ctx)

    # ---- SELECT list ---------------------------------------------------
    # allow_window=True only here — windows are valid in SELECT
    # list (and in ORDER BY by reuse), forbidden everywhere else.
    select_ctx = replace(ctx, allow_window=True)
    n_targets = rng.randint(1, cfg.max_select_items)
    targets: list[SelectTarget] = []
    for _ in range(n_targets):
        target_type = _pick_select_type(rng)
        e = gen_expr(select_ctx, target_type)
        targets.append(SelectTarget(expr=e))

    # ---- WHERE ---------------------------------------------------------
    where: Expr | None = None
    if FEATURE_WHERE in flags and rng.random() < cfg.p_where:
        # WHERE forbids windows; use the original ctx (allow_window=False).
        where = gen_expr(ctx, BOOL)

    # ---- ORDER BY ------------------------------------------------------
    order_by = _maybe_gen_order_by(ctx, targets)

    # ---- LIMIT ---------------------------------------------------------
    limit = _maybe_gen_limit(ctx)

    return Select(
        targets=tuple(targets),
        from_=from_,
        with_ctes=with_ctes,
        where=where,
        order_by=order_by,
        limit=limit,
    )


def _gen_aggregate_select(
    ctx: GenContext,
    *,
    with_ctes: tuple[CteDef, ...] = (),
) -> Select:
    """An aggregating SELECT with explicit GROUP BY.

    PARSE-tier-correct by construction: every non-aggregate SELECT-list
    item is one of the chosen GROUP BY expressions (reused as the same
    AST object), so PG's "must appear in GROUP BY or be in an
    aggregate" rule is satisfied. HAVING is constructed as
    `aggregate COMP literal` so it always references aggregated state.

    Implicit-single-group aggregates (no GROUP BY, all SELECT items
    aggregates) are deliberately not generated here — they come back
    in a later milestone.
    """
    cfg = ctx.config
    rng = ctx.rng
    flags = cfg.feature_flags

    from_ = _gen_from_clause(ctx)

    # ---- GROUP BY: pick K column refs from the populated scope ---------
    visible = ctx.scope.visible_columns()
    if not visible:
        # Defensive: schema generator always gives every table at least
        # an `id` column, so this can't happen. Fall back rather than
        # crashing if it ever does.
        return _gen_non_aggregate_select(replace(ctx, allow_aggregates=False))

    n_group = rng.randint(1, min(cfg.max_group_by_items, len(visible)))
    chosen_bindings = rng.sample(visible, n_group)
    # The same AST objects get reused as both GROUP BY items and
    # SELECT-list grouped items — that's how the "verbatim in GROUP
    # BY" PARSE-correctness rule is satisfied at the AST level
    # (frozen dataclasses compare structurally, so equality holds).
    grouped_exprs: tuple[Expr, ...] = tuple(
        ColumnRef(b.type, b.table_alias, b.column)
        for b in chosen_bindings
    )

    # Optional grouping-set extension: wrap the column list in
    # ROLLUP, CUBE, or GROUPING SETS. PG accepts mixing constructs
    # within one GROUP BY but the simplest realistic shape is "the
    # whole GROUP BY is one construct"; emit that.
    # Either the original column tuple OR a single-element tuple
    # holding a GroupingSet — the AST field accepts the union.
    group_by_clause: tuple[Expr | GroupingSet, ...] = grouped_exprs
    if (FEATURE_GROUPING_SET in flags
            and rng.random() < cfg.p_grouping_set):
        group_by_clause = (_gen_grouping_set(ctx, grouped_exprs),)

    # ---- SELECT list: mix grouped exprs and aggregates ----------------
    # allow_window=True for this section — windows are valid in
    # aggregating SELECT lists too (e.g., row_number() over partitions
    # of grouped results).
    select_ctx = replace(ctx, allow_window=True)
    n_targets = rng.randint(1, cfg.max_select_items)
    targets: list[SelectTarget] = []
    for _ in range(n_targets):
        # 50/50 split: half grouped, half aggregate. The mix keeps
        # output looking like real analytic SQL — `region, count(*)`
        # rather than all-agg or all-grouped.
        if rng.random() < 0.5:
            e = rng.choice(grouped_exprs)
        else:
            e = _gen_aggregate_funccall(select_ctx)
        targets.append(SelectTarget(expr=e))

    # ---- WHERE: same as non-agg path; aggregates forbidden -----------
    where: Expr | None = None
    if FEATURE_WHERE in flags and rng.random() < cfg.p_where:
        where_ctx = replace(ctx, allow_aggregates=False)
        where = gen_expr(where_ctx, BOOL)

    # ---- HAVING: aggregate COMP literal -------------------------------
    having: Expr | None = None
    if FEATURE_HAVING in flags and rng.random() < cfg.p_having:
        having = _gen_having_expr(ctx)

    # ---- ORDER BY: pull from SELECT list (always GROUP-BY-consistent)
    order_by = _maybe_gen_order_by(ctx, targets)

    # ---- LIMIT --------------------------------------------------------
    limit = _maybe_gen_limit(ctx)

    return Select(
        targets=tuple(targets),
        from_=from_,
        with_ctes=with_ctes,
        where=where,
        group_by=group_by_clause,
        having=having,
        order_by=order_by,
        limit=limit,
    )


# ---- Shared sub-generators -------------------------------------------------

def _gen_from_clause(ctx: GenContext) -> tuple[FromItem, ...]:
    """Pick FROM items, populate scope, return the FROM tuple.

    Each FROM position is built incrementally — and for the explicit-
    JOIN path, each item's ON condition is generated IMMEDIATELY
    after the item is added to scope (before the next item exists).
    This enforces three structural rules at once:

      * LATERAL derived tables see preceding siblings (which were
        added in earlier iterations) but not later ones (which
        haven't been generated yet).

      * JoinExpr ON conditions for `tA JOIN tB ON ...` see only tA
        and tB (and earlier items in the join), not later joins —
        PG's parse-analysis rejects `t1 JOIN t2 ON t4.x = t2.y`
        because t4 isn't yet in scope at that point in the FROM
        tree.

      * Comma-FROM still adds each item before the next, so a LATERAL
        derived table at position i sees items 0..i-1 the same way
        the explicit-join path does.

    Mutates `ctx.scope` and `ctx.alias_counter` as side effects.
    """
    cfg = ctx.config
    rng = ctx.rng
    flags = cfg.feature_flags

    # DETERMINISM: `ctx.schema.tables` is an insertion-ordered tuple
    # built by the schema generator from a deterministic RNG draw, so
    # the rng.sample() output is stable across runs. Switching to a
    # set-derived container here would silently break the seed →
    # output guarantee.
    n_from = rng.randint(1, min(cfg.max_from_items, len(ctx.schema.tables)))
    chosen = rng.sample(ctx.schema.tables, n_from)
    # Reserve `n_from` consecutive alias indices from the query-wide
    # counter. Shared across all derived contexts so sibling /
    # nested subqueries don't collide on aliases.
    start = ctx.alias_counter.take(n_from)

    use_explicit = (
        n_from > 1
        and FEATURE_INNER_JOIN in flags
        and rng.random() < cfg.p_explicit_join
    )

    if use_explicit:
        # ALGORITHM: left-deep JOIN tree, built position-by-position.
        # We generate each ON condition right after its right-hand item
        # is added to scope. This is what keeps ON-clause name
        # resolution honest — at the time we generate `t1 JOIN t2 ON
        # ...`, only t1 and t2 are in scope. PG's parse analysis
        # rejects `t1 JOIN t2 ON t3.x = t2.y` even when t3 appears later
        # in the FROM, because the ON is parsed before t3 enters scope.
        first_alias = f"t{start}"
        first_item, _ = _make_from_item(
            ctx, first_alias, chosen[0], position=0,
        )
        tree: FromItem = first_item
        for i in range(1, n_from):
            alias = f"t{start + i}"
            item, base_t = _make_from_item(
                ctx, alias, chosen[i], position=i,
            )
            kind = _pick_join_kind(ctx)
            # ctx.scope at this point has items 0..i in it (item i
            # was just added by _make_from_item). The ON sees
            # exactly those — what PG's left-to-right rule requires.
            on = _gen_join_condition(ctx, alias, base_t)
            tree = JoinExpr(left=tree, right=item, kind=kind, on=on)
        return (tree,)

    # Comma-FROM path: incremental in scope-building too (so LATERAL
    # in a comma-FROM sees preceding siblings), but no ON conditions.
    items: list[FromItem] = []
    for i, t in enumerate(chosen):
        alias = f"t{start + i}"
        item, _ = _make_from_item(ctx, alias, t, position=i)
        items.append(item)
    return tuple(items)


def _make_from_item(
    ctx: GenContext,
    alias: str,
    table: Table,
    *,
    position: int,
) -> tuple[FromItem, Optional[Table]]:
    """Build one FROM item and register it in scope.

    Returns (item, optional_base_table) — the second element is the
    underlying Table for base-table FROMs, None for derived tables
    AND for CTE references (FK biasing applies only to base tables).

    Decisions, in priority order:
      1. CTE reference — gated on FEATURE_CTE and at least one CTE
         being visible in scope (has_visible_ctes). Picked first
         when available because it's the most "structural" FROM
         shape — once a WITH clause defines CTEs, generated queries
         should actually reference them often enough to exercise
         the resolution machinery.
      2. Derived table — gated on FEATURE_DERIVED_TABLE and
         subquery-depth budget. If derived, LATERAL vs non-LATERAL
         is gated on FEATURE_LATERAL and position > 0.
      3. Base table — the fallback / default.

    Scope addition happens BEFORE returning so the next call (for the
    next FROM position) sees this item as a preceding sibling.
    Critical for LATERAL semantics.
    """
    cfg = ctx.config
    rng = ctx.rng
    flags = cfg.feature_flags

    # CTE reference: only meaningful when at least one CTE is visible.
    use_cte = (
        FEATURE_CTE in flags
        and ctx.scope.has_visible_ctes()
        and rng.random() < cfg.p_cte_in_from
    )
    if use_cte:
        # Pick deterministically from the visible CTE pool (already
        # in deterministic order — dict insertion + chain walk).
        cte_names = ctx.scope.visible_cte_names()
        cte_name = rng.choice(cte_names)
        cte_columns = ctx.scope.lookup_cte(cte_name)
        # has_visible_ctes was True, so lookup must succeed; assert
        # here as a generator-bug tripwire.
        assert cte_columns is not None
        # Register the local alias's bindings (using the CTE's column
        # info but tagged with this local alias).
        ctx.scope.add_derived(alias, cte_columns)
        return CteRef(cte_name=cte_name, alias=alias), None

    use_derived = (
        FEATURE_DERIVED_TABLE in flags
        and not ctx.at_subquery_leaf()  # need budget for the inner SELECT
        and rng.random() < cfg.p_derived_table_in_from
    )

    if use_derived:
        lateral = (
            position > 0
            and FEATURE_LATERAL in flags
            and rng.random() < cfg.p_lateral_when_derived
        )
        dt = gen_derived_table(ctx, alias, lateral=lateral)
        cols = [
            (st.alias if st.alias is not None else "c1",
             st.expr.pg_type)
            for st in dt.select.targets
        ]
        ctx.scope.add_derived(alias, cols)
        return dt, None

    # Base table
    ctx.scope.add_table(alias, table)
    return TableRef(table.name, alias), table


def _maybe_gen_order_by(
    ctx: GenContext,
    targets: list[SelectTarget],
) -> tuple[OrderByItem, ...]:
    cfg = ctx.config
    rng = ctx.rng
    if (FEATURE_ORDER_BY not in cfg.feature_flags
            or not targets
            or rng.random() >= cfg.p_order_by):
        return ()
    n_order = rng.randint(1, len(targets))
    idx = sorted(rng.sample(range(len(targets)), n_order))
    return tuple(
        OrderByItem(
            expr=targets[i].expr,
            direction=rng.choice(("ASC", "DESC")),
        )
        for i in idx
    )


def _maybe_gen_limit(ctx: GenContext) -> Expr | None:
    cfg = ctx.config
    if FEATURE_LIMIT in cfg.feature_flags and ctx.rng.random() < cfg.p_limit:
        return Literal(INT4, ctx.rng.choice(_LIMIT_VALUES))
    return None


# ---- Grouping-set construction --------------------------------------------

def _gen_grouping_set(
    ctx: GenContext,
    grouped_exprs: tuple[Expr, ...],
) -> GroupingSet:
    """Wrap `grouped_exprs` in a ROLLUP, CUBE, or GROUPING SETS
    construct. The choice is uniform across the three keywords.

    Element shape per kind:

      * ROLLUP: each grouped expr is its own single-expr element,
        producing `ROLLUP (a, b, c)` — the canonical hierarchical-
        rollup shape.
      * CUBE: same — `CUBE (a, b, c)` produces 2^n grouping sets.
      * GROUPING SETS: enumerates 2..N random subsets of grouped_exprs
        (including possibly the empty set `()` for the grand total).
        Output looks like `GROUPING SETS ((a, b), (c), ())`.

    Multi-expr per element (e.g. `ROLLUP ((a, b), c)` where (a,b) is
    treated as a single rollup level) is structurally supported by
    GroupingSet but not generated here — single-expr elements are
    the dominant real-world shape and keep output readable.
    """
    rng = ctx.rng
    kind = rng.choice(("ROLLUP", "CUBE", "GROUPING SETS"))

    # Annotated explicitly — without it mypy unifies the two branches
    # to the narrower `tuple[tuple[Expr], ...]` from the ROLLUP/CUBE
    # path (single-expr inner tuples) and then rejects the GROUPING
    # SETS path which produces variable-length inner tuples.
    elements: tuple[tuple[Expr, ...], ...]
    if kind in ("ROLLUP", "CUBE"):
        elements = tuple((expr,) for expr in grouped_exprs)
    else:  # GROUPING SETS
        # 2..min(4, 2^N) sets — enough variety, kept low for
        # readability. Each set is a random subset; duplicates
        # across sets are accepted (PG just collapses them).
        # Always include the empty grouping at least sometimes.
        max_n = min(4, 2 ** len(grouped_exprs))
        n_sets = rng.randint(2, max_n) if max_n >= 2 else 2
        sets: list[tuple[Expr, ...]] = []
        for _ in range(n_sets):
            n = rng.randint(0, len(grouped_exprs))
            sample = rng.sample(grouped_exprs, n) if n > 0 else []
            sets.append(tuple(sample))
        elements = tuple(sets)

    return GroupingSet(kind=kind, elements=elements)


# ---- Aggregate construction helpers ---------------------------------------

def _gen_aggregate_funccall(ctx: GenContext) -> FuncCall:
    """Pick a random aggregate function and build its FuncCall.

    Aggregate selection is over `catalog.functions` (ordered tuple,
    stable across runs); each declared overload is one candidate,
    so aggregates with more overloads (e.g. `min`/`max`) appear
    proportionally more often than rarely-overloaded ones (e.g.
    `string_agg`, `bool_and`).

    Args are generated under `in_aggregate=True` (blocks nested
    aggregates) AND `allow_window=False` (blocks windows inside
    aggregates — PG evaluates aggregates *before* windows in its
    pipeline, so `count(row_number() OVER (...))` is a parse-
    analysis error). Both flag resets needed when allow_window
    might be inherited from a SELECT-list-context parent ctx.
    """
    aggs = [f for f in ctx.catalog.functions if f.kind == FuncKind.AGGREGATE]
    f = ctx.rng.choice(aggs)
    # Ordered-set aggregates (percentile_cont, percentile_disc, ...)
    # MUST be called with WITHIN GROUP — special-case before any
    # other path. Same gate as gen_expr's agg branch.
    if f.name in ORDERED_SET_AGGREGATES:
        return gen_ordered_set_agg(ctx, f)
    # FILTER eligibility — every aggregate accepts FILTER in PG.
    # Gen up-front so it composes with the star form below.
    filter_expr = (
        gen_filter_predicate(ctx)
        if should_emit_filter(ctx.rng)
        else None
    )
    # `count(*)` substitution — same gate as gen_expr's agg branch.
    if f.name == "count" and should_emit_count_star(ctx.rng):
        return FuncCall(
            f.returns, "count", (), star=True, filter_=filter_expr,
        )
    arg_ctx = replace(
        ctx.descend(),
        in_aggregate=True,
        allow_window=False,
    )
    # Wrap in explicit casts when the actual arg type doesn't match
    # the param type — same overload-resolution defense as gen_expr.
    args = tuple(
        coerce_to_param_type(gen_expr(arg_ctx, arg_t), arg_t)
        for arg_t in f.args
    )
    return FuncCall(f.returns, f.name, args, filter_=filter_expr)


def _gen_having_expr(ctx: GenContext) -> Expr:
    """Construct a HAVING expression as `aggregate COMP literal`.

    This shape — an aggregate against a literal of the same type — is
    the most realistic HAVING form (`HAVING count(*) > 10`,
    `HAVING sum(amount) >= 1000`) and guarantees PARSE-tier
    correctness: the aggregate side carries the post-grouping
    semantic, the literal side has no GROUP BY constraint at all.

    Aggregates whose return type isn't comparable in our catalog
    (e.g. array_agg returns int4[], no array comparison ops
    registered) are filtered out — using one of those would fall
    into the defensive branch which produces a trivial `TRUE`.
    """
    cat = ctx.catalog
    bool_ops = cat.binary_ops_returning(BOOL)
    # Types T such that there's a `T COMP T → BOOL` op available.
    # Using a set is fine here: we only do membership tests, never
    # iterate (set iteration order would be a determinism hazard).
    comparable = {o.left for o in bool_ops if o.left == o.right}

    candidate_aggs = [
        f for f in cat.functions
        if f.kind == FuncKind.AGGREGATE and f.returns in comparable
    ]
    if not candidate_aggs:
        # Pathological catalog (no comparable aggregate returns at
        # all). Emit a trivially-TRUE HAVING — parses, runs, doesn't
        # filter anything.
        return Literal(BOOL, True)

    f = ctx.rng.choice(candidate_aggs)
    # Ordered-set aggregates need WITHIN GROUP — same special-case
    # gate as the other two agg-construction sites. Returns the
    # FuncCall without FILTER attached (the helper doesn't add one);
    # the HAVING comparison wraps it as usual.
    if f.name in ORDERED_SET_AGGREGATES:
        agg_expr: Expr = gen_ordered_set_agg(ctx, f)
    else:
        # FILTER + count(*) attachments — same gates as the other agg
        # construction sites. Both compose: `count(*) FILTER (WHERE ...)`
        # is canonical and the most common filtered-aggregate form.
        filter_expr = (
            gen_filter_predicate(ctx)
            if should_emit_filter(ctx.rng)
            else None
        )
        if f.name == "count" and should_emit_count_star(ctx.rng):
            agg_expr = FuncCall(
                f.returns, "count", (), star=True, filter_=filter_expr,
            )
        else:
            arg_ctx = replace(ctx.descend(), in_aggregate=True, allow_window=False)
            args = tuple(
                coerce_to_param_type(gen_expr(arg_ctx, arg_t), arg_t)
                for arg_t in f.args
            )
            agg_expr = FuncCall(f.returns, f.name, args, filter_=filter_expr)

    matching_ops = [
        o for o in bool_ops
        if o.left == f.returns and o.right == f.returns
    ]
    op = ctx.rng.choice(matching_ops)
    # The equality filter above guarantees o.right == f.returns (a
    # concrete PgType), so o.right is non-None. mypy can't narrow
    # through the equality, so assert locally.
    assert op.right is not None
    rhs = gen_literal(ctx.rng, op.right)
    return BinaryOp(BOOL, op.symbol, agg_expr, rhs)


# ===========================================================================
# Internals: JOIN-tree construction and ON-condition FK biasing
# ===========================================================================

# (formerly: _build_join_tree and _alias_of — removed when
# _gen_from_clause was refactored to interleave scope-population with
# ON-condition generation, eliminating the forward-reference bug
# where ON for `t1 JOIN t2` could reference t3+ that wasn't yet in
# scope at parse-analysis time.)


def _pick_join_kind(ctx: GenContext) -> str:
    """Pick INNER vs LEFT for an explicit join.

    LEFT is only an option when FEATURE_LEFT_JOIN is unlocked at the
    current dial level. RIGHT and FULL are deferred — they're
    syntactically supported by the AST but the generator avoids them
    to keep the join semantics simpler to reason about.
    """
    if (FEATURE_LEFT_JOIN in ctx.config.feature_flags
            and ctx.rng.random() < ctx.config.p_left_join_when_explicit):
        return "LEFT"
    return "INNER"


def _gen_join_condition(
    ctx: GenContext,
    right_alias: str,
    right_table: Optional[Table],
) -> Expr:
    """Generate an ON condition for joining `right_alias` to the
    existing scope tables.

    Strategy:
      1. If the right side is a base table (`right_table` is not None),
         look for FK relationships either way between it and any
         already-aliased BASE table. Each found FK gives a candidate
         equality `left_alias.id = right_alias.fk_col` (or vice versa).
      2. If candidates exist, pick one at random — this is the FK bias.
      3. Otherwise (right is derived, OR no FK candidates), fall back
         to a random BOOL expression. Derived tables have no FKs so
         this path is the only option for a derived right side.

    Composite FKs (multi-column) are skipped — the schema generator
    never produces them; if that changes, this function needs an
    AND-of-equalities path.
    """
    aliased = ctx.scope.aliased_tables()  # base tables only
    others = [(a, t) for a, t in aliased if a != right_alias]

    # Each candidate is (left_alias, left_col, right_alias, right_col).
    # Order is deterministic because both `right_table.foreign_keys`
    # and `aliased_tables()` are insertion-ordered.
    candidates: list[tuple[str, str, str, str]] = []

    # FK biasing only applies when the right side is a base table.
    # Derived tables have no FKs, so right_table is None for derived
    # and we skip directly to the gen_expr fallback.
    if right_table is not None:
        # Both directions are collected because either makes a valid
        # equi-join. Walking right→left FKs first then left→right
        # produces a stable insertion order; the rng.choice below is
        # the only place RNG enters the FK-bias decision.
        # FKs FROM the new right table TO an already-present table.
        for fk in right_table.foreign_keys:
            if len(fk.columns) != 1 or len(fk.ref_columns) != 1:
                continue
            for other_alias, other_table in others:
                if fk.ref_table == other_table.name:
                    candidates.append((
                        other_alias, fk.ref_columns[0],
                        right_alias, fk.columns[0],
                    ))

        # FKs FROM an already-present table TO the new right table.
        for other_alias, other_table in others:
            for fk in other_table.foreign_keys:
                if len(fk.columns) != 1 or len(fk.ref_columns) != 1:
                    continue
                if fk.ref_table == right_table.name:
                    candidates.append((
                        other_alias, fk.columns[0],
                        right_alias, fk.ref_columns[0],
                    ))

    if candidates:
        left_a, left_c, right_a, right_c = ctx.rng.choice(candidates)
        left_t = ctx.scope.lookup_alias(left_a)
        right_t = ctx.scope.lookup_alias(right_a)
        # lookup_alias returns None only if the alias was never added,
        # which can't happen here (we added them all upstream).
        assert left_t is not None and right_t is not None
        left_type = left_t.column(left_c).type
        right_type = right_t.column(right_c).type
        return BinaryOp(
            BOOL, "=",
            ColumnRef(left_type, left_a, left_c),
            ColumnRef(right_type, right_a, right_c),
        )

    # No FK candidate. Fall back to a random BOOL via gen_expr.
    # Aggregates are forbidden in ON clauses by PG parse analysis
    # regardless of whether the surrounding query aggregates, so
    # explicitly disable them here even if the caller's ctx allowed
    # them. This is enforced at the function level so every caller
    # gets it for free.
    return gen_expr(replace(ctx, allow_aggregates=False), BOOL)


# ===========================================================================
# Internals: target-type picking
# ===========================================================================

def _pick_select_type(rng: random.Random) -> PgType:
    types = [t for t, _ in _SELECT_TYPE_WEIGHTS]
    weights = [w for _, w in _SELECT_TYPE_WEIGHTS]
    return rng.choices(types, weights=weights, k=1)[0]


__all__ = ["gen_select"]
