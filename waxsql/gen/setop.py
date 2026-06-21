"""Set-operation generator (UNION / INTERSECT / EXCEPT).

Role: combines N SELECTs into one statement. The hard constraint
that gives the file its shape: both (all) sides of a set-op must
produce the SAME column types in the SAME order. PG enforces this
positionally — column #i of every arm must implicitly-cast to a
common type with column #i of every other arm. The generator
sidesteps the cross-arm cast negotiation entirely by FIXING arm 1's
exact target types and reusing them as the spec for arms 2..N (see
`target_types` below). Arms 2..N don't get gen_select's full freedom;
they call _gen_arm_select with a pinned type list.

One public entry point: `gen_setop(ctx)` builds a SetOp wrapping
N Selects with matching column counts and types.

Two phases:

  1. **Arm 1**: a full `gen_select(arm1_ctx)` in its own child scope.
     Could be aggregate, could have any target shape. We extract its
     target types after generation, then strip its with_ctes /
     order_by / limit / offset (those belong to the SetOp wrapper,
     not the arm).

  2. **Arms 2+**: simpler Selects that match arm 1's target types
     position-by-position. Built via `_gen_arm_select` which produces
     a non-aggregate SELECT with FROM, optional WHERE, and one
     SelectTarget per requested type (generated via gen_expr against
     the arm's local scope).

Each arm gets its own scope via `descend_subquery(correlated=False)`
so arm-local FROM aliases don't collide with the surrounding query
or with other arms. The shared `alias_counter` (and `cte_counter`)
ensures unique naming across the whole query.

After all arms are generated, the SetOp wrapper optionally gets
its own ORDER BY (positional reference: `ORDER BY 1`) and LIMIT —
both PG-valid only at the combined-statement level for milestone 7
(per-arm ORDER BY/LIMIT requires explicit parens).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

from ..ast import (
    Expr, Literal, OrderByItem, Select, SelectTarget, SetOp,
)
from ..config import (
    FEATURE_LIMIT, FEATURE_ORDER_BY, FEATURE_WHERE,
)
from ..context import GenContext
from ..types import BOOL, INT4, PgType
from .expr import gen_expr


_LIMIT_VALUES: tuple[int, ...] = (1, 5, 10, 25, 50, 100)


def gen_setop(ctx: GenContext) -> SetOp:
    """Generate a SetOp combining N arms with the chosen operator.

    Arm 1 is a full gen_select; subsequent arms match arm 1's
    target types via _gen_arm_select OR (with probability
    `cfg.p_nested_set_op_arm`) by recursively generating another
    SetOp whose own first arm matches the parent's target types.

    Each arm has its own child scope; the shared alias_counter
    prevents cross-arm alias collision. Nested SetOps consume
    `subquery_depth_remaining` via the per-arm descend_subquery,
    so unbounded nesting is impossible.

    The SetOp wrapper's ORDER BY/LIMIT (if generated) reference
    output columns positionally (`ORDER BY 1`) — the only form
    that works without naming the unified output columns
    explicitly.
    """
    # Lazy import — gen_select / _gen_arm_select live in gen/select.py,
    # which currently doesn't import gen/setop.py but might in future.
    from .select import gen_select

    cfg = ctx.config
    rng = ctx.rng

    n_arms = rng.randint(2, cfg.max_set_op_arms)
    op = rng.choice(("UNION", "INTERSECT", "EXCEPT"))
    all_ = rng.random() < cfg.p_set_op_all

    # Arm 1: full SELECT in its own child scope. MUST be generated
    # before arms 2+ — its target types drive everything downstream
    # (positional set-op compatibility means every later arm's
    # targets are typed-matched to arm 1's, not the other way).
    arm1_ctx = ctx.descend_subquery(correlated=False)
    arm1 = _strip_arm_clauses(gen_select(arm1_ctx))
    # `target_types` is the positional type signature every later arm
    # must match. We extract it from arm 1 AFTER stripping the
    # combined-statement clauses but BEFORE iterating arms 2..N —
    # gen_select's own type choices become the canonical spec for
    # the whole set-op.
    target_types: list[PgType] = [t.expr.pg_type for t in arm1.targets]

    # `arms` may hold either Selects or nested SetOps (Track B #5).
    # The explicit union annotation keeps mypy honest about the
    # mixed-element list — without it, the inferred type from the
    # initial `[arm1]` (Select-only) would reject the SetOp append.
    arms: list[Select | SetOp] = [arm1]
    for _ in range(n_arms - 1):
        arm_ctx = ctx.descend_subquery(correlated=False)
        # Nested-SetOp candidate gating:
        #   * Need budget — descending into a nested SetOp consumes
        #     another subquery_depth in turn for its own arms.
        #   * Probability gate. Kept moderate-low; deeply nested
        #     setops aren't realistic SQL and produce huge outputs.
        arm: Select | SetOp
        if (not arm_ctx.at_subquery_leaf()
                and rng.random() < cfg.p_nested_set_op_arm):
            arm = _gen_nested_setop_arm(arm_ctx, target_types)
        else:
            arm = _gen_arm_select(arm_ctx, target_types)
        arms.append(arm)

    # SetOp-level ORDER BY (positional) and LIMIT.
    order_by: tuple[OrderByItem, ...] = ()
    if (FEATURE_ORDER_BY in cfg.feature_flags
            and rng.random() < cfg.p_order_by):
        # Positional reference: `ORDER BY <n> ASC|DESC`. PG interprets
        # bare integer literals in ORDER BY as 1-based output-column
        # references — works regardless of what the unified output
        # columns are named.
        pos = rng.randint(1, len(target_types))
        direction = rng.choice(("ASC", "DESC"))
        order_by = (OrderByItem(
            expr=Literal(INT4, pos),
            direction=direction,
        ),)

    limit: Optional[Expr] = None
    if (FEATURE_LIMIT in cfg.feature_flags
            and rng.random() < cfg.p_limit):
        limit = Literal(INT4, rng.choice(_LIMIT_VALUES))

    return SetOp(
        op=op,
        all=all_,
        arms=tuple(arms),
        order_by=order_by,
        limit=limit,
    )


# ===========================================================================
# Internals
# ===========================================================================

def _gen_nested_setop_arm(
    ctx: GenContext,
    target_types: list[PgType],
) -> SetOp:
    """Generate a nested SetOp suitable for use as an arm of an
    outer SetOp.

    All arms of the nested SetOp must produce the same target_types
    as the outer's arm 1 (set-op type compatibility is positional).
    The nested SetOp also doesn't get its own ORDER BY/LIMIT —
    those would be ambiguous with the outer's, and PG requires
    explicit per-arm parens for them anyway. The wrapping parens
    happen in the printer based on AST structure.

    Currently always 2-armed (no further nesting). Could recurse to
    arbitrary depth via subquery_depth_remaining, but two-deep
    nesting (`A UNION (B INTERSECT C)`) is the realistic case;
    deeper chains produce mostly-unreadable output.
    """
    rng = ctx.rng
    op = rng.choice(("UNION", "INTERSECT", "EXCEPT"))
    all_ = rng.random() < ctx.config.p_set_op_all

    # All arms of the nested SetOp use _gen_arm_select with the
    # given target types, ensuring positional type alignment with
    # the OUTER setop's arm 1.
    n_arms = 2
    inner_arms: list[Select] = []
    for _ in range(n_arms):
        sub_ctx = ctx.descend_subquery(correlated=False)
        inner_arms.append(_gen_arm_select(sub_ctx, target_types))

    # No order_by / limit / offset — those belong only to the
    # outermost SetOp wrapper.
    return SetOp(op=op, all=all_, arms=tuple(inner_arms))


def _strip_arm_clauses(s: Select) -> Select:
    """Remove clauses that belong to the SetOp wrapper, not individual
    arms — with_ctes, order_by, limit, offset.

    PG's grammar requires per-arm ORDER BY/LIMIT/OFFSET to be wrapped
    in parens. Stripping is simpler than parenthesizing, and the
    SetOp wrapper carries equivalents at the combined-statement level."""
    return replace(
        s,
        with_ctes=(),
        order_by=(),
        limit=None,
        offset=None,
    )


# CONTRACT: every Select returned here has exactly len(target_types)
# targets, in order, with each target's pg_type implicitly-castable
# from target_types[i]. This is what the outer SetOp wrapper relies
# on for cross-arm compatibility. Violating it produces a PG error
# like "each UNION query must have the same number of columns" or
# "UNION types int and text cannot be matched".
def _gen_arm_select(
    ctx: GenContext,
    target_types: list[PgType],
) -> Select:
    """Build a Select for an arms-2+ SetOp arm: FROM clause, target
    list of pre-determined types, optional WHERE.

    Bypasses gen_select's aggregate-vs-non-aggregate dispatch: arm
    Selects in milestone 7 are always non-aggregate (matching aggregate
    target types from arm 1 across arms is handled by the type
    machinery, not the GROUP BY logic). Aggregate-style arms can come
    in a follow-up.
    """
    # Lazy imports for the same cycle-breaking reason as gen/cte.py
    # — gen/select.py imports from gen/setop.py at the top level, so
    # importing _gen_from_clause at module top would close the loop.
    from .select import _gen_from_clause

    cfg = ctx.config
    rng = ctx.rng
    flags = cfg.feature_flags

    # FROM clause — populates arm-local scope.
    from_ = _gen_from_clause(ctx)

    # Target list: one SelectTarget per requested type, generated
    # against the arm's scope. allow_aggregates=False because this
    # path is non-aggregate.
    expr_ctx = replace(ctx, allow_aggregates=False)
    targets = tuple(
        SelectTarget(expr=gen_expr(expr_ctx, t))
        for t in target_types
    )

    # Optional WHERE.
    where: Optional[Expr] = None
    if FEATURE_WHERE in flags and rng.random() < cfg.p_where:
        where = gen_expr(expr_ctx, BOOL)

    return Select(
        targets=targets,
        from_=from_,
        where=where,
    )


__all__ = ["gen_setop"]
