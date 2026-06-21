"""Tests for waxsql.context.

GenContext is mostly a record type, so the tests focus on the parts
that have actual behavior: descend(), at_leaf(), and the frozen/
shared-mutable-state semantics that the determinism guarantees rely on.
"""
import random
from dataclasses import replace

import pytest

from waxsql import default_catalog, generate_schema
from waxsql.config import query_config_for_complexity
from waxsql.context import GenContext
from waxsql.scope import Scope


def _ctx(depth: int = 5) -> GenContext:
    return GenContext(
        rng=random.Random(0),
        scope=Scope(),
        schema=generate_schema(seed=0, complexity=3),
        catalog=default_catalog(),
        config=query_config_for_complexity(5),
        depth_remaining=depth,
    )


def test_descend_decrements_depth():
    ctx = _ctx(depth=3)
    assert ctx.descend().depth_remaining == 2
    assert ctx.descend(cost=2).depth_remaining == 1


def test_descend_shares_rng_object():
    """Critical for determinism: descending must NOT clone the RNG;
    otherwise the same draw would happen multiple times in different
    branches and the (seed → output) map would break."""
    ctx = _ctx()
    child = ctx.descend()
    assert ctx.rng is child.rng


def test_descend_shares_scope_object():
    """Descending into a sub-expression doesn't change scope; scope
    only changes when entering a subquery."""
    ctx = _ctx()
    assert ctx.descend().scope is ctx.scope


def test_at_leaf_true_only_when_budget_exhausted():
    assert not _ctx(depth=3).at_leaf()
    assert not _ctx(depth=1).at_leaf()
    assert _ctx(depth=0).at_leaf()
    assert _ctx(depth=-5).at_leaf()


def test_replace_for_aggregate_flag_is_isolated():
    """Flipping `allow_aggregates` in a child must not leak back."""
    ctx = _ctx()
    no_aggs = replace(ctx, allow_aggregates=False)
    assert ctx.allow_aggregates is True
    assert no_aggs.allow_aggregates is False


def test_context_is_frozen():
    """Direct field assignment should be blocked — modifications go
    through replace()."""
    ctx = _ctx()
    with pytest.raises(Exception):
        ctx.depth_remaining = 99  # type: ignore[misc]


def test_default_flags_match_outermost_select_body():
    ctx = _ctx()
    assert ctx.in_aggregate is False
    assert ctx.allow_aggregates is True   # SELECT-list / HAVING are OK
    assert ctx.allow_window is False      # only true inside OVER()
    assert ctx.allow_subquery is True


def test_subquery_depth_defaults_to_zero():
    """Safe default: a GenContext built without thinking about
    subqueries doesn't allow any. Only generate_query (which sets
    it from cfg.max_subquery_depth) opts into nesting."""
    ctx = _ctx()
    assert ctx.subquery_depth_remaining == 0
    assert ctx.at_subquery_leaf() is True


def test_descend_subquery_decrements_subquery_depth():
    ctx = replace(_ctx(), subquery_depth_remaining=2)
    child = ctx.descend_subquery(correlated=True)
    assert child.subquery_depth_remaining == 1


def test_descend_subquery_resets_expression_depth():
    """The subquery's expression tree starts with a full budget,
    independent of the outer expression's remaining depth."""
    ctx = replace(_ctx(depth=1), subquery_depth_remaining=2)
    child = ctx.descend_subquery(correlated=True)
    assert child.depth_remaining == ctx.config.max_expr_depth
    assert child.depth_remaining > ctx.depth_remaining


def test_descend_subquery_resets_aggregate_flags():
    """A subquery in WHERE has its own expression context — the
    outer 'no aggregates' constraint doesn't propagate inward."""
    ctx = replace(
        _ctx(),
        subquery_depth_remaining=2,
        allow_aggregates=False,    # outer is in WHERE
        in_aggregate=True,         # outer was inside an aggregate arg
    )
    child = ctx.descend_subquery(correlated=True)
    assert child.allow_aggregates is True
    assert child.in_aggregate is False


def test_descend_subquery_creates_correlated_child_scope():
    """The child scope inherits parent visibility when correlated=True
    — Scope's push_subquery handles the actual chain, but verify the
    helper hands off the flag correctly."""
    from waxsql.scope import Scope
    from waxsql.schema import Column, Table
    from waxsql.types import INT8, TEXT

    outer_table = Table(
        name="t_outer",
        columns=(Column("id", INT8, nullable=False),
                 Column("name", TEXT)),
        primary_key=("id",),
    )
    outer_scope = Scope()
    outer_scope.add_table("o", outer_table)
    ctx = replace(_ctx(), scope=outer_scope, subquery_depth_remaining=2)

    correlated = ctx.descend_subquery(correlated=True)
    aliases = {b.table_alias for b in correlated.scope.visible_columns()}
    assert "o" in aliases  # outer table visible

    uncorr = ctx.descend_subquery(correlated=False)
    aliases = {b.table_alias for b in uncorr.scope.visible_columns()}
    assert "o" not in aliases  # outer table NOT visible


def test_descend_subquery_shares_rng():
    """Determinism: the rng must be shared, not cloned, across
    descents into subqueries."""
    ctx = replace(_ctx(), subquery_depth_remaining=2)
    child = ctx.descend_subquery(correlated=True)
    assert child.rng is ctx.rng


def test_alias_counter_shared_across_descents():
    """The alias_counter is a mutable monotonic counter; advancing it
    in a child context (via _gen_from_clause) must be visible in the
    parent and in sibling contexts produced from the same parent."""
    ctx = replace(_ctx(), subquery_depth_remaining=2)
    child = ctx.descend_subquery(correlated=True)
    assert child.alias_counter is ctx.alias_counter
    child.alias_counter.value = 42
    assert ctx.alias_counter.value == 42


def test_cte_counter_shared_across_descents():
    """Same mutable-cell mechanism as alias_counter, separate
    namespace — CTE name generation across nested gen_select calls
    doesn't collide because the counter advances monotonically
    across all of them."""
    ctx = replace(_ctx(), subquery_depth_remaining=2)
    child = ctx.descend_subquery(correlated=True)
    assert child.cte_counter is ctx.cte_counter
    child.cte_counter.value = 99
    assert ctx.cte_counter.value == 99


def test_alias_and_cte_counters_are_independent():
    """Two separate counters means CTE names and table aliases stay
    in their own number sequences regardless of how many of either
    are generated."""
    ctx = _ctx()
    assert ctx.alias_counter is not ctx.cte_counter
    assert ctx.alias_counter.value == 1
    assert ctx.cte_counter.value == 1
