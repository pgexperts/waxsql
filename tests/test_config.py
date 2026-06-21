"""Tests for waxsql.config.

The dial mapping is the primary thing under test: complexity 0 should
be trivial, complexity 10 should unlock everything, intermediate
levels should grow monotonically.
"""
from waxsql.config import (
    FEATURE_AGGREGATE, FEATURE_CTE, FEATURE_DERIVED_TABLE, FEATURE_EXISTS,
    FEATURE_HAVING, FEATURE_INNER_JOIN, FEATURE_IN_SUBQUERY,
    FEATURE_LATERAL, FEATURE_LEFT_JOIN, FEATURE_LIMIT, FEATURE_ORDER_BY,
    FEATURE_RECURSIVE_CTE, FEATURE_SCALAR_SUBQUERY, FEATURE_SET_OP,
    FEATURE_WHERE,
    query_config_for_complexity,
)


def test_complexity_zero_unlocks_no_features():
    """At c=0 we want SELECT col FROM t — nothing else."""
    cfg = query_config_for_complexity(0)
    assert cfg.feature_flags == frozenset()
    assert cfg.max_from_items == 1


def test_complexity_one_unlocks_where_and_inner_join():
    cfg = query_config_for_complexity(1)
    assert FEATURE_WHERE in cfg.feature_flags
    assert FEATURE_INNER_JOIN in cfg.feature_flags
    assert FEATURE_LEFT_JOIN not in cfg.feature_flags


def test_complexity_two_unlocks_order_by_and_limit():
    cfg = query_config_for_complexity(2)
    assert FEATURE_ORDER_BY in cfg.feature_flags
    assert FEATURE_LIMIT in cfg.feature_flags


def test_complexity_four_unlocks_left_join():
    assert FEATURE_LEFT_JOIN not in query_config_for_complexity(3).feature_flags
    assert FEATURE_LEFT_JOIN in query_config_for_complexity(4).feature_flags


def test_complexity_three_unlocks_aggregate():
    assert FEATURE_AGGREGATE not in query_config_for_complexity(2).feature_flags
    assert FEATURE_AGGREGATE in query_config_for_complexity(3).feature_flags


def test_complexity_five_unlocks_having():
    """HAVING is staggered one notch behind AGGREGATE so the dial
    introduces one feature per notch where possible."""
    assert FEATURE_HAVING not in query_config_for_complexity(4).feature_flags
    assert FEATURE_HAVING in query_config_for_complexity(5).feature_flags
    # HAVING also requires AGGREGATE — having one without the other
    # would let gen_select fire HAVING for non-aggregating queries,
    # which is a PG syntax error.
    cfg = query_config_for_complexity(5)
    assert FEATURE_AGGREGATE in cfg.feature_flags


def test_max_group_by_items_grows_with_complexity():
    assert query_config_for_complexity(0).max_group_by_items >= 1
    assert (query_config_for_complexity(10).max_group_by_items
            > query_config_for_complexity(3).max_group_by_items)


def test_complexity_four_unlocks_scalar_subquery():
    assert FEATURE_SCALAR_SUBQUERY not in query_config_for_complexity(3).feature_flags
    assert FEATURE_SCALAR_SUBQUERY in query_config_for_complexity(4).feature_flags


def test_complexity_five_unlocks_exists_and_in_subquery():
    assert FEATURE_EXISTS not in query_config_for_complexity(4).feature_flags
    assert FEATURE_IN_SUBQUERY not in query_config_for_complexity(4).feature_flags
    assert FEATURE_EXISTS in query_config_for_complexity(5).feature_flags
    assert FEATURE_IN_SUBQUERY in query_config_for_complexity(5).feature_flags


def test_max_subquery_depth_grows_with_complexity():
    """Below the unlock notch (c<4), depth is 0; from c=4 onward it
    grows slowly so deep nesting only happens at high complexity."""
    assert query_config_for_complexity(3).max_subquery_depth == 0
    assert query_config_for_complexity(4).max_subquery_depth >= 1
    assert (query_config_for_complexity(10).max_subquery_depth
            >= query_config_for_complexity(4).max_subquery_depth)


def test_subquery_weights_are_positive():
    """Zero weights would mean the production never fires; that
    breaks the feature even when the flag is set."""
    for c in range(0, 11):
        cfg = query_config_for_complexity(c)
        assert cfg.scalar_subquery_weight > 0, c
        assert cfg.exists_weight > 0, c
        assert cfg.in_subquery_weight > 0, c


def test_complexity_five_unlocks_derived_table():
    assert FEATURE_DERIVED_TABLE not in query_config_for_complexity(4).feature_flags
    assert FEATURE_DERIVED_TABLE in query_config_for_complexity(5).feature_flags


def test_complexity_six_unlocks_lateral():
    """LATERAL is gated one notch behind DERIVED_TABLE so the dial
    progression introduces one feature per notch where possible."""
    assert FEATURE_LATERAL not in query_config_for_complexity(5).feature_flags
    assert FEATURE_LATERAL in query_config_for_complexity(6).feature_flags
    # LATERAL only meaningful with derived tables; verify both are
    # set together at c >= 6.
    cfg = query_config_for_complexity(6)
    assert FEATURE_DERIVED_TABLE in cfg.feature_flags


def test_derived_and_lateral_probabilities_sane():
    """Sanity bounds on the new probability fields."""
    for c in range(0, 11):
        cfg = query_config_for_complexity(c)
        assert 0.0 <= cfg.p_derived_table_in_from <= 1.0, c
        assert 0.0 <= cfg.p_lateral_when_derived <= 1.0, c
    # p_derived shouldn't dominate — base tables should remain the
    # common case.
    assert query_config_for_complexity(10).p_derived_table_in_from < 0.5


def test_complexity_seven_unlocks_cte():
    assert FEATURE_CTE not in query_config_for_complexity(6).feature_flags
    assert FEATURE_CTE in query_config_for_complexity(7).feature_flags


def test_max_ctes_per_with_grows_with_complexity():
    """At c=7 (unlock notch) max is 1; grows to 3 at c=10."""
    assert query_config_for_complexity(7).max_ctes_per_with == 1
    assert query_config_for_complexity(10).max_ctes_per_with >= 2


def test_cte_probabilities_sane():
    for c in range(0, 11):
        cfg = query_config_for_complexity(c)
        assert 0.0 <= cfg.p_with_clause <= 1.0, c
        assert 0.0 <= cfg.p_cte_in_from <= 1.0, c
    # WITH shouldn't dominate either.
    assert query_config_for_complexity(10).p_with_clause < 0.5


def test_complexity_nine_unlocks_set_op():
    assert FEATURE_SET_OP not in query_config_for_complexity(8).feature_flags
    assert FEATURE_SET_OP in query_config_for_complexity(9).feature_flags


def test_setop_probabilities_sane():
    for c in range(0, 11):
        cfg = query_config_for_complexity(c)
        assert 0.0 <= cfg.p_set_op_query <= 1.0, c
        assert 0.0 <= cfg.p_set_op_all <= 1.0, c
        assert cfg.max_set_op_arms >= 2, c
    # SetOps shouldn't dominate.
    assert query_config_for_complexity(10).p_set_op_query < 0.5


def test_complexity_ten_unlocks_recursive_cte():
    assert FEATURE_RECURSIVE_CTE not in query_config_for_complexity(9).feature_flags
    assert FEATURE_RECURSIVE_CTE in query_config_for_complexity(10).feature_flags


def test_recursive_cte_probability_sane():
    for c in range(0, 11):
        cfg = query_config_for_complexity(c)
        assert 0.0 <= cfg.p_recursive_when_cte <= 1.0, c
    # Recursive CTEs shouldn't be the dominant CTE form.
    assert query_config_for_complexity(10).p_recursive_when_cte < 0.6


def test_complexity_caps_grow_monotonically():
    prev = query_config_for_complexity(0)
    for c in range(1, 11):
        cur = query_config_for_complexity(c)
        assert cur.max_expr_depth >= prev.max_expr_depth, c
        assert cur.max_from_items >= prev.max_from_items, c
        assert cur.max_select_items >= prev.max_select_items, c
        prev = cur


def test_feature_set_is_monotonic():
    """A higher dial should never *remove* features."""
    prev_flags = query_config_for_complexity(0).feature_flags
    for c in range(1, 11):
        cur_flags = query_config_for_complexity(c).feature_flags
        assert prev_flags <= cur_flags, c
        prev_flags = cur_flags


def test_complexity_clamps_to_range():
    """Negative or above-10 dials shouldn't blow up."""
    a = query_config_for_complexity(-5)
    b = query_config_for_complexity(0)
    c = query_config_for_complexity(99)
    d = query_config_for_complexity(10)
    assert a == b
    assert c == d


def test_config_is_frozen_and_hashable():
    cfg = query_config_for_complexity(5)
    assert hash(cfg) is not None
    assert cfg == cfg


def test_probabilities_are_valid():
    """Sanity: every probability is in [0, 1]."""
    for c in range(0, 11):
        cfg = query_config_for_complexity(c)
        for p in (cfg.p_where, cfg.p_order_by, cfg.p_limit,
                  cfg.p_explicit_join, cfg.p_left_join_when_explicit,
                  cfg.p_aggregate_query, cfg.p_having):
            assert 0.0 <= p <= 1.0, (c, p)


def test_aggregate_query_probability_below_half():
    """Sanity: most queries should NOT aggregate. If this drifts above
    0.5, the milestone-1 non-aggregate code path would stop being the
    common case and seed-stability across milestones would suffer."""
    for c in range(0, 11):
        cfg = query_config_for_complexity(c)
        assert cfg.p_aggregate_query < 0.5, c
