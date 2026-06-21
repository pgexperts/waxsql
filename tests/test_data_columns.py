"""Tests for the column-name override registry. See waxsql/gen/data/columns.py."""
from __future__ import annotations

import random

from waxsql.gen.data.columns import strategy_for, _NAME_PATTERNS
from waxsql.gen.data.strategies import strategy_for_type
from waxsql.schema import Column
from waxsql.types import TEXT


def test_registry_is_a_tuple():
    """Ordering must be stable across Python versions; tuple not dict."""
    assert isinstance(_NAME_PATTERNS, tuple)


def test_strategy_for_falls_through_to_type_when_no_pattern_matches():
    col = Column(name="random_column_xyz", type=TEXT, nullable=False)
    strat = strategy_for(col)
    expected = strategy_for_type(TEXT)
    # Same identity, same behavior: applied to the same rng, identical output.
    rng_a = random.Random(0)
    rng_b = random.Random(0)
    assert strat(rng_a, col) == expected(rng_b, col)


def test_strategy_for_returns_strategy_consistent_with_type():
    col = Column(name="anything", type=TEXT, nullable=False)
    strat = strategy_for(col)
    rng = random.Random(0)
    v = strat(rng, col)
    assert isinstance(v, str)
