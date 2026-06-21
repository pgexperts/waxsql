"""Tests for per-PgType value strategies. See waxsql/gen/data/strategies.py."""
from __future__ import annotations

import datetime as _dt
import random
import uuid as _uuid
from decimal import Decimal

from waxsql.gen.data.strategies import pick_word, WORDLIST, strategy_for_type
from waxsql.types import (
    BOOL, DATE, FLOAT8, INT4, INT8, INTERVAL, JSONB, NUMERIC, TEXT, TIMESTAMPTZ,
    UUID, PgType, TypeCategory, array_of,
)


def test_wordlist_is_nonempty():
    assert len(WORDLIST) > 50


def test_wordlist_entries_are_simple_lowercase_words():
    for w in WORDLIST:
        assert w.islower()
        assert w.isalpha()
        assert 2 <= len(w) <= 20


def test_pick_word_is_deterministic():
    a = [pick_word(random.Random(7)) for _ in range(20)]
    b = [pick_word(random.Random(7)) for _ in range(20)]
    assert a == b


def test_pick_word_returns_a_word_from_the_list():
    word = pick_word(random.Random(1))
    assert word in WORDLIST


def _column(type_: PgType):
    """Minimal Column stand-in for strategies that don't consult the column."""
    from waxsql.schema import Column
    return Column(name="x", type=type_, nullable=False)


def test_int4_strategy_in_int4_range():
    rng = random.Random(0)
    strat = strategy_for_type(INT4)
    for _ in range(50):
        v = strat(rng, _column(INT4))
        assert isinstance(v, int)
        assert -(2**31) + 1 <= v <= (2**31) - 1


def test_int8_strategy_in_int8_range():
    rng = random.Random(0)
    strat = strategy_for_type(INT8)
    for _ in range(50):
        v = strat(rng, _column(INT8))
        assert isinstance(v, int)
        assert -(2**62) <= v <= (2**62)


def test_text_strategy_returns_word_from_wordlist():
    rng = random.Random(0)
    strat = strategy_for_type(TEXT)
    v = strat(rng, _column(TEXT))
    assert v in WORDLIST


def test_varchar_strategy_respects_typmod():
    short = PgType("varchar", TypeCategory.STRING, typmod=(5,))
    rng = random.Random(0)
    strat = strategy_for_type(short)
    for _ in range(50):
        v = strat(rng, _column(short))
        assert isinstance(v, str)
        assert len(v) <= 5


def test_bool_strategy_returns_bool():
    rng = random.Random(0)
    strat = strategy_for_type(BOOL)
    v = strat(rng, _column(BOOL))
    assert isinstance(v, bool)


def test_uuid_strategy_returns_uuid():
    rng = random.Random(0)
    strat = strategy_for_type(UUID)
    v = strat(rng, _column(UUID))
    assert isinstance(v, _uuid.UUID)


def test_strategies_are_deterministic_for_same_seed():
    """Same seed + same type → same value across two runs."""
    for ty in (INT4, INT8, TEXT, BOOL, UUID, NUMERIC, FLOAT8, DATE, TIMESTAMPTZ, INTERVAL, JSONB):
        strat = strategy_for_type(ty)
        a = [strat(random.Random(42), _column(ty)) for _ in range(5)]
        b = [strat(random.Random(42), _column(ty)) for _ in range(5)]
        assert a == b, ty.name


def test_float8_strategy_returns_float():
    rng = random.Random(0)
    strat = strategy_for_type(FLOAT8)
    v = strat(rng, _column(FLOAT8))
    assert isinstance(v, float)


def test_numeric_strategy_returns_decimal():
    rng = random.Random(0)
    strat = strategy_for_type(NUMERIC)
    v = strat(rng, _column(NUMERIC))
    assert isinstance(v, Decimal)


def test_numeric_strategy_respects_precision_and_scale():
    # numeric(5, 2): up to 5 digits total, 2 after the decimal point.
    # Magnitude must stay under 10^(5-2) = 1000.
    typed = PgType("numeric", TypeCategory.NUMERIC, typmod=(5, 2))
    rng = random.Random(0)
    strat = strategy_for_type(typed)
    for _ in range(50):
        v = strat(rng, _column(typed))
        assert isinstance(v, Decimal)
        assert abs(v) < Decimal("1000")
        # At most `scale` digits after the decimal:
        _, _, exp = v.as_tuple()
        assert exp >= -2, f"too many fractional digits: {v}"


def test_date_strategy_returns_date_near_epoch():
    rng = random.Random(0)
    strat = strategy_for_type(DATE)
    v = strat(rng, _column(DATE))
    assert isinstance(v, _dt.date)
    # Within +/- 5 years of the fixed epoch.
    assert _dt.date(2020, 1, 1) <= v <= _dt.date(2030, 1, 1)


def test_timestamptz_strategy_has_tzinfo():
    rng = random.Random(0)
    strat = strategy_for_type(TIMESTAMPTZ)
    v = strat(rng, _column(TIMESTAMPTZ))
    assert isinstance(v, _dt.datetime)
    assert v.tzinfo is not None


def test_interval_strategy_returns_timedelta():
    rng = random.Random(0)
    strat = strategy_for_type(INTERVAL)
    for _ in range(50):
        v = strat(rng, _column(INTERVAL))
        assert isinstance(v, _dt.timedelta)
        assert _dt.timedelta(0) <= v <= _dt.timedelta(days=120, seconds=86_399)


def test_jsonb_strategy_returns_dict():
    rng = random.Random(0)
    strat = strategy_for_type(JSONB)
    v = strat(rng, _column(JSONB))
    assert isinstance(v, dict)


def test_jsonb_keys_are_strings_values_are_scalars():
    rng = random.Random(0)
    strat = strategy_for_type(JSONB)
    for _ in range(20):
        v = strat(rng, _column(JSONB))
        assert all(isinstance(k, str) for k in v)
        for value in v.values():
            assert isinstance(value, (str, int, float, bool, type(None)))


def test_jsonb_strategy_size_bounded():
    rng = random.Random(0)
    strat = strategy_for_type(JSONB)
    for _ in range(20):
        v = strat(rng, _column(JSONB))
        assert 1 <= len(v) <= 4


def test_array_of_int4_strategy_returns_list_of_ints():
    arr_t = array_of(INT4)
    rng = random.Random(0)
    strat = strategy_for_type(arr_t)
    v = strat(rng, _column(arr_t))
    assert isinstance(v, list)
    assert all(isinstance(x, int) for x in v)


def test_array_strategy_length_bounded():
    arr_t = array_of(TEXT)
    rng = random.Random(0)
    strat = strategy_for_type(arr_t)
    for _ in range(50):
        v = strat(rng, _column(arr_t))
        assert 0 <= len(v) <= 5


def test_array_strategy_is_deterministic():
    arr_t = array_of(INT4)
    strat = strategy_for_type(arr_t)
    a = [strat(random.Random(99), _column(arr_t)) for _ in range(5)]
    b = [strat(random.Random(99), _column(arr_t)) for _ in range(5)]
    assert a == b
