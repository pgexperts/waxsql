"""Tests for waxsql.catalog."""
from waxsql.catalog import default_catalog
from waxsql.types import INT4, INT8, NUMERIC, TEXT, BOOL, TIMESTAMPTZ


def test_catalog_loads():
    cat = default_catalog()
    assert len(cat.functions) > 30
    assert len(cat.operators) > 30


def test_aggs_returning_int8_includes_count_and_sum():
    cat = default_catalog()
    aggs = cat.aggs_returning(INT8)
    names = {f.name for f in aggs}
    assert "count" in names
    assert "sum" in names


def test_scalar_funcs_returning_text():
    cat = default_catalog()
    fs = cat.scalar_funcs_returning(TEXT)
    names = {f.name for f in fs}
    assert "upper" in names
    assert "lower" in names
    assert "concat" in names


def test_window_funcs_present():
    cat = default_catalog()
    wins = cat.window_funcs_returning(INT8)
    names = {f.name for f in wins}
    assert "row_number" in names
    assert "rank" in names


def test_srfs_present():
    cat = default_catalog()
    srfs = cat.srfs_returning(INT4)
    names = {f.name for f in srfs}
    assert "generate_series" in names


def test_binary_ops_returning_bool():
    cat = default_catalog()
    ops = cat.binary_ops_returning(BOOL)
    syms = {o.symbol for o in ops}
    assert "=" in syms
    assert "<" in syms
    assert "AND" in syms
    assert "LIKE" in syms


def test_binary_ops_returning_numeric():
    cat = default_catalog()
    ops = cat.binary_ops_returning(NUMERIC)
    syms = {o.symbol for o in ops}
    assert "+" in syms
    assert "*" in syms
    assert "%" in syms


def test_temporal_arithmetic_present():
    cat = default_catalog()
    ops = cat.binary_ops_returning(TIMESTAMPTZ)
    # TIMESTAMPTZ ± INTERVAL → TIMESTAMPTZ
    assert any(o.symbol == "+" and o.left.name == "timestamptz" for o in ops)


def test_lookup_results_are_deterministic():
    cat = default_catalog()
    a = [(f.name, f.kind.value, tuple(x.name for x in f.args))
         for f in cat.funcs_returning(INT4)]
    b = [(f.name, f.kind.value, tuple(x.name for x in f.args))
         for f in cat.funcs_returning(INT4)]
    assert a == b


def test_funcs_returning_walks_implicit_casts():
    """A NUMERIC-returning function should be visible when asking for FLOAT8."""
    from waxsql.types import FLOAT8
    cat = default_catalog()
    fs = cat.funcs_returning(FLOAT8)
    names = {f.name for f in fs}
    # round() returns NUMERIC, which casts to FLOAT8
    assert "round" in names
