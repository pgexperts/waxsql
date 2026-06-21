"""Tests for waxsql.types."""
from waxsql.types import (
    INT4, INT8, NUMERIC, FLOAT8, TEXT, VARCHAR, BOOL, JSONB,
    array_of, implicitly_castable,
)


def test_self_cast_always_true():
    for t in (INT4, INT8, NUMERIC, FLOAT8, TEXT, VARCHAR, BOOL, JSONB):
        assert implicitly_castable(t, t)


def test_implicit_int4_to_int8():
    assert implicitly_castable(INT4, INT8)


def test_no_implicit_int8_to_int4():
    assert not implicitly_castable(INT8, INT4)


def test_int_to_numeric_chain():
    assert implicitly_castable(INT4, NUMERIC)
    assert implicitly_castable(INT4, FLOAT8)
    assert implicitly_castable(NUMERIC, FLOAT8)


def test_varchar_to_text():
    assert implicitly_castable(VARCHAR, TEXT)
    # text → varchar is NOT implicit in PostgreSQL
    assert not implicitly_castable(TEXT, VARCHAR)


def test_arrays_match_only_on_element():
    assert implicitly_castable(array_of(INT4), array_of(INT4))
    assert not implicitly_castable(array_of(INT4), array_of(INT8))
    assert not implicitly_castable(array_of(INT4), INT4)
    assert not implicitly_castable(INT4, array_of(INT4))


def test_pgtype_sql_renders_typmod():
    from waxsql.types import PgType, TypeCategory
    n = PgType("numeric", TypeCategory.NUMERIC, typmod=(10, 2))
    assert n.sql() == "numeric(10,2)"


def test_array_sql_form():
    assert array_of(INT4).sql() == "int4[]"
    assert array_of(array_of(TEXT)).sql() == "text[][]"
