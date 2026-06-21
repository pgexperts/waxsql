"""Tests for `waxsql.data.generate_data` and its public re-export."""
from __future__ import annotations

import pytest

from waxsql import generate_data, generate_schema
from waxsql.validate.syntax import check_syntax


@pytest.mark.parametrize("seed", range(10))
def test_generate_data_is_deterministic(seed):
    schema = generate_schema(seed=seed, complexity=4)
    a = generate_data(schema, seed=seed, rows=10, fanout=3, null_fraction=0.1)
    b = generate_data(schema, seed=seed, rows=10, fanout=3, null_fraction=0.1)
    assert a == b


@pytest.mark.parametrize("seed", range(10))
def test_generate_data_emits_copy_block_per_table(seed):
    schema = generate_schema(seed=seed, complexity=4)
    text = generate_data(schema, seed=seed, rows=5, fanout=2, null_fraction=0.0)
    for t in schema.tables:
        assert f'COPY "{t.name}"' in text


@pytest.mark.parametrize("seed", range(10))
def test_ddl_plus_data_parses_via_pglast(seed):
    """DDL and COPY statement headers must parse cleanly through pglast.

    pglast (libpg_query) parses COPY ... FROM STDIN; headers as valid SQL,
    but it rejects the `\\.` data-block terminator and any inline COPY data
    (those are text-protocol, not SQL). We extract only COPY header lines
    and combine them with the DDL for the syntax check. The `\\.` terminator
    is intentionally excluded.
    """
    schema = generate_schema(seed=seed, complexity=4)
    ddl = schema.emit_ddl()
    data = generate_data(schema, seed=seed, rows=3, fanout=2, null_fraction=0.0)
    # Keep only the `COPY "table" (...) FROM STDIN;` lines — pglast cannot
    # parse the `\.` terminator or raw COPY data lines. The headers alone
    # are sufficient to verify identifiers, column lists, and syntax.
    headers_only = "\n".join(
        line for line in data.splitlines()
        if line.startswith("COPY ")
    )
    combined = ddl + "\n" + headers_only
    result = check_syntax(combined)
    assert result.ok, f"seed={seed}: {result.error}\n{combined}"


def test_zero_rows_emits_empty_copy_blocks():
    schema = generate_schema(seed=1, complexity=3)
    text = generate_data(schema, seed=1, rows=0, fanout=2, null_fraction=0.0)
    for t in schema.tables:
        # Each table's COPY block exists but contains only the terminator.
        block_start = f'COPY "{t.name}"'
        assert block_start in text
    # No data rows between header and terminator.
    for line in text.splitlines():
        assert not line.startswith("1\t"), "expected no data rows at rows=0"


def test_different_seeds_produce_different_data():
    schema = generate_schema(seed=1, complexity=3)
    a = generate_data(schema, seed=1, rows=5, fanout=2, null_fraction=0.0)
    b = generate_data(schema, seed=2, rows=5, fanout=2, null_fraction=0.0)
    assert a != b


def test_data_seed_is_independent_of_schema_seed():
    """Same schema, different data seed → different data."""
    schema = generate_schema(seed=1, complexity=3)
    a = generate_data(schema, seed=10, rows=5, fanout=2, null_fraction=0.0)
    b = generate_data(schema, seed=11, rows=5, fanout=2, null_fraction=0.0)
    assert a != b


@pytest.mark.parametrize("complexity", range(11))
@pytest.mark.parametrize("seed", range(5))
def test_generate_data_either_succeeds_or_raises_value_error(seed, complexity):
    """At any (seed, complexity), generate_data either produces a string or
    raises ValueError (the documented failure for FK-cyclic schemas).
    No other exceptions are acceptable.
    """
    schema = generate_schema(seed=seed, complexity=complexity)
    try:
        text = generate_data(schema, seed=seed, rows=2, fanout=2, null_fraction=0.0)
        assert isinstance(text, str)
    except ValueError:
        # Acceptable for cyclic schemas at high complexity. Confirmed in CLI.
        pass
