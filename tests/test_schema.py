"""Tests for waxsql.schema.

The headline test is `test_schema_ddl_parses` which asserts that every
generated schema's DDL parses cleanly via pglast. This is the round-trip
guarantee — if we ever break the DDL emitter, this lights up immediately.
"""
import pytest

from waxsql import generate_schema, schema_config_for_complexity
from waxsql.validate.syntax import check_syntax


# --- DDL round-trip ---------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("complexity", [0, 3, 5, 7, 10])
def test_schema_ddl_parses(seed, complexity):
    """Every generated schema must produce DDL that pglast accepts."""
    schema = generate_schema(seed=seed, complexity=complexity)
    ddl = schema.emit_ddl()
    result = check_syntax(ddl)
    assert result.ok, (
        f"DDL did not parse at seed={seed}, complexity={complexity}\n"
        f"Error: {result.error}\n"
        f"At position: {result.error_position}\n"
        f"DDL:\n{ddl}"
    )


# --- Determinism ------------------------------------------------------------

def test_same_seed_same_schema():
    a = generate_schema(seed=42, complexity=5)
    b = generate_schema(seed=42, complexity=5)
    assert a == b
    assert a.emit_ddl() == b.emit_ddl()


def test_different_seeds_differ():
    a = generate_schema(seed=1, complexity=5)
    b = generate_schema(seed=2, complexity=5)
    assert a != b


# --- Structural invariants --------------------------------------------------

def test_complexity_scales_table_count():
    small = generate_schema(seed=0, complexity=0)
    large = generate_schema(seed=0, complexity=10)
    assert len(small.tables) < len(large.tables)


def test_every_table_has_pk():
    for seed in range(5):
        schema = generate_schema(seed=seed, complexity=7)
        for t in schema.tables:
            assert t.primary_key, f"{t.name} has no PK"
            assert "id" in t.primary_key


def test_table_names_unique():
    for seed in range(10):
        schema = generate_schema(seed=seed, complexity=10)
        names = [t.name for t in schema.tables]
        assert len(names) == len(set(names))


def test_column_names_unique_within_table():
    for seed in range(10):
        schema = generate_schema(seed=seed, complexity=10)
        for t in schema.tables:
            names = [c.name for c in t.columns]
            assert len(names) == len(set(names)), (
                f"Duplicate column in {t.name}: {names}"
            )


def test_fks_reference_real_tables_and_columns():
    for seed in range(10):
        schema = generate_schema(seed=seed, complexity=7)
        table_names = {t.name for t in schema.tables}
        for t in schema.tables:
            for fk in t.foreign_keys:
                assert fk.ref_table in table_names
                ref_t = schema.table(fk.ref_table)
                ref_col_names = {c.name for c in ref_t.columns}
                for rc in fk.ref_columns:
                    assert rc in ref_col_names


def test_fk_columns_exist_on_source_table():
    for seed in range(10):
        schema = generate_schema(seed=seed, complexity=7)
        for t in schema.tables:
            col_names = {c.name for c in t.columns}
            for fk in t.foreign_keys:
                for c in fk.columns:
                    assert c in col_names


def test_fks_acyclic_at_low_complexity():
    """At complexity < 8 (allow_cyclic_fks=False), FKs only point to
    earlier tables (or self if allow_self_fks holds)."""
    for seed in range(10):
        schema = generate_schema(seed=seed, complexity=5)
        cfg = schema_config_for_complexity(5)
        assert not cfg.allow_cyclic_fks  # sanity: precondition for the test
        idx = {t.name: i for i, t in enumerate(schema.tables)}
        for i, t in enumerate(schema.tables):
            for fk in t.foreign_keys:
                ref_idx = idx[fk.ref_table]
                if cfg.allow_self_fks:
                    assert ref_idx <= i
                else:
                    assert ref_idx < i


def test_index_names_unique_within_table():
    for seed in range(10):
        schema = generate_schema(seed=seed, complexity=8)
        for t in schema.tables:
            names = [ix.name for ix in t.indexes]
            assert len(names) == len(set(names)), (
                f"Duplicate index in {t.name}: {names}"
            )


# --- Negative test for the validator itself --------------------------------

def test_syntax_rejects_garbage():
    """Sanity check: the validator actually rejects malformed SQL."""
    result = check_syntax("SELECT FROM WHERE GROUP BY HAVING")
    assert not result.ok
    assert result.error
