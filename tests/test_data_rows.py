"""Tests for topo order and row materialization. See waxsql/gen/data/rows.py."""
from __future__ import annotations

import random

import pytest

from waxsql.gen.data.rows import (
    depth_of,
    generate_row,
    rows_for_table,
    topological_order,
)
from waxsql.schema import generate_schema


@pytest.mark.parametrize("seed", range(10))
def test_topological_order_visits_every_table_once(seed):
    schema = generate_schema(seed=seed, complexity=5)
    order = topological_order(schema)
    names_in_order = [t.name for t in order]
    schema_names = [t.name for t in schema.tables]
    assert sorted(names_in_order) == sorted(schema_names)


@pytest.mark.parametrize("seed", range(10))
def test_topological_order_parents_precede_children(seed):
    schema = generate_schema(seed=seed, complexity=5)
    order = topological_order(schema)
    position = {t.name: i for i, t in enumerate(order)}
    for t in schema.tables:
        for fk in t.foreign_keys:
            # Self-referential FKs impose no ordering constraint; skip them.
            if fk.ref_table == t.name:
                continue
            assert position[fk.ref_table] < position[t.name], (
                f"FK from {t.name} → {fk.ref_table} violates topo order"
            )


@pytest.mark.parametrize("seed", range(10))
def test_topological_order_is_deterministic(seed):
    schema = generate_schema(seed=seed, complexity=5)
    a = [t.name for t in topological_order(schema)]
    b = [t.name for t in topological_order(schema)]
    assert a == b


def test_depth_of_root_is_zero():
    schema = generate_schema(seed=1, complexity=5)
    roots = [t for t in schema.tables if not t.foreign_keys]
    assert roots, "expected at least one root table"
    for r in roots:
        assert depth_of(r, schema) == 0


def test_depth_of_child_exceeds_parent():
    schema = generate_schema(seed=1, complexity=5)
    for t in schema.tables:
        for fk in t.foreign_keys:
            # Self-referential FKs don't create a parent/child depth gap.
            if fk.ref_table == t.name:
                continue
            parent = next(p for p in schema.tables if p.name == fk.ref_table)
            assert depth_of(t, schema) > depth_of(parent, schema)


# ---------------------------------------------------------------------------
# Task 11: generate_row + rows_for_table
# ---------------------------------------------------------------------------


def test_generate_row_uses_pk_from_argument():
    schema = generate_schema(seed=1, complexity=2)
    table = next(t for t in schema.tables if not t.foreign_keys)
    row = generate_row(
        table=table,
        pk=42,
        rng=random.Random(0),
        id_store={},
        null_fraction=0.0,
    )
    # PK column is at the conventional first position.
    assert row[0] == 42


def test_generate_row_resolves_fk_from_id_store():
    schema = generate_schema(seed=1, complexity=4)
    # Find a table with at least one cross-table FK (skip pure self-FK tables).
    child = next(
        (
            t for t in schema.tables
            if any(fk.ref_table != t.name for fk in t.foreign_keys)
        ),
        None,
    )
    if child is None:
        pytest.skip("schema has no cross-table FK-bearing table at this seed/complexity")
    fk = next(fk for fk in child.foreign_keys if fk.ref_table != child.name)
    parent_name = fk.ref_table
    parent_ids = [10, 20, 30, 40, 50]
    # Populate id_store with all tables the child touches so no KeyError.
    id_store = {
        other_fk.ref_table: [1, 2, 3]
        for other_fk in child.foreign_keys
        if other_fk.ref_table != child.name
    }
    id_store[parent_name] = parent_ids
    row = generate_row(
        table=child,
        pk=1,
        rng=random.Random(0),
        id_store=id_store,
        null_fraction=0.0,
    )
    # FK column value must come from the parent's IDs.
    fk_col_idx = [c.name for c in child.columns].index(fk.columns[0])
    assert row[fk_col_idx] in parent_ids


def test_generate_row_emits_nulls_for_nullable_columns():
    schema = generate_schema(seed=2, complexity=4)
    table = schema.tables[0]
    nullable = [c for c in table.columns if c.nullable]
    if not nullable:
        pytest.skip("no nullable columns at this seed/complexity")
    saw_null = False
    rng = random.Random(0)
    for i in range(200):
        row = generate_row(
            table=table,
            pk=i,
            rng=rng,
            id_store={t.name: [1] for t in schema.tables},
            null_fraction=0.5,
        )
        if any(v is None for c, v in zip(table.columns, row, strict=True) if c.nullable):
            saw_null = True
            break
    assert saw_null, "expected at least one NULL across 200 trials at 50% null_fraction"


def test_generate_row_never_emits_null_for_not_null_columns():
    schema = generate_schema(seed=2, complexity=4)
    table = schema.tables[0]
    not_null_cols = [c for c in table.columns if not c.nullable]
    assert not_null_cols, "expected at least one NOT NULL column (PK)"
    for i in range(50):
        row = generate_row(
            table=table,
            pk=i,
            rng=random.Random(i),
            id_store={t.name: [1] for t in schema.tables},
            null_fraction=0.99,
        )
        for c, v in zip(table.columns, row, strict=True):
            if not c.nullable:
                assert v is not None, f"NOT NULL column {c.name} got NULL"


def test_rows_for_table_uses_fanout_by_depth():
    schema = generate_schema(seed=1, complexity=5)
    base = 10
    fanout = 4
    for t in schema.tables:
        n = rows_for_table(t, schema, base=base, fanout=fanout)
        assert n >= base


def test_generate_row_handles_self_fk_when_not_nullable():
    """Self-FK columns reference rows [1..pk]; row 1 references itself when NOT NULL."""
    schema = generate_schema(seed=1, complexity=5)
    self_fk_tables = [
        t for t in schema.tables
        if any(fk.ref_table == t.name for fk in t.foreign_keys)
    ]
    if not self_fk_tables:
        pytest.skip("no self-FK tables at this seed/complexity")
    for t in self_fk_tables:
        # Build a clean id_store WITHOUT this table (simulating mid-materialization).
        id_store: dict[str, list[int]] = {
            other.name: [1, 2, 3] for other in schema.tables if other.name != t.name
        }
        # Row pk=1 — can only sample [1..1] for NOT NULL self-FK.
        row = generate_row(
            table=t,
            pk=1,
            rng=random.Random(0),
            id_store=id_store,
            null_fraction=0.0,
        )
        # All self-FK columns are either NULL (if nullable) or 1 (only valid choice).
        for fk in t.foreign_keys:
            if fk.ref_table != t.name:
                continue
            for fk_col in fk.columns:
                idx = [c.name for c in t.columns].index(fk_col)
                col = next(c for c in t.columns if c.name == fk_col)
                if col.nullable:
                    assert row[idx] in (None, 1)
                else:
                    assert row[idx] == 1
