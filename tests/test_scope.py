"""Tests for waxsql.scope.

Scope is small but its visibility rules are easy to get subtly wrong,
so the tests cover each rule explicitly: insertion order, type-cast
filtering, parent-chain walking, correlation gating, and the
duplicate-alias error.
"""
import pytest

from waxsql.scope import Binding, Scope
from waxsql.schema import Column, Table
from waxsql.types import INT4, INT8, NUMERIC, TEXT


def _t(name: str, *cols: Column) -> Table:
    """Tiny helper: build a Table with the given columns and an
    implicit `id BIGINT NOT NULL` PK matching the schema generator."""
    return Table(
        name=name,
        columns=(Column("id", INT8, nullable=False),) + cols,
        primary_key=("id",),
    )


def test_empty_scope_has_no_visible_columns():
    s = Scope()
    assert s.visible_columns() == []
    assert s.aliased_tables() == []


def test_add_table_populates_bindings_in_column_order():
    t = _t("customers", Column("name", TEXT), Column("age", INT4))
    s = Scope()
    s.add_table("c", t)
    bindings = s.visible_columns()
    # Expected order: id, name, age (the table's column order).
    assert [b.column for b in bindings] == ["id", "name", "age"]
    assert all(b.table_alias == "c" for b in bindings)


def test_visible_columns_filter_by_implicit_cast():
    """Asking for NUMERIC should yield int4 and int8 (both cast),
    plus numeric itself if present. Text-only columns are excluded."""
    t = _t("x",
           Column("a", INT4),
           Column("b", INT8),
           Column("c", NUMERIC),
           Column("name", TEXT))
    s = Scope()
    s.add_table("x", t)
    nums = {b.column for b in s.visible_columns(of_type=NUMERIC)}
    # id is INT8, also castable to numeric.
    assert nums == {"id", "a", "b", "c"}
    texts = {b.column for b in s.visible_columns(of_type=TEXT)}
    assert texts == {"name"}


def test_duplicate_alias_in_one_scope_raises():
    """PostgreSQL forbids duplicate FROM aliases; the scope mirrors
    the rule to catch generator bugs at construction time."""
    t = _t("x")
    s = Scope()
    s.add_table("x", t)
    with pytest.raises(ValueError):
        s.add_table("x", t)


def test_correlated_child_sees_parent_bindings():
    parent_t = _t("p", Column("name", TEXT))
    child_t = _t("c", Column("count", INT4))
    parent = Scope()
    parent.add_table("p", parent_t)
    child = parent.push_subquery(correlated=True)
    child.add_table("c", child_t)
    cols = {(b.table_alias, b.column) for b in child.visible_columns()}
    assert ("p", "name") in cols
    assert ("c", "count") in cols


def test_uncorrelated_child_does_not_see_parent_bindings():
    """Non-LATERAL FROM subqueries can't see their siblings/outer."""
    parent_t = _t("p", Column("name", TEXT))
    child_t = _t("c", Column("count", INT4))
    parent = Scope()
    parent.add_table("p", parent_t)
    child = parent.push_subquery(correlated=False)
    child.add_table("c", child_t)
    cols = {(b.table_alias, b.column) for b in child.visible_columns()}
    assert ("p", "name") not in cols
    assert ("c", "count") in cols


def test_correlation_chain_stops_at_first_uncorrelated_link():
    """Outer -> uncorrelated -> correlated should not see outer."""
    outer_t = _t("o", Column("ox", INT4))
    mid_t = _t("m", Column("mx", INT4))
    inner_t = _t("i", Column("ix", INT4))
    outer = Scope()
    outer.add_table("o", outer_t)
    mid = outer.push_subquery(correlated=False)
    mid.add_table("m", mid_t)
    inner = mid.push_subquery(correlated=True)
    inner.add_table("i", inner_t)
    aliases = {b.table_alias for b in inner.visible_columns()}
    assert aliases == {"i", "m"}  # outer is blocked by mid


def test_aliased_tables_is_scope_local_in_insertion_order():
    """For FK-biased JOIN generation, we want this scope's tables
    only, in the order they were added."""
    a = _t("a", Column("x", INT4))
    b = _t("b", Column("y", INT4))
    parent = Scope()
    parent.add_table("p", a)
    s = parent.push_subquery(correlated=True)
    s.add_table("a", a)
    s.add_table("b", b)
    pairs = s.aliased_tables()
    assert [alias for alias, _ in pairs] == ["a", "b"]
    # Parent's "p" alias is NOT included.
    assert all(alias != "p" for alias, _ in pairs)


def test_lookup_alias_returns_table_or_none():
    t = _t("x", Column("a", INT4))
    s = Scope()
    s.add_table("xx", t)
    assert s.lookup_alias("xx") is t
    assert s.lookup_alias("yy") is None


def test_binding_is_frozen_and_hashable():
    """Bindings should behave like the rest of the model objects:
    structurally comparable, hashable, immutable."""
    a = Binding("t1", "id", INT8, nullable=False)
    b = Binding("t1", "id", INT8, nullable=False)
    assert a == b
    assert {a, b} == {a}
    with pytest.raises(Exception):
        a.column = "foo"  # type: ignore[misc]


def test_visible_columns_returns_a_fresh_list_each_call():
    """Caller may mutate the returned list (e.g. shuffle/sample)
    without affecting future lookups."""
    t = _t("x", Column("a", INT4))
    s = Scope()
    s.add_table("x", t)
    a = s.visible_columns()
    a.clear()
    assert s.visible_columns()  # the clear() did not affect the scope


# --- DerivedTable scope semantics (milestone 4) ----------------------------

def test_add_derived_creates_bindings_with_synthetic_columns():
    """A derived alias registers its columns as bindings; the
    expression generator can pick them via visible_columns the same
    way it picks base-table columns."""
    s = Scope()
    s.add_derived("sq", [("c1", INT8), ("c2", TEXT)])
    bindings = s.visible_columns()
    assert {(b.table_alias, b.column, b.type) for b in bindings} == {
        ("sq", "c1", INT8),
        ("sq", "c2", TEXT),
    }


def test_derived_columns_are_always_nullable():
    """We don't propagate NOT NULL through SELECT-list expressions —
    derived columns get nullable=True regardless of inner provenance."""
    s = Scope()
    s.add_derived("sq", [("c1", INT8)])
    [b] = s.visible_columns()
    assert b.nullable is True


def test_aliased_tables_excludes_derived():
    """FK biasing only applies to base tables; aliased_tables()
    filters out derived-table aliases."""
    base = _t("orders")
    s = Scope()
    s.add_table("t1", base)
    s.add_derived("sq", [("c1", INT8)])
    pairs = s.aliased_tables()
    assert [(a, t.name) for a, t in pairs] == [("t1", "orders")]


def test_lookup_alias_returns_none_for_derived():
    """Per the docstring contract: lookup_alias returns None for both
    'absent' and 'present but derived'. has_alias is the unambiguous
    check."""
    base = _t("orders")
    s = Scope()
    s.add_table("t1", base)
    s.add_derived("sq", [("c1", INT8)])
    assert s.lookup_alias("t1") is base
    assert s.lookup_alias("sq") is None         # derived → None
    assert s.lookup_alias("missing") is None    # absent → None
    assert s.has_alias("t1") is True
    assert s.has_alias("sq") is True
    assert s.has_alias("missing") is False


def test_add_derived_rejects_alias_collision_with_base():
    """Cross-kind collision: same alias used for both a base table
    and a derived would shadow at PG resolution time. Reject early."""
    base = _t("orders")
    s = Scope()
    s.add_table("x", base)
    with pytest.raises(ValueError):
        s.add_derived("x", [("c1", INT8)])


def test_add_table_rejects_alias_collision_with_derived():
    """Symmetric direction: derived already in scope, then base
    tries to take the same alias."""
    s = Scope()
    s.add_derived("x", [("c1", INT8)])
    with pytest.raises(ValueError):
        s.add_table("x", _t("orders"))


def test_correlated_child_sees_derived_columns_from_parent():
    """A LATERAL subquery's child scope (correlated=True) walks up
    to the parent — derived-table columns introduced in the parent
    must be visible in the child."""
    parent = Scope()
    parent.add_derived("sq", [("c1", INT8)])
    child = parent.push_subquery(correlated=True)
    cols = {(b.table_alias, b.column) for b in child.visible_columns()}
    assert ("sq", "c1") in cols


# --- CTE management (milestone 5) ------------------------------------------

def test_add_cte_creates_lookup_match():
    """A CTE registered under a name should be retrievable by that
    name with the same column info."""
    s = Scope()
    s.add_cte("cte1", [("c1", INT8)])
    assert s.lookup_cte("cte1") == [("c1", INT8)]


def test_lookup_cte_returns_none_for_undefined():
    s = Scope()
    assert s.lookup_cte("nonexistent") is None


def test_add_cte_rejects_duplicate_name():
    """PG enforces CTE name uniqueness within a single WITH clause."""
    s = Scope()
    s.add_cte("cte1", [("c1", INT8)])
    with pytest.raises(ValueError):
        s.add_cte("cte1", [("c1", TEXT)])


def test_lookup_cte_walks_parent_chain():
    """CTEs defined in outer scopes must be visible from nested
    scopes — that's the static-scoping rule for CTEs."""
    parent = Scope()
    parent.add_cte("cte_outer", [("c1", INT8)])
    child = parent.push_subquery(correlated=True)
    assert child.lookup_cte("cte_outer") == [("c1", INT8)]


def test_lookup_cte_walks_chain_even_when_uncorrelated():
    """CTE visibility is independent of correlation/LATERAL semantics:
    even a non-LATERAL FROM subquery (correlated=False) still sees
    outer CTEs. That's the key difference between CTE visibility
    (static) and column visibility (correlation-gated)."""
    parent = Scope()
    parent.add_cte("cte_outer", [("c1", INT8)])
    # Verify columns DO get gated by correlated=False
    parent.add_derived("derived", [("c1", INT8)])
    uncorrelated = parent.push_subquery(correlated=False)

    # CTE should still be visible despite uncorrelated=True
    assert uncorrelated.lookup_cte("cte_outer") == [("c1", INT8)]
    # But derived columns should NOT be visible (correlated=False)
    cols = {(b.table_alias, b.column) for b in uncorrelated.visible_columns()}
    assert ("derived", "c1") not in cols


def test_has_visible_ctes_is_true_when_any_in_chain():
    """Sanity: has_visible_ctes finds CTEs anywhere up the chain."""
    parent = Scope()
    child = parent.push_subquery(correlated=False)
    assert child.has_visible_ctes() is False  # nothing defined yet
    parent.add_cte("c", [("c1", INT8)])
    assert child.has_visible_ctes() is True


def test_visible_cte_names_returns_chain():
    """Names in insertion order, child first then parent's."""
    parent = Scope()
    parent.add_cte("outer1", [("c1", INT8)])
    parent.add_cte("outer2", [("c1", TEXT)])
    child = parent.push_subquery(correlated=True)
    child.add_cte("inner", [("c1", INT8)])
    names = child.visible_cte_names()
    # Child's first, then parent's, in insertion order within each
    assert names == ["inner", "outer1", "outer2"]


def test_visible_cte_names_dedupes_shadowed_names():
    """When a child scope defines a CTE that shadows a parent's CTE
    by the same name, the visible-names list should contain that
    name exactly once — and the child (closer) binding should win.

    Today's generator only emits top-level WITHs, so this never
    fires in the wild; the test exists as a latent-correctness
    guard for the eventual nested-WITH path. Without dedupe, a
    caller picking a CTE name from this list could resolve to the
    outer binding by accident — `lookup_cte` walks closest-first
    and would actually return the inner CTE, producing a
    name/binding mismatch.
    """
    parent = Scope()
    parent.add_cte("shared", [("c1", INT8)])
    parent.add_cte("outer_only", [("c1", INT8)])
    child = parent.push_subquery(correlated=True)
    child.add_cte("shared", [("c1", TEXT)])  # shadows parent's "shared"
    child.add_cte("inner_only", [("c2", INT8)])
    names = child.visible_cte_names()
    # 'shared' appears once: child's binding wins on first-seen order.
    # Then the rest of the child's names, then the parent's remaining
    # non-shadowed names — parent's 'shared' suppressed by dedupe.
    assert names == ["shared", "inner_only", "outer_only"]
    # And lookup_cte resolves 'shared' to the closer (child) binding.
    assert child.lookup_cte("shared") == [("c1", TEXT)]


def test_lookup_cte_returns_fresh_list_each_call():
    """Caller can mutate the returned list without affecting future
    lookups — same invariant as visible_columns."""
    s = Scope()
    s.add_cte("cte1", [("c1", INT8)])
    a = s.lookup_cte("cte1")
    a.append(("rogue", TEXT))
    # The next lookup should NOT include the rogue entry.
    assert s.lookup_cte("cte1") == [("c1", INT8)]
