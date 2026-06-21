"""Per-table row materialization and FK resolution.

The walk is one topological pass over the FK DAG. For each table, we
materialize all rows in memory as Python tuples, capture the PK column
values into a per-table ID store, and hand the tuples to the emitter.
Children resolve FK columns by sampling from the parent's ID list,
which is guaranteed populated by the topological order.

Self-referential FKs (a column in table T referencing T.id) are handled
as a special case: row with pk=N can reference any row in [1..N] including
itself, because PostgreSQL checks FK constraints per-row by default (NOT
DEFERRABLE). Nullable self-FK columns honour null_fraction (the null roll
fires above, before the FK branch, and controls whether the column is NULL);
NOT NULL self-FK columns always sample from [1..pk].

In-memory materialization is fine at demo scales (default 100 rows ×
fanout 5 across a few levels of depth is comfortably under a megabyte).
If volume ever becomes a real constraint, a streaming variant can be
added behind the same public API.
"""
from __future__ import annotations

import random
from collections.abc import Mapping

from waxsql.gen.data.columns import strategy_for
from waxsql.schema import Schema, Table


def topological_order(schema: Schema) -> list[Table]:
    """Return the schema's tables in FK-topological order: every table
    appears before any table that has an FK referencing it.

    Within each topological rank, tables are sorted alphabetically by
    name. This eliminates dict-insertion-order dependency and keeps the
    walk byte-identical across Python builds.

    Raises ValueError if a cycle is detected (the current generator does
    not produce cycles, but we assert loudly rather than loop forever).

    Algorithm: Kahn-style level-set walk. Each iteration extracts the
    set of tables whose remaining parents are empty (i.e. all their FK
    targets have already been placed), sorts that set alphabetically,
    and appends them. A "no ready tables but remaining is non-empty"
    state means a cycle and raises.
    """
    by_name = {t.name: t for t in schema.tables}
    # Collect UNIQUE external parents per table: FKs that point to a
    # *different* table. Self-referential FKs (`t → t`) are intentionally
    # excluded because they impose no ordering constraint — the table
    # trivially appears before itself.
    parents: dict[str, set[str]] = {
        t.name: {fk.ref_table for fk in t.foreign_keys if fk.ref_table != t.name}
        for t in schema.tables
    }
    remaining = set(by_name)
    out: list[Table] = []
    while remaining:
        # sorted(...) is essential here, not cosmetic: `set` iteration
        # order is not stable across Python builds (hash-randomized for
        # strings since 3.3), and any set-driven choice would silently
        # break the determinism contract.
        ready = sorted(
            n for n in remaining
            if not (parents[n] & remaining)
        )
        if not ready:
            # No table has all its parents already placed AND `remaining`
            # is non-empty → cycle. The data generator can't currently
            # handle cycles (would need deferred constraints + UPDATE
            # patches); the CLI catches this ValueError and produces a
            # clean usage message pointing at the cycle.
            raise ValueError(
                f"FK cycle in schema: remaining {sorted(remaining)!r}"
            )
        for n in ready:
            out.append(by_name[n])
            remaining.discard(n)
    return out


def depth_of(table: Table, schema: Schema) -> int:
    """The longest FK chain from any root to `table`.

    Roots have depth 0. A table directly referencing a root has depth 1.
    When a table has multiple FK parents, depth is 1 + max(parent depths)
    — the longest chain, not the shortest.

    Depth feeds the row-count formula: `rows * fanout ** depth`. Choosing
    longest-chain (not shortest) means a leaf with even one deep ancestor
    gets the deep row count — appropriate, since the leaf has to spread
    across the parent's larger ID pool to avoid clumping.

    The internal `memo` dict is local to this call, so memoization
    avoids re-walking shared ancestry WITHIN a single invocation but
    is NOT shared across calls. `rows_for_table` calls `depth_of`
    once per table in the schema, so the total work for a full
    data-generation pass is O(N · D) where N is the number of
    tables and D the maximum chain length — fine at the schema
    generator's N=12 ceiling. Promote to a shared memo threaded
    through `topological_order`/`rows_for_table` only if N grows
    materially. The `stack` parameter threads the current path for
    cycle detection; a cycle raises ValueError rather than
    recursing to Python's stack limit.
    """
    by_name = {t.name: t for t in schema.tables}
    memo: dict[str, int] = {}

    def visit(name: str, stack: tuple[str, ...] = ()) -> int:
        if name in stack:
            # Cycle detected on the recursion path itself — different
            # from the topological_order cycle check, which is global.
            # Tuple-based stack is cheap (small N) and keeps the path
            # for the error message.
            raise ValueError(f"FK cycle through {name}: {stack}")
        if name in memo:
            return memo[name]
        t = by_name[name]
        # Only consider FKs to *other* tables. Self-referential FKs
        # (`t → t`) would recurse forever; they carry no depth information
        # because the row materializer handles self-references separately
        # (sample from [1..pk] in the same row generation).
        external_parents = [fk.ref_table for fk in t.foreign_keys if fk.ref_table != name]
        d = 0 if not external_parents else 1 + max(
            visit(p, stack + (name,)) for p in external_parents
        )
        memo[name] = d
        return d

    return visit(table.name)


# PK column name convention is fixed by the schema generator: every table
# has an `id BIGINT NOT NULL` PK at the first column position. We don't
# search for it dynamically; if that invariant ever changes, this module
# will produce nonsense for that table and tests will fail loudly, which
# is the desired behavior.
_PK_COLUMN_NAME = "id"


def rows_for_table(table: Table, schema: Schema, *, base: int, fanout: int) -> int:
    """Row count for `table`: `base * fanout ** depth`. No ceiling; the
    user is responsible for not asking for combinations that produce
    absurd output. Depth is 0 for root tables, so they always get exactly
    `base` rows regardless of fanout.

    Why the power law: it lets child tables outnumber their parents
    geometrically, which is what real one-to-many shapes look like
    (orders >> customers, line_items >> orders). With base=100, fanout=5,
    depth=3: 12500 rows — large enough to exercise EXPLAIN's choices,
    small enough to fit comfortably in memory and load in seconds.
    """
    return base * fanout ** depth_of(table, schema)


def generate_row(
    *,
    table: Table,
    pk: int,
    rng: random.Random,
    id_store: Mapping[str, list[int]],
    null_fraction: float,
) -> tuple[object, ...]:
    """Generate one row for `table` as a Python tuple.

    PK column gets `pk`; FK columns sample from `id_store[ref_table]`;
    other columns dispatch through `strategy_for`. Nullable columns —
    including nullable FK columns — roll for NULL first; only columns
    that survive the null roll reach FK or strategy dispatch.

    Self-referential FKs bypass `id_store` entirely: PostgreSQL's default
    NOT DEFERRABLE constraint checking allows row pk=N to reference any
    row in [1..N] (itself included, since the row exists by check time).
    Both nullable and non-nullable self-FK columns sample from [1..pk] once
    they reach the FK branch — the null roll above already gave nullable
    columns their chance to become NULL, so null_fraction is honoured
    correctly instead of forcing self-FK nullables to always be NULL.
    """
    # Build a column-name → ref_table lookup once per row. With most
    # schemas having ≤ a handful of FKs per table this is cheap.
    # Dict order doesn't matter here — we lookup by name, not iterate.
    fk_by_col: dict[str, str] = {}
    for fk in table.foreign_keys:
        for child_col, _parent_col in zip(fk.columns, fk.ref_columns, strict=True):
            fk_by_col[child_col] = fk.ref_table

    values: list[object] = []
    # Column ORDER matters: every consumed rng call advances the shared
    # RNG state, so a swap here changes downstream output. The iteration
    # order follows `table.columns`, which is the schema generator's
    # declared order — same source of truth as the DDL emitter.
    for col in table.columns:
        if col.name == _PK_COLUMN_NAME:
            # PK is supplied by the caller (sequential 1..n). PK never
            # rolls for NULL and never goes through a strategy — bypass.
            values.append(pk)
            continue
        if col.nullable and rng.random() < null_fraction:
            # Null roll consumes one rng tick — this consumption happens
            # for EVERY nullable column regardless of FK status, which
            # keeps the FK and non-FK paths byte-stable when a column's
            # type changes between FK and non-FK in future refactors.
            values.append(None)
            continue
        if col.name in fk_by_col:
            ref_table = fk_by_col[col.name]
            if ref_table == table.name:
                # Self-FK: PG enforces FKs per-row (default NOT DEFERRABLE),
                # so row pk=N can reference any row in [1..N] including itself.
                # Without this branch we'd try to read id_store[table.name]
                # which is empty (populated only after this loop in data.py).
                # For nullable columns, the null roll has already fired above
                # — any column that reaches this branch survived that roll
                # and should get a real value, not unconditional NULL.
                values.append(rng.randint(1, pk))
                continue
            parent_ids = id_store[ref_table]
            # If the parent table has zero rows, this FK column must be NULL.
            # A NOT NULL FK into an empty parent table is a generator error —
            # the topological walk should have materialized parents first.
            # This branch ONLY fires when the user passes --rows=0 (or the
            # parent's depth-adjusted count rounds to 0, which can't happen
            # with the current `base * fanout**depth` formula and base > 0).
            if not parent_ids:
                if not col.nullable:
                    raise ValueError(
                        f"NOT NULL FK {table.name}.{col.name} references "
                        f"empty parent {ref_table}"
                    )
                values.append(None)
                continue
            values.append(rng.choice(parent_ids))
            continue
        strat = strategy_for(col)
        values.append(strat(rng, col))
    return tuple(values)
