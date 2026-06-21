"""Public API for the data generator.

`generate_data(schema, *, seed, rows, fanout, null_fraction)` walks the
schema's FK DAG topologically and produces a string containing one
`COPY ... FROM STDIN; ...; \\.` block per table, in dependency order.

Determinism contract: same (schema, seed, rows, fanout, null_fraction)
produces byte-identical output across Python versions. The schema and
data generators each construct their own `random.Random(seed)`, so
they share no RNG state even when given the same seed.

Role in the system: this is the thin orchestration layer. The heavy
lifting (topological walk, per-row dispatch, COPY text formatting,
strategy registries) lives under `waxsql.gen.data.*`; this module
just stitches them together and owns the public function signature.
Keeping the public surface here means callers can `from waxsql import
generate_data` without ever touching the internal package layout.
"""
from __future__ import annotations

import random

from waxsql.gen.data.emit import emit_copy_block
from waxsql.gen.data.rows import (
    generate_row,
    rows_for_table,
    topological_order,
)
from waxsql.schema import Schema


def generate_data(
    schema: Schema,
    *,
    seed: int,
    rows: int = 100,
    fanout: int = 5,
    null_fraction: float = 0.05,
) -> str:
    """Emit COPY blocks for every table in `schema`, in FK-topological order.

    `rows` is the base row count; tables deeper in the FK DAG get
    `rows * fanout ** depth`. `null_fraction` is the per-nullable-column
    probability of emitting NULL. Same arguments + same schema → byte-
    identical output.

    `id_store` is populated as each table is finished — critically, the
    current table's own IDs are added AFTER all its rows are materialized.
    This lets the self-FK branch in `generate_row` use `rng.randint(1, pk)`
    for forward-safe references without consulting `id_store` for the
    current table.

    Raises ValueError if the schema contains an FK cycle. Cycle handling
    (deferred constraints + UPDATE-patches) is a known follow-up; today
    the CLI catches this and surfaces a clean usage error.
    """
    # One RNG seeded once and threaded through every row. The data
    # generator does NOT share RNG state with the schema generator —
    # each constructs its own `random.Random(seed)` — so passing the
    # same seed to both produces independent deterministic streams.
    rng = random.Random(seed)
    # Per-table list of PK values produced so far. Read by child tables
    # to resolve their FK columns; written ONCE per table, after that
    # table's rows are fully materialized (see ordering note below).
    id_store: dict[str, list[int]] = {}
    blocks: list[str] = []
    for table in topological_order(schema):
        n = rows_for_table(table, schema, base=rows, fanout=fanout)
        table_rows: list[tuple[object, ...]] = []
        ids: list[int] = []
        for pk in range(1, n + 1):
            # PKs are sequential 1..n. The schema generator emits
            # `id BIGINT NOT NULL` (no sequence), so the data generator
            # owns the PK numbering — no gaps, no collisions.
            row = generate_row(
                table=table,
                pk=pk,
                rng=rng,
                id_store=id_store,
                null_fraction=null_fraction,
            )
            ids.append(pk)
            table_rows.append(row)
        # Populate id_store only after all rows for this table are done.
        # Children (tables with an FK to this table) will be visited later
        # in topological order and can safely read from id_store then.
        # This ordering is load-bearing: a child must not be able to see
        # a parent's IDs mid-materialization (which could only happen
        # under a buggy reordering), and self-FK resolution within the
        # inner loop intentionally uses `rng.randint(1, pk)` instead of
        # consulting id_store, so the current table is absent from the
        # store throughout its own row loop.
        id_store[table.name] = ids
        columns = tuple(c.name for c in table.columns)
        blocks.append(emit_copy_block(table.name, columns, table_rows))
    # Blocks are joined with a single "\n" so callers can split/process
    # them; each block already ends with "\.\n" so adjacent blocks are
    # separated by a blank line in the rendered output.
    return "\n".join(blocks)
