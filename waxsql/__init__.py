"""waxsql — random PostgreSQL query generator (wax-fruit SQL).

Public API surface at 1.0: the schema model, the type system, the
catalog, the validation modes (SYNTAX / PARSE / PLAN), and the
query generator.

The two headline functions:

  * `generate_schema(seed, complexity)` — random Schema, deterministic
    in seed.
  * `generate_query(seed, schema, complexity)` — random Query against
    that schema, deterministic in seed (independent RNG stream from
    the schema generator's, so "same schema, different queries"
    works by varying just the query seed).

Render queries with `print_query(q)`. Validate at three tiers:
`waxsql.validate.syntax.check_syntax(sql)` (no DB),
`waxsql.validate.parse.check_parse(sql, conn)` (live PG, PREPARE),
or `waxsql.validate.plan.check_plan(sql, conn)` (live PG, EXPLAIN).

Role in the system: this module IS the public surface. The `__all__`
at the bottom is the supported API contract; anything reachable only
via dotted paths into submodules is implementation detail subject to
churn. The data generator (`generate_data`) is re-exported here so
callers don't need to know it lives under `waxsql.data`.

Deliberately kept out of `__all__` to hold a small, stable 1.0 contract:

  * The full AST node vocabulary (`ColumnRef`, `FuncCall`, `Subquery`,
    `WindowSpec`, `CteDef`, ...) lives in `waxsql.ast` — the canonical
    AST module. Import from there to construct or walk a query tree.
    Only `Query` (the `generate_query` return type) and the `Select` /
    `SetOp` shapes its `.select` can hold are surfaced here.
  * The query-generator internals (`GenContext`, `Scope`, `Binding`,
    `ComplexityConfig`, `query_config_for_complexity`, the `FEATURE_*`
    flags) live in `waxsql.context` / `waxsql.scope` / `waxsql.config`.
    They drive generation but are not a stable extension API — and the
    functions that consume them (`gen_select` / `gen_setop`) aren't
    public either, so exporting the context types alone would be
    incoherent.
"""
from __future__ import annotations

import random
from importlib.metadata import PackageNotFoundError, version
from typing import Optional

# Only the workflow-facing AST types are surfaced at top level; the rest
# of the node vocabulary stays in `waxsql.ast` (see module docstring).
from .ast import Query, Select, SetOp
from .catalog import Catalog, FuncKind, FuncSig, OpSig, default_catalog
# query_config_for_complexity and FEATURE_SET_OP are imported for use in
# generate_query's body below; they are intentionally NOT re-exported
# (generator internals — see module docstring).
from .config import FEATURE_SET_OP, query_config_for_complexity
from .context import GenContext
from .data import generate_data
from .gen import gen_select, gen_setop
from .printer import print_expr, print_query
from .schema import (
    Column, ForeignKey, Index, Schema, SchemaConfig, Table,
    generate_schema, generate_schema_with_config,
    quote_ident, schema_config_for_complexity,
)
from .scope import Scope
from .types import (
    BOOL, DATE, FLOAT8, INT4, INT8, INTERVAL, JSONB, NUMERIC,
    PgType, SCALAR_TYPES, TEXT, TIMESTAMPTZ, TypeCategory,
    UUID, VARCHAR, array_of, implicitly_castable,
)
from .validate import ValidationMode

# Version is the single source of truth in pyproject.toml. Read it at
# import time via importlib.metadata so we never have two constants to
# keep in sync. The PackageNotFoundError fallback handles the rare
# "running directly from the source tree without an install" case.
try:
    __version__ = version("waxsql")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"


def generate_query(
    seed: int,
    *,
    schema: Schema,
    complexity: int = 5,
    catalog: Optional[Catalog] = None,
) -> Query:
    """Generate a random Query against `schema`, deterministic in `seed`.

    Same `(seed, schema, complexity, catalog)` always produces an
    identical Query (and identical printed SQL). The query-generator
    RNG is independent of the schema-generator RNG — pass the same
    seed to both for "fully reproducible session", or vary the query
    seed to fuzz queries against a fixed schema.

    `catalog` defaults to `default_catalog()` if not supplied.
    """
    cat = catalog if catalog is not None else default_catalog()
    cfg = query_config_for_complexity(complexity)
    # Construct the GenContext exactly once here, at the entry point.
    # Every downstream generator gets a derivative of this ctx via
    # GenContext.descend / descend_subquery / dataclasses.replace —
    # and crucially, every derivative shares this rng instance. That
    # shared mutable rng is what makes (seed, schema, complexity) ↦
    # query a deterministic function: reordering or duplicating the
    # rng-consuming calls anywhere in the generator tree perturbs the
    # output. Resist any urge to fork a fresh Random() inside a
    # downstream generator "for cleanliness"; it would silently break
    # the determinism test and the fuzzer's bug-reproduction guarantee.
    ctx = GenContext(
        rng=random.Random(seed),
        scope=Scope(),
        schema=schema,
        catalog=cat,
        config=cfg,
        depth_remaining=cfg.max_expr_depth,
        # Opt in to the subquery budget; the GenContext default is 0
        # (no subqueries) so callers must explicitly enable them.
        subquery_depth_remaining=cfg.max_subquery_depth,
    )
    # Top-level dispatch: SetOp (UNION/INTERSECT/EXCEPT) when the
    # feature is unlocked AND the dice roll fires; plain Select
    # otherwise. Set-ops live at the OUTERMOST level — the generator
    # doesn't produce set ops inside subqueries or CTE bodies.
    body: Select | SetOp
    if (FEATURE_SET_OP in cfg.feature_flags
            and ctx.rng.random() < cfg.p_set_op_query):
        body = gen_setop(ctx)
    else:
        body = gen_select(ctx)
    return Query(select=body)


__all__ = [
    # Types
    "PgType", "TypeCategory",
    "INT4", "INT8", "NUMERIC", "FLOAT8", "TEXT", "VARCHAR", "BOOL",
    "DATE", "TIMESTAMPTZ", "INTERVAL", "UUID", "JSONB",
    "SCALAR_TYPES", "array_of", "implicitly_castable",
    # Schema
    "Schema", "Table", "Column", "ForeignKey", "Index",
    "SchemaConfig", "schema_config_for_complexity",
    "generate_schema", "generate_schema_with_config",
    "quote_ident",
    # Catalog
    "Catalog", "FuncSig", "OpSig", "FuncKind", "default_catalog",
    # Validation
    "ValidationMode",
    # Query result type + the two shapes its `.select` can hold. The rest
    # of the AST node vocabulary lives in `waxsql.ast` (see docstring).
    "Query", "Select", "SetOp",
    # Generation + rendering
    "generate_query", "print_query", "print_expr",
    # Data generator
    "generate_data",
]
