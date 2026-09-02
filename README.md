# waxsql

[![CI](https://github.com/pgexperts/waxsql/actions/workflows/ci.yml/badge.svg)](https://github.com/pgexperts/waxsql/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/waxsql.svg)](https://pypi.org/project/waxsql/)
[![Python](https://img.shields.io/pypi/pyversions/waxsql.svg)](https://pypi.org/project/waxsql/)

A deterministic, type-driven random PostgreSQL query generator. SQL the
equivalent of wax fruit — looks real, doesn't compute anything useful, won't
spoil. Useful for fuzzing query tools, exercising parsers and planners,
generating reproducible workloads, and producing realistic-but-meaningless
SQL on tap.

```python
from waxsql import generate_query, generate_schema, print_query

schema = generate_schema(seed=42, complexity=8)
query  = generate_query(seed=42, schema=schema, complexity=8)

print(schema.emit_ddl())   # CREATE TABLE / ALTER TABLE / CREATE INDEX
print(print_query(query))  # SELECT ... — type-correct against the schema
```

Same `(seed, complexity)` always produces the same output, byte for byte.
The complexity dial scales from `SELECT col FROM t` up through deeply-nested
CTE / subquery / window-function / grouping-set trees. "Correct" means
**type-driven correct**: every expression respects PostgreSQL's type system,
scope rules, aggregate-context restrictions, and overload-resolution
semantics — generated SQL clears parse-analysis cleanly.

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Theory of operation](#theory-of-operation)
- [The complexity dial](#the-complexity-dial)
- [Validation tiers](#validation-tiers)
- [Public API reference](#public-api-reference)
- [Surprising corners](#surprising-corners)
- [Development](#development)
- [License](#license)

---

## Install

waxsql is a pure-Python package on PyPI. The package itself has zero runtime
dependencies; everything PostgreSQL-related is an optional extra.

### With `pip`

```bash
# Just the generator. Use this if you only need to produce SQL strings.
pip install waxsql

# Add the SYNTAX-tier validator (uses pglast / libpg_query — no DB needed).
pip install 'waxsql[syntax]'

# Add the live-DB validators (psycopg). PARSE and PLAN are equivalent in
# the dependency set; the names exist so callers can express intent.
pip install 'waxsql[parse]'
pip install 'waxsql[plan]'

# Everything (generator + all validators).
pip install 'waxsql[all]'
```

### With `uv`

```bash
# Standalone install into the active venv.
uv pip install waxsql

# Or as a project dependency:
uv add waxsql
uv add 'waxsql[all]'
```

### Optional-dependency matrix

| Extra      | Pulls in                                | Enables                                |
|------------|-----------------------------------------|----------------------------------------|
| (none)     | nothing                                 | `generate_schema`, `generate_query`, `print_query` |
| `[syntax]` | `pglast >=7,<8`                         | `check_syntax(sql)` — no DB needed     |
| `[parse]`  | `psycopg[binary] >=3.1`                 | `check_parse(sql, conn)`               |
| `[plan]`   | `psycopg[binary] >=3.1`                 | `check_plan(sql, conn)`                |
| `[cli]`    | `click >=8`                             | the `waxsql` console script (`gen`, `validate`) |
| `[pprint]` | `pglast >=7,<8` + `pygments >=2`        | `gen --pprint` — reformatted and colorized SQL output |
| `[all]`    | `pglast >=7,<8` + `psycopg[binary] >=3.1` + `click >=8` + `pygments >=2` | all validators + the CLI |
| `[dev]`    | `[all]` plus `pytest`, `ruff`, `mypy`   | full test/lint/type pipeline           |

Python 3.10 or newer is required. `pglast` v7 currently tracks PostgreSQL 17;
v8 (PG18 support) is in development upstream — when it lands, the `[syntax]`
pin will move.

### Pre-release versions

If you want a release candidate or dev build (`waxsql 1.1.0rc1`,
`waxsql 1.1.0.dev1`), `pip install` ignores those by default. Pass `--pre`:

```bash
pip install --pre waxsql
```

---

## Quick start

### Use it from the command line

Install the `[cli]` extra and use the bundled `waxsql` command — no Python needed:

```bash
pip install 'waxsql[cli]'

# One-shot demo: random seed, default complexity, schema + query both.
waxsql gen

# Reproducible run with a fixed seed.
waxsql gen --seed 42 --complexity 8

# Pipe straight into psql.
waxsql gen --seed 42 -c 8 | psql -d scratch

# Pipe through the validator (gen output's header tells validate
# which schema to install — auto-schema is on by default).
waxsql gen --seed 42 -c 8 | waxsql validate --tier plan
```

`waxsql gen --help` and `waxsql validate --help` document all the flags. A few notable `gen` options:

- `--pprint` — reformat the generated DDL and queries with indentation and,
  when writing to a terminal, syntax-color them. Display-only: the output
  is for reading, not for piping into `validate` (color codes and the
  reflowed layout aren't meant to round-trip). Requires the `[pprint]`
  extra: `pip install 'waxsql[pprint]'`.

The Python API examples below cover the same ground for callers who want to drive the library directly.

### 1. Generate a schema and one query against it

```python
from waxsql import generate_query, generate_schema, print_query

schema = generate_schema(seed=42, complexity=5)
query  = generate_query(seed=42, schema=schema, complexity=5)

print(print_query(query))
```

Both functions are deterministic in their seeds. Re-running the snippet on
any machine, any Python version (≥3.10), produces the same SQL.

### 2. Generate many queries against one fixed schema

The schema and query generators have **independent RNG streams**. Hold the
schema seed fixed and vary the query seed to produce a workload of
unrelated queries against a stable target — useful for soak-testing a
query-rewriting tool, a planner, or a connection-pooling proxy.

```python
from waxsql import generate_query, generate_schema, print_query

schema = generate_schema(seed=42, complexity=6)

for query_seed in range(100):
    q = generate_query(seed=query_seed, schema=schema, complexity=6)
    print(print_query(q))
    print(";")
```

### 3. Validate a generated query through pglast (no DB)

```python
from waxsql import generate_query, generate_schema, print_query
from waxsql.validate.syntax import check_syntax

schema = generate_schema(seed=1, complexity=10)
q      = generate_query(seed=1, schema=schema, complexity=10)
sql    = print_query(q)

result = check_syntax(sql)
assert result.ok, f"pglast rejected: {result.error}\n{sql}"
```

`check_syntax` is microsecond-fast and needs no PostgreSQL connection —
it shells out to the libpg_query C library bundled with `pglast`.

### 4. Validate against a live PostgreSQL via PREPARE

PARSE-tier validation catches name- and type-resolution errors that pure
syntax-checking can't see. The validator wraps each PREPARE in a savepoint
so a single failure doesn't abort the surrounding transaction.

```python
import psycopg
from waxsql import generate_query, generate_schema, print_query
from waxsql.validate.parse import check_parse, install_schema

schema = generate_schema(seed=1, complexity=10)
q      = generate_query(seed=1, schema=schema, complexity=10)
sql    = print_query(q)

# autocommit=False is required — the validators use savepoints, which only
# work inside a transaction. psycopg opens that transaction implicitly on
# the first statement; no explicit BEGIN needed (and issuing one would
# produce a spurious "there is already a transaction in progress" warning).
with psycopg.connect("dbname=waxsql_scratch", autocommit=False) as conn:
    try:
        install_schema(schema, conn)        # CREATE TABLE etc.
        result = check_parse(sql, conn)
        assert result.ok, f"PG rejected: {result.error}\n{sql}"
    finally:
        conn.rollback()                     # discard the throwaway schema
```

`check_plan(sql, conn)` is identical in shape but runs `EXPLAIN` instead
of `PREPARE`, exercising the planner as well as parse-analysis. See
[Validation tiers](#validation-tiers) for when to reach for which.

---

## Theory of operation

waxsql is closer in spirit to SQLsmith than to a yacc-driven fuzzer. Three
ideas hold the whole thing together:

### Type-driven generation

Every expression is generated **with a target type**. The catalog
(`waxsql.catalog`) answers "what produces type T?" from the function pool,
the operator pool, and (via the scope) column references. The generator
never emits `int + text` because no operator satisfies that request.

The implicit-cast graph (`_IMPLICIT_CASTS` in `waxsql.types`) governs what
counts as "produces T" — a function returning `int4` satisfies a request
for `int8` because `int4` implicitly casts to `int8`. The graph is hand-
curated against PostgreSQL's `pg_cast`.

This is why the output isn't just syntactically valid — it clears
**parse-analysis** at PostgreSQL's full type-resolver. SQL that the
PostgreSQL grammar accepts but the type system rejects (e.g.
`WHERE current_timestamp + 'hello'`) is structurally absent from the
generator's possible outputs.

### Determinism

Same `(seed, complexity)` produces byte-identical SQL across runs and
Python versions. This is what makes the generator usable for fuzzing
(reproduce a bug from a seed) and for golden-output testing.

The discipline that keeps it true:

- The generator never reads the global `random` module — every randomized
  decision goes through an injected `random.Random` instance.
- Set iteration is forbidden in any RNG-affecting code path; sets are
  always `sorted(...)` first because Python's set iteration order isn't
  stable across builds. (Dict order is stable since 3.7, so dicts are fine.)
- Schema and query each get an independently seeded RNG stream, so
  varying the query seed against a fixed schema seed works as expected.

### Round-trip validation

Every generated artifact can be round-tripped through PostgreSQL's actual
parser via `pglast` — the `check_syntax` function runs your SQL through
the real `libpg_query`. The test suite enforces this on every code path
that emits SQL, parametrized over many seeds. There are no print-time
shortcuts that would let invalid SQL escape.

---

## The complexity dial

`complexity` is a single integer 0..10 that controls both how rich the
schema is and how feature-rich the queries are. It works by **unlocking
features in stages**:

| `complexity` | Features unlocked at this notch                                      |
|--------------|----------------------------------------------------------------------|
| 0            | Trivial `SELECT col FROM t` only                                     |
| 1            | `WHERE`, `INNER JOIN`, multiple `FROM` items                         |
| 2            | `ORDER BY`, `LIMIT`                                                  |
| 3            | Aggregates and `GROUP BY`                                            |
| 4            | `LEFT JOIN`, scalar subqueries                                       |
| 5            | `HAVING`, `EXISTS`, `IN (SELECT ...)`, derived tables                |
| 6            | `LATERAL` (only meaningful with derived tables, hence the gate)      |
| 7            | Common Table Expressions (`WITH`)                                    |
| 8            | Window functions (`func() OVER (...)`)                               |
| 9            | Set operations (`UNION` / `INTERSECT` / `EXCEPT [ALL]`)              |
| 10           | `WITH RECURSIVE`, `ROLLUP` / `CUBE` / `GROUPING SETS`                |

Structural caps (max expression depth, max FROM items, max subquery nesting,
max CTEs per WITH list) also grow with the dial — see `waxsql.config.
query_config_for_complexity` for the formulas.

The schema generator's dial is similar in spirit:

| `complexity` | Schema effect                                                        |
|--------------|----------------------------------------------------------------------|
| 0..10        | `table_count = 2 + c` (so 2..12 tables)                              |
| 0..10        | `max_columns = 4 + c` (so up to 14 cols/table)                       |
| 0..10        | `fk_density = 0.30 + 0.04*c` (more FKs at higher complexity)         |
| ≥5           | Self-referencing FKs allowed (`tbl.parent_id REFERENCES tbl(id)`)    |
| ≥8           | Cyclic FK graphs allowed                                             |

If the canned dial doesn't fit your needs, drop down to the underlying
config objects — see [Public API reference](#public-api-reference) below.

---

## Validation tiers

Three modes, strictly stronger. Each tier catches a strict superset of
what the previous tier catches.

| Tier   | Mechanism                | Cost      | Coverage at c=10 |
|--------|--------------------------|-----------|------------------|
| SYNTAX | `pglast` (libpg_query)   | µs/check  | 100%             |
| PARSE  | `PREPARE` + savepoint    | ms/check  | 100%             |
| PLAN   | `EXPLAIN` + savepoint    | ms/check  | ~95–96%          |

Rates are observed at c=10 on deterministic hardware; CI gates at floors
that absorb per-seed variance (PARSE 100% at the tested complexities,
PLAN ≥90% at c=10).

The PLAN-tier residual is constant-foldable runtime errors PG catches
eagerly during planning (mostly division by zero through arithmetic that
folds to a literal zero). They'd be the same errors `EXECUTE` would raise
on real data.

### When to use which

- **SYNTAX** is the right default. It catches every grammar error, runs
  in microseconds, and needs no PostgreSQL — install with `[syntax]` and
  call `check_syntax(sql)`.
- **PARSE** is for verifying that name/type resolution succeeds against a
  real schema. If you're feeding waxsql output into a tool that does its
  own parse-analysis (e.g. a query rewriter), PARSE is the floor your
  fuzzer should clear.
- **PLAN** adds planner-time errors on top of PARSE. Run it when you want
  to be sure PostgreSQL would actually accept the SQL for execution, not
  just compilation.

### Live-DB validation pattern

Both PARSE and PLAN need a transaction-mode psycopg connection. The
canonical shape:

```python
import psycopg
from waxsql import generate_query, generate_schema, print_query
from waxsql.validate.parse import check_parse, install_schema
# from waxsql.validate.plan import check_plan   # same shape, runs EXPLAIN

schema = generate_schema(seed=0, complexity=10)

with psycopg.connect("dbname=scratch", autocommit=False) as conn:
    try:
        install_schema(schema, conn)

        # Sweep many queries against the same installed schema.
        # Each check_parse savepoints around the PREPARE so a single
        # failure doesn't abort the sweep.
        for query_seed in range(1000):
            q = generate_query(seed=query_seed, schema=schema, complexity=10)
            sql = print_query(q)
            r = check_parse(sql, conn)
            if not r.ok:
                print(f"REJECTED at seed={query_seed}: {r.error}\n{sql}\n")
    finally:
        conn.rollback()  # nothing persists
```

The default DSN used by the test suite is `dbname=waxsql_test`; override
via the `WAXSQL_PG_DSN` environment variable if you want to point the
test suite at a different cluster.

---

## Public API reference

Everything documented here is exported from the top-level `waxsql` package.
Internal helpers are not part of the public surface and may change without
notice; if you reach into a submodule that isn't listed here, you're on
your own at upgrade time.

### Generation

```python
generate_schema(seed: int, complexity: int = 5) -> Schema
```
Generate a random schema. Deterministic in `(seed, complexity)`.

```python
generate_query(seed: int, *, schema: Schema, complexity: int = 5,
               catalog: Optional[Catalog] = None) -> Query
```
Generate a random `Query` against `schema`. Deterministic in
`(seed, schema, complexity, catalog)`. Note the `*` — `schema` is
**keyword-only**.

```python
generate_schema_with_config(rng: random.Random, cfg: SchemaConfig) -> Schema
```
Lower-level entry point that accepts a pre-seeded RNG and a hand-built
`SchemaConfig`. Use this if `schema_config_for_complexity` doesn't fit
your needs.

```python
generate_data(schema: Schema, *, seed: int, rows: int = 100,
              fanout: int = 5, null_fraction: float = 0.05) -> str
```
Emit one deterministic `COPY ... FROM STDIN` block per table, in
FK-topological order — a fully self-contained data section to pair with
`schema.emit_ddl()`. Deterministic in `(schema, seed, rows, fanout,
null_fraction)`; tables deeper in the FK DAG get `rows * fanout ** depth`
rows. Raises `ValueError` on an FK-cyclic schema (deferred-constraint
cycle handling is a known follow-up).

### Rendering

```python
print_query(q: Query) -> str
```
Render a `Query` AST as a SQL string. Despite the name, this **returns**
the string; it does not write to stdout. (See [Surprising corners](#surprising-corners).)

```python
print_expr(e: Expr) -> str
```
Render a single expression. Useful when debugging the generator.

```python
schema.emit_ddl() -> str
```
Method on `Schema`. Returns the full `CREATE TABLE` / `ALTER TABLE` /
`CREATE INDEX` script for the schema. Tables are emitted first, then
all foreign keys (so cyclic FK graphs work), then indexes.

### Schema model

All frozen dataclasses; safely hashable, safely shared.

| Symbol         | Role                                                          |
|----------------|---------------------------------------------------------------|
| `Schema`       | Top-level container. `tables: tuple[Table, ...]`, `.table(name)`, `.emit_ddl()` |
| `Table`        | Has `name`, `columns`, `foreign_keys`, `indexes`              |
| `Column`       | Has `name`, `type` (a `PgType`), `not_null`                   |
| `ForeignKey`   | Source-column → target-table.id reference                     |
| `Index`        | Single- or multi-column index spec                            |
| `SchemaConfig` | The dial-derived knobs for `generate_schema_with_config`      |
| `quote_ident(name)` | Quote a SQL identifier (always; safe to over-quote)      |

### Type system

| Symbol            | Notes                                                    |
|-------------------|----------------------------------------------------------|
| `PgType`          | Frozen dataclass. The atomic unit of the type system.    |
| `TypeCategory`    | Enum: NUMERIC / STRING / BOOLEAN / TEMPORAL / etc.       |
| `INT4`, `INT8`, `NUMERIC`, `FLOAT8`, `TEXT`, `VARCHAR`, `BOOL`, `DATE`, `TIMESTAMPTZ`, `INTERVAL`, `UUID`, `JSONB` | Built-in singletons |
| `SCALAR_TYPES`    | Tuple of all built-in scalar `PgType` values             |
| `array_of(t)`     | Construct an array type (`int4[]`, `text[]`, etc.)       |
| `implicitly_castable(src, dst)` | Walks the implicit-cast graph            |

### Catalog

| Symbol            | Notes                                                    |
|-------------------|----------------------------------------------------------|
| `Catalog`         | Function and operator pools, plus type → producer index  |
| `FuncSig`         | Function signature: name, arg types, return type, kind   |
| `OpSig`           | Binary/unary operator signature                          |
| `FuncKind`        | Enum: SCALAR / AGGREGATE / WINDOW                        |
| `default_catalog()` | The standard hand-curated catalog used by the generator |

### Query AST

| Symbol            | Notes                                                    |
|-------------------|----------------------------------------------------------|
| `Query`           | Outermost node — wraps a `Select` or `SetOp`             |
| `Select`          | A single SELECT statement                                |
| `SetOp`           | UNION / INTERSECT / EXCEPT combining multiple selects    |
| `SelectTarget`    | One entry in the SELECT list (expression + optional alias) |
| `OrderByItem`     | One ORDER BY entry (expression + ASC/DESC + nulls placement) |
| `FromItem`        | Anything that can appear in FROM (table, derived, CTE ref) |
| `TableRef`        | A reference to a base table                              |
| `JoinExpr`        | An explicit JOIN node                                    |
| `Expr` (Protocol) | Marker protocol for expression nodes                     |
| `ColumnRef`, `Literal`, `FuncCall`, `BinaryOp`, `UnaryOp`, `Cast` | Concrete expression nodes |

### Validation

| Symbol                              | Module                  | Notes                          |
|-------------------------------------|-------------------------|--------------------------------|
| `ValidationMode`                    | `waxsql.validate`       | Enum (NONE / SYNTAX / PARSE / PLAN). Informational only — see surprises. |
| `check_syntax(sql)`                 | `waxsql.validate.syntax`| Returns `SyntaxResult`. No DB. |
| `check_parse(sql, conn)`            | `waxsql.validate.parse` | Returns `ParseResult`. PREPARE.|
| `check_plan(sql, conn)`             | `waxsql.validate.plan`  | Returns `PlanResult`. EXPLAIN. |
| `install_schema(schema, conn)`      | `waxsql.validate.parse` | DDL deploy for live-DB checks. |

All three result types share the shape `(ok: bool, error: Optional[str], ...)`.

### Configuration & generator internals

These are exported for callers who want to drive the generator directly
rather than through the canned `complexity` dial.

| Symbol                            | Notes                                          |
|-----------------------------------|------------------------------------------------|
| `ComplexityConfig`                | Dial-derived knobs for query generation        |
| `SchemaConfig`                    | Dial-derived knobs for schema generation       |
| `query_config_for_complexity(c)`  | Build a `ComplexityConfig` from 0..10          |
| `schema_config_for_complexity(c)` | Build a `SchemaConfig` from 0..10              |
| `GenContext`                      | The per-call state object (rng, scope, schema, catalog, config, depth budgets) |
| `Scope`, `Binding`                | Visible-columns lookup used during generation  |
| `FEATURE_*`                       | String constants for the feature-flag set      |

### Module map

```
waxsql/
├── __init__.py        ← public surface; everything in __all__ comes from here
├── types.py           PgType, type categories, implicit cast graph
├── catalog.py         FuncSig, OpSig, default catalog
├── schema.py          Schema model + random generator + DDL emitter
├── data.py            generate_data: deterministic COPY blocks for a schema
├── ast.py             AST dataclasses for queries
├── printer.py         AST → SQL with precedence/parens
├── pretty.py          SQL reformat + color for `gen --pprint`
├── scope.py           Binding stack, visible-columns lookup
├── context.py         GenContext: rng, scope, depth budget, dial
├── config.py          Complexity dial → weights/budgets
├── cli.py             `waxsql` console script (gen / data / validate)
├── gen/
│   ├── expr.py        Typed expression generator
│   ├── select.py      SELECT/FROM/WHERE/GROUP BY/HAVING/ORDER/LIMIT
│   ├── subquery.py    Scalar / EXISTS / IN subqueries + derived tables
│   ├── window.py      Window function specs (PARTITION/ORDER/FRAME)
│   ├── cte.py         WITH (recursive and not)
│   ├── setop.py       UNION/INTERSECT/EXCEPT
│   └── data/          Data-generator internals
│       ├── strategies.py   Per-type value strategies + wordlist
│       ├── columns.py      Column-name override registry
│       ├── rows.py         Topo walk + row materialization
│       └── emit.py         COPY block formatting
└── validate/
    ├── __init__.py    ValidationMode enum
    ├── syntax.py      pglast wrapper (no DB)
    ├── parse.py       PREPARE-based + install_schema
    └── plan.py        EXPLAIN-based
```

---

## Surprising corners

A handful of API choices that are likely to trip up a first-time reader.

### `schema=` is keyword-only

```python
generate_query(42, schema=schema)   # OK
generate_query(42, schema)          # TypeError: schema is keyword-only
```

The signature uses a `*` to force keyword passing — this is intentional
so that a future addition of a `complexity` positional argument can't
silently re-bind in existing call sites.

### `print_query` returns a string; it does not print

```python
sql = print_query(q)      # CORRECT — capture the return value
print_query(q)            # WRONG-ish — you get the string back but discard it
```

The name is a historical artifact (think "pretty-print") rather than an
imperative. The whole `print_*` family in `waxsql.printer` is functional:
they convert AST → string.

### Same seed for schema and query is by convention only

The schema and query generators have **independent** RNG streams. Passing
the same seed to both is a useful idiom for "fully reproducible session,"
but the two seeds are otherwise unrelated:

```python
schema = generate_schema(seed=42, complexity=8)

# Same schema, 100 different queries:
for s in range(100):
    q = generate_query(seed=s, schema=schema, complexity=8)
    ...

# Same query, but only across schemas built with the same SCHEMA seed:
schema2 = generate_schema(seed=42, complexity=8)   # identical to schema
q2      = generate_query(seed=7, schema=schema2, complexity=8)
# q2 == generate_query(seed=7, schema=schema, complexity=8)  # True
```

### `ValidationMode` is a label, not a dispatcher

```python
from waxsql import ValidationMode
# This enum exists, but there is NO `validate(sql, mode=ValidationMode.PARSE)`
# function. It's classification metadata for callers' own code.
```

To actually run a validation, call the tier-specific function directly:
`check_syntax(sql)`, `check_parse(sql, conn)`, or `check_plan(sql, conn)`.

### Live-DB validators need autocommit OFF

```python
conn = psycopg.connect(dsn, autocommit=True)   # WRONG — savepoints don't work
conn = psycopg.connect(dsn, autocommit=False)  # CORRECT
```

`check_parse` and `check_plan` use `SAVEPOINT` / `ROLLBACK TO SAVEPOINT`
to isolate per-query failures, and savepoints only exist inside a
transaction. Combine with the `BEGIN` / `install_schema` / `ROLLBACK`
shape shown above.

### `install_schema` lives in `waxsql.validate.parse`, not the top level

It's not re-exported from the top-level package because it's only
meaningful in the live-DB validation context. Import it explicitly:

```python
from waxsql.validate.parse import install_schema
```

### The schema generator never produces composite primary keys

Every table has `id BIGINT NOT NULL`. This is deliberate — composites
complicate FK matching and JOIN generation in ways that don't earn their
keep. If you need composite-PK coverage for a particular tool, hand-write
the schema and feed it into `generate_query` directly.

### Tables and columns in generated DDL are deterministically random names

You'll get identifiers like `tbl_a3f2`, not `customer` / `order`. The
generator is type-driven, not domain-driven; column names are opaque
on purpose to discourage callers from accidentally encoding semantic
assumptions about the output.

### The CLI's auto-schema header is a convention, not a stable file format

`waxsql gen` prefixes its output with `-- waxsql <version>  seed=N  complexity=X`,
and `waxsql validate --auto-schema` (default on) parses that header to
regenerate the matching schema. The `seed=N` and `complexity=X` keys are
guaranteed to remain parseable across CLI versions, but the surrounding
format may grow new fields. If you're piping `gen` output into something
other than `validate`, treat the header as a SQL comment to strip rather
than a format to depend on.

---

## Development

```bash
# Editable install with everything (test deps, lint, type-checker, validators).
pip install -e '.[dev]'

# Full test suite. ~70s with PARSE/PLAN tiers if a PG is reachable;
# the live-DB tiers skip cleanly if not.
pytest

# SYNTAX-tier only — fast, no PG needed.
pytest --ignore=tests/test_parse.py --ignore=tests/test_plan.py

# Lint and type-check (CI runs both on every push).
ruff check waxsql tests
mypy waxsql

# Quick smoke from the CLI:
python -c "from waxsql import generate_schema; print(generate_schema(42, 6).emit_ddl())"
```

The default DSN for live-DB tests is `dbname=waxsql_test`; override via
`WAXSQL_PG_DSN` (e.g. `WAXSQL_PG_DSN='host=localhost port=5433 dbname=fuzz' pytest`).

Bugs, questions, and feature ideas belong in
[GitHub Issues](https://github.com/pgexperts/waxsql/issues).

The release procedure is documented in [`RELEASING.md`](RELEASING.md).
The architecture and design rationale, including the choices behind the
type-driven approach and the determinism discipline, are in
[`ARCHITECTURE.md`](ARCHITECTURE.md). Possible future directions are in
[`FUTURE.md`](FUTURE.md).

---

## License

MIT.
