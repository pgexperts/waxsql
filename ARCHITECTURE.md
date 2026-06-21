# waxsql

A random PostgreSQL query generator: SQL the equivalent of wax fruit. The goal is **syntactically correct** SQL queries against a random schema, with a complexity dial that scales from `SELECT * FROM t` up through deeply-nested CTE/subquery/window-function trees.

"Correct" here means **type-driven correct**: every expression respects the type system, scope rules, and aggregate-context restrictions. Output is meant to clear PostgreSQL's parse-analysis stage, not just the grammar. Closer in spirit to SQLsmith than to a yacc-driven fuzzer.

## Architectural pillars (load-bearing — discuss before breaking)

### 1. Type-driven expression generation

Every expression is generated with a target type. The catalog (`waxsql/catalog.py`) answers "what produces type T?" from the function pool, the operator pool, and (eventually) column references via the scope. We never emit `int + text` because no operator satisfies that request.

The implicit cast graph (`_IMPLICIT_CASTS` in `types.py`) governs what counts as "produces T" — a function returning `int4` satisfies a request for `int8` because `int4` implicitly casts to `int8`. Keep this graph honest with PostgreSQL's actual cast rules. When in doubt, check `pg_cast`.

### 2. Determinism

Same `(seed, complexity)` must produce byte-identical output across runs and Python versions. This is what makes the generator usable for fuzzing (reproduce a bug from a seed) and for testing (golden output).

The discipline that keeps this true:

- **Never iterate over a `set` in code that affects RNG choices.** Set iteration order isn't stable across Python builds. Use `sorted(...)` first. Dict iteration order *is* stable (Python 3.7+) so dicts are fine.
- **Split RNG streams** when you have orthogonal random decisions. Schema and query generators get independently-seeded RNGs derived from the master seed; this lets "same schema, different queries" work.
- **No time, environment, or other non-deterministic input.** If you reach for the global `random` module instead of an injected `rng`, you're doing something wrong.

The test `test_same_seed_same_schema` enforces this for the schema generator. Add an analogous test for any new generator.

### 3. Round-trip validation

Every generated artifact must parse via `pglast` (libpg_query bindings — the actual PostgreSQL parser as a static library). The pattern:

```python
schema = generate_schema(seed=seed, complexity=complexity)
ddl = schema.emit_ddl()
result = check_syntax(ddl)
assert result.ok, f"... {result.error}\n{ddl}"
```

This catches a huge class of generator bugs immediately. **Any new code path that emits SQL must have a parametrized test that round-trips many seeds through pglast.** Don't trust the printer; let libpg_query confirm.

`pglast` is pinned to v7 (PostgreSQL 17). v8 (PG18) is in development on the `lelit/pglast` `v8` branch but not on PyPI as of May 2026. When v8 ships, bump the pin in `pyproject.toml` and re-run the suite.

### 4. Validation tiers

Three modes, strictly stronger:

- **SYNTAX** (default): pglast. No DB. ~µs/check. Catches every grammar error.
- **PARSE**: PREPARE against a live DB. Catches name/type errors. ~ms/check. Not yet implemented.
- **PLAN**: EXPLAIN against a live DB. Catches planner-time errors. When input headers carry `with-data=true`, the data section is loaded and `ANALYZE` is run before EXPLAIN, so plans reflect real statistics rather than empty-table defaults.

When implementing PARSE/PLAN, use savepoints around each PREPARE so a single failure doesn't abort the surrounding transaction.

### 5. Data generation

`waxsql/data.py` produces deterministic COPY blocks for any generated schema. Same `(schema, seed, rows, fanout, null_fraction)` → byte-identical output. The generator walks the FK DAG topologically, samples FK column values from per-table ID stores it builds along the way, and dispatches per-column through a tuple-of-patterns registry that falls through to a type-strategy registry. Plausibility today is mostly type-driven (wordlist text, bounded numerics, a 5-year date window); the column-name override registry exists as a hook for semantic plausibility as schema generation grows richer.

`validate --tier plan` consumes the data section when the input header has `with-data=true`, loading COPY blocks under savepoints and running `ANALYZE` before EXPLAIN — so plans reflect realistic statistics.

## Module layout

```
waxsql/
├── types.py            PgType, type categories, implicit cast graph    [DONE]
├── catalog.py          FuncSig, OpSig, default catalog                 [DONE]
├── schema.py           Schema model + random generator + DDL emitter   [DONE]
├── data.py             Public API: generate_data                       [DONE]
├── ast.py              AST dataclasses for queries                     [DONE]
├── printer.py          AST → SQL with precedence/parens                [DONE]
├── pretty.py           SQL reformat + color for gen --pprint           [DONE]
├── scope.py            Binding stack, visible-columns lookup           [DONE]
├── context.py          GenContext: rng, scope, depth budget, dial      [DONE]
├── config.py           Complexity dial → weights/budgets               [DONE]
├── gen/
│   ├── data/
│   │   ├── strategies.py   Per-type value strategies + wordlist        [DONE]
│   │   ├── columns.py      Column-name override registry               [DONE]
│   │   ├── rows.py         Topo walk + row materialization             [DONE]
│   │   └── emit.py         COPY block formatting                       [DONE]
│   ├── expr.py         Typed expression generator                      [DONE]
│   ├── select.py       SELECT/FROM/WHERE/GROUP BY/HAVING/ORDER/LIMIT   [DONE]
│   ├── subquery.py     Scalar / EXISTS / IN subqueries + derived       [DONE]
│   ├── window.py       Window function specs (PARTITION/ORDER/FRAME)   [DONE]
│   ├── cte.py          WITH (recursive and not)                        [DONE]
│   └── setop.py        UNION/INTERSECT/EXCEPT                          [DONE]
└── validate/
    ├── __init__.py     ValidationMode enum                             [DONE]
    ├── syntax.py       pglast wrapper                                  [DONE]
    ├── parse.py        PREPARE-based                                   [DONE]
    └── plan.py         EXPLAIN-based                                   [DONE]
```

## Conventions

- **Frozen dataclasses** for models (Schema, Table, Column, PgType, AST nodes). Hashable, can't accidentally mutate, pass cheaply through the call stack.
- **`from __future__ import annotations`** at the top of every module.
- **Docstrings explain the *why*, not the *what*.** The code already says what it does. Use the docstring for the design reason — why this abstraction exists, why this pattern over the alternative, what easy mistake the comment is heading off.
- **Type hints everywhere**, including return types. Keep it mypy-clean by hand even though mypy isn't wired up.
- **Tests use parametrization aggressively.** `@pytest.mark.parametrize("seed", range(20))` is the standard pattern; bump the range if a bug slips through. Full suite runs in <1s, so generosity costs nothing.

## Things deliberately out of scope (don't add without discussion)

- **Composite primary keys** in the schema generator. All tables get `id BIGINT NOT NULL`. Composites complicate FK matching and JOIN generation.
- **Special-syntax functions** in the catalog: `POSITION...IN`, `EXTRACT...FROM`, `OVERLAY...PLACING`, `TRIM...FROM`, `SUBSTRING...FOR`. Add only when the printer can render them correctly; emitting them as plain function calls produces invalid SQL.
- **Scraping `pg_proc`.** The catalog is hand-curated for good reasons (polymorphism, variadics, set-returning, special syntax). Resist the urge to "just import everything."
- **Nullability tracking through outer joins.** Useful eventually, lots of bookkeeping, easy to get wrong. Assume everything might be NULL.
- **CHECK / UNIQUE (beyond PK) / DOMAIN constraint awareness in the data generator.** Schema doesn't emit those today; revisit when it does.
- **Semantic per-column plausibility.** The data generator has a hook for column-name → strategy overrides, but the registry ships nearly-empty. Today plausibility is type-driven; growing the registry is a separate (small) ongoing task.
- **FK-cyclic schemas in the data generator.** At complexity ≥ 8 the schema generator can produce cross-table FK cycles; the data generator currently raises a clean usage error rather than handling them. Proper cycle handling (deferred constraints + UPDATE patches) is a known follow-up.

## Workflow

```bash
pip install -e '.[dev]'
pytest                            # full suite, <1s
pytest tests/test_schema.py -v    # schema-specific
python -c "from waxsql import generate_schema; print(generate_schema(42, 6).emit_ddl())"
```

When adding new generators, mirror the schema test pattern: parametrize over seeds and complexity, round-trip through pglast, assert ok.

## Reference: the foundational design conversation

The original design discussion that produced the current architecture covered the type-driven vs grammar-driven choice, the scope object's lateral/aggregate/CTE subtleties, the validation tier design, and the schema generator's FK-graph shape. If you find yourself uncertain about a foundational decision, the reasoning is in the docstrings of the relevant module — start there before re-deriving.
