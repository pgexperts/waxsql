# Future directions

waxsql 1.0 covers the SELECT-statement domain end-to-end — from
random schema through random query through three-tier validation
(SYNTAX / PARSE / PLAN). What follows is a non-prescriptive sketch
of where the project could go next, grouped by ambition. Each
direction stands alone; none is on a critical path.

## Within the current architecture (incremental)

These are the natural follow-ons for someone who wants more of the
same — same generator, same validation tiers, just tighter or
broader:

- **Push PLAN-tier toward 100%.** The ~4–5% residual at c=10 is
  constant-foldable runtime errors PG catches eagerly when all
  inputs are constant — `x / (y - y)` folds to division by zero;
  `substr(s, x, neg_const)` folds to a length error. Catching
  these statically requires a small constant-fold analysis pass
  on the divisor / length / key sub-expression. Diminishing
  returns above ~99%, but the work is well-scoped.

- **The 1-in-200 PARSE residual.** A TIMESTAMPTZ-arithmetic edge
  case where a derived-table column's actual inner-SELECT type
  diverges from the registered binding type. Probably a small fix
  in the derived-table type-tracking code; the rate is too low
  to surface reliably without a focused investigation seed.

- **Feature-coverage statistical tests.** No test currently
  asserts that every shipped feature fires at some minimum rate.
  A sweep that runs N seeds at c=10 and checks ratios — at least
  M queries with WITH RECURSIVE, M with FILTER, M with frame
  clauses, M with GROUPING SETS, etc. — would catch silent
  feature regressions. Defensive infrastructure, ~half day.

- **Differential testing across PG versions.** Run the same SQL
  through PG 14 / 15 / 16 / 17 / 18-beta and surface where they
  disagree on PARSE/PLAN. This is what waxsql is best-suited to:
  generating large diverse query corpora cheaply. Mostly
  infrastructure work — Docker per PG version, side-by-side
  EXPLAIN comparisons.

- **EXECUTE-tier validation (Track D).** EXPLAIN ANALYZE on the
  empty schema. Adds runtime checking, but with no rows most
  per-row errors don't fire. Worth doing once the empty-schema
  limitation gets addressed (see "minimal data generation"
  below).

## Generator depth (still within SELECT)

These add more variety to what the SELECT generator can produce,
without changing the surrounding architecture:

- **Expression-level CASE / COALESCE / NULLIF**. Already in the
  catalog as scalar funcs but the WHEN/ELSE branching of CASE
  isn't modeled. Useful for exercising aggregate-in-CASE rules
  and type-resolution corners.

- **Column-list-on-derived-table** syntax: `(SELECT ...) AS dt(a, b, c)`.
  PG accepts this for renaming derived columns; we always use
  synthetic c1/c2/cN.

- **Lateral subqueries in SELECT list**, not just FROM. Less
  common but exercises a separate planning path.

- **Hypothetical-set aggregates** alongside ordered-set:
  `rank(value) WITHIN GROUP (ORDER BY ...)` and the dense_rank /
  percent_rank / cume_dist variants. Same `within_group`
  attachment, different semantics.

- **Schema column-type biasing.** The schema generator picks types
  uniformly from a pool. Biasing toward more numeric / fewer
  text columns (or the reverse, configurable) would diversify
  what queries look like across schemas.

- **Minimal data generation** for EXPLAIN ANALYZE. Insert a
  handful of rows per table after CREATE so PG's planner has
  statistics to work with. Opens a path to actual EXECUTE-tier
  validation without the empty-schema dodge.

## Extending the SQL surface (architectural)

These are projects of similar weight to the original m1–9
sequence. Each opens a new domain:

- **DML generation: INSERT / UPDATE / DELETE / MERGE.** The
  largest natural extension. Type-driven correctness applies the
  same way: column lists, RETURNING clauses, ON CONFLICT,
  multi-row VALUES, INSERT...SELECT. MERGE (PG 15+) adds
  conditional WHEN clauses with their own scope rules. Probably
  several weeks to do well.

- **Schema-evolution generation.** ALTER TABLE shapes (ADD COLUMN,
  DROP COLUMN, ALTER COLUMN TYPE with USING clauses), CREATE
  INDEX (including expression indexes, partial indexes,
  covering), partition setups (RANGE / LIST / HASH partitioning,
  attach/detach), inheritance trees. Each shape has its own
  validity rules; the schema generator already gives a base to
  evolve from.

- **Custom types: composite / enum / domain.** Composite types
  exercise a different scope-resolution path (`row.field`); enum
  types are simple but interact with comparison ordering; domain
  types layer constraints on existing types. Each is small but
  adds non-trivial surface to the catalog/scope modeling.

- **Functions, procedures, and triggers.** PG's function language
  variants (SQL, PL/pgSQL, ...) are their own languages.
  Generating functions with valid bodies and triggers wiring
  them into table events would substantially expand the scope.
  Plan-tier validation gets harder here — function bodies aren't
  validated by EXPLAIN.

- **Rules and views.** PG's rule system (CREATE RULE, ON
  SELECT/UPDATE/DELETE/INSERT) and views (including materialized
  views with REFRESH semantics). Smaller than triggers but with
  their own rewriting interactions.

- **LISTEN / NOTIFY / advisory locks.** Specialized but small;
  exercises PG's session-state APIs.

## Tooling and integration

The library could grow a CLI and integration surface for
downstream consumers:

- **CLI tool.** `waxsql gen --seed N --complexity X` that
  emits SQL to stdout. `waxsql validate --tier plan SQL_FILE`
  for validating arbitrary SQL through the same machinery.
  `waxsql sweep --seeds 0-100 --complexity 10 --report failures`
  for quick PARSE/PLAN-rate measurement.

- **Output formats.** Beyond raw SQL: JSON of the AST (for tools
  that want to mutate), a "labeled" form (each clause tagged
  with the feature that produced it), a structured trace (which
  generator productions fired in what order — useful for
  debugging RNG behavior).

- **pgbench integration.** Use generated queries as load. The
  determinism and complexity dial map well onto pgbench's
  parametric workload model.

- **pgTAP test generation.** Generate test queries paired with
  expected-pass assertions for tools like pgTAP that test a
  database from inside it.

- **Web playground.** A hosted "fiddle" — paste a seed and
  complexity, see the schema and query, try the validation
  tiers. Useful for sharing reproducers and demoing the project.

- **VS Code / language-server integration.** Generate sample SQL
  for a workspace-aware schema; use the validator as a linter
  for hand-written SQL against that schema.

## Cross-database

waxsql's architecture (catalog + printer + type system) is
PG-specific but the generator/validator structure isn't. Other
databases could plug in:

- **PG-compatible dialects: CockroachDB, YugabyteDB.** Mostly
  catalog tweaks (different function pool, different operator
  set) plus a few printer changes for non-standard syntax.

- **SQL-standard subset.** Target ANSI rather than PG-specific
  — useful for tools that need broad SQL coverage.

- **MySQL, SQLite, SQL Server.** Each needs its own catalog,
  type system, and printer. The generator core (gen_expr,
  gen_select, etc.) could potentially stay shared if abstracted.

## Generator architecture (research-y)

These are speculative directions that would change how the
generator itself works:

- **Property-based testing integration.** Hypothesis-style
  strategies that produce waxsql Schema/Query objects, with
  shrinking. Would let downstream tools express invariants like
  "my SQL prettifier round-trips for any waxsql-generated query."

- **Constraint-aware generation.** Queries that satisfy specific
  shape predicates: "produce a query with at least one
  correlated subquery and at least one window function" —
  useful for targeted testing of optimizer paths.

- **Catalog-from-introspection.** Derive the catalog (function
  signatures, operator overloads) from a real DB's pg_proc /
  pg_operator instead of hand-curating. Risky because PG's
  catalog has polymorphic and special-syntax entries waxsql's
  flat model can't represent — but would expand coverage
  substantially if done carefully.

- **Cost-aware generation.** Produce queries within a planner-
  cost budget (estimated by EXPLAIN). Useful for generating
  "small" or "huge" queries on demand.

- **Mutation-based fuzzing.** Take a known-good waxsql query,
  apply small mutations (swap an op, change a literal, drop a
  clause), check whether validation tiers still pass. Comp-
  lements the from-scratch generation approach.

## Specialized applications

If the project gets used in a specific domain, these directions
might emerge:

- **Equivalence-preserving query rewriter.** Generate two
  semantically-equivalent queries from the same seed (different
  shape, same logical result). Direct application: testing PG's
  optimizer for plan-cost differences on equivalent inputs.

- **SQL-injection-pattern generation.** Generate queries that
  exercise specific SQL-injection-defense paths in
  framework-level query builders. Training data for security
  tooling.

- **Query-pattern mining seeding.** Extract realistic shape
  distributions from production query logs and bias the
  generator's RNG to reproduce those distributions. Test data
  that resembles real load.

- **ML training data.** Large corpora of (schema, query,
  validation-result) triples for training models that read or
  generate SQL.

## Closing notes

The 1.0 architecture is robust enough that most of the above
directions slot in without re-litigating foundations. The
patterns that pay back across all of them: type-driven
generation, deterministic RNG, savepoint-isolated validation,
and the construction-site `coerce / replace` helpers that catch
PG's idiosyncrasies before they reach the printer.

The project is not actively planning toward any of these. They
are simply the directions that make sense given what 1.0 made
possible. Anyone picking up the codebase is welcome to take any
of them in whatever order suits.
