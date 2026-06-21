"""Schema model and random schema generator.

The Schema is an immutable description of tables, columns, primary keys,
foreign keys, and indexes. It serves two purposes:

  1. Input to the query generator (which tables/columns exist, what
     types they have, what FK relationships connect them).

  2. Source of DDL that can be loaded into PostgreSQL for PARSE/PLAN
     validation modes, or just for human inspection of test scenarios.

Schema generation is deterministic in the seed and uses its own RNG
stream — the public `generate()` function in __init__.py splits the
master seed so that "same schema, different queries" is possible by
re-seeding only the query generator.

A few design choices worth flagging:

  * Every table has an `id` BIGINT PK. Composite keys are interesting
    but add complexity to FK generation and JOIN matching. Add later.

  * FKs are emitted as separate ALTER TABLE statements after all CREATE
    TABLEs. This sidesteps any topological-sort headache when the FK
    graph has cycles (allowed at high complexity) and makes the DDL
    work even when tables reference each other.

  * Index names and FK names embed the table name, which keeps them
    unique across the schema (PostgreSQL constraint and index names
    are schema-scoped, not table-scoped).

  * Iteration over column lists, candidate referent indices, etc. is
    always done over sorted/list-typed inputs. Set iteration order is
    not guaranteed to be stable across Python builds, so all RNG-
    sensitive iteration goes through `sorted(...)`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Optional

from .types import (
    PgType,
    INT4, INT8, NUMERIC, FLOAT8,
    TEXT, VARCHAR, BOOL,
    DATE, TIMESTAMPTZ, INTERVAL,
    UUID, JSONB,
)


# ---------------------------------------------------------------------------
# Identifier quoting
# ---------------------------------------------------------------------------

# A small but useful set of PG reserved words. We don't try to be exhaustive
# because our name generator pulls from a curated word list, but better safe
# than sorry: any candidate name in this set gets double-quoted.
RESERVED_WORDS: frozenset[str] = frozenset({
    "all", "analyse", "analyze", "and", "any", "array", "as", "asc",
    "asymmetric", "both", "case", "cast", "check", "collate", "column",
    "constraint", "create", "current_date", "current_role", "current_time",
    "current_timestamp", "current_user", "default", "deferrable", "desc",
    "distinct", "do", "else", "end", "except", "false", "fetch", "for",
    "foreign", "from", "grant", "group", "having", "in", "initially",
    "intersect", "into", "lateral", "leading", "limit", "localtime",
    "localtimestamp", "not", "null", "offset", "on", "only", "or",
    "order", "placing", "primary", "references", "returning", "select",
    "session_user", "some", "symmetric", "table", "then", "to", "trailing",
    "true", "union", "unique", "user", "using", "variadic", "when", "where",
    "window", "with",
})


def quote_ident(name: str) -> str:
    """Quote an SQL identifier if it would otherwise be ambiguous.

    Bare identifiers are fine if they're lowercase, valid Python
    identifier characters, and not a reserved word. Otherwise wrap in
    double quotes and escape any embedded quotes.
    """
    if (name.isidentifier()
            and name == name.lower()
            and name not in RESERVED_WORDS):
        return name
    return '"' + name.replace('"', '""') + '"'


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Column:
    name: str
    type: PgType
    nullable: bool = True
    default: Optional[str] = None  # raw SQL expression; rarely used here

    def to_sql(self) -> str:
        bits = [quote_ident(self.name), self.type.sql()]
        if not self.nullable:
            bits.append("NOT NULL")
        if self.default is not None:
            bits.append(f"DEFAULT {self.default}")
        return " ".join(bits)


@dataclass(frozen=True)
class ForeignKey:
    name: str
    columns: tuple[str, ...]
    ref_table: str
    ref_columns: tuple[str, ...]
    on_delete: str = "NO ACTION"
    on_update: str = "NO ACTION"


@dataclass(frozen=True)
class Index:
    """An index. Either `columns` (plain b-tree on listed columns) or
    `expressions` (functional/expression index). `where` makes it partial.
    """
    name: str
    columns: tuple[str, ...] = ()
    expressions: tuple[str, ...] = ()
    unique: bool = False
    where: Optional[str] = None


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...] = ()
    foreign_keys: tuple[ForeignKey, ...] = ()
    indexes: tuple[Index, ...] = ()

    def column(self, name: str) -> Column:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)


@dataclass(frozen=True)
class Schema:
    tables: tuple[Table, ...]

    def table(self, name: str) -> Table:
        for t in self.tables:
            if t.name == name:
                return t
        raise KeyError(name)

    def emit_ddl(self) -> str:
        """Emit complete CREATE TABLE / ALTER TABLE / CREATE INDEX script.

        Order: all tables first, then all FKs (so cyclic FK graphs work),
        then all indexes.
        """
        parts: list[str] = []
        for t in self.tables:
            parts.append(_emit_create_table(t))
        for t in self.tables:
            for fk in t.foreign_keys:
                parts.append(_emit_alter_add_fk(t.name, fk))
        for t in self.tables:
            for idx in t.indexes:
                parts.append(_emit_create_index(t.name, idx))
        return "\n\n".join(parts) + "\n"


def _emit_create_table(t: Table) -> str:
    cols_sql = [c.to_sql() for c in t.columns]
    if t.primary_key:
        pk_cols = ", ".join(quote_ident(c) for c in t.primary_key)
        cols_sql.append(f"PRIMARY KEY ({pk_cols})")
    body = ",\n  ".join(cols_sql)
    return f"CREATE TABLE {quote_ident(t.name)} (\n  {body}\n);"


def _emit_alter_add_fk(table_name: str, fk: ForeignKey) -> str:
    cols = ", ".join(quote_ident(c) for c in fk.columns)
    ref_cols = ", ".join(quote_ident(c) for c in fk.ref_columns)
    return (
        f"ALTER TABLE {quote_ident(table_name)} "
        f"ADD CONSTRAINT {quote_ident(fk.name)} "
        f"FOREIGN KEY ({cols}) "
        f"REFERENCES {quote_ident(fk.ref_table)} ({ref_cols}) "
        f"ON DELETE {fk.on_delete} ON UPDATE {fk.on_update};"
    )


def _emit_create_index(table_name: str, idx: Index) -> str:
    unique = "UNIQUE " if idx.unique else ""
    if idx.expressions:
        cols = ", ".join(f"({e})" for e in idx.expressions)
    else:
        cols = ", ".join(quote_ident(c) for c in idx.columns)
    where = f" WHERE {idx.where}" if idx.where else ""
    return (
        f"CREATE {unique}INDEX {quote_ident(idx.name)} "
        f"ON {quote_ident(table_name)} ({cols}){where};"
    )


# ---------------------------------------------------------------------------
# Random schema generator
# ---------------------------------------------------------------------------

# Curated word lists. Output is dramatically more readable when tables are
# `customers` and `orders` rather than `t_47` and `t_48`. Costs nothing.
_NOUNS: tuple[str, ...] = (
    "customer", "order", "product", "invoice", "shipment", "address",
    "account", "transaction", "category", "tag", "comment", "review",
    "ticket", "event", "session", "device", "service", "region", "country",
    "city", "warehouse", "vendor", "supplier", "employee", "department",
    "project", "task", "milestone", "file", "folder", "role",
    "permission", "subscription", "plan", "feature", "discount", "coupon",
    "payment", "refund", "channel", "post", "thread", "notification",
    "alert", "metric", "report", "audit", "snapshot", "backup",
    "branch", "release", "build", "artifact", "config", "policy", "rule",
    "filter", "rating", "vote", "follower", "friend", "contact", "lead",
    "campaign", "keyword", "asset", "license", "contract", "clause",
    "schedule", "appointment", "reservation", "booking", "host",
    "room", "floor", "building", "site", "zone", "lane", "route", "stop",
    "trip", "vehicle", "driver", "package", "manifest", "container",
    "shelf", "bin", "lot", "batch", "sku", "variant", "color", "size",
    "material", "fabric", "ingredient", "recipe", "cart",
)

_ADJECTIVES: tuple[str, ...] = (
    "active", "archived", "pending", "primary", "secondary", "external",
    "internal", "draft", "published", "private", "public", "shared",
    "default", "custom", "raw", "processed", "verified", "trial",
    "premium", "legacy", "current", "historical", "scheduled",
)


@dataclass
class SchemaConfig:
    """Tunables for random schema generation.

    Each parameter is exposed so users can drive the generator manually
    if the complexity-dial preset doesn't fit their needs.
    """
    table_count: int
    min_columns: int
    max_columns: int
    fk_density: float           # P(non-id int column becomes an FK), 0..1
    allow_cyclic_fks: bool      # if False, FKs only point at earlier tables
    allow_self_fks: bool        # if True, a table may FK to itself
    index_density: float        # mean extra indexes per table (Gaussian)
    type_weights: dict[PgType, float] = field(default_factory=dict)


def schema_config_for_complexity(complexity: int) -> SchemaConfig:
    """Map a 0..10 complexity dial onto a SchemaConfig.

    The cyclic / self-FK thresholds are deliberately high because those
    structures matter mostly to recursive CTE and graph-style query
    generation, which only kicks in at the top of the dial.
    """
    c = max(0, min(10, complexity))
    return SchemaConfig(
        table_count=2 + c,                   # 2..12
        min_columns=3,
        max_columns=4 + c,                   # up to 14
        fk_density=0.30 + 0.04 * c,          # 0.30..0.70
        allow_cyclic_fks=c >= 8,
        allow_self_fks=c >= 5,
        index_density=0.5 + 0.1 * c,
        type_weights=_default_type_weights(),
    )


def _default_type_weights() -> dict[PgType, float]:
    """Bias toward the types real schemas use most. Arrays absent for
    now — they make join/where generation more annoying than it's worth
    at this stage."""
    return {
        INT4: 4.0, INT8: 3.0, NUMERIC: 1.5, FLOAT8: 0.5,
        TEXT: 4.0, VARCHAR: 1.0, BOOL: 2.0,
        DATE: 1.0, TIMESTAMPTZ: 2.5, INTERVAL: 0.3,
        UUID: 1.0, JSONB: 0.8,
    }


def generate_schema(seed: int, complexity: int = 5) -> Schema:
    """Generate a random schema deterministic in `seed`.

    Same (seed, complexity) → identical Schema object.
    """
    rng = random.Random(seed)
    cfg = schema_config_for_complexity(complexity)
    return _generate_schema_impl(rng, cfg)


def generate_schema_with_config(seed: int, cfg: SchemaConfig) -> Schema:
    """Generate a random schema with an explicit SchemaConfig."""
    rng = random.Random(seed)
    return _generate_schema_impl(rng, cfg)


def _generate_schema_impl(rng: random.Random, cfg: SchemaConfig) -> Schema:
    # Three-pass build: tables-with-columns first, then FKs (which need
    # all table names to exist as referent candidates), then indexes
    # (which need the FKs to exist so they can be auto-indexed). Frozen
    # dataclasses mean each pass returns a fresh list of replaced Tables.
    # The freshly-replaced list discipline matters: the FK pass reads
    # `drafts[ref_idx]` for each new FK, so passes that mutate the
    # input list during iteration would be off-by-one races. Returning
    # a new list keeps each pass logically atomic.
    table_names = _unique_names(rng, cfg.table_count, plural=True)

    drafts: list[Table] = []
    for name in table_names:
        ncols = rng.randint(cfg.min_columns, cfg.max_columns)
        cols = _generate_columns(rng, ncols, cfg.type_weights)
        # Every table gets an `id BIGINT NOT NULL` PK, prepended.
        cols = (Column("id", INT8, nullable=False),) + cols
        drafts.append(Table(name=name, columns=cols, primary_key=("id",)))

    drafts = _add_foreign_keys(rng, drafts, cfg)
    drafts = _add_indexes(rng, drafts, cfg)
    return Schema(tables=tuple(drafts))


def _unique_names(rng: random.Random, n: int, *, plural: bool) -> list[str]:
    """Generate `n` distinct identifier-safe names."""
    # `seen` is used only for membership tests (`in`, `add`); we never
    # iterate it. That's the only reason a set is OK here under the
    # determinism rules — set iteration order is unstable across Python
    # builds. The RNG-affecting iteration goes through `nouns`/`adjs`,
    # which are sorted lists.
    seen: set[str] = set()
    out: list[str] = []
    nouns = sorted(_NOUNS)
    adjs = sorted(_ADJECTIVES)
    # Bound the number of attempts to avoid pathological cases.
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        noun = rng.choice(nouns)
        if plural:
            noun = _pluralize(noun)
        # 20% of the time, prefix with an adjective for variety; also
        # use the prefix as a fallback if the bare noun collides.
        if rng.random() < 0.2 or noun in seen:
            adj = rng.choice(adjs)
            candidate = f"{adj}_{noun}"
        else:
            candidate = noun
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    if len(out) < n:
        # Fall back to numeric suffixes if we somehow exhausted the pool.
        i = 0
        while len(out) < n:
            candidate = f"t_{i}"
            if candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
            i += 1
    return out


def _pluralize(noun: str) -> str:
    """Trivial English pluralizer. Good enough for table names."""
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and (len(noun) < 2 or noun[-2] not in "aeiou"):
        return noun[:-1] + "ies"
    return noun + "s"


def _generate_columns(
    rng: random.Random,
    n: int,
    type_weights: dict[PgType, float],
) -> tuple[Column, ...]:
    """Generate `n` columns with names not colliding with `id`."""
    # `seen` is membership-only — never iterated. Set is safe here for
    # the same reason as in _unique_names. The RNG-touching iteration
    # is `nouns`/`adjs`/`types`, all sorted lists.
    seen: set[str] = {"id"}
    cols: list[Column] = []
    nouns = sorted(_NOUNS)
    adjs = sorted(_ADJECTIVES)
    # Sort by .name (not the PgType itself) because PgType is frozen
    # but doesn't define a comparison order — sorting by name gives a
    # stable, build-independent order without requiring an __lt__.
    types = sorted(type_weights.keys(), key=lambda t: t.name)
    weights = [type_weights[t] for t in types]
    attempts = 0
    while len(cols) < n and attempts < n * 50:
        attempts += 1
        if rng.random() < 0.5:
            name = rng.choice(nouns)
        else:
            name = f"{rng.choice(adjs)}_{rng.choice(nouns)}"
        if name in seen:
            continue
        seen.add(name)
        t = rng.choices(types, weights=weights)[0]
        # 60% nullable matches typical real-world schema shape closely
        # enough — and gives the query generator regular exercise of
        # both NULL-aware and NULL-naive code paths. This is not the
        # same as the data generator's null_fraction (which controls
        # whether a nullable column actually contains NULL).
        nullable = rng.random() < 0.6
        cols.append(Column(name=name, type=t, nullable=nullable))
    return tuple(cols)


def _add_foreign_keys(
    rng: random.Random,
    drafts: list[Table],
    cfg: SchemaConfig,
) -> list[Table]:
    """Add FKs to each table per the config's density and cycle rules.

    FK columns are selected from existing int4/int8 non-id columns; the
    referent is always some other table's `id`. This keeps types
    compatible without us having to re-type columns mid-generation.

    Tradeoff: not retro-renaming the columns means the FK column name
    (`metric`, `region`, ...) often has nothing to do with what it
    references. That's a fidelity loss versus real schemas but keeps
    the generation pass single-shot — column generation doesn't have
    to know which columns will later become FKs. The alternative
    (assign FKs first, generate columns to match) makes the dependency
    direction backwards from how a schema is normally built up.
    """
    n = len(drafts)
    new_drafts: list[Table] = []

    for i, t in enumerate(drafts):
        fks: list[ForeignKey] = []

        # Build candidate referent index list.
        #
        # The non-cyclic path is "earlier tables only" because forward
        # references would create cycles by definition: if table i can
        # point at table j > i, and table j can point at table i, the
        # graph has a 2-cycle. Restricting to j < i gives a DAG by
        # construction. The data generator currently relies on this
        # DAG-ness to topo-walk FKs (see ARCHITECTURE.md "out of scope": FK-
        # cyclic schemas in the data generator).
        if cfg.allow_cyclic_fks:
            candidates = list(range(n))
        else:
            candidates = list(range(i))  # earlier tables only
        if cfg.allow_self_fks and i not in candidates:
            candidates.append(i)
        # Self-FK control runs AFTER the cyclic branch added `i` — the
        # cyclic case unconditionally added every index, including `i`,
        # so without this filter we'd allow self-FKs at any complexity
        # the moment cyclic FKs unlock. The two flags must be honored
        # independently.
        if not cfg.allow_self_fks:
            candidates = [j for j in candidates if j != i]

        for col in t.columns:
            if col.name == "id":
                continue
            if col.type.name not in ("int4", "int8"):
                continue
            if rng.random() > cfg.fk_density:
                continue
            if not candidates:
                continue
            # `sorted(candidates)` looks redundant — `candidates` is
            # built from range(...) which is already sorted — but the
            # `if not allow_self_fks` filter above can break that
            # invariant for the cyclic path (where we appended `i`
            # before filtering). The sort is cheap and load-bearing
            # for determinism: rng.choice consumes the iterable's
            # actual order, not a canonical one.
            ref_idx = rng.choice(sorted(candidates))
            ref_table = drafts[ref_idx]
            fks.append(ForeignKey(
                name=f"fk_{t.name}_{col.name}",
                columns=(col.name,),
                ref_table=ref_table.name,
                ref_columns=("id",),
                on_delete=rng.choice((
                    "NO ACTION", "CASCADE", "SET NULL", "RESTRICT"
                )),
            ))
        new_drafts.append(replace(t, foreign_keys=tuple(fks)))
    return new_drafts


def _add_indexes(
    rng: random.Random,
    drafts: list[Table],
    cfg: SchemaConfig,
) -> list[Table]:
    """Add indexes: always on FK columns, plus a few extras per density.

    The "always index FK columns" rule mirrors what most real schemas
    do (and what PG's planner expects for efficient joins). Without
    these indexes, planner-tier validation would produce uniformly
    seq-scan-heavy plans that don't exercise the index-using code
    paths the generator is trying to fuzz.
    """
    new_drafts: list[Table] = []
    for t in drafts:
        idxs: list[Index] = []
        for fk in t.foreign_keys:
            idxs.append(Index(
                name=f"ix_{t.name}_{'_'.join(fk.columns)}",
                columns=fk.columns,
            ))
        n_extra = max(0, int(rng.gauss(cfg.index_density, 0.5)))
        non_id_cols = sorted(
            (c for c in t.columns if c.name != "id"),
            key=lambda c: c.name,
        )
        for k in range(n_extra):
            if not non_id_cols:
                break
            # Three-way taxonomy of generated indexes (cumulative roll):
            #   < 0.60 → single-column b-tree
            #   < 0.85 → two-column composite (falls through to partial
            #            branch if there are fewer than 2 columns)
            #   else   → partial b-tree predicated on a bool column
            #            (silently produces no index if no bool columns)
            # The partial branch can no-op, so n_extra is an upper bound
            # on the number of extra indexes, not an exact count.
            roll = rng.random()
            if roll < 0.6:
                col = rng.choice(non_id_cols)
                idxs.append(Index(
                    name=f"ix_{t.name}_{col.name}_x{k}",
                    columns=(col.name,),
                ))
            elif roll < 0.85 and len(non_id_cols) >= 2:
                cols = rng.sample(non_id_cols, 2)
                idxs.append(Index(
                    name=f"ix_{t.name}_{'_'.join(c.name for c in cols)}_x{k}",
                    columns=tuple(c.name for c in cols),
                ))
            else:
                bool_cols = [c for c in non_id_cols if c.type == BOOL]
                if bool_cols:
                    pred_col = rng.choice(bool_cols)
                    target_col = rng.choice(non_id_cols)
                    idxs.append(Index(
                        name=f"ix_{t.name}_{target_col.name}_partial_x{k}",
                        columns=(target_col.name,),
                        where=f"{quote_ident(pred_col.name)} = TRUE",
                    ))
        new_drafts.append(replace(t, indexes=tuple(idxs)))
    return new_drafts
