"""PostgreSQL type system model.

Mirrors the abstractions PostgreSQL itself uses (pg_type.typcategory) so
that as we add more types and casts later, the structure already lines up
with how the planner reasons about coercion.

This is a deliberately small slice of PostgreSQL's actual type system —
~12 scalar types plus arrays. Expand `_IMPLICIT_CASTS` and `SCALAR_TYPES`
as the generator needs more variety.

This module is the load-bearing foundation under the type-driven
expression generator: every "what produces type T?" lookup in the
catalog, every column visibility filter in scope.py, and every
function/operator argument check runs through `implicitly_castable`.
Mistakes here propagate as "valid-looking SQL that fails parse-analysis"
across the whole generator. Cross-reference with pg_cast when changing
anything below.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TypeCategory(str, Enum):
    """Type categories from pg_type.typcategory.

    Used by the planner to decide implicit coercion in contexts like
    UNION resolution and operator/function dispatch. We track it on every
    type so the catalog can answer "is this thing usable here" without
    reinventing PostgreSQL's logic.
    """
    ARRAY = "A"
    BOOLEAN = "B"
    COMPOSITE = "C"
    DATETIME = "D"
    ENUM = "E"
    GEOMETRIC = "G"
    NETWORK = "I"
    NUMERIC = "N"
    PSEUDO = "P"
    RANGE = "R"
    STRING = "S"
    TIMESPAN = "T"
    USER = "U"
    BITSTRING = "V"
    UNKNOWN = "X"


@dataclass(frozen=True)
class PgType:
    """A PostgreSQL type.

    `name` matches pg_type.typname (so `int8`, not `bigint`); we rely on
    PostgreSQL accepting both spellings in DDL. `element` is set only for
    array types, in which case `name` is conventionally the underscore-
    prefixed form (`_int4` for `int4[]`), again matching pg_type.

    `typmod` is the type modifier tuple, e.g. (10, 2) for `numeric(10,2)`
    or (50,) for `varchar(50)`. Empty tuple means no modifier.

    Frozen so PgType instances are hashable and usable as dict keys, which
    matters for type weight tables and catalog indexes.
    """
    name: str
    category: TypeCategory
    element: Optional["PgType"] = None
    typmod: tuple[int, ...] = ()

    # The `is_*` predicates are convenience wrappers. They exist so that
    # callers don't have to import TypeCategory just to ask the obvious
    # question, and so future re-categorization (e.g. splitting NUMERIC
    # into INTEGRAL/REAL) only has to touch this file.
    def is_array(self) -> bool:
        return self.element is not None

    def is_numeric(self) -> bool:
        return self.category == TypeCategory.NUMERIC

    def is_string(self) -> bool:
        return self.category == TypeCategory.STRING

    def sql(self) -> str:
        """Render as a SQL type expression suitable for DDL or CAST."""
        if self.element is not None:
            return f"{self.element.sql()}[]"
        if self.typmod:
            return f"{self.name}({','.join(str(t) for t in self.typmod)})"
        return self.name


# Day-one scalar set. Picked to give the generator interesting variety
# (numeric, string, temporal, structured) without drowning the catalog
# in every cast rule PostgreSQL ships with.
INT4 = PgType("int4", TypeCategory.NUMERIC)
INT8 = PgType("int8", TypeCategory.NUMERIC)
NUMERIC = PgType("numeric", TypeCategory.NUMERIC)
FLOAT8 = PgType("float8", TypeCategory.NUMERIC)
TEXT = PgType("text", TypeCategory.STRING)
VARCHAR = PgType("varchar", TypeCategory.STRING)
BOOL = PgType("bool", TypeCategory.BOOLEAN)
DATE = PgType("date", TypeCategory.DATETIME)
TIMESTAMPTZ = PgType("timestamptz", TypeCategory.DATETIME)
INTERVAL = PgType("interval", TypeCategory.TIMESPAN)
UUID = PgType("uuid", TypeCategory.USER)
JSONB = PgType("jsonb", TypeCategory.USER)


def array_of(t: PgType) -> PgType:
    """Construct an array type over `t`. Mirrors pg_type's `_typname` convention."""
    return PgType(name=f"_{t.name}", category=TypeCategory.ARRAY, element=t)


SCALAR_TYPES: tuple[PgType, ...] = (
    INT4, INT8, NUMERIC, FLOAT8,
    TEXT, VARCHAR, BOOL,
    DATE, TIMESTAMPTZ, INTERVAL,
    UUID, JSONB,
)


# Implicit cast graph. Each key maps to the set of target type names that
# the source coerces to *implicitly* (no CAST needed). This is a small
# subset of pg_cast — enough to keep the generator honest about what it
# can pass where, without trying to be a complete oracle for PG semantics.
#
# Convention: every type implicitly casts to itself, so the target set
# always contains the source's own name.
#
# Direction matters: this is a source→target relation, not symmetric.
# `int4 → int8` is listed; `int8 → int4` is not. The numeric chain
# (int4 → int8 → numeric → float8) reflects PG's standard promotion
# ladder. A type missing from this dict still casts to itself via the
# `src == tgt` short-circuit in implicitly_castable, so adding a new
# scalar without an entry here degrades to "no implicit casts" rather
# than to broken behavior.
#
# Transitivity is precomputed, not derived. `int4` lists `float8`
# directly even though PG reaches float8 only via the int8 → numeric
# → float8 chain. The lookup must be O(1) because it runs once per
# candidate type per expression-generator decision; we'd rather
# maintain the closure by hand than walk the graph at every check.
# Anyone editing this dict must keep the closure consistent.
_IMPLICIT_CASTS: dict[str, frozenset[str]] = {
    "int4":        frozenset({"int4", "int8", "numeric", "float8"}),
    "int8":        frozenset({"int8", "numeric", "float8"}),
    "numeric":     frozenset({"numeric", "float8"}),
    "float8":      frozenset({"float8"}),
    "text":        frozenset({"text"}),
    "varchar":     frozenset({"varchar", "text"}),
    "bool":        frozenset({"bool"}),
    "date":        frozenset({"date", "timestamptz"}),
    "timestamptz": frozenset({"timestamptz"}),
    "interval":    frozenset({"interval"}),
    "uuid":        frozenset({"uuid"}),
    "jsonb":       frozenset({"jsonb"}),
}


def implicitly_castable(src: PgType, tgt: PgType) -> bool:
    """True iff a value of type `src` can be used where `tgt` is expected
    without an explicit CAST.

    Arrays are handled with a deliberately strict rule: arrays cast iff
    their element types match exactly. PostgreSQL's actual array casting
    rules are more permissive in some cases, but the strict rule keeps
    the generator from emitting things that *might* parse but rarely
    type-check.
    """
    # Identity short-circuit before the dict lookup. Two reasons:
    # (1) it's the common case and avoids a hash/lookup per check;
    # (2) it ensures a type with no entry in _IMPLICIT_CASTS still
    # casts to itself — see the dict comment about "degrades to
    # no implicit casts" when an entry is missing.
    if src == tgt:
        return True
    # Mixed scalar/array combinations are always rejected. PG allows
    # some such coercions via container cast machinery, but generating
    # them requires special-cased SQL output (e.g. ARRAY[expr]); the
    # generator doesn't emit those today, so refusing here keeps the
    # generator's notion of cast-availability conservative.
    if src.is_array() or tgt.is_array():
        if src.is_array() and tgt.is_array():
            return src.element == tgt.element
        return False
    return tgt.name in _IMPLICIT_CASTS.get(src.name, frozenset())
