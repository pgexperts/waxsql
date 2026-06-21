"""Catalog of functions and operators available for query generation.

Hand-curated rather than scraped from pg_proc. The reasons:

  1. pg_proc has thousands of entries with polymorphic types (anyelement,
     anyarray) and special-case argument handling (variadic, set-returning,
     aggregates with FILTER, etc.) that don't translate cleanly to a
     simple signature model.

  2. The generator only needs *enough* surface area to produce varied
     queries. ~50 hand-picked entries give richer output than 5,000
     entries we don't know how to use.

  3. Curation lets us exclude things with awkward syntax — POSITION,
     EXTRACT, OVERLAY, TRIM-with-FROM-clause — that need bespoke printer
     handling. Add those when the printer can support them.

The catalog is split into FuncSig (callable name + arg list + return
type) and OpSig (left/right operand types + return). The generator asks
"what produces type T?" and the catalog answers from both pools.

Determinism contract: every lookup that returns a list is responsible
for sorting that list by a stable key. The expression generator picks
weighted-randomly from those lists, so a non-deterministic ordering
would silently desynchronize generated queries from their seed even
when the underlying tuple of FuncSig/OpSig hasn't changed (e.g. when
Python's hash randomization shuffles set iteration). See ARCHITECTURE.md
"Determinism" pillar.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .types import (
    PgType,
    INT4, INT8, NUMERIC, FLOAT8,
    TEXT, BOOL,
    DATE, TIMESTAMPTZ, INTERVAL,
    UUID, JSONB,
    array_of, implicitly_castable,
)


class FuncKind(str, Enum):
    """How a function may be invoked.

    SCALAR functions can appear in any expression context.
    AGGREGATE functions need a query level with grouping semantics.
    WINDOW functions need an OVER clause.
    SET_RETURNING functions are valid in FROM (and a few other places).

    The generator uses this to pick functions whose call site matches
    where the expression is being placed.
    """
    SCALAR = "scalar"
    AGGREGATE = "agg"
    WINDOW = "window"
    SET_RETURNING = "srf"


@dataclass(frozen=True)
class FuncSig:
    """A function signature.

    For overloaded functions (e.g. abs, sum, min, max) we register one
    FuncSig per concrete signature rather than trying to model
    polymorphism. Slightly verbose, completely unambiguous.

    Frozen so it's hashable and can serve as a key in any future
    candidate-cache; also makes the "one FuncSig per concrete
    signature" rule structurally enforced rather than just
    documentary.
    """
    name: str
    args: tuple[PgType, ...]
    returns: PgType
    kind: FuncKind = FuncKind.SCALAR
    variadic: bool = False


@dataclass(frozen=True)
class OpSig:
    """A binary or unary operator signature.

    `left=None` means prefix unary (e.g. -x); `right=None` would mean
    postfix unary (rare in modern SQL — factorial `!` was deprecated).
    For the starter catalog, all operators are binary.
    """
    symbol: str
    left: Optional[PgType]
    right: Optional[PgType]
    returns: PgType


@dataclass
class Catalog:
    """Indexed pool of functions and operators.

    The lookup methods sort their results by stable keys (name, then
    discriminating attributes) so that iteration order is reproducible
    across Python runs. This matters because the generator picks weighted
    random elements from these lists, and a different ordering would
    silently desynchronize generated queries from their seed.
    """
    functions: tuple[FuncSig, ...]
    operators: tuple[OpSig, ...]

    def funcs_returning(self, t: PgType) -> list[FuncSig]:
        """All functions whose return type implicitly casts to `t`.

        The sort key (name, kind, arg-type names) is chosen so that
        two FuncSigs that differ only in overload (`abs(int4)` vs
        `abs(int8)`) sort by arg-type — giving a fully stable order
        across runs. Don't replace the lambda with a simple `f.name`
        key: PG-style overloads would then cluster in arbitrary order
        and the generator would pick differently across Python builds.
        """
        return sorted(
            (f for f in self.functions if implicitly_castable(f.returns, t)),
            key=lambda f: (f.name, f.kind.value, tuple(a.name for a in f.args)),
        )

    # The kind-filtered convenience methods all delegate to
    # funcs_returning, so they inherit its sort order automatically.
    # Don't shortcut by filtering self.functions directly — that would
    # bypass the sort, breaking determinism.
    def aggs_returning(self, t: PgType) -> list[FuncSig]:
        return [f for f in self.funcs_returning(t) if f.kind == FuncKind.AGGREGATE]

    def scalar_funcs_returning(self, t: PgType) -> list[FuncSig]:
        return [f for f in self.funcs_returning(t) if f.kind == FuncKind.SCALAR]

    def window_funcs_returning(self, t: PgType) -> list[FuncSig]:
        return [f for f in self.funcs_returning(t) if f.kind == FuncKind.WINDOW]

    def srfs_returning(self, t: PgType) -> list[FuncSig]:
        return [f for f in self.funcs_returning(t) if f.kind == FuncKind.SET_RETURNING]

    def binary_ops_returning(self, t: PgType) -> list[OpSig]:
        # The `left is not None and right is not None` guard filters out
        # unary OpSigs. The starter catalog registers no unary operators,
        # but OpSig models them (`left=None` for prefix unary), so the
        # filter is defensive against future additions — without it, a
        # registered unary op would silently leak into the binary pool
        # and produce malformed `<unary> X Y` SQL.
        binary = [
            o for o in self.operators
            if o.left is not None and o.right is not None
            and implicitly_castable(o.returns, t)
        ]
        # Lambda accesses .name on what mypy still sees as Optional;
        # the comprehension above guarantees both are non-None, but
        # type-narrowing doesn't propagate into the lambda. Assert
        # locally for the type-checker — the runtime check is a no-op.
        return sorted(
            binary,
            key=lambda o: (
                o.symbol,
                o.left.name if o.left else "",
                o.right.name if o.right else "",
            ),
        )


def default_catalog() -> Catalog:
    """The starter catalog used by the generator out of the box.

    Roughly 50 functions covering numeric, string, temporal, JSON, UUID,
    aggregates, window, and set-returning categories; plus operators for
    arithmetic, comparison, logical, string concat, LIKE/ILIKE, temporal
    arithmetic, and JSON access.

    Anything requiring special-case syntax (POSITION...IN, EXTRACT...FROM,
    OVERLAY...PLACING, TRIM...FROM, SUBSTRING...FOR) is omitted; they need
    printer support before they're safe to register.
    """
    fs: list[FuncSig] = [
        # ---- Numeric scalar ----
        FuncSig("abs", (INT4,), INT4),
        FuncSig("abs", (INT8,), INT8),
        FuncSig("abs", (NUMERIC,), NUMERIC),
        FuncSig("abs", (FLOAT8,), FLOAT8),
        FuncSig("ceil", (NUMERIC,), NUMERIC),
        FuncSig("floor", (NUMERIC,), NUMERIC),
        FuncSig("round", (NUMERIC,), NUMERIC),
        FuncSig("round", (NUMERIC, INT4), NUMERIC),
        FuncSig("sqrt", (FLOAT8,), FLOAT8),
        FuncSig("power", (FLOAT8, FLOAT8), FLOAT8),
        FuncSig("mod", (INT4, INT4), INT4),
        FuncSig("mod", (INT8, INT8), INT8),
        FuncSig("greatest", (INT4, INT4), INT4),
        FuncSig("least", (INT4, INT4), INT4),
        FuncSig("greatest", (NUMERIC, NUMERIC), NUMERIC),
        FuncSig("least", (NUMERIC, NUMERIC), NUMERIC),

        # ---- String scalar ----
        FuncSig("length", (TEXT,), INT4),
        FuncSig("char_length", (TEXT,), INT4),
        FuncSig("upper", (TEXT,), TEXT),
        FuncSig("lower", (TEXT,), TEXT),
        FuncSig("initcap", (TEXT,), TEXT),
        FuncSig("trim", (TEXT,), TEXT),
        FuncSig("btrim", (TEXT,), TEXT),
        FuncSig("ltrim", (TEXT,), TEXT),
        FuncSig("rtrim", (TEXT,), TEXT),
        FuncSig("substr", (TEXT, INT4, INT4), TEXT),
        FuncSig("substr", (TEXT, INT4), TEXT),
        FuncSig("concat", (TEXT, TEXT), TEXT),
        FuncSig("replace", (TEXT, TEXT, TEXT), TEXT),
        FuncSig("md5", (TEXT,), TEXT),
        FuncSig("left", (TEXT, INT4), TEXT),
        FuncSig("right", (TEXT, INT4), TEXT),
        FuncSig("reverse", (TEXT,), TEXT),
        FuncSig("repeat", (TEXT, INT4), TEXT),

        # ---- Date/time scalar ----
        FuncSig("now", (), TIMESTAMPTZ),
        FuncSig("current_date", (), DATE),
        # date_trunc's first arg type is TEXT in our catalog, but PG
        # only accepts specific UNIT keywords ('day', 'hour', ...).
        # Generation special-cases the first arg in gen/expr.py's
        # func branch via `_DATE_TRUNC_UNITS` — the bare TEXT
        # signature here lets the catalog dispatch find it normally;
        # the special arg rewrite is what keeps the unit valid.
        # (Bad units fail at runtime, not PARSE/PLAN — the latter
        # because the cast wrapping masks PG's constant-folding —
        # so we'd fail EXECUTE without this special-case.)
        FuncSig("date_trunc", (TEXT, TIMESTAMPTZ), TIMESTAMPTZ),
        FuncSig("age", (TIMESTAMPTZ, TIMESTAMPTZ), INTERVAL),
        FuncSig("age", (TIMESTAMPTZ,), INTERVAL),

        # ---- UUID ----
        FuncSig("gen_random_uuid", (), UUID),

        # ---- JSON ----
        FuncSig("jsonb_typeof", (JSONB,), TEXT),
        # jsonb_array_length is back, with a special-case in
        # gen/expr.py's func branch: when emitting it, the JSONB arg
        # is replaced with a guaranteed-array literal. The catalog
        # signature stays generic JSONB so dispatch finds it; the
        # arg-rewrite ensures the value is always array-shaped.
        # (PG validates array-ness at run time AND at planning time
        # if the value is constant; my always-typed text literals
        # would mask the constant from the planner anyway, so this
        # is a runtime-correctness fix rather than a PLAN-tier fix.)
        FuncSig("jsonb_array_length", (JSONB,), INT4),
        FuncSig("jsonb_build_object", (TEXT, TEXT), JSONB),
        FuncSig("jsonb_build_array", (TEXT, TEXT), JSONB),
        FuncSig("to_jsonb", (TEXT,), JSONB),

        # ---- Null handling ----
        FuncSig("coalesce", (TEXT, TEXT), TEXT),
        FuncSig("coalesce", (INT4, INT4), INT4),
        FuncSig("nullif", (INT4, INT4), INT4),
        FuncSig("nullif", (TEXT, TEXT), TEXT),

        # ---- Aggregates ----
        FuncSig("count", (INT4,), INT8, kind=FuncKind.AGGREGATE),
        FuncSig("count", (TEXT,), INT8, kind=FuncKind.AGGREGATE),
        FuncSig("sum", (INT4,), INT8, kind=FuncKind.AGGREGATE),
        FuncSig("sum", (INT8,), NUMERIC, kind=FuncKind.AGGREGATE),
        FuncSig("sum", (NUMERIC,), NUMERIC, kind=FuncKind.AGGREGATE),
        FuncSig("avg", (INT4,), NUMERIC, kind=FuncKind.AGGREGATE),
        FuncSig("avg", (NUMERIC,), NUMERIC, kind=FuncKind.AGGREGATE),
        FuncSig("min", (INT4,), INT4, kind=FuncKind.AGGREGATE),
        FuncSig("min", (NUMERIC,), NUMERIC, kind=FuncKind.AGGREGATE),
        FuncSig("min", (TEXT,), TEXT, kind=FuncKind.AGGREGATE),
        FuncSig("min", (TIMESTAMPTZ,), TIMESTAMPTZ, kind=FuncKind.AGGREGATE),
        FuncSig("max", (INT4,), INT4, kind=FuncKind.AGGREGATE),
        FuncSig("max", (NUMERIC,), NUMERIC, kind=FuncKind.AGGREGATE),
        FuncSig("max", (TEXT,), TEXT, kind=FuncKind.AGGREGATE),
        FuncSig("max", (TIMESTAMPTZ,), TIMESTAMPTZ, kind=FuncKind.AGGREGATE),
        FuncSig("string_agg", (TEXT, TEXT), TEXT, kind=FuncKind.AGGREGATE),
        FuncSig("array_agg", (INT4,), array_of(INT4), kind=FuncKind.AGGREGATE),
        FuncSig("array_agg", (TEXT,), array_of(TEXT), kind=FuncKind.AGGREGATE),
        FuncSig("bool_and", (BOOL,), BOOL, kind=FuncKind.AGGREGATE),
        FuncSig("bool_or", (BOOL,), BOOL, kind=FuncKind.AGGREGATE),

        # ---- Ordered-set aggregates (require WITHIN GROUP) ----
        # These cannot be called without a WITHIN GROUP clause; the
        # generator special-cases their emission in gen/expr.py to
        # always attach one. Their declared return type holds when
        # the WITHIN GROUP ORDER BY column is FLOAT8 (the common
        # case our generator targets); other ORDER BY types would
        # change the return type per PG's polymorphic resolution,
        # but we don't exercise those.
        FuncSig("percentile_cont", (FLOAT8,), FLOAT8, kind=FuncKind.AGGREGATE),
        FuncSig("percentile_disc", (FLOAT8,), FLOAT8, kind=FuncKind.AGGREGATE),

        # ---- Window-only ----
        # (Aggregates can also be used as window functions via OVER, but
        # those are handled at the AST level rather than re-registered.)
        FuncSig("row_number", (), INT8, kind=FuncKind.WINDOW),
        FuncSig("rank", (), INT8, kind=FuncKind.WINDOW),
        FuncSig("dense_rank", (), INT8, kind=FuncKind.WINDOW),
        FuncSig("lag", (INT4,), INT4, kind=FuncKind.WINDOW),
        FuncSig("lead", (INT4,), INT4, kind=FuncKind.WINDOW),
        FuncSig("first_value", (INT4,), INT4, kind=FuncKind.WINDOW),
        FuncSig("last_value", (INT4,), INT4, kind=FuncKind.WINDOW),

        # ---- Set-returning ----
        FuncSig("generate_series", (INT4, INT4), INT4, kind=FuncKind.SET_RETURNING),
        FuncSig("generate_series", (INT8, INT8), INT8, kind=FuncKind.SET_RETURNING),
        FuncSig("generate_series", (TIMESTAMPTZ, TIMESTAMPTZ, INTERVAL),
                TIMESTAMPTZ, kind=FuncKind.SET_RETURNING),
        FuncSig("unnest", (array_of(INT4),), INT4, kind=FuncKind.SET_RETURNING),
        FuncSig("unnest", (array_of(TEXT),), TEXT, kind=FuncKind.SET_RETURNING),
    ]

    ops: list[OpSig] = []

    # Same-type arithmetic for the four numeric types.
    for sym in ("+", "-", "*", "/"):
        for t in (INT4, INT8, NUMERIC, FLOAT8):
            ops.append(OpSig(sym, t, t, t))

    # Modulo: defined for integer/numeric only (FLOAT8 modulo uses mod()).
    for t in (INT4, INT8, NUMERIC):
        ops.append(OpSig("%", t, t, t))

    # A few cross-type numeric operators for variety. PG implicitly
    # promotes, so we get richer outputs by registering these explicitly.
    # Not exhaustive — registering every promotion would multiply the
    # arithmetic operator pool four-fold without adding much expressive
    # range. The handful below ensures the cross-type path gets exercised
    # regularly during fuzzing.
    ops.append(OpSig("+", INT4, INT8, INT8))
    ops.append(OpSig("+", INT8, INT4, INT8))
    ops.append(OpSig("+", INT4, NUMERIC, NUMERIC))
    ops.append(OpSig("+", NUMERIC, INT4, NUMERIC))

    # Comparison: every comparable type → bool.
    for sym in ("=", "<>", "<", "<=", ">", ">="):
        for t in (INT4, INT8, NUMERIC, FLOAT8, TEXT, DATE, TIMESTAMPTZ, BOOL, UUID):
            ops.append(OpSig(sym, t, t, BOOL))

    # Logical
    ops.append(OpSig("AND", BOOL, BOOL, BOOL))
    ops.append(OpSig("OR", BOOL, BOOL, BOOL))

    # String
    ops.append(OpSig("||", TEXT, TEXT, TEXT))
    ops.append(OpSig("LIKE", TEXT, TEXT, BOOL))
    ops.append(OpSig("ILIKE", TEXT, TEXT, BOOL))

    # Date/time arithmetic. NOTE: `date ± interval` is intentionally NOT
    # registered — PG types it as `timestamp` (without time zone), a type
    # waxsql doesn't model; declaring it TIMESTAMPTZ was a type-system lie
    # masked only by PG's implicit timestamp→timestamptz cast (ISSUES.md
    # #50). `timestamptz ± interval` is correctly typed and kept.
    ops.append(OpSig("+", TIMESTAMPTZ, INTERVAL, TIMESTAMPTZ))
    ops.append(OpSig("-", TIMESTAMPTZ, INTERVAL, TIMESTAMPTZ))
    ops.append(OpSig("-", TIMESTAMPTZ, TIMESTAMPTZ, INTERVAL))

    # JSON access
    ops.append(OpSig("->", JSONB, TEXT, JSONB))
    ops.append(OpSig("->>", JSONB, TEXT, TEXT))
    ops.append(OpSig("@>", JSONB, JSONB, BOOL))
    ops.append(OpSig("?", JSONB, TEXT, BOOL))

    return Catalog(functions=tuple(fs), operators=tuple(ops))
