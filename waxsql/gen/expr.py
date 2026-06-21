"""Typed expression generator.

Role: the type-driven heart of the query generator (pillar 1 in the
project's architectural pillars). Every other generator that needs an
expression — SELECT targets, WHERE predicates, JOIN ON conditions,
HAVING, FILTER, window-function arguments, etc. — funnels through
`gen_expr` so that type compatibility is enforced by construction
rather than checked after the fact.

The core recursion of the query generator. Given a `GenContext` and a
target `PgType`, produce an `Expr` whose actual type implicitly casts
to the target. The candidate productions are:

  * **Column reference** — pull a binding from the current scope whose
    type implicitly casts to the target. Available at any depth.

  * **Literal** — synthesize a typed constant via `gen_literal`. Always
    available (the catch-all when nothing else fits).

  * **Function call** — pick a scalar function from the catalog whose
    return type matches; recursively generate one arg per signature
    parameter. Recursive — only available when the depth budget
    allows.

  * **Binary operator** — same as function call but for binary ops.

  * **Aggregate function call** — only when `ctx.allow_aggregates`
    is True AND we're not already inside an aggregate (`in_aggregate`
    False). Same shape as a scalar function call, but the recursive
    arg-generation context has `in_aggregate=True`, which prevents
    nested aggregates like `sum(count(...))` (PG syntax error).
    Recursive, gated on the depth budget.

  * **Subquery productions** — `(SELECT ...)` (scalar), `EXISTS (...)`,
    or `<expr> IN (...)`. Treated as leaves of the *outer* expression
    tree (no expression-depth budget consumed for them) but rationed
    by their own `subquery_depth_remaining` budget. EXISTS and IN are
    only candidates when target_type is BOOL; scalar subqueries are
    candidates for any target type. All three require `allow_subquery`
    AND the matching feature flag.

The depth budget rations recursion. Every recursive call descends by
one; when `ctx.at_leaf()`, only column refs and literals are
considered. Recursive productions are *additionally* downweighted as
depth shrinks (`(depth_remaining/max_depth) ** leaf_bias`), so trees
don't always max out their budget — they collapse early often enough
to keep output varied.

Determinism rules respected here:

  * All randomness comes from `ctx.rng`, never the global `random`.
  * Catalog lookup methods return sorted lists — iteration order is
    stable across Python builds.
  * Scope's `visible_columns` returns bindings in insertion order,
    which is also stable for any deterministic FROM-clause walk.
"""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import replace

from ..ast import (
    BinaryOp, Cast, ColumnRef, Expr, FuncCall, Literal, OrderByItem,
)
from ..catalog import FuncKind, FuncSig
from ..config import (
    FEATURE_EXISTS, FEATURE_IN_SUBQUERY, FEATURE_SCALAR_SUBQUERY,
    FEATURE_WINDOW_FUNCTION,
)
from ..context import GenContext
from ..types import (
    BOOL, DATE, FLOAT8, INT4, INT8, INTERVAL, JSONB, NUMERIC,
    PgType, TEXT, TIMESTAMPTZ, UUID, VARCHAR, implicitly_castable,
)
from .window import gen_window_spec


# Probability of emitting a typed NULL when the literal production is
# selected. Keeps a steady NULL rate flowing through expressions so
# that the printer's `NULL::T` path stays exercised, without drowning
# the output in NULLs.
_NULL_PROB: float = 0.05

# Probability of emitting `count(*)` instead of `count(<arg>)` when
# the chosen aggregate happens to be count. Both forms are common in
# real SQL — `count(col)` counts non-NULL values, `count(*)` counts
# all rows including all-NULL. Half-and-half is a reasonable balance;
# tune if outputs feel monotonous in either direction.
_P_COUNT_STAR: float = 0.5


# Valid unit keywords for date_trunc's first arg. PG accepts these
# 13 strings (and a few abbreviations); arbitrary TEXT errors at
# runtime ("unit X not recognized"). The list is fixed at PG-spec
# values — see https://www.postgresql.org/docs/current/functions-
# datetime.html#FUNCTIONS-DATETIME-TRUNC for canonical names.
_DATE_TRUNC_UNITS: tuple[str, ...] = (
    "microseconds", "milliseconds", "second", "minute",
    "hour", "day", "week", "month", "quarter", "year",
    "decade", "century", "millennium",
)

# Fixed JSONB-array literals for the jsonb_array_length arg-rewrite.
# Ordinary JSONB literals (`_JSONB_LIT` above) are objects; passing
# them to jsonb_array_length errors with "cannot get array length of
# a non-array". A small pool gives output variety while guaranteeing
# array-ness.
_JSONB_ARRAY_LITS: tuple[str, ...] = (
    "[1, 2, 3]", '["alpha", "beta"]', "[true, false]", "[]",
)

# Ordered-set aggregates: must be called with a WITHIN GROUP clause
# (PG rejects them otherwise). Set membership drives the agg-branch
# special-case in gen_expr (and the parallel sites in gen/select.py)
# that routes through `gen_ordered_set_agg` rather than the regular
# construction path. Public name (no leading underscore) because
# select.py imports it.
ORDERED_SET_AGGREGATES: frozenset[str] = frozenset({
    "percentile_cont", "percentile_disc",
})


def gen_ordered_set_agg(ctx: GenContext, f: FuncSig) -> FuncCall:
    """Build an ordered-set aggregate FuncCall with the required
    WITHIN GROUP clause.

    For percentile_cont / percentile_disc: the call arg is a fraction
    (FLOAT8 in [0,1]); the WITHIN GROUP ORDER BY is a numeric
    expression (we use FLOAT8 to match the catalog's declared return
    type). PG actually allows polymorphic ORDER BY types, but the
    return type changes accordingly — sticking with FLOAT8 keeps the
    catalog's declared return correct.

    The fraction is drawn from a small fixed pool (0.25, 0.5, 0.75)
    rather than a random float so output stays readable. Could
    extend to arbitrary literals if more variety is wanted.

    FILTER and OVER are deliberately not attached here — composing
    every clause variant on every aggregate kind is more code than
    polish-tier value justifies. Ordered-set aggs in real SQL are
    almost always used bare anyway.
    """
    rng = ctx.rng

    # Args: a single FLOAT8 fraction literal in [0, 1].
    fraction = rng.choice((0.25, 0.5, 0.75))
    args = (Literal(FLOAT8, fraction),)

    # WITHIN GROUP ORDER BY: one FLOAT8 expression. The expression
    # is evaluated per-row (pre-aggregation), so the same forbidden-
    # combinations apply as for regular aggregate args:
    #   * No nested aggregates (in_aggregate=True empties aggs pool).
    #   * No window functions (allow_window=False blocks the window
    #     branch). PG rejects `agg(... window(...) OVER (...) ...)`
    #     with "aggregate function calls cannot contain window
    #     function calls" — and that includes window-style use of
    #     other aggregates like `percentile_cont(x) OVER (...)`.
    wg_ctx = replace(
        ctx.descend(),
        in_aggregate=True,
        allow_window=False,
        in_window=False,
    )
    wg_expr = gen_expr(wg_ctx, FLOAT8)
    wg_expr = coerce_to_param_type(wg_expr, FLOAT8)
    within_group = (
        OrderByItem(expr=wg_expr, direction=rng.choice(("ASC", "DESC"))),
    )

    return FuncCall(
        f.returns, f.name, args, within_group=within_group,
    )


def _replace_literal_zero(expr: Expr, expected_type: PgType) -> Expr:
    """If `expr` is a numeric literal with value 0 (or 0.0), return a
    fresh literal of `expected_type` with value 1 instead. Otherwise
    return `expr` unchanged.

    Used at division/modulo emission sites to suppress the trivial
    `x / 0` / `x % 0` case that PG constant-folds to a 22012 error.
    Doesn't try to recursively detect zero-folding through arithmetic
    (`y - y`, `0 * x`, ...) — those are statistically rare and
    catching them robustly would require a constant-folding pass.
    The bare-literal case is the dominant one.
    """
    if not isinstance(expr, Literal):
        return expr
    if expr.value is None:
        return expr
    # Numeric types only; bool/text/etc. literals can't be zero in a
    # division-relevant sense.
    if expr.pg_type not in (INT4, INT8, NUMERIC, FLOAT8):
        return expr
    if expr.value == 0:
        # Literal of `expected_type` with value 1. Keeps the type
        # discipline that the divisor's type matches the operator's
        # right-hand param (set up by coerce_to_param_type just before
        # this call).
        return Literal(expected_type, 1 if expected_type in (INT4, INT8) else 1.0)
    return expr


# ---------------------------------------------------------------------------
# Per-function argument rewriters
# ---------------------------------------------------------------------------
#
# A few catalog functions declare a generic arg type (TEXT / JSONB) but
# require a CONSTRAINED value at PLAN time: date_trunc's first arg must be
# a unit keyword, jsonb_array_length's arg must be a JSON array, mod's
# divisor must not constant-fold to zero. The catalog can't express "this
# TEXT must be one of these keywords", so the generator patches the
# generated args. Centralizing the "which function needs which rewrite"
# knowledge in one registry (vs scattered `if f.name == ...` branches at
# the call site) gives that knowledge a single home — a future catalog
# change updates one table instead of hunting call sites. Each rewriter
# mutates `args_list` in place and consumes `ctx.rng` identically to its
# former inline form; at most one fires per call (catalog names are
# unique), so determinism is unchanged.


def _rewrite_mod_args(args_list: list[Expr], ctx: GenContext, f: FuncSig) -> None:
    # `mod(x, 0)` constant-folds to a 22012 division-by-zero error at PLAN
    # time. Same suppression as the `/` / `%` operator branch — replace a
    # bare-literal-zero second arg with 1.
    if len(args_list) >= 2:
        args_list[1] = _replace_literal_zero(args_list[1], f.args[1])


def _rewrite_date_trunc_args(args_list: list[Expr], ctx: GenContext, f: FuncSig) -> None:
    # date_trunc's first arg must be a recognized UNIT keyword. The catalog
    # signature is generic TEXT; replace whatever gen_expr produced with a
    # fresh literal from the whitelist. PG validates the unit at run AND
    # PLAN time (when not behind a cast), so a generic text literal would
    # make EXECUTE fail without this rewrite.
    if len(args_list) >= 1:
        args_list[0] = Literal(TEXT, ctx.rng.choice(_DATE_TRUNC_UNITS))


def _rewrite_jsonb_array_length_args(
    args_list: list[Expr], ctx: GenContext, f: FuncSig
) -> None:
    # jsonb_array_length requires its arg to be a JSON ARRAY, not an
    # object. Our default JSONB literal is `{"k":"v"}` (object), and
    # jsonb_build_object produces objects — replace with a guaranteed-array
    # literal. Same safety pattern as date_trunc, different constraint.
    if len(args_list) >= 1:
        args_list[0] = Literal(JSONB, ctx.rng.choice(_JSONB_ARRAY_LITS))


# Function name -> in-place argument rewriter. Resolved by exact-name
# lookup (never iterated), so this dict does not violate the
# no-iteration-over-collections-in-RNG-paths determinism rule.
_ARG_REWRITERS: dict[str, Callable[[list[Expr], GenContext, FuncSig], None]] = {
    "mod": _rewrite_mod_args,
    "date_trunc": _rewrite_date_trunc_args,
    "jsonb_array_length": _rewrite_jsonb_array_length_args,
}


def coerce_to_param_type(arg: Expr, param_type: PgType) -> Expr:
    """Wrap `arg` in an explicit `::param_type` cast iff its actual
    type isn't exactly `param_type`.

    Why this matters: the catalog declares each FuncSig's return type
    based on the declared param types. PG resolves the actual call by
    looking at the runtime arg types — and when our generator passes
    an INT4 to a NUMERIC slot (legal via implicit cast), PG's overload
    resolution may pick a DIFFERENT overload returning a DIFFERENT
    type. Classic case: `floor(int8)` resolves to `floor(double
    precision) → double precision` in PG (preferred-type tiebreak),
    not `floor(numeric) → numeric` as our catalog claims.

    Wrapping the arg with `::numeric` matches PG's resolution to our
    catalog's metadata: PG sees `floor(int8::numeric)` and unambiguously
    picks `floor(numeric)`. The same trick fixes 42804 unknown-type
    failures from bare string literals — `'foo'::text` is concretely
    typed where `'foo'` would be 'unknown' until PG infers context.

    Single shared helper so all four expression-construction sites —
    gen_expr's func/op/agg/window dispatch, plus select.py's two
    aggregate-construction helpers — apply identical coercion.
    """
    if arg.pg_type == param_type:
        return arg
    return Cast(pg_type=param_type, expr=arg, target_type=param_type)


def should_emit_count_star(rng: random.Random) -> bool:
    """Coin flip for the `count(*)` substitution. Lives here (next
    to the probability constant) so all three agg-construction sites
    — this module's agg/window dispatch, gen/select.py's
    `_gen_aggregate_funccall` and `_gen_having_expr` — share one
    source of truth for the rate. Caller is responsible for checking
    `f.name == "count"` first; this helper just provides the dice."""
    return rng.random() < _P_COUNT_STAR


# Probability of attaching a `FILTER (WHERE ...)` clause to an
# aggregate call. Lower than _P_COUNT_STAR because count(*) replaces
# an existing form (every count picks one form or the other), while
# FILTER is purely additive — every aggregate with a filter is
# strictly noisier than the same aggregate without one. 20% strikes
# a balance: visible but not dominant.
_P_FILTER: float = 0.2


def should_emit_filter(rng: random.Random) -> bool:
    """Coin flip for attaching a FILTER clause. PG only accepts
    FILTER on aggregate (and ordered-set/hypothetical-set) calls;
    callers MUST check `f.kind == FuncKind.AGGREGATE` before using
    this dice. Window-style aggregates are also eligible — the
    `count(*) FILTER (WHERE x > 0) OVER (...)` shape is valid PG."""
    return rng.random() < _P_FILTER


def gen_filter_predicate(ctx: GenContext) -> "Expr":
    """Build a BOOL expression for use as a FILTER (WHERE ...)
    predicate.

    Generated under in_aggregate=True (no nested aggregates inside
    FILTER — PG: "aggregate function calls cannot be nested") and
    allow_window=False (no windows — PG: "window functions are not
    allowed in FILTER"). depth-decremented so the predicate doesn't
    blow the parent's depth budget."""
    pred_ctx = replace(
        ctx.descend(),
        in_aggregate=True,
        allow_window=False,
        in_window=False,
    )
    return gen_expr(pred_ctx, BOOL)


# Curated word list for string literals. Deliberately small and free
# of apostrophes / non-ASCII so the printer's literal escaper has
# nothing surprising to handle. Add more here if string-heavy output
# starts feeling repetitive — there's no determinism implication.
_WORDS: tuple[str, ...] = (
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
    "foo", "bar", "baz", "quux", "lorem", "ipsum",
    "north", "south", "east", "west",
    "one", "two", "three", "four", "five",
)

# Small "nice" numbers. Generators that want a richer numeric range
# can use the operator pool to compose them; this just gives the
# literal-leaf production something deterministic to draw from.
_INTS: tuple[int, ...] = (0, 1, 2, 3, 5, 7, 10, 42, 100, 1000)
_FLOATS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.14, 10.0, 100.0)

# Fixed temporal/UUID/JSONB literals. The schema and operator pool
# provide the variety; the literal leaf just needs *some* parseable
# value of the right type.
_DATE_LIT = "2024-01-01"
_TIMESTAMPTZ_LIT = "2024-01-01 12:00:00+00"
_INTERVAL_LIT = "1 day"
_UUID_LIT = "00000000-0000-0000-0000-000000000000"
_JSONB_LIT = '{"k":"v"}'


# ---------------------------------------------------------------------------
# Literal generation
# ---------------------------------------------------------------------------

def gen_literal(rng: random.Random, t: PgType) -> Literal:
    """Generate a typed literal of the requested type.

    With probability `_NULL_PROB` the literal is NULL (which the
    printer renders as `NULL::T`). Otherwise a type-appropriate
    constant is drawn from the curated lists above.

    Always returns a Literal — never raises, never returns None.
    Unknown types fall through to a typed NULL.
    """
    if rng.random() < _NULL_PROB:
        return Literal(t, None)

    if t == INT4 or t == INT8:
        return Literal(t, rng.choice(_INTS))
    if t == NUMERIC or t == FLOAT8:
        return Literal(t, rng.choice(_FLOATS))
    if t == TEXT or t == VARCHAR:
        return Literal(t, rng.choice(_WORDS))
    if t == BOOL:
        # Avoid importing a `True`/`False` constant just to give them
        # equal weight; rng.random() < 0.5 is one fewer dependency.
        return Literal(t, rng.random() < 0.5)
    if t == DATE:
        return Literal(t, _DATE_LIT)
    if t == TIMESTAMPTZ:
        return Literal(t, _TIMESTAMPTZ_LIT)
    if t == INTERVAL:
        return Literal(t, _INTERVAL_LIT)
    if t == UUID:
        return Literal(t, _UUID_LIT)
    if t == JSONB:
        return Literal(t, _JSONB_LIT)

    # Unrecognized type (e.g. an array, or a future user type): fall
    # back to typed NULL. The printer handles `NULL::<sql>` for any
    # PgType because PgType.sql() always renders.
    return Literal(t, None)


# ---------------------------------------------------------------------------
# Expression generation
# ---------------------------------------------------------------------------

def gen_expr(ctx: GenContext, target_type: PgType) -> Expr:
    """Generate an expression whose value type implicitly casts to
    `target_type`.

    See module docstring for the production set and the depth-budget
    discipline. The function is total — it always returns an Expr,
    even when the catalog has nothing useful for the requested type
    (literal fallback).
    """
    # INVARIANT: aggregate-context check happens BEFORE column-ref
    # resolution. Inside an aggregate's argument that itself lives in
    # a correlated subquery, an outer-column reference would force
    # the OUTER query into implicit-single-group mode (PG: 42803).
    # The local-bindings narrowing here is what prevents that mode
    # shift — same reasoning as PG's own parse-analysis ordering.
    # Column-ref candidates. Inside an aggregate's args within a
    # correlated subquery, restrict to LOCAL bindings only — outer-
    # column refs in this position trigger PG's implicit-grouping
    # inference on the OUTER query (PARSE-tier error 42803,
    # "column must appear in the GROUP BY clause"). Outside aggregates,
    # outer-column refs are the whole point of correlation, so the
    # full visible_columns walk is correct.
    if ctx.in_aggregate and ctx.in_correlated_subquery:
        cols = ctx.scope.local_bindings(of_type=target_type)
    else:
        cols = ctx.scope.visible_columns(of_type=target_type)
    funcs = ctx.catalog.scalar_funcs_returning(target_type)
    ops = ctx.catalog.binary_ops_returning(target_type)
    # Aggregates are only candidates when the surrounding context
    # allows them AND we're not already inside one (no nested aggs).
    aggs = (
        ctx.catalog.aggs_returning(target_type)
        if (ctx.allow_aggregates and not ctx.in_aggregate)
        else []
    )
    # Window-style calls: union of WINDOW-only functions (row_number,
    # rank, lag, ...) and AGGREGATE functions (which can also be used
    # as windows via OVER). Eligible when allow_window AND not
    # in_window AND the feature flag is set. The candidate is the
    # FuncSig itself; we'll attach a WindowSpec at dispatch time.
    window_candidates: list[FuncSig] = []
    if (ctx.allow_window
            and not ctx.in_window
            and FEATURE_WINDOW_FUNCTION in ctx.config.feature_flags):
        for f in ctx.catalog.functions:
            # Ordered-set aggregates (percentile_cont, percentile_disc,
            # ...) cannot be used as plain window functions —
            # `percentile_cont(x) OVER (...)` is a parse error in PG.
            # They MUST go through the WITHIN GROUP path, which is
            # in the regular agg branch. Exclude from window pool.
            if (f.kind in (FuncKind.WINDOW, FuncKind.AGGREGATE)
                    and f.name not in ORDERED_SET_AGGREGATES
                    and implicitly_castable(f.returns, target_type)):
                window_candidates.append(f)

    # Build (weight, kind) candidate list. Each kind names a
    # production we'll dispatch to below.
    # DETERMINISM: the append order below is fixed and never depends
    # on iteration over a set (the only set in scope here is implicit
    # in candidate filtering, which uses pre-sorted catalog lookups).
    # This is what makes `rng.choices` give identical picks across
    # Python builds for the same (seed, target_type, ctx) tuple.
    candidates: list[tuple[float, str]] = []

    # Leaves are always available.
    if cols:
        candidates.append((ctx.config.column_ref_weight, "col"))
    candidates.append((ctx.config.literal_weight, "lit"))

    # Subqueries are also leaves at the OUTER expression level — they
    # represent a single value (or a BOOL test) that the inner SELECT
    # produces. They DON'T consume expression depth (the descent
    # resets `depth_remaining` for the inner expression tree). Their
    # rationing comes from `subquery_depth_remaining`, a separate
    # budget. So no leaf_factor downweighting here.
    #
    # Note the gate uses `at_subquery_leaf()`, NOT `at_leaf()` — the
    # two budgets are independent. An expression deep in its parent's
    # depth budget can still spawn a subquery if the subquery budget
    # has room, and vice versa.
    if (ctx.allow_subquery
            and not ctx.at_subquery_leaf()):
        flags = ctx.config.feature_flags
        if FEATURE_SCALAR_SUBQUERY in flags:
            candidates.append(
                (ctx.config.scalar_subquery_weight, "scalar_sub")
            )
        # EXISTS and IN return BOOL exactly. BOOL doesn't implicitly
        # cast to anything else in our type graph, so the gate is a
        # plain target-type check.
        if target_type == BOOL:
            if FEATURE_EXISTS in flags:
                candidates.append(
                    (ctx.config.exists_weight, "exists")
                )
            if FEATURE_IN_SUBQUERY in flags:
                candidates.append(
                    (ctx.config.in_subquery_weight, "in_sub")
                )

    # Recursive productions only when the budget allows — and even
    # then, downweighted as the budget shrinks. The exponent is
    # `leaf_bias` from the config: 1.0 is linear decay, >1.0 favors
    # leaves more aggressively, <1.0 favors recursion.
    if not ctx.at_leaf():
        max_d = max(1, ctx.config.max_expr_depth)  # avoid /0
        leaf_factor = (ctx.depth_remaining / max_d) ** ctx.config.leaf_bias
        if funcs:
            candidates.append(
                (ctx.config.func_call_weight * leaf_factor, "func")
            )
        if ops:
            candidates.append(
                (ctx.config.binary_op_weight * leaf_factor, "op")
            )
        if aggs:
            candidates.append(
                (ctx.config.aggregate_call_weight * leaf_factor, "agg")
            )
        if window_candidates:
            # Window calls are recursive (args can be expressions),
            # so they get the same leaf_factor downweighting as
            # other recursive productions.
            candidates.append(
                (ctx.config.window_call_weight * leaf_factor, "window")
            )

    # Determinism note: candidate-list construction order above is
    # fixed (col, lit, subquery branches, recursive branches), so the
    # `kinds` / `weights` parallel lists are identical across runs for
    # a given (target_type, ctx) pair. `rng.choices` consumes that
    # ordering plus a single rng draw — same seed → same kind picked.
    weights = [w for w, _ in candidates]
    kinds = [k for _, k in candidates]
    kind = ctx.rng.choices(kinds, weights=weights, k=1)[0]

    if kind == "col":
        b = ctx.rng.choice(cols)
        return ColumnRef(b.type, b.table_alias, b.column)

    if kind == "lit":
        return gen_literal(ctx.rng, target_type)

    if kind == "func":
        f = ctx.rng.choice(funcs)
        # Each argument is generated at depth-1 from the parent (not
        # cumulatively across siblings — depth tracks tree depth,
        # not call-list length). Args are coerced to their declared
        # param type via explicit cast — see coerce_to_param_type
        # docstring for why our claimed return type only matches PG
        # when args match exactly.
        child_ctx = ctx.descend()
        args_list = [
            coerce_to_param_type(gen_expr(child_ctx, arg_t), arg_t)
            for arg_t in f.args
        ]
        # A few catalog functions need their generated args massaged for
        # PLAN-time validity (unit keyword, JSON array, non-zero divisor).
        # The _ARG_REWRITERS registry holds the per-function logic; at
        # most one fires (names are unique). See the registry definition
        # for the determinism note.
        rewriter = _ARG_REWRITERS.get(f.name)
        if rewriter is not None:
            rewriter(args_list, ctx, f)
        return FuncCall(f.returns, f.name, tuple(args_list))

    if kind == "op":
        o = ctx.rng.choice(ops)
        child_ctx = ctx.descend()
        # left/right are not None here: catalog.binary_ops_returning
        # filters out unary ops (which would have left=None or
        # right=None).
        assert o.left is not None and o.right is not None
        # Same coercion rationale as the func branch — operator
        # overload resolution is even more sensitive than function
        # resolution because there's no fallback "best-match" rule
        # for operators (PG either finds an exact match or errors).
        left = coerce_to_param_type(gen_expr(child_ctx, o.left), o.left)
        right = coerce_to_param_type(gen_expr(child_ctx, o.right), o.right)
        # Avoid `x / 0` and `x % 0` — PG constant-folds these and
        # raises 22012 "division by zero" at PLAN time. We only catch
        # the bare-literal-zero case here; complex zero-folding (e.g.
        # `x / (y - y)`) would still slip through to runtime, but PG
        # only constant-folds when both operands are constant, and a
        # constant-folded zero requires a literal zero somewhere in
        # the chain — replacing the bare-literal case statistically
        # eliminates it.
        if o.symbol in ("/", "%"):
            right = _replace_literal_zero(right, o.right)
        return BinaryOp(o.returns, o.symbol, left, right)

    if kind == "agg":
        # Aggregates are gated by BOTH `allow_aggregates` (set per-clause
        # by gen_select: True in SELECT-list/HAVING, False in WHERE/ON
        # /target-lists-of-subqueries) AND `in_aggregate=False` (no
        # nested aggregates: `sum(count(...))` is a hard parse error).
        # Both gates already applied above when building `aggs`.
        f = ctx.rng.choice(aggs)
        # Ordered-set aggregates (percentile_cont, percentile_disc,
        # mode, ...) MUST be called with a WITHIN GROUP clause; bare
        # `percentile_cont(0.5)` is a parse error. Special-case
        # before the regular agg path so we always attach the clause.
        if f.name in ORDERED_SET_AGGREGATES:
            return gen_ordered_set_agg(ctx, f)
        # FILTER decision is made up-front so it composes with the
        # star form (`count(*) FILTER (WHERE ...)`) — PG accepts both
        # together. All aggregates are FILTER-eligible by PG's rules.
        filter_expr = (
            gen_filter_predicate(ctx)
            if should_emit_filter(ctx.rng)
            else None
        )
        # `count(*)` special form: only valid for count, only at probability
        # `_P_COUNT_STAR`. The catalog still carries `count(INT4)` and
        # `count(TEXT)` as the source of count's INT8-returning behavior;
        # this just substitutes the star form at emission time. PG rejects
        # the * placeholder for any other aggregate (`sum(*)` etc.), so
        # the gate is name-specific.
        if f.name == "count" and should_emit_count_star(ctx.rng):
            return FuncCall(
                f.returns, "count", (), star=True, filter_=filter_expr,
            )
        # Args are generated under in_aggregate=True, which removes
        # aggregates from the candidate pool of the recursive call —
        # blocking sum(count(...)) and similar nested-aggregate forms
        # that PG rejects with a parse-analysis error.
        child_ctx = replace(ctx.descend(), in_aggregate=True)
        args = tuple(
            coerce_to_param_type(gen_expr(child_ctx, arg_t), arg_t)
            for arg_t in f.args
        )
        return FuncCall(f.returns, f.name, args, filter_=filter_expr)

    if kind == "window":
        f = ctx.rng.choice(window_candidates)
        # FILTER is only valid on AGGREGATE-kind window functions —
        # `row_number() FILTER (WHERE ...)` is a parse error because
        # row_number isn't an aggregate. The same gate that drives
        # count(*) eligibility (kind==AGGREGATE) drives FILTER
        # eligibility here.
        filter_expr = (
            gen_filter_predicate(ctx)
            if (f.kind == FuncKind.AGGREGATE
                and should_emit_filter(ctx.rng))
            else None
        )
        # `count(*) OVER (...)` is the canonical "running row count by
        # partition" idiom — same name-specific gate as the agg path.
        # Skip arg generation entirely for the star form; the OVER
        # spec is generated against the OUTER ctx as usual.
        if (f.kind == FuncKind.AGGREGATE
                and f.name == "count"
                and should_emit_count_star(ctx.rng)):
            spec = gen_window_spec(ctx)
            return FuncCall(
                f.returns, "count", (),
                over=spec, star=True, filter_=filter_expr,
            )
        # Window args have three forbidden things at once: nested
        # windows (PG rejects sum(row_number() OVER (...)) OVER (...)),
        # aggregates inside non-aggregate window args (the rules get
        # subtle), and ordinary aggregates inside aggregate-as-window
        # args (sum(count(...)) OVER (...) is invalid). Three flag
        # resets cover the whole forbidden-combinations space.
        arg_ctx = replace(
            ctx.descend(),
            in_window=True,
            allow_window=False,
            allow_aggregates=False,
            in_aggregate=True,  # also blocks aggs in window args
        )
        args = tuple(
            coerce_to_param_type(gen_expr(arg_ctx, arg_t), arg_t)
            for arg_t in f.args
        )
        # The window spec itself uses the OUTER ctx's scope (which
        # has the same visible columns as the surrounding SELECT
        # list); column refs in PARTITION BY / ORDER BY are valid
        # there.
        spec = gen_window_spec(ctx)
        return FuncCall(
            f.returns, f.name, args, over=spec, filter_=filter_expr,
        )

    if kind in ("scalar_sub", "exists", "in_sub"):
        # Lazy import: gen/subquery.py imports gen_expr at module
        # top level (it generates the inner WHERE/target via
        # gen_expr), so a top-level import here would close the
        # cycle. The import is cached in sys.modules after first
        # use — only the first call pays the lookup cost.
        # NOTE: the subquery branch consumes one slot from
        # `subquery_depth_remaining`, not from the expression-depth
        # budget — see the at_subquery_leaf gate above.
        from . import subquery as _sq
        # 50/50 correlated vs uncorrelated. Correlated subqueries are
        # more realistic-looking but not always achievable (the
        # forcer falls back to literal RHS if no compatible inner
        # column exists); the runtime decision is per-subquery.
        correlated = ctx.rng.random() < 0.5
        if kind == "scalar_sub":
            return _sq.gen_scalar_subquery(
                ctx, target_type, correlated=correlated,
            )
        if kind == "exists":
            return _sq.gen_exists_subquery(ctx, correlated=correlated)
        # kind == "in_sub"
        return _sq.gen_in_subquery(ctx, correlated=correlated)

    # Unreachable — `kinds` is always non-empty (literal is always
    # in the candidate set). The fallback is here defensively to
    # satisfy type-checkers and to fail loudly if a future change
    # accidentally drops the literal candidate.
    raise RuntimeError(f"no production picked (kinds={kinds})")
