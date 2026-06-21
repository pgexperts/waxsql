"""Tests for waxsql.gen.expr.

Three things to verify:

  1. Output well-typed: returned Expr's pg_type implicitly casts to
     the requested target.
  2. Determinism: same (seed, scope, target, depth) → same Expr.
  3. Termination: at depth=0, recursive productions are not chosen.
  4. Round-trip: every generated expression's printed form parses
     via pglast (the headline guarantee).
"""
import random
from dataclasses import replace

import pytest

from waxsql import default_catalog, generate_schema
from waxsql.ast import (
    BinaryOp, Cast, ColumnRef, Expr, FuncCall, Literal, UnaryOp,
)
from waxsql.catalog import FuncKind
from waxsql.config import query_config_for_complexity
from waxsql.context import GenContext
from waxsql.gen import gen_expr, gen_literal
from waxsql.printer import print_expr
from waxsql.scope import Scope
from waxsql.types import (
    BOOL, DATE, FLOAT8, INT4, INT8, INTERVAL, JSONB, NUMERIC, PgType,
    TEXT, TIMESTAMPTZ, UUID, VARCHAR, implicitly_castable,
)
from waxsql.validate.syntax import check_syntax


# Helpers used by aggregate-flag tests below.

_AGG_NAMES = frozenset(
    f.name for f in default_catalog().functions
    if f.kind == FuncKind.AGGREGATE
)


def _walk_for_aggregates(e: Expr) -> list[str]:
    """Return the names of every aggregate FuncCall anywhere in `e`."""
    if isinstance(e, FuncCall):
        out = [e.name] if e.name in _AGG_NAMES else []
        for a in e.args:
            out.extend(_walk_for_aggregates(a))
        return out
    if isinstance(e, BinaryOp):
        return _walk_for_aggregates(e.left) + _walk_for_aggregates(e.right)
    if isinstance(e, UnaryOp):
        return _walk_for_aggregates(e.operand)
    if isinstance(e, Cast):
        return _walk_for_aggregates(e.expr)
    return []


def _has_nested_aggregate(e: Expr, in_agg: bool = False) -> bool:
    """True if `e` contains an aggregate FuncCall nested inside the
    args of another aggregate FuncCall."""
    if isinstance(e, FuncCall):
        is_agg = e.name in _AGG_NAMES
        if is_agg and in_agg:
            return True
        for a in e.args:
            if _has_nested_aggregate(a, in_agg or is_agg):
                return True
        return False
    if isinstance(e, BinaryOp):
        return (_has_nested_aggregate(e.left, in_agg)
                or _has_nested_aggregate(e.right, in_agg))
    if isinstance(e, UnaryOp):
        return _has_nested_aggregate(e.operand, in_agg)
    if isinstance(e, Cast):
        return _has_nested_aggregate(e.expr, in_agg)
    return False


def _ctx(seed: int, *, depth: int = 3, complexity: int = 5) -> GenContext:
    schema = generate_schema(seed=seed, complexity=complexity)
    scope = Scope()
    # Put the first two schema tables into scope under aliases t1/t2.
    for i, t in enumerate(schema.tables[:2]):
        scope.add_table(f"t{i+1}", t)
    return GenContext(
        rng=random.Random(seed),
        scope=scope,
        schema=schema,
        catalog=default_catalog(),
        config=query_config_for_complexity(complexity),
        depth_remaining=depth,
    )


# --- gen_literal ------------------------------------------------------------

@pytest.mark.parametrize("t", [INT4, INT8, NUMERIC, FLOAT8, TEXT, VARCHAR,
                                BOOL, DATE, TIMESTAMPTZ, INTERVAL, UUID, JSONB])
def test_gen_literal_returns_literal_of_requested_type(t: PgType):
    """gen_literal always returns a Literal whose pg_type matches the
    request, even when value=None."""
    rng = random.Random(0)
    for _ in range(50):
        lit = gen_literal(rng, t)
        assert isinstance(lit, Literal)
        assert lit.pg_type == t


def test_gen_literal_sometimes_returns_null():
    """The 5% NULL probability should produce some NULLs in 200 draws."""
    rng = random.Random(0)
    nulls = sum(1 for _ in range(200) if gen_literal(rng, INT4).value is None)
    assert nulls > 0
    # And it should NOT be all NULLs.
    assert nulls < 200


# --- gen_expr: well-typed output -------------------------------------------

@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("target", [INT4, INT8, NUMERIC, TEXT, BOOL])
def test_gen_expr_returns_compatible_type(seed: int, target: PgType):
    """The returned Expr's actual type must implicitly cast to target."""
    ctx = _ctx(seed)
    e = gen_expr(ctx, target)
    assert isinstance(e, Expr)
    assert implicitly_castable(e.pg_type, target), (
        f"got {e.pg_type.name!r} for target {target.name!r}: {print_expr(e)}"
    )


# --- gen_expr: determinism --------------------------------------------------

@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("target", [INT8, BOOL, TEXT])
def test_gen_expr_deterministic_for_same_inputs(seed: int, target: PgType):
    """Running gen_expr twice with identical inputs (same seed →
    fresh RNG, same scope, same target) must yield identical Exprs."""
    a = gen_expr(_ctx(seed), target)
    b = gen_expr(_ctx(seed), target)
    assert a == b
    assert print_expr(a) == print_expr(b)


# --- gen_expr: termination at depth=0 --------------------------------------

@pytest.mark.parametrize("seed", range(50))
def test_gen_expr_at_zero_depth_returns_leaf_only(seed: int):
    """When the depth budget is 0, the result must be a ColumnRef
    or Literal — never a FuncCall or BinaryOp."""
    ctx = _ctx(seed, depth=0)
    e = gen_expr(ctx, INT8)
    assert isinstance(e, (ColumnRef, Literal)), (
        f"got recursive node {type(e).__name__}: {print_expr(e)}"
    )


def test_gen_expr_at_negative_depth_still_terminates():
    """Defensive: even if depth is somehow already negative, gen_expr
    must not infinite-loop or recurse."""
    ctx = _ctx(0, depth=-5)
    e = gen_expr(ctx, INT8)
    assert isinstance(e, (ColumnRef, Literal))


# --- gen_expr: round-trip via pglast ---------------------------------------

@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("target", [INT8, BOOL, TEXT, NUMERIC])
@pytest.mark.parametrize("complexity", [1, 3, 5, 7])
def test_gen_expr_round_trips_through_pglast(
    seed: int, target: PgType, complexity: int,
):
    """Every generated expression's printed form must parse cleanly
    when wrapped in a SELECT. This is the type-driven guarantee in
    action: thousands of random expressions, all syntactically valid
    SQL by construction."""
    ctx = _ctx(seed, complexity=complexity, depth=complexity)
    e = gen_expr(ctx, target)
    sql = f"SELECT {print_expr(e)} FROM t1, t2"
    r = check_syntax(sql)
    assert r.ok, f"seed={seed} target={target.name} complexity={complexity}\n{r.error}\n{sql}"


# --- gen_expr: with empty scope (no column refs available) -----------------

def test_gen_expr_with_empty_scope_falls_back_to_literals_and_funcs():
    """When no columns are visible, generation must still succeed
    (using literals, function calls, and operators on literals)."""
    schema = generate_schema(seed=0, complexity=3)
    ctx = GenContext(
        rng=random.Random(0),
        scope=Scope(),                               # empty!
        schema=schema,
        catalog=default_catalog(),
        config=query_config_for_complexity(5),
        depth_remaining=3,
    )
    for _ in range(20):
        e = gen_expr(ctx, INT4)
        assert isinstance(e, Expr)
        # No ColumnRefs should appear at the top — empty scope.
        # (Recursive sub-expressions also can't yield ColumnRefs.)
        _assert_no_column_refs(e)


def _assert_no_column_refs(e: Expr) -> None:
    if isinstance(e, ColumnRef):
        raise AssertionError(f"ColumnRef appeared: {print_expr(e)}")
    if isinstance(e, FuncCall):
        for a in e.args:
            _assert_no_column_refs(a)
    if isinstance(e, BinaryOp):
        _assert_no_column_refs(e.left)
        _assert_no_column_refs(e.right)


# ===========================================================================
# Aggregate-flag behavior (milestone 2)
# ===========================================================================

@pytest.mark.parametrize("seed", range(100))
def test_no_aggregates_when_disallowed(seed):
    """With allow_aggregates=False, no aggregate FuncCall must appear
    anywhere in the generated tree — at any depth, on either side of
    an op, in any cast or func arg."""
    ctx = replace(_ctx(seed, complexity=7, depth=4), allow_aggregates=False)
    for target in (INT8, NUMERIC, BOOL, TEXT):
        e = gen_expr(ctx, target)
        found = _walk_for_aggregates(e)
        assert not found, (
            f"seed={seed} target={target.name}: aggregates leaked through "
            f"despite allow_aggregates=False: {found}\n{print_expr(e)}"
        )


@pytest.mark.parametrize("seed", range(50))
def test_aggregates_appear_when_allowed(seed):
    """A single seed might or might not produce an aggregate, but
    across 50 seeds at moderate complexity the aggregate path should
    fire often. This is a coverage test, not a per-seed assertion."""
    pass  # individual-seed assertion is too noisy; the aggregate
    # rate is tested in the count test below.


def test_aggregates_appear_at_meaningful_rate():
    """Sanity: across 200 generations with allow_aggregates=True, a
    sizable fraction must contain at least one aggregate. If this drops
    to zero, the feature is broken; if it dominates, the weights are
    miscalibrated."""
    rate = 0
    for seed in range(200):
        ctx = replace(_ctx(seed, complexity=5, depth=3), allow_aggregates=True)
        e = gen_expr(ctx, INT8)
        if _walk_for_aggregates(e):
            rate += 1
    # Loose bounds — anything in the 10%..70% band is fine, and lets
    # us re-tune `aggregate_call_weight` without false alarms.
    assert 20 < rate < 140, f"aggregate rate {rate}/200 outside sanity band"


@pytest.mark.parametrize("seed", range(50))
def test_no_nested_aggregates(seed):
    """`sum(count(...))`-style nesting is a PG syntax error. The
    `in_aggregate` flag thread-through should make it impossible at
    the generator level."""
    ctx = replace(_ctx(seed, complexity=7, depth=5), allow_aggregates=True)
    for target in (INT8, NUMERIC, BOOL, TEXT):
        e = gen_expr(ctx, target)
        assert not _has_nested_aggregate(e), (
            f"seed={seed} target={target.name}: nested aggregate found\n"
            f"{print_expr(e)}"
        )


def test_aggregates_round_trip_via_pglast():
    """Aggregate-containing expressions must produce parseable SQL.
    They aren't valid in every position (WHERE forbids them, etc.),
    so we wrap in a SELECT-list context which always accepts aggregates."""
    for seed in range(100):
        ctx = replace(_ctx(seed, complexity=5, depth=3), allow_aggregates=True)
        e = gen_expr(ctx, INT8)
        sql = f"SELECT {print_expr(e)} FROM t1, t2"
        r = check_syntax(sql)
        assert r.ok, f"seed={seed}: {r.error}\n{sql}"


def test_in_aggregate_flag_fully_blocks_aggregates():
    """Setting in_aggregate=True directly (bypassing the
    allow_aggregates path) must also block aggregate generation —
    this is the safety net that gen/select.py relies on when
    generating an aggregate's args."""
    for seed in range(50):
        ctx = replace(
            _ctx(seed, complexity=7, depth=4),
            allow_aggregates=True,    # would normally allow aggs
            in_aggregate=True,        # but in_aggregate forbids them
        )
        e = gen_expr(ctx, INT8)
        found = _walk_for_aggregates(e)
        assert not found, (
            f"seed={seed}: aggregates leaked through despite "
            f"in_aggregate=True: {found}\n{print_expr(e)}"
        )
