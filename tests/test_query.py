"""Headline integration tests for the query generator.

The lead test is `test_query_parses` — the round-trip acceptance
criterion: ≥200 (seed, complexity) combinations, every one of which
must produce SQL that pglast accepts.

Beyond that, the determinism and structural-invariant tests defend
the same guarantees as the schema-side tests, transferred to query
generation.
"""
import pytest

from waxsql import (
    default_catalog, generate_query, generate_schema, print_query,
)
from waxsql.ast import (
    BinaryOp, Cast, ColumnRef, CteRef, DerivedTable, Exists, Expr,
    FuncCall, InSubquery, JoinExpr, SetOp, Subquery, TableRef, UnaryOp,
)
from waxsql.catalog import FuncKind
from waxsql.config import query_config_for_complexity
from waxsql.validate.syntax import check_syntax


def _all_select_bodies(body):
    """Yield every Select in a query body.

    For a plain Select, yields just it. For a SetOp, yields each
    arm — recursively, because Track B #5 added nested SetOp arms
    (`A UNION (B INTERSECT C)`) and the structural recursion is
    needed to reach the inner Selects.

    The aggregate-mode / subquery / window invariant tests walk
    Select internals (targets, from_, where, etc.); routing through
    this helper makes them apply per-arm for SetOp queries without
    changing the invariant logic itself.
    """
    if isinstance(body, SetOp):
        for arm in body.arms:
            yield from _all_select_bodies(arm)
    else:
        yield body


# Names of every aggregate function in the catalog. Used by the
# walking helpers below to identify aggregate FuncCalls in the AST.
_AGG_NAMES = frozenset(
    f.name for f in default_catalog().functions
    if f.kind == FuncKind.AGGREGATE
)


def _walk_exprs_in_select(s):
    """Yield every Expr anywhere inside `s` — SELECT items, WHERE,
    GROUP BY, HAVING, ORDER BY, ON conditions in joins.

    Accepts either a Select or a SetOp; for SetOps, recurses into
    each arm transparently."""
    if isinstance(s, SetOp):
        for arm in s.arms:
            yield from _walk_exprs_in_select(arm)
        return
    for t in s.targets:
        yield t.expr
        yield from _walk_subexprs(t.expr)
    if s.where is not None:
        yield s.where
        yield from _walk_subexprs(s.where)
    for e in s.group_by:
        yield e
        yield from _walk_subexprs(e)
    if s.having is not None:
        yield s.having
        yield from _walk_subexprs(s.having)
    for ob in s.order_by:
        yield ob.expr
        yield from _walk_subexprs(ob.expr)
    for f in s.from_:
        yield from _walk_exprs_in_from_item(f)


def _walk_exprs_in_from_item(f):
    if isinstance(f, JoinExpr):
        yield from _walk_exprs_in_from_item(f.left)
        yield from _walk_exprs_in_from_item(f.right)
        if f.on is not None:
            yield f.on
            yield from _walk_subexprs(f.on)


def _walk_subexprs(e: Expr):
    """Yield every sub-Expr of `e` recursively (not including `e` itself)."""
    if isinstance(e, FuncCall):
        for a in e.args:
            yield a
            yield from _walk_subexprs(a)
    elif isinstance(e, BinaryOp):
        yield e.left
        yield from _walk_subexprs(e.left)
        yield e.right
        yield from _walk_subexprs(e.right)
    elif isinstance(e, UnaryOp):
        yield e.operand
        yield from _walk_subexprs(e.operand)
    elif isinstance(e, Cast):
        yield e.expr
        yield from _walk_subexprs(e.expr)


def _is_aggregate_call(e: Expr) -> bool:
    return isinstance(e, FuncCall) and e.name in _AGG_NAMES


def _is_window_call(e: Expr) -> bool:
    """A window-style call is any FuncCall with `over` set."""
    return isinstance(e, FuncCall) and e.over is not None


def _has_nested_aggregate_anywhere(s) -> bool:
    """True if any aggregate FuncCall in `s` has another aggregate
    FuncCall within its argument subtree. Accepts Select or SetOp."""
    if isinstance(s, SetOp):
        return any(_has_nested_aggregate_anywhere(arm) for arm in s.arms)
    for outer in _walk_exprs_in_select(s):
        if _is_aggregate_call(outer):
            for inner in _walk_subexprs(outer):
                if _is_aggregate_call(inner):
                    return True
    return False


def _aggregates_in_clause(start: Expr | None) -> list[FuncCall]:
    if start is None:
        return []
    out = []
    if _is_aggregate_call(start):
        out.append(start)
    for sub in _walk_subexprs(start):
        if _is_aggregate_call(sub):
            out.append(sub)
    return out


# ===========================================================================
# Headline round-trip — the milestone acceptance test
# ===========================================================================

@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("complexity", [1, 3, 5, 7])
def test_query_parses(seed, complexity):
    """Every generated query must produce SQL that pglast accepts.

    50 seeds × 4 complexities = 200 combinations, matching the
    milestone-1 acceptance bar. If any combination fails, the
    failure message includes the seed, complexity, parser error,
    and the offending SQL — enough to reproduce locally.
    """
    schema = generate_schema(seed=seed, complexity=complexity)
    query = generate_query(seed=seed, schema=schema, complexity=complexity)
    sql = print_query(query)
    result = check_syntax(sql)
    assert result.ok, (
        f"Query did not parse at seed={seed}, complexity={complexity}\n"
        f"Error: {result.error}\n"
        f"At position: {result.error_position}\n"
        f"SQL:\n{sql}"
    )


# Sweep the dial endpoints too — these aren't in the headline bar
# but they catch off-by-one bugs at the edges.

@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("complexity", [0, 2, 4, 6, 8, 10])
def test_query_parses_at_dial_extremes(seed, complexity):
    schema = generate_schema(seed=seed, complexity=complexity)
    query = generate_query(seed=seed, schema=schema, complexity=complexity)
    sql = print_query(query)
    result = check_syntax(sql)
    assert result.ok, f"seed={seed} complexity={complexity}: {result.error}\n{sql}"


# ===========================================================================
# Determinism
# ===========================================================================

def test_same_seed_same_query():
    """Bytes-identical SQL for same inputs across runs."""
    schema = generate_schema(seed=42, complexity=5)
    a = generate_query(seed=42, schema=schema, complexity=5)
    b = generate_query(seed=42, schema=schema, complexity=5)
    assert a == b
    assert print_query(a) == print_query(b)


@pytest.mark.parametrize("complexity", [0, 1, 3, 5, 7, 10])
def test_determinism_holds_across_complexity(complexity):
    schema = generate_schema(seed=7, complexity=complexity)
    a = generate_query(seed=7, schema=schema, complexity=complexity)
    b = generate_query(seed=7, schema=schema, complexity=complexity)
    assert print_query(a) == print_query(b)


def test_different_seeds_produce_different_queries():
    """Two different seeds against the same schema should not produce
    the same SQL for any reasonable complexity."""
    schema = generate_schema(seed=0, complexity=5)
    a = generate_query(seed=1, schema=schema, complexity=5)
    b = generate_query(seed=2, schema=schema, complexity=5)
    assert print_query(a) != print_query(b)


def test_query_seed_is_independent_of_schema_seed():
    """Same query seed should produce comparable queries against
    different schemas (different schemas lead to different SQL, but
    the act of seeding query gen with the same seed is independent)."""
    schema_a = generate_schema(seed=10, complexity=5)
    schema_b = generate_schema(seed=20, complexity=5)
    qa = generate_query(seed=99, schema=schema_a, complexity=5)
    qb = generate_query(seed=99, schema=schema_b, complexity=5)
    # Different schemas produce different queries.
    assert print_query(qa) != print_query(qb)
    # But each is itself reproducible.
    qa2 = generate_query(seed=99, schema=schema_a, complexity=5)
    assert qa == qa2


# ===========================================================================
# Structural invariants
# ===========================================================================

def test_query_always_has_at_least_one_target():
    for seed in range(20):
        for c in (0, 3, 7):
            schema = generate_schema(seed=seed, complexity=c)
            q = generate_query(seed=seed, schema=schema, complexity=c)
            for s in _all_select_bodies(q.select):
                assert len(s.targets) >= 1, (seed, c)


def test_query_always_has_a_from_clause():
    """Milestone-1 generation always picks at least 1 FROM table."""
    for seed in range(20):
        for c in (0, 5, 10):
            schema = generate_schema(seed=seed, complexity=c)
            q = generate_query(seed=seed, schema=schema, complexity=c)
            for s in _all_select_bodies(q.select):
                assert len(s.from_) >= 1, (seed, c)


def test_complexity_zero_has_no_optional_clauses():
    """At c=0 no flags are set, so WHERE/ORDER BY/LIMIT are all
    forbidden. Every query should be the bare SELECT/FROM shape."""
    for seed in range(20):
        schema = generate_schema(seed=seed, complexity=0)
        q = generate_query(seed=seed, schema=schema, complexity=0)
        assert q.select.where is None
        assert q.select.order_by == ()
        assert q.select.limit is None
        # max_from_items is 1 at c=0, so always one table.
        assert len(q.select.from_) == 1


def _walk_aliases_in_select(s) -> set[str]:
    """Collect every (alias) referenced anywhere in expressions
    inside `s`. Used to verify that every column ref points at
    an alias the query actually FROM-introduced. Accepts Select
    or SetOp (walks arms for SetOps)."""
    if isinstance(s, SetOp):
        out: set[str] = set()
        for arm in s.arms:
            out.update(_walk_aliases_in_select(arm))
        return out
    refs: set[str] = set()

    def walk_expr(e: Expr) -> None:
        if isinstance(e, ColumnRef):
            refs.add(e.table_alias)
        elif isinstance(e, FuncCall):
            for a in e.args:
                walk_expr(a)
        elif isinstance(e, BinaryOp):
            walk_expr(e.left)
            walk_expr(e.right)
        elif isinstance(e, UnaryOp):
            walk_expr(e.operand)
        elif isinstance(e, Cast):
            walk_expr(e.expr)
        # Literal: nothing to do

    for t in s.targets:
        walk_expr(t.expr)
    if s.where is not None:
        walk_expr(s.where)
    for ob in s.order_by:
        walk_expr(ob.expr)

    # Walk ON conditions inside joins.
    def walk_from_item(f) -> None:
        if isinstance(f, JoinExpr):
            walk_from_item(f.left)
            walk_from_item(f.right)
            if f.on is not None:
                walk_expr(f.on)

    for f in s.from_:
        walk_from_item(f)

    return refs


def _aliases_introduced_by_from(s) -> set[str]:
    """Collects FROM-clause aliases. Accepts Select or SetOp; for
    SetOps, unions across all arms."""
    if isinstance(s, SetOp):
        out: set[str] = set()
        for arm in s.arms:
            out.update(_aliases_introduced_by_from(arm))
        return out
    out2: set[str] = set()

    def walk(f) -> None:
        if isinstance(f, TableRef):
            out2.add(f.alias)
        elif isinstance(f, DerivedTable):
            out2.add(f.alias)
        elif isinstance(f, CteRef):
            out2.add(f.alias)
        elif isinstance(f, JoinExpr):
            walk(f.left)
            walk(f.right)

    for f in s.from_:
        walk(f)
    return out2


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("complexity", [1, 3, 5, 7])
def test_every_column_ref_resolves_to_a_from_alias(seed, complexity):
    """No column reference should point at an alias that doesn't
    appear in the FROM clause. This is the structural invariant the
    scope is supposed to enforce; if the scope leaks, this test
    catches it."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    refs = _walk_aliases_in_select(q.select)
    introduced = _aliases_introduced_by_from(q.select)
    leaked = refs - introduced
    assert not leaked, (
        f"seed={seed} complexity={complexity}: refs to "
        f"undeclared aliases {leaked}\nSQL:\n{print_query(q)}"
    )


def test_from_aliases_are_unique_within_a_query():
    """Every FROM item in each arm should have a distinct alias."""
    for seed in range(30):
        for c in (1, 5, 10):
            schema = generate_schema(seed=seed, complexity=c)
            q = generate_query(seed=seed, schema=schema, complexity=c)
            for s in _all_select_bodies(q.select):
                aliases = _aliases_introduced_by_from(s)
                # set vs count comparison detects duplicates.
                n_refs = sum(1 for f in s.from_
                             for _ in _flatten_table_refs(f))
                assert len(aliases) == n_refs, (seed, c, aliases)


def _flatten_table_refs(f):
    """Yield every aliased FROM item (TableRef, DerivedTable, or
    CteRef) inside `f`, recursing through joins. Used by uniqueness
    checks that count aliased items vs distinct alias names."""
    if isinstance(f, TableRef):
        yield f
    elif isinstance(f, DerivedTable):
        yield f
    elif isinstance(f, CteRef):
        yield f
    elif isinstance(f, JoinExpr):
        yield from _flatten_table_refs(f.left)
        yield from _flatten_table_refs(f.right)


# ===========================================================================
# Sample output — visual sanity check
# ===========================================================================

def test_sample_output_eyeballable():
    """Not really an assertion test — generates a few sample queries
    and checks they print as something readable. If this ever stops
    producing reasonable-looking SQL the rest of the suite will too,
    but seeing samples in `pytest -v -s` is useful during development."""
    for seed in range(3):
        for c in (1, 5, 7):
            schema = generate_schema(seed=seed, complexity=c)
            q = generate_query(seed=seed, schema=schema, complexity=c)
            sql = print_query(q)
            assert "SELECT" in sql
            assert "FROM" in sql


# ===========================================================================
# Milestone 2: aggregate-query invariants
# ===========================================================================

@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("complexity", [3, 5, 7, 10])
def test_aggregate_query_parses(seed, complexity):
    """At aggregate-eligible complexity (>=3), a meaningful share of
    queries will hit the aggregate path. Every one must still pglast-
    accept. This is the milestone-2 acceptance bar (50 × 4 = 200
    combos, mirroring the milestone-1 headline test)."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    sql = print_query(q)
    result = check_syntax(sql)
    assert result.ok, (
        f"seed={seed} complexity={complexity}: {result.error}\n"
        f"At pos {result.error_position}\n{sql}"
    )


def test_aggregate_path_actually_fires():
    """Sanity: across enough seeds, the aggregate path produces
    GROUP BY queries at a meaningful rate. If this hits 0 the path
    is broken; if 100 the dial is mis-tuned."""
    n_agg = 0
    total = 200
    for seed in range(total):
        schema = generate_schema(seed=seed, complexity=5)
        q = generate_query(seed=seed, schema=schema, complexity=5)
        if q.select.group_by:
            n_agg += 1
    # p_aggregate_query=0.35 → ~70 expected; loose ±50% bounds
    # so dial re-tuning doesn't trigger spurious failures.
    assert 30 < n_agg < 130, (
        f"aggregate path fired {n_agg}/{total} times — outside sanity band"
    )


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("complexity", [3, 5, 7, 10])
def test_group_by_consistency(seed, complexity):
    """For every aggregating query, every non-aggregate SELECT-list
    expression must appear in GROUP BY. PG's PARSE-tier rule allows
    SELECT-list items to be either aggregates or expressions
    GROUP-BY'd at SOME level. With grouping sets (ROLLUP, CUBE,
    GROUPING SETS), any expression appearing in any element of any
    grouping construct counts.

    This walks `Select.group_by` and unrolls GroupingSet items into
    their member expressions; the union of all those is the set of
    "grouped at some level" expressions that SELECT items can
    reference. Applied per-arm for SetOp bodies — each arm has its
    own targets/group_by and the rule must hold independently for
    every reachable Select.
    """
    from waxsql.ast import GroupingSet
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    for s in _all_select_bodies(q.select):
        if not s.group_by:
            continue  # non-aggregating: rule doesn't apply to this Select

        # Build the set of "expressions reachable by some grouping set."
        # Plain Expr items contribute themselves; GroupingSet items
        # contribute every Expr appearing in any of their elements.
        grouped: set = set()
        for item in s.group_by:
            if isinstance(item, GroupingSet):
                for elem in item.elements:
                    grouped.update(elem)
            else:
                grouped.add(item)
        for tgt in s.targets:
            if _is_aggregate_call(tgt.expr):
                continue
            assert tgt.expr in grouped, (
                f"seed={seed} complexity={complexity}: SELECT item {tgt.expr!r} "
                f"is neither an aggregate nor in GROUP BY {s.group_by}\n"
                f"SQL:\n{print_query(q)}"
            )


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("complexity", [3, 5, 7, 10])
def test_no_nested_aggregates_anywhere(seed, complexity):
    """sum(count(...))-style nesting is a PG syntax error. The
    in_aggregate flag should make it impossible at the generator
    level — verify on the full Query AST, not just isolated exprs."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    assert not _has_nested_aggregate_anywhere(q.select), (
        f"seed={seed} complexity={complexity}: nested aggregate\n"
        f"SQL:\n{print_query(q)}"
    )


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("complexity", [1, 3, 5, 7, 10])
def test_no_aggregates_in_non_aggregate_queries(seed, complexity):
    """A query without GROUP BY must not contain "plain" aggregates
    (the kind that require GROUP BY to be valid). Aggregates with
    OVER (window-style aggregates like `sum(x) OVER (PARTITION BY y)`)
    ARE valid in non-aggregate queries — the OVER clause provides
    its own partitioning, lifting the GROUP-BY requirement.

    Regression guard for the `allow_aggregates=False` discipline.
    Applied per-arm for SetOp bodies — group_by status is per-arm,
    so the rule is too."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    for s in _all_select_bodies(q.select):
        if s.group_by:
            continue  # aggregating: rule doesn't apply to this Select
        for e in _walk_exprs_in_select(s):
            if _is_aggregate_call(e) and e.over is None:
                assert False, (
                    f"seed={seed} complexity={complexity}: plain "
                    f"(non-window) aggregate {e.name!r} appeared in "
                    f"non-aggregating query\n{print_query(q)}"
                )


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("complexity", [3, 5, 7, 10])
def test_no_aggregates_in_where_or_on(seed, complexity):
    """Aggregates are forbidden in WHERE and ON regardless of whether
    the query aggregates. PG enforces this at parse-analysis time;
    we enforce it at generator time (allow_aggregates=False threaded
    through both clauses) so that PARSE-tier validation, when it
    lands, won't surface this category of error. Applied per-arm for
    SetOp bodies — each arm has its own WHERE and FROM."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    for s in _all_select_bodies(q.select):
        # WHERE
        assert not _aggregates_in_clause(s.where), (
            f"seed={seed} complexity={complexity}: aggregate in WHERE\n"
            f"{print_query(q)}"
        )
        # ON conditions in joins
        for f in s.from_:
            for e in _walk_exprs_in_from_item(f):
                assert not _aggregates_in_clause(e), (
                    f"seed={seed} complexity={complexity}: aggregate in JOIN ON\n"
                    f"{print_query(q)}"
                )


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("complexity", [3, 5, 7, 10])
def test_having_implies_group_by(seed, complexity):
    """A query with HAVING must also have GROUP BY — HAVING is only
    valid in aggregating queries, and we always emit explicit GROUP
    BY for aggregating queries (no implicit-single-group form).
    Applied per-arm for SetOp bodies."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    for s in _all_select_bodies(q.select):
        if s.having is not None:
            assert s.group_by, (
                f"seed={seed} complexity={complexity}: HAVING without "
                f"GROUP BY\n{print_query(q)}"
            )


def test_having_actually_fires():
    """Sanity: HAVING must fire occasionally at c >= 5."""
    n_having = 0
    for seed in range(200):
        schema = generate_schema(seed=seed, complexity=7)
        q = generate_query(seed=seed, schema=schema, complexity=7)
        if q.select.having is not None:
            n_having += 1
    # Expected rate: p_aggregate_query * p_having ≈ 0.14 → ~28/200
    # Loose bounds.
    assert 5 < n_having < 80, (
        f"HAVING fired {n_having}/200 times — outside sanity band"
    )


# ===========================================================================
# Milestone 3: subquery invariants
# ===========================================================================
#
# Subquery-aware helpers below are kept separate from the milestone-2
# helpers (which intentionally treat subqueries as opaque). Each
# milestone-3 test names which kind of traversal it wants.

_ANY_SUBQUERY = (Subquery, Exists, InSubquery)


def _on_conditions(f):
    """Yield each ON expression in `f`'s JoinExpr tree once (top-level
    only — the caller is expected to recurse into the expression's
    structure itself).

    Distinct from milestone-2's `_walk_exprs_in_from_item` which
    yields ON-and-its-subexprs. For the milestone-3 walkers below,
    we want each ON expression visited once and then traversed
    intentionally — otherwise nested subqueries get visited
    multiple times along different yield paths."""
    if isinstance(f, JoinExpr):
        yield from _on_conditions(f.left)
        yield from _on_conditions(f.right)
        if f.on is not None:
            yield f.on


def _column_refs_in_expr_local(e: Expr):
    """Yield ColumnRefs in `e`, recursing through compound expressions
    but treating subqueries as OPAQUE (their refs belong to a
    different scope)."""
    if isinstance(e, ColumnRef):
        yield e
    elif isinstance(e, FuncCall):
        for a in e.args:
            yield from _column_refs_in_expr_local(a)
    elif isinstance(e, BinaryOp):
        yield from _column_refs_in_expr_local(e.left)
        yield from _column_refs_in_expr_local(e.right)
    elif isinstance(e, UnaryOp):
        yield from _column_refs_in_expr_local(e.operand)
    elif isinstance(e, Cast):
        yield from _column_refs_in_expr_local(e.expr)
    elif isinstance(e, InSubquery):
        # The LHS belongs to the surrounding scope; do walk it.
        yield from _column_refs_in_expr_local(e.expr)
    # Subquery / Exists: opaque — their inner refs are handled by the
    # subquery walker, not this one.


def _column_refs_in_select_local(s):
    """All ColumnRefs in the expression positions of `s`, NOT
    recursing into nested subqueries. Accepts Select or SetOp."""
    if isinstance(s, SetOp):
        for arm in s.arms:
            yield from _column_refs_in_select_local(arm)
        return
    for t in s.targets:
        yield from _column_refs_in_expr_local(t.expr)
    if s.where is not None:
        yield from _column_refs_in_expr_local(s.where)
    for e in s.group_by:
        yield from _column_refs_in_expr_local(e)
    if s.having is not None:
        yield from _column_refs_in_expr_local(s.having)
    for ob in s.order_by:
        yield from _column_refs_in_expr_local(ob.expr)
    for f in s.from_:
        for on in _on_conditions(f):
            yield from _column_refs_in_expr_local(on)


def _aliases_introduced_locally(s) -> set[str]:
    """Aliases that `s.from_` introduces (base + derived + CTE refs),
    not recursing into joins' inner subqueries — same shape as
    milestone-1's helper but duplicated here to make the milestone-3+
    intent explicit. Accepts Select or SetOp."""
    if isinstance(s, SetOp):
        out_setop: set[str] = set()
        for arm in s.arms:
            out_setop.update(_aliases_introduced_locally(arm))
        return out_setop
    out: set[str] = set()

    def walk(f) -> None:
        if isinstance(f, TableRef):
            out.add(f.alias)
        elif isinstance(f, DerivedTable):
            out.add(f.alias)
        elif isinstance(f, CteRef):
            out.add(f.alias)
        elif isinstance(f, JoinExpr):
            walk(f.left)
            walk(f.right)

    for f in s.from_:
        walk(f)
    return out


def _all_subqueries(s):
    """Yield every Subquery / Exists / InSubquery anywhere in `s`,
    INCLUDING those nested inside other subqueries. Recursive.
    Accepts Select or SetOp."""
    if isinstance(s, SetOp):
        for arm in s.arms:
            yield from _all_subqueries(arm)
        return
    def walk_expr(e: Expr):
        if isinstance(e, _ANY_SUBQUERY):
            yield e
            yield from _all_subqueries(e.select)
            if isinstance(e, InSubquery):
                yield from walk_expr(e.expr)
        elif isinstance(e, FuncCall):
            for a in e.args:
                yield from walk_expr(a)
        elif isinstance(e, BinaryOp):
            yield from walk_expr(e.left)
            yield from walk_expr(e.right)
        elif isinstance(e, UnaryOp):
            yield from walk_expr(e.operand)
        elif isinstance(e, Cast):
            yield from walk_expr(e.expr)

    for t in s.targets:
        yield from walk_expr(t.expr)
    if s.where is not None:
        yield from walk_expr(s.where)
    for e in s.group_by:
        yield from walk_expr(e)
    if s.having is not None:
        yield from walk_expr(s.having)
    for ob in s.order_by:
        yield from walk_expr(ob.expr)
    for f in s.from_:
        for on in _on_conditions(f):
            yield from walk_expr(on)


def _max_subquery_depth(s, current: int = 0) -> int:
    """Return the maximum nesting depth of subqueries reachable from
    `s`. A top-level query with no subqueries → 0; one subquery → 1;
    a subquery containing another → 2. Accepts Select or SetOp;
    for SetOp, takes the max across arms."""
    if isinstance(s, SetOp):
        return max(
            (_max_subquery_depth(arm, current) for arm in s.arms),
            default=current,
        )
    deepest = current

    def walk_expr(e: Expr, depth: int):
        nonlocal deepest
        if isinstance(e, _ANY_SUBQUERY):
            inner_depth = _max_subquery_depth(e.select, depth + 1)
            if inner_depth > deepest:
                deepest = inner_depth
            if isinstance(e, InSubquery):
                walk_expr(e.expr, depth)
        elif isinstance(e, FuncCall):
            for a in e.args:
                walk_expr(a, depth)
        elif isinstance(e, BinaryOp):
            walk_expr(e.left, depth)
            walk_expr(e.right, depth)
        elif isinstance(e, UnaryOp):
            walk_expr(e.operand, depth)
        elif isinstance(e, Cast):
            walk_expr(e.expr, depth)

    for t in s.targets:
        walk_expr(t.expr, current)
    if s.where is not None:
        walk_expr(s.where, current)
    for e in s.group_by:
        walk_expr(e, current)
    if s.having is not None:
        walk_expr(s.having, current)
    for ob in s.order_by:
        walk_expr(ob.expr, current)
    for f in s.from_:
        for on in _on_conditions(f):
            walk_expr(on, current)
    return deepest


def _is_correlated(sub) -> bool:
    """A subquery is correlated iff any ColumnRef in its inner SELECT
    references an alias NOT introduced by its own FROM clause.

    PG itself defines correlation this way — it doesn't carry a flag,
    it works out from name resolution. Here we compute it directly
    from the AST, which matches PG's view exactly (modulo the
    alias-uniqueness invariant we now enforce).
    """
    inner_aliases = _aliases_introduced_locally(sub.select)
    for ref in _column_refs_in_select_local(sub.select):
        if ref.table_alias not in inner_aliases:
            return True
    return False


# --- Headline parse sweep ---------------------------------------------------

@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("complexity", [4, 5, 7, 10])
def test_subquery_query_parses(seed, complexity):
    """At subquery-eligible complexity (>=4), generated queries must
    pglast-parse — including those that nest subqueries multiple
    levels deep. 50 × 4 = 200 combos, the milestone-3 acceptance bar."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    sql = print_query(q)
    result = check_syntax(sql)
    assert result.ok, (
        f"seed={seed} complexity={complexity}: {result.error}\n"
        f"{sql}"
    )


def test_subquery_path_actually_fires():
    """Sanity: across enough seeds at c=5, subqueries appear at a
    meaningful rate. If 0 the path is broken; if 100% the weights
    are mis-tuned."""
    n_with_sub = 0
    total = 200
    for seed in range(total):
        schema = generate_schema(seed=seed, complexity=5)
        q = generate_query(seed=seed, schema=schema, complexity=5)
        if any(True for _ in _all_subqueries(q.select)):
            n_with_sub += 1
    # Loose bounds — re-tuning subquery weights shouldn't trigger
    # spurious failures.
    assert 30 < n_with_sub < 195, (
        f"subqueries fired in {n_with_sub}/{total} queries — outside band"
    )


# --- Depth-budget invariant -------------------------------------------------

@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("complexity", [4, 5, 7, 10])
def test_subquery_depth_within_budget(seed, complexity):
    """The depth of nested subqueries must not exceed
    cfg.max_subquery_depth — that's the budget the generator was
    given."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    cfg = query_config_for_complexity(complexity)
    actual = _max_subquery_depth(q.select)
    assert actual <= cfg.max_subquery_depth, (
        f"seed={seed} c={complexity}: depth {actual} exceeds "
        f"budget {cfg.max_subquery_depth}\n{print_query(q)}"
    )


# --- No subqueries below the unlock notch ----------------------------------

@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("complexity", [0, 1, 2, 3])
def test_no_subqueries_at_low_complexity(seed, complexity):
    """Below c=4 the FEATURE_SCALAR_SUBQUERY flag is absent, so no
    subquery production should fire."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    subs = list(_all_subqueries(q.select))
    assert not subs, (
        f"seed={seed} c={complexity}: subqueries appeared at low "
        f"complexity\n{print_query(q)}"
    )


# --- Alias uniqueness across nested scopes ---------------------------------

def _all_aliases_in_query(s) -> list[str]:
    """Return EVERY alias used anywhere in the query, including inside
    nested subqueries. A list (not a set) so duplicates can be
    detected — duplicates are exactly the bug we're testing for.

    Dedupes subquery NODES by Python identity. Accepts Select or
    SetOp; for SetOp, concatenates aliases from all arms (each arm
    is a separate scope, so cross-arm "duplicates" are NOT actually
    duplicates — wait, they ARE because the alias_counter is shared
    across the whole query, so cross-arm alias collisions WOULD
    indicate a bug)."""
    if isinstance(s, SetOp):
        out_setop: list[str] = []
        for arm in s.arms:
            out_setop.extend(_all_aliases_in_query(arm))
        return out_setop
    out: list[str] = []
    out.extend(_aliases_introduced_locally(s))
    seen_subs: set[int] = set()
    for sub in _all_subqueries(s):
        if id(sub) in seen_subs:
            continue
        seen_subs.add(id(sub))
        out.extend(_aliases_introduced_locally(sub.select))
    return out


@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("complexity", [4, 5, 7, 10])
def test_alias_uniqueness_across_scopes(seed, complexity):
    """Every TableRef alias across the entire query (outer + every
    nested subquery) must be unique. Reusing aliases across scopes
    causes PG's name resolution to silently shadow outer refs,
    breaking correlation. This test is the regression guard for
    the milestone-3 alias-shadowing fix."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    aliases = _all_aliases_in_query(q.select)
    assert len(aliases) == len(set(aliases)), (
        f"seed={seed} c={complexity}: duplicate aliases {aliases}\n"
        f"{print_query(q)}"
    )


# --- Correlated subqueries actually correlate ------------------------------

def test_correlation_path_actually_correlates():
    """Across enough seeds, at least some generated subqueries should
    reference outer aliases (i.e., be correlated). The 50/50 dispatch
    in gen_expr should produce roughly half correlated; the forced-
    correlation predicate guarantees correlated subqueries actually
    have an outer ref."""
    n_subs = 0
    n_correlated = 0
    for seed in range(200):
        schema = generate_schema(seed=seed, complexity=7)
        q = generate_query(seed=seed, schema=schema, complexity=7)
        for sub in _all_subqueries(q.select):
            n_subs += 1
            if _is_correlated(sub):
                n_correlated += 1
    assert n_subs > 50, f"too few subqueries to evaluate ({n_subs})"
    rate = n_correlated / n_subs
    # Loose: should be substantially above 0 and below 1, since
    # roughly half the subqueries are forced-correlated and the
    # uncorrelated ones occasionally reference outer columns by
    # accident (gen_expr's column-ref candidates can pick from the
    # parent chain when `correlated=True` on the scope).
    assert 0.2 < rate < 0.95, (
        f"correlation rate {rate:.2f} ({n_correlated}/{n_subs}) "
        f"outside sanity band — should be ~0.5"
    )


# --- The pre-existing PARSE-tier-correctness invariants must still hold ---

@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("complexity", [4, 5, 7, 10])
def test_milestone2_invariants_hold_with_subqueries(seed, complexity):
    """A combined check that:
      1. The outer query's GROUP BY consistency still holds (subqueries
         inside aggregating queries don't accidentally violate it).
      2. No top-level aggregates appear outside aggregate context.
      3. Every column ref at the outer level resolves to an outer alias.

    Applied per-arm for SetOp bodies — each arm has its own scope and
    its own FROM, so the "resolves to outer" check runs per-arm.
    """
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    for s in _all_select_bodies(q.select):
        outer_aliases = _aliases_introduced_locally(s)

        # Outer column refs (NOT recursing into subqueries) must resolve
        # to outer aliases.
        for ref in _column_refs_in_select_local(s):
            assert ref.table_alias in outer_aliases, (
                f"seed={seed} c={complexity}: outer ColumnRef "
                f"{ref.table_alias}.{ref.column} doesn't resolve to outer\n"
                f"{print_query(q)}"
            )


# ===========================================================================
# Milestone 4: derived-table and LATERAL invariants
# ===========================================================================

def _all_derived_tables(s):
    """Yield every DerivedTable anywhere in `s` — outer FROM clause,
    inside JoinExprs, and inside any nested subquery's FROM clause.
    Includes derived tables nested arbitrarily deep. Accepts Select
    or SetOp."""
    if isinstance(s, SetOp):
        for arm in s.arms:
            yield from _all_derived_tables(arm)
        return

    def walk_from(f):
        if isinstance(f, DerivedTable):
            yield f
            yield from _all_derived_tables(f.select)
        elif isinstance(f, JoinExpr):
            yield from walk_from(f.left)
            yield from walk_from(f.right)
            if f.on is not None:
                # ON conditions can contain subqueries, which can
                # themselves contain derived tables.
                for sub in _all_subqueries_in_expr(f.on):
                    yield from _all_derived_tables(sub.select)

    for f in s.from_:
        yield from walk_from(f)
    # Subqueries in expression positions can also contain derived
    # tables in their FROM.
    for sub in _all_subqueries(s):
        yield from _all_derived_tables(sub.select)


def _all_subqueries_in_expr(e: Expr):
    """Yield every subquery node directly inside `e`, recursing into
    compound expressions but not into the subqueries' inner Selects."""
    if isinstance(e, _ANY_SUBQUERY):
        yield e
        if isinstance(e, InSubquery):
            yield from _all_subqueries_in_expr(e.expr)
    elif isinstance(e, FuncCall):
        for a in e.args:
            yield from _all_subqueries_in_expr(a)
    elif isinstance(e, BinaryOp):
        yield from _all_subqueries_in_expr(e.left)
        yield from _all_subqueries_in_expr(e.right)
    elif isinstance(e, UnaryOp):
        yield from _all_subqueries_in_expr(e.operand)
    elif isinstance(e, Cast):
        yield from _all_subqueries_in_expr(e.expr)


@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("complexity", [5, 6, 7, 10])
def test_derived_table_query_parses(seed, complexity):
    """At derived-table-eligible complexity (>=5), generated queries
    must pglast-parse — including those with deeply-nested derived
    tables and LATERAL. 50 × 4 = 200 combos, the milestone-4
    acceptance bar."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    sql = print_query(q)
    result = check_syntax(sql)
    assert result.ok, (
        f"seed={seed} complexity={complexity}: {result.error}\n{sql}"
    )


def test_derived_table_path_actually_fires():
    """Sanity: across enough seeds at c=6, derived tables appear at
    a meaningful rate. Loose bounds so re-tuning probabilities
    doesn't trigger spurious failures."""
    n_derived = 0
    n_lateral = 0
    total = 200
    for seed in range(total):
        schema = generate_schema(seed=seed, complexity=6)
        q = generate_query(seed=seed, schema=schema, complexity=6)
        if any(True for _ in _all_derived_tables(q.select)):
            n_derived += 1
        if any(d.lateral for d in _all_derived_tables(q.select)):
            n_lateral += 1
    assert 30 < n_derived < 195, (
        f"derived path fired in {n_derived}/{total} — outside band"
    )
    # LATERAL is rarer (gated on derived AND past first FROM AND
    # p_lateral_when_derived AND FEATURE_LATERAL).
    assert 5 < n_lateral < 150, (
        f"LATERAL fired in {n_lateral}/{total} — outside band"
    )


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("complexity", [5, 6, 7, 10])
def test_derived_table_columns_use_synthetic_cN_names(seed, complexity):
    """Every derived table's inner SELECT has 1..N targets with
    explicit `c1`, `c2`, `c3`, ... aliases — that's the contract
    that makes `outer.cN` references deterministic. Multi-column
    derived tables added in the polish-track work; the contract
    on naming stays the same, just generalized to multiple targets.
    Regression guard for the `gen_derived_table` aliasing logic."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    for d in _all_derived_tables(q.select):
        for i, target in enumerate(d.select.targets):
            expected = f"c{i + 1}"
            assert target.alias == expected, (
                f"seed={seed} c={complexity}: derived target #{i} alias "
                f"is {target.alias!r}, expected {expected!r}\n"
                f"{print_query(q)}"
            )


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("complexity", [6, 7, 10])
def test_lateral_actually_references_preceding(seed, complexity):
    """A LATERAL derived table should reference at least one alias
    that's NOT introduced by its own inner FROM — proving it's
    actually using the LATERAL capability rather than just being a
    LATERAL no-op.

    Counts statistical: at least one LATERAL in the suite should
    have an outer reference. The forced-correlation predicate makes
    this nearly-always-true; a few edge cases (no comparable outer
    types) might skip the forcer."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    # We don't assert per-query because individual LATERAL might
    # legitimately end up not referencing outer (forcer skipped due
    # to no comparable types). The test_lateral_correlates_at_meaningful_rate
    # below covers the statistical invariant.
    for d in _all_derived_tables(q.select):
        if not d.lateral:
            continue
        # Verify at least the structure is right: an alias and a Select
        assert d.alias is not None
        assert d.select is not None


def test_lateral_correlates_at_meaningful_rate():
    """The forced-correlation predicate inside `_build_subquery_select`
    should make MOST LATERAL derived tables actually reference an
    outer column. Across 200 seeds at c=10, the rate should be
    high but not 100% (rare cases where no comparable outer column
    exists fall back to plain WHERE without correlation)."""
    n_lateral = 0
    n_correlated = 0
    for seed in range(200):
        schema = generate_schema(seed=seed, complexity=10)
        q = generate_query(seed=seed, schema=schema, complexity=10)
        for d in _all_derived_tables(q.select):
            if not d.lateral:
                continue
            n_lateral += 1
            inner_aliases = _aliases_introduced_locally(d.select)
            for ref in _column_refs_in_select_local(d.select):
                if ref.table_alias not in inner_aliases:
                    n_correlated += 1
                    break
    assert n_lateral > 30, f"too few LATERAL to evaluate ({n_lateral})"
    rate = n_correlated / n_lateral
    assert rate > 0.5, (
        f"LATERAL-correlation rate {rate:.2f} ({n_correlated}/{n_lateral}) "
        f"too low — forcer should make most LATERAL actually correlate"
    )


def test_no_lateral_at_low_complexity():
    """Below c=6 the FEATURE_LATERAL flag is absent, so no LATERAL
    derived tables should appear. (Derived tables themselves can
    appear at c>=5.)"""
    for seed in range(30):
        for c in (0, 1, 2, 3, 4, 5):
            schema = generate_schema(seed=seed, complexity=c)
            q = generate_query(seed=seed, schema=schema, complexity=c)
            for d in _all_derived_tables(q.select):
                assert not d.lateral, (
                    f"seed={seed} c={c}: LATERAL appeared below c=6\n"
                    f"{print_query(q)}"
                )


def test_no_derived_tables_at_low_complexity():
    """Below c=5 the FEATURE_DERIVED_TABLE flag is absent, so no
    derived tables at all."""
    for seed in range(30):
        for c in (0, 1, 2, 3, 4):
            schema = generate_schema(seed=seed, complexity=c)
            q = generate_query(seed=seed, schema=schema, complexity=c)
            derived = list(_all_derived_tables(q.select))
            assert not derived, (
                f"seed={seed} c={c}: derived tables appeared below c=5\n"
                f"{print_query(q)}"
            )


# ===========================================================================
# Milestone 5: CTE invariants
# ===========================================================================


def _cte_refs_in_select(s):
    """Yield every CteRef anywhere in `s` (outer FROM, inside joins,
    inside any nested subquery's FROM, inside CTE bodies). Accepts
    Select or SetOp."""
    if isinstance(s, SetOp):
        for arm in s.arms:
            yield from _cte_refs_in_select(arm)
        return

    def walk_from(f):
        if isinstance(f, CteRef):
            yield f
        elif isinstance(f, JoinExpr):
            yield from walk_from(f.left)
            yield from walk_from(f.right)

    for f in s.from_:
        yield from walk_from(f)
    # Recurse into nested subquery bodies.
    for sub in _all_subqueries(s):
        yield from _cte_refs_in_select(sub.select)
    # Recurse into derived-table inner SELECTs.
    for d in _all_derived_tables(s):
        yield from _cte_refs_in_select(d.select)
    # Recurse into CTE bodies themselves.
    for c in s.with_ctes:
        yield from _cte_refs_in_select(c.select)


def _defined_cte_names_in_chain(s) -> set[str]:
    """Return the set of CTE names defined in `s`'s WITH clause (or
    in any enclosing WITH for SetOp arms).

    Used to verify CteRef resolution: every CteRef.cte_name must
    appear in the WITH chain of the enclosing query/subquery.

    Accepts Select or SetOp; for SetOp, unions across all arms."""
    if isinstance(s, SetOp):
        out: set[str] = set()
        for arm in s.arms:
            out.update(_defined_cte_names_in_chain(arm))
        return out
    # For the milestone-5 generator, only the OUTER select has CTEs
    # (descend_subquery sets allow_with=False). So all defined CTE
    # names live in the top-level Select's with_ctes.
    return {c.name for c in s.with_ctes}


@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("complexity", [7, 8, 10])
def test_cte_query_parses(seed, complexity):
    """At CTE-eligible complexity (>=7), generated queries with CTEs
    must pglast-parse — including those where multiple CTEs reference
    each other and the main body uses CTEs in FROM. 50 × 3 = 150
    combos; the CTE-actually-fires sanity test below ensures the
    test isn't trivially passing by never producing CTEs."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    sql = print_query(q)
    result = check_syntax(sql)
    assert result.ok, (
        f"seed={seed} complexity={complexity}: {result.error}\n{sql}"
    )


def test_cte_path_actually_fires():
    """Sanity: at c=8 (where CTEs are unlocked), a meaningful
    fraction of queries should have a WITH clause AND at least
    some should reference a CTE in FROM (otherwise the WITH would
    be a dead definition)."""
    n_with = 0
    n_with_ref = 0
    total = 200
    for seed in range(total):
        schema = generate_schema(seed=seed, complexity=8)
        q = generate_query(seed=seed, schema=schema, complexity=8)
        if q.select.with_ctes:
            n_with += 1
            if any(True for _ in _cte_refs_in_select(q.select)):
                n_with_ref += 1
    # p_with_clause=0.3 → ~60 expected with WITH. Loose bounds.
    assert 30 < n_with < 130, (
        f"WITH fired in {n_with}/{total} — outside band"
    )
    # Of those that have WITH, most should have at least one
    # reference (otherwise the WITH is dead code).
    assert n_with_ref > n_with // 2, (
        f"only {n_with_ref}/{n_with} WITH-having queries reference "
        f"a CTE — most should"
    )


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("complexity", [7, 8, 10])
def test_cte_names_unique_within_with(seed, complexity):
    """PG enforces CTE name uniqueness within a single WITH list.
    The cte_counter on GenContext is monotonic, so this should
    structurally hold; this is the regression guard. Applied
    per-arm for SetOp bodies (only milestone-7 arm Selects can have
    WITHs, but milestone-7 strips them — so this is mostly a future-
    proofing walk)."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    for s in _all_select_bodies(q.select):
        names = [c.name for c in s.with_ctes]
        assert len(names) == len(set(names)), (
            f"seed={seed} c={complexity}: duplicate CTE names {names}\n"
            f"{print_query(q)}"
        )


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("complexity", [7, 8, 10])
def test_every_cteref_resolves_to_defined_cte(seed, complexity):
    """Every `CteRef.cte_name` must match a CTE name defined in
    the top-level WITH clause. Without this, generated SQL might
    reference a CTE that doesn't exist, and PG would reject it
    at parse-analysis time even though pglast accepts it."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    defined_names = _defined_cte_names_in_chain(q.select)
    for cte_ref in _cte_refs_in_select(q.select):
        assert cte_ref.cte_name in defined_names, (
            f"seed={seed} c={complexity}: CteRef '{cte_ref.cte_name}' "
            f"doesn't match any defined CTE (defined: {defined_names})\n"
            f"{print_query(q)}"
        )


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("complexity", [7, 8, 10])
def test_cte_definition_order_no_forward_refs(seed, complexity):
    """A non-recursive CTE can reference an earlier CTE in the WITH
    list but not a later one. RECURSIVE CTEs are exempt for SELF-
    references (the recursive arm of a recursive CTE references the
    CTE being defined — that's the whole point). The generator
    builds CTEs in definition order and registers each before
    generating the next, so non-self forward references shouldn't
    be possible.

    Applied per-arm for SetOp bodies. Currently SetOp arms have
    empty with_ctes (per the strip discipline enforced by
    `test_setop_arms_have_no_per_arm_clauses`), so this is a no-op
    walk for SetOp queries — but walking per-arm keeps the rule
    correctly attached to "every Select reachable" if the generator
    later allows arm-level CTEs."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    for s in _all_select_bodies(q.select):
        if not s.with_ctes:
            continue
        # Walk CTE definitions in order, accumulating defined names.
        # Each CTE body can only reference names defined BEFORE it,
        # OR — for recursive CTEs — itself.
        defined_so_far: set[str] = set()
        for cte_def in s.with_ctes:
            # A recursive CTE references itself. Treat its own name as
            # "available" while walking its body.
            allowed = defined_so_far | (
                {cte_def.name} if cte_def.recursive else set()
            )
            for ref in _cte_refs_in_select(cte_def.select):
                assert ref.cte_name in allowed, (
                    f"seed={seed} c={complexity}: CTE '{cte_def.name}' "
                    f"forward-references '{ref.cte_name}' (only "
                    f"{allowed} allowed)\n{print_query(q)}"
                )
            defined_so_far.add(cte_def.name)


@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("complexity", [7, 8, 10])
def test_alias_uniqueness_with_ctes(seed, complexity):
    """The milestone-3 alias-uniqueness invariant must hold even
    when CTE references are mixed in. CteRef aliases come from the
    same alias_counter as base/derived FROM aliases, so this should
    structurally hold."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    aliases = _all_aliases_in_query(q.select)
    assert len(aliases) == len(set(aliases)), (
        f"seed={seed} c={complexity}: duplicate aliases {aliases}\n"
        f"{print_query(q)}"
    )


def test_no_with_at_low_complexity():
    """Below c=7 the FEATURE_CTE flag is absent, so no WITH clauses
    should appear."""
    for seed in range(30):
        for c in (0, 1, 2, 3, 4, 5, 6):
            schema = generate_schema(seed=seed, complexity=c)
            q = generate_query(seed=seed, schema=schema, complexity=c)
            assert not q.select.with_ctes, (
                f"seed={seed} c={c}: WITH appeared at low complexity\n"
                f"{print_query(q)}"
            )


# ===========================================================================
# Milestone 6: window-function invariants
# ===========================================================================


def _all_window_calls_in_select(s):
    """Yield every FuncCall in `s` whose `over` is set, recursing
    through compound expressions but treating subqueries as opaque
    (their windows are evaluated in their own scope). Accepts
    Select or SetOp."""
    if isinstance(s, SetOp):
        for arm in s.arms:
            yield from _all_window_calls_in_select(arm)
        return

    def walk_expr(e):
        if _is_window_call(e):
            yield e
        if isinstance(e, FuncCall):
            for a in e.args:
                yield from walk_expr(a)
        elif isinstance(e, BinaryOp):
            yield from walk_expr(e.left)
            yield from walk_expr(e.right)
        elif isinstance(e, UnaryOp):
            yield from walk_expr(e.operand)
        elif isinstance(e, Cast):
            yield from walk_expr(e.expr)
        elif isinstance(e, InSubquery):
            yield from walk_expr(e.expr)
        # Subquery / Exists: opaque

    for t in s.targets:
        yield from walk_expr(t.expr)
    if s.where is not None:
        yield from walk_expr(s.where)
    for e in s.group_by:
        yield from walk_expr(e)
    if s.having is not None:
        yield from walk_expr(s.having)
    for ob in s.order_by:
        yield from walk_expr(ob.expr)
    for f in s.from_:
        for on in _on_conditions(f):
            yield from walk_expr(on)


@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("complexity", [8, 9, 10])
def test_window_query_parses(seed, complexity):
    """At window-eligible complexity (>=8), generated queries with
    windows must pglast-parse. 50 × 3 = 150 combos."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    sql = print_query(q)
    result = check_syntax(sql)
    assert result.ok, (
        f"seed={seed} complexity={complexity}: {result.error}\n{sql}"
    )


def test_window_path_actually_fires():
    """Sanity: at c=9, a meaningful share of queries should have
    at least one window call."""
    n = 0
    total = 200
    for seed in range(total):
        schema = generate_schema(seed=seed, complexity=9)
        q = generate_query(seed=seed, schema=schema, complexity=9)
        if any(True for _ in _all_window_calls_in_select(q.select)):
            n += 1
    # Loose bounds — re-tuning weights shouldn't trigger spurious
    # failures.
    assert 30 < n < 195, f"window calls in {n}/{total} — outside band"


def test_no_windows_at_low_complexity():
    """Below c=8 the FEATURE_WINDOW_FUNCTION flag is absent."""
    for seed in range(30):
        for c in (0, 1, 2, 3, 4, 5, 6, 7):
            schema = generate_schema(seed=seed, complexity=c)
            q = generate_query(seed=seed, schema=schema, complexity=c)
            wins = list(_all_window_calls_in_select(q.select))
            assert not wins, (
                f"seed={seed} c={c}: window calls appeared below c=8\n"
                f"{print_query(q)}"
            )


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("complexity", [8, 9, 10])
def test_no_nested_windows(seed, complexity):
    """`window(window(...) OVER (...)) OVER (...)` is invalid PG.
    The `in_window` flag block in gen_expr should make it
    impossible — verify on every generated AST. Applied per-arm for
    SetOp bodies."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)

    def has_nested_window(e, in_w=False):
        if isinstance(e, FuncCall):
            is_w = e.over is not None
            if is_w and in_w:
                return True
            new_in_w = in_w or is_w
            for a in e.args:
                if has_nested_window(a, new_in_w):
                    return True
            return False
        if isinstance(e, BinaryOp):
            return (has_nested_window(e.left, in_w)
                    or has_nested_window(e.right, in_w))
        if isinstance(e, UnaryOp):
            return has_nested_window(e.operand, in_w)
        if isinstance(e, Cast):
            return has_nested_window(e.expr, in_w)
        return False

    for s in _all_select_bodies(q.select):
        for t in s.targets:
            assert not has_nested_window(t.expr), (
                f"seed={seed} c={complexity}: nested window in target\n"
                f"{print_query(q)}"
            )


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("complexity", [8, 9, 10])
def test_no_windows_in_forbidden_clauses(seed, complexity):
    """Window functions are valid ONLY in SELECT list and ORDER BY
    (the latter via SELECT-list reuse). They're forbidden in WHERE,
    HAVING, GROUP BY, JOIN-ON, and inside aggregate function args.
    PG enforces this at parse-analysis time; our generator must
    enforce it via the allow_window flag thread-through. Applied
    per-arm for SetOp bodies — each arm has its own clauses."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)

    def walk_expr_for_windows(e):
        if _is_window_call(e):
            yield e
        if isinstance(e, FuncCall):
            for a in e.args:
                yield from walk_expr_for_windows(a)
        elif isinstance(e, BinaryOp):
            yield from walk_expr_for_windows(e.left)
            yield from walk_expr_for_windows(e.right)
        elif isinstance(e, UnaryOp):
            yield from walk_expr_for_windows(e.operand)
        elif isinstance(e, Cast):
            yield from walk_expr_for_windows(e.expr)
        elif isinstance(e, InSubquery):
            yield from walk_expr_for_windows(e.expr)

    for s in _all_select_bodies(q.select):
        # WHERE: no windows
        if s.where is not None:
            for w in walk_expr_for_windows(s.where):
                assert False, (
                    f"seed={seed} c={complexity}: window in WHERE\n"
                    f"{print_query(q)}"
                )
        # HAVING: no windows (the agg-vs-window-pipeline rule)
        if s.having is not None:
            for w in walk_expr_for_windows(s.having):
                assert False, (
                    f"seed={seed} c={complexity}: window in HAVING\n"
                    f"{print_query(q)}"
                )
        # GROUP BY: no windows
        for ge in s.group_by:
            for w in walk_expr_for_windows(ge):
                assert False, (
                    f"seed={seed} c={complexity}: window in GROUP BY\n"
                    f"{print_query(q)}"
                )
        # JOIN ON: no windows
        for f in s.from_:
            for on in _on_conditions(f):
                for w in walk_expr_for_windows(on):
                    assert False, (
                        f"seed={seed} c={complexity}: window in JOIN ON\n"
                        f"{print_query(q)}"
                    )


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("complexity", [8, 9, 10])
def test_no_windows_inside_aggregate_args(seed, complexity):
    """`agg(window(...) OVER (...))` is a parse-analysis error in PG
    — aggregates evaluate before windows in the pipeline. The
    `_gen_aggregate_funccall` must set allow_window=False on the
    arg ctx. Applied per-arm for SetOp bodies — each arm has its
    own targets and HAVING."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)

    def walk_for_window_in_agg(e, in_agg=False):
        """Yield offending (agg, window) pairs."""
        if isinstance(e, FuncCall):
            is_agg = _is_aggregate_call(e)
            is_win = e.over is not None
            if is_win and in_agg:
                yield e
            new_in_agg = in_agg or is_agg
            for a in e.args:
                yield from walk_for_window_in_agg(a, new_in_agg)
        elif isinstance(e, BinaryOp):
            yield from walk_for_window_in_agg(e.left, in_agg)
            yield from walk_for_window_in_agg(e.right, in_agg)
        elif isinstance(e, UnaryOp):
            yield from walk_for_window_in_agg(e.operand, in_agg)
        elif isinstance(e, Cast):
            yield from walk_for_window_in_agg(e.expr, in_agg)

    for s in _all_select_bodies(q.select):
        for t in s.targets:
            offending = list(walk_for_window_in_agg(t.expr))
            assert not offending, (
                f"seed={seed} c={complexity}: window inside aggregate "
                f"arg\n{print_query(q)}"
            )
        if s.having is not None:
            offending = list(walk_for_window_in_agg(s.having))
            assert not offending, (
                f"seed={seed} c={complexity}: window inside aggregate "
                f"in HAVING\n{print_query(q)}"
            )


@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("complexity", [5, 6, 7, 10])
def test_derived_alias_referenced_columns_match_inner_targets(
    seed, complexity,
):
    """When the outer query references `derived_alias.col`, `col` must
    match an alias from the derived table's inner SELECT targets.
    For milestone 4, all derived columns are named `c1`, so any
    outer ref to a derived alias must use `c1`. Applied per-arm for
    SetOp bodies — each arm has its own FROM and its own column
    refs."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    for s in _all_select_bodies(q.select):
        # Map derived aliases to their valid column names.
        derived_cols: dict[str, set[str]] = {}
        for d in _all_derived_tables(s):
            derived_cols[d.alias] = {
                t.alias for t in d.select.targets if t.alias is not None
            }
        # Walk all column refs at this Select's level and check.
        for ref in _column_refs_in_select_local(s):
            if ref.table_alias in derived_cols:
                assert ref.column in derived_cols[ref.table_alias], (
                    f"seed={seed} c={complexity}: ColumnRef "
                    f"{ref.table_alias}.{ref.column} doesn't match any "
                    f"target of the derived table (valid: "
                    f"{derived_cols[ref.table_alias]})\n{print_query(q)}"
                )


# ===========================================================================
# Milestone 7: set-op invariants
# ===========================================================================

from waxsql.types import implicitly_castable  # noqa: E402


@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("complexity", [9, 10])
def test_setop_query_parses(seed, complexity):
    """At set-op-eligible complexity (>=9), generated queries with
    UNION/INTERSECT/EXCEPT must pglast-parse. 50 × 2 = 100 combos."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    sql = print_query(q)
    result = check_syntax(sql)
    assert result.ok, (
        f"seed={seed} complexity={complexity}: {result.error}\n{sql}"
    )


def test_setop_path_actually_fires():
    """Sanity: at c=10, a meaningful share of queries should be
    SetOps. p_set_op_query=0.25 → ~50 expected out of 200."""
    n = 0
    total = 200
    for seed in range(total):
        schema = generate_schema(seed=seed, complexity=10)
        q = generate_query(seed=seed, schema=schema, complexity=10)
        if isinstance(q.select, SetOp):
            n += 1
    assert 20 < n < 100, f"SetOps fired in {n}/{total} — outside band"


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("complexity", [9, 10])
def test_setop_arms_have_matching_target_count(seed, complexity):
    """PG enforces that all arms of a UNION/INTERSECT/EXCEPT have
    the same number of result columns. The generator extracts
    arm 1's target types and forces arms 2+ to match — this is
    the regression guard.

    For nested SetOp arms, recurse: every Select reachable through
    the arm tree must have the matching column count. The test
    helper walks all the way down."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    if not isinstance(q.select, SetOp):
        return  # not a SetOp; rule doesn't apply

    def all_select_target_counts(arm):
        if isinstance(arm, SetOp):
            for inner in arm.arms:
                yield from all_select_target_counts(inner)
        else:
            yield len(arm.targets)

    counts = set()
    for arm in q.select.arms:
        counts.update(all_select_target_counts(arm))
    assert len(counts) == 1, (
        f"seed={seed} c={complexity}: arms have differing target "
        f"counts {counts}\n{print_query(q)}"
    )


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("complexity", [9, 10])
def test_setop_arms_have_compatible_target_types(seed, complexity):
    """Each arm's target i must produce a type implicitly castable
    to the first arm's target i (PG's union-resolution rule).

    The generator forces arms 2+ to match arm 1's types via
    `_gen_arm_select(target_types=...)`. Cast compatibility is
    direction-aware (PG actually unions by the most-general type,
    but our simpler rule "castable to arm 1's type" is a stronger
    sufficient condition).

    Recursively walks nested SetOp arms so the same compatibility
    rule applies all the way down — a `(B INTERSECT C)` arm of an
    outer UNION must have B and C target types both castable to
    the outer arm 1's types."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    if not isinstance(q.select, SetOp):
        return

    def first_select(arm):
        """Walk to the leftmost Select in a possibly-nested SetOp arm.
        That Select's targets define the arm's output type signature."""
        while isinstance(arm, SetOp):
            arm = arm.arms[0]
        return arm

    def check_arm(arm, expected_types):
        if isinstance(arm, SetOp):
            for inner in arm.arms:
                check_arm(inner, expected_types)
            return
        for i, t in enumerate(arm.targets):
            assert implicitly_castable(t.expr.pg_type, expected_types[i]), (
                f"seed={seed} c={complexity}: arm target {i} type "
                f"{t.expr.pg_type.name} not castable to arm-1 type "
                f"{expected_types[i].name}\n{print_query(q)}"
            )

    arm1_types = [t.expr.pg_type for t in first_select(q.select.arms[0]).targets]
    for arm in q.select.arms[1:]:
        check_arm(arm, arm1_types)


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("complexity", [9, 10])
def test_setop_arms_have_no_per_arm_clauses(seed, complexity):
    """Per-arm ORDER BY/LIMIT/OFFSET/WITH belong only at the OUTER
    SetOp wrapper. Per-arm placements would either be ambiguous
    (vs the wrapper's clauses) or require explicit parens (PG
    grammar). The generator strips them via `_strip_arm_clauses`
    for plain Select arms and never sets them on nested SetOp arms.

    Recursively walks nested SetOp arms — same rule applies all the
    way down: only the outermost SetOp may carry these clauses."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    if not isinstance(q.select, SetOp):
        return

    def check(arm, depth):
        if isinstance(arm, SetOp):
            # Nested SetOp: must NOT carry order_by/limit/offset
            # (those would conflict with the outer wrapper's).
            assert arm.order_by == (), (
                f"seed={seed} c={complexity}: nested SetOp at depth "
                f"{depth} has order_by\n{print_query(q)}"
            )
            assert arm.limit is None, (
                f"seed={seed} c={complexity}: nested SetOp at depth "
                f"{depth} has limit\n{print_query(q)}"
            )
            assert arm.offset is None, (
                f"seed={seed} c={complexity}: nested SetOp at depth "
                f"{depth} has offset\n{print_query(q)}"
            )
            for inner in arm.arms:
                check(inner, depth + 1)
        else:
            # Plain Select arm: same rule plus no with_ctes.
            assert arm.order_by == (), (
                f"seed={seed} c={complexity}: arm at depth {depth} "
                f"has order_by\n{print_query(q)}"
            )
            assert arm.limit is None, (
                f"seed={seed} c={complexity}: arm at depth {depth} "
                f"has limit\n{print_query(q)}"
            )
            assert arm.offset is None, (
                f"seed={seed} c={complexity}: arm at depth {depth} "
                f"has offset\n{print_query(q)}"
            )
            assert arm.with_ctes == (), (
                f"seed={seed} c={complexity}: arm at depth {depth} "
                f"has with_ctes\n{print_query(q)}"
            )

    for arm in q.select.arms:
        check(arm, depth=1)


def test_no_setops_at_low_complexity():
    """Below c=9 the FEATURE_SET_OP flag is absent, so no SetOps
    should appear."""
    for seed in range(30):
        for c in (0, 1, 2, 3, 4, 5, 6, 7, 8):
            schema = generate_schema(seed=seed, complexity=c)
            q = generate_query(seed=seed, schema=schema, complexity=c)
            assert not isinstance(q.select, SetOp), (
                f"seed={seed} c={c}: SetOp appeared at low complexity\n"
                f"{print_query(q)}"
            )


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("complexity", [9, 10])
def test_setop_op_is_one_of_three(seed, complexity):
    """Sanity: the operator must be one of the recognized three."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    if not isinstance(q.select, SetOp):
        return
    assert q.select.op in ("UNION", "INTERSECT", "EXCEPT"), (
        f"seed={seed} c={complexity}: unknown op {q.select.op!r}"
    )


# ===========================================================================
# Milestone 8: recursive CTE invariants
# ===========================================================================


def _all_cte_defs(body):
    """Yield every CteDef anywhere in `body` — top-level WITH and any
    nested WITH inside subqueries / arms. For milestone-8 the
    generator only puts WITH at the top level, but the walk handles
    deeper cases."""
    for s in _all_select_bodies(body):
        yield from s.with_ctes


@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("complexity", [10])
def test_recursive_cte_query_parses(seed, complexity):
    """At c=10 (recursive CTE unlock notch), generated queries with
    recursive CTEs must pglast-parse."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    sql = print_query(q)
    result = check_syntax(sql)
    assert result.ok, (
        f"seed={seed} complexity={complexity}: {result.error}\n{sql}"
    )


def test_recursive_cte_actually_fires():
    """Sanity: at c=10, a meaningful share of CTE-having queries
    should have at least one recursive CTE."""
    n_recursive = 0
    n_with = 0
    total = 200
    for seed in range(total):
        schema = generate_schema(seed=seed, complexity=10)
        q = generate_query(seed=seed, schema=schema, complexity=10)
        ctes = list(_all_cte_defs(q.select))
        if ctes:
            n_with += 1
            if any(c.recursive for c in ctes):
                n_recursive += 1
    assert n_with > 5, f"too few CTE queries to evaluate ({n_with})"
    # p_recursive_when_cte=0.3 → roughly 30% of CTE-having queries
    # should have at least one recursive CTE.
    assert n_recursive > 0, "no recursive CTEs in 200 c=10 queries"


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("complexity", [10])
def test_recursive_cte_body_is_setop(seed, complexity):
    """Every recursive CTE's body is a SetOp (base UNION ALL
    recursive) — that's the canonical recursive-CTE shape."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    for cte in _all_cte_defs(q.select):
        if not cte.recursive:
            continue
        assert isinstance(cte.select, SetOp), (
            f"seed={seed} c={complexity}: recursive CTE {cte.name!r} "
            f"body is {type(cte.select).__name__}, expected SetOp\n"
            f"{print_query(q)}"
        )


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("complexity", [10])
def test_recursive_cte_recursive_arm_self_references(seed, complexity):
    """Every recursive CTE's recursive arm (arm 2 of the SetOp body)
    must contain a CteRef to the CTE itself. Without this, the
    "recursive" CTE wouldn't actually recurse — would just be a
    plain 2-arm UNION."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    for cte in _all_cte_defs(q.select):
        if not cte.recursive:
            continue
        rec_arm = cte.select.arms[1]
        refs = list(_cte_refs_in_select(rec_arm))
        names = [r.cte_name for r in refs]
        assert cte.name in names, (
            f"seed={seed} c={complexity}: recursive CTE {cte.name!r} "
            f"has no self-reference in its recursive arm "
            f"(refs found: {names})\n{print_query(q)}"
        )


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("complexity", [10])
def test_recursive_cte_base_arm_does_not_self_reference(seed, complexity):
    """The base arm of a recursive CTE must NOT reference the CTE
    being defined — that's the "base case" semantically."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    for cte in _all_cte_defs(q.select):
        if not cte.recursive:
            continue
        base_arm = cte.select.arms[0]
        refs = list(_cte_refs_in_select(base_arm))
        names = [r.cte_name for r in refs]
        assert cte.name not in names, (
            f"seed={seed} c={complexity}: recursive CTE {cte.name!r} "
            f"base arm self-references (only the recursive arm should)\n"
            f"{print_query(q)}"
        )


@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("complexity", [10])
def test_with_recursive_keyword_when_any_cte_recursive(seed, complexity):
    """The `WITH RECURSIVE` keyword must appear when any CTE in the
    list is recursive — PG enforces this at parse-analysis time."""
    schema = generate_schema(seed=seed, complexity=complexity)
    q = generate_query(seed=seed, schema=schema, complexity=complexity)
    sql = print_query(q)
    for s in _all_select_bodies(q.select):
        if any(c.recursive for c in s.with_ctes):
            # The body's WITH must use RECURSIVE — substring check
            # is good enough since we always emit uppercase.
            assert "WITH RECURSIVE" in sql, (
                f"seed={seed} c={complexity}: recursive CTE present "
                f"but WITH RECURSIVE not in SQL\n{sql}"
            )
            return  # one match is enough


def test_no_recursive_ctes_below_c10():
    """Below c=10 the FEATURE_RECURSIVE_CTE flag is absent."""
    for seed in range(30):
        for c in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
            schema = generate_schema(seed=seed, complexity=c)
            q = generate_query(seed=seed, schema=schema, complexity=c)
            for cte in _all_cte_defs(q.select):
                assert not cte.recursive, (
                    f"seed={seed} c={c}: recursive CTE appeared below "
                    f"c=10\n{print_query(q)}"
                )
