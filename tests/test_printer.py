"""Tests for waxsql.printer.

These are the hand-crafted-AST round-trip tests called out in
HANDOFF.md: build an AST, print it, parse it back via pglast, and
either (a) confirm pglast accepts it, or (b) walk the parsed
structure and confirm it matches the original AST's intent.

The structural tests (the "trap tests") matter most: they catch the
class of printer bugs where the SQL parses fine but means something
different from what the AST said. The headline example is
precedence-driven misgrouping of AND/OR.
"""
from pglast import parse_sql
from pglast.enums import BoolExprType, SubLinkType

from waxsql.ast import (
    BinaryOp, Cast, ColumnRef, CteCycle, CteDef, CteRef, CteSearch,
    DerivedTable, Exists, FrameBound, FrameClause, FuncCall, InSubquery,
    JoinExpr, Literal, NamedWindow, OrderByItem, Query, Select, SelectTarget,
    SetOp, Subquery, TableRef, UnaryOp, WindowRef, WindowSpec,
)
from waxsql.printer import print_expr, print_query
from waxsql.types import (
    BOOL, DATE, FLOAT8, INT4, INT8, INTERVAL, JSONB, NUMERIC,
    TEXT, TIMESTAMPTZ, UUID, VARCHAR,
)
from waxsql.validate.syntax import check_syntax


# --- Smoke: every AST node prints to parseable SQL --------------------------

def _ok(sql: str) -> None:
    r = check_syntax(sql)
    assert r.ok, f"{r.error}\n--SQL--\n{sql}"


def test_simple_select_parses():
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "t1", "id")),),
        from_=(TableRef("customers", "t1"),),
    ))
    _ok(print_query(q))


def test_select_with_join_where_order_limit_parses():
    """The HANDOFF.md target query."""
    q = Query(select=Select(
        targets=(
            SelectTarget(expr=ColumnRef(INT8, "t1", "id")),
            SelectTarget(
                expr=FuncCall(TEXT, "upper", (ColumnRef(TEXT, "t2", "name"),)),
                alias="uname",
            ),
        ),
        from_=(JoinExpr(
            left=TableRef("customers", "t1"),
            right=TableRef("orders", "t2"),
            kind="INNER",
            on=BinaryOp(BOOL, "=",
                        ColumnRef(INT8, "t1", "id"),
                        ColumnRef(INT8, "t2", "customer")),
        ),),
        where=BinaryOp(BOOL, "=", ColumnRef(BOOL, "t1", "active"), Literal(BOOL, True)),
        order_by=(OrderByItem(expr=ColumnRef(INT8, "t1", "id")),),
        limit=Literal(INT8, 10),
    ))
    _ok(print_query(q))


def test_left_deep_join_tree_parses():
    """Three tables joined via a left-deep tree of JoinExprs."""
    j_inner = JoinExpr(
        left=TableRef("a", "t1"),
        right=TableRef("b", "t2"),
        kind="INNER",
        on=BinaryOp(BOOL, "=", ColumnRef(INT8, "t1", "id"), ColumnRef(INT8, "t2", "a")),
    )
    j_outer = JoinExpr(
        left=j_inner,
        right=TableRef("c", "t3"),
        kind="LEFT",
        on=BinaryOp(BOOL, "=", ColumnRef(INT8, "t1", "id"), ColumnRef(INT8, "t3", "a")),
    )
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "t1", "id")),),
        from_=(j_outer,),
    ))
    _ok(print_query(q))


def test_cross_join_parses():
    j = JoinExpr(
        left=TableRef("a", "t1"),
        right=TableRef("b", "t2"),
        kind="CROSS",
    )
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "t1", "id")),),
        from_=(j,),
    ))
    _ok(print_query(q))


def test_multiple_from_items_comma_join_parses():
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "t1", "id")),),
        from_=(TableRef("a", "t1"), TableRef("b", "t2")),
    ))
    _ok(print_query(q))


def test_using_join_parses():
    j = JoinExpr(
        left=TableRef("a", "t1"),
        right=TableRef("b", "t2"),
        kind="INNER",
        using=("id",),
    )
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "t1", "id")),),
        from_=(j,),
    ))
    _ok(print_query(q))


# --- Literals ---------------------------------------------------------------

def test_literal_int_renders():
    assert print_expr(Literal(INT4, 42)) == "42"
    assert print_expr(Literal(INT8, 9999999999)) == "9999999999"


def test_literal_bool_renders_lowercase():
    assert print_expr(Literal(BOOL, True)) == "true"
    assert print_expr(Literal(BOOL, False)) == "false"


def test_literal_text_quotes_and_escapes_singles():
    """TEXT/VARCHAR literals always carry an explicit `::text` cast.
    Bare quoted strings would be parsed as 'unknown' type by PG and
    fail polymorphic-resolution contexts (42804). The cast pins the
    type unambiguously."""
    assert print_expr(Literal(TEXT, "hello")) == "'hello'::text"
    assert print_expr(Literal(TEXT, "it's")) == "'it''s'::text"


def test_literal_numeric_keeps_decimal_point():
    """Integer-valued floats must still print with a decimal so PG's
    lexer treats them as numeric, not int."""
    assert "." in print_expr(Literal(NUMERIC, 2.0))
    assert "." in print_expr(Literal(FLOAT8, 1.5))


def test_literal_null_is_typed_cast():
    """Bare NULL has surprising overload-resolution behavior; the
    printer always emits a typed cast."""
    assert print_expr(Literal(INT4, None)) == "NULL::int4"
    assert print_expr(Literal(TEXT, None)) == "NULL::text"


def test_typed_string_literals_for_temporal_and_jsonb():
    # The cast form `'...'::type` is the uniform path; pglast accepts it.
    sql = f"SELECT {print_expr(Literal(DATE, '2024-01-01'))}"
    _ok(sql)
    sql = f"SELECT {print_expr(Literal(TIMESTAMPTZ, '2024-01-01 12:00:00+00'))}"
    _ok(sql)
    sql = f"SELECT {print_expr(Literal(INTERVAL, '1 day'))}"
    _ok(sql)
    sql = f"SELECT {print_expr(Literal(UUID, '00000000-0000-0000-0000-000000000000'))}"
    _ok(sql)
    # Construct the JSONB literal outside the f-string: nested
    # backslash-escaped quotes inside an f-string are a 3.12+ syntax;
    # extracting the value keeps the test parseable on 3.10 / 3.11.
    jsonb_value = '{"k":"v"}'
    sql = f"SELECT {print_expr(Literal(JSONB, jsonb_value))}"
    _ok(sql)
    sql = f"SELECT {print_expr(Literal(VARCHAR, 'abc'))}"
    _ok(sql)


# --- Identifier quoting (delegated to schema.quote_ident) ------------------

def test_reserved_word_alias_is_quoted():
    """`order` is a reserved word; the printer must quote it."""
    q = Query(select=Select(
        targets=(SelectTarget(
            expr=ColumnRef(INT8, "order", "id"),
        ),),
        from_=(TableRef("orders", "order"),),
    ))
    sql = print_query(q)
    assert '"order"' in sql
    _ok(sql)


# --- Function calls --------------------------------------------------------

def test_function_with_args_renders_parens():
    e = FuncCall(TEXT, "upper", (Literal(TEXT, "x"),))
    assert print_expr(e) == "upper('x'::text)"


def test_bare_keyword_function_renders_without_parens():
    """current_date(), current_user(), etc. would be parse errors."""
    e = FuncCall(DATE, "current_date", ())
    assert print_expr(e) == "current_date"
    _ok(f"SELECT {print_expr(e)}")


def test_zero_arg_real_function_renders_with_parens():
    """now() and gen_random_uuid() are real functions; parens required."""
    e = FuncCall(TIMESTAMPTZ, "now", ())
    assert print_expr(e) == "now()"
    _ok(f"SELECT {print_expr(e)}")


def test_nested_function_calls_parse():
    e = FuncCall(TEXT, "upper", (
        FuncCall(TEXT, "lower", (Literal(TEXT, "Hi"),)),
    ))
    assert print_expr(e) == "upper(lower('Hi'::text))"
    _ok(f"SELECT {print_expr(e)}")


# --- count(*) special form -------------------------------------------------

def test_count_star_renders_with_asterisk():
    """`count(*)` is PG's only star-arg aggregate; the printer's
    star-flag path should produce the exact `count(*)` text."""
    e = FuncCall(INT8, "count", (), star=True)
    assert print_expr(e) == "count(*)"
    _ok(f"SELECT {print_expr(e)} FROM (SELECT 1) AS t(x)")


def test_count_star_with_over_renders_window():
    """`count(*) OVER (...)` is canonical SQL for "running row count
    by partition." The star and OVER attachments must compose."""
    e = FuncCall(
        INT8, "count", (),
        over=WindowSpec(),  # empty OVER ()
        star=True,
    )
    assert print_expr(e) == "count(*) OVER ()"
    _ok(f"SELECT {print_expr(e)} FROM (SELECT 1) AS t(x)")


def test_funccall_star_with_args_raises():
    """The post-init invariant must fire — `count(*, x)` is malformed
    and the AST should refuse to construct it."""
    import pytest
    with pytest.raises(ValueError, match="cannot have args"):
        FuncCall(INT8, "count", (Literal(INT4, 1),), star=True)


# --- FILTER (WHERE ...) clause ---------------------------------------------

def test_filter_clause_renders_after_args():
    """`agg(arg) FILTER (WHERE pred)` — the standard filtered-
    aggregate form. Order matters: FILTER comes between the args
    paren and any subsequent OVER (window) clause."""
    e = FuncCall(
        INT8, "sum",
        (Literal(INT4, 1),),
        filter_=Literal(BOOL, True),
    )
    assert print_expr(e) == "sum(1) FILTER (WHERE true)"
    _ok(f"SELECT {print_expr(e)} FROM (SELECT 1) AS t(x)")


def test_count_star_with_filter_renders():
    """`count(*) FILTER (WHERE x > 0)` — the most common filtered
    aggregate in real SQL. Verifies star and FILTER compose."""
    e = FuncCall(
        INT8, "count", (),
        star=True,
        filter_=Literal(BOOL, True),
    )
    assert print_expr(e) == "count(*) FILTER (WHERE true)"
    _ok(f"SELECT {print_expr(e)} FROM (SELECT 1) AS t(x)")


def test_filter_with_over_orders_filter_before_over():
    """PG grammar: FILTER goes BEFORE OVER. `count(*) OVER (...) FILTER (...)`
    is a syntax error; `count(*) FILTER (...) OVER (...)` is correct."""
    e = FuncCall(
        INT8, "count", (),
        star=True,
        filter_=Literal(BOOL, True),
        over=WindowSpec(),
    )
    assert print_expr(e) == "count(*) FILTER (WHERE true) OVER ()"
    _ok(f"SELECT {print_expr(e)} FROM (SELECT 1) AS t(x)")


# --- Window frame clauses --------------------------------------------------

def test_frame_rows_unbounded_preceding_to_current_row():
    """The classic running-aggregate frame."""
    spec = WindowSpec(frame=FrameClause(
        unit="ROWS",
        start=FrameBound(kind="unbounded_preceding"),
        end=FrameBound(kind="current_row"),
    ))
    e = FuncCall(INT8, "count", (), star=True, over=spec)
    assert print_expr(e) == (
        "count(*) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
    )
    _ok(f"SELECT {print_expr(e)} FROM (SELECT 1) AS t(x)")


def test_frame_rows_n_preceding_to_n_following():
    """Centered moving-window frame with integer offsets."""
    spec = WindowSpec(frame=FrameClause(
        unit="ROWS",
        start=FrameBound(kind="preceding", offset=Literal(INT4, 3)),
        end=FrameBound(kind="following", offset=Literal(INT4, 3)),
    ))
    e = FuncCall(INT8, "count", (), star=True, over=spec)
    assert print_expr(e) == (
        "count(*) OVER (ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING)"
    )
    _ok(f"SELECT {print_expr(e)} FROM (SELECT 1) AS t(x)")


def test_frame_single_bound_form():
    """Single-bound form: `ROWS UNBOUNDED PRECEDING` (PG implicitly
    treats as `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`)."""
    spec = WindowSpec(frame=FrameClause(
        unit="ROWS",
        start=FrameBound(kind="unbounded_preceding"),
    ))
    e = FuncCall(INT8, "count", (), star=True, over=spec)
    assert print_expr(e) == "count(*) OVER (ROWS UNBOUNDED PRECEDING)"
    _ok(f"SELECT {print_expr(e)} FROM (SELECT 1) AS t(x)")


def test_frame_with_partition_and_order_renders_in_order():
    """Frame goes AFTER partition/order. Order: PARTITION → ORDER → frame."""
    spec = WindowSpec(
        partition_by=(ColumnRef(INT8, "t", "x"),),
        order_by=(OrderByItem(expr=ColumnRef(INT8, "t", "x")),),
        frame=FrameClause(
            unit="ROWS",
            start=FrameBound(kind="current_row"),
            end=FrameBound(kind="unbounded_following"),
        ),
    )
    e = FuncCall(INT8, "count", (), star=True, over=spec)
    s = print_expr(e)
    assert "PARTITION BY" in s
    pi = s.index("PARTITION BY")
    oi = s.index("ORDER BY")
    fi = s.index("ROWS BETWEEN")
    assert pi < oi < fi, f"clause order wrong in {s!r}"


def test_framebound_offset_required_for_preceding():
    """The post-init invariant: preceding/following kinds must
    have an offset; the unbounded/current_row kinds must NOT."""
    import pytest
    with pytest.raises(ValueError, match="requires an offset"):
        FrameBound(kind="preceding")  # missing offset
    with pytest.raises(ValueError, match="must not have an offset"):
        FrameBound(kind="current_row", offset=Literal(INT4, 1))


def test_frame_with_exclude_renders_after_bounds():
    """`ROWS BETWEEN ... AND ... EXCLUDE CURRENT ROW` — EXCLUDE goes
    last in the frame clause. The body is just the kind name; the
    printer prepends EXCLUDE."""
    spec = WindowSpec(frame=FrameClause(
        unit="ROWS",
        start=FrameBound(kind="unbounded_preceding"),
        end=FrameBound(kind="current_row"),
        exclude="CURRENT ROW",
    ))
    e = FuncCall(INT8, "count", (), star=True, over=spec)
    assert print_expr(e) == (
        "count(*) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW "
        "EXCLUDE CURRENT ROW)"
    )
    _ok(f"SELECT {print_expr(e)} FROM (SELECT 1) AS t(x)")


def test_frame_exclude_all_four_kinds_parse():
    """All four EXCLUDE bodies are valid PG. NO OTHERS is the default
    (semantically same as no EXCLUDE) but accepted explicitly."""
    for body in ("CURRENT ROW", "GROUP", "TIES", "NO OTHERS"):
        spec = WindowSpec(frame=FrameClause(
            unit="ROWS",
            start=FrameBound(kind="unbounded_preceding"),
            end=FrameBound(kind="current_row"),
            exclude=body,
        ))
        e = FuncCall(INT8, "count", (), star=True, over=spec)
        _ok(f"SELECT {print_expr(e)} FROM (SELECT 1) AS t(x)")


def test_frame_range_with_offsets_renders_and_parses():
    """`RANGE BETWEEN 5 PRECEDING AND CURRENT ROW` over a single
    numeric ORDER BY column — the canonical range-offset shape."""
    spec = WindowSpec(
        order_by=(OrderByItem(expr=ColumnRef(INT8, "t", "x")),),
        frame=FrameClause(
            unit="RANGE",
            start=FrameBound(kind="preceding", offset=Literal(INT4, 5)),
            end=FrameBound(kind="current_row"),
        ),
    )
    e = FuncCall(INT8, "count", (), star=True, over=spec)
    sql = f"SELECT {print_expr(e)} FROM (SELECT 1::int8 AS x) AS t"
    assert "RANGE BETWEEN 5 PRECEDING AND CURRENT ROW" in sql
    _ok(sql)


def test_within_group_renders_after_args_before_filter():
    """`percentile_cont(0.5) WITHIN GROUP (ORDER BY x) FILTER (WHERE y)`
    — WITHIN GROUP comes immediately after args, FILTER after that."""
    e = FuncCall(
        FLOAT8, "percentile_cont",
        (Literal(FLOAT8, 0.5),),
        within_group=(OrderByItem(expr=ColumnRef(FLOAT8, "t", "x")),),
        filter_=Literal(BOOL, True),
    )
    s = print_expr(e)
    assert s == (
        "percentile_cont(0.5) WITHIN GROUP (ORDER BY t.x ASC) "
        "FILTER (WHERE true)"
    )
    _ok(f"SELECT {s} FROM (SELECT 1::float8 AS x) AS t")


def test_frame_groups_with_offsets_renders_and_parses():
    """`GROUPS BETWEEN 3 PRECEDING AND 3 FOLLOWING` over an ORDER BY
    column — GROUPS counts peer groups, integer offset always OK."""
    spec = WindowSpec(
        order_by=(OrderByItem(expr=ColumnRef(INT8, "t", "x")),),
        frame=FrameClause(
            unit="GROUPS",
            start=FrameBound(kind="preceding", offset=Literal(INT4, 3)),
            end=FrameBound(kind="following", offset=Literal(INT4, 3)),
        ),
    )
    e = FuncCall(INT8, "count", (), star=True, over=spec)
    sql = f"SELECT {print_expr(e)} FROM (SELECT 1::int8 AS x) AS t"
    assert "GROUPS BETWEEN 3 PRECEDING AND 3 FOLLOWING" in sql
    _ok(sql)


# --- Named windows (WINDOW clause + OVER name) -----------------------------

def test_window_ref_renders_without_parens():
    """`OVER name` form: a bare identifier, no parens. The parens
    are what distinguishes the inline-spec form from the named-ref
    form in PG's grammar."""
    e = FuncCall(INT8, "count", (), star=True, over=WindowRef("w1"))
    assert print_expr(e) == "count(*) OVER w1"


def test_select_with_window_clause_renders_after_having_before_order_by():
    """PG grammar order: HAVING → WINDOW → ORDER BY. The WINDOW
    clause lists `name AS (spec)` entries, with the spec rendered
    via the same code path as inline OVER specs."""
    spec = WindowSpec(
        partition_by=(ColumnRef(INT8, "t1", "id"),),
    )
    s = Select(
        targets=(
            SelectTarget(expr=FuncCall(
                INT8, "count", (), star=True, over=WindowRef("w1"),
            )),
        ),
        from_=(TableRef("customers", "t1"),),
        windows=(NamedWindow(name="w1", spec=spec),),
    )
    sql = print_query(Query(select=s))
    assert "WINDOW w1 AS (PARTITION BY t1.id)" in sql
    assert "OVER w1" in sql
    _ok(sql)


# --- Operators -------------------------------------------------------------

def test_word_operators_have_spaces():
    e = BinaryOp(BOOL, "AND", Literal(BOOL, True), Literal(BOOL, False))
    sql = print_expr(e)
    # The critical thing: 'true' and 'AND' must not concatenate.
    assert "true AND false" in sql or "(true) AND (false)" in sql


def test_symbolic_operators_render():
    e = BinaryOp(INT4, "+", Literal(INT4, 1), Literal(INT4, 2))
    assert print_expr(e) == "1 + 2"


def test_unary_minus_renders():
    e = UnaryOp(INT4, "-", Literal(INT4, 5))
    assert print_expr(e) == "-5"


def test_cast_renders_double_colon():
    e = Cast(INT8, Literal(INT4, 1), INT8)
    assert print_expr(e) == "1::int8"


# --- The headline trap test: precedence is preserved -----------------------

def test_and_or_precedence_is_preserved_through_round_trip():
    """`(a OR b) AND c` must round-trip as `(a OR b) AND c`.

    Without parenthesization, PG's precedence (AND tighter than OR)
    would re-parse as `a OR (b AND c)` — silently wrong. This is the
    most important printer test; if it ever fails, every AST with an
    AND-of-ORs is being misrendered.
    """
    a = ColumnRef(BOOL, "t1", "a")
    b = ColumnRef(BOOL, "t1", "b")
    c = ColumnRef(BOOL, "t1", "c")
    expr = BinaryOp(BOOL, "AND", BinaryOp(BOOL, "OR", a, b), c)

    q = Query(select=Select(
        targets=(SelectTarget(expr=Literal(INT4, 1)),),
        from_=(TableRef("t", "t1"),),
        where=expr,
    ))
    sql = print_query(q)

    parsed = parse_sql(sql)
    where = parsed[0].stmt.whereClause
    # Top of the WHERE tree must be AND (matches the AST's outer op).
    assert where.boolop == BoolExprType.AND_EXPR, (
        f"Expected top AND, got {where.boolop}\nSQL was:\n{sql}"
    )
    # One of AND's children must itself be a BoolExpr with OR.
    or_children = [
        a for a in where.args
        if hasattr(a, "boolop") and a.boolop == BoolExprType.OR_EXPR
    ]
    assert or_children, (
        f"Expected an OR child under AND, got args={where.args}\nSQL:\n{sql}"
    )


def test_or_and_precedence_inverse_does_not_need_parens():
    """`a OR (b AND c)` matches PG's natural precedence; the parens
    around the AND are redundant but harmless. The structure must
    still re-parse correctly either way."""
    a = ColumnRef(BOOL, "t1", "a")
    b = ColumnRef(BOOL, "t1", "b")
    c = ColumnRef(BOOL, "t1", "c")
    expr = BinaryOp(BOOL, "OR", a, BinaryOp(BOOL, "AND", b, c))

    q = Query(select=Select(
        targets=(SelectTarget(expr=Literal(INT4, 1)),),
        from_=(TableRef("t", "t1"),),
        where=expr,
    ))
    sql = print_query(q)

    where = parse_sql(sql)[0].stmt.whereClause
    assert where.boolop == BoolExprType.OR_EXPR
    and_children = [
        a for a in where.args
        if hasattr(a, "boolop") and a.boolop == BoolExprType.AND_EXPR
    ]
    assert and_children


def test_cast_of_compound_expression_parens_correctly():
    """`(a + b)::int8` — the cast must apply to the whole binary, not
    just to `b`."""
    inner = BinaryOp(INT4, "+", Literal(INT4, 1), Literal(INT4, 2))
    e = Cast(INT8, inner, INT8)
    sql = f"SELECT {print_expr(e)}"
    _ok(sql)
    # Verify the printed form actually contains parens around the inner.
    assert "(1 + 2)::int8" == print_expr(e)


# --- Order BY / NULLS / DESC ------------------------------------------------

def test_order_by_desc_with_nulls_first_parses():
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "t1", "id")),),
        from_=(TableRef("t", "t1"),),
        order_by=(OrderByItem(
            expr=ColumnRef(INT8, "t1", "id"),
            direction="DESC",
            nulls="FIRST",
        ),),
    ))
    _ok(print_query(q))


# --- Subquery node round-trips (milestone 3) -------------------------------

def _make_inner_select(target_expr, *, where=None) -> Select:
    """Tiny helper: a one-target SELECT for use as a subquery body."""
    return Select(
        targets=(SelectTarget(expr=target_expr),),
        from_=(TableRef("orders", "t2"),),
        where=where,
    )


def test_scalar_subquery_round_trips_as_expr_sublink():
    """`WHERE t1.x = (SELECT ...)` — pglast represents the right side
    as a SubLink with subLinkType=EXPR_SUBLINK."""
    inner = _make_inner_select(
        FuncCall(INT8, "max", (ColumnRef(INT8, "t2", "amount"),)),
    )
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "t1", "id")),),
        from_=(TableRef("customers", "t1"),),
        where=BinaryOp(BOOL, "=",
                       ColumnRef(INT8, "t1", "id"),
                       Subquery(INT8, inner)),
    ))
    sql = print_query(q)
    _ok(sql)
    where = parse_sql(sql)[0].stmt.whereClause
    # Right side of `t1.id = (subquery)` is the SubLink.
    sub = where.rexpr
    assert sub.subLinkType == SubLinkType.EXPR_SUBLINK, sub.subLinkType


def test_exists_subquery_round_trips_as_exists_sublink():
    inner = _make_inner_select(Literal(INT8, 1))
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "t1", "id")),),
        from_=(TableRef("customers", "t1"),),
        where=Exists(BOOL, inner, negated=False),
    ))
    sql = print_query(q)
    _ok(sql)
    where = parse_sql(sql)[0].stmt.whereClause
    assert where.subLinkType == SubLinkType.EXISTS_SUBLINK


def test_not_exists_subquery_round_trips_as_not_exists():
    """NOT EXISTS becomes `NOT (EXISTS (...))` in pglast — a BoolExpr
    of NOT_EXPR wrapping a SubLink of EXISTS_SUBLINK. Either form
    PG accepts; ours must produce the parseable shape."""
    inner = _make_inner_select(Literal(INT8, 1))
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "t1", "id")),),
        from_=(TableRef("customers", "t1"),),
        where=Exists(BOOL, inner, negated=True),
    ))
    sql = print_query(q)
    _ok(sql)
    where = parse_sql(sql)[0].stmt.whereClause
    assert where.boolop == BoolExprType.NOT_EXPR
    inner_sub = where.args[0]
    assert inner_sub.subLinkType == SubLinkType.EXISTS_SUBLINK


def test_in_subquery_round_trips_as_any_sublink():
    """`IN (subquery)` is sugar for `= ANY (subquery)`. pglast records
    this as a SubLink of ANY_SUBLINK."""
    inner = _make_inner_select(ColumnRef(INT8, "t2", "category"))
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(TEXT, "t1", "name")),),
        from_=(TableRef("products", "t1"),),
        where=InSubquery(BOOL,
                         ColumnRef(INT8, "t1", "category"),
                         inner,
                         negated=False),
    ))
    sql = print_query(q)
    _ok(sql)
    where = parse_sql(sql)[0].stmt.whereClause
    assert where.subLinkType == SubLinkType.ANY_SUBLINK


def test_not_in_subquery_round_trips_as_not_any():
    """NOT IN parses as `NOT (ANY_SUBLINK)`."""
    inner = _make_inner_select(ColumnRef(INT8, "t2", "category"))
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(TEXT, "t1", "name")),),
        from_=(TableRef("products", "t1"),),
        where=InSubquery(BOOL,
                         ColumnRef(INT8, "t1", "category"),
                         inner,
                         negated=True),
    ))
    sql = print_query(q)
    _ok(sql)
    where = parse_sql(sql)[0].stmt.whereClause
    assert where.boolop == BoolExprType.NOT_EXPR
    inner_sub = where.args[0]
    assert inner_sub.subLinkType == SubLinkType.ANY_SUBLINK


def test_subquery_in_select_list_parses():
    """Scalar subqueries appear naturally in WHERE, but the syntax
    permits them anywhere a value of the right type is wanted —
    including the SELECT list. Verify that path too."""
    inner = _make_inner_select(
        FuncCall(INT8, "count", (ColumnRef(INT8, "t2", "id"),))
    )
    q = Query(select=Select(
        targets=(
            SelectTarget(expr=ColumnRef(INT8, "t1", "id")),
            SelectTarget(expr=Subquery(INT8, inner), alias="order_count"),
        ),
        from_=(TableRef("customers", "t1"),),
    ))
    _ok(print_query(q))


def test_inline_subquery_preserves_embedded_newline_in_literal():
    """Regression guard: `_print_select_inline` must NOT corrupt a
    TEXT literal that contains a newline. Earlier versions did
    `_print_select(s).replace("\\n", " ")`, which collapsed clause
    boundaries AND any newline inside a literal — silently producing
    `'foo bar'` for `Literal(TEXT, "foo\\nbar")`. The fix
    parameterizes the join character (sep=" " in the inline case)
    so the renderer never emits the newline in the first place.

    The literal generator never produces newline-containing strings,
    so the bug was latent. This test fires only when an embedded
    Literal carries a newline value — preventing future regressions."""
    inner = _make_inner_select(
        Literal(TEXT, "foo\nbar"),
    )
    q = Query(select=Select(
        targets=(
            SelectTarget(expr=Subquery(TEXT, inner), alias="snippet"),
        ),
        from_=(TableRef("customers", "t1"),),
    ))
    sql = print_query(q)
    # The literal must round-trip with its newline intact (as PG's
    # standard-conforming string-literal syntax expresses it).
    assert "foo\nbar" in sql, (
        f"literal newline corrupted by inline rendering:\n{sql!r}"
    )


def test_embedded_subquery_renders_inline():
    """The `_print_select_inline` helper should collapse the subquery
    body to a single line so the embedding clause stays readable."""
    inner = _make_inner_select(
        ColumnRef(INT8, "t2", "id"),
        where=BinaryOp(BOOL, "=",
                       ColumnRef(INT8, "t2", "customer"),
                       ColumnRef(INT8, "t1", "id")),
    )
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "t1", "id")),),
        from_=(TableRef("customers", "t1"),),
        where=Exists(BOOL, inner),
    ))
    sql = print_query(q)
    # The WHERE line containing EXISTS (...) should be one line —
    # no embedded newlines inside the parens.
    where_line = next(line for line in sql.split("\n") if "EXISTS" in line)
    open_paren = where_line.index("(")
    close_paren = where_line.rindex(")")
    # The whole subquery body lives between those parens, on this line.
    body = where_line[open_paren + 1:close_paren]
    assert "\n" not in body, f"subquery body is not inline: {body!r}"
    assert "SELECT" in body and "FROM" in body


# --- DerivedTable round-trips (milestone 4) --------------------------------

def _derived(*, lateral: bool = False, alias: str = "sq",
             column_aliases=()) -> DerivedTable:
    """Tiny helper: a one-target derived table for testing."""
    inner = Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "t9", "id"), alias="c1"),),
        from_=(TableRef("orders", "t9"),),
    )
    return DerivedTable(inner, alias, column_aliases=column_aliases,
                        lateral=lateral)


def test_non_lateral_derived_table_round_trips():
    """`SELECT sq.c1 FROM (SELECT col FROM t9) AS sq` — pglast
    represents the FROM item as a RangeSubselect with lateral=False."""
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "sq", "c1")),),
        from_=(_derived(),),
    ))
    sql = print_query(q)
    _ok(sql)
    fr = parse_sql(sql)[0].stmt.fromClause
    assert type(fr[0]).__name__ == "RangeSubselect"
    assert fr[0].lateral is False
    assert fr[0].alias.aliasname == "sq"


def test_lateral_derived_table_round_trips():
    """LATERAL is rendered as a prefix; pglast records it on the
    RangeSubselect's lateral field."""
    q = Query(select=Select(
        targets=(
            SelectTarget(expr=ColumnRef(INT8, "t1", "id")),
            SelectTarget(expr=ColumnRef(INT8, "sq", "c1")),
        ),
        from_=(
            TableRef("customers", "t1"),
            _derived(lateral=True),
        ),
    ))
    sql = print_query(q)
    _ok(sql)
    fr = parse_sql(sql)[0].stmt.fromClause
    assert type(fr[1]).__name__ == "RangeSubselect"
    assert fr[1].lateral is True


def test_derived_table_in_join_round_trips():
    """`t1 JOIN (SELECT ...) AS sq ON true` — derived table on the
    right side of a JoinExpr. JoinExpr.print_from_item dispatches
    on FromItem subtype, so DerivedTable on either side just works."""
    j = JoinExpr(
        left=TableRef("customers", "t1"),
        right=_derived(),
        kind="INNER",
        on=Literal(BOOL, True),
    )
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "t1", "id")),),
        from_=(j,),
    ))
    _ok(print_query(q))


def test_derived_table_with_column_aliases_round_trips():
    """`(SELECT ...) AS sq(a, b)` — the column-alias list. Reserved
    feature; the milestone-4 generator doesn't emit it but the
    printer must support it for future use."""
    d = _derived(column_aliases=("a",))
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "sq", "a")),),
        from_=(d,),
    ))
    sql = print_query(q)
    _ok(sql)
    fr = parse_sql(sql)[0].stmt.fromClause
    assert fr[0].alias.colnames is not None
    assert [c.sval for c in fr[0].alias.colnames] == ["a"]


def test_derived_table_body_renders_inline():
    """Same one-line-body invariant as expression-position subqueries
    — derived tables in the FROM clause stay on one line."""
    d = _derived()
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "sq", "c1")),),
        from_=(d,),
    ))
    sql = print_query(q)
    from_line = next(line for line in sql.split("\n") if "FROM" in line)
    # The body inside (...) AS sq should have no newlines.
    open_paren = from_line.index("(")
    close_paren = from_line.rindex(")")
    body = from_line[open_paren + 1:close_paren]
    assert "\n" not in body, f"derived body not inline: {body!r}"


# --- WITH clause / CteDef / CteRef round-trips (milestone 5) ---------------

def _trivial_cte_select(*, target_alias: str = "c1") -> Select:
    """One-target SELECT useful as a CTE body."""
    return Select(
        targets=(SelectTarget(
            expr=ColumnRef(INT8, "t9", "id"),
            alias=target_alias,
        ),),
        from_=(TableRef("orders", "t9"),),
    )


def test_single_cte_round_trips():
    """`WITH cte1 AS (SELECT ...) SELECT c.c1 FROM cte1 AS c` —
    pglast represents the prefix as a WithClause with one
    CommonTableExpr."""
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "c", "c1")),),
        from_=(CteRef("cte1", "c"),),
        with_ctes=(CteDef("cte1", _trivial_cte_select()),),
    ))
    sql = print_query(q)
    _ok(sql)
    stmt = parse_sql(sql)[0].stmt
    assert stmt.withClause is not None
    assert len(stmt.withClause.ctes) == 1
    assert stmt.withClause.ctes[0].ctename == "cte1"


def test_multiple_ctes_round_trip_in_order():
    """Two CTEs separated by comma in the WITH list. The second can
    reference the first because we generate them in declaration
    order — same constraint PG enforces (without RECURSIVE)."""
    cte1 = CteDef("cte1", _trivial_cte_select())
    inner2 = Select(
        targets=(SelectTarget(
            expr=ColumnRef(INT8, "first_alias", "c1"),
            alias="c1",
        ),),
        from_=(CteRef("cte1", "first_alias"),),
    )
    cte2 = CteDef("cte2", inner2)
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "c", "c1")),),
        from_=(CteRef("cte2", "c"),),
        with_ctes=(cte1, cte2),
    ))
    sql = print_query(q)
    _ok(sql)
    stmt = parse_sql(sql)[0].stmt
    assert [c.ctename for c in stmt.withClause.ctes] == ["cte1", "cte2"]


def test_materialized_modifier_round_trips():
    """The MATERIALIZED keyword (PG 12+) goes between AS and the body.
    NOT MATERIALIZED is the symmetric case."""
    cte_mat = CteDef("cte_mat", _trivial_cte_select(), materialized=True)
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "c", "c1")),),
        from_=(CteRef("cte_mat", "c"),),
        with_ctes=(cte_mat,),
    ))
    sql = print_query(q)
    _ok(sql)
    assert "MATERIALIZED" in sql

    cte_not_mat = CteDef(
        "cte_nm", _trivial_cte_select(), materialized=False,
    )
    q2 = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "c", "c1")),),
        from_=(CteRef("cte_nm", "c"),),
        with_ctes=(cte_not_mat,),
    ))
    sql2 = print_query(q2)
    _ok(sql2)
    assert "NOT MATERIALIZED" in sql2


def test_cte_with_column_aliases_round_trips():
    """`WITH c(x) AS (SELECT ...)` — explicit column-alias list,
    same shape as DerivedTable's column_aliases."""
    cte = CteDef(
        "cte_cols", _trivial_cte_select(), column_aliases=("x",),
    )
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "c", "x")),),
        from_=(CteRef("cte_cols", "c"),),
        with_ctes=(cte,),
    ))
    sql = print_query(q)
    _ok(sql)
    stmt = parse_sql(sql)[0].stmt
    assert stmt.withClause.ctes[0].aliascolnames is not None
    assert [n.sval for n in stmt.withClause.ctes[0].aliascolnames] == ["x"]


def test_cte_ref_in_join_round_trips():
    """A CTE reference can appear on either side of a JoinExpr —
    same shape as TableRef in that respect."""
    cte = CteDef("cte1", _trivial_cte_select())
    j = JoinExpr(
        left=TableRef("customers", "t1"),
        right=CteRef("cte1", "c"),
        kind="INNER",
        on=Literal(BOOL, True),
    )
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "c", "c1")),),
        from_=(j,),
        with_ctes=(cte,),
    ))
    _ok(print_query(q))


def test_cte_body_renders_inline():
    """CTE bodies use the same `_print_select_inline` helper as
    derived tables and scalar subqueries — body stays on one line
    so the WITH prefix line stays readable for multi-CTE forms."""
    cte = CteDef("cte1", _trivial_cte_select())
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "c", "c1")),),
        from_=(CteRef("cte1", "c"),),
        with_ctes=(cte,),
    ))
    sql = print_query(q)
    with_line = next(line for line in sql.split("\n") if line.startswith("WITH"))
    open_paren = with_line.index("(")
    close_paren = with_line.rindex(")")
    body = with_line[open_paren + 1:close_paren]
    assert "\n" not in body, f"CTE body not inline: {body!r}"


# --- Recursive CTE: SEARCH and CYCLE clauses --------------------------------

def _trivial_recursive_body() -> SetOp:
    """A minimal `base UNION ALL recursive` body for a recursive CTE
    test fixture. The arms structurally mimic what gen_recursive_cte_def
    produces but are hand-built for printer-test predictability."""
    base = Select(
        targets=(SelectTarget(
            expr=Literal(INT4, 1), alias="c1",
        ),),
        from_=(),
    )
    rec = Select(
        targets=(SelectTarget(
            expr=ColumnRef(INT4, "r", "c1"), alias="c1",
        ),),
        from_=(CteRef("cte1", "r"),),
    )
    return SetOp(op="UNION", all=True, arms=(base, rec))


def test_recursive_cte_with_search_renders_breadth_first():
    cte = CteDef(
        name="cte1",
        select=_trivial_recursive_body(),
        recursive=True,
        search=CteSearch(
            breadth_first=True,
            by_columns=("c1",),
            set_column="search_seq",
        ),
    )
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT4, "r", "c1")),),
        from_=(CteRef("cte1", "r"),),
        with_ctes=(cte,),
    ))
    sql = print_query(q)
    assert "SEARCH BREADTH FIRST BY c1 SET search_seq" in sql
    _ok(sql)


def test_recursive_cte_with_cycle_renders_path_clause():
    cte = CteDef(
        name="cte1",
        select=_trivial_recursive_body(),
        recursive=True,
        cycle=CteCycle(
            columns=("c1",),
            cycle_mark_column="is_cycle",
            path_column="cycle_path",
        ),
    )
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT4, "r", "c1")),),
        from_=(CteRef("cte1", "r"),),
        with_ctes=(cte,),
    ))
    sql = print_query(q)
    assert "CYCLE c1 SET is_cycle USING cycle_path" in sql
    _ok(sql)


def test_recursive_cte_search_and_cycle_render_in_order():
    """PG grammar: SEARCH comes before CYCLE."""
    cte = CteDef(
        name="cte1",
        select=_trivial_recursive_body(),
        recursive=True,
        search=CteSearch(
            breadth_first=False,
            by_columns=("c1",),
            set_column="search_seq",
        ),
        cycle=CteCycle(
            columns=("c1",),
            cycle_mark_column="is_cycle",
            path_column="cycle_path",
        ),
    )
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT4, "r", "c1")),),
        from_=(CteRef("cte1", "r"),),
        with_ctes=(cte,),
    ))
    sql = print_query(q)
    assert "SEARCH DEPTH FIRST" in sql
    assert "CYCLE" in sql
    assert sql.index("SEARCH") < sql.index("CYCLE"), (
        f"SEARCH must come before CYCLE in: {sql}"
    )
    _ok(sql)


def test_select_without_with_unchanged():
    """The default with_ctes=() must produce the same output as
    before milestone 5 — no spurious WITH keyword in the SQL."""
    q = Query(select=Select(
        targets=(SelectTarget(expr=ColumnRef(INT8, "t1", "id")),),
        from_=(TableRef("customers", "t1"),),
    ))
    sql = print_query(q)
    assert "WITH" not in sql
    _ok(sql)
