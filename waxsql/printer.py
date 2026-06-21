"""AST → SQL printer.

Pure function on the AST. The printer is the *only* place that knows
about SQL surface syntax — quoting, parenthesization, literal
serialization, and the handful of nodes whose SQL form differs from
what their function/operator signature suggests (the SQL-keyword
nullary "functions" like `current_date`).

Design rules:

  * Conservative parenthesization. Binary operators always wrap
    binary/unary operand subexpressions in parens, even when PG's
    precedence rules would not require it. The cost is verbosity; the
    benefit is that "the AST said `(a OR b) AND c`" prints as something
    that re-parses to `(a OR b) AND c` — never `a OR (b AND c)`. The
    printer-round-trip test in tests/test_printer.py is what enforces
    this property.

  * No string concatenation as control flow. Every clause is rendered
    as a separately-built fragment, then joined with newlines. Easy to
    read, easy to debug, no risk of "missed a space" bugs.

  * Identifiers always go through `quote_ident`. Don't pass raw names
    to f-strings; reserved-word collisions and uppercase/case-folding
    issues hide there.

  * SQL keyword choice: lowercase for short keywords (`from`, `where`,
    `and`, `or`, `is`, etc.), uppercase only where convention is
    overwhelmingly uppercase (`SELECT`, `FROM`, `WHERE`, `JOIN`, ...).
    Generated SQL is for humans to read, not for shouting at.

  * Literal rendering is type-driven. The Literal node carries its
    own pg_type, and `_render_literal` switches on it. NULL renders as
    `NULL::<type>` — bare untyped NULL has surprising overload-
    resolution behavior in PostgreSQL.
"""
from __future__ import annotations

from .ast import (
    BinaryOp, Cast, ColumnRef, CteDef, CteRef, DerivedTable, Exists, Expr,
    FrameBound, FrameClause, FromItem, FuncCall, GroupingSet, InSubquery,
    JoinExpr, Literal, OrderByItem, Query, Select, SelectTarget,
    SetOp, Subquery, TableRef, UnaryOp, WindowRef, WindowSpec,
)
from .schema import quote_ident
from .types import (
    BOOL, FLOAT8, INT4, INT8, NUMERIC, PgType,
)


# Names that PostgreSQL parses as keyword expressions, NOT function
# calls. `current_date()` (with parens, zero args) is a syntax error;
# `current_date` (bare keyword) is the correct form.
#
# Two-arg variants like `current_timestamp(2)` exist as well but use
# special grammar productions — the catalog doesn't model those, so we
# only special-case the zero-arg form.
_BARE_KEYWORD_FUNCS: frozenset[str] = frozenset({
    "current_date", "current_time", "current_timestamp",
    "localtime", "localtimestamp",
    "current_user", "session_user", "user",
})


# ===========================================================================
# Public entry points
# ===========================================================================

def print_query(q: Query) -> str:
    """Render a Query as a SQL statement (no trailing semicolon).

    The trailing-semicolon decision is left to the caller; pglast is
    happy either way, and downstream batching is easier without one.

    `q.select` may be a Select OR a SetOp (UNION/INTERSECT/EXCEPT
    combining multiple SELECTs); dispatch on type.
    """
    body = q.select
    if isinstance(body, SetOp):
        return _print_set_op(body)
    return _print_select(body)


def _print_set_op(s: SetOp, *, sep: str = "\n") -> str:
    """Render `arm1 OP[ ALL] arm2 [OP[ ALL] arm3 ...] [ORDER BY ...] [LIMIT ...]`.

    Each arm is a Select rendered top-level OR a nested SetOp
    rendered inside parens. Parens are required for nested SetOps
    because PG's set-op precedence (INTERSECT > UNION = EXCEPT)
    would otherwise re-associate the operators differently from
    the AST structure. Always-parenthesizing nested arms matches
    pg_dump's behavior for round-trip safety.

    `sep` is the join character between clause fragments (and
    between arms). Default "\\n" gives multi-line readable output.
    Inline contexts (subqueries, derived tables) pass " " so the
    body fits on one line. Building the joined form directly avoids
    a render-then-replace pass that could corrupt embedded literals
    containing the join character.
    """
    op_kw = f"{s.op} ALL" if s.all else s.op
    arm_strs = [_print_set_op_arm(arm, sep=sep) for arm in s.arms]
    parts: list[str] = []
    parts.append((f"{sep}{op_kw}{sep}").join(arm_strs))
    if s.order_by:
        # SetOp ORDER BY uses positional refs (`ORDER BY 1`) since
        # the unified output column names aren't reliably nameable.
        # Opt out of the integer-cast auto-fix.
        parts.append("ORDER BY " + ", ".join(
            _print_order_by(o, allow_positional=True) for o in s.order_by
        ))
    if s.limit is not None:
        parts.append("LIMIT " + _print_expr(s.limit))
    if s.offset is not None:
        parts.append("OFFSET " + _print_expr(s.offset))
    return sep.join(parts)


def _print_set_op_arm(arm: Select | SetOp, *, sep: str = "\n") -> str:
    """Render a single SetOp arm. Plain Select arms render as-is;
    nested SetOp arms get wrapped in parens (mandatory — see
    _print_set_op docstring for the precedence-precedence rationale).
    `sep` propagates to keep nested arms in the same inline/multi-line
    mode as their parent."""
    if isinstance(arm, SetOp):
        return f"({_print_set_op(arm, sep=sep)})"
    return _print_select(arm, sep=sep)


def print_expr(e: Expr) -> str:
    """Render a single expression. Useful for tests and diagnostics."""
    return _print_expr(e)


# ===========================================================================
# SELECT and clause rendering
# ===========================================================================

def _print_select(s: Select, *, sep: str = "\n") -> str:
    """Render a Select. `sep` is the join character between clause
    fragments — default "\\n" for top-level multi-line output, " "
    for inline contexts (called via _print_select_inline). Parameter-
    izing the join is what lets the inline case avoid a render-then-
    replace pass that would corrupt embedded literals containing the
    join character."""
    parts: list[str] = []

    # WITH clause comes first when present. Each CteDef renders on
    # one line; multiple CTEs are comma-separated. The whole WITH
    # gets its own line for readability.
    #
    # `WITH RECURSIVE` is emitted when ANY CTE in the list is
    # recursive — PG's grammar requires the keyword once per WITH
    # list, not per-CTE. Without RECURSIVE, a CTE that references
    # its own name fails parse-analysis with "relation does not
    # exist" (CTE names aren't visible to themselves under plain WITH).
    if s.with_ctes:
        kw = "WITH RECURSIVE" if any(c.recursive for c in s.with_ctes) else "WITH"
        parts.append(kw + " " + ", ".join(_print_cte_def(c) for c in s.with_ctes))

    parts.append("SELECT " + ", ".join(_print_target(t) for t in s.targets))

    if s.from_:
        parts.append("FROM " + ", ".join(_print_from_item(f) for f in s.from_))

    if s.where is not None:
        parts.append("WHERE " + _print_expr(s.where))

    if s.group_by:
        parts.append("GROUP BY " + ", ".join(
            _print_grouping_set(g) if isinstance(g, GroupingSet) else _print_expr(g)
            for g in s.group_by
        ))

    if s.having is not None:
        parts.append("HAVING " + _print_expr(s.having))

    if s.windows:
        # PG grammar: WINDOW comes after HAVING and before ORDER BY.
        # Each entry is `name AS (spec)`; the spec rendering is the
        # same code path used by inline OVER clauses.
        parts.append("WINDOW " + ", ".join(
            f"{quote_ident(w.name)} AS ({_print_window_spec(w.spec)})"
            for w in s.windows
        ))

    if s.order_by:
        parts.append("ORDER BY " + ", ".join(_print_order_by(o) for o in s.order_by))

    if s.limit is not None:
        parts.append("LIMIT " + _print_expr(s.limit))

    if s.offset is not None:
        parts.append("OFFSET " + _print_expr(s.offset))

    return sep.join(parts)


def _print_target(t: SelectTarget) -> str:
    # The target expression renders without any compound-wrapping —
    # SELECT-list position is unambiguous in PG's grammar (commas
    # delimit), so extra parens would only add noise. Contrast with
    # operand positions inside BinaryOp / UnaryOp / Cast where the
    # _wrap_if_compound dance is required.
    body = _print_expr(t.expr)
    if t.alias is not None:
        return f"{body} AS {quote_ident(t.alias)}"
    return body


def _print_order_by(
    o: OrderByItem,
    *,
    allow_positional: bool = False,
) -> str:
    """Render one ORDER BY item.

    SUBTLE: in PG's grammar, a bare integer constant in ORDER BY is
    interpreted as a 1-based output-column position, NOT as a sort
    key value. `ORDER BY 0` means "position 0" (always invalid),
    `ORDER BY 5` means "position 5" (only valid if there are 5+
    targets). When our generator reuses a SELECT-list item that
    happens to be a bare integer Literal as an ORDER BY expr,
    PG mis-interprets — so by default we emit an explicit cast
    (`5::int4`) which disables the positional-ref rule.

    Even parens don't disable the rule (`ORDER BY (0)` is still
    positional); only a cast does.

    The SetOp ORDER BY path INTENTIONALLY uses positional refs
    (`ORDER BY 1 ASC` to sort by the unified first output column,
    since the unified column names aren't always nameable). It
    passes `allow_positional=True` to opt out of the auto-cast.
    """
    expr_str = _print_expr(o.expr)
    if (not allow_positional
            and isinstance(o.expr, Literal)
            and o.expr.value is not None):
        # Force literal interpretation via cast.
        # PG rejects ANY bare non-NULL literal in ORDER BY: "ORDER BY
        # position N is not in select list" for integers, "non-integer
        # constant in ORDER BY" for everything else (booleans, strings,
        # numerics, jsonb, ...). NULL is exempt — `ORDER BY NULL` is
        # accepted as a no-op sort key. Casting disables both rules
        # uniformly across all literal types.
        expr_str = f"{expr_str}::{o.expr.pg_type.sql()}"
    bits = [expr_str, o.direction]
    if o.nulls is not None:
        bits.append("NULLS " + o.nulls)
    return " ".join(bits)


# ===========================================================================
# FROM items
# ===========================================================================

def _print_from_item(f: FromItem) -> str:
    # isinstance-chain dispatch instead of a registry / visitor pattern:
    # the closed set of FromItem subclasses is small, and a flat chain
    # keeps the printer single-file and grep-friendly. The TypeError
    # tail is load-bearing — any new FromItem subclass will fail loudly
    # at runtime rather than silently rendering as nothing.
    if isinstance(f, TableRef):
        return f"{quote_ident(f.table)} AS {quote_ident(f.alias)}"
    if isinstance(f, JoinExpr):
        return _print_join(f)
    if isinstance(f, DerivedTable):
        return _print_derived_table(f)
    if isinstance(f, CteRef):
        return f"{quote_ident(f.cte_name)} AS {quote_ident(f.alias)}"
    raise TypeError(f"Unknown FromItem: {type(f).__name__}")


def _print_cte_def(c: CteDef) -> str:
    """`name [(col1, col2, ...)] AS [MATERIALIZED|NOT MATERIALIZED] (SELECT ...)`.

    The body is rendered inline (newlines collapsed to spaces) so a
    multi-CTE WITH stays on its own line in the outer formatting.
    For recursive CTEs (milestone 8), the body is a SetOp (base
    UNION recursive); the inline-rendering helper handles either
    Select or SetOp via its dispatch on type.

    The MATERIALIZED keyword (PG 12+) goes between AS and the
    parenthesized body. None means no modifier — let PG choose.
    """
    name = quote_ident(c.name)
    cols = ""
    if c.column_aliases:
        cols = "(" + ", ".join(quote_ident(a) for a in c.column_aliases) + ")"
    if c.materialized is True:
        modifier = "MATERIALIZED "
    elif c.materialized is False:
        modifier = "NOT MATERIALIZED "
    else:
        modifier = ""
    body = _print_query_body_inline(c.select)
    out = f"{name}{cols} AS {modifier}({body})"
    if c.search is not None:
        sw = c.search
        order_kw = "BREADTH FIRST" if sw.breadth_first else "DEPTH FIRST"
        by_cols = ", ".join(quote_ident(b) for b in sw.by_columns)
        out += (
            f" SEARCH {order_kw} BY {by_cols} "
            f"SET {quote_ident(sw.set_column)}"
        )
    if c.cycle is not None:
        cy = c.cycle
        cycle_cols = ", ".join(quote_ident(b) for b in cy.columns)
        out += (
            f" CYCLE {cycle_cols} "
            f"SET {quote_ident(cy.cycle_mark_column)} "
            f"USING {quote_ident(cy.path_column)}"
        )
    return out


def _print_query_body_inline(body: Select | SetOp) -> str:
    """Render a query body (Select OR SetOp) as a single line for
    embedding inside a CteDef body, scalar subquery, derived table,
    etc. Same single-line invariant as _print_select_inline; the
    only difference is dispatch on body type. Both branches build
    the inline form by passing sep=" " to the renderer rather than
    rendering with "\\n" and replacing — the replace approach would
    corrupt any embedded literal that contained a newline."""
    if isinstance(body, SetOp):
        return _print_set_op(body, sep=" ")
    return _print_select_inline(body)


def _print_derived_table(d: DerivedTable) -> str:
    """`[LATERAL ](SELECT ...) AS alias[(col1, col2, ...)]`.

    LATERAL is a prefix modifier — PG's grammar parses it before the
    parens, not after the alias. The body is rendered inline (same
    helper milestone 3 used for expression-position subqueries) to
    keep the FROM clause readable.
    """
    prefix = "LATERAL " if d.lateral else ""
    body = _print_select_inline(d.select)
    cols = ""
    if d.column_aliases:
        cols = "(" + ", ".join(quote_ident(c) for c in d.column_aliases) + ")"
    return f"{prefix}({body}) AS {quote_ident(d.alias)}{cols}"


def _print_join(j: JoinExpr) -> str:
    left = _print_from_item(j.left)
    right = _print_from_item(j.right)

    # Nested JoinExprs on either side need parens to keep associativity
    # explicit. PG's syntax accepts the parens, and this avoids any
    # subtle disagreement between our left-deep build order and the
    # parser's reduction order.
    if isinstance(j.left, JoinExpr):
        left = f"({left})"
    if isinstance(j.right, JoinExpr):
        right = f"({right})"

    if j.kind == "CROSS":
        return f"{left} CROSS JOIN {right}"

    head = f"{left} {j.kind} JOIN {right}"
    if j.on is not None:
        return f"{head} ON {_print_expr(j.on)}"
    if j.using:
        cols = ", ".join(quote_ident(c) for c in j.using)
        return f"{head} USING ({cols})"
    # Non-CROSS join with neither ON nor USING is malformed; surface it
    # rather than silently emitting un-parseable SQL.
    raise ValueError(f"{j.kind} JOIN requires ON or USING")


# ===========================================================================
# Expression rendering
# ===========================================================================

def _print_expr(e: Expr) -> str:
    # Central dispatch for every Expr subclass. Order is by frequency
    # — column refs and literals dominate generated output, so they
    # short-circuit first. Adding a new Expr type means adding a clause
    # here; the trailing TypeError catches the "forgot to register"
    # case at the first round-trip test.
    if isinstance(e, ColumnRef):
        return f"{quote_ident(e.table_alias)}.{quote_ident(e.column)}"
    if isinstance(e, Literal):
        return _render_literal(e.value, e.pg_type)
    if isinstance(e, FuncCall):
        return _print_func_call(e)
    if isinstance(e, BinaryOp):
        return _print_binary(e)
    if isinstance(e, UnaryOp):
        return _print_unary(e)
    if isinstance(e, Cast):
        return _print_cast(e)
    if isinstance(e, Subquery):
        return _print_subquery(e)
    if isinstance(e, Exists):
        return _print_exists(e)
    if isinstance(e, InSubquery):
        return _print_in_subquery(e)
    raise TypeError(f"Unknown Expr: {type(e).__name__}")


def _print_func_call(f: FuncCall) -> str:
    if f.star:
        # `name(*)` special form. In practice only `count(*)` is
        # valid PG, but the printer doesn't enforce that — the AST
        # post-init guarantees args is empty when star is set, and
        # the generator only emits star for count. OVER and FILTER
        # pass through the same way they do for arg-bearing calls
        # (`count(*) OVER (...)`, `count(*) FILTER (WHERE ...)` are
        # both canonical SQL).
        base = f"{f.name}(*)"
    elif not f.args and f.name in _BARE_KEYWORD_FUNCS and f.filter_ is None:
        # PostgreSQL parses these without parens. Adding parens is a
        # syntax error.
        # (Window functions and FILTER never appear with bare-keyword
        # nullary functions — current_date doesn't take either — so
        # the filter_-is-None guard is defensive against malformed
        # AST construction rather than a real generator output path.)
        return f.name
    else:
        args = ", ".join(_print_expr(a) for a in f.args)
        base = f"{f.name}({args})"
    # PG grammar: `func(args) [WITHIN GROUP (ORDER BY ...)]
    # [FILTER (WHERE ...)] [OVER (...)]`. WITHIN GROUP comes first
    # of the trailing clauses (used by ordered-set aggregates like
    # percentile_cont). Then FILTER, then OVER.
    if f.within_group:
        wg = ", ".join(_print_order_by(o) for o in f.within_group)
        base = f"{base} WITHIN GROUP (ORDER BY {wg})"
    if f.filter_ is not None:
        base = f"{base} FILTER (WHERE {_print_expr(f.filter_)})"
    if f.over is not None:
        # Window-style call. Two forms:
        #   * Inline WindowSpec → `OVER (PARTITION BY ... ORDER BY ...)`
        #   * Named-window WindowRef → `OVER name` (no parens — PG
        #     grammar distinguishes these by the parens presence)
        if isinstance(f.over, WindowRef):
            return f"{base} OVER {quote_ident(f.over.name)}"
        return f"{base} OVER ({_print_window_spec(f.over)})"
    return base


def _print_window_spec(w: WindowSpec) -> str:
    """Render the body of an OVER clause: PARTITION BY exprs, then
    ORDER BY items, then frame (if any). All sections optional —
    `OVER ()` (empty body) is valid PG (entire result set as one
    partition)."""
    parts: list[str] = []
    if w.partition_by:
        parts.append("PARTITION BY " + ", ".join(
            _print_expr(e) for e in w.partition_by
        ))
    if w.order_by:
        parts.append("ORDER BY " + ", ".join(
            _print_order_by(o) for o in w.order_by
        ))
    if w.frame is not None:
        parts.append(_print_frame_clause(w.frame))
    return " ".join(parts)


def _print_frame_bound(b: FrameBound) -> str:
    """Render one bound: UNBOUNDED PRECEDING / N PRECEDING /
    CURRENT ROW / N FOLLOWING / UNBOUNDED FOLLOWING. The kind→text
    mapping is a small fixed table; defining it inline here keeps
    the bound rendering self-contained."""
    if b.kind == "unbounded_preceding":
        return "UNBOUNDED PRECEDING"
    if b.kind == "current_row":
        return "CURRENT ROW"
    if b.kind == "unbounded_following":
        return "UNBOUNDED FOLLOWING"
    # preceding/following take an offset expression. AST post-init
    # guarantees b.offset is non-None for these kinds.
    assert b.offset is not None  # for type-checkers; enforced by AST
    direction = "PRECEDING" if b.kind == "preceding" else "FOLLOWING"
    return f"{_print_expr(b.offset)} {direction}"


def _print_frame_clause(fc: FrameClause) -> str:
    """Render a window frame clause. Single-bound form when end is
    None: `unit start` (PG implicitly treats as `unit BETWEEN start
    AND CURRENT ROW`). BETWEEN form when end is set:
    `unit BETWEEN start AND end`. Optional EXCLUDE clause appended
    as `EXCLUDE <body>` where body is one of CURRENT ROW / GROUP /
    TIES / NO OTHERS."""
    if fc.end is None:
        body = f"{fc.unit} {_print_frame_bound(fc.start)}"
    else:
        body = (
            f"{fc.unit} BETWEEN "
            f"{_print_frame_bound(fc.start)} AND "
            f"{_print_frame_bound(fc.end)}"
        )
    if fc.exclude is not None:
        body = f"{body} EXCLUDE {fc.exclude}"
    return body


def _print_grouping_set(gs: GroupingSet) -> str:
    """Render a GROUP BY grouping-set construct: ROLLUP, CUBE, or
    GROUPING SETS.

    Element rendering depends on the kind:

      * ROLLUP/CUBE: single-expr elements render bare
        (`ROLLUP (a, b, c)`); multi-expr elements get parens
        (`ROLLUP ((a, b), c)`). PG accepts either, but the bare
        form is the conventional one for the common single-col case.

      * GROUPING SETS: every element gets parens, including the empty
        tuple → `()` (the grand-total grouping). Even single-expr
        elements paren'd: `GROUPING SETS ((a), (b), ())`.
    """
    items: list[str] = []
    for elem in gs.elements:
        if not elem:
            # Empty tuple — the grand-total grouping. Only meaningful
            # in GROUPING SETS, but harmless if it appears in
            # ROLLUP/CUBE (PG accepts).
            items.append("()")
        elif gs.kind == "GROUPING SETS" or len(elem) > 1:
            inner = ", ".join(_print_expr(e) for e in elem)
            items.append(f"({inner})")
        else:
            # Single-expr element in ROLLUP/CUBE — bare.
            items.append(_print_expr(elem[0]))
    return f"{gs.kind} ({', '.join(items)})"


def _print_binary(b: BinaryOp) -> str:
    # Both operands go through _wrap_if_compound. This is the
    # conservative-parens policy in action: we don't consult a
    # precedence table, we just always wrap nested compound
    # expressions. The round-trip tests confirm the resulting SQL
    # re-parses to the same AST shape.
    left = _wrap_if_compound(b.left)
    right = _wrap_if_compound(b.right)
    # Spaces around the symbol always: word-form ops (AND, OR, LIKE,
    # ILIKE) need them for tokenization; symbolic ops (+, -, *, ||)
    # don't strictly need them but read better.
    return f"{left} {b.symbol} {right}"


def _print_unary(u: UnaryOp) -> str:
    operand = _wrap_if_compound(u.operand)
    # `sep` controls whether to insert a space between the operator
    # symbol and the operand: symbolic ops bind tightly to their
    # operand (`-x`, `+x`), while word-form ops need a separator or
    # the lexer would merge them into one identifier (`NOTx` would
    # parse as a single name, not as NOT applied to x). Compound
    # operands are already parenthesized by `_wrap_if_compound`
    # above; `sep` is purely about the symbol↔operand boundary.
    sep = "" if u.symbol in ("-", "+") else " "
    return f"{u.symbol}{sep}{operand}"


def _print_cast(c: Cast) -> str:
    inner = _wrap_if_compound(c.expr)
    return f"{inner}::{c.target_type.sql()}"


def _print_subquery(s: Subquery) -> str:
    """Scalar subquery: `(SELECT ... FROM ...)`. Inner Select is
    rendered inline (newlines collapsed to spaces) to keep embedded
    SQL on one line. Outer parens are intrinsic — no precedence
    interaction with surrounding ops."""
    return f"({_print_select_inline(s.select)})"


def _print_exists(e: Exists) -> str:
    """`[NOT ]EXISTS (SELECT ...)`. NOT is rendered as a separate
    keyword; PG parses `NOT EXISTS` as `NOT (EXISTS ...)` either way."""
    prefix = "NOT EXISTS" if e.negated else "EXISTS"
    return f"{prefix} ({_print_select_inline(e.select)})"


def _print_in_subquery(i: InSubquery) -> str:
    """`<expr> [NOT ]IN (SELECT col FROM ...)`. The left expression
    gets the same compound-wrapping treatment as a binary-op operand,
    keeping `(a + b) IN (...)` correctly grouped."""
    left = _wrap_if_compound(i.expr)
    op = "NOT IN" if i.negated else "IN"
    return f"{left} {op} ({_print_select_inline(i.select)})"


def _print_select_inline(s: Select) -> str:
    """Render a Select as a single line, suitable for embedding inside
    expression position (subqueries).

    Implementation note: passes sep=" " into _print_select so the
    clause join character itself is the space. Earlier versions did
    `_print_select(s).replace("\\n", " ")`, which would have corrupted
    any embedded TEXT literal that contained a newline. The literal
    generator never produces such strings today, so the bug was
    latent — the parameterized sep approach prevents it without
    relying on the literal pool's restraint."""
    return _print_select(s, sep=" ")


def _wrap_if_compound(e: Expr) -> str:
    """Return the rendered expression, wrapped in parens if it's a
    compound (binary, unary, or cast).

    Conservative: always wrap. The verbosity is preferable to a subtle
    precedence bug. The pglast round-trip tests are the safety net that
    keeps us honest.
    """
    s = _print_expr(e)
    if isinstance(e, (BinaryOp, UnaryOp, Cast)):
        return f"({s})"
    return s


# ===========================================================================
# Literal rendering
# ===========================================================================

def _render_literal(value: int | float | str | bool | None, t: PgType) -> str:
    """Render a Python value as a typed SQL literal.

    NULL always renders with an explicit cast (`NULL::int4`). Bare
    untyped NULL has occasionally-surprising behavior during PG's
    function-overload resolution, so the cast keeps the query
    well-typed regardless of where it appears.
    """
    if value is None:
        return f"NULL::{t.sql()}"

    if t == BOOL:
        # PostgreSQL accepts `true`/`false` as boolean constants; use
        # them in lowercase to match the rest of our keyword style.
        return "true" if value else "false"

    if t in (INT4, INT8):
        # Integer literals print bare. We never generate negatives at
        # the literal level — the operator generator can produce them
        # via `0 - x` or `UnaryOp(-, ...)` if needed.
        return str(int(value))

    if t in (NUMERIC, FLOAT8):
        # Force a decimal point so PG's lexer treats it as a numeric
        # literal, not an integer that happens to fit. `repr(2.0)` →
        # `'2.0'`, but `str(2.0)` → `'2.0'` too — they agree on floats.
        s = repr(float(value))
        return s if "." in s or "e" in s or "E" in s else s + ".0"

    # Catch-all for string-quoted typed literals: text/varchar (always
    # cast — see below), date/time/uuid/jsonb (need the cast for PG's
    # parser to accept the string body as the right type), and any
    # future scalar that doesn't have its own bare-form rendering above.
    #
    # Why text/varchar always carry the cast too: bare text literals
    # are typed 'unknown' by PG until inferred from context; in
    # polymorphic contexts (jsonb_build_object's VARIADIC "any",
    # coalesce of all-bare-strings, etc.) PG can't infer them and
    # errors with 42804 "could not determine polymorphic type because
    # input has type unknown." The explicit cast pre-empts the
    # inference entirely. Cost: noisier output. Benefit: closes a
    # whole PARSE-tier leak class. (Track A, commit 731015a.)
    #
    # The caller is responsible for using a string body PostgreSQL
    # can parse for the target type (e.g. ISO 8601 for dates).
    return f"{_quote_string_literal(str(value))}::{t.sql()}"


def _quote_string_literal(s: str) -> str:
    """Wrap `s` as a PostgreSQL standard-conforming string literal.

    Doubles embedded single quotes; assumes no other escaping is
    needed because our literal generator pulls from a curated word
    list with no backslashes or non-printable characters. This
    assumption is documented at the call site in the literal
    generator; if it's ever violated, the safest fix is to switch to
    PG's E-string form (`E'...'`) and add `\\` escaping.

    Standard-conforming (single-quote-doubled) rather than E-string
    is chosen so output is portable across servers regardless of
    `standard_conforming_strings` GUC setting — that GUC has been on
    by default since PG 9.1 but explicit single-quote-doubling works
    universally without depending on it.
    """
    return "'" + s.replace("'", "''") + "'"


__all__ = ["print_query", "print_expr"]
