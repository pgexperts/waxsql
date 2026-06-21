"""AST for generated SELECT queries.

Pure data — no rendering, no generation logic. The printer in `printer.py`
turns these into SQL; the generators in `gen/` turn random choices into
these. Splitting the pipeline this way keeps each stage testable in
isolation: the printer can be exercised against handcrafted ASTs, and
the generators can be exercised by inspecting their output AST without
parsing SQL back out.

Design notes:

  * Every `Expr` subclass carries its own `pg_type`. The generator
    knows the target type when it emits each node, so storing it on
    the node lets the printer (and future planners) reason about types
    without going back to the catalog.

  * All AST nodes are frozen dataclasses. Hashable, structurally
    comparable (handy for tests), and immune to "inner generator
    mutated my outer node" bugs.

  * `Expr` and `FromItem` are nominal marker bases (plain classes
    declaring the expected attributes). Concrete subclasses are the
    @dataclass(frozen=True). This avoids the dataclass-inheritance
    ordering rule (parents-before-children, defaults-after-non-
    defaults) that bites when the base wants required fields and a
    subclass wants to add more required fields.

  * The AST models the full SELECT-statement surface (GROUP BY,
    HAVING, OFFSET, Cast, UnaryOp, etc.) uniformly, whether or not
    every generator path emits each shape. A complete printer is
    easier to test than one with conditional branches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from .types import PgType


# Allowed value carriers for a SQL literal in our AST. The printer's
# `_render_literal` switches on `pg_type` to render the value; the
# value's Python type must be one of these. Date/timestamp/interval/
# UUID/JSONB literals carry their PG-textual form as `str` (see the
# `_DATE_LIT` / `_TIMESTAMPTZ_LIT` / etc. constants in `gen/expr.py`),
# so the union doesn't need datetime/uuid/Decimal members. Defined
# at module scope so callers (and printer.py's `_render_literal`)
# can refer to a single named type instead of inlining the union.
LiteralValue = int | float | str | bool | None


# ---------------------------------------------------------------------------
# Expression hierarchy
# ---------------------------------------------------------------------------

@runtime_checkable
class Expr(Protocol):
    """Structural protocol for expression nodes.

    Concrete expression types (`ColumnRef`, `Literal`, `FuncCall`,
    `BinaryOp`, `UnaryOp`, `Cast`, `Subquery`, `Exists`, `InSubquery`)
    are `@dataclass(frozen=True)` classes that each declare a
    `pg_type` field. `Expr` is a `Protocol` (PEP 544) rather than a
    plain base class so that mypy treats it as a *structural* type:
    any class with a `PgType`-typed `pg_type` attribute satisfies
    `Expr` where one is expected, regardless of nominal inheritance.

    Scope of the static check (read this if you're tempted to declare
    a new Expr subclass): Protocol attributes are NOT abstract. mypy
    rejects a *use* that fails the structural check — e.g., passing
    an instance without `pg_type` where an `Expr` is expected — but
    it does NOT force `class X(Expr):` itself to define `pg_type`.
    Forgetting the field in a subclass declaration will silently
    produce an under-typed class; only the call site catches it.
    If declaration-time enforcement becomes important, switch to
    `abc.ABC` with `@property @abstractmethod def pg_type`. dataclass
    fields satisfy abstract properties, so subclass shapes would not
    need to change. The Protocol form is preferred today because the
    structural check is strictly more permissive (free dataclasses
    not in this file can satisfy Expr without importing it) and the
    runtime cost is lower than ABC's metaclass machinery.

    `@runtime_checkable` keeps `isinstance(x, Expr)` working: the
    check verifies attribute presence rather than nominal
    inheritance — slightly more permissive than a nominal check
    (anything with a `pg_type` attribute counts as an Expr instance),
    but for the test suite's purposes ("did the generator return
    something with a typed AST shape?") structural is the right
    semantics.
    """
    pg_type: PgType


@dataclass(frozen=True)
class ColumnRef(Expr):
    """A qualified column reference: `alias.column`.

    We always qualify with the FROM-item alias rather than the raw table
    name. This keeps generated SQL unambiguous when the same table
    appears more than once (self-joins) and removes the need to track
    "which table did this column come from" outside the scope.
    """
    pg_type: PgType
    table_alias: str
    column: str


@dataclass(frozen=True)
class Literal(Expr):
    """A typed literal.

    `value=None` represents SQL NULL; the printer emits `NULL::<type>`
    to keep PostgreSQL from defaulting NULL's inferred type to `text`
    in surprising ways. Bare untyped NULL is a known footgun in
    PostgreSQL function-resolution code.
    """
    pg_type: PgType
    value: LiteralValue


@dataclass(frozen=True)
class FuncCall(Expr):
    """A function call: `name(arg1, arg2, ...) [OVER (...)]`.

    The printer specially renders the SQL-keyword "nullary functions"
    (current_date, current_timestamp, ...) without parentheses, since
    PostgreSQL's grammar treats those as keyword expressions, not
    function invocations — `current_date()` is a parse error.

    `over` is None for ordinary scalar/aggregate calls; a WindowSpec
    when the call is window-style (`func(args) OVER (...)`). Adding
    it as an optional field on FuncCall (rather than a separate
    WindowFuncCall node) keeps existing FuncCall callers unchanged
    — pre-milestone-6 construction sites pass nothing and get None.

    `star=True` represents the `name(*)` special form. In standard PG
    this is only valid for `count(*)` — every other aggregate rejects
    `*` as a placeholder. The generator enforces that restriction;
    the AST does not, because adding the constraint here would force
    a name-list dependency between ast.py and the catalog. The post-
    init check enforces only the invariants that are universally true
    regardless of which function is involved (no args when starred).

    `filter_` (Optional[Expr] returning BOOL) renders as a trailing
    `FILTER (WHERE <expr>)` clause. Only valid for aggregate
    functions in PG — `upper('x') FILTER (WHERE ...)` is a parse
    error. As with `star`, the AST doesn't validate the aggregate-
    only constraint (would require coupling to the catalog); the
    generator does. The trailing underscore avoids shadowing the
    `filter` builtin.

    `within_group` (tuple of OrderByItem, default empty) renders as
    `WITHIN GROUP (ORDER BY ...)`. Required syntactically for
    ordered-set aggregates (percentile_cont, percentile_disc, mode,
    hypothetical-set rank/dense_rank/etc.); rejected by PG on any
    other function. As with star/filter_, the AST trusts the
    generator to only set this on appropriate functions.

    Clause order in PG grammar: `name(args) [WITHIN GROUP (...)]
    [FILTER (WHERE ...)] [OVER (...)]`. The printer renders in that
    order; the AST fields are independent.
    """
    pg_type: PgType
    name: str
    args: tuple[Expr, ...]
    # `over` carries either an inline WindowSpec (the OVER (...) form)
    # or a WindowRef (OVER name, where name resolves against the
    # enclosing Select.windows). The named-window hoist pass in
    # gen/window.py rewrites inline specs into refs after generation.
    over: "WindowSpec | WindowRef | None" = None
    star: bool = False
    filter_: Optional[Expr] = None
    within_group: tuple["OrderByItem", ...] = ()

    def __post_init__(self) -> None:
        # `name(*)` and `name(arg, ...)` are mutually exclusive
        # syntactic forms — PG rejects `count(*, x)` etc. Catching
        # this at construction time prevents the printer from emitting
        # a malformed `count(*)` followed by mystery args.
        if self.star and self.args:
            raise ValueError(
                f"FuncCall(star=True) cannot have args; "
                f"got {len(self.args)} arg(s) for {self.name!r}"
            )


@dataclass(frozen=True)
class BinaryOp(Expr):
    """A binary operator application: `left SYMBOL right`.

    `symbol` is the literal operator text from the catalog ("+", "AND",
    "LIKE", "->>", ...). The printer always pads with spaces so the
    word-form operators don't collide with their operands.

    No precedence field: the printer parenthesizes operands
    conservatively rather than walking a precedence table. See
    printer.py `_wrap_if_compound` — emitting extra parens is harmless,
    missing one is a parse error, so the trade favors verbosity.
    """
    pg_type: PgType
    symbol: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class UnaryOp(Expr):
    """A prefix unary operator: `SYMBOL operand`.

    Used for NOT and unary minus/plus. Postfix unaries are not
    modeled — none of the operators in the catalog need them.
    """
    pg_type: PgType
    symbol: str
    operand: Expr


@dataclass(frozen=True)
class Cast(Expr):
    """An explicit type cast.

    Two SQL renderings exist (`expr::type` and `CAST(expr AS type)`);
    the printer picks the `::` form for compactness. The `pg_type`
    field intentionally duplicates `target_type`; it's there so every
    Expr satisfies the "has pg_type" invariant uniformly.
    """
    pg_type: PgType
    expr: Expr
    target_type: PgType


# ---------------------------------------------------------------------------
# Subquery expressions
# ---------------------------------------------------------------------------
#
# Three forms cover the bulk of subquery usage in real SQL:
#
#   * Subquery: `(SELECT col FROM ...)` in expression position.
#     Returns a single value; pg_type is the type of that value.
#
#   * Exists: `[NOT ]EXISTS (SELECT ...)`. Always BOOL. The inner
#     SELECT's targets are ignored at runtime — canonical idiom is
#     `SELECT 1`.
#
#   * InSubquery: `<expr> [NOT ]IN (SELECT col FROM ...)`. Always BOOL.
#     The inner SELECT must produce a single column whose type matches
#     `expr`'s type.
#
# Forward reference to Select via the string annotation; this works
# because the module uses `from __future__ import annotations`, so all
# annotations are lazily-resolved strings rather than runtime types.

@dataclass(frozen=True)
class Subquery(Expr):
    """Scalar subquery: `(SELECT col FROM ...)`.

    PG accepts a multi-row scalar subquery at parse time but errors at
    runtime if it returns more than one row. The generator pairs every
    Subquery with a `LIMIT 1` on the inner Select to be runtime-safe;
    the AST itself doesn't enforce that — it's the generator's job.
    """
    pg_type: PgType
    select: "Select"


@dataclass(frozen=True)
class Exists(Expr):
    """`[NOT ]EXISTS (SELECT ...)` test. Always BOOL.

    The inner SELECT's targets are semantically irrelevant — PG only
    tests whether the subquery yields any rows. Canonical idiom is
    `SELECT 1 FROM ...`; the generator emits exactly that shape.
    """
    pg_type: PgType  # always BOOL; field present for Expr-uniformity
    select: "Select"
    negated: bool = False


@dataclass(frozen=True)
class InSubquery(Expr):
    """`<expr> [NOT ]IN (SELECT col FROM ...)` test. Always BOOL.

    The inner SELECT must produce exactly one column whose type matches
    `expr.pg_type` (or implicitly casts to it). The generator enforces
    this; the AST doesn't validate it.
    """
    pg_type: PgType  # always BOOL
    expr: Expr
    select: "Select"
    negated: bool = False


# ---------------------------------------------------------------------------
# FROM clause
# ---------------------------------------------------------------------------

class FromItem:
    """Marker base for things that can appear in a FROM clause.

    Concrete subclasses: `TableRef`, `JoinExpr`, `DerivedTable`,
    `CteRef`. The hierarchy stays open for additional FROM kinds
    (table-valued functions, VALUES lists, etc.) without affecting
    existing subclasses.
    """
    pass


@dataclass(frozen=True)
class TableRef(FromItem):
    """A table reference with a mandatory alias.

    Always-aliased even when the alias equals the table name; this
    sidesteps the "is this column qualified by table name or alias"
    distinction in PostgreSQL's name resolution and keeps generated
    SQL self-consistent.
    """
    table: str
    alias: str


@dataclass(frozen=True)
class DerivedTable(FromItem):
    """A FROM-clause subquery: `[LATERAL ](SELECT ...) AS alias`.

    The inner SELECT acts as a virtual table; the outer query
    references its columns through `alias.<column>`.

    With `lateral=True`, the inner SELECT may reference aliases from
    preceding sibling FROM items (PG's LATERAL semantics). Without
    LATERAL, the inner is independent of all siblings — the same
    distinction as correlated vs uncorrelated subqueries in the
    expression position, applied to FROM.

    `column_aliases` corresponds to the optional `AS sq(a, b, c)`
    column-list syntax. Reserved here for later milestones; the
    milestone-4 generator leaves it empty and relies on the inner
    SELECT's target aliases instead.
    """
    select: "Select"
    alias: str
    column_aliases: tuple[str, ...] = ()
    lateral: bool = False


# Allowed JoinExpr.kind values. The set is the source of truth — keep
# the printer's switch and the generator's choices in sync with it.
JOIN_KINDS: frozenset[str] = frozenset({
    "INNER", "LEFT", "RIGHT", "FULL", "CROSS",
})


@dataclass(frozen=True)
class JoinExpr(FromItem):
    """A binary JOIN tree node.

    Multi-table joins build a left-deep tree (`((t1 JOIN t2) JOIN t3)`).
    For CROSS joins, both `on` and `using` are empty; for non-CROSS,
    exactly one of them is set.

    Invariant the generator must uphold (printer raises on violation):
    a non-CROSS join must have `on` set XOR `using` non-empty. Bare
    `LEFT JOIN t` with no qualifier is a parse error in PG.

    The printer is responsible for parenthesizing nested join trees
    correctly. We accept any nesting; the printer's job is to make it
    print parseably.
    """
    left: FromItem
    right: FromItem
    kind: str  # one of JOIN_KINDS
    on: Optional[Expr] = None
    using: tuple[str, ...] = ()


@dataclass(frozen=True)
class CteRef(FromItem):
    """A reference to a CTE in a FROM clause: `cte_name AS alias`.

    Same shape as TableRef but the source is a CTE definition (in
    the enclosing WITH) rather than a base table. Resolution of
    `cte_name` against a defined CTE is the generator's responsibility;
    the AST itself doesn't validate that the name exists.

    The local `alias` introduces column bindings derived from the
    CTE's inner SELECT targets — same way DerivedTable does, just
    that the inner SELECT lives in a sibling CteDef rather than
    here.
    """
    cte_name: str
    alias: str


# ---------------------------------------------------------------------------
# WITH clause / CTE definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CteSearch:
    """`SEARCH BREADTH|DEPTH FIRST BY col, ... SET seqcol` clause on
    a recursive CTE.

    Adds one synthetic column (`set_column`) to the CTE's exposed
    columns — the generator updates the scope's column list to
    include it. PG uses the synthetic column to expose the search
    order, allowing the outer query to ORDER BY it for reliable
    BFS/DFS traversal output."""
    breadth_first: bool  # True = BREADTH FIRST, False = DEPTH FIRST
    by_columns: tuple[str, ...]
    set_column: str


@dataclass(frozen=True)
class CteCycle:
    """`CYCLE col, ... SET cyclecol USING pathcol` clause on a
    recursive CTE.

    Adds two synthetic columns: `cycle_mark_column` (BOOL — true on
    rows where a cycle was detected) and `path_column` (an array of
    row tuples tracing the recursion path so far). Both get added to
    the CTE's exposed columns by the generator's scope-registration
    step. Used to defend against infinite recursion in graph walks.

    PG also accepts an extended form `SET cycle_mark TO val DEFAULT
    val2` for non-default cycle-detection markers; not modeled
    here (defaults TO TRUE / DEFAULT FALSE are what 99% of real
    queries want)."""
    columns: tuple[str, ...]
    cycle_mark_column: str
    path_column: str


@dataclass(frozen=True)
class CteDef:
    """One entry in a WITH clause:
    `name [(col1, col2, ...)] AS [MATERIALIZED|NOT MATERIALIZED] (SELECT ...)
     [SEARCH ...] [CYCLE ...]`.

    `column_aliases` is empty by default; reserved for the explicit-
    column-list syntax (same pattern as DerivedTable.column_aliases).

    `materialized` is None for PG's default behavior (which since
    PG 12 may inline single-use CTEs); True forces MATERIALIZED;
    False forces NOT MATERIALIZED. The milestone-5 generator leaves
    this at None — the printer emits no modifier in that case.

    `recursive` is True for recursive CTEs (added in milestone 8).
    The keyword `RECURSIVE` is per-WITH-list, not per-CteDef — the
    printer scans the WITH list and emits `WITH RECURSIVE` if ANY
    CteDef has recursive=True. The flag is per-CteDef here because
    that's where the generator decides "this one will self-reference."

    `select` widens to `Select | SetOp` in milestone 8: a recursive
    CTE's body is `base UNION ALL recursive` (a SetOp). Non-recursive
    CTEs continue to hold a plain Select.

    `search` and `cycle` are PG-specific recursive-CTE clauses, valid
    only when `recursive=True`. Each adds synthetic exposed columns
    (see CteSearch / CteCycle docstrings). Both can coexist on the
    same CTE — `... SEARCH ... CYCLE ...` is valid PG."""
    name: str
    select: "Select | SetOp"
    column_aliases: tuple[str, ...] = ()
    materialized: Optional[bool] = None
    recursive: bool = False
    search: Optional[CteSearch] = None
    cycle: Optional[CteCycle] = None


# ---------------------------------------------------------------------------
# SELECT clause pieces
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectTarget:
    """One item in the SELECT list: an expression with optional alias.

    No `SELECT *` form is modeled — the generator always projects an
    explicit target list. Star-projection collides badly with the
    type-driven pipeline (the projected types depend on the FROM
    items rather than being a property of the SELECT itself), and
    nothing downstream needs it."""
    expr: Expr
    alias: Optional[str] = None


@dataclass(frozen=True)
class OrderByItem:
    """One item in ORDER BY.

    `direction` is "ASC" or "DESC". `nulls` is "FIRST", "LAST", or None
    (let PostgreSQL apply its default: NULLS LAST for ASC, NULLS FIRST
    for DESC).

    Strings (not enums) for the same reason FrameClause.unit and
    GroupingSet.kind use strings: tiny fixed alphabet, mirrors the
    grammar tokens 1:1, the printer can splat them in directly. PG
    validates at parse time, so a stray value surfaces immediately
    via the round-trip test rather than silently mis-rendering.
    """
    expr: Expr
    direction: str = "ASC"
    nulls: Optional[str] = None


@dataclass(frozen=True)
class FrameBound:
    """One bound of a window frame (the `start` or `end` of a
    BETWEEN ... AND ... extent, or the sole bound of a single-bound
    extent).

    PG's grammar gives five choices:
      UNBOUNDED PRECEDING | <offset> PRECEDING | CURRENT ROW
      | <offset> FOLLOWING | UNBOUNDED FOLLOWING

    We model them via a string `kind` field — same convention as
    OrderByItem.direction. Strings keep printer dispatch trivial and
    avoid an enum import boilerplate. Valid kinds:
        "unbounded_preceding"
        "preceding"
        "current_row"
        "following"
        "unbounded_following"

    `offset` is the `<offset>` expression for preceding/following
    kinds (typically a non-negative integer literal); None for the
    unbounded kinds and CURRENT ROW. Post-init validates the
    pairing — preceding/following without offset (or unbounded/
    current_row WITH offset) would be malformed SQL.
    """
    kind: str
    offset: Optional[Expr] = None

    def __post_init__(self) -> None:
        needs_offset = self.kind in ("preceding", "following")
        has_offset = self.offset is not None
        if needs_offset and not has_offset:
            raise ValueError(
                f"FrameBound(kind={self.kind!r}) requires an offset"
            )
        if not needs_offset and has_offset:
            raise ValueError(
                f"FrameBound(kind={self.kind!r}) must not have an offset"
            )


@dataclass(frozen=True)
class FrameClause:
    """A window frame clause: the `ROWS BETWEEN ... AND ...` part
    that follows PARTITION BY and ORDER BY inside an OVER clause.

    PG grammar:
      { RANGE | ROWS | GROUPS } frame_extent [ frame_exclusion ]

    `unit` is "ROWS", "RANGE", or "GROUPS" (literal strings, mirroring
    the grammar tokens for printer convenience).

    `start` is the lower bound. `end` is the upper bound; when None,
    the printer emits the single-bound form `unit start`, which PG
    interprets as `unit BETWEEN start AND CURRENT ROW`. Both forms
    are valid; the explicit BETWEEN is more common and clearer.

    `exclude` is the EXCLUDE clause's body — one of "CURRENT ROW",
    "GROUP", "TIES", "NO OTHERS" — or None for the default (same
    as omitting the clause entirely). The printer prepends "EXCLUDE"
    so callers store just the body. Kept as a string rather than an
    enum for symmetry with `unit` and `direction` elsewhere; the
    set is small and validated by PG at parse time anyway.
    """
    unit: str
    start: FrameBound
    end: Optional[FrameBound] = None
    exclude: Optional[str] = None


@dataclass(frozen=True)
class WindowSpec:
    """The `OVER (PARTITION BY ... ORDER BY ... [frame])` clause
    attached to a window-style function call.

    Empty `partition_by` and empty `order_by` together produce the
    `OVER ()` form (entire result set as one partition). Some window
    functions (lag, lead, first_value, last_value) are typically used
    with ORDER BY but PG accepts them without it syntactically.

    `frame`, when set, is a structured FrameClause that the printer
    renders as `ROWS BETWEEN ... AND ...` (or RANGE/GROUPS variants).
    Was a raw string in earlier milestones; switched to structured
    representation when frame generation landed so the printer can
    guarantee well-formed output and the generator can compose
    bounds deterministically without string concatenation.
    """
    partition_by: tuple[Expr, ...] = ()
    order_by: tuple[OrderByItem, ...] = ()
    frame: Optional[FrameClause] = None


@dataclass(frozen=True)
class WindowRef:
    """Reference to a named window declared in the enclosing SELECT's
    WINDOW clause. Used as `OVER w` (no parens around the name).

    Modeled as a separate AST node from WindowSpec so the printer
    dispatches structurally without a name-vs-spec field hack on
    WindowSpec itself. FuncCall.over accepts either type:
    `Optional[WindowSpec | WindowRef]`.

    PG's grammar also allows `OVER (w PARTITION BY extra-col)` —
    a named window with inline extension. That form is not modeled
    yet; if needed it would be a third FuncCall.over option."""
    name: str


@dataclass(frozen=True)
class NamedWindow:
    """One entry in a SELECT's WINDOW clause: a name bound to a spec.

    Multiple FuncCall.over WindowRefs can reference the same name —
    that's the whole point of the WINDOW clause, deduplicating window
    specs across multiple aggregates. The Select.windows tuple
    declares all names visible from that SELECT's body."""
    name: str
    spec: WindowSpec


@dataclass(frozen=True)
class GroupingSet:
    """A grouping-set construct in a GROUP BY clause: ROLLUP, CUBE,
    or GROUPING SETS.

    PG grammar:
      ROLLUP ( expr_list_or_paren_list, ... )
      CUBE ( expr_list_or_paren_list, ... )
      GROUPING SETS ( ( expr_list ), ( expr_list ), ... )

    `kind` is the keyword: "ROLLUP", "CUBE", or "GROUPING SETS"
    (literal strings for printer convenience, mirroring `unit` on
    FrameClause and `direction` on OrderByItem).

    `elements` is a tuple-of-tuples. Each outer entry is one
    "element" of the grouping construct:

      * For ROLLUP/CUBE: each element is typically a single-expression
        tuple (`ROLLUP (a, b, c)`); multi-expr elements get parens
        (`ROLLUP ((a, b), c)` — first element is the compound (a,b)).

      * For GROUPING SETS: each element is one grouping set,
        rendered with explicit parens including the empty-tuple case
        (which becomes `()` — the grand-total grouping).

    The same expressions that appear in `elements` may be referenced
    by SELECT-list items; PG's "must appear in GROUP BY" rule is
    satisfied by structural equality (same Expr instance or
    equivalent frozen-dataclass value).
    """
    kind: str
    elements: tuple[tuple[Expr, ...], ...]


# ---------------------------------------------------------------------------
# Top-level statement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Select:
    """A SELECT statement.

    Field order roughly matches the SQL clause order — `targets`
    first because that's the SELECT body, then `from_` and any
    optional clauses. `with_ctes` is positionally after the required
    fields (the dataclass rule that defaulted fields follow non-
    defaulted ones), but the printer emits it as a `WITH ...` prefix
    AHEAD of SELECT in the output.

    `from_` is a tuple of FromItems that the printer joins with commas
    (a cross product). To express explicit JOINs, nest a JoinExpr
    inside a single from_ slot.

    `with_ctes` is empty by default; when non-empty, the printer
    prefixes the SELECT with `WITH cte1 AS (...), cte2 AS (...) ...`.
    Defaulting empty means callers that don't care about CTEs can
    construct a Select without thinking about the WITH list.
    """
    targets: tuple[SelectTarget, ...]
    from_: tuple[FromItem, ...]
    with_ctes: tuple[CteDef, ...] = ()
    where: Optional[Expr] = None
    # Each item is either a plain Expr (regular GROUP BY column) or a
    # GroupingSet (ROLLUP/CUBE/GROUPING SETS extension). The printer
    # dispatches on type. PG accepts mixing the two within one
    # GROUP BY: `GROUP BY a, ROLLUP (b, c)` is valid.
    group_by: tuple["Expr | GroupingSet", ...] = ()
    having: Optional[Expr] = None
    windows: tuple[NamedWindow, ...] = ()
    order_by: tuple[OrderByItem, ...] = ()
    limit: Optional[Expr] = None
    offset: Optional[Expr] = None


SET_OPS: frozenset[str] = frozenset({"UNION", "INTERSECT", "EXCEPT"})


@dataclass(frozen=True)
class SetOp:
    """A set operation combining N SELECT-style arms — UNION,
    INTERSECT, or EXCEPT — with optional ALL modifier.

    `arms` is a tuple of length 2+; each arm is a Select (nested
    SetOps deferred to milestone 8+). PG requires every arm to
    produce the same number of columns with implicitly-castable
    types; the generator enforces this by extracting the first
    arm's target types and forcing subsequent arms to match.

    `order_by` / `limit` / `offset` belong to the COMBINED result,
    not individual arms — PG's grammar requires per-arm ORDER BY/
    LIMIT to be parenthesized inside an arm, and milestone 7 keeps
    them at the SetOp level only.

    Lives at the same nesting level as Select — `Query.select` is
    typed `Union[Select, SetOp]` — so callers can hold a top-level
    query body uniformly without dispatching on shape at the
    wrapper layer.
    """
    op: str  # one of SET_OPS
    all: bool
    # Each arm is either a Select or a nested SetOp. Track B #5
    # added the nested case (`A UNION (B INTERSECT C)`); the printer
    # and the test-walking helpers all dispatch on isinstance.
    arms: tuple["Select | SetOp", ...]
    order_by: tuple[OrderByItem, ...] = ()
    limit: Optional[Expr] = None
    offset: Optional[Expr] = None


@dataclass(frozen=True)
class Query:
    """Top-level wrapper for a generated query.

    `select` (despite the name) holds either a Select or a SetOp.
    The name is kept for API continuity since milestone 1; treat
    it as "query body" semantically. The Union allows top-level
    UNION/INTERSECT/EXCEPT from milestone 7 onward without
    breaking the public Query type.
    """
    select: "Select | SetOp"


__all__ = [
    "Expr", "ColumnRef", "Literal", "FuncCall", "BinaryOp", "UnaryOp", "Cast",
    "Subquery", "Exists", "InSubquery",
    "FromItem", "TableRef", "JoinExpr", "DerivedTable", "CteRef", "JOIN_KINDS",
    "CteDef", "CteSearch", "CteCycle",
    "SelectTarget", "OrderByItem",
    "WindowSpec", "WindowRef", "NamedWindow", "FrameBound", "FrameClause",
    "GroupingSet",
    "Select", "Query",
    "SetOp", "SET_OPS",
]
