"""Complexity-dial configuration for query generation.

Parallel to `schema.SchemaConfig` / `schema_config_for_complexity`,
but for the query generator rather than the schema generator. The
philosophy is the same: a 0..10 dial maps onto a fully-specified
config, but the config can also be hand-built when the preset doesn't
match the use case.

Two separate knobs:

  * `feature_flags` GATE which features are *available*. Lower
    complexity unlocks fewer features (e.g. LEFT JOIN appears only
    at c >= 4). The generator checks `feature in cfg.feature_flags`
    before considering the feature at all.

  * Probability constants flavor *how often* available features
    fire. These don't slide with complexity — keeping them constant
    means dialing complexity changes the *shape* of generated queries
    (more joins, more select items, deeper exprs) rather than just
    the rate at which fixed features show up.

This split keeps "what does c=5 look like?" predictable: the dial
unlocks feature buckets; the buckets fire at fixed rates.
"""
from __future__ import annotations

from dataclasses import dataclass


# Recognized feature-flag names. Exposed as a constant so the
# generator's `if 'left_join' in flags:` checks are typo-resistant
# at the test level.
#
# Using string literals (not an Enum) keeps the feature_flags set
# trivial to dump/diff in test failures: a frozenset of plain
# strings prints readably under pytest -v. The cost is the
# named-constant discipline below — but mypy + the __all__ export
# keep typos from compiling.
FEATURE_INNER_JOIN = "inner_join"
FEATURE_LEFT_JOIN = "left_join"
FEATURE_WHERE = "where"
FEATURE_ORDER_BY = "order_by"
FEATURE_LIMIT = "limit"
FEATURE_AGGREGATE = "aggregate"
FEATURE_HAVING = "having"
FEATURE_SCALAR_SUBQUERY = "scalar_subquery"
FEATURE_EXISTS = "exists"
FEATURE_IN_SUBQUERY = "in_subquery"
FEATURE_DERIVED_TABLE = "derived_table"
FEATURE_LATERAL = "lateral"
FEATURE_CTE = "cte"
FEATURE_WINDOW_FUNCTION = "window_function"
FEATURE_SET_OP = "set_op"
FEATURE_RECURSIVE_CTE = "recursive_cte"
FEATURE_GROUPING_SET = "grouping_set"


@dataclass(frozen=True)
class ComplexityConfig:
    """Tunables for the query generator.

    Frozen for the same reason the rest of the model is frozen:
    immutability prevents one branch of generation from accidentally
    mutating a config field in a way that affects siblings, and the
    config is hashable for any future caching.
    """

    # ---- Structural caps -------------------------------------------------
    max_expr_depth: int
    """Maximum expression-tree depth. The expression generator
    decrements `depth_remaining` on each recursive call; at zero,
    only leaf productions (column ref, literal) are allowed."""

    max_from_items: int
    """Maximum number of tables in the FROM clause. The generator
    picks 1..max inclusive; with N > 1 items, joins (or comma cross
    products) connect them."""

    max_select_items: int
    """Maximum number of expressions in the SELECT list."""

    max_group_by_items: int
    """Maximum number of expressions in the GROUP BY clause for an
    aggregating query. Only consulted when the query is chosen to
    aggregate; for non-aggregating queries this field is ignored."""

    max_subquery_depth: int
    """Maximum nesting depth of subqueries (scalar / EXISTS / IN). The
    outermost query is depth 0; a subquery inside it is depth 1; a
    subquery inside that is depth 2. Counted separately from
    `max_expr_depth` because subquery nesting is a different
    rationing dimension — a deep `a + b * c + ...` expression
    shouldn't share a budget with `(SELECT ... WHERE x IN (SELECT ...))`.
    Set to 0 to disable subqueries even when the feature flag is set.

    Also covers CTE inner-SELECT nesting: each CTE definition's body
    consumes one level of subquery budget."""

    max_ctes_per_with: int
    """Maximum number of CTE definitions in a single WITH clause.
    Capped low (1..3 across the dial) so the WITH list stays readable
    in eyeballed output. The combinatorial blow-up in nested-query
    complexity comes from max_subquery_depth, not from CTE count."""

    # ---- Feature gates ---------------------------------------------------
    feature_flags: frozenset[str]
    """Set of FEATURE_* identifiers that are unlocked at this dial
    setting. The generator silently skips features whose flag is
    absent (no error — just doesn't generate them)."""

    # ---- Per-clause firing probabilities --------------------------------
    p_where: float
    """P(emit WHERE), conditional on FEATURE_WHERE being set."""

    p_order_by: float
    """P(emit ORDER BY), conditional on FEATURE_ORDER_BY being set."""

    p_limit: float
    """P(emit LIMIT), conditional on FEATURE_LIMIT being set."""

    p_explicit_join: float
    """P(use INNER/LEFT JOIN syntax over comma cross product) when
    multiple FROM items exist. Comma joins remain valid PG syntax,
    but explicit joins read more like real SQL."""

    p_left_join_when_explicit: float
    """When emitting an explicit join AND FEATURE_LEFT_JOIN is set,
    P(LEFT JOIN over INNER JOIN). At 0.0, every explicit join is
    INNER even if LEFT is unlocked."""

    p_aggregate_query: float
    """P(this query aggregates), conditional on FEATURE_AGGREGATE
    being set. Kept under 0.5 so non-aggregate queries remain the
    common case — both paths get exercised by the headline parse
    sweep."""

    p_having: float
    """P(emit HAVING), conditional on the query already aggregating
    AND FEATURE_HAVING being set. HAVING without aggregation is a
    PG syntax error; the gate inside gen_select enforces this."""

    p_derived_table_in_from: float
    """P(a given FROM item becomes a derived table), conditional on
    FEATURE_DERIVED_TABLE being set. Kept under 0.5 so most FROM
    items remain base tables — derived tables are a flavor, not the
    dominant shape, of real SQL."""

    p_lateral_when_derived: float
    """P(LATERAL prefix on a derived table), conditional on
    FEATURE_LATERAL being set AND the derived table being past the
    first FROM position (LATERAL on the first FROM item is a no-op
    — nothing precedes it to reference)."""

    p_with_clause: float
    """P(emit a WITH clause), conditional on FEATURE_CTE being set.
    Kept under 0.5 so non-CTE queries remain the common case —
    both paths get exercised by the headline parse sweep."""

    p_cte_in_from: float
    """P(a given FROM item becomes a CteRef rather than a base/derived
    table), conditional on at least one CTE being visible in scope.
    The flag check happens AFTER has_visible_ctes; useless without
    visible CTEs."""

    p_partition_by: float
    """P(a window spec includes a PARTITION BY clause). Most real-
    world window calls partition; the default is biased high."""

    p_order_by_in_window: float
    """P(a window spec includes an ORDER BY). Some functions (lag,
    lead, first_value, last_value, row_number) are typically used
    with ORDER BY; the default is biased high."""

    max_partition_by_items: int
    """Cap on the number of expressions in PARTITION BY. Kept low
    (1..2) so window specs stay readable."""

    max_order_by_in_window_items: int
    """Cap on the number of items in a window's ORDER BY. Same
    rationale as max_partition_by_items."""

    p_window_frame: float
    """P(a window spec includes an explicit frame clause like
    `ROWS BETWEEN N PRECEDING AND CURRENT ROW`). Independent of
    p_partition_by / p_order_by_in_window — frames can appear
    even on `OVER ()`. Real SQL most often omits the frame
    (PG's default RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT
    ROW is usually fine), so this is biased moderate-low."""

    max_derived_table_columns: int
    """Cap on the number of columns in a derived-table subquery
    (`(SELECT c1, c2, c3 FROM ...) AS dt`). Capped low so most
    derived tables stay single-column (the most common real-world
    shape) but multi-column shapes get exercised regularly."""

    max_cte_columns: int
    """Cap on the number of columns in a non-recursive CTE
    (`WITH cte AS (SELECT c1, c2, c3 FROM ...)`). Same rationale
    as max_derived_table_columns. Recursive CTEs stay single-
    column for now — the multi-column recursive case requires both
    arms to agree on each column type, more bookkeeping than the
    polish item warrants."""

    max_set_op_arms: int
    """Cap on the number of arms in a UNION/INTERSECT/EXCEPT.
    Capped low (2..3) so output stays readable; nested set ops
    are deferred to milestone 8+."""

    p_set_op_query: float
    """P(this query is a SetOp rather than a plain Select), gated
    on FEATURE_SET_OP. Kept under 0.5 so plain SELECTs remain the
    common case."""

    p_nested_set_op_arm: float
    """P(an arm of a SetOp is itself a SetOp). When non-zero,
    produces shapes like `A UNION (B INTERSECT C)` with the
    parens mandated by PG's set-op precedence (INTERSECT binds
    tighter than UNION/EXCEPT). Each nested SetOp consumes another
    `subquery_depth_remaining`, so arbitrary nesting is impossible."""

    p_grouping_set: float
    """P(an aggregate query's GROUP BY uses a ROLLUP/CUBE/GROUPING
    SETS extension instead of a flat column list), gated on
    FEATURE_GROUPING_SET. When firing, the grouped columns are
    wrapped in one of the three constructs uniformly. Plain
    column-list GROUP BY remains the common case so output stays
    readable."""

    p_set_op_all: float
    """When emitting a set op, P(use the ALL variant — `UNION ALL`
    etc.). The non-ALL form is the deduplicating variant."""

    p_recursive_when_cte: float
    """P(a CTE definition is recursive), conditional on
    FEATURE_RECURSIVE_CTE being set. Recursive CTEs are advanced;
    keeping this modest avoids drowning normal-CTE coverage."""

    # ---- Expression-generator weights -----------------------------------
    leaf_bias: float
    """Exponent applied to `depth_remaining / max_expr_depth` when
    weighting recursive productions. With leaf_bias=1.0 the bias
    decays linearly with depth; >1.0 favors leaves more aggressively
    (shorter expression trees on average); <1.0 favors recursion
    (taller but spiker trees)."""

    func_call_weight: float
    binary_op_weight: float
    column_ref_weight: float
    literal_weight: float
    aggregate_call_weight: float
    """Weight for an aggregate function call when it's a candidate
    in the expression generator (only when allow_aggregates is True
    and we're not already inside an aggregate). Tuned modestly so
    aggregates appear regularly in HAVING and inside expression
    arguments without dominating the candidate pool."""

    window_call_weight: float
    """Weight for a window-style function call (`func(args) OVER (...)`)
    when it's a candidate in the expression generator. Only eligible
    when allow_window=True (set by gen_select for SELECT-list expr
    generation) and not in_window (no nested windows)."""

    scalar_subquery_weight: float
    """Weight for a scalar subquery `(SELECT col FROM ...)` candidate
    in the expression generator. Eligible at any target type when
    the feature is unlocked and there's subquery-depth budget."""

    exists_weight: float
    """Weight for an `[NOT ]EXISTS (...)` candidate. Only eligible
    when the target type is BOOL and FEATURE_EXISTS is set."""

    in_subquery_weight: float
    """Weight for an `<expr> [NOT ]IN (...)` candidate. Only eligible
    when the target type is BOOL and FEATURE_IN_SUBQUERY is set."""


def query_config_for_complexity(complexity: int) -> ComplexityConfig:
    """Map a 0..10 complexity dial onto a ComplexityConfig.

    The dial unlocks features in stages:
      c == 0: trivial SELECT col FROM t1, no joins, no clauses
      c >= 1: WHERE, INNER JOIN unlock; max_from_items grows past 1
      c >= 2: ORDER BY, LIMIT unlock
      c >= 3: AGGREGATE unlocks (queries can do GROUP BY)
      c >= 4: LEFT JOIN, SCALAR SUBQUERY unlock
      c >= 5: HAVING, EXISTS, IN-SUBQUERY, DERIVED TABLE unlock
      c >= 6: LATERAL unlocks (only meaningful with derived tables)
      c >= 7: CTE unlocks (WITH clauses)
      c >= 8: WINDOW FUNCTION unlocks (`func() OVER (...)`)
      c >= 9: SET OP unlocks (UNION/INTERSECT/EXCEPT [ALL])
      c >= 10: RECURSIVE CTE unlocks (`WITH RECURSIVE ...`)

    Why complexity 0 emits flat `SELECT col FROM t`: the dial is
    intentionally a strict superset — every shape at level N also
    appears at every level > N. That property lets a user "lower
    the dial until the bug reproduces" without changing the qualitative
    character of what's generated. If c=0 produced something different
    in spirit from c=1, the dial would become a discontinuous
    classifier rather than a continuous knob.

    Why structural caps double-ish: each step roughly doubles the
    surface area the generator can hit at that level (`c // 2`,
    `c // 3`, etc.). Linear scaling would leave the top of the dial
    underpowered relative to the unlocked features; exponential would
    blow up parse-tree size and slow the test suite. Halving steps
    are a deliberate sweet spot for a <1s test suite.
    """
    # `c` is clamped — callers occasionally pass complexity from CLI
    # input or fuzz seeds and we'd rather degrade smoothly than raise.
    c = max(0, min(10, complexity))

    # The notch ordering below is load-bearing: features land in the
    # order they're typically built up in real SQL — predicates first
    # (WHERE, joins), then result shaping (ORDER BY, LIMIT), then
    # aggregation, then the more advanced clausal features (HAVING,
    # subqueries), then CTEs, then windows, then set ops. Reordering
    # changes which fixtures fire at each c-level and breaks
    # determinism guarantees that callers may depend on for golden
    # output. Add new features at the END of an existing notch when
    # possible; only introduce new notches when a feature is genuinely
    # gated on something earlier being unlocked first.
    flags: set[str] = set()
    if c >= 1:
        flags.add(FEATURE_WHERE)
        flags.add(FEATURE_INNER_JOIN)
    if c >= 2:
        flags.add(FEATURE_ORDER_BY)
        flags.add(FEATURE_LIMIT)
    if c >= 3:
        flags.add(FEATURE_AGGREGATE)
    if c >= 4:
        flags.add(FEATURE_LEFT_JOIN)
        flags.add(FEATURE_SCALAR_SUBQUERY)
    if c >= 5:
        # HAVING is grouped with the subquery batch because both rely
        # on the c=3 AGGREGATE unlock to produce useful output: HAVING
        # only fires on aggregating queries, and subqueries gain a lot
        # of expressive power once the aggregate path is live.
        flags.add(FEATURE_HAVING)
        flags.add(FEATURE_EXISTS)
        flags.add(FEATURE_IN_SUBQUERY)
        flags.add(FEATURE_DERIVED_TABLE)
    if c >= 6:
        # LATERAL is a no-op without DERIVED_TABLE; the c=6 placement
        # ensures derived tables exist by the time LATERAL unlocks.
        flags.add(FEATURE_LATERAL)
    if c >= 7:
        flags.add(FEATURE_CTE)
    if c >= 8:
        flags.add(FEATURE_WINDOW_FUNCTION)
    if c >= 9:
        flags.add(FEATURE_SET_OP)
    if c >= 10:
        # Top-of-dial features: recursive CTEs require the plain CTE
        # machinery (c=7) to already work, and grouping sets require
        # the AGGREGATE machinery (c=3). Both are kept off until the
        # full dial is in use because they exercise narrow PG paths
        # that swamp other coverage if they fire too readily.
        flags.add(FEATURE_RECURSIVE_CTE)
        flags.add(FEATURE_GROUPING_SET)

    return ComplexityConfig(
        # Structural caps grow with complexity. // 2 / // 3 / // 4
        # keep the numbers small so debug output stays human-readable
        # even at the high end.
        max_expr_depth=1 + c // 2,         # 1 .. 6
        max_from_items=1 + c // 3,         # 1 .. 4
        max_select_items=1 + c // 2,       # 1 .. 6
        max_group_by_items=1 + c // 4,     # 1 .. 3
        # Subquery depth: 0 below the unlock notch (subqueries gated
        # by feature flag so this doesn't fire anyway), then grows
        # slowly. At c=10 we allow 3 levels of nesting — deeper than
        # most hand-written SQL. The (c - 4) // 3 step is intentionally
        # gentler than max_expr_depth's c // 2: subquery nesting
        # combinatorially explodes parse-tree size in a way that flat
        # expression depth does not, so the budget grows in coarser
        # steps to keep generated SQL human-readable at the top of dial.
        max_subquery_depth=max(0, 1 + (c - 4) // 3) if c >= 4 else 0,
        # CTEs are capped low (1..3) so the WITH list stays readable
        # — depth complexity comes from max_subquery_depth, not count.
        max_ctes_per_with=max(1, 1 + (c - 7) // 2) if c >= 7 else 1,
        feature_flags=frozenset(flags),
        # Constant probabilities — see module docstring on why.
        p_where=0.7,
        p_order_by=0.5,
        p_limit=0.3,
        p_explicit_join=0.85,
        p_left_join_when_explicit=0.3,
        # Aggregating is intentionally a minority outcome (~35%) so
        # the non-aggregate path keeps getting exercised by every
        # parametrized parse sweep. HAVING fires roughly half the
        # time when the query is already aggregating.
        p_aggregate_query=0.35,
        p_having=0.4,
        # Derived tables: same minority-outcome reasoning. ~25% of
        # FROM items become derived; LATERAL fires ~half the time
        # when a derived table is past the first FROM position.
        p_derived_table_in_from=0.25,
        p_lateral_when_derived=0.5,
        # CTEs: ~30% of queries get a WITH clause when the feature
        # is unlocked; once a CTE exists in scope, a given FROM item
        # has a ~25% chance of being a CteRef (vs base/derived).
        p_with_clause=0.3,
        p_cte_in_from=0.25,
        # Window specs: most window calls use PARTITION BY and
        # ORDER BY in real SQL. Capped low (1..2) for readability.
        p_partition_by=0.7,
        p_order_by_in_window=0.7,
        max_partition_by_items=2,
        max_order_by_in_window_items=2,
        # Frames: ~30% of windows get an explicit frame. Most real
        # SQL omits frames (PG default is fine), but generating them
        # exercises the printer's frame path and PG's grammar
        # validation more thoroughly.
        p_window_frame=0.3,
        # Derived tables and non-recursive CTEs: 1..3 columns. Cap
        # is low enough that single-column remains the most common
        # output (the plurality, not the dominant majority); the
        # multi-column shape gets exercised in maybe a third of
        # derived/CTE emissions.
        max_derived_table_columns=3,
        max_cte_columns=3,
        # Set ops: 2 arms at unlock notch, growing to 3 at c=10.
        # Probability ~0.25 keeps SetOps a minority outcome so
        # plain SELECTs stay the common case.
        max_set_op_arms=2 if c < 10 else 3,
        p_set_op_query=0.25,
        p_set_op_all=0.5,
        # Nested set ops: low probability so most SetOps remain
        # the simple flat shape. When it does fire, exercises the
        # printer's mandatory-paren path for sub-SetOp arms.
        p_nested_set_op_arm=0.2,
        # Grouping sets: moderate probability. When firing, the
        # whole GROUP BY uses one of ROLLUP/CUBE/GROUPING SETS
        # (uniformly chosen), exercising the multi-grouping path
        # that PG's planner has its own handling for.
        p_grouping_set=0.3,
        # Recursive CTEs: ~30% of CTEs become recursive when the
        # feature is unlocked. Both forms (recursive and plain) get
        # exercised by every CTE-eligible parse sweep.
        p_recursive_when_cte=0.3,
        # Expression shape. Weights are relative — only ratios matter.
        # Column refs are weighted heavily because they're what makes
        # the query reference the schema at all; without enough column
        # refs, output is just literal arithmetic.
        leaf_bias=1.0,
        func_call_weight=2.0,
        binary_op_weight=2.0,
        column_ref_weight=4.0,
        literal_weight=1.0,
        aggregate_call_weight=2.0,
        # Subqueries are recursive productions on the same scale as
        # function calls. Modest weights so they appear regularly
        # without dominating — output should still mostly be ordinary
        # column refs and operators.
        scalar_subquery_weight=1.0,
        exists_weight=1.0,
        in_subquery_weight=1.0,
        # Window calls are heavyweight (the OVER clause adds visible
        # text); modest weight keeps them flavorful, not dominant.
        window_call_weight=1.5,
    )


__all__ = [
    "ComplexityConfig",
    "query_config_for_complexity",
    "FEATURE_INNER_JOIN", "FEATURE_LEFT_JOIN",
    "FEATURE_WHERE", "FEATURE_ORDER_BY", "FEATURE_LIMIT",
    "FEATURE_AGGREGATE", "FEATURE_HAVING", "FEATURE_GROUPING_SET",
    "FEATURE_SCALAR_SUBQUERY", "FEATURE_EXISTS", "FEATURE_IN_SUBQUERY",
    "FEATURE_DERIVED_TABLE", "FEATURE_LATERAL",
    "FEATURE_CTE",
    "FEATURE_WINDOW_FUNCTION",
    "FEATURE_SET_OP",
    "FEATURE_RECURSIVE_CTE",
]
