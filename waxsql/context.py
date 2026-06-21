"""Generation context — the bag of state threaded through every
generator call.

A `GenContext` holds:

  * `rng` — the random.Random instance. Shared across the whole
    generation run; advancing it in one call affects what later calls
    see, which is exactly what makes the (seed → output) mapping
    deterministic.

  * `scope` — the current binding scope. Determines which columns are
    visible at expression generation time.

  * `schema`, `catalog` — read-only references to the world. Passed
    through so generators can look up tables, FKs, functions,
    operators without re-importing globals.

  * `config` — the ComplexityConfig. Tells generators how aggressive
    to be (how many FROM items, how deep, which features are unlocked).

  * `depth_remaining` — the depth budget. Decremented on each
    recursive descent; at zero, generators are required to pick leaf
    productions only. This is what guarantees termination.

  * Expression-context flags (`in_aggregate`, `allow_aggregates`,
    `allow_window`, `allow_subquery`) — track what kind of expression
    we're inside, gating which production candidates the expression
    generator considers at each call site.

`GenContext` is frozen. To "modify" it (descend into a sub-expression,
swap scope for a subquery, flip an aggregate flag), use `replace(...)`
or one of the helper methods like `descend()`. Forcing all changes
through replace makes "use this for one descent, then revert" the
default control-flow shape, which prevents whole categories of
"forgot to reset the flag" bugs.

Why this is threaded explicitly through every call instead of being
held in a module-level / thread-local / context-var slot:

  * Testability. A test can build a GenContext directly with a
    hand-chosen rng/scope/config and call a generator function in
    isolation. Thread-local state would force tests through a
    setup-fixture-then-call shape that's harder to read and easier
    to leak between tests.

  * Parallelism. Multiple generation runs (e.g. parallel fuzzing
    workers in one process) can interleave without contending on
    a shared context slot. Each worker holds its own GenContext.

  * Reproducibility. The state that controls "what gets generated"
    is visible in the function signature. Reading a generator and
    tracing what's in scope doesn't require knowing about an
    ambient module-level variable that some other call mutated.

The cost is verbosity — `ctx` shows up in nearly every signature.
That's a price worth paying for the property that "given this ctx,
this function does X" is true regardless of where it's called from.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

from .catalog import Catalog
from .config import ComplexityConfig
from .schema import Schema
from .scope import Scope


@dataclass
class Counter:
    """Mutable monotonic counter used as a cell on the frozen GenContext.

    Replaces the bare `list[int]`-of-one idiom previously used for alias
    and CTE numbering. The point is type-visibility: `alias_counter:
    Counter` says "shared mutable counter" outright, and an accidental
    reset via `dataclasses.replace(ctx, alias_counter=Counter())` reads
    as obviously suspicious in a way `alias_counter=[1]` never did — that
    silent-reset footgun is exactly what this guards against. Determinism
    depends on the produced value sequence being identical across runs,
    so the only operation is a monotonic advance.
    """

    value: int = 1

    def take(self, n: int = 1) -> int:
        """Return the current value, then advance by `n` (default 1).

        The returned value names an alias or CTE; because `replace()`
        preserves this same Counter instance (not a copy), the
        post-advance state is immediately visible to sibling generators
        that share the context — which is what keeps sibling subqueries
        from picking colliding alias/CTE numbers.
        """
        current = self.value
        self.value += n
        return current


@dataclass(frozen=True)
class GenContext:
    """Generation state passed through the recursive generator calls.

    Frozen — see module docstring. The contained `rng` and `scope`
    objects are themselves mutable; that's intentional. The frozen
    wrapper protects against re-binding the field, not against the
    pointed-to object's natural mutation.
    """

    rng: random.Random
    scope: Scope
    schema: Schema
    catalog: Catalog
    config: ComplexityConfig
    depth_remaining: int

    # Expression-context flags. Defaults reflect "outermost SELECT
    # body": no aggregate context active, but aggregates are allowed
    # (in the SELECT list and HAVING; the SELECT generator flips the
    # `allow_aggregates` flag off when descending into WHERE).
    in_aggregate: bool = False
    allow_aggregates: bool = True
    allow_window: bool = False  # True in SELECT-list expr generation
    in_window: bool = False     # True inside a window FuncCall's args
    allow_subquery: bool = True
    # True iff the current ctx is a body of a correlated subquery
    # (or LATERAL derived table) — anything where the inner can see
    # outer columns via the parent-chain walk in scope. When True
    # AND in_aggregate=True, gen_expr restricts column-ref candidates
    # to the LOCAL scope only — outer-column refs inside aggregates
    # trigger PG's implicit-grouping inference (PARSE-tier 42803).
    in_correlated_subquery: bool = False
    # Whether a WITH clause may be generated at this gen_select call.
    # Top-level queries default to True; descend_subquery flips it
    # off so subqueries / derived tables / CTE bodies don't generate
    # their own nested WITHs (PG allows nested WITH but milestone 5
    # keeps WITH top-level only — defer nested to a later milestone).
    allow_with: bool = True

    # Subquery-nesting budget. 0 (the default) means "no subqueries
    # allowed", so code that builds a GenContext without thinking
    # about subqueries gets the safe behavior. The public
    # generate_query sets this from cfg.max_subquery_depth.
    #
    # Counted separately from depth_remaining because subquery
    # nesting is a different rationing dimension — a deep
    # `a + b + c` expression shouldn't share a budget with
    # `(SELECT ... WHERE x IN (SELECT ...))`.
    subquery_depth_remaining: int = 0

    # Mutable monotonic counter for FROM-clause alias generation.
    # Shared across all GenContexts in a single query (replace()
    # preserves the same Counter instance), so sibling subqueries
    # don't pick colliding aliases. See the Counter docstring for why
    # this is a dedicated type rather than a bare list-of-one cell.
    #
    # Generator-internal: not part of the public API. Tests that
    # build GenContext directly can ignore it.
    alias_counter: Counter = field(default_factory=Counter)

    # Companion monotonic counter for CTE-name generation. Separate
    # from alias_counter because CTE names live in a different
    # namespace (lookup_cte vs lookup_alias) — combining them would
    # interleave `cte1, cte5, ...` with `t2, t3, t4, t6, ...` for
    # no semantic benefit. Same shared-mutable-cell mechanism so
    # nested subqueries' CTE names don't collide either.
    cte_counter: Counter = field(default_factory=Counter)

    def descend(self, cost: int = 1) -> "GenContext":
        """Return a copy with `depth_remaining` decremented.

        Every recursive expression-generator call must descend by at
        least 1; that's what enforces termination. Higher-cost
        productions can pass `cost > 1` to penalize expensive shapes
        (e.g. function calls more than column refs); current callers
        all use the default cost of 1, but the parameter is there for
        when biasing depth budget by production cost becomes useful.

        Note this only decrements the expression-depth budget. Use
        `descend_subquery` when entering a SELECT body — that's a
        different rationing dimension and inherits-or-resets a
        different set of flags.
        """
        return replace(self, depth_remaining=self.depth_remaining - cost)

    def at_leaf(self) -> bool:
        """True iff the depth budget is exhausted; generators must
        pick leaf productions only.

        Centralizing this check makes the budget rule one line in
        the generator instead of an inline `<= 0` test scattered
        across files.
        """
        return self.depth_remaining <= 0

    def at_subquery_leaf(self) -> bool:
        """True iff no further subqueries can be opened. Used by
        gen_expr to gate subquery candidates the same way at_leaf
        gates recursive expression productions."""
        return self.subquery_depth_remaining <= 0

    def descend_subquery(self, *, correlated: bool) -> "GenContext":
        """Return a child context for entering a subquery body.

        The child context gets:

          * a fresh child scope (correlated/uncorrelated per arg),
            so inner FROM tables don't pollute the outer scope and
            outer columns are visible only when correlated;
          * decremented `subquery_depth_remaining`;
          * fresh `depth_remaining` — the subquery's expression tree
            starts with a full budget, not whatever's left of the
            outer expression's;
          * `allow_aggregates=True`, `in_aggregate=False` — the
            subquery is its own expression context, so the outer
            "no aggregates here" constraint doesn't propagate.

        The shared rng / schema / catalog / config remain shared.

        IMPORTANT — this method is the single chokepoint for entering
        a child query body. Every expression-context flag must be
        explicitly considered here: inheriting an outer flag through
        replace()'s "preserve unspecified fields" default is exactly
        the bug pattern this central method exists to prevent (e.g.
        an inherited allow_window=True would leak window functions
        into a subquery's WHERE clause). When adding a new flag to
        GenContext, decide here whether it propagates or resets
        before doing anything else.
        """
        return replace(
            self,
            scope=self.scope.push_subquery(correlated=correlated),
            subquery_depth_remaining=self.subquery_depth_remaining - 1,
            depth_remaining=self.config.max_expr_depth,
            allow_aggregates=True,
            in_aggregate=False,
            # No nested WITH inside a subquery — milestone 5 keeps
            # WITH top-level only.
            allow_with=False,
            # Window flags reset too. allow_window inherited as True
            # from a SELECT-list-context parent would leak windows
            # into the SUBQUERY's WHERE/HAVING/ON clauses (PARSE-tier
            # error 42P20 "window functions are not allowed in WHERE").
            # The subquery's gen_select will independently flip
            # allow_window=True for ITS own SELECT-list expressions.
            allow_window=False,
            in_window=False,
            # Track whether THIS ctx is a correlated-subquery body.
            # Used by gen_expr's aggregate-arg column-ref candidate
            # restriction (see in_correlated_subquery field doc).
            in_correlated_subquery=correlated,
        )


__all__ = ["GenContext"]
