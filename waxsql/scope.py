"""Binding stack for query generation.

A `Scope` answers two questions for the expression generator:

  1. "Which columns can I reference here?" — `visible_columns(of_type)`.
  2. "Which tables are in the current FROM clause, and under what
     aliases?" — `aliased_tables()` (used by the SELECT generator to
     bias JOIN conditions toward FK-related tables).

Scopes form a parent chain. A nested subquery's scope points at the
outer query's scope; whether the lookup walks up the chain depends on
whether the subquery is correlated (and, equivalently for FROM-clause
subqueries, whether it's LATERAL). This is the same mechanism that
PostgreSQL's parser uses internally for name resolution; modelling it
the same way means the generator's notion of "what's visible" matches
PostgreSQL's notion when we eventually wire up PARSE-level validation.

The aggregate / GROUP BY / window flags do NOT live here. Those are
expression-context state (they flip per-call inside an aggregate
argument, etc.) and belong on GenContext. Scope is purely about
binding visibility.

There's no explicit pop. Scopes nest via parent pointers, and a
"popped" scope is one the caller simply stops referencing — typically
by returning from the function that built it, or by replacing the
GenContext.scope field with the parent. This is enforced structurally
because GenContext is frozen: `descend_subquery` produces a NEW
GenContext with a child scope, and once that GenContext goes out of
scope, the child scope does too. The discipline is "every scope
borrowed for a subquery is local to that subquery's generation call".
Violating this means stale bindings leak into sibling generation —
the prototypical "I see columns from a sibling I shouldn't see" bug.

Two related visibility mechanisms:

  * CTE table-level visibility. A `_cte_defs` dict on each scope holds
    CTE definitions; CTE lookup walks the chain unconditionally — CTEs
    are visible regardless of correlation.

  * Subquery support. `push_subquery(correlated=...)` creates a child
    scope; the `correlated` flag at construction time decides whether
    parent-chain column lookups walk past this level.

What's deliberately not modeled:

  * Nullability propagation through outer joins. Bindings carry their
    declared `nullable` flag, but the generator currently treats
    everything as potentially-NULL anyway. Refining this requires
    join-tree analysis at generation time and is its own piece of work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .schema import Table
from .types import PgType, implicitly_castable


@dataclass(frozen=True)
class Binding:
    """One column visible at a particular scope level.

    `table_alias` matches the alias used in the FROM clause (not the
    underlying table name) — see the printer's ColumnRef handling for
    why we always reference columns through the alias.
    """
    table_alias: str
    column: str
    type: PgType
    nullable: bool


class Scope:
    """Mutable binding container with an immutable parent link.

    A fresh Scope has no bindings; the caller adds them as it processes
    the FROM clause. Use `push_subquery(...)` to create a child scope
    when entering a subquery; the parent is kept alive (and visible,
    if `correlated=True`) for the lifetime of the child.

    The `correlated` flag is set once at construction and not changed:
    it reflects what kind of nesting created this scope. For LATERAL
    FROM-subqueries and most expression-position subqueries, pass
    `correlated=True`; for non-LATERAL FROM-subqueries that may not
    reference their siblings, pass `correlated=False`.

    Scope is NOT frozen, by deliberate exception to the project-wide
    immutability convention. The reason: binding lists grow over a
    single FROM-clause pass — `add_table` is called once per FROM
    item — and rebuilding the whole Scope object for each addition
    would force the caller into an awkward fold-style loop. The
    mutation is confined to one writer (the FROM-clause builder) per
    scope instance, and the parent link is set in __init__ and never
    changed, so the parent-chain walk remains effectively immutable.
    """

    def __init__(
        self,
        parent: Optional["Scope"] = None,
        *,
        correlated: bool = True,
    ) -> None:
        self._parent = parent
        self._correlated = correlated
        # Bindings in insertion order. The expression generator picks
        # weighted-randomly from this list, so the order has to be
        # stable across runs. Insertion order is naturally stable when
        # `add_table` / `add_derived` is called in a deterministic
        # sequence.
        self._bindings: list[Binding] = []
        # alias -> Table-or-None, in insertion order. Base-table
        # aliases get their Table; derived-table aliases get None
        # (derived tables don't have an underlying base Table to
        # carry, but the alias still occupies a slot for collision
        # detection and for ordered enumeration).
        #
        # Used by the SELECT generator's join-condition FK biasing
        # (which filters out the derived-None entries since FKs only
        # exist on base tables).
        self._aliases: list[tuple[str, Optional[Table]]] = []
        # CTE definitions visible at this scope level. Maps CTE name
        # to its column info — list of (col_name, col_type) pairs.
        # Bindings aren't stored here because the table_alias on a
        # binding depends on the LOCAL alias used in a future CteRef,
        # which doesn't exist yet at CTE-definition time. The CteRef-
        # add-to-scope step constructs Bindings with the right alias
        # via `add_derived`.
        #
        # Lookup walks the parent chain UNCONDITIONALLY — CTEs are
        # visible from any nested scope regardless of `_correlated`.
        # That flag gates column visibility, not CTE visibility:
        # those are two separate static-scoping rules in PG.
        self._cte_defs: dict[str, list[tuple[str, PgType]]] = {}

    # -- mutation -----------------------------------------------------------

    def add_table(self, alias: str, table: Table) -> None:
        """Register `table` under `alias` and add all its columns as
        bindings visible at this scope level.

        Aliases must be unique within a single Scope (PostgreSQL
        enforces this for FROM-clause aliases). The check is intended
        to catch generator bugs, not to police user input. Both base
        and derived aliases participate in the uniqueness check.
        """
        if any(a == alias for a, _ in self._aliases):
            raise ValueError(f"alias {alias!r} already in scope")
        self._aliases.append((alias, table))
        for col in table.columns:
            self._bindings.append(Binding(
                table_alias=alias,
                column=col.name,
                type=col.type,
                nullable=col.nullable,
            ))

    def add_derived(
        self,
        alias: str,
        columns: list[tuple[str, PgType]],
    ) -> None:
        """Register a derived-table alias whose columns come from a
        FROM-clause subquery's targets, not a base Table.

        `columns` is a list of (column_name, column_type) pairs —
        typically one entry per inner SELECT target. The generator
        uses synthetic column names (`c1`, `c2`, ...) on the inner
        targets and passes those names here; that keeps column
        resolution `derived.c1` deterministic regardless of what the
        inner expression evaluated to.

        Derived columns are always treated as nullable: we don't
        propagate NOT NULL constraints through SELECT-list
        expressions, consistent with the generator-wide rule that
        treats everything as potentially-NULL.
        """
        if any(a == alias for a, _ in self._aliases):
            raise ValueError(f"alias {alias!r} already in scope")
        self._aliases.append((alias, None))
        for col_name, col_type in columns:
            self._bindings.append(Binding(
                table_alias=alias,
                column=col_name,
                type=col_type,
                nullable=True,
            ))

    # -- queries ------------------------------------------------------------

    def local_bindings(
        self,
        of_type: Optional[PgType] = None,
    ) -> list[Binding]:
        """Bindings introduced at THIS scope level only — no parent
        chain walk.

        Used by gen_expr when generating aggregate args inside a
        correlated subquery: outer-column refs inside such aggregates
        trigger PG's implicit-grouping inference on the outer query
        (PARSE-tier error 42803). Restricting to local bindings
        prevents the leak. The visible_columns method (which DOES
        walk the chain) remains the right tool everywhere else.
        """
        if of_type is None:
            return list(self._bindings)
        return [
            b for b in self._bindings
            if implicitly_castable(b.type, of_type)
        ]

    def visible_columns(
        self,
        of_type: Optional[PgType] = None,
    ) -> list[Binding]:
        """All bindings visible at this scope level.

        Walks up the parent chain when this scope is correlated;
        stops at the first uncorrelated scope. With `of_type` set,
        filters to bindings whose declared type implicitly casts to
        the requested type — same coercion rule the catalog uses.

        The returned list is freshly built on each call; the caller
        owns it and may freely sort or sample without affecting Scope.

        The walk order is innermost-first, then outward. Sample
        consumers (gen/expr.py column-ref candidate selection)
        weight earlier entries — i.e., closer scopes — implicitly
        through this ordering, which matches the intuition that
        "the current query's columns" are more relevant than
        "the enclosing query's columns" for a casual column pick.
        """
        out: list[Binding] = []
        s: Optional[Scope] = self
        while s is not None:
            if of_type is None:
                out.extend(s._bindings)
            else:
                out.extend(
                    b for b in s._bindings
                    if implicitly_castable(b.type, of_type)
                )
            if not s._correlated:
                break
            s = s._parent
        return out

    def aliased_tables(self) -> list[tuple[str, Table]]:
        """Return (alias, table) pairs from this scope only, BASE
        TABLES ONLY — derived-table aliases are filtered out.

        Used for FK-biased JOIN condition generation, which only
        applies to base tables (derived tables don't have FKs).
        Scope-local because the use case is within a single SELECT's
        FROM clause; parent-scope tables aren't JOIN candidates.
        """
        return [(a, t) for a, t in self._aliases if t is not None]

    def lookup_alias(self, alias: str) -> Optional[Table]:
        """Resolve `alias` to its underlying Table at this scope
        level. Returns None if the alias is absent OR if it's a
        derived-table alias (no underlying Table).

        The two None cases are deliberately collapsed because the
        only current caller (FK-biased JOIN-condition generation)
        treats both the same way: "can't generate an FK predicate
        against this." Use `has_alias()` for the unambiguous
        existence check.
        """
        for a, t in self._aliases:
            if a == alias:
                return t  # may legitimately be None for derived
        return None

    def has_alias(self, alias: str) -> bool:
        """True iff `alias` is registered at this scope level,
        regardless of base-table vs derived. Distinct from
        `lookup_alias`, which returns None for both "absent" and
        "present but derived"."""
        return any(a == alias for a, _ in self._aliases)

    # -- CTE management -----------------------------------------------------

    def add_cte(
        self,
        name: str,
        columns: list[tuple[str, PgType]],
    ) -> None:
        """Register a CTE definition under `name` with its column
        info. Same shape as `add_derived`'s columns parameter —
        list of (col_name, col_type) pairs.

        CTE name uniqueness is per-scope (one WITH clause). PG
        enforces this; reject early to surface generator bugs.
        Cross-scope shadowing (an inner WITH defining the same name
        as an outer) is allowed by PG but outside milestone-5 scope.
        """
        if name in self._cte_defs:
            raise ValueError(f"CTE name {name!r} already defined in scope")
        self._cte_defs[name] = list(columns)

    def lookup_cte(
        self,
        name: str,
    ) -> Optional[list[tuple[str, PgType]]]:
        """Resolve a CTE name to its column info, walking the parent
        chain UNCONDITIONALLY.

        Unconditional walk because CTE visibility is static-scope:
        a CTE defined in an outer query is visible from every nested
        SELECT, regardless of correlation/LATERAL semantics. That's
        different from `visible_columns`, which gates parent-chain
        walking on the `_correlated` flag.

        Returns a fresh list per call so callers can freely mutate it.
        Returns None when the name isn't found anywhere in the chain.
        """
        s: Optional[Scope] = self
        while s is not None:
            if name in s._cte_defs:
                return list(s._cte_defs[name])
            s = s._parent
        return None

    def has_visible_ctes(self) -> bool:
        """True iff at least one CTE is defined in this scope or any
        ancestor. Used by the FROM-clause generator to gate the
        "use a CTE reference?" decision — meaningless when no CTEs
        are in scope."""
        s: Optional[Scope] = self
        while s is not None:
            if s._cte_defs:
                return True
            s = s._parent
        return False

    def visible_cte_names(self) -> list[str]:
        """All CTE names visible at this scope level, walking the
        parent chain. Insertion-order within each scope, child
        scopes' CTEs first — de-duped so a name shadowed by a
        closer scope appears exactly once, with the child binding
        winning (because the child's copy is emitted before the
        walk reaches the parent's entry for that name).

        Today's generator only emits top-level WITHs, so the
        shadow case never fires in production; the dedupe is a
        latent-correctness guard for the eventual nested-WITH
        path. Without it, the list would contain the same name
        twice and a caller picking a CTE to reference by name
        could resolve the wrong binding — `lookup_cte` walks
        closest-first and would return the child's binding,
        producing a name/binding mismatch.

        Used when the generator needs to pick a CTE to reference
        from the FROM clause. Order is deterministic — both dict
        insertion order (Python 3.7+) and the parent-chain walk
        order are stable. The membership-check set is consulted
        only for `in`-tests, never iterated, so this does not
        violate the project's no-set-iteration-in-RNG-paths rule.
        """
        out: list[str] = []
        seen: set[str] = set()
        s: Optional[Scope] = self
        while s is not None:
            for name in s._cte_defs:
                if name not in seen:
                    seen.add(name)
                    out.append(name)
            s = s._parent
        return out

    # -- nesting ------------------------------------------------------------

    def push_subquery(self, *, correlated: bool) -> "Scope":
        """Construct a child scope for a nested query.

        `correlated=True` is the default for subqueries in expression
        position (e.g. `WHERE x = (SELECT ...)`) and for LATERAL FROM
        subqueries. `correlated=False` is for plain FROM subqueries,
        which by SQL standard cannot reference their siblings.

        Caller is responsible for not leaking the child scope past
        the subquery's generation call — see module docstring.
        Practically, that means the child is held only by a
        descended-then-discarded GenContext, never assigned to a
        long-lived attribute.
        """
        return Scope(parent=self, correlated=correlated)


__all__ = ["Binding", "Scope"]
