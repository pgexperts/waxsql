"""Query-generation entry points.

Role: the public surface of the `gen/` subpackage. Re-exports a flat
set of generator functions so callers (validators, tests, the CLI)
can `from waxsql.gen import gen_select` without touching the internal
file layout. The submodules themselves still import each other
directly — the re-exports here are for *external* convenience, not
intra-package wiring.

The package exposes a flat set of generator functions, one per
SQL surface. The two foundational ones are:

  * `expr.gen_expr(ctx, target_type)` — typed expression generator.
    The recursive core; every other generator that needs to emit an
    expression goes through this.

  * `select.gen_select(ctx)` — SELECT-statement generator. Picks a
    FROM clause, populates the scope, then asks `gen_expr` for each
    select-list / WHERE / ORDER BY expression.

The rest (`subquery`, `cte`, `setop`, `window`) are full peers, each
emitting a different SQL surface but all called from `gen_select` (or
recursively from each other) for their part of the tree.

CONTRACT: these generators MUTATE `ctx` as they run. The rng advances
deterministically (intentional — that's the determinism guarantee),
but `ctx.scope`, `ctx.alias_counter`, and `ctx.cte_counter`
also get mutated as tables are added to the FROM clause and aliases
are minted. Callers must NOT assume `ctx` is unchanged across a
generator call. The `descend_subquery` chokepoint on GenContext is
where children get a fresh scope so cross-arm mutations don't
collide; production code always goes through it.
"""
from .cte import gen_cte_def, gen_recursive_cte_def
from .expr import gen_expr, gen_literal
from .select import gen_select
from .setop import gen_setop
from .subquery import (
    gen_derived_table, gen_exists_subquery, gen_in_subquery,
    gen_scalar_subquery,
)
from .window import gen_window_spec

__all__ = [
    "gen_expr", "gen_literal", "gen_select",
    "gen_scalar_subquery", "gen_exists_subquery", "gen_in_subquery",
    "gen_derived_table",
    "gen_cte_def", "gen_recursive_cte_def",
    "gen_window_spec",
    "gen_setop",
]
