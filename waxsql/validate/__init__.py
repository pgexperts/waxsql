"""Validation modes for generated SQL.

Role in the system: the public surface for "how thoroughly do we
check this query?" Every CLI entry point and test takes a
ValidationMode and dispatches to the right submodule. Keeping the
enum here (separate from the implementations in `syntax.py`,
`parse.py`, `plan.py`) means callers can refer to a mode without
importing psycopg or pglast — those imports are deferred to the
submodule that actually needs them.

Three layers, each strictly stronger than the previous in BOTH
cost and catch-rate — pick the cheapest one that catches the
failure class you care about. The ordering (SYNTAX < PARSE < PLAN)
is load-bearing: anything PARSE catches, PLAN also catches, and
anything SYNTAX catches, the other two also catch. That's why a
test that fails at PARSE is automatically a failure at PLAN — the
tiers compose.

  SYNTAX  — parse via libpg_query (pglast). No DB needed. Catches every
            grammar error PostgreSQL itself catches but no name/type
            resolution. Microseconds per check.

  PARSE   — PREPARE against a live DB. Runs full parse analysis: name
            resolution, type checking, aggregate/GROUP BY rules, function
            lookup. Milliseconds per check. Implemented in `parse.py`.

  PLAN    — EXPLAIN against a live DB. Runs the full planner pipeline:
            parse-analysis + rewriting + plan-tree construction. Catches
            operator-class lookup failures (ORDER BY / DISTINCT / GROUP
            BY on types without comparison operators) and the subset of
            runtime errors PG can constant-fold at planning time
            (division by zero on literal divisors, etc.). Implemented
            in `plan.py`.
"""
from enum import Enum, auto


# Enum (not a string constant) so callers can't pass typos that fail
# silently — every dispatch path is forced through the typed match.
# `auto()` for values because nothing outside this module should
# depend on the integer identity; only the symbolic name is API.
class ValidationMode(Enum):
    # NONE: skip validation entirely. Reserved for future "generate-only"
    # callers (benchmarks, reproducer dumps); the test suite always runs
    # at SYNTAX or higher to keep generator bugs visible.
    NONE = auto()
    SYNTAX = auto()
    PARSE = auto()
    PLAN = auto()


__all__ = ["ValidationMode"]
