"""Plan-tier validation via EXPLAIN against a live PostgreSQL.

Role in the system: the strongest validation tier — strictly stronger
than PARSE in catch-rate (any PARSE failure is also a PLAN failure)
but roughly the same in cost on an empty schema, because EXPLAIN on
zero rows is mostly tree construction. The CLI surfaces PLAN as
`validate --tier plan`.

Runs PG's planner on the SQL — full parse-analysis (everything PARSE
catches) plus rewriting plus plan-tree construction. Catches the
sliver of issues that fail at planning time but not at parse-analysis:

  * Operator-class lookup failures for ORDER BY / DISTINCT /
    GROUP BY (e.g. types with `=` but no `<` ordering operator)
  * Some statistics-dependent type-coercion edges
  * Permission failures (auth happens at planning, not parse)
  * Cost-estimation issues that prevent plan-tree construction

Does NOT catch runtime errors (division by zero, type-cast failures
on actual data values, scalar-subquery cardinality violations).
EXPLAIN doesn't execute; for those, EXPLAIN ANALYZE would be needed
— and that requires data in the schema, which waxsql deliberately
doesn't generate.

Cost: roughly the same as PARSE on a small schema with no rows —
EXPLAIN is mostly statistics lookup and tree construction, both
cheap when there are no real rows to estimate against.

The savepoint-around-EXPLAIN pattern matches `parse.py`'s — a
single failing query rolls back JUST that statement, not the
schema setup or prior successes. Without the savepoint, the first
planner error would put the surrounding transaction into the
aborted state, and every subsequent EXPLAIN would error out before
PG even tried to plan it, losing all signal beyond the first
failure. Unlike PARSE, EXPLAIN doesn't create a prepared statement,
so no DEALLOCATE step is needed (the savepoint release is
sufficient cleanup).

Statistics note (per ARCHITECTURE.md Pillar 4): when the input header
carries `with-data=true`, the test harness is expected to load
COPY blocks and run ANALYZE BEFORE calling `check_plan`, so the
EXPLAIN here reflects real statistics rather than empty-table
defaults — without ANALYZE, plans collapse to seq-scans-only and
lose all signal for join order, index choice, etc. The data
loading and ANALYZE invocation live in the caller (CLI / harness),
not in this module; `check_plan` is intentionally just the EXPLAIN +
savepoint dance.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # See `validate/parse.py` for the rationale: TYPE_CHECKING-gated
    # import gives mypy the real type without forcing psycopg as a
    # runtime dep for SYNTAX-tier-only users.
    import psycopg


@dataclass(frozen=True)
class PlanResult:
    """The outcome of a single PLAN-tier check.

    `ok=True` means PG's planner produced a complete plan tree.
    `ok=False` carries the PG error message; `error_code` carries
    the SQLSTATE if available.

    Same shape as ParseResult so callers (and aggregation tests)
    can use the two interchangeably — a PlanResult is just a
    ParseResult that ran the planner too.
    """
    ok: bool
    error: Optional[str] = None
    error_code: Optional[str] = None


def check_plan(sql: str, conn: psycopg.Connection) -> PlanResult:
    """EXPLAIN the SQL against the live `conn`. Wraps the EXPLAIN
    in a savepoint so a planner failure rolls back JUST that
    statement, not the surrounding transaction.

    EXPLAIN goes through parse-analysis AND rewriting AND planning,
    so it catches everything PARSE catches plus planner-side issues.
    The query is NOT executed — `EXPLAIN ANALYZE` would do that, but
    runtime errors are out of scope here (waxsql generates queries
    against an empty schema, where most runtime errors don't fire
    because zero rows means no per-row evaluation).

    `conn` must be a psycopg connection in a transaction (autocommit
    off). The caller is responsible for connection lifecycle and
    schema setup — typically reusing `parse.install_schema`.

    Raises if psycopg isn't available.
    """
    try:
        import psycopg
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "PLAN validation requires psycopg. "
            "Install with: pip install 'waxsql[plan]'"
        ) from e

    # Savepoint name matches the parse.py pattern (fixed identifier,
    # not per-call-unique) for the same reason: psycopg serializes
    # statements on a connection, and the savepoint is released
    # before this function returns, so collision is impossible.
    with conn.cursor() as cur:
        cur.execute("SAVEPOINT _waxsql_plan")
        try:
            # Plain EXPLAIN — no ANALYZE, no BUFFERS, no FORMAT JSON.
            # We only care that planning succeeds; the plan text itself
            # is uninteresting (and discarded — psycopg buffers result
            # rows but doesn't fetch them unless asked).
            #
            # No EXPLAIN ANALYZE: that would execute the query, which
            # could throw runtime errors (div-by-zero, cast failures)
            # that are out of scope for this tier. It would also pay
            # for real execution on every check — orders of magnitude
            # more expensive than plan-only EXPLAIN.
            cur.execute(f"EXPLAIN {sql}")
        except psycopg.Error as e:
            # SQLSTATE provenance mirrors parse.py — `e.diag.sqlstate`
            # is the 5-char code (e.g. "42883" for undefined_function).
            # Captured BEFORE cleanup so a cleanup failure can't lose it.
            sqlstate = (
                e.diag.sqlstate if hasattr(e, "diag") and e.diag else None
            )
            # Planning failed — savepoint rollback discards the failed
            # statement. EXPLAIN doesn't create persistent state on
            # success either, so no DEALLOCATE needed in the success path.
            #
            # Suppressed for the same reason as parse.py: if the connection
            # died mid-EXPLAIN, the ROLLBACK itself raises and would
            # otherwise mask the original planner error we're reporting.
            with contextlib.suppress(Exception):
                cur.execute("ROLLBACK TO SAVEPOINT _waxsql_plan")
                cur.execute("RELEASE SAVEPOINT _waxsql_plan")
            return PlanResult(
                ok=False,
                error=str(e).strip(),
                error_code=sqlstate,
            )
        cur.execute("RELEASE SAVEPOINT _waxsql_plan")
    return PlanResult(ok=True)


__all__ = ["PlanResult", "check_plan"]
