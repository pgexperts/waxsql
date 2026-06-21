"""Parse-tier validation via PREPARE against a live PostgreSQL.

Role in the system: the middle tier — strictly stronger than SYNTAX,
strictly cheaper than PLAN. Used when the test harness has a live
DB available and wants to catch the entire class of name/type
errors that pglast silently passes.

Runs full PG parse-analysis: column resolution, type checking,
aggregate-context rules, function lookup, view validity. Catches
the entire class of "PG accepts at SYNTAX tier but rejects at
parse-analysis" issues that pglast (libpg_query) silently passes:

  * Column refs to undefined aliases or non-existent columns
  * Aggregates in disallowed contexts (WHERE, JOIN ON, aggregate args)
  * Window functions in disallowed contexts (WHERE, HAVING, agg args)
  * GROUP BY consistency violations
  * Type mismatches (e.g. `int = uuid` with no implicit cast)
  * Set-op arms with mismatched column counts/types

What it does NOT catch:

  * Operator-class lookup failures for ORDER BY / DISTINCT /
    GROUP BY (deferred to PLAN tier — would need EXPLAIN)
  * Runtime errors (division by zero, type cast failures on
    actual data, etc.)

Cost: roughly 1ms per query on a local PG with the schema already
loaded. Schema setup itself is ~1ms per CREATE TABLE plus FK/index
DDL — typically 50-300ms total for a milestone-sized schema.

The savepoint-around-PREPARE pattern is what lets a single failing
query not poison the surrounding transaction's state (the schema
setup, prior PREPAREs, ...). Without it, the first parse error
would put the surrounding transaction into an aborted state and
every subsequent check in the batch would error with "current
transaction is aborted, commands ignored until end of transaction
block" — losing all signal beyond the first failure. The test
harness owns the connection and transaction; this module is a thin
wrapper around the PREPARE + savepoint dance.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # psycopg is an OPTIONAL runtime dep (only required when callers
    # actually use the PARSE tier); but for static type-checkers we
    # want the annotations to carry the real types. The TYPE_CHECKING
    # block is False at runtime (so no import happens for users of
    # SYNTAX-tier only) but True for mypy/pyright. The `from __future__
    # import annotations` above makes all annotations lazy strings,
    # so the runtime never tries to resolve `psycopg.Connection`.
    import psycopg

    from ..schema import Schema


@dataclass(frozen=True)
class ParseResult:
    """The outcome of a single PARSE-tier check.

    `ok=True` means PG accepted the SQL at parse-analysis time.
    `ok=False` carries the PG error message; `error_code` carries
    the SQLSTATE if the error came from psycopg's structured error
    object (mostly always available for parse failures).
    """
    ok: bool
    error: Optional[str] = None
    error_code: Optional[str] = None


def check_parse(sql: str, conn: psycopg.Connection) -> ParseResult:
    """PREPARE the SQL against the live `conn`. Wraps the PREPARE in
    a savepoint so a parse failure rolls back JUST that statement,
    not the surrounding transaction.

    Uses an anonymous-prepared-statement-then-rollback pattern:
    the PREPARE itself does parse-analysis; the savepoint rollback
    cleans up the prepared statement so the name is reusable for
    the next call.

    `conn` must be a psycopg connection in a transaction (autocommit
    off). The caller is responsible for connection lifecycle and
    schema setup.

    Raises if psycopg isn't available (the package is an optional
    `[parse]` extra).
    """
    try:
        import psycopg
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "PARSE validation requires psycopg. "
            "Install with: pip install 'waxsql[parse]'"
        ) from e

    # Savepoint name is a fixed identifier (`_waxsql_parse`) rather
    # than per-call-unique because PREPARE/savepoint pairs are strictly
    # sequential within a single check_parse call — the savepoint is
    # always released before this function returns, so name collision
    # across concurrent callers on the same connection is impossible
    # (psycopg serializes statements on a single connection anyway).
    with conn.cursor() as cur:
        cur.execute("SAVEPOINT _waxsql_parse")
        try:
            # PREPARE forces full parse-analysis at PREPARE time (this
            # is the whole point of using PREPARE rather than, say,
            # EXPLAIN — we only want to pay for parsing, not planning,
            # because the PLAN tier exists for the planner check).
            cur.execute(f"PREPARE _waxsql_check AS {sql}")
        except psycopg.Error as e:
            # `e.diag.sqlstate` is the 5-char SQLSTATE code (e.g.
            # "42703" for undefined_column). Far more useful for
            # programmatic filtering than the human-readable message.
            # Captured BEFORE cleanup so a cleanup failure can't lose it.
            sqlstate = (
                e.diag.sqlstate if hasattr(e, "diag") and e.diag else None
            )
            # Parse failed — savepoint rollback discards the failed
            # statement and any partial state. The ROLLBACK + RELEASE
            # ordering matters: ROLLBACK TO returns us to the pre-PREPARE
            # state, then RELEASE removes the (now-empty) savepoint frame
            # so the surrounding transaction's savepoint stack doesn't
            # accumulate dead entries across many failed checks.
            #
            # Suppressed: if the connection died mid-PREPARE, the ROLLBACK
            # itself raises — without suppression that secondary error
            # would mask the original parse failure we're trying to
            # report. The caller's outer transaction handling deals with a
            # truly-dead connection; here we just make sure the original
            # `e` is what propagates.
            with contextlib.suppress(Exception):
                cur.execute("ROLLBACK TO SAVEPOINT _waxsql_parse")
                cur.execute("RELEASE SAVEPOINT _waxsql_parse")
            return ParseResult(
                ok=False,
                error=str(e).strip(),
                error_code=sqlstate,
            )
        # PREPARE succeeded. PG's transaction handling does NOT remove a
        # PREPARE via savepoint rollback — verified empirically against
        # PG 18.3 (and long-standing behavior across PG 12+): after
        # `SAVEPOINT s; PREPARE p; ROLLBACK TO SAVEPOINT s`, re-PREPAREing
        # `p` fails with SQLSTATE 42P05 (duplicate_prepared_statement), so
        # the statement survived the rollback. See
        # tests/test_parse.py::test_prepared_statement_survives_savepoint_rollback.
        # Hence the explicit DEALLOCATE: on the success path we RELEASE
        # (not roll back) the savepoint, and RELEASE never undoes the
        # PREPARE, so without DEALLOCATE the name `_waxsql_check` would
        # persist and the next call's PREPARE would collide. Order:
        # DEALLOCATE before RELEASE — if we released first, the DEALLOCATE
        # would still work but we'd have a brief window where the savepoint
        # is gone but the prepared statement isn't, a worse failure mode if
        # an interleaved error occurs.
        cur.execute("DEALLOCATE _waxsql_check")
        cur.execute("RELEASE SAVEPOINT _waxsql_parse")
    return ParseResult(ok=True)


def install_schema(schema: Schema, conn: psycopg.Connection) -> None:
    """Execute the schema DDL against `conn`. The caller should run
    this once per (seed, complexity) inside a transaction that's
    later rolled back, so the schema doesn't persist between tests.

    `schema` is a waxsql.Schema; `conn` is a psycopg connection.

    This is the canonical high-level pattern WHEN THE CALLER HAS A
    `Schema` OBJECT — library users, the conftest test fixture, and
    the README examples. The CLI's `validate --tier {parse,plan}`
    path deliberately does NOT call this function: `_resolve_schema_
    source` returns a raw DDL string (because the `--schema-from
    FILE` source has no Schema object behind it), and the CLI
    executes that string directly via `cur.execute(ddl)`. Both
    patterns are first-class and produce the same on-disk effect;
    pick whichever matches what's in hand. If you're holding a
    Schema, use `install_schema(schema, conn)`. If you're holding
    a DDL string already, `cur.execute(ddl)` is fine. There is no
    correctness difference between them.
    """
    # Single execute() of the entire DDL string: psycopg sends the
    # whole multi-statement payload to PG, which parses each statement
    # in turn. If any statement fails, the surrounding transaction
    # aborts — but since the caller is expected to rollback at the
    # end of the test anyway, that's the desired behavior. No
    # savepoint here because schema setup is all-or-nothing: a
    # partial schema isn't a useful state for the test harness.
    ddl = schema.emit_ddl()
    with conn.cursor() as cur:
        cur.execute(ddl)


__all__ = ["ParseResult", "check_parse", "install_schema"]
