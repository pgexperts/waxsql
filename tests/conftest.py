"""Shared fixtures for the live-DB validation tests.

`pg_conn` and `install_and_check` are used by both `test_parse.py`
and `test_plan.py`; extracted here to remove the previous duplication.

DELIBERATELY no `pytest.importorskip("psycopg")` at module top —
that would skip the conftest itself, which would prevent collection
of every other test file in the directory. The skip happens inside
the `pg_conn` fixture body, which only runs when something requests
the fixture.
"""
from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from typing import Any

import pytest

from waxsql import generate_query, generate_schema, print_query
from waxsql.validate.parse import install_schema


# Default DSN works on any libpq install with a `waxsql_test` database.
# Override via the env var; libpq's standard PG* variables are also
# respected because we pass the DSN directly to psycopg.connect().
_DSN = os.environ.get("WAXSQL_PG_DSN", "dbname=waxsql_test")


@pytest.fixture(scope="session")
def pg_conn() -> Any:
    """A long-lived PG connection used by PARSE-tier and PLAN-tier
    tests. Skips the requesting test if psycopg or PG aren't available.

    Session-scoped so the connection is reused across tests; each
    `install_and_check` invocation wraps its work in a transaction
    that gets rolled back, so cross-test isolation is per-transaction
    rather than per-connection.
    """
    psycopg = pytest.importorskip("psycopg")
    try:
        conn = psycopg.connect(_DSN, autocommit=False)
    except psycopg.Error as e:
        pytest.skip(f"no PG connection at {_DSN!r}: {e}")
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def install_and_check(pg_conn: Any) -> Callable:
    """Factory: returns a `runner(seed, complexity, check_fn)` callable.

    Each call generates a fresh schema + query at the given (seed,
    complexity), opens a transaction, installs the schema, runs
    `check_fn(sql, conn)`, then rolls back. Returns `(result, sql)`.

    The fixture-factory shape (returning a callable that takes the
    check function as a parameter) is what lets PARSE and PLAN tests
    share the same install/transaction shape — the only difference
    between them is which `check_fn` runs against the prepared
    schema. Caller passes `check_parse` or `check_plan`.

    Session-scoped because it captures `pg_conn` (which is also
    session-scoped). The captured connection is reused; the
    per-invocation transaction provides isolation.
    """
    def runner(seed: int, complexity: int, check_fn: Callable) -> tuple[Any, str]:
        schema = generate_schema(seed=seed, complexity=complexity)
        q = generate_query(seed=seed, schema=schema, complexity=complexity)
        sql = print_query(q)
        # No explicit BEGIN: psycopg (autocommit=False) opens the
        # transaction implicitly when install_schema issues its first
        # DDL statement. Explicit BEGIN here would produce the spurious
        # "there is already a transaction in progress" warning that
        # cli.py:626-629 explicitly calls out as the psycopg antipattern.
        try:
            install_schema(schema, pg_conn)
            result = check_fn(sql, pg_conn)
        finally:
            # Suppress so a poisoned/dead connection (e.g. a server-side
            # crash mid-check) can't turn teardown into a second exception
            # that masks the real test failure. The session-scoped
            # connection is reset best-effort here; a genuinely dead
            # connection would surface on the next test that requests it
            # (acceptable — that class of failure is closed today, see #23).
            with contextlib.suppress(Exception):
                pg_conn.rollback()
        return result, sql
    return runner
