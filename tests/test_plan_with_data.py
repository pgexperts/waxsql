"""Live-DB tests for `validate --tier plan` with COPY-block data pre-loading.

Skipped automatically when there's no PG connection available. Mirrors
the gating in `tests/test_plan.py`.

The key integration being tested here is that `waxsql gen --with-data`
piped into `waxsql validate --tier plan` correctly:
  1. Parses the waxsql header to detect the with-data flag
  2. Regenerates the COPY blocks deterministically from header parameters
  3. Loads them into the live DB via psycopg's COPY API
  4. Runs ANALYZE so the planner sees real statistics
  5. Then runs EXPLAIN against the query

Without step 2-4, EXPLAIN would succeed but plan on empty tables;
with it, cost estimates reflect actual row counts and distributions.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest


# Skip the whole file if psycopg or a DSN aren't available — same pattern
# tests/test_plan.py uses today.
psycopg = pytest.importorskip("psycopg")

DSN = os.environ.get("WAXSQL_PG_DSN", "dbname=waxsql_test")

# Resolve the `waxsql` console script from the same Python environment
# that's running the test suite. This ensures the tests work whether or
# not the venv bin directory is on PATH (it's on PATH in an activated
# venv, but not when pytest is invoked via a full path like `/.venv/bin/python -m
# pytest`). We prefer the sibling-to-Python approach over hard-coding a
# path so the tests stay portable across CI setups.
_WAXSQL_BIN = os.path.join(os.path.dirname(sys.executable), "waxsql")

# The editable install in the venv may point to the main repo rather than
# this worktree. PYTHONPATH ensures the worktree's package is found first
# so the subprocess sees the same code the test runner uses. In a CI
# install without a separate worktree, this path entry is harmless (a
# directory that happens to also contain the installed package).
_WORKTREE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": _WORKTREE_ROOT}


@pytest.fixture
def have_db():
    """Probe the DB; skip the requesting test if the DB is unreachable.

    Using a fixture rather than a module-level skip means the test
    collection still succeeds even when no DB is available, which lets
    offline CI see which tests *would* run.
    """
    try:
        with psycopg.connect(DSN, autocommit=True) as c:
            with c.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception:
        pytest.skip(f"PG not available at {DSN!r}")


def _run_cli(args: list[str], input_text: str = "") -> subprocess.CompletedProcess:
    """Run the installed `waxsql` console script in a subprocess.

    Using the console script (not `python -m waxsql.cli`) validates the
    pyproject.toml entry-point registration end-to-end — the same path
    that CI's smoke test exercises. We derive the binary path from
    sys.executable so the tests work inside any virtual environment
    regardless of whether the venv's bin dir is on PATH.

    PYTHONPATH is set to the worktree root so the subprocess picks up
    the worktree's `waxsql/` package ahead of any stale editable-install
    mapping in the venv's site-packages finder.
    """
    return subprocess.run(
        [_WAXSQL_BIN, *args],
        input=input_text,
        capture_output=True,
        text=True,
        env=_SUBPROCESS_ENV,
    )


def test_plan_tier_with_data_loads_copy_blocks_and_analyzes(have_db):
    """Pipe `gen --with-data` into `validate --tier plan` end-to-end.

    This is the core M3 integration: the full pipeline from gen output
    containing COPY blocks through to a successful EXPLAIN with populated
    statistics. Exit code 0 means validate parsed the header, loaded the
    data, ran ANALYZE, and got a clean EXPLAIN result.
    """
    gen = _run_cli(
        ["gen", "--seed", "11", "--complexity", "3", "--with-data", "--rows", "10"],
    )
    assert gen.returncode == 0, gen.stderr

    validate = _run_cli(
        ["validate", "--tier", "plan", "--dsn", DSN, "-"],
        input_text=gen.stdout,
    )
    assert validate.returncode == 0, validate.stderr


def test_plan_tier_without_with_data_still_works(have_db):
    """Regression: existing plan-tier validation path is unchanged.

    Gen output without --with-data must still pass validate --tier plan
    exactly as before M3. The new code path only fires when the header
    contains with-data=true; missing or false header takes the old path.
    """
    gen = _run_cli(["gen", "--seed", "11", "--complexity", "3"])
    assert gen.returncode == 0

    validate = _run_cli(
        ["validate", "--tier", "plan", "--dsn", DSN, "-"],
        input_text=gen.stdout,
    )
    assert validate.returncode == 0, validate.stderr


def test_savepoint_recovery_after_failed_copy(have_db):
    """Pins the recovery contract the CLI depends on (cli.py:732-746).

    The validate-with-data path wraps each COPY block in a SAVEPOINT and,
    on `psycopg.Error`, issues ROLLBACK TO SAVEPOINT and RELEASE
    SAVEPOINT on THE SAME CURSOR that just failed. PostgreSQL allows
    `ROLLBACK TO SAVEPOINT` from `PQ_TRANS_INERROR` (that's its whole
    purpose — recovering from a failed statement), so the sequence is
    expected to work; but it was previously defensive code with no
    positive test coverage. If a future psycopg upgrade or PG server-
    side change silently broke this contract, the CLI's error handler
    would become unreachable: the recovery commands themselves would
    raise and mask the original COPY error, leaving the user with a
    "secondary failure" diagnostic instead of "this COPY block was
    malformed."

    The test mirrors the exact CLI sequence with a deliberately-bad
    COPY block (wrong column count) and asserts:
      1. The COPY raises psycopg.Error with the expected SQLSTATE.
      2. The same cursor can successfully execute the savepoint
         cleanup commands without raising.
      3. The surrounding transaction remains usable for subsequent
         statements (a SELECT against the temp table succeeds).
    """
    with psycopg.connect(DSN, autocommit=False) as conn:
        with conn.cursor() as cur:
            # Temp table scoped to this session; rolled back at end so
            # nothing leaks between test runs even if the same DSN is
            # reused.
            cur.execute(
                "CREATE TEMPORARY TABLE _wax_copy_recovery (id INT, name TEXT)"
            )

            cur.execute("SAVEPOINT _waxsql_copy")
            caught: psycopg.Error | None = None
            try:
                # Force the failure mode the comment in cli.py:743-744
                # talks about: a COPY block whose data doesn't match the
                # column count. The COPY statement declares 2 columns
                # (id, name) but the row supplies 3 tab-separated fields,
                # so PG raises "extra data after last expected column"
                # — SQLSTATE 22P04, bad_copy_file_format.
                with cur.copy(
                    "COPY _wax_copy_recovery (id, name) FROM STDIN"
                ) as copy:
                    copy.write("1\talice\textra_field\n")
            except psycopg.Error as e:
                caught = e
            assert caught is not None, "Malformed COPY did not raise as expected"

            # Confirm we caught the original COPY-format error, not a
            # secondary savepoint-cleanup failure. Either the SQLSTATE
            # is the canonical 22P04, or the text indicates the
            # column-count complaint — defensive check in case a
            # future PG version emits a different SQLSTATE.
            sqlstate = caught.diag.sqlstate if caught.diag is not None else None
            error_text = str(caught).lower()
            assert sqlstate == "22P04" or "extra data" in error_text, (
                f"Expected COPY-format error; got sqlstate={sqlstate}, msg={caught}"
            )

            # The contract: these four lines run on the SAME cursor
            # that issued the failed COPY. If psycopg/PG ever broke
            # this, one of these statements would raise, and the
            # CLI's user-facing diagnostic would silently regress.
            cur.execute("ROLLBACK TO SAVEPOINT _waxsql_copy")
            cur.execute("RELEASE SAVEPOINT _waxsql_copy")

            # Surrounding transaction is usable: a regular SELECT
            # succeeds and reports zero rows (the failed COPY didn't
            # insert anything because the SAVEPOINT rolled back).
            cur.execute("SELECT count(*) FROM _wax_copy_recovery")
            row = cur.fetchone()
            assert row is not None
            assert row[0] == 0, "Failed COPY should not have inserted any rows"
        conn.rollback()
