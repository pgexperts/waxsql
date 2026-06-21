"""PARSE-tier validation tests.

These tests run generated queries through PG's parse-analysis (via
PREPARE) instead of just pglast's syntax check. They catch the
class of bugs where pglast happily accepts SQL that PG would
reject at parse time:

  * Aggregates in disallowed contexts (WHERE, JOIN ON, agg args)
  * Window functions in disallowed contexts
  * GROUP BY consistency violations
  * Type mismatches PG rejects but the cast graph permits
  * Set-op arms with mismatched column counts/types
  * Forward references / undefined column refs

Tests are skipped if no PG connection is available. Set the env
variable WAXSQL_PG_DSN (or rely on libpq's standard PG* env vars)
to point at a writable test database. The default DSN is
`dbname=waxsql_test`.

Each test runs in a transaction that gets rolled back at the end,
so the test database state is fully reset between runs. The schema
itself is set up once per (seed, complexity) inside that transaction,
then many generated queries are PREPARE'd against it within nested
savepoints — a single failing query doesn't roll back the schema.

Why this is a separate test file: PARSE-tier validation requires a
live PG and psycopg, both optional dependencies. SYNTAX-tier tests
(the rest of the suite) run anywhere and stay fast (~30s for
~9000 tests). This file is opt-in.

The `pg_conn` and `install_and_check` fixtures live in
`tests/conftest.py`; the latter is a factory that returns a callable
parameterized by the check function (here: `check_parse`).
"""
from __future__ import annotations

import pytest

# psycopg is required at PARSE-tier; skip the whole file if absent.
# (The conftest's pg_conn fixture also skips, but importorskip here
# avoids loading psycopg-dependent test code in the absence case.)
pytest.importorskip("psycopg")

from waxsql.validate.parse import check_parse  # noqa: E402


# ---------------------------------------------------------------------------
# Smoke test — confirm the pipeline works end-to-end
# ---------------------------------------------------------------------------

def test_parse_pipeline_works(install_and_check):
    """The simplest generated query (c=0) must parse cleanly through
    the live-PG pipeline. If this test fails, the validate/parse.py
    machinery itself is broken."""
    result, sql = install_and_check(0, 0, check_parse)
    assert result.ok, (
        f"PARSE failed for the simplest possible query.\n"
        f"Error: {result.error}\nSQL: {sql}"
    )


def test_prepared_statement_survives_savepoint_rollback(pg_conn):
    """Pins the empirical behavior the explicit DEALLOCATE in check_parse
    relies on: PostgreSQL does NOT remove a PREPARE when the enclosing
    savepoint is rolled back. Verified against PG 18.3 and long-standing
    across PG 12+. If this ever regresses (re-PREPARE succeeds instead of
    raising 42P05 duplicate_prepared_statement), the DEALLOCATE rationale
    in check_parse should be revisited — see issue #45.
    """
    import psycopg
    try:
        with pg_conn.cursor() as cur:
            cur.execute("SAVEPOINT _wax_p45")
            cur.execute("PREPARE _wax_p45_stmt AS SELECT 1")
            cur.execute("ROLLBACK TO SAVEPOINT _wax_p45")
            # Re-PREPARE the same name. If the rollback had removed the
            # prepared statement this would succeed; the documented
            # behavior is that it survives, so PG raises 42P05.
            survived = False
            try:
                cur.execute("PREPARE _wax_p45_stmt AS SELECT 1")
            except psycopg.Error as e:
                survived = (e.diag.sqlstate == "42P05") if e.diag else False
        assert survived, (
            "PREPARE was removed by savepoint rollback — the DEALLOCATE "
            "rationale in check_parse needs revisiting (#45)"
        )
    finally:
        # The failed re-PREPARE leaves the transaction in error state; a
        # full rollback resets the shared session connection. But the
        # prepared statement SURVIVES rollback (the very thing asserted),
        # so DEALLOCATE ALL clears it too — otherwise a same-process re-run
        # would hit 42P05 at the FIRST PREPARE and false-pass (#56).
        pg_conn.rollback()
        with pg_conn.cursor() as cur:
            cur.execute("DEALLOCATE ALL")


# ---------------------------------------------------------------------------
# Per-complexity PARSE rates
# ---------------------------------------------------------------------------
#
# These tests measure what fraction of generated queries pass PARSE
# at each complexity level. At 1.0 the rates are 100% across every
# complexity; the strict thresholds below catch any regression.
# Remaining edge cases (and longer-term direction) live in FUTURE.md
# under the type-system-hardening section.


@pytest.mark.parametrize("seed", range(20))
def test_milestone1_parses_cleanly(install_and_check, seed):
    """Milestone-1 queries (c=0..2: SELECT/FROM/JOIN/WHERE/ORDER BY/
    LIMIT, no aggregates, no subqueries) should ALL parse — no known
    PARSE-tier leaks at this level."""
    for c in (0, 1, 2):
        result, sql = install_and_check(seed, c, check_parse)
        assert result.ok, (
            f"PARSE regression at milestone 1 (seed={seed}, c={c}):\n"
            f"[{result.error_code}] {result.error}\nSQL:\n{sql}"
        )


def _measure_parse_rate(install_and_check, complexity: int, n_seeds: int) -> tuple[int, int, dict]:
    """Run n_seeds queries at given complexity, report (passed,
    failed, errors_by_sqlstate)."""
    n_pass = 0
    n_fail = 0
    by_code: dict = {}
    for seed in range(n_seeds):
        result, _sql = install_and_check(seed, complexity, check_parse)
        if result.ok:
            n_pass += 1
        else:
            n_fail += 1
            by_code.setdefault(result.error_code, 0)
            by_code[result.error_code] += 1
    return n_pass, n_fail, by_code


def test_milestone3_parse_rate_above_threshold(install_and_check):
    """At c=4 (subqueries unlocked) the PARSE rate is 100% after
    Track A type-system hardening. The strict 1.0 threshold flags
    ANY regression — at this complexity level there's no excuse
    for a generator-produced query to fail PARSE."""
    n_pass, n_fail, by_code = _measure_parse_rate(
        install_and_check, complexity=4, n_seeds=30,
    )
    rate = n_pass / (n_pass + n_fail)
    assert rate == 1.0, (
        f"PARSE rate at c=4 dropped to {rate:.2f} ({n_pass}/{n_pass+n_fail}); "
        f"errors by SQLSTATE: {by_code}"
    )


def test_milestone8_parse_rate_above_threshold(install_and_check):
    """At c=10 (every feature unlocked) the PARSE rate is 100% on
    multi-thousand-seed sweeps after the Track A type-system
    hardening pass (commits a6ccf17 + 731015a).

    The two fixes that closed everything:
      * Wrap function/operator args in explicit casts when the
        actual arg type doesn't equal the declared param type.
        Closes 42883 (operator-doesn't-exist via overload-
        resolution surprise) and 42725 (ambiguous function/op).
      * Always emit TEXT/VARCHAR literals with `::text` cast.
        Closes 42804 (polymorphic-type-unknown from bare literals
        in VARIADIC "any" contexts like jsonb_build_object).

    The strict 1.0 threshold flags ANY regression. If you see this
    test fail, find the new leak and decide whether to plug it
    (extend coerce_to_param_type or the printer's literal path) or
    deliberately downgrade this threshold with a documented exception.
    """
    n_pass, n_fail, by_code = _measure_parse_rate(
        install_and_check, complexity=10, n_seeds=30,
    )
    rate = n_pass / (n_pass + n_fail)
    assert rate == 1.0, (
        f"PARSE rate at c=10 dropped to {rate:.2f} ({n_pass}/{n_pass+n_fail}); "
        f"errors by SQLSTATE: {by_code}"
    )
