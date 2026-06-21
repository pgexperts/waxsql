"""PLAN-tier validation tests.

These tests run generated queries through PG's planner (via EXPLAIN)
instead of just PREPARE. They catch the sliver of bugs that pass
parse-analysis but fail at planning time:

  * Operator-class lookup failures for ORDER BY / DISTINCT / GROUP BY
  * Some statistics-dependent type-coercion edges
  * Constant-foldable runtime errors (division by zero, invalid
    substring lengths, NULL-key in jsonb_build_object, etc.) that
    PG evaluates eagerly when all inputs are constant

What PLAN-tier doesn't catch: actual runtime errors on real data.
EXPLAIN ANALYZE would, but waxsql generates against an empty schema
where most runtime errors don't fire.

Tests skip if no PG connection is available. Same DSN convention
as test_parse.py — the WAXSQL_PG_DSN env variable, default
`dbname=waxsql_test`.

The `pg_conn` and `install_and_check` fixtures are shared via
`tests/conftest.py`; the latter is a factory that returns a callable
parameterized by the check function (here: `check_plan`).
"""
from __future__ import annotations

import pytest

# psycopg is required at PLAN-tier; skip the whole file if absent.
pytest.importorskip("psycopg")

from waxsql.validate.plan import check_plan  # noqa: E402


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_plan_pipeline_works(install_and_check):
    """The simplest generated query must EXPLAIN cleanly."""
    result, sql = install_and_check(0, 0, check_plan)
    assert result.ok, (
        f"PLAN failed for the simplest possible query.\n"
        f"Error: {result.error}\nSQL: {sql}"
    )


# ---------------------------------------------------------------------------
# Per-milestone PLAN rates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
def test_milestone1_plans_cleanly(install_and_check, seed):
    """Milestone-1 queries (c=0..2) should ALL plan — no PLAN-tier
    leaks at this level. The base SELECT/FROM/JOIN/WHERE/ORDER BY/
    LIMIT shape doesn't reach any of the constant-fold-to-error
    patterns that catch us at higher complexity."""
    for c in (0, 1, 2):
        result, sql = install_and_check(seed, c, check_plan)
        assert result.ok, (
            f"PLAN regression at milestone 1 (seed={seed}, c={c}):\n"
            f"[{result.error_code}] {result.error}\nSQL:\n{sql}"
        )


def _measure_plan_rate(install_and_check, complexity: int, n_seeds: int) -> tuple[int, int, dict]:
    n_pass = 0
    n_fail = 0
    by_code: dict = {}
    for seed in range(n_seeds):
        result, _sql = install_and_check(seed, complexity, check_plan)
        if result.ok:
            n_pass += 1
        else:
            n_fail += 1
            by_code.setdefault(result.error_code, 0)
            by_code[result.error_code] += 1
    return n_pass, n_fail, by_code


def test_plan_rate_at_c4_above_threshold(install_and_check):
    """At c=4 (subqueries unlocked, no aggregates) the PLAN rate is
    100% on 200-seed sweeps — the simple expression shapes here
    don't reach any constant-foldable runtime-equivalent errors.
    Strict 1.0 threshold catches any regression."""
    n_pass, n_fail, by_code = _measure_plan_rate(
        install_and_check, complexity=4, n_seeds=30,
    )
    rate = n_pass / (n_pass + n_fail)
    assert rate == 1.0, (
        f"PLAN rate at c=4 dropped to {rate:.2f} ({n_pass}/{n_pass+n_fail}); "
        f"errors by SQLSTATE: {by_code}"
    )


def test_plan_rate_at_c10_above_threshold(install_and_check):
    """At c=10 (every feature unlocked) the PLAN rate is ~96% on
    multi-hundred-seed sweeps. The remaining failures are all
    constant-foldable runtime errors PG raises eagerly:

      * 22012 division by zero (~3-4%): an arithmetic expression
        constant-folds to a zero divisor. The literal-zero case is
        suppressed in gen_expr's op branch and gen mod() helper, but
        zero-folding through (x - x), (0 * y), etc. would require a
        constant-folding analysis pass to fully catch.

      * 22011 negative substring length (~0.2%): substr's length
        arg constant-folds negative.

      * 22023 jsonb_build_object NULL key (~0.2%): the first arg of
        jsonb_build_object can't be NULL; if our random generator
        picks NULL::text for the key slot, PG errors.

    These are runtime-equivalent errors that PG happens to catch at
    planning time via constant-folding. They'd be the same errors
    that EXECUTE would raise on actual data. Closing them fully
    would require either heavier static analysis or removing the
    relevant operators/functions from the catalog (excessive).

    Threshold 0.9 leaves headroom for seed variance. If you see
    new SQLSTATE codes appear, investigate before adjusting."""
    n_pass, n_fail, by_code = _measure_plan_rate(
        install_and_check, complexity=10, n_seeds=30,
    )
    rate = n_pass / (n_pass + n_fail)
    assert rate >= 0.9, (
        f"PLAN rate at c=10 dropped to {rate:.2f} ({n_pass}/{n_pass+n_fail}); "
        f"errors by SQLSTATE: {by_code}"
    )
