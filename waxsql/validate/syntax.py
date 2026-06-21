"""Syntax validation via pglast (libpg_query bindings).

Role in the system: the fast inner-loop validator used by the round-trip
test pattern (Pillar 3). Every parametrized test that generates SQL
pipes the output through `check_syntax` and asserts ok=True — this is
how the generator catches malformed token streams the moment a printer
or generator regression appears, without paying for a live PG.

This is the cheapest and most universally available validation layer:
it requires nothing beyond the pglast wheel — no PostgreSQL server, no
network, no per-query cost worth measuring. It catches the entire class
of generator bugs that produce malformed token streams.

pglast is pinned to v7 (libpg_query for PG17). v8 (PG18) is in
development on the upstream `lelit/pglast` v8 branch but not on PyPI
as of May 2026 — when it ships, bump the `pyproject.toml` pin and
re-run the suite. The version pin is load-bearing: libpg_query is
the actual PostgreSQL parser compiled as a static library, so the
grammar accepted here is exactly the grammar PG accepts (modulo
version skew, which is why we pin).

What it does NOT catch:

  * Undefined column or table references
  * Type mismatches
  * Aggregates in WHERE
  * Anything requiring catalog lookup
  * Operator-class lookup failures, constant-foldable runtime errors

For those, use the PARSE tier (`waxsql.validate.parse.check_parse`),
which runs the live PG parser via PREPARE, or the PLAN tier
(`waxsql.validate.plan.check_plan`), which runs the planner via
EXPLAIN. Both ship at v1.0; both require psycopg and a live PG.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Frozen dataclass per project convention (Conventions section of
# ARCHITECTURE.md): hashable, can't accidentally mutate, cheap to pass.
# Same general shape as ParseResult/PlanResult so callers can branch
# on `.ok` uniformly, but this tier carries an `error_position`
# because pglast surfaces a 1-based character offset on failure —
# the live-DB tiers carry a SQLSTATE instead (different provenance).
@dataclass(frozen=True)
class SyntaxResult:
    ok: bool
    error: Optional[str] = None
    error_position: Optional[int] = None  # 1-based char offset, if available


def check_syntax(sql: str) -> SyntaxResult:
    """Parse `sql` via pglast. Returns SyntaxResult with ok=False on any
    parse error.

    Raises RuntimeError if pglast is not installed — install with the
    `syntax` extra: `pip install 'waxsql[syntax]'`.
    """
    # Lazy import: pglast is technically optional (`[syntax]` extra),
    # so importing at module-load would force the dependency on every
    # consumer of `ValidationMode`, including callers who only ever
    # use NONE. Deferring keeps the import cost off the hot path for
    # users who don't need syntax validation.
    try:
        from pglast import parse_sql
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Syntax validation requires pglast. "
            "Install with: pip install 'waxsql[syntax]'"
        ) from e

    try:
        parse_sql(sql)
    except Exception as e:
        # Broad `Exception` catch is deliberate: pglast normally raises
        # ParseError, but version skew or upstream changes could surface
        # other exception types. Masking them as "non-ok with the
        # exception text" beats letting an unrelated exception type
        # punch through and crash the test harness.
        # pglast raises pglast.parser.ParseError with a `.location` attr
        # (1-based character offset) on parse failure. Other exception
        # types are unexpected but worth surfacing rather than masking.
        loc = getattr(e, "location", None)
        return SyntaxResult(ok=False, error=str(e), error_position=loc)
    return SyntaxResult(ok=True)
