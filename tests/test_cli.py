"""Tests for waxsql.cli — the click-based CLI.

Uses click.testing.CliRunner so no subprocess is involved. Tests that
exercise the missing-dep path simulate the ImportError by patching
sys.modules; the live-DB tests reuse the pg_conn session fixture from
tests/conftest.py and skip cleanly without a connection.
"""
from __future__ import annotations

import os
import sys
import tempfile
from unittest import mock

import pytest

# importorskip rather than a hard import so the test file is collectable
# even if click hasn't been installed in the dev environment yet (it
# normally is, via [dev], but failing-soft here matches how the rest of
# the suite handles optional deps).
click = pytest.importorskip("click")
from click.testing import CliRunner  # noqa: E402

from waxsql.cli import (  # noqa: E402
    _extract_first_select,
    _parse_header,
    _render_header,
    _resolve_schema_source,
    _split_copy_blocks,
    main,
)

cli_main = main  # alias used in the Gen/Validate test classes for clarity


def test_main_help_exits_zero() -> None:
    """Ensures both subcommands are wired into the click group.
    A user's first interaction is typically `waxsql --help`; if gen or
    validate is missing from the listing, the tool is silently incomplete
    without any import error to signal the gap.
    """
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "gen" in result.output
    assert "validate" in result.output


def test_main_version_exits_zero() -> None:
    """Ensures the version reported by the CLI matches the package's installed
    version — important for reproducibility when users report issues, since the
    CLI's --version is the primary way users identify which build they ran."""
    from waxsql import __version__
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_missing_click_prints_friendly_hint(capsys) -> None:
    """Users who install plain `waxsql` without the [cli] extra get an
    actionable install hint rather than a raw ImportError traceback — this UX
    choice is why the try/except guard precedes all click code.  The
    sys.modules manipulation below forces a fresh re-import so we can simulate
    the missing-dep path in a session where click is already loaded."""
    # waxsql.cli is already cached in sys.modules from the `from waxsql.cli
    # import main` at the top of this file, so we must evict both it and the
    # click entry before patching, otherwise `import waxsql.cli` below would
    # return the cached module without re-executing the module-level try/except.
    saved_click = sys.modules.pop("click", None)
    saved_cli   = sys.modules.pop("waxsql.cli", None)
    try:
        with mock.patch.dict(sys.modules, {"click": None}):
            with pytest.raises(SystemExit) as exc_info:
                import waxsql.cli  # noqa: F401
            assert exc_info.value.code == 3
        captured = capsys.readouterr()
        assert "pip install 'waxsql[cli]'" in captured.err
    finally:
        # Restore so subsequent tests can import waxsql.cli normally.
        if saved_click is not None:
            sys.modules["click"] = saved_click
        if saved_cli is not None:
            sys.modules["waxsql.cli"] = saved_cli
        else:
            sys.modules.pop("waxsql.cli", None)


class TestHeader:
    """The `-- waxsql <version> seed=N complexity=X` header is the
    convention by which `gen` output identifies itself for `validate
    --auto-schema`. Round-trip property + edge cases here because any
    future format drift would silently break the gen→validate pipeline.
    """

    def test_round_trip_simple(self) -> None:
        rendered = _render_header(seed=42, complexity=5)
        parsed = _parse_header(rendered)
        assert parsed is not None
        assert parsed.seed == 42
        assert parsed.complexity == 5

    def test_round_trip_complexity_zero(self) -> None:
        rendered = _render_header(seed=0, complexity=0)
        parsed = _parse_header(rendered)
        assert parsed is not None
        assert parsed.seed == 0
        assert parsed.complexity == 0

    def test_round_trip_negative_seed(self) -> None:
        # Seeds are arbitrary 64-bit ints; negative values must round-trip.
        rendered = _render_header(seed=-1234567890, complexity=10)
        parsed = _parse_header(rendered)
        assert parsed is not None
        assert parsed.seed == -1234567890
        assert parsed.complexity == 10

    def test_no_header_returns_none(self) -> None:
        assert _parse_header("SELECT 1") is None
        assert _parse_header("") is None
        assert _parse_header("\n\n") is None

    def test_malformed_header_returns_none(self) -> None:
        # Looks like a header but missing required fields.
        assert _parse_header("-- waxsql seed=42") is None
        assert _parse_header("-- not waxsql seed=42 complexity=5") is None

    def test_header_must_be_first_nonempty_line(self) -> None:
        # We deliberately don't scan for headers further down — keeps the
        # round-trip property obvious and the parser dumb.
        text = "SELECT 1;\n-- waxsql 1.0.0  seed=42  complexity=5\n"
        assert _parse_header(text) is None

    def test_header_tolerates_leading_blank_lines(self) -> None:
        text = "\n\n-- waxsql 1.0.0  seed=42  complexity=5\nSELECT 1;\n"
        parsed = _parse_header(text)
        assert parsed is not None
        assert parsed.seed == 42
        assert parsed.complexity == 5

    def test_render_includes_version(self) -> None:
        from waxsql import __version__
        out = _render_header(seed=1, complexity=1)
        assert __version__ in out
        assert "seed=1" in out
        assert "complexity=1" in out

    def test_render_header_with_data_flags(self) -> None:
        out = _render_header(
            seed=7, complexity=3,
            with_data=True, rows=50, fanout=4, null_fraction=0.1,
        )
        assert "seed=7" in out
        assert "complexity=3" in out
        assert "with-data=true" in out
        assert "rows=50" in out
        assert "fanout=4" in out
        assert "null-fraction=0.1" in out

    def test_parse_header_round_trip_extended(self) -> None:
        rendered = _render_header(
            seed=7, complexity=3,
            with_data=True, rows=50, fanout=4, null_fraction=0.1,
        )
        parsed = _parse_header(rendered)
        assert parsed is not None
        assert parsed.seed == 7
        assert parsed.complexity == 3
        assert parsed.with_data is True
        assert parsed.rows == 50
        assert parsed.fanout == 4
        assert parsed.null_fraction == 0.1

    def test_parse_header_backwards_compat_without_data_keys(self) -> None:
        """Old gen output (no with-data/rows/fanout) parses with defaults."""
        parsed = _parse_header("-- waxsql 0.1.0  seed=1  complexity=2\n")
        assert parsed is not None
        assert parsed.seed == 1
        assert parsed.complexity == 2
        assert parsed.with_data is False
        assert parsed.rows == 100
        assert parsed.fanout == 5
        assert parsed.null_fraction == 0.05

    def test_malformed_seed_returns_none(self) -> None:
        """A header that matches the prefix regex but carries a
        non-integer seed should return None rather than raising
        ValueError up to the CLI. The validator's no-header path
        then takes over gracefully."""
        assert _parse_header("-- waxsql 1.0.0  seed=abc  complexity=5\n") is None

    def test_malformed_complexity_returns_none(self) -> None:
        assert _parse_header("-- waxsql 1.0.0  seed=42  complexity=xyz\n") is None

    def test_malformed_optional_value_returns_none(self) -> None:
        """Even optional fields, when present-but-malformed, fall
        through to None — not raise. A user with a hand-edited
        header shouldn't get an uncaught ValueError traceback."""
        text = (
            "-- waxsql 1.0.0  seed=1  complexity=2  "
            "with-data=true  rows=many  fanout=4  null-fraction=0.1\n"
        )
        assert _parse_header(text) is None

    def test_header_tolerates_crlf_line_endings(self) -> None:
        """CRLF input shouldn't make int(value) raise. Python's
        splitlines normally handles \\r\\n cleanly, but if anything
        upstream in the pipeline double-encodes or assembles from
        bytes irregularly, a lingering \\r would have surfaced as
        a ValueError. The rstrip in _parse_header is defense in
        depth against that class of bug."""
        text = "-- waxsql 1.0.0  seed=42  complexity=5\r\nSELECT 1;\r\n"
        parsed = _parse_header(text)
        assert parsed is not None
        assert parsed.seed == 42
        assert parsed.complexity == 5


class TestRedactDsn:
    """Password material in DSN strings must not leak to stderr via
    the connection-failure error message. Tests the redaction helper
    directly so we don't need a live PG to exercise the failure path.
    """

    def test_keyvalue_password_redacted(self) -> None:
        from waxsql.cli import _redact_dsn
        out = _redact_dsn("dbname=foo password=hunter2 host=h")
        assert "hunter2" not in out
        assert "password=***" in out
        # Non-password fields survive intact.
        assert "dbname=foo" in out
        assert "host=h" in out

    def test_uri_password_redacted(self) -> None:
        from waxsql.cli import _redact_dsn
        out = _redact_dsn("postgresql://alice:hunter2@db.example.com:5432/mydb")
        assert "hunter2" not in out
        # The `:` and `@` delimiters stay so the structure is
        # still recognizable, just with the password masked.
        assert ":***@" in out
        # User and host parts preserved.
        assert "alice" in out
        assert "db.example.com" in out

    def test_uri_password_redacted_no_user(self) -> None:
        """`postgresql://:secret@host/db` is a valid URI without a
        user portion; the regex must still find the password."""
        from waxsql.cli import _redact_dsn
        out = _redact_dsn("postgresql://:hunter2@db.example.com/mydb")
        assert "hunter2" not in out
        assert ":***@" in out

    def test_no_password_unchanged(self) -> None:
        from waxsql.cli import _redact_dsn
        assert _redact_dsn("dbname=foo host=h") == "dbname=foo host=h"
        assert (
            _redact_dsn("postgresql://alice@db.example.com/mydb")
            == "postgresql://alice@db.example.com/mydb"
        )

    def test_kv_password_with_punctuation(self) -> None:
        """Password values can include any non-whitespace; ensure the
        whole token gets masked even if it contains punctuation."""
        from waxsql.cli import _redact_dsn
        out = _redact_dsn("dbname=foo password=p@ss:w0rd! host=h")
        assert "p@ss:w0rd!" not in out
        assert "password=***" in out
        assert "host=h" in out


class TestSplitCopyBlocks:
    r"""`_split_copy_blocks` chunks a generate_data stream on the `\.`
    terminator. generate_data always terminates every block it emits;
    a truncated or hand-edited stream might not, and silently dropping
    the orphan tail would look like "some rows just vanished" with no
    error — so an unterminated final block must raise."""

    def test_splits_terminated_blocks(self) -> None:
        stream = (
            'COPY "t" ("id") FROM STDIN;\n1\n\\.\n'
            "\n"
            'COPY "u" ("id") FROM STDIN;\n2\n\\.\n'
        )
        blocks = _split_copy_blocks(stream)
        assert len(blocks) == 2
        assert blocks[0].startswith('COPY "t"')
        assert blocks[1].startswith('COPY "u"')

    def test_raises_on_unterminated_trailing_block(self) -> None:
        stream = (
            'COPY "t" ("id") FROM STDIN;\n1\n\\.\n'
            "\n"
            'COPY "u" ("id") FROM STDIN;\n2\n'  # missing \. terminator
        )
        with pytest.raises(ValueError, match="unterminated COPY block"):
            _split_copy_blocks(stream)


class TestGenBasic:
    """`waxsql gen` — default behavior, --seed, --complexity, header.
    These tests establish the determinism guarantee and the psql-ready
    output shape that the headline demo use case depends on.
    """

    def test_explicit_seed_is_deterministic(self) -> None:
        """Same `--seed` and `--complexity` always produce the same output.
        Determinism is the property that makes the tool useful for bug
        reproduction: a user can share a seed and we can replay their exact run.
        """
        runner = CliRunner()
        a = runner.invoke(cli_main, ["gen", "--seed", "42", "-c", "5"])
        b = runner.invoke(cli_main, ["gen", "--seed", "42", "-c", "5"])
        assert a.exit_code == 0
        assert b.exit_code == 0
        assert a.output == b.output

    def test_random_seed_default_is_nondeterministic(self) -> None:
        """No `--seed` → different output across invocations (almost surely).
        The random default is intentional: every new user sees different SQL,
        demonstrating the generator's variety rather than a fixed example.
        """
        runner = CliRunner()
        a = runner.invoke(cli_main, ["gen"])
        b = runner.invoke(cli_main, ["gen"])
        assert a.exit_code == 0
        assert b.exit_code == 0
        # Astronomically unlikely to collide with a 64-bit random seed.
        assert a.output != b.output

    def test_header_present_by_default(self) -> None:
        """The header is the mechanism that makes random runs reproducible.
        Verifying it contains seed= and complexity= on the first line ensures
        the validate --auto-schema feature can parse it back.
        """
        runner = CliRunner()
        result = runner.invoke(cli_main, ["gen", "--seed", "42", "-c", "5"])
        assert result.exit_code == 0
        first_line = result.output.splitlines()[0]
        assert first_line.startswith("-- waxsql")
        assert "seed=42" in first_line
        assert "complexity=5" in first_line

    def test_random_seed_replay_works(self) -> None:
        """The seed printed in the header should reproduce the run exactly.
        This is the key UX promise: `waxsql gen --seed <printed> -c <printed>`
        reproduces byte-for-byte, so a shared seed is a complete bug report.
        """
        runner = CliRunner()
        first = runner.invoke(cli_main, ["gen"])
        # Extract seed from the header.
        import re as _re
        m = _re.search(r"seed=(-?\d+)\s+complexity=(\d+)", first.output)
        assert m is not None
        seed, c = m.group(1), m.group(2)
        replay = runner.invoke(cli_main, ["gen", "--seed", seed, "-c", c])
        assert first.output == replay.output

    def test_default_emits_schema_and_query(self) -> None:
        """Default output is both schema and query — the pasteable demo output.
        The `-- schema:` and `-- query:` labels are visual section markers;
        their presence confirms the full output shape is intact.
        """
        runner = CliRunner()
        result = runner.invoke(cli_main, ["gen", "--seed", "42", "-c", "5"])
        assert result.exit_code == 0
        assert "-- schema:" in result.output
        assert "-- query:" in result.output
        assert "CREATE TABLE" in result.output
        assert "SELECT" in result.output

    def test_query_has_trailing_semicolon(self) -> None:
        """`waxsql gen` is opinionated for the demo case: psql-ready SQL.
        The library's print_query() omits the trailing `;` (caller's choice);
        the CLI adds it so the output is pasteable into psql without editing.
        """
        runner = CliRunner()
        result = runner.invoke(cli_main, ["gen", "--seed", "42", "-c", "5"])
        assert result.exit_code == 0
        # Strip trailing whitespace, then the last non-empty line should end with ;
        non_empty = [ln for ln in result.output.splitlines() if ln.strip()]
        assert non_empty[-1].rstrip().endswith(";")

    def test_complexity_default_is_5(self) -> None:
        """No `--complexity` flag → uses 5 (matches library default).
        Documents the contract so a change to the default is visible in tests.
        """
        runner = CliRunner()
        a = runner.invoke(cli_main, ["gen", "--seed", "42"])
        b = runner.invoke(cli_main, ["gen", "--seed", "42", "-c", "5"])
        assert a.output == b.output


class TestGenPprint:
    """`waxsql gen --pprint` reformats + (on a TTY) colorizes the
    generated SQL. In CliRunner, stdout is not a TTY, so color is
    auto-off — which lets us assert on the reflowed plain text and
    round-trip it back through pglast to prove we didn't corrupt it.
    """

    def test_pprint_query_reflows_and_still_parses(self) -> None:
        from pglast import parse_sql
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["gen", "--seed", "7", "-c", "6",
             "--query-only", "--no-header", "--pprint"],
        )
        assert result.exit_code == 0, result.output
        sql = result.output
        # CliRunner stdout is not a TTY -> no color.
        assert "\x1b[" not in sql
        # The reflowed query is still valid SQL.
        parse_sql(sql)
        # Reflow actually happened (multi-line output).
        assert "\n" in sql.strip()

    def test_pprint_leaves_header_and_is_deterministic(self) -> None:
        runner = CliRunner()
        a = runner.invoke(cli_main, ["gen", "--seed", "7", "-c", "6", "--pprint"])
        b = runner.invoke(cli_main, ["gen", "--seed", "7", "-c", "6", "--pprint"])
        assert a.exit_code == 0, a.output
        # Header comment is still present and plain (not reflowed).
        assert a.output.startswith("-- waxsql")
        # --pprint output is deterministic for a fixed (seed, complexity).
        assert a.output == b.output

    def test_pprint_missing_deps_exits_friendly(self, monkeypatch) -> None:
        # Force prettify_sql to behave as if the optional deps are absent.
        import waxsql.pretty

        def _boom(sql, *, color):
            raise RuntimeError(
                "--pprint requires pglast and pygments. "
                "Install with: pip install 'waxsql[pprint]'"
            )

        monkeypatch.setattr(waxsql.pretty, "prettify_sql", _boom)
        runner = CliRunner()
        result = runner.invoke(
            cli_main, ["gen", "--seed", "1", "-c", "2", "--pprint"]
        )
        assert result.exit_code == 3
        assert "waxsql[pprint]" in result.output


class TestGenScopeFlags:
    """--schema-only and --query-only: mutually exclusive escape hatches.
    They exist to support callers who need only one half of the output —
    e.g. a schema-only pipe into psql, or a query-only pipe into a
    syntax checker. The header is preserved in either mode so the run
    is still reproducible from the output alone.
    """

    def test_schema_only_emits_no_select(self) -> None:
        """--schema-only should suppress the query section entirely.
        The labeled-section test (absence of '-- query:') is more
        robust than checking for SELECT absence, since SELECT could
        theoretically appear in a CHECK constraint someday.
        """
        runner = CliRunner()
        result = runner.invoke(cli_main, ["gen", "--seed", "42", "--schema-only"])
        assert result.exit_code == 0
        assert "CREATE TABLE" in result.output
        assert "-- query:" not in result.output

    def test_query_only_emits_no_create(self) -> None:
        """--query-only should suppress the schema section entirely."""
        runner = CliRunner()
        result = runner.invoke(cli_main, ["gen", "--seed", "42", "--query-only"])
        assert result.exit_code == 0
        assert "SELECT" in result.output
        assert "CREATE TABLE" not in result.output
        assert "-- schema:" not in result.output

    def test_both_flags_is_usage_error(self) -> None:
        """Passing both --schema-only and --query-only together is a usage
        error (exit 2). They're logically contradictory — 'only this' AND
        'only that' has no valid interpretation.
        """
        runner = CliRunner()
        result = runner.invoke(
            cli_main, ["gen", "--seed", "42", "--schema-only", "--query-only"]
        )
        assert result.exit_code == 2
        assert "mutually exclusive" in result.output.lower() \
            or "cannot be used together" in result.output.lower()

    def test_schema_only_still_has_header(self) -> None:
        """The header is preserved even in schema-only mode so the DDL output
        is self-describing — a consumer can always tell which seed generated it.
        """
        runner = CliRunner()
        result = runner.invoke(cli_main, ["gen", "--seed", "42", "--schema-only"])
        first_line = result.output.splitlines()[0]
        assert first_line.startswith("-- waxsql")

    def test_query_only_still_has_header(self) -> None:
        """Same reason as schema_only: the header is what makes the output
        reproducible, so suppressing the schema section must not suppress it.
        """
        runner = CliRunner()
        result = runner.invoke(cli_main, ["gen", "--seed", "42", "--query-only"])
        first_line = result.output.splitlines()[0]
        assert first_line.startswith("-- waxsql")


class TestGenCountAndHeader:
    """--count N (multiple queries against one schema) and --no-header.
    These features serve batch use cases (comparing N different query
    shapes against the same schema) and tool-pipeline use cases (stripping
    the comment header when the downstream tool doesn't tolerate it).
    """

    def test_count_emits_n_queries(self) -> None:
        """N queries means N `-- query` sections, each with a trailing `;`.
        The schema should appear exactly once regardless of N.
        """
        runner = CliRunner()
        result = runner.invoke(cli_main, ["gen", "--seed", "42", "-c", "5", "-n", "3"])
        assert result.exit_code == 0
        # Three "-- query …" labels, one schema.
        assert result.output.count("-- query") == 3
        assert result.output.count("CREATE TABLE") >= 1

    def test_count_has_separator_with_index(self) -> None:
        """Multi-query output uses indexed separators `-- query K/N` so a
        reader knows how many queries to expect and which one they're looking at.
        """
        runner = CliRunner()
        result = runner.invoke(cli_main, ["gen", "--seed", "42", "-c", "5", "-n", "3"])
        assert "-- query 1/3" in result.output
        assert "-- query 2/3" in result.output
        assert "-- query 3/3" in result.output

    def test_count_one_uses_unindexed_label(self) -> None:
        """Single-query case (the common one) keeps the `-- query:` label so
        the N=1 output is identical to the no-flag output. Changing this
        would break existing pipelines that parse the label.
        """
        runner = CliRunner()
        a = runner.invoke(cli_main, ["gen", "--seed", "42", "-c", "5"])
        b = runner.invoke(cli_main, ["gen", "--seed", "42", "-c", "5", "-n", "1"])
        assert a.output == b.output

    def test_count_query_seeds_are_consecutive(self) -> None:
        """Same seed always yields the same N-query batch — required so that
        `--seed S -n 3` is a reproducible, self-contained bug report.
        """
        runner = CliRunner()
        a = runner.invoke(cli_main, ["gen", "--seed", "100", "-c", "5", "-n", "3"])
        b = runner.invoke(cli_main, ["gen", "--seed", "100", "-c", "5", "-n", "3"])
        assert a.output == b.output

    def test_count_with_query_only_emits_n_queries_no_schema(self) -> None:
        """--query-only + --count N should emit N query sections, no schema.
        Verifies the flags compose correctly.
        """
        runner = CliRunner()
        result = runner.invoke(
            cli_main, ["gen", "--seed", "42", "-c", "5", "-n", "3", "--query-only"]
        )
        assert result.exit_code == 0
        assert "CREATE TABLE" not in result.output
        assert "-- schema:" not in result.output
        assert result.output.count("-- query") == 3

    def test_no_header_suppresses_first_line(self) -> None:
        """--no-header removes the `-- waxsql ...` line from the output.
        The first content line should be the schema label, not the header.
        This is the escape hatch for pipeline tools that choke on SQL comments
        at the very top.
        """
        runner = CliRunner()
        result = runner.invoke(
            cli_main, ["gen", "--seed", "42", "-c", "5", "--no-header"]
        )
        assert result.exit_code == 0
        first_line = result.output.splitlines()[0]
        assert not first_line.startswith("-- waxsql")
        # First content line should be the schema label.
        assert first_line == "-- schema:"


class TestResolveSchemaSource:
    """The validate subcommand's schema-source precedence helper.

    Returns DDL (string) or None. Pure beyond reading --schema-from PATH.
    Precedence matters because validate can receive a schema from multiple
    possible sources; the ordering is: explicit file > explicit seed >
    auto-parsed header > nothing. Tests here document that order explicitly
    so future changes to the precedence are caught as test failures.
    """

    def test_returns_none_when_nothing_supplied(self) -> None:
        """No schema source → None. The caller (validate) decides if this
        is an error; the helper stays neutral so SYNTAX-tier callers can
        ignore the schema entirely.
        """
        result = _resolve_schema_source(
            input_text="SELECT 1",
            schema_from=None, schema_seed=None, schema_complexity=5,
            auto_schema=True,
        )
        assert result is None

    def test_returns_none_when_auto_schema_off_and_input_has_header(self) -> None:
        """auto_schema=False disables header parsing even if a header is present.
        Needed so callers can opt out of the auto-schema behavior.
        """
        result = _resolve_schema_source(
            input_text=_render_header(42, 5) + "\n-- schema:\nCREATE TABLE t (id BIGINT);\n",
            schema_from=None, schema_seed=None, schema_complexity=5,
            auto_schema=False,
        )
        assert result is None

    def test_auto_schema_parses_header(self) -> None:
        """auto_schema + a recognizable header → regenerate and return the DDL.
        We return the REGENERATED DDL (not the text from the input) so the
        schema is always canonical and matches what the library actually produces.
        """
        result = _resolve_schema_source(
            input_text=_render_header(42, 5) + "\nSELECT 1;\n",
            schema_from=None, schema_seed=None, schema_complexity=5,
            auto_schema=True,
        )
        assert result is not None
        assert "CREATE TABLE" in result
        # Same seed ⇒ same DDL as a direct generate_schema call.
        from waxsql import generate_schema
        expected = generate_schema(seed=42, complexity=5).emit_ddl()
        assert result == expected

    def test_explicit_schema_seed_beats_auto(self) -> None:
        """schema_seed overrides auto-schema even when a header is present.
        The explicit seed is higher precedence — the user said what schema to use.
        """
        result = _resolve_schema_source(
            input_text=_render_header(42, 5) + "\nSELECT 1;\n",
            schema_from=None, schema_seed=99, schema_complexity=8,
            auto_schema=True,
        )
        from waxsql import generate_schema
        expected = generate_schema(seed=99, complexity=8).emit_ddl()
        assert result == expected

    def test_schema_from_beats_everything(self) -> None:
        """--schema-from PATH wins over both --schema-seed and auto-schema.
        It's the most explicit source, so it takes the highest precedence.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sql", delete=False
        ) as f:
            f.write("CREATE TABLE custom (id BIGINT);\n")
            path = f.name
        try:
            result = _resolve_schema_source(
                input_text=_render_header(42, 5) + "\nSELECT 1;\n",
                schema_from=path, schema_seed=99, schema_complexity=8,
                auto_schema=True,
            )
            assert result == "CREATE TABLE custom (id BIGINT);\n"
        finally:
            os.unlink(path)

    def test_schema_seed_requires_complexity(self) -> None:
        """When schema_seed is supplied, schema_complexity controls the dial.
        Documents the contract between these two params (CLI gives complexity
        a default of 5, so the helper never sees a None complexity).
        """
        result = _resolve_schema_source(
            input_text="SELECT 1",
            schema_from=None, schema_seed=42, schema_complexity=5,
            auto_schema=False,
        )
        assert result is not None
        assert "CREATE TABLE" in result


class TestValidateSyntax:
    """`waxsql validate --tier syntax` — pure pglast, no DB.
    This is the default tier because it's the cheapest (no network, no
    schema) and catches every grammar error. The output discipline (silent
    on success, error to stderr on failure) follows Unix conventions so
    the command is composable in pipelines.
    """

    _GOOD_SQL = "SELECT 1;\n"
    _BAD_SQL  = "SELEKT 1;\n"  # typo'd keyword — pglast rejects it

    def test_good_sql_from_stdin_silent_exit_0(self) -> None:
        """Silent on success is the Unix pipeline convention — a successful
        validate should not pollute downstream consumers with output.
        """
        runner = CliRunner()
        result = runner.invoke(cli_main, ["validate"], input=self._GOOD_SQL)
        assert result.exit_code == 0
        assert result.output == ""  # silent on success

    def test_good_sql_from_stdin_verbose_prints_ok(self) -> None:
        """--verbose produces human-readable confirmation — useful when the
        command is run interactively rather than as part of a pipeline.
        """
        runner = CliRunner()
        result = runner.invoke(
            cli_main, ["validate", "-v"], input=self._GOOD_SQL,
        )
        assert result.exit_code == 0
        assert "OK" in result.output

    def test_bad_sql_from_stdin_exit_1_with_stderr(self) -> None:
        """Validation failure goes to stderr so stdout stays clean for pipelines.
        Exit 1 (not 2) signals a validation failure rather than a usage error.
        """
        # Click 8.x exposes result.stdout and result.stderr separately without
        # any mix_stderr flag — the streams are always distinct in this version.
        runner = CliRunner()
        result = runner.invoke(cli_main, ["validate"], input=self._BAD_SQL)
        assert result.exit_code == 1
        assert result.stdout == ""  # no stdout output on failure either
        assert result.stderr != ""
        # Don't pin the exact pglast error string; just check we're reporting an error.
        assert "syntax" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_good_sql_from_file(self) -> None:
        """validate accepts a positional file argument as an alternative to stdin."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("query.sql", "w") as f:
                f.write(self._GOOD_SQL)
            result = runner.invoke(cli_main, ["validate", "query.sql"])
            assert result.exit_code == 0

    def test_dash_means_stdin(self) -> None:
        """'-' as the file argument is an explicit stdin marker — required by
        the spec so scripts can pass '-' unconditionally without special-casing.
        """
        runner = CliRunner()
        result = runner.invoke(cli_main, ["validate", "-"], input=self._GOOD_SQL)
        assert result.exit_code == 0

    def test_default_tier_is_syntax(self) -> None:
        """Default tier is 'syntax' — the cheapest, no-DB option.
        If the default were 'parse', this would try to connect to PG and
        likely fail or skip, making the out-of-box experience bad.
        """
        runner = CliRunner()
        result = runner.invoke(cli_main, ["validate"], input=self._GOOD_SQL)
        assert result.exit_code == 0

    def test_pipe_from_gen_works(self) -> None:
        """`waxsql gen | waxsql validate --tier syntax` — the headline demo.
        gen output includes the schema DDL + query; pglast should handle the
        full multi-statement output since it validates statement-by-statement.
        """
        runner = CliRunner()
        gen_result = runner.invoke(cli_main, ["gen", "--seed", "42", "-c", "5"])
        assert gen_result.exit_code == 0
        validate_result = runner.invoke(
            cli_main, ["validate", "--tier", "syntax"],
            input=gen_result.output,
        )
        assert validate_result.exit_code == 0


class TestExtractFirstSelect:
    """_extract_first_select pulls the first SELECT/WITH statement out of
    gen output (which also includes DDL). PREPARE and EXPLAIN take a single
    statement; this helper bridges the gap so `gen | validate --tier parse`
    just works without the user stripping the DDL manually.
    """

    def test_returns_select_from_mixed_input(self) -> None:
        """Extracts the SELECT from gen output that also contains DDL."""
        text = "-- comment\nCREATE TABLE t (id BIGINT);\nSELECT id FROM t;\n"
        result = _extract_first_select(text)
        assert result.strip().upper().startswith("SELECT")
        assert "CREATE TABLE" not in result

    def test_returns_with_from_cte_input(self) -> None:
        """CTEs start with WITH, not SELECT. Both are valid query starts."""
        text = "CREATE TABLE t (id BIGINT);\nWITH cte AS (SELECT 1) SELECT * FROM cte;\n"
        result = _extract_first_select(text)
        assert result.strip().upper().startswith("WITH")

    def test_returns_text_unchanged_if_no_select(self) -> None:
        """If no SELECT/WITH is found, pass through unchanged so hand-written
        single-statement input still works without special-casing.
        """
        # DDL-only input (no SELECT/WITH) passes through unchanged.
        ddl_only = "CREATE TABLE t (id BIGINT);\n"
        result = _extract_first_select(ddl_only)
        assert result == ddl_only  # passed through unchanged

    def test_strips_trailing_semicolon(self) -> None:
        """PREPARE and EXPLAIN don't want a trailing `;`; we strip it so the
        validator doesn't have to.
        """
        text = "SELECT id FROM t;"
        result = _extract_first_select(text)
        assert not result.strip().endswith(";")

    def test_simple_select_is_returned_unchanged(self) -> None:
        """A plain SELECT without any DDL prefix is returned as-is (minus the `;`)."""
        text = "SELECT 1"
        result = _extract_first_select(text)
        assert "SELECT" in result


class TestValidateParsePlan:
    """`waxsql validate --tier {parse,plan}` — requires a live PG.

    Uses the pg_conn session fixture from conftest.py; tests skip
    cleanly if psycopg or PG isn't available. Each test runs the CLI
    via CliRunner but points WAXSQL_PG_DSN at the same PG the fixture
    uses. Transaction isolation is handled inside the validate command
    (connect → install schema → check → rollback → close).
    """

    def _make_gen_output(self, runner, seed: int, complexity: int) -> str:
        """Helper: produce gen output for a given seed/complexity."""
        result = runner.invoke(
            cli_main, ["gen", "--seed", str(seed), "-c", str(complexity)],
        )
        assert result.exit_code == 0
        return result.output

    def test_parse_tier_with_auto_schema_pipe(self, pg_conn, monkeypatch) -> None:
        """gen | validate --tier parse — the headline pipeline payoff.
        The gen-output header lets validate reconstruct the matching schema
        without any extra flags. This is the killer use case.
        """
        import os
        dsn = os.environ.get("WAXSQL_PG_DSN", "dbname=waxsql_test")
        monkeypatch.setenv("WAXSQL_PG_DSN", dsn)

        runner = CliRunner()
        gen_output = self._make_gen_output(runner, seed=42, complexity=5)
        # Feed the full gen output to validate — auto-schema will pick up
        # the header and reinstall the matching schema before running PREPARE.
        result = runner.invoke(
            cli_main, ["validate", "--tier", "parse", "-v"],
            input=gen_output,
        )
        assert result.exit_code == 0, f"unexpected failure: {result.output!r}"
        assert "OK" in result.output

    def test_parse_tier_explicit_schema_seed(self, pg_conn, monkeypatch) -> None:
        """--schema-seed N / --schema-complexity X regenerate the schema
        explicitly. The gen-output header is NOT used (--no-auto-schema).
        """
        import os
        dsn = os.environ.get("WAXSQL_PG_DSN", "dbname=waxsql_test")
        monkeypatch.setenv("WAXSQL_PG_DSN", dsn)

        runner = CliRunner()
        gen_output = self._make_gen_output(runner, seed=42, complexity=5)

        result = runner.invoke(
            cli_main,
            ["validate", "--tier", "parse",
             "--schema-seed", "42", "--schema-complexity", "5",
             "--no-auto-schema"],
            input=gen_output,
        )
        assert result.exit_code == 0, f"unexpected failure: {result.output!r}"

    def test_parse_tier_missing_schema_is_usage_error(self, pg_conn, monkeypatch) -> None:
        """parse/plan with no schema source AND no header → exit 2.
        Clear message so the user knows how to provide a schema.
        """
        import os
        dsn = os.environ.get("WAXSQL_PG_DSN", "dbname=waxsql_test")
        monkeypatch.setenv("WAXSQL_PG_DSN", dsn)

        runner = CliRunner()
        result = runner.invoke(
            cli_main, ["validate", "--tier", "parse", "--no-auto-schema"],
            input="SELECT 1;\n",
        )
        assert result.exit_code == 2
        assert "schema" in result.output.lower() or "schema" in result.stderr.lower()

    def test_plan_tier_with_auto_schema_pipe(self, pg_conn, monkeypatch) -> None:
        """gen | validate --tier plan — same pipeline shape, runs EXPLAIN
        instead of PREPARE. Exercises the planner, not just the parser.
        """
        import os
        dsn = os.environ.get("WAXSQL_PG_DSN", "dbname=waxsql_test")
        monkeypatch.setenv("WAXSQL_PG_DSN", dsn)

        runner = CliRunner()
        gen_output = self._make_gen_output(runner, seed=42, complexity=5)
        result = runner.invoke(
            cli_main, ["validate", "--tier", "plan", "-v"],
            input=gen_output,
        )
        assert result.exit_code == 0, f"unexpected failure: {result.output!r}"
        assert "OK" in result.output


class TestValidateMissingDeps:
    """Friendly error when an optional dep isn't installed.
    The user gets a pip install hint and exit 3, matching the missing-click
    behavior at the top of cli.py. Consistency matters: all missing deps
    should behave the same way.
    """

    def test_parse_tier_without_psycopg(self, monkeypatch) -> None:
        """Simulate psycopg not being importable — verifies exit 3 and
        the install hint, so a user without [parse] gets actionable feedback.
        """
        monkeypatch.setitem(sys.modules, "psycopg", None)
        runner = CliRunner()
        result = runner.invoke(
            cli_main, ["validate", "--tier", "parse"],
            input="SELECT 1;\n",
        )
        assert result.exit_code == 3
        assert "pip install" in result.output or "pip install" in result.stderr
        assert "[parse]" in result.output or "[parse]" in result.stderr or "[plan]" in result.output or "[plan]" in result.stderr
