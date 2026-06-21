"""Tests for SQL pretty-printing. See waxsql/pretty.py."""
from __future__ import annotations

import io
import sys

import pytest

from waxsql.pretty import prettify_sql, should_colorize


class _FakeStream:
    """Minimal stand-in for a stdout-like object with a controllable
    isatty() — io.StringIO always reports isatty() False, so we need a
    fake to exercise the True branch."""

    def __init__(self, isatty: bool) -> None:
        self._isatty = isatty

    def isatty(self) -> bool:
        return self._isatty


def test_should_colorize_true_for_tty_without_no_color(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert should_colorize(_FakeStream(isatty=True)) is True


def test_should_colorize_false_for_non_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    # A plain StringIO is not a terminal.
    assert should_colorize(io.StringIO()) is False


def test_should_colorize_false_when_no_color_set(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    # Even on a TTY, NO_COLOR opts out (https://no-color.org).
    assert should_colorize(_FakeStream(isatty=True)) is False


def test_prettify_reformats_query_color_false():
    out = prettify_sql("SELECT a, b FROM t1 WHERE a > 1", color=False)
    # comma_at_eoln=True → trailing commas, aligned continuation.
    assert out == "SELECT a,\n       b\nFROM t1\nWHERE a > 1"


def test_prettify_preserves_trailing_semicolon():
    # pglast.prettify strips a trailing ';'; prettify_sql restores it
    # so gen's psql-ready output is preserved.
    assert prettify_sql("SELECT a FROM t1;", color=False) == "SELECT a\nFROM t1;"


def test_prettify_no_semicolon_when_input_has_none():
    assert prettify_sql("SELECT a FROM t1", color=False) == "SELECT a\nFROM t1"


def test_prettify_no_trailing_newline():
    out = prettify_sql("SELECT a FROM t1", color=False)
    assert not out.endswith("\n")


def test_prettify_reformats_ddl_with_semicolon():
    # DDL reflow: one column per line, trailing commas (comma_at_eoln),
    # PRIMARY KEY on its own line, keywords upper-cased, and the trailing
    # ';' preserved so the output stays psql-ready. Locks the DDL contract
    # at the unit level (the CLI gen --pprint path covers it only
    # indirectly via the round-trip test).
    ddl = "CREATE TABLE foo (id bigint not null, name text, primary key(id));"
    out = prettify_sql(ddl, color=False)
    assert out == (
        "CREATE TABLE foo (\n"
        "  id bigint NOT NULL,\n"
        "  name text,\n"
        "  PRIMARY KEY (id)\n"
        ");"
    )


def test_prettify_color_true_emits_ansi():
    out = prettify_sql("SELECT a FROM t1", color=True)
    # pygments terminal output contains ANSI escape sequences.
    assert "\x1b[" in out


def test_prettify_color_false_has_no_ansi():
    out = prettify_sql("SELECT a FROM t1", color=False)
    assert "\x1b[" not in out


def test_prettify_color_true_no_trailing_newline():
    # pygments.highlight appends a newline; prettify_sql strips it.
    out = prettify_sql("SELECT a FROM t1", color=True)
    assert not out.endswith("\n")


def test_prettify_color_true_preserves_semicolon():
    out = prettify_sql("SELECT a FROM t1;", color=True)
    # Strip ANSI to check the underlying text ends with ';'.
    import re
    plain = re.sub(r"\x1b\[[0-9;]*m", "", out)
    assert plain.endswith(";")


def test_prettify_missing_pglast_raises_runtimeerror(monkeypatch):
    # Setting sys.modules['pglast'] = None makes `import pglast` raise
    # ImportError on the NEXT call — possible only because prettify_sql
    # imports pglast lazily inside the function body.
    monkeypatch.setitem(sys.modules, "pglast", None)
    with pytest.raises(RuntimeError, match=r"waxsql\[pprint\]"):
        prettify_sql("SELECT 1", color=False)


def test_prettify_missing_pygments_raises_runtimeerror(monkeypatch):
    # pglast present, pygments absent: only the color path needs pygments.
    # Setting pygments to None makes `from pygments import highlight` raise
    # ImportError (the first pygments import in the color branch).
    monkeypatch.setitem(sys.modules, "pygments", None)
    with pytest.raises(RuntimeError, match=r"waxsql\[pprint\]"):
        prettify_sql("SELECT 1", color=True)
