"""Pretty-printing for generated SQL: reformat + optional terminal color.

Display-only transformation behind `waxsql gen --pprint`. Reformats SQL
via pglast's parse-tree serializer and, when writing to a terminal,
colorizes via pygments. Deliberately NOT on the generation hot path —
this is a human-facing presentation layer, kept separate from the
canonical machine-pipe output (which stays plain so it can be re-fed to
`validate`).

Both pglast and pygments are optional (the `[pprint]` extra). They are
imported lazily inside `prettify_sql` so that importing this module —
and therefore `waxsql.cli` — never hard-requires them. Same discipline
as `validate/parse.py`'s lazy `import psycopg`.
"""
from __future__ import annotations

import os
from typing import TextIO


_INSTALL_HINT = (
    "--pprint requires pglast and pygments. "
    "Install with: pip install 'waxsql[pprint]'"
)


def should_colorize(stream: TextIO) -> bool:
    """True iff `stream` is an interactive terminal and color isn't
    opted out via the NO_COLOR convention (https://no-color.org).

    Only COLOR is gated on this — reformatting under --pprint always
    happens. The split is deliberate: `gen --pprint > file.sql` and
    `gen --pprint | psql` get clean, escape-code-free SQL, while an
    interactive terminal gets the colorized view.
    """
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty()) and os.environ.get("NO_COLOR") is None


def prettify_sql(sql: str, *, color: bool) -> str:
    """Reformat `sql` and optionally colorize it for terminal display.

    Reformatting uses ``pglast.prettify(sql, comma_at_eoln=True)`` —
    conventional trailing commas. Coloring uses pygments.

    Trailing-semicolon handling: pglast.prettify drops a trailing ``;``
    (and omits the final ``;`` of a multi-statement script). To preserve
    gen's psql-ready output, if the INPUT ends with ``;`` we restore one
    after reformatting and before coloring, so it colors uniformly with
    the rest of the statement.

    Returns the formatted string with no trailing newline.
    """
    try:
        import pglast
    except ImportError as e:
        raise RuntimeError(_INSTALL_HINT) from e

    had_semicolon = sql.rstrip().endswith(";")
    formatted = pglast.prettify(sql, comma_at_eoln=True)
    if had_semicolon:
        formatted += ";"

    if color:
        try:
            from pygments import highlight
            from pygments.formatters import Terminal256Formatter
            from pygments.lexers import PostgresLexer
        except ImportError as e:
            raise RuntimeError(_INSTALL_HINT) from e

        # pygments.highlight appends a trailing newline; strip it so the
        # caller controls line breaks between output segments.
        formatted = highlight(
            formatted, PostgresLexer(), Terminal256Formatter()
        ).rstrip("\n")

    return formatted


__all__ = ["prettify_sql", "should_colorize"]
