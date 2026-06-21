"""COPY block formatting.

PostgreSQL's text-format COPY uses tab as the column separator, `\\N`
as the NULL sentinel, and backslash-escapes for tab, newline, carriage
return, and backslash itself. This module owns all of that; strategies
return native Python values and the emitter formats them.

Role in the system: this is the bottom of the value-rendering stack.
Everything above it (strategies, row materializer) traffics in native
Python objects; nothing above this module knows anything about tab
encoding, NULL sentinels, COPY framing, or PG array literal syntax.
That separation lets the strategy registry stay trivially testable
(no string round-trips needed to compare values).
"""
from __future__ import annotations

import datetime as _dt
import json
import uuid
from decimal import Decimal
from collections.abc import Iterable, Sequence


# PG text-format COPY: the literal two-character sequence `\N` (backslash-N)
# in an unescaped position is the NULL marker. Raw string so the backslash
# stays a backslash — not an escape introducer for Python.
NULL_SENTINEL = r"\N"

# COPY text-format escape rules. Order matters: backslash MUST come
# first so we don't double-escape escapes we just inserted. If `\t`
# came first, the subsequent backslash pass would turn the `\` in
# `\t` into `\\`, producing `\\t` instead of the intended `\t`.
_ESCAPES: tuple[tuple[str, str], ...] = (
    ("\\", r"\\"),
    ("\t", r"\t"),
    ("\n", r"\n"),
    ("\r", r"\r"),
)


def _escape_text(s: str) -> str:
    # Linear pass through the ordered _ESCAPES table. str.replace is
    # already optimized in CPython; rolling our own char-by-char loop
    # would be slower and no clearer.
    for raw, escaped in _ESCAPES:
        s = s.replace(raw, escaped)
    return s


def encode_value(v: object) -> str:
    """Encode a Python value for PG's text-format COPY.

    Returns the COPY representation of `v`. NULL is represented by
    `\\N`; bools as `t`/`f`; datetimes/dates as ISO-8601 (with `+00:00`
    suffix for timestamptz); intervals as ISO-8601 duration; dicts as
    compact JSON; lists as PG array literals.

    Dispatch ORDER is load-bearing — see the bool-before-int note below
    and the str-after-everything note for the array element path.
    """
    if v is None:
        return NULL_SENTINEL
    if isinstance(v, bool):
        # bool MUST come before int — bool is a subclass of int in
        # Python (isinstance(True, int) is True), so an earlier int
        # branch would swallow booleans and emit them as "1"/"0".
        return "t" if v else "f"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, Decimal):
        # Decimal __str__ is canonical (no trailing zeros stripping, no
        # scientific notation for reasonable values); PG numeric parses
        # it directly.
        return str(v)
    if isinstance(v, uuid.UUID):
        return str(v)
    if isinstance(v, _dt.datetime):
        # str() on tz-aware datetime gives "YYYY-MM-DD HH:MM:SS+HH:MM"
        # which is what PG accepts for timestamptz literals.
        # Note: the date branch below is reached only by naïve `date`
        # values — datetime is a subclass of date, so this branch
        # MUST come before the date branch.
        return str(v)
    if isinstance(v, _dt.date):
        return v.isoformat()
    if isinstance(v, _dt.timedelta):
        # ISO-8601 duration: PnDTnS. Limited to days/seconds because
        # timedelta doesn't carry month/year parts. PG accepts this.
        # If we ever need year/month resolution we'll have to leave
        # timedelta behind for a richer carrier type.
        days = v.days
        seconds = v.seconds
        return f"P{days}DT{seconds}S"
    if isinstance(v, dict):
        # Compact JSON (no spaces) keeps COPY output narrow and matches
        # what PG emits for jsonb on dump. separators kwarg is required;
        # the default json.dumps inserts ", " and ": " padding.
        #
        # The result is then run through `_escape_text` because
        # json.dumps emits backslash-escape sequences (`\t`, `\n`, `\"`,
        # `\uXXXX`) for control chars, quotes, and non-ASCII. Those
        # backslashes mean nothing to JSON's *reader* (it resolves
        # them) but everything to COPY's *reader* (which uses
        # backslash as its own escape introducer). Without
        # re-escaping, a JSON value containing e.g. an embedded tab
        # would arrive at PG with the `\` already consumed by COPY,
        # leaving invalid JSON for jsonb_in. _escape_text doubles
        # every `\` so COPY resolves them back to single `\` before
        # handing the JSON to PG.
        # allow_nan=False: inf/nan would serialize as Infinity/NaN, which
        # jsonb_in rejects. Fail at generation time, not COPY-load time.
        return _escape_text(json.dumps(v, separators=(",", ":"), allow_nan=False))
    if isinstance(v, list):
        # PG text-format array literal: `{elem,elem,...}`. Per-element
        # formatting goes through `_array_element`, which knows the
        # two-layer escape rules required for strings/dicts inside an
        # array literal that itself sits inside a COPY cell.
        return "{" + ",".join(_array_element(e) for e in v) + "}"
    if isinstance(v, str):
        return _escape_text(v)
    raise TypeError(f"no COPY encoder for type {type(v).__name__}: {v!r}")


def _array_element(v: object) -> str:
    """Format a single element for inclusion in a PG array literal.

    A PG array literal lives INSIDE a COPY cell, so two parsing
    layers apply to its bytes:

      1. PG's COPY row reader resolves row-format escapes first
         (`\\\\`, `\\t`, `\\n`, `\\r`).
      2. PG's array_in then parses the resolved cell text, with
         its own escape grammar for quoted elements: `\\\\` resolves
         to `\\`, `\\"` resolves to `"`. (Note: array_in does NOT
         recognize `\\t`/`\\n`/`\\r` — anywhere it sees a literal
         tab/newline/CR inside a quoted element, it just keeps
         the character as part of the string.)

    String and dict elements need BOTH escape passes; numeric/bool/
    datetime elements need NEITHER (PG's array_in accepts them
    unquoted via the element type's own input function). Lists
    recurse for multidimensional arrays — they do NOT get quoted,
    because PG's array literal supports `{{1,2},{3,4}}` directly.

    None becomes the bare token `NULL` (case-insensitive), NOT the
    row-level `\\N` sentinel — inside a `{...}` literal, `\\N`
    is interpreted as the two-character string `\\N`, not SQL NULL.
    """
    if v is None:
        # Bare unquoted token: PG's array_in treats this as SQL NULL.
        # The row-level `\N` sentinel would be misread as the literal
        # 2-char string, not NULL — a silent data corruption.
        return "NULL"
    if isinstance(v, list):
        # Multidimensional array: emit the inner `{...}` literal
        # WITHOUT quoting, so the outer array_in sees it as a
        # sub-array, not a string element. Today `array_of(array_of(
        # ...))` isn't constructible by the data generator, but the
        # path is correct for the eventual extension.
        return encode_value(v)
    if isinstance(v, dict):
        # Array-of-jsonb: each element is a JSON-encoded value
        # wrapped as a quoted array-element. PG's array_in extracts
        # the quoted content (resolving its escapes) and hands the
        # resulting JSON text to jsonb_in for parsing. Two-layer
        # escape handles backslashes from JSON's own escapes plus
        # the quote-and-escape required by the array element format.
        return _quote_array_element(json.dumps(v, separators=(",", ":"), allow_nan=False))
    if isinstance(v, str):
        return _quote_array_element(v)
    # Scalar non-string types (int, float, bool, Decimal, UUID,
    # datetime, date, timedelta): PG's array_in accepts these as
    # unquoted tokens — quoting would actually be wrong here,
    # because array_in would then call the element type's input
    # function on a *string* form rather than the bare token.
    return encode_value(v)


def _quote_array_element(s: str) -> str:
    """Quote-and-escape a string for use as a PG array literal element.

    Encodes in REVERSE order of how PG's parsers resolve the bytes:
    array_in's escapes are applied FIRST in our code (because they
    resolve LAST when PG reads), then COPY's escapes are applied
    LAST in our code (because they resolve FIRST when PG reads).

    Concretely:

      * Step 1 (array_in escapes): `\\\\` → `\\\\\\\\` (one backslash
        becomes two), `"` → `\\"` (one quote becomes backslash-quote).
        These are the only escapes array_in recognizes inside a
        double-quoted element.

      * Step 2 (COPY escapes via `_escape_text`): re-escapes every
        backslash (so each `\\\\` from step 1 becomes `\\\\\\\\`,
        and any pre-existing literal backslash gets doubled) AND
        backslash-escapes tab/newline/CR characters in the source
        (so they don't terminate the COPY cell prematurely on
        reading; once resolved by COPY they become literal
        whitespace inside the quoted element, which array_in
        keeps verbatim).

    Composition example: a source `\\` (one backslash) → step 1
    → `\\\\` (two) → step 2 → `\\\\\\\\` (four) → wrapped in `"..."`.
    PG resolution: COPY reads four backslashes → resolves to two
    → array_in reads two backslashes → resolves to one. Round-trip
    preserves the original character.
    """
    arr_escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    copy_escaped = _escape_text(arr_escaped)
    return '"' + copy_escaped + '"'


def emit_copy_block(
    table_name: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[object]],
) -> str:
    """Format one COPY block.

    Output shape:
        COPY "table" ("col1", "col2") FROM STDIN;
        v11<TAB>v12
        v21<TAB>v22
        \\.

    Identifiers are double-quoted to handle case-sensitivity and
    reserved-word collisions uniformly. PG accepts quoted identifiers
    that happen not to need quoting, so the extra punctuation is harmless.

    The empty-rows case (no row tuples) is intentionally legal: it
    produces `COPY ... FROM STDIN;\\n\\.\\n` — header immediately
    followed by terminator. psql and psycopg both accept this; the
    `waxsql gen --rows=0` flag relies on it.
    """
    col_list = ", ".join(f'"{c}"' for c in columns)
    # Build via parts + join: O(n) regardless of row count, vs O(n^2)
    # for repeated string += concatenation.
    parts = [f'COPY "{table_name}" ({col_list}) FROM STDIN;\n']
    for row in rows:
        parts.append("\t".join(encode_value(v) for v in row))
        parts.append("\n")
    # PG COPY text-format end-of-data sentinel: a line containing only
    # `\.` (the same literal sequence used in psql). Followed by a
    # newline so concatenated blocks separate cleanly.
    parts.append("\\.\n")
    return "".join(parts)
