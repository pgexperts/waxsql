"""Tests for COPY block formatting. See waxsql/gen/data/emit.py."""
from __future__ import annotations

import datetime as _dt
import uuid

from decimal import Decimal

import pytest

from waxsql.gen.data.emit import (
    NULL_SENTINEL,
    encode_value,
    emit_copy_block,
)


def test_null_sentinel_is_pg_default():
    assert NULL_SENTINEL == r"\N"


def test_encode_none_returns_sentinel():
    assert encode_value(None) == NULL_SENTINEL


def test_encode_int_returns_decimal_string():
    assert encode_value(42) == "42"
    assert encode_value(-7) == "-7"


def test_encode_bool_returns_t_or_f():
    """PG accepts 't'/'f' (and 'true'/'false') in COPY; 't'/'f' is the canonical form."""
    assert encode_value(True) == "t"
    assert encode_value(False) == "f"


def test_encode_string_escapes_tab_newline_backslash():
    assert encode_value("hello") == "hello"
    assert encode_value("a\tb") == r"a\tb"
    assert encode_value("a\nb") == r"a\nb"
    assert encode_value("a\\b") == r"a\\b"
    assert encode_value("a\rb") == r"a\rb"


def test_encode_decimal_returns_canonical_string():
    assert encode_value(Decimal("3.14")) == "3.14"


def test_encode_uuid_returns_dashed_hex():
    u = uuid.UUID(int=0)
    assert encode_value(u) == str(u)


def test_encode_datetime_returns_iso_with_tz():
    dt = _dt.datetime(2025, 1, 2, 3, 4, 5, tzinfo=_dt.timezone.utc)
    assert encode_value(dt) == "2025-01-02 03:04:05+00:00"


def test_encode_date_returns_iso():
    assert encode_value(_dt.date(2025, 1, 2)) == "2025-01-02"


def test_encode_timedelta_returns_iso_interval():
    td = _dt.timedelta(days=1, seconds=2)
    # PG accepts ISO-8601 interval form `P1DT2S`. Easier to parse than
    # "1 day 0:00:02" and unambiguous.
    assert encode_value(td) == "P1DT2S"


def test_encode_dict_returns_json_string():
    import json
    d = {"a": 1, "b": "x"}
    assert encode_value(d) == json.dumps(d, separators=(",", ":"))


def test_encode_list_returns_pg_array_literal():
    # PG array literals in COPY text format: {1,2,3}
    assert encode_value([1, 2, 3]) == "{1,2,3}"
    assert encode_value([]) == "{}"


def test_emit_copy_block_basic_shape():
    rows = [(1, "alpha"), (2, "beta")]
    columns = ("id", "name")
    block = emit_copy_block("widgets", columns, rows)
    assert block.startswith('COPY "widgets" ("id", "name") FROM STDIN;\n')
    assert "1\talpha\n" in block
    assert "2\tbeta\n" in block
    assert block.rstrip().endswith(r"\.")


def test_emit_copy_block_handles_nulls():
    rows = [(1, None), (None, "x")]
    columns = ("id", "name")
    block = emit_copy_block("t", columns, rows)
    assert "1\t\\N\n" in block
    assert "\\N\tx\n" in block


def test_emit_copy_block_empty_rows():
    """Zero-row tables emit a valid (empty) COPY block."""
    block = emit_copy_block("t", ("id",), [])
    assert block.startswith('COPY "t" ("id") FROM STDIN;\n')
    assert block.rstrip().endswith(r"\.")


# ----- Two-layer escape behavior (issue #33) ----------------------------
# These tests pin the byte-level output for values that previously
# rendered incorrectly. The bug class is: array elements / dict values
# live inside a COPY cell, so they need BOTH array_in escapes (for the
# `{...}` syntax) and COPY row-format escapes (for the cell content),
# applied in the right order so the bytes survive both parsing passes.
#
# Today's strategy set never produces the trigger characters (no tabs/
# newlines in wordlist text, no array-of-text, no array elements with
# NULL allowed), so these are LATENT bugs — pinned here so a future
# strategy expansion can't reintroduce them silently.


def test_encode_dict_escapes_backslashes_for_copy():
    """JSON containing backslash-escapes (here: `\\t` from a tab in a
    string value) must have those `\\` chars re-escaped before being
    handed to COPY. Otherwise COPY misinterprets the leading `\\`
    as its own escape introducer and corrupts the JSON payload.
    """
    import json
    out = encode_value({"k": "a\tb"})
    inner = json.dumps({"k": "a\tb"}, separators=(",", ":"))
    # Every backslash in the JSON output is doubled for COPY.
    assert out == inner.replace("\\", "\\\\")


def test_array_of_strings_escapes_tab():
    """A tab inside a string array element must be backslash-escaped at
    the COPY layer (array_in keeps literal tabs as element content, but
    a literal tab in the COPY source would terminate the cell)."""
    out = encode_value(["a\tb"])
    # Bytes: { " a \ t b " } — 9 characters.
    assert out == r'{"a\tb"}'


def test_array_of_strings_escapes_newline_and_cr():
    assert encode_value(["a\nb"]) == r'{"a\nb"}'
    assert encode_value(["a\rb"]) == r'{"a\rb"}'


def test_array_of_strings_escapes_backslash():
    """One source backslash produces FOUR backslashes in the COPY
    bytes: array_in's escape doubles to two, then COPY's doubling
    redoubles to four. PG round-trip then halves back to one."""
    out = encode_value(["a\\b"])  # actual string: a, \, b (3 chars)
    expected = "{" + '"' + "a" + "\\" * 4 + "b" + '"' + "}"
    assert out == expected


def test_array_of_strings_escapes_double_quote():
    """A `"` inside a string element becomes `\\\\"` in COPY bytes:
    array_in's escape `\\"` is one `\\` plus the quote; COPY then
    doubles that backslash to two. PG resolution: COPY emits one
    backslash plus quote; array_in resolves that to a literal `"`."""
    out = encode_value(['a"b'])
    # Expected bytes between the wrapping " and ": a, \, \, ", b
    expected = "{" + '"' + "a" + "\\" * 2 + '"' + "b" + '"' + "}"
    assert out == expected


def test_array_with_none_emits_bare_null_token():
    """Inside an array literal, NULL is the unquoted token `NULL`,
    not the row-level `\\N` sentinel. The element column today is
    forced nullable=False so this path is latent, but a future
    strategy allowing NULL elements would have silently emitted
    the literal 2-character string `\\N` for every null."""
    assert encode_value([1, None, 3]) == "{1,NULL,3}"


def test_array_of_dicts_quotes_and_escapes_json():
    """Array-of-jsonb: each element is the JSON-encoded value,
    wrapped as a quoted array element with two-layer escape so
    PG's jsonb_in receives intact JSON after COPY+array_in
    resolution. The JSON output `{"a":1}` contains `"` chars
    that need array_in-level escapes, which then get COPY-level
    doubling."""
    out = encode_value([{"a": 1}, {"a": 2}])
    # For `{"a":1}`: array_in escapes turn the 2 `"` into `\"`
    # (2 chars each), then COPY doubles each `\` to `\\`.
    # Per element: { \\ " a \\ " : 1 } wrapped in " ... "
    elem1 = '"{' + "\\" * 2 + '"a' + "\\" * 2 + '":1}"'
    elem2 = '"{' + "\\" * 2 + '"a' + "\\" * 2 + '":2}"'
    assert out == "{" + elem1 + "," + elem2 + "}"


def test_nested_array_uses_multidim_syntax_not_quoted():
    """PG arrays are natively multidimensional: `{{1,2},{3,4}}`.
    A nested list element must emit as a sub-array literal, NOT
    as a quoted string containing `{1,2}` — that would make the
    inner array a STRING element of the outer one, which is the
    opposite of what was modeled."""
    assert encode_value([[1, 2], [3, 4]]) == "{{1,2},{3,4}}"


def test_array_of_scalars_unquoted():
    """Regression: numeric and bool elements remain unquoted (PG's
    array_in accepts the bare tokens via int4_in / bool_in)."""
    assert encode_value([1, 2, 3]) == "{1,2,3}"
    assert encode_value([True, False]) == "{t,f}"
    assert encode_value([1.5, 2.5]) == "{1.5,2.5}"


def test_encode_dict_rejects_non_finite_floats():
    """inf/nan would serialize as Infinity/NaN — invalid JSON that jsonb_in
    rejects. encode_value must fail at generation time (allow_nan=False),
    not produce a COPY block that explodes at load time (#51)."""
    with pytest.raises(ValueError):
        encode_value({"k": float("inf")})
    with pytest.raises(ValueError):
        encode_value({"k": float("nan")})


def test_array_of_dicts_rejects_non_finite_floats():
    """Same guard on the array-element JSON path."""
    with pytest.raises(ValueError):
        encode_value([{"k": float("nan")}])
