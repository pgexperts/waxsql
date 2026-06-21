"""Command-line interface to waxsql.

Three subcommands mirror the library's halves:
  - gen:      produce a random schema + query at a chosen complexity / seed
              (optionally with COPY blocks for table data)
  - data:     produce only COPY blocks for a deterministic schema
  - validate: run SQL through the SYNTAX, PARSE, or PLAN tier

This module is loaded by the `waxsql` console_scripts entry point in
pyproject.toml. The CLI's runtime dependency on `click` is declared via
the optional [cli] extra; a friendly install hint is printed before any
click code runs if the dep is missing, so a user who installed plain
`waxsql` and ran the command gets a clear message instead of a raw
ImportError traceback.

Role: this is the user-facing pipe surface. Everything here is glue —
flag parsing, stream routing, header round-tripping, error formatting.
No generator logic lives here; if you find yourself reaching for one,
the right answer is almost always "import it from `waxsql.*`".
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

# ----- Friendly missing-dep check ---------------------------------------
# Must run before any @click decorator is evaluated. If [cli] isn't
# installed, exit 3 with an install hint rather than letting the
# ModuleNotFoundError propagate up through the console-script wrapper.
try:
    import click
except ImportError:
    sys.stderr.write(
        "waxsql: the CLI requires the 'click' package, which ships with the\n"
        "[cli] optional extra. Install with:\n\n"
        "    pip install 'waxsql[cli]'\n\n"
    )
    sys.exit(3)

import contextlib
import random as _random
import re
from typing import Any, Optional

from waxsql import __version__, generate_query, generate_schema, print_query

# ----- Header format ----------------------------------------------------
# The `gen` subcommand prefixes its output with a header describing the
# seed and complexity used; the `validate` subcommand's --auto-schema
# feature parses that header back to regenerate the matching schema.
#
# Base format (stable across CLI versions):
#
#     -- waxsql <version>  seed=<int>  complexity=<int>
#
# When data generation is requested, additional keys follow:
#
#     -- waxsql <version>  seed=<int>  complexity=<int>  with-data=true  rows=<int>  fanout=<int>  null-fraction=<float>
#
# The regex matches the prefix loosely and we extract key=value pairs
# post-hoc, which lets unknown future keys be ignored gracefully rather
# than causing a parse failure. The required keys (seed, complexity) are
# still enforced after extraction.


@dataclass(frozen=True)
class Header:
    """Parsed contents of a `-- waxsql ... seed=N complexity=K ...` line.

    Older headers had only seed and complexity; newer headers add keys for
    data generation. Missing keys take sensible defaults so old gen output
    continues to parse cleanly.

    The defaults here (with_data=False, rows=100, fanout=5, null_fraction=0.05)
    deliberately match the defaults of `generate_data` and the gen subcommand
    flags. That alignment is what lets a header produced by an older version
    (without data keys) be replayed by a newer validator without surprises.
    """

    seed: int
    complexity: int
    with_data: bool = False
    rows: int = 100
    fanout: int = 5
    null_fraction: float = 0.05


# Matches "-- waxsql X.Y.Z" followed by anything — we parse key=value pairs
# from the captured rest rather than encoding the full format in the regex.
# This is deliberately loose: the strictness lives in the required-key check
# inside _parse_header, not in the regex. Future unknown keys are simply
# ignored, so older validators reading newer gen output won't reject it.
_HEADER_RE = re.compile(r"^--\s+waxsql\s+\S+(?P<rest>.*)$")
# Key names are lowercase letters and hyphens only (e.g. `null-fraction`).
# Values are non-whitespace tokens — no spaces allowed in values today;
# the format predates any value that would need it.
_KV_RE = re.compile(r"(?P<k>[a-z-]+)=(?P<v>\S+)")


def _render_header(
    seed: int,
    complexity: int,
    *,
    with_data: bool = False,
    rows: int = 100,
    fanout: int = 5,
    null_fraction: float = 0.05,
) -> str:
    """Render the gen-output header. Round-trips with `_parse_header`.

    When `with_data` is False (the historical default), only seed and
    complexity are emitted so existing gen output remains byte-stable.
    Data keys are appended only when they carry meaning.
    """
    parts = [f"-- waxsql {__version__}", f"seed={seed}", f"complexity={complexity}"]
    if with_data:
        parts.extend([
            "with-data=true",
            f"rows={rows}",
            f"fanout={fanout}",
            f"null-fraction={null_fraction}",
        ])
    return "  ".join(parts)


def _parse_header(text: str) -> Optional[Header]:
    """Parse a waxsql header from the first non-empty line of `text`.

    Returns None if the first non-empty line isn't a recognizable waxsql
    header (the common case — most SQL fed to the validator won't have one).
    Tolerates leading blank lines so a header preceded by whitespace parses.

    Returns a `Header` with defaults filled in for any missing optional keys
    so old gen output (seed+complexity only) continues to parse cleanly.
    Requires seed and complexity to be present; returns None otherwise.
    """
    for raw_line in text.splitlines():
        # Defense-in-depth: strip trailing whitespace (including any \r
        # that survived a non-canonical line-ending normalization in the
        # pipeline). splitlines normally handles CRLF cleanly, but if
        # the input was decoded twice or assembled from bytes irregularly,
        # a lingering \r would have made `int("5\r")` raise. Stripping is
        # cheaper than auditing every input path that could feed us text.
        line = raw_line.rstrip()
        if not line.strip():
            continue
        m = _HEADER_RE.match(line)
        if not m:
            return None
        kv = {kv_m.group("k"): kv_m.group("v") for kv_m in _KV_RE.finditer(m.group("rest"))}
        if "seed" not in kv or "complexity" not in kv:
            return None
        # Conversions wrapped so a malformed-but-regex-matching header
        # (e.g. a hand-edited `seed=abc` or `rows=many`) falls through
        # to None rather than raising ValueError up to the caller. The
        # validator's no-header path handles None gracefully; an
        # uncaught ValueError here would crash the CLI with a traceback
        # that hides the real "I tried to parse your header and gave up"
        # diagnostic.
        try:
            return Header(
                seed=int(kv["seed"]),
                complexity=int(kv["complexity"]),
                with_data=kv.get("with-data", "false").lower() == "true",
                rows=int(kv.get("rows", 100)),
                fanout=int(kv.get("fanout", 5)),
                null_fraction=float(kv.get("null-fraction", 0.05)),
            )
        except ValueError:
            return None
    return None


# ----- DSN redaction ----------------------------------------------------
# Connection-failure error messages echo the DSN to stderr for diagnostics.
# That stream may be captured by logs or piped through less-trusted
# tools, so we mask any password material before printing. The user
# supplied the secret themselves, but echoing it verbatim is a small,
# avoidable exposure — and matches the discipline psycopg's own
# `conninfo` redaction follows.
#
# Two patterns handle the two DSN forms psycopg accepts:
#   - key-value: `dbname=x password=secret host=h`
#   - URI:       `postgresql://user:secret@host/db`
#
# Conservative: doesn't try to handle quoted values (`password='se cret'`)
# because no reasonable user/CI tool produces those; the fallthrough
# leaves a quote-wrapped password partially exposed, which is still
# strictly better than the current "echo verbatim" baseline.
_DSN_PASSWORD_KV_RE = re.compile(r"(password=)\S+")
_DSN_PASSWORD_URI_RE = re.compile(r"(://[^:/@]*:)[^@]+(@)")


def _redact_dsn(dsn: str) -> str:
    """Replace password fields in a DSN with `***` for safe logging.

    Used on the connect-failure error message that goes to stderr.
    Other DSN components (host, dbname, user, options, application_name)
    are preserved intact — they're useful for diagnostics and don't
    constitute credentials.
    """
    dsn = _DSN_PASSWORD_KV_RE.sub(r"\1***", dsn)
    dsn = _DSN_PASSWORD_URI_RE.sub(r"\1***\2", dsn)
    return dsn


def _resolve_schema_source(
    *,
    input_text: str,
    schema_from: Optional[str],
    schema_seed: Optional[int],
    schema_complexity: int,
    auto_schema: bool,
) -> Optional[str]:
    """Decide which DDL to install for parse/plan validation.

    Returns the DDL string (a CREATE TABLE / ALTER TABLE / CREATE INDEX
    script) or None if no schema source was supplied. Precedence:

      1. --schema-from PATH  (most explicit — caller provided a file)
      2. --schema-seed N     (explicit regenerate, uses --schema-complexity)
      3. --auto-schema parses the input's header and regenerates from it
         (convenience: gen | validate pipelines just work, no extra flags)
      4. None of the above → return None (caller decides if that's fatal)

    Pure beyond reading --schema-from PATH. Raises FileNotFoundError if
    the path doesn't exist (let it propagate; click will format it as a
    usage error).

    All parameters are keyword-only — with six arguments and an order-sensitive
    precedence chain, positional calls would be error-prone and unreadable.
    """
    if schema_from is not None:
        with open(schema_from, encoding="utf-8") as f:
            return f.read()

    if schema_seed is not None:
        return generate_schema(
            seed=schema_seed, complexity=schema_complexity
        ).emit_ddl()

    if auto_schema:
        parsed = _parse_header(input_text)
        if parsed is not None:
            return generate_schema(
                seed=parsed.seed, complexity=parsed.complexity
            ).emit_ddl()

    return None


def _split_copy_blocks(data_sql: str) -> list[str]:
    """Split a `generate_data` output string into individual COPY blocks.

    Each block is `COPY ... FROM STDIN;\\n<rows>\\n\\.\\n`. They're
    separated by blank lines in the combined stream. Blocks are collected
    by accumulating lines until the `\\.` terminator is found, so partial
    or empty COPY blocks (rows=0) are handled correctly.

    Leading blank lines between blocks are skipped so every returned block
    starts with its COPY header line. This matters because `_execute_copy_block`
    treats `lines[0]` as the COPY statement to pass to `cur.copy()`.

    We split on the `\\.` terminator rather than on blank-line boundaries
    because COPY data rows can in principle contain blank-looking content
    (a row with all empty strings would render as a single tab character).
    Anchoring on `\\.` is grammatically correct for PG text-format COPY.
    """
    blocks: list[str] = []
    current: list[str] = []
    for line in data_sql.splitlines(keepends=True):
        # Skip blank lines between blocks (don't accumulate into current).
        # Once `current` has any content, blank lines are treated as
        # row separators within the block (rare but possible) and kept.
        if not current and not line.strip():
            continue
        current.append(line)
        if line.strip() == r"\.":
            blocks.append("".join(current))
            current = []
    # A non-empty `current` here means the stream ended mid-block: the
    # final COPY block never hit its `\.` terminator. generate_data always
    # terminates every block it emits, so this only fires on truncated or
    # hand-edited input — raise rather than silently returning fewer blocks
    # than the caller will go on to load (which would look like "some rows
    # just didn't show up" with no error).
    if current:
        raise ValueError("unterminated COPY block: missing '\\.' terminator")
    return blocks


def _execute_copy_block(cur: Any, block: str) -> None:
    """Execute one COPY block via psycopg's `copy()` context manager.

    The block's first line is `COPY "t" (...) FROM STDIN;`; the body is
    tab-encoded rows; the terminator is `\\.`. We strip the terminator
    before feeding because psycopg's `copy()` provides its own end-of-data
    marker — sending `\\.` as a data row would corrupt the stream.

    The `cur` parameter is untyped because psycopg is an optional dep and
    a forward reference to `psycopg.Cursor` would require a TYPE_CHECKING
    guard; `Any` is the same pattern used for `check_fn` above.

    Side effect: data rows are inserted into the table named by the COPY
    header. The caller must hold a transaction (or a savepoint) around
    this call; the validate command wraps each block in a SAVEPOINT for
    granular error reporting.
    """
    lines = block.splitlines()
    # Strip trailing `;` — psycopg's `cur.copy(stmt)` doesn't want the
    # statement terminator (it's not a normal SQL execute).
    header = lines[0].rstrip(";")
    # Drop the trailing `\.` terminator; psycopg handles end-of-stream.
    # Using a filter here means stray `\.` lines anywhere in the block
    # would be dropped, but `_split_copy_blocks` guarantees there's
    # exactly one terminator at the end.
    body_lines = [line for line in lines[1:] if line.strip() != r"\."]
    with cur.copy(header) as copy:
        for line in body_lines:
            copy.write(line + "\n")


def _extract_first_select(text: str) -> str:
    """Pull the first SELECT or WITH statement out of `text`.

    The gen subcommand emits its output as multiple `;`-terminated
    statements (CREATE TABLE ...; ALTER TABLE ...; SELECT ...;); the
    parse/plan validators take a single statement at a time. This
    helper finds the first SELECT/WITH and returns it through its
    trailing `;` (exclusive). If no SELECT/WITH is found, returns the
    whole text unchanged so a hand-written single-statement input
    still works without special-casing.

    The trailing semicolon is stripped because PREPARE and EXPLAIN
    both reject a statement-terminating `;` — the validator supplies
    its own statement boundary.
    """
    lines = text.splitlines()
    select_start = None
    # Use a word-boundary match so that identifiers starting with 'SELECT'
    # or 'WITH' (e.g. 'SELECTED', 'WITHOUT') don't accidentally trigger
    # extraction. re.match is anchored at the start of the stripped line.
    _select_re = re.compile(r"(?i)(SELECT|WITH)\b")
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if _select_re.match(stripped):
            select_start = i
            break
    if select_start is None:
        return text  # no SELECT/WITH found; pass through

    body = "\n".join(lines[select_start:])
    # Strip trailing semicolon — PREPARE/EXPLAIN don't want one.
    body = body.rstrip().rstrip(";").rstrip()
    return body


@click.group()
@click.version_option(version=__version__, prog_name="waxsql")
def main() -> None:
    """Random PostgreSQL query generator."""


# Subcommands (gen, validate) are registered in subsequent tasks.


def _pick_random_seed() -> int:
    """Pick a fresh seed for `waxsql gen` with no `--seed` flag.

    Uses SystemRandom (backed by os.urandom) rather than the global random
    module to avoid crossing streams with any downstream RNG users. The range
    matches what `random.Random(seed)` accepts comfortably (non-negative 63-bit
    int), and it's large enough that collision across independent invocations
    is astronomically unlikely.

    This is the ONLY place in the codebase that intentionally consumes
    OS entropy. Everywhere else in waxsql is strictly deterministic; this
    function exists precisely to bootstrap a fresh deterministic seed
    when the user didn't supply one. The seed is then echoed in the
    output header so the run remains reproducible.
    """
    return _random.SystemRandom().randint(0, 2**63 - 1)


def _apply_pprint(sql: str, *, color: bool) -> str:
    """Pretty-print `sql` for `gen --pprint`, converting the pretty
    module's optional-dependency RuntimeError into a friendly CLI exit
    (exit 3, matching the click missing-dep guard at module top)."""
    from waxsql.pretty import prettify_sql

    try:
        return prettify_sql(sql, color=color)
    except RuntimeError as e:
        click.echo(str(e), err=True)
        sys.exit(3)


@main.command("gen")
@click.option(
    "-s", "--seed", type=int, default=None,
    help="RNG seed. If omitted, picks one randomly and prints it in the "
         "output header so the run is reproducible.",
)
@click.option(
    "-c", "--complexity", type=click.IntRange(0, 10), default=5, show_default=True,
    help="Complexity dial 0..10.",
)
@click.option(
    "--schema-only", is_flag=True, default=False,
    help="Emit only the CREATE TABLE script (no query).",
)
@click.option(
    "--query-only", is_flag=True, default=False,
    help="Emit only the SELECT statement (no schema DDL).",
)
@click.option(
    "-n", "--count", type=click.IntRange(min=1), default=1, show_default=True,
    help="Number of queries to emit against the same schema. "
         "Each gets its own query seed starting from --seed.",
)
@click.option(
    "--no-header", is_flag=True, default=False,
    help="Suppress the leading `-- waxsql seed=N complexity=X` header line.",
)
@click.option(
    "--with-data", "with_data", is_flag=True, default=False,
    help="Emit COPY blocks for table data between DDL and queries. "
         "Ignored when --schema-only or --query-only is set.",
)
@click.option(
    "--rows", type=click.IntRange(min=0), default=100, show_default=True,
    help="Base row count per table when --with-data is set. "
         "0 emits empty COPY blocks (header + immediate terminator).",
)
@click.option(
    "--fanout", type=click.IntRange(min=1), default=5, show_default=True,
    help="FK-depth row multiplier when --with-data is set.",
)
@click.option(
    "--null-fraction", "null_fraction", type=float, default=0.05, show_default=True,
    help="Per-nullable-column NULL probability when --with-data is set.",
)
@click.option(
    "--pprint", is_flag=True, default=False,
    help="Reformat and (on a terminal) colorize the generated SQL for "
         "human reading. Display-only — not for piping into validate. "
         "Requires the [pprint] extra.",
)
def gen(
    seed: Optional[int],
    complexity: int,
    schema_only: bool,
    query_only: bool,
    count: int,
    no_header: bool,
    with_data: bool,
    rows: int,
    fanout: int,
    null_fraction: float,
    pprint: bool,
) -> None:
    """Generate a random schema and/or query against it.

    By default emits both: schema DDL first, then the query, both
    semicolon-terminated and psql-ready. --schema-only / --query-only
    are mutually exclusive escape hatches. --count N produces N queries
    against the same schema (seeds are seed, seed+1, ..., seed+N-1), so
    `--seed S -n 3` always yields the same three-query batch.

    --with-data inserts COPY blocks between the DDL and queries, producing
    a fully self-contained psql script. It is silently ignored when
    --schema-only or --query-only is set, because the combined stream
    (header + DDL + data + query) requires both halves to be present.
    """
    if schema_only and query_only:
        raise click.UsageError(
            "--schema-only and --query-only are mutually exclusive."
        )

    # --with-data only makes sense in the combined stream (DDL + data +
    # queries). Silently suppress it for partial outputs so callers don't
    # need to remember to also add --schema-only flags when building pipelines.
    emit_data = with_data and not schema_only and not query_only

    # --pprint reformats always; color is gated on an interactive
    # terminal. should_colorize needs no optional deps (just isatty +
    # NO_COLOR); the pglast/pygments requirement surfaces later when
    # _apply_pprint actually calls prettify_sql.
    pprint_color = False
    if pprint:
        from waxsql.pretty import should_colorize
        pprint_color = should_colorize(sys.stdout)

    if seed is None:
        seed = _pick_random_seed()

    schema = generate_schema(seed=seed, complexity=complexity)

    out: list[str] = []
    if not no_header:
        out.append(_render_header(
            seed, complexity,
            with_data=emit_data, rows=rows, fanout=fanout,
            null_fraction=null_fraction,
        ))

    if not query_only:
        out.append("-- schema:")
        ddl = schema.emit_ddl().rstrip()
        if pprint:
            ddl = _apply_pprint(ddl, color=pprint_color)
        out.append(ddl)

    if emit_data:
        # COPY blocks go between DDL and queries. We flush `out` first so
        # click.echo handles the DDL, then emit the COPY text with nl=False
        # (the generate_data output already ends with "\n"), then continue
        # building the query section.
        from waxsql.data import generate_data
        click.echo("\n".join(out))
        out = []
        click.echo()  # blank line separating DDL from COPY blocks
        try:
            data_text = generate_data(
                schema, seed=seed, rows=rows, fanout=fanout,
                null_fraction=null_fraction,
            )
        except ValueError as e:
            # ValueError from generate_data is almost always an FK cycle
            # at complexity ≥ 8 (the schema generator can produce them;
            # the data generator can't yet untangle them via deferred
            # constraints + UPDATE patches). Catch here and re-raise as
            # a usage error rather than a stack trace.
            click.echo(
                f"waxsql gen: cannot generate data for this schema: {e}\n"
                f"This typically means the schema has FK cycles, which "
                f"data generation does not yet support. Try a lower --complexity.",
                err=True,
            )
            sys.exit(1)
        click.echo(data_text, nl=False)

    if not schema_only:
        for i in range(count):
            if not query_only or i > 0:
                # Blank separator: between schema (or data) and queries
                # (i==0 with a schema present), and between consecutive queries.
                out.append("")
            label = "-- query:" if count == 1 else f"-- query {i + 1}/{count}"
            out.append(label)
            q = generate_query(
                seed=seed + i, schema=schema, complexity=complexity,
            )
            q_sql = print_query(q) + ";"
            if pprint:
                q_sql = _apply_pprint(q_sql, color=pprint_color)
            out.append(q_sql)

    click.echo("\n".join(out))


@main.command("validate")
@click.argument(
    "sql_file", type=click.Path(exists=True, dir_okay=False, allow_dash=True),
    required=False, default="-",
)
@click.option(
    "-t", "--tier",
    type=click.Choice(["syntax", "parse", "plan"], case_sensitive=False),
    default="syntax", show_default=True,
    help="Validation tier.",
)
@click.option(
    "--dsn", default=None,
    help="psycopg DSN for parse/plan tiers. Defaults to $WAXSQL_PG_DSN "
         "or 'dbname=waxsql_test'.",
)
@click.option(
    "--schema-from", "schema_from", type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Install DDL from this file before validating (parse/plan only).",
)
@click.option(
    "--schema-seed", "schema_seed", type=int, default=None,
    help="Regenerate schema from seed instead (parse/plan only).",
)
@click.option(
    "--schema-complexity", "schema_complexity",
    type=click.IntRange(0, 10), default=5, show_default=True,
    help="Companion to --schema-seed.",
)
@click.option(
    "--auto-schema/--no-auto-schema", "auto_schema",
    default=True, show_default=True,
    help="If input begins with a `-- waxsql seed=N complexity=X` header, "
         "regenerate that schema automatically.",
)
@click.option(
    "-v", "--verbose", is_flag=True, default=False,
    help='Print "OK" on success (silent by default, Unix style).',
)
def validate(
    sql_file: str,
    tier: str,
    dsn: Optional[str],
    schema_from: Optional[str],
    schema_seed: Optional[int],
    schema_complexity: int,
    auto_schema: bool,
    verbose: bool,
) -> None:
    """Validate SQL through the SYNTAX, PARSE, or PLAN tier.

    Reads SQL from SQL_FILE, or from stdin if SQL_FILE is omitted or '-'.
    SYNTAX is the default tier (no DB required, fast). PARSE and PLAN
    tiers require a live PG connection and a schema to install. The
    gen-output header (`-- waxsql seed=N complexity=X`) is parsed
    automatically by default so `gen | validate` just works.
    """
    # ----- Read input --------------------------------------------------------
    # click.Path with allow_dash=True does NOT auto-read stdin; we do it here.
    # The distinction between None and '-' matters: both mean stdin, but the
    # user can also explicitly pass '-' as the file argument.
    if sql_file == "-":
        sql_text = click.get_text_stream("stdin").read()
    else:
        with open(sql_file, encoding="utf-8") as f:
            sql_text = f.read()

    # ----- SYNTAX tier -------------------------------------------------------
    if tier.lower() == "syntax":
        try:
            from waxsql.validate.syntax import check_syntax
        except ImportError:
            # pglast isn't installed — shouldn't happen in normal use (it's a
            # required dep of waxsql itself), but we handle it gracefully for
            # completeness and consistency with the [parse]/[plan] guard below.
            click.echo(
                "waxsql validate --tier syntax requires the [syntax] extra.\n"
                "Install with: pip install 'waxsql[syntax]'",
                err=True,
            )
            sys.exit(3)

        # pglast handles multi-statement SQL (gen output includes DDL + query)
        # and SQL comments (the header is a comment line) without issue.
        result = check_syntax(sql_text)
        if not result.ok:
            click.echo(f"SYNTAX error: {result.error}", err=True)
            sys.exit(1)
        if verbose:
            click.echo("OK")
        return

    # ----- PARSE / PLAN tiers ------------------------------------------------
    # Both tiers need psycopg + a live PG. Same connection lifecycle; only
    # the check function differs (PREPARE vs EXPLAIN).
    try:
        import psycopg
    except ImportError:
        click.echo(
            f"waxsql validate --tier {tier} requires psycopg, which ships\n"
            f"with the [parse] (or [plan]) optional extra. Install with:\n\n"
            f"    pip install 'waxsql[{tier}]'",
            err=True,
        )
        sys.exit(3)

    # check_fn typed as Any: check_parse and check_plan have different return
    # types (ParseResult vs PlanResult) that mypy can't reconcile in a single
    # variable without a Union — Any is simpler and correct here since both
    # results have the same .ok / .error / .error_code duck-type interface.
    check_fn: Any
    if tier.lower() == "parse":
        from waxsql.validate.parse import check_parse
        check_fn = check_parse
    else:  # "plan"
        from waxsql.validate.plan import check_plan
        check_fn = check_plan

    # Resolve schema source per the precedence rules in _resolve_schema_source.
    ddl = _resolve_schema_source(
        input_text=sql_text,
        schema_from=schema_from,
        schema_seed=schema_seed,
        schema_complexity=schema_complexity,
        auto_schema=auto_schema,
    )
    if ddl is None:
        raise click.UsageError(
            f"--tier {tier} needs a schema; pass --schema-from PATH, "
            f"--schema-seed N, or pipe in waxsql gen output (which "
            f"includes a parseable header)."
        )

    # Resolve DSN: --dsn flag > $WAXSQL_PG_DSN > default.
    import os as _os
    resolved_dsn = (
        dsn
        or _os.environ.get("WAXSQL_PG_DSN")
        or "dbname=waxsql_test"
    )

    # Open a transaction-mode connection, install the schema DDL, run the
    # check, then roll back so nothing persists in the DB. Same pattern as
    # the install_and_check fixture in tests/conftest.py, packaged for
    # one-shot use. We execute the DDL directly (not via install_schema)
    # because _resolve_schema_source returns a raw DDL string, not a Schema
    # object — and both sources (--schema-from file and regenerated) are
    # equally valid as raw SQL.
    try:
        conn = psycopg.connect(resolved_dsn, autocommit=False)
    except psycopg.Error as e:
        # _redact_dsn masks any `password=...` or URI-form credentials
        # so an exposed log (CI artifact, piped-to-less-trusted-tool)
        # doesn't leak the caller's secret. Other DSN parts (host,
        # dbname, etc.) survive intact for diagnostic value.
        click.echo(
            f"could not connect to PG ({_redact_dsn(resolved_dsn)!r}): {e}",
            err=True,
        )
        sys.exit(1)

    try:
        with conn.cursor() as cur:
            # psycopg (autocommit=False) opens a transaction implicitly on
            # the first statement — no explicit BEGIN needed or wanted.
            # Issuing BEGIN on an already-open psycopg transaction produces
            # a spurious "WARNING: there is already a transaction in progress"
            # from PG and is the psycopg anti-pattern.
            cur.execute(ddl)

        # If the input was a `gen --with-data` stream, regenerate the data
        # deterministically from the header parameters and load it before
        # the query check. ANALYZE so the planner sees the populated-table
        # statistics. Order matters: DDL → COPY blocks → ANALYZE → EXPLAIN.
        # Running ANALYZE before all COPYs would only see partial data for
        # the tables loaded so far; we want one ANALYZE covering all tables.
        header = _parse_header(sql_text)
        if header is not None and header.with_data:
            from waxsql.data import generate_data
            from waxsql.schema import generate_schema as _gen_schema

            # Regenerate the data from the header's parameters. Because
            # generate_data is deterministic in (schema, seed, rows, fanout,
            # null_fraction), this produces byte-identical output to what
            # the original gen command emitted — but we don't keep the
            # gen-output COPY text, we regenerate. That avoids needing to
            # parse arbitrary COPY blocks back out of the input stream
            # (which would conflict with extracting the trailing SELECT).
            schema = _gen_schema(seed=header.seed, complexity=header.complexity)
            try:
                data_sql = generate_data(
                    schema,
                    seed=header.seed,
                    rows=header.rows,
                    fanout=header.fanout,
                    null_fraction=header.null_fraction,
                )
            except ValueError as e:
                # Same cycle-handling branch as in `gen` — kept consistent
                # so failure modes are recognizable across subcommands.
                click.echo(
                    f"waxsql validate: cannot generate data for this schema: {e}\n"
                    f"This typically means the schema has FK cycles, which "
                    f"data generation does not yet support. Try a lower --complexity.",
                    err=True,
                )
                sys.exit(1)
            with conn.cursor() as cur:
                # Each COPY block runs in its own savepoint so a single load
                # failure surfaces cleanly with the offending table, rather
                # than aborting the surrounding transaction and losing the
                # entire error context.
                for block in _split_copy_blocks(data_sql):
                    cur.execute("SAVEPOINT _waxsql_copy")
                    try:
                        _execute_copy_block(cur, block)
                    except psycopg.Error as e:
                        cur.execute("ROLLBACK TO SAVEPOINT _waxsql_copy")
                        cur.execute("RELEASE SAVEPOINT _waxsql_copy")
                        # PG's error doesn't include the table; the block's
                        # first line is the COPY header which does.
                        table_header = block.splitlines()[0]
                        click.echo(
                            f"COPY error loading {table_header!r}: {e}",
                            err=True,
                        )
                        sys.exit(1)
                    cur.execute("RELEASE SAVEPOINT _waxsql_copy")
                # ANALYZE after all COPYs so the planner sees statistics
                # for every table, not just those loaded so far.
                cur.execute("ANALYZE")

        # Extract the first SELECT/WITH statement — PREPARE and EXPLAIN
        # take a single statement; gen output may contain DDL first.
        sql_to_check = _extract_first_select(sql_text)
        # Use a separate variable name to avoid the SyntaxResult type mypy
        # inferred for `result` in the SYNTAX branch above (even though the
        # SYNTAX branch always returns before we get here, mypy tracks the
        # narrowed type across both branches of the if/else chain).
        check_result = check_fn(sql_to_check, conn)
    except psycopg.Error as e:
        # Schema install (`cur.execute(ddl)`) and `ANALYZE` run inside this
        # try with no inner handler; a malformed `--schema-from` DDL or an
        # ANALYZE failure would otherwise escape as a raw traceback. Surface
        # it as a clean CLI error, matching the connect/COPY-load handlers.
        # (check_parse/check_plan don't raise psycopg.Error — they return a
        # result — so this only catches the setup statements.)
        click.echo(f"validation setup failed (schema/ANALYZE): {e}", err=True)
        sys.exit(1)
    finally:
        # If check_fn raised a non-psycopg exception (e.g. BrokenPipeError
        # from a dead connection), rollback/close can raise again and mask
        # the original error. Suppress so the original propagates to the
        # caller — cleanup failures on an already-failing path are noise.
        with contextlib.suppress(Exception):
            conn.rollback()
        with contextlib.suppress(Exception):
            conn.close()

    if not check_result.ok:
        code_part = f"[{check_result.error_code}] " if check_result.error_code else ""
        click.echo(f"{tier.upper()} error: {code_part}{check_result.error}", err=True)
        sys.exit(1)
    if verbose:
        click.echo("OK")


@main.command("data")
@click.option(
    "--seed", type=int, required=True,
    help="Schema/data seed (required).",
)
@click.option(
    "--complexity", type=click.IntRange(0, 10), default=5, show_default=True,
    help="Schema complexity dial 0..10.",
)
@click.option(
    "--rows", type=click.IntRange(min=0), default=100, show_default=True,
    help="Base row count per table (multiplied by fanout**depth). "
         "0 emits empty COPY blocks (header + immediate terminator).",
)
@click.option(
    "--fanout", type=click.IntRange(min=1), default=5, show_default=True,
    help="FK-depth row multiplier.",
)
@click.option(
    "--null-fraction", "null_fraction", type=float, default=0.05, show_default=True,
    help="Per-nullable-column NULL probability.",
)
def data(
    seed: int,
    complexity: int,
    rows: int,
    fanout: int,
    null_fraction: float,
) -> None:
    """Generate row data (COPY blocks) for a deterministic schema.

    Output is COPY blocks only — no DDL, no queries, no header. Pipe
    `waxsql gen --schema-only` (or your own DDL) before this output to
    produce a loadable psql script. The schema is regenerated from
    `--seed` and `--complexity`, so the same pair always yields the same
    tables and the same data.
    """
    # Local imports keep startup time low when click is installed but the
    # data/schema modules haven't been used yet. The pattern mirrors the
    # optional-import guard in `validate` for psycopg/pglast.
    from waxsql.data import generate_data
    from waxsql.schema import generate_schema

    schema = generate_schema(seed=seed, complexity=complexity)
    try:
        text = generate_data(
            schema, seed=seed, rows=rows, fanout=fanout, null_fraction=null_fraction,
        )
    except ValueError as e:
        click.echo(
            f"waxsql data: cannot generate data for this schema: {e}\n"
            f"This typically means the schema has FK cycles, which "
            f"data generation does not yet support. Try a lower --complexity.",
            err=True,
        )
        sys.exit(1)
    # nl=False: the COPY block string from emit_copy_block already ends with
    # "\n" after the "\." terminator, so we let the content dictate line endings.
    click.echo(text, nl=False)
