"""CLI integration tests for the `waxsql data` subcommand.

Tests cover subcommand registration, COPY-block output shape, determinism,
absence of the gen-style header, and the requirement for --seed.
"""
from __future__ import annotations

from click.testing import CliRunner

from waxsql.cli import main as cli


def test_data_subcommand_exists() -> None:
    runner = CliRunner()
    r = runner.invoke(cli, ["data", "--help"])
    assert r.exit_code == 0
    assert "data" in r.output.lower()


def test_data_subcommand_emits_copy_blocks() -> None:
    runner = CliRunner()
    r = runner.invoke(cli, ["data", "--seed", "1", "--complexity", "3", "--rows", "5"])
    assert r.exit_code == 0, r.output
    assert "COPY " in r.output
    assert r.output.rstrip().endswith(r"\.")


def test_data_subcommand_is_deterministic() -> None:
    runner = CliRunner()
    r1 = runner.invoke(cli, ["data", "--seed", "7", "--complexity", "3", "--rows", "5"])
    r2 = runner.invoke(cli, ["data", "--seed", "7", "--complexity", "3", "--rows", "5"])
    assert r1.exit_code == 0 and r2.exit_code == 0
    assert r1.output == r2.output


def test_data_subcommand_no_header() -> None:
    """`data` output is COPY blocks only — no '-- waxsql' header line.

    The header belongs to combined `gen --with-data` streams.  Standalone
    data output is meant to be concatenated onto user-managed DDL.
    """
    runner = CliRunner()
    r = runner.invoke(cli, ["data", "--seed", "1", "--complexity", "3", "--rows", "3"])
    assert r.exit_code == 0
    assert not r.output.lstrip().startswith("-- waxsql")


def test_data_subcommand_requires_seed() -> None:
    runner = CliRunner()
    r = runner.invoke(cli, ["data", "--complexity", "3"])
    assert r.exit_code != 0
    assert "seed" in r.output.lower() or "missing" in r.output.lower()


def test_data_subcommand_zero_rows_emits_empty_copy_blocks() -> None:
    """`--rows 0` is a documented degenerate case: COPY headers with
    immediate terminators, no data rows. Useful for stress-testing the
    pipeline (`waxsql data ... | psql ...`) without data volume.
    """
    runner = CliRunner()
    r = runner.invoke(cli, ["data", "--seed", "1", "--complexity", "3", "--rows", "0"])
    assert r.exit_code == 0, r.output
    assert "COPY " in r.output
    assert r.output.rstrip().endswith(r"\.")
    # No data rows: every non-COPY, non-terminator line is whitespace.
    for line in r.output.splitlines():
        if line.startswith("COPY ") or line.strip() in (r"\.", ""):
            continue
        assert False, f"unexpected data line at rows=0: {line!r}"


def test_gen_with_data_emits_header_ddl_data_queries() -> None:
    """Combined stream order must be: header → CREATE TABLE → COPY → SELECT.

    The header and DDL come from the existing gen path; the COPY blocks are
    injected between DDL and queries only when --with-data is set.
    """
    runner = CliRunner()
    r = runner.invoke(
        cli, ["gen", "--seed", "1", "--complexity", "3",
              "--with-data", "--rows", "5"],
    )
    assert r.exit_code == 0, r.output
    out = r.output
    # Positions of the four landmarks — in increasing order proves correct
    # stream ordering without caring about the exact line numbers.
    h = out.index("-- waxsql")
    d = out.index("CREATE TABLE")
    c = out.index("COPY ")
    q = out.index("SELECT")
    assert h < d < c < q


def test_gen_with_data_header_includes_data_keys() -> None:
    """The header line must carry with-data, rows, and fanout keys when
    --with-data is set, so the output can be self-describing and round-trip
    through the validate command's header parser.
    """
    runner = CliRunner()
    r = runner.invoke(
        cli, ["gen", "--seed", "1", "--complexity", "3",
              "--with-data", "--rows", "7", "--fanout", "3"],
    )
    assert r.exit_code == 0
    assert "with-data=true" in r.output
    assert "rows=7" in r.output
    assert "fanout=3" in r.output


def test_gen_without_with_data_is_unchanged() -> None:
    """Regression: existing `gen` output without --with-data is unchanged.

    No data-flag keys in the header, and no COPY blocks in the output.
    This guards against accidentally polluting the historical output format.
    """
    runner = CliRunner()
    r = runner.invoke(cli, ["gen", "--seed", "1", "--complexity", "3"])
    assert r.exit_code == 0
    assert "with-data" not in r.output  # no data-flag keys in header
    assert "COPY " not in r.output       # no COPY blocks


def test_data_subcommand_clean_error_on_cyclic_schema() -> None:
    """At sufficiently high complexity the schema generator produces FK
    cycles. The data CLI must surface this as a clean error, not a traceback.

    We sweep complexity 9 across 20 seeds: at least one must hit a cyclic
    schema (empirically ~19/20 at complexity=9). For every non-zero exit we
    require the output to be a clean usage message — no Python traceback.
    """
    runner = CliRunner()
    saw_error = False
    for seed in range(20):
        r = runner.invoke(
            cli, ["data", "--seed", str(seed), "--complexity", "9", "--rows", "2"]
        )
        if r.exit_code != 0:
            saw_error = True
            combined = (r.output or "") + (r.stderr or "")
            # Clean error: no Python traceback anywhere in the output.
            assert "Traceback" not in combined, (
                f"seed={seed}: got a raw traceback instead of a clean error:\n{combined}"
            )
            assert any(
                kw in combined.lower()
                for kw in ("fk", "cycle", "complexity", "cannot generate")
            ), f"seed={seed}: error message lacks expected keywords:\n{combined}"
    assert saw_error, (
        "expected at least one seed in [0,20) to hit a cyclic schema at complexity=9"
    )
