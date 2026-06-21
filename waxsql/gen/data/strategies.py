"""Per-PgType value strategies. Each strategy maps (rng, Column) → object;
the emitter is responsible for formatting that object for COPY.

The split between strategies and the emitter is deliberate: strategies
return native Python values (Decimal, datetime, UUID, dict, list) and
have no knowledge of tab encoding or NULL sentinels. That keeps the
strategy registry trivially testable and lets the emitter own all of
PostgreSQL's COPY text-format escape rules in one place.

Determinism invariants (load-bearing across this whole module):
  * Only the injected `rng` is allowed as a source of randomness.
    No `uuid.uuid4()`, no `random.gauss()` from the global module,
    no `datetime.now()`, no environment lookups.
  * `_EPOCH` is a fixed date constant — wall-clock input would mean
    the same seed produces different data tomorrow, breaking the
    "reproduce a bug from a seed years later" guarantee.
  * Dict iteration order is stable in Python 3.7+, so `_TYPE_STRATEGIES`
    is fine; but anything that iterates a `set` for rng decisions must
    `sorted()` first (none currently do).
"""
from __future__ import annotations

import datetime as _dt
import random
import uuid
from dataclasses import dataclass
from decimal import Decimal
from collections.abc import Callable

from waxsql.schema import Column
from waxsql.types import PgType


# Hand-curated short list of simple English words. Used as the default
# text/varchar value source: each cell gets one of these. Not a name
# dictionary, not Lorem Ipsum — just enough variety that SELECT * FROM t
# doesn't look like line noise. Grow as taste dictates.
WORDLIST: tuple[str, ...] = (
    "alpha", "amber", "anchor", "apple", "arrow", "atlas", "azure",
    "badger", "basin", "beacon", "berry", "birch", "blossom", "boulder",
    "breeze", "bridge", "bronze", "buffalo", "cabin", "candle", "canyon",
    "cedar", "cherry", "cinder", "clover", "cobalt", "comet", "copper",
    "coral", "cottage", "crater", "crescent", "crimson", "crystal",
    "dahlia", "dawn", "delta", "diamond", "dolphin", "dove", "dusk",
    "ember", "emerald", "falcon", "feather", "fern", "festival", "fjord",
    "forest", "fossil", "frost", "galaxy", "garnet", "gentian", "geyser",
    "glacier", "glade", "granite", "harbor", "harvest", "haven", "hazel",
    "heron", "hickory", "horizon", "indigo", "iris", "island", "ivory",
    "jasper", "juniper", "kestrel", "lagoon", "lantern", "lavender",
    "library", "linden", "lotus", "magnet", "maple", "marble", "marigold",
    "marsh", "meadow", "mercury", "midnight", "mineral", "mint", "mirage",
    "morning", "mosaic", "mountain", "nectar", "nimbus", "nocturne",
    "oasis", "obsidian", "ocean", "olive", "opal", "orchid", "otter",
    "panther", "parsley", "peach", "pebble", "pelican", "petal", "phlox",
    "pine", "plateau", "poppy", "prairie", "prism", "quartz", "quail",
    "quill", "rainbow", "raven", "redwood", "river", "robin", "rose",
    "ruby", "saffron", "sage", "salmon", "sapphire", "scarlet", "sequoia",
    "shadow", "shoreline", "silver", "slate", "solstice", "sparrow",
    "spruce", "starling", "stone", "summit", "sunset", "swallow", "tangerine",
    "thicket", "thistle", "thunder", "tide", "topaz", "tulip", "turquoise",
    "twilight", "umber", "valley", "velvet", "verdant", "violet", "walnut",
    "waterfall", "willow", "winter", "wisteria", "yarrow", "yew", "zenith",
    "zephyr", "zircon",
)


def pick_word(rng: random.Random) -> str:
    """Pick a deterministic word from `WORDLIST`. Uses `rng.choice` rather
    than `rng.randint`+index so adding/removing wordlist entries shifts
    output predictably rather than scrambling it.
    """
    return rng.choice(WORDLIST)


# ---------------------------------------------------------------------------
# Per-type strategy functions
# ---------------------------------------------------------------------------

# A strategy is a pure function: (rng, column) -> Python value.
# Native types are returned; the emitter formats them for COPY.
Strategy = Callable[[random.Random, Column], object]


def _int4(rng: random.Random, col: Column) -> int:
    # PG int4 is signed 32-bit: -2147483648..2147483647. The -1 floor
    # avoids the minimum-int corner that some downstream string
    # formatters mishandle; the loss of one value is irrelevant.
    return rng.randint(-(2**31) + 1, (2**31) - 1)


def _int8(rng: random.Random, col: Column) -> int:
    # PG int8 is signed 64-bit but we deliberately bound this tighter
    # than the full range. Full-range int8 values aren't interesting
    # for demo data and produce visually ugly output.
    # Trade-off: planner-cost estimates that depend on value distribution
    # see a slightly narrower range; this hasn't bitten anything yet.
    return rng.randint(-(2**62), (2**62))


def _text(rng: random.Random, col: Column) -> str:
    return pick_word(rng)


def _varchar(rng: random.Random, col: Column) -> str:
    word = pick_word(rng)
    # Respect typmod when present; varchar(N) rejects values longer than N.
    # When typmod is absent (`varchar` with no length), cap at 32 — generous
    # enough for any single wordlist entry, modest enough to keep COPY
    # output readable. Note: pick_word ALWAYS fires (consumes one rng tick)
    # even if the cap would truncate to empty — keeps the rng stream
    # independent of typmod, so adding a length to a column doesn't
    # cascade into downstream byte-shifts.
    cap = col.type.typmod[0] if col.type.typmod else 32
    return word[:cap]


def _bool(rng: random.Random, col: Column) -> bool:
    return rng.choice((True, False))


def _uuid(rng: random.Random, col: Column) -> uuid.UUID:
    # rng.getrandbits(128) keeps determinism within our RNG; uuid.uuid4()
    # would reach for os.urandom and break that. The resulting UUID won't
    # have the version-4 bit pattern set — PG doesn't care, the uuid column
    # accepts any 128-bit value, and the determinism contract trumps RFC
    # 4122 cosmetic correctness.
    return uuid.UUID(int=rng.getrandbits(128))


def _float8(rng: random.Random, col: Column) -> float:
    # Bounded so output stays human-readable. Avoid `random.gauss()` —
    # we want a flat distribution for COPY, not a bell curve.
    return rng.uniform(-1_000_000.0, 1_000_000.0)


def _numeric(rng: random.Random, col: Column) -> Decimal:
    # numeric(precision, scale): `precision` total digits, `scale` after
    # the decimal point. Magnitude < 10^(precision-scale); fractional
    # digits = scale. Without typmod, fall back to a sensible default.
    # The default (10, 4) is arbitrary but matches a common business-data
    # shape (six integer digits, four fractional) and keeps output narrow.
    if col.type.typmod and len(col.type.typmod) >= 2:
        precision, scale = col.type.typmod[0], col.type.typmod[1]
    elif col.type.typmod and len(col.type.typmod) == 1:
        # PG allows `numeric(P)` (scale implicit 0). Mirror that.
        precision, scale = col.type.typmod[0], 0
    else:
        precision, scale = 10, 4
    integer_digits = precision - scale
    upper = 10**integer_digits - 1
    # Pick a raw integer in the value space [−upper·10^scale, upper·10^scale],
    # then divide by 10^scale. Doing integer arithmetic first keeps every
    # produced Decimal exactly representable (no float rounding intrusion).
    raw = rng.randint(-upper * 10**scale, upper * 10**scale)
    return Decimal(raw) / (Decimal(10) ** scale)


# Fixed reference epoch — NOT `datetime.now()`. Determinism contract:
# same seed must produce same output years from now, which means no
# wall-clock input anywhere in the generator.
_EPOCH = _dt.date(2025, 1, 1)
_WINDOW_DAYS = 5 * 365  # ±5 years


def _date(rng: random.Random, col: Column) -> _dt.date:
    days = rng.randint(-_WINDOW_DAYS, _WINDOW_DAYS)
    return _EPOCH + _dt.timedelta(days=days)


def _timestamptz(rng: random.Random, col: Column) -> _dt.datetime:
    # Three rng calls in fixed order: days, seconds, microseconds.
    # Changing the order would shift every downstream value — these
    # are part of the determinism contract.
    days = rng.randint(-_WINDOW_DAYS, _WINDOW_DAYS)
    seconds = rng.randint(0, 86_399)
    micros = rng.randint(0, 999_999)
    # tz: stick with UTC. timestamptz stores UTC internally regardless
    # of the input tz; emitting UTC keeps COPY output canonical and
    # avoids planner statistics being affected by client TZ settings.
    return _dt.datetime(
        _EPOCH.year, _EPOCH.month, _EPOCH.day, tzinfo=_dt.timezone.utc,
    ) + _dt.timedelta(days=days, seconds=seconds, microseconds=micros)


def _interval(rng: random.Random, col: Column) -> _dt.timedelta:
    # Bounded interval: at most a few months. PG intervals can encode
    # year/month parts that timedelta cannot; the emitter formats as
    # ISO-8601-like and the server accepts it cleanly.
    days = rng.randint(0, 120)
    seconds = rng.randint(0, 86_399)
    return _dt.timedelta(days=days, seconds=seconds)


def _jsonb(rng: random.Random, col: Column) -> dict:
    """Shallow random object: 1-4 string keys, scalar values.

    Deliberately not nested. The spec allows for richer JSON later, but
    for parse/plan validation and demo readability, shallow is enough.

    Key collisions across the loop are tolerated: when two iterations
    pick the same word, the later iteration overwrites the earlier.
    That's why output dicts may have fewer than `n_keys` entries; this
    is intentional and keeps the rng-call count fixed at `n_keys` keys
    regardless of collisions (every iteration consumes the same ticks).
    """
    n_keys = rng.randint(1, 4)
    out: dict[str, object] = {}
    for _ in range(n_keys):
        key = pick_word(rng)
        # Cap the dict at n_keys; collisions just overwrite, which is fine.
        # Each branch below consumes EXACTLY ONE rng call (after the kind
        # roll), so the total rng consumption per _jsonb is deterministic:
        # 1 (n_keys) + n_keys × (1 key + 1 kind + 0-or-1 value).
        kind = rng.randint(0, 4)
        if kind == 0:
            out[key] = pick_word(rng)
        elif kind == 1:
            out[key] = rng.randint(-1000, 1000)
        elif kind == 2:
            out[key] = round(rng.uniform(-1000.0, 1000.0), 3)
        elif kind == 3:
            out[key] = rng.choice((True, False))
        else:
            # kind == 4: literal JSON null. No rng call. The asymmetry
            # (4 of 5 branches consume an rng tick, 1 doesn't) is
            # intentional — null is a real JSON value, not an error path,
            # and balancing rng consumption would force a dummy call.
            out[key] = None
    return out


@dataclass(frozen=True)
class _ColumnAdapter:
    """Lightweight Column stand-in used when array strategies recurse on
    the element type. Avoids depending on the full Column constructor
    surface in case it grows constraints we don't care about here.
    """
    name: str
    type: PgType
    nullable: bool


def _array(element_strategy: Strategy, element_type: PgType) -> Strategy:
    """Build a strategy that returns a 0-5 element list. The factory
    closes over the element strategy so call-time work is minimal —
    we pay the `strategy_for_type` lookup once at factory time, not
    once per row.

    The closure pattern matters because `strategy_for_type` is called
    every time a row generates an array column; without memoization via
    the factory, recursive array types would re-resolve their element
    strategies on every row.
    """
    def gen(rng: random.Random, col: Column) -> list:
        n = rng.randint(0, 5)
        # Element nullability is intentionally suppressed; NULL injection
        # is an outer-row concern, not an element-level one. A PG array
        # can contain NULL elements, but we don't generate them today —
        # would require differently-quoted output ({NULL} not {""}) and
        # the emit layer doesn't currently model that distinction.
        elem_col = _ColumnAdapter(name=col.name, type=element_type, nullable=False)
        return [element_strategy(rng, elem_col) for _ in range(n)]  # type: ignore[arg-type]
    return gen


# Type-name → strategy lookup. Keyed on `PgType.name` (matches pg_type.typname).
# Adding a new scalar PgType requires adding an entry here AND making sure
# the emit module's `encode_value` knows how to format the strategy's
# return type. The two registries together define what the data generator
# can produce.
_TYPE_STRATEGIES: dict[str, Strategy] = {
    "int4": _int4,
    "int8": _int8,
    "text": _text,
    "varchar": _varchar,
    "bool": _bool,
    "uuid": _uuid,
    "float8": _float8,
    "numeric": _numeric,
    "date": _date,
    "timestamptz": _timestamptz,
    "interval": _interval,
    "jsonb": _jsonb,
}


def strategy_for_type(t: PgType) -> Strategy:
    """Return the strategy for type `t`, or raise KeyError if no scalar strategy
    is registered for the element type.

    Array types are handled directly: we recurse on the element type and
    wrap the result in `_array`. The element strategy is looked up once at
    factory time (not per row), keeping data generation cheap. Recursion
    bottoms out at scalar types, which are in `_TYPE_STRATEGIES`.
    """
    if t.is_array():
        assert t.element is not None  # guaranteed by is_array()
        return _array(strategy_for_type(t.element), t.element)
    return _TYPE_STRATEGIES[t.name]
