"""Column-name override registry.

This is a hook for the eventual 'column named email actually has emails'
story. Today the registry is nearly empty and `strategy_for` falls
through to the type strategy in nearly every case. The registry is a
tuple (not a dict) because order matters — first match wins — and
because tuple iteration is deterministic across Python versions.

Role in the system: `strategy_for(col)` is the single per-column
dispatch point used by `rows.generate_row`. Centralizing the lookup
here means future semantic plausibility (emails, names, URLs, etc.)
can grow inside this module without touching the row materializer or
the type-strategy registry.
"""
from __future__ import annotations

import re

from waxsql.gen.data.strategies import Strategy, strategy_for_type
from waxsql.schema import Column


# Tuple of (compiled-pattern, strategy). First match wins. Empty today —
# this is the seam for future semantic plausibility. Adding an entry
# here doesn't require schema changes; the dispatch happens per column
# at row-generation time.
#
# Why a tuple of (pattern, strategy) pairs and not a dict of name → strategy:
# (1) ordering matters when patterns can overlap (e.g. `email_verified_at`
# should match a timestamp pattern, not the email pattern), and tuple
# iteration order is part of the source; (2) regex matching against
# every column name avoids exact-name brittleness.
_NAME_PATTERNS: tuple[tuple[re.Pattern, Strategy], ...] = ()


def strategy_for(column: Column) -> Strategy:
    """Return the strategy to use for `column`. Patterns are consulted
    in order; the first one that matches the column name wins. Falls
    through to `strategy_for_type(column.type)` when nothing matches.

    Today this almost always falls through to the type strategy —
    plausibility is type-driven, the name-override seam is intentionally
    underused. The fallthrough is the common case, not an error path.
    """
    for pat, strat in _NAME_PATTERNS:
        if pat.search(column.name):
            return strat
    return strategy_for_type(column.type)
