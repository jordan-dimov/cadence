"""Integration test for the Morpholog-backed safety gate.

This runs the real `MorphologGovernor`, which drives the Morpholog runtime
over cadence.morph against a PostgreSQL database. It is skipped unless that
setup is present, so the everyday `pytest` run (and a fresh checkout) needs
nothing extra. Continuous integration provides the database and the
`morpholog` binary and runs it for real (see .github/workflows/morpholog.yml).

What it proves is the whole point of Section 8 of the guide: the position
limit and the kill switch are enforced by the runtime, not by the Python, so
a breaching order is refused at the source.
"""

import os
import shutil
import uuid

import pytest

_have_db = bool(os.environ.get("DATABASE_URL"))
_have_binary = bool(os.environ.get("MORPHOLOG_BIN") or shutil.which("morpholog"))

pytestmark = pytest.mark.skipif(
    not (_have_db and _have_binary),
    reason="needs DATABASE_URL and the morpholog binary (MORPHOLOG_BIN or on PATH)",
)

# Skips cleanly if the client has not been generated into src/cadence yet.
pytest.importorskip("cadence.morpholog_client")

from cadence.risk import MorphologGovernor, Order  # noqa: E402


def _fresh_names():
    """Unique book and strategy ids, so each run is independent of any state
    left in the database by an earlier run."""
    suffix = uuid.uuid4().hex[:8]
    return f"book-{suffix}", f"strat-{suffix}"


def test_runtime_enforces_position_limit_and_kill_switch():
    book, strat = _fresh_names()
    gov = MorphologGovernor()
    gov.open_book(book, limit=100.0)
    gov.enable_strategy(strat)

    # Within the limit: the runtime admits it.
    assert gov.admit(Order(f"o1-{book}", strat, book, 50, 80)).admitted

    # This would push the net position to 110, past the limit of 100. The
    # runtime invariant refuses it; no Python check is involved.
    breaching = gov.admit(Order(f"o2-{book}", strat, book, 60, 80))
    assert not breaching.admitted
    assert breaching.reason  # a reason is reported

    # After the kill switch, the strategy's orders are refused at the source.
    gov.engage_kill_switch(strat)
    assert not gov.admit(Order(f"o3-{book}", strat, book, 10, 80)).admitted


def test_attribution_is_recorded_and_within_limit_orders_accumulate():
    book, strat = _fresh_names()
    gov = MorphologGovernor()
    gov.open_book(book, limit=100.0)
    gov.enable_strategy(strat)

    # Two orders that together stay within the limit both go through.
    assert gov.admit(Order(f"a-{book}", strat, book, 40, 80)).admitted
    assert gov.admit(Order(f"b-{book}", strat, book, 40, 80)).admitted
    # A third would take the total to 120, over the limit: refused.
    assert not gov.admit(Order(f"c-{book}", strat, book, 40, 80)).admitted
