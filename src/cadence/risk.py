"""Section 8: The safety gate every order must pass.

A bot that places orders on its own is dangerous if nothing checks it. So
before any order reaches the exchange, it passes through a gate that can say
no. The gate enforces a few simple rules:

  - A position limit: never let a book build up a bigger bet than allowed.
  - A kill switch: if we switch a strategy off, none of its orders get
    through.
  - A circuit breaker: if we halt a book, nothing trades on it.
  - Attribution: every order must say which strategy created it, so we can
    always trace later who did what (regulators now require exactly this).

This module ships the gate in two interchangeable versions, and the contrast
between them is the real lesson of this section:

  - `InProcessGovernor`: plain Python, holds everything in memory. Perfect
    for learning and for running the examples with nothing else installed.
  - `MorphologGovernor`: hands the rules to Morpholog, a separate engine that
    stores the record properly and refuses, at the source, to ever record an
    order that breaks a rule.

On a small single-program demo like this one, the plain Python version is all
you need, and Morpholog can look like overkill. Its value shows up when the
demo grows up: when the program restarts and must not forget its position;
when several bots trade the same book at once and must share one honest view
of it; when an auditor needs a record that provably has not been tampered
with and can be replayed to any moment in the past. Those are the things that
turn the 50 lines below into thousands, and that is the job Morpholog is for.
The guide builds the plain version first, then deliberately breaks it to show
where Morpholog earns its place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Order:
    """One order the bot wants to place.

    Notice you cannot even create an order without naming the `strategy` that
    produced it and the `book` it belongs to. That is attribution built in:
    an unlabelled order is impossible. `signed_qty` carries direction, a
    positive number buys, a negative number sells.
    """

    order_id: str
    strategy: str
    book: str
    signed_qty: float
    price: float


@dataclass(frozen=True)
class Decision:
    """The gate's answer: was the order allowed, and if not, why not."""

    admitted: bool
    reason: str = ""


class Governor(Protocol):
    """The shape of a safety gate. The bot proposes orders; the gate decides.

    Both versions below follow this same shape, so the bot does not know or
    care which one it is talking to.
    """

    def open_book(self, book: str, limit: float) -> None: ...
    def enable_strategy(self, strategy: str) -> None: ...
    def engage_kill_switch(self, strategy: str) -> None: ...
    def halt_book(self, book: str) -> None: ...
    def admit(self, order: Order) -> Decision: ...


@dataclass
class InProcessGovernor:
    """The plain-Python safety gate. Simple, and exactly the code you no
    longer have to write (and trust) once Morpholog enforces the same rules.
    """

    _enabled: set[str] = field(default_factory=set)
    _open: set[str] = field(default_factory=set)
    _limit: dict[str, float] = field(default_factory=dict)
    _net: dict[str, float] = field(default_factory=dict)
    _log: list[Order] = field(default_factory=list)
    _ids: set[str] = field(default_factory=set)

    def open_book(self, book: str, limit: float) -> None:
        """Open a book for trading, with a maximum position size."""
        if limit < 0:
            raise ValueError("the limit cannot be negative")
        self._open.add(book)
        self._limit[book] = limit
        self._net.setdefault(book, 0.0)

    def enable_strategy(self, strategy: str) -> None:
        """Allow a strategy's orders through the gate."""
        self._enabled.add(strategy)

    def engage_kill_switch(self, strategy: str) -> None:
        """Switch a strategy off. Its orders will now be refused."""
        self._enabled.discard(strategy)

    def halt_book(self, book: str) -> None:
        """Stop all trading on a book."""
        self._open.discard(book)

    def admit(self, order: Order) -> Decision:
        """Decide whether to let an order through, and remember it if so."""
        if order.order_id in self._ids:
            return Decision(False, "we have already seen this order id")
        if order.strategy not in self._enabled:
            return Decision(False, f"strategy '{order.strategy}' is switched off")
        if order.book not in self._open:
            return Decision(False, f"book '{order.book}' is not open")
        limit = self._limit[order.book]
        new_position = self._net[order.book] + order.signed_qty
        if abs(new_position) > limit:
            return Decision(
                False,
                f"would push book '{order.book}' over its limit "
                f"({abs(new_position):.1f} beyond the allowed {limit:.1f})",
            )
        # Allowed: record the order and update the running position.
        self._net[order.book] = new_position
        self._log.append(order)
        self._ids.add(order.order_id)
        return Decision(True)

    @property
    def order_log(self) -> list[Order]:
        """The list of orders that were allowed through, in order.

        In the plain-Python version this is just a list, which is exactly the
        weakness: anyone could change it or lose it on a restart. The
        Morpholog version turns it into a proper, tamper-evident record.
        """
        return list(self._log)


class MorphologUnavailable(RuntimeError):
    """Raised when the Morpholog-backed gate is asked for but its engine is
    not set up. The plain-Python gate above needs none of that."""


class MorphologGovernor:
    """The same safety gate, but enforced by Morpholog instead of by Python.

    The difference that matters: the position limit is written down as a rule
    inside Morpholog, and Morpholog simply will not record an order that
    breaks it, no matter how the order arrives. The rules live in the file
    cadence.morph next to this code, where a risk officer or auditor can read
    them without reading any Python.

    Setting it up needs the `morpholog` program, a throwaway database, and a
    one-off step to generate the typed client:

        morpholog generate python-client cadence.morph --out src/cadence/_morph_client

    The methods are left for the guide's Section 8 to fill in, so that the
    examples here keep running with nothing extra installed. The point the
    guide makes is that almost none of the rule-checking ends up in Python
    once Morpholog holds the rules: the plain version above is what you get
    to delete.
    """

    def __init__(self, database_url: str | None = None) -> None:
        self._database_url = database_url or os.environ.get("DATABASE_URL")
        try:
            # This client only exists after the "generate" step above.
            from cadence._morph_client import Morpholog  # type: ignore
        except ImportError as exc:  # pragma: no cover - needs external setup
            raise MorphologUnavailable(
                "Morpholog client not generated yet. Run: "
                "morpholog generate python-client cadence.morph "
                "--out src/cadence/_morph_client"
            ) from exc
        if not self._database_url:
            raise MorphologUnavailable(
                "set DATABASE_URL to a throwaway database first"
            )
        self._client = Morpholog("cadence.morph", self._database_url)

    # Each method below matches one rule-changing action in cadence.morph.
    def open_book(self, book: str, limit: float) -> None:  # pragma: no cover
        raise NotImplementedError("filled in during guide Section 8")

    def enable_strategy(self, strategy: str) -> None:  # pragma: no cover
        raise NotImplementedError("filled in during guide Section 8")

    def engage_kill_switch(self, strategy: str) -> None:  # pragma: no cover
        raise NotImplementedError("filled in during guide Section 8")

    def halt_book(self, book: str) -> None:  # pragma: no cover
        raise NotImplementedError("filled in during guide Section 8")

    def admit(self, order: Order) -> Decision:  # pragma: no cover
        raise NotImplementedError("filled in during guide Section 8")


def make_governor(kind: str | None = None) -> Governor:
    """Pick which safety gate to use.

    By default you get the plain-Python gate, which needs nothing extra. Set
    the environment variable CADENCE_GOVERNOR=morpholog to use the
    Morpholog-backed one instead.
    """
    kind = kind or os.environ.get("CADENCE_GOVERNOR", "inprocess")
    if kind == "morpholog":
        return MorphologGovernor()
    return InProcessGovernor()
