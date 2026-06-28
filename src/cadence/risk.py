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
import subprocess
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
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

    def __post_init__(self) -> None:
        # A zero-size or non-positive-price order is a bug, not a trade.
        if self.signed_qty == 0:
            raise ValueError("order quantity cannot be zero")
        if self.price <= 0:
            raise ValueError("order price must be positive")


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

    The rules live in cadence.morph, next to this code, where a risk officer
    or auditor can read them without reading any Python. Morpholog will not
    record an order that breaks them, no matter how the order arrives. Setting
    it up needs the `morpholog` program, a throwaway database, and a one-off
    step to generate the typed client (see the README):

        morpholog generate python-client cadence.morph --out src/cadence

    Each method below maps onto one transformation in cadence.morph. An order
    that would breach the position limit, name a switched-off strategy, or land
    on a halted book comes back as a refusal (a `Rejected` outcome), so the
    runtime, not this Python, is what enforces the rules.

    The PostgreSQL database is deliberate: it gives the record a real, durable
    home that survives a restart. Morpholog then builds its tamper-evidence on
    top (an append-only, hash-chained audit it can replay and verify); the
    database supplies durability, Morpholog supplies the tamper-evidence. The
    cost is running as a separate program; a resident `morpholog serve`
    process avoids paying the start-up cost per call.
    """

    ACTOR = "cadence-bot"

    def __init__(
        self, database_url: str | None = None, morph_file: str = "cadence.morph"
    ) -> None:
        self._database_url = database_url or os.environ.get("DATABASE_URL")
        if not self._database_url:
            raise MorphologUnavailable(
                "set DATABASE_URL to a PostgreSQL database first"
            )
        try:
            # This package only exists after the "generate" step above.
            from cadence.morpholog_client import Morpholog, models
        except ImportError as exc:  # pragma: no cover - needs external setup
            raise MorphologUnavailable(
                "Morpholog client not generated yet. Run: "
                "morpholog generate python-client cadence.morph --out src/cadence"
            ) from exc
        self._models = models
        self._client = Morpholog(morph_file, self._database_url)
        # Provision the schema for this programme (idempotent).
        self._client.init(skip_if_exists=True)

    def _submit_as(self, request: object, actor: str) -> Decision:
        """Propose one transformation as `actor`; map the outcome to a Decision.
        The actor matters wherever a rule names it (the segregation-of-duties
        check below), which is why it is a parameter rather than always ACTOR."""
        from cadence.morpholog_client import envelopes

        outcome = self._client.submit(request, actor)
        if isinstance(outcome, envelopes.Committed):
            return Decision(True)
        return Decision(False, outcome.reason)

    def _submit(self, request: object) -> Decision:
        return self._submit_as(request, self.ACTOR)

    def open_book(self, book: str, limit: float) -> None:
        decision = self._submit(
            self._models.OpenBookRequest(book=book, limit=Decimal(str(limit)))
        )
        if not decision.admitted:
            raise ValueError(f"open_book refused: {decision.reason}")

    def enable_strategy(self, strategy: str) -> None:
        self._submit(self._models.EnableStrategyRequest(strategy=strategy))

    def engage_kill_switch(self, strategy: str) -> None:
        # Best effort: a kill switch on an already-off strategy is a no-op.
        self._submit(self._models.EngageKillSwitchRequest(strategy=strategy))

    def halt_book(self, book: str) -> None:
        # Best effort: halting an already-halted book is a no-op.
        self._submit(self._models.HaltBookRequest(book=book))

    def admit(self, order: Order) -> Decision:
        return self._submit(
            self._models.AdmitOrderRequest(
                order_id=order.order_id,
                strategy=order.strategy,
                book=order.book,
                signed_qty=Decimal(str(order.signed_qty)),
                price=Decimal(str(order.price)),
            )
        )

    def record_fill(self, fill_id: str, order_id: str, qty: float) -> Decision:
        """Record a fill the exchange reported against an order. The runtime
        refuses a fill against an unknown order, and refuses any fill that would
        take the order past the size it was admitted for (the overfill rule).

        This is what the in-process FillReconciler does in Python, but here it
        is part of the same governed record as the order, so the order and how
        much of it has filled cannot drift apart.
        """
        return self._submit(
            self._models.RecordFillRequest(
                fill_id=fill_id,
                order_id=order_id,
                qty=Decimal(str(qty)),
            )
        )

    # --- Segregation of duties (who may change the rules) -----------------
    # These are governance the in-process gate has no way to express: they
    # turn on *who* is acting, not just what they are doing.

    def assign_trader(self, book: str, trader: str) -> None:
        """Record who trades a book (setup)."""
        self._submit(self._models.AssignTraderRequest(book=book, trader=trader))

    def grant_limit_authority(self, principal: str) -> None:
        """Grant the risk-desk power to change position limits."""
        self._submit(self._models.GrantLimitAuthorityRequest(principal=principal))

    def change_position_limit(
        self, book: str, current_limit: float, new_limit: float, actor: str
    ) -> Decision:
        """Change a book's position limit, acting as `actor`. Refused unless
        the actor holds limit authority AND is not the book's own trader: a
        trader can never raise their own limit (four-eyes), and the runtime,
        not the trading code, is what enforces the separation."""
        return self._submit_as(
            self._models.ChangePositionLimitRequest(
                book=book,
                current_limit=Decimal(str(current_limit)),
                new_limit=Decimal(str(new_limit)),
            ),
            actor,
        )

    # --- Audit: proving the governed record ------------------------------
    # The order-and-fill record can be exported as a portable, offline-
    # verifiable evidence pack: an auditor or regulator checks it without the
    # database and without trusting the operator. This is operator integrity
    # evidence over the trail, not the regulator's submission format itself,
    # which a participant generates from these governed records.

    def checkpoint(self) -> None:
        """Seal a checkpoint (a tamper-evident tree head) over the current
        record, so it can be exported and verified as of this point."""
        self._client.checkpoint()

    def export_evidence(self, path: str) -> None:
        """Seal a checkpoint and export the whole governed record to `path`
        as a self-verifying evidence pack (canonical JSON)."""
        self.checkpoint()
        assert self._database_url is not None  # guaranteed by __init__
        # Export writes the canonical pack JSON to stdout; persist it verbatim
        # so the offline verifier sees exactly what was exported.
        proc = subprocess.run(
            [self._client.binary, "evidence", "export",
             "--database-url", self._database_url],
            capture_output=True, text=True, check=True,
        )
        Path(path).write_text(proc.stdout)

    def verify_evidence(self, path: str) -> bool:
        """Offline-check an exported evidence pack (no database). True if the
        record's tamper-evidence holds; False if any recorded row was altered."""
        from cadence.morpholog_client import envelopes

        verdict = self._client.evidence_verify(path)
        return isinstance(verdict, envelopes.TreeIntact)


def make_governor(kind: str | None = None) -> Governor:
    """Pick which safety gate to use.

    By default you get the plain-Python gate, which needs nothing extra. Set
    the environment variable CADENCE_GOVERNOR=morpholog to use the
    Morpholog-backed one instead.
    """
    kind = kind or os.environ.get("CADENCE_GOVERNOR", "inprocess")
    if kind == "inprocess":
        return InProcessGovernor()
    if kind == "morpholog":
        return MorphologGovernor()
    # A typo should not silently fall back to the weaker gate.
    raise ValueError(
        f"unknown governor kind {kind!r} (use 'inprocess' or 'morpholog')"
    )
