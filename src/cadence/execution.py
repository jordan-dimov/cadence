"""Section 5: Actually placing the orders, without moving the price.

Deciding to sell 15 MWh is one thing. Getting it done well is another. If you
dump all 15 onto the exchange at once, you eat through the best prices and the
rest fills at worse and worse levels: you have pushed the price against
yourself. So instead you break the order into smaller pieces and feed them in
over time. Deciding the timing and size of those pieces is what this module
does.

Three ways to schedule the pieces, each fixing a weakness of the one before:

  - Spread them evenly over time (simple, but ignores how urgent it is).
  - Trade harder as the deadline approaches (smarter about urgency).
  - Size each piece to how much is on offer right now, so as not to push the
    price (smartest; `liquidity_aware_clip`).

The deadline matters a lot in power. Every product has a "gate closure": a
cut-off after which you can no longer trade it. Miss it with power still
unsold and you are forced into the imbalance market, which is usually
expensive. So waiting has a real, measurable cost, which we work out below.

One more hard-won lesson lives here too: an order you send is not always
filled in full. It can be partly filled, or rejected. If your code assumes
every order completes, it will think it is finished while it is still
exposed. `FillReconciler` keeps the bot honest about what really happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import OrderBook


def twap_schedule(total_qty: float, n_slices: int) -> np.ndarray:
    """Split the order into equal pieces, one per time slot.

    The name TWAP stands for "time-weighted average price", which is just a
    fancy way of saying "trade the same amount each interval". Simple and
    predictable, but it pays no attention to deadlines or market conditions.
    """
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    return np.full(n_slices, total_qty / n_slices)


def urgency_schedule(
    total_qty: float, n_slices: int, alpha: float = 1.0
) -> np.ndarray:
    """Split the order into pieces that get bigger or smaller over time,
    depending on how urgent things are.

    The `alpha` dial controls the shape:
      - below 1: trade more early, while there is plenty of time and
        liquidity, and ease off later.
      - exactly 1: equal pieces (same as the simple schedule above).
      - above 1: hold back early and rush at the end (riskier, because you
        are betting you can still get filled near the deadline).

    Whatever the shape, the pieces always add up to the full amount.
    """
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    # How much time is left, as a fraction, falling from 1 toward 0.
    time_left = np.linspace(1.0, 1.0 / n_slices, n_slices)
    weights = time_left**alpha
    weights = weights / weights.sum()
    return weights * total_qty


def liquidity_aware_clip(
    remaining_qty: float, book: OrderBook, max_fraction: float = 0.5
) -> float:
    """Decide how big the next piece should be from how much is on offer now.

    When the book is deep we can trade a bigger piece; when it is thin we
    trade a smaller one, so we do not push the price against ourselves. We
    never take more than `max_fraction` of what is resting at the best price
    level, nor more than what is left to trade. (This sizes a buy against the
    best ask; selling works the same way against the best bid.)
    """
    if remaining_qty < 0:
        raise ValueError("remaining_qty cannot be negative")
    depth_at_best = float(book.ask_sizes[0])
    return min(remaining_qty, max_fraction * depth_at_best)


def cost_of_waiting(
    fill_prob_at_close: float, expected_imbalance_price: float
) -> float:
    """How much it costs, on average, to leave trading too late.

    If there is some chance you fail to finish before the gate closes, that
    leftover gets settled in the expensive imbalance market. The expected
    cost is simply: the chance you are caught out, times how much being
    caught out costs. This is the number that justifies trading more urgently
    in power than you would in, say, shares.
    """
    if not 0.0 <= fill_prob_at_close <= 1.0:
        raise ValueError("a probability must be between 0 and 1")
    return fill_prob_at_close * expected_imbalance_price


@dataclass
class FillReconciler:
    """Keeps track of how much of an order has actually been filled.

    You ask to trade `target`. The exchange fills it bit by bit. After each
    fill you call `record_fill`, and `residual` tells you how much is still
    outstanding. Never assume an order filled in full: that assumption is one
    of the most common ways a live bot quietly ends up exposed.
    """

    target: float
    filled: float = 0.0
    fills: list[float] = field(default_factory=list)

    def record_fill(self, qty: float) -> None:
        """Record that `qty` of the order just got filled."""
        if qty < 0:
            raise ValueError("a fill cannot be negative")
        self.filled += qty
        self.fills.append(qty)

    @property
    def residual(self) -> float:
        """How much of the order is still waiting to be filled."""
        return self.target - self.filled

    @property
    def is_complete(self) -> bool:
        """True once the whole order has been filled."""
        return abs(self.residual) < 1e-9
