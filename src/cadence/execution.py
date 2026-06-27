"""Section 5: Execution algorithms.

The order-type concepts belong to the Market Microstructure guide; here we
*implement* schedulers that work a decided volume into the market without
giving away edge, and we confront the operational reality the backtest
never shows you: partial fills. A scheduler that assumes every clip fills
in full will, on the first partial, believe it is done while still exposed.

Three schedulers, each motivated by the limit of the last:
  - TWAP: even across time. Blind to urgency and book state.
  - urgency-adjusted: trade harder toward gate closure; the cost of waiting
    in power is not just worse price but the probability of being stuck at
    gate closure times the expected imbalance price.
  - liquidity-aware: adapt clip size to book depth (Section 6 / repo TODO).

`FillReconciler` carries the Section 2 insight: position is a belief,
reconciled against fills before the next decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def twap_schedule(total_qty: float, n_slices: int) -> np.ndarray:
    """Split a quantity evenly across `n_slices` intervals."""
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    return np.full(n_slices, total_qty / n_slices)


def urgency_schedule(
    total_qty: float, n_slices: int, alpha: float = 1.0
) -> np.ndarray:
    """Urgency-weighted schedule. alpha < 1 front-loads (trade early, into
    the best liquidity); alpha = 1 is TWAP; alpha > 1 back-loads (wait for
    information, riskier into gate closure). Weights sum to `total_qty`."""
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    # Time remaining fraction falls from 1 to ~0; weight by its power.
    remaining = np.linspace(1.0, 1.0 / n_slices, n_slices)
    weights = remaining**alpha
    weights = weights / weights.sum()
    return weights * total_qty


def cost_of_waiting(
    fill_prob_at_close: float, expected_imbalance_price: float
) -> float:
    """The energy-specific term that makes power execution different: the
    expected cost (GBP/MWh) of failing to execute before gate closure.
    `fill_prob_at_close` is the probability the residual is still unfilled
    at close. The guide derives why this belongs in the urgency function."""
    if not 0.0 <= fill_prob_at_close <= 1.0:
        raise ValueError("fill probability must be in [0, 1]")
    return fill_prob_at_close * expected_imbalance_price


@dataclass
class FillReconciler:
    """Tracks target versus filled, so the bot's believed position stays
    honest under partial fills. The classic live bug is here if you remove
    it: assuming full fills and losing track of residual exposure."""

    target: float
    filled: float = 0.0
    fills: list[float] = field(default_factory=list)

    def record_fill(self, qty: float) -> None:
        if qty < 0:
            raise ValueError("fill qty must be non-negative")
        self.filled += qty
        self.fills.append(qty)

    @property
    def residual(self) -> float:
        """What is still working. Never assume this is zero."""
        return self.target - self.filled

    @property
    def is_complete(self) -> bool:
        return abs(self.residual) < 1e-9
