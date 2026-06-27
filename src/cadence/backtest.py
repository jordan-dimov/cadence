"""Section 7: Would this strategy actually have made money?

Before risking real money, you replay a strategy over past data and see how
it would have done. That is a backtest. It sounds simple, and it is full of
traps that make a losing strategy look like a winner.

The two biggest traps:

  - Peeking at the future. If your strategy ever uses information it could not
    have known at the time, even by accident, the backtest is a fantasy. We
    guard against this by always acting on *yesterday's* decision, never
    today's.
  - Pretending trading is free. Every trade costs something (fees, and the
    small loss from crossing the gap between buy and sell prices). Ignore
    that and any strategy looks better than it really is. We charge a cost
    for every trade, and you can watch the apparent profit shrink once you
    do.

The full guide adds more (modelling how your own trading moves the price, and
testing on rolling chunks of time). This module is the honest core: replay,
charge costs, and report not just profit but how bumpy the ride was.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BacktestResult:
    """How a strategy did over the test period.

    - `pnl`: total profit or loss ("P&L"), in GBP per MWh terms.
    - `n_trades`: how many times the strategy changed its position.
    - `sharpe`: profit measured against how bumpy the ride was. Higher is
      better: it rewards steady gains and punishes wild swings.
    - `max_drawdown`: the worst peak-to-trough fall along the way, i.e. the
      most you would have been down at any point.
    - `per_step_pnl`: the profit or loss in each individual step.
    """

    pnl: float
    n_trades: int
    sharpe: float
    max_drawdown: float
    per_step_pnl: np.ndarray


def backtest_signal(
    spread: np.ndarray,
    signals: np.ndarray,
    cost_per_trade: float = 0.5,
) -> BacktestResult:
    """Replay a buy/sell/do-nothing strategy over past data and score it.

    `spread` is the price gap over time; `signals` is what the strategy
    decided at each step (+1 buy, -1 sell, 0 nothing). `cost_per_trade` is
    what we charge, per unit, each time the position changes.

    The important detail: we hold *yesterday's* decision into *today's* price
    move. That one-step delay is what stops the strategy from cheating by
    reacting to a move it should not yet have seen.
    """
    if spread.shape != signals.shape:
        raise ValueError("spread and signals must be the same length")
    # Yesterday's decision, applied to today: shift the signals by one step.
    position = np.concatenate([[0], signals[:-1]])
    todays_move = np.diff(spread, prepend=spread[0])
    profit_before_costs = position * todays_move

    # Charge a cost every time we change our position.
    position_changes = np.abs(np.diff(position, prepend=0))
    costs = position_changes * cost_per_trade
    profit_after_costs = profit_before_costs - costs

    # Track the running total to find the worst dip along the way.
    running_total = np.cumsum(profit_after_costs)
    peak_so_far = np.maximum.accumulate(running_total)
    dip_from_peak = running_total - peak_so_far

    spread_of_returns = profit_after_costs.std()
    if spread_of_returns > 0:
        sharpe = float(
            profit_after_costs.mean()
            / spread_of_returns
            * np.sqrt(len(profit_after_costs))
        )
    else:
        sharpe = 0.0

    return BacktestResult(
        pnl=float(running_total[-1]),
        n_trades=int((position_changes > 0).sum()),
        sharpe=sharpe,
        max_drawdown=float(dip_from_peak.min()),
        per_step_pnl=profit_after_costs,
    )
