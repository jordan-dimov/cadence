"""Section 7: Backtesting, and knowing it actually works.

The engineer's guide to not fooling yourself. The single most expensive
error is look-ahead bias; the second is a backtest that fills at the touch
with no cost and no market impact and therefore always flatters the
strategy. This module models costs explicitly and reports risk-adjusted
metrics, because "it made money in the backtest" is not an answer.

The full guide adds market-impact modelling and walk-forward validation
(TODO here); this core is enough to show the Section 6 bot's edge shrink
once costs are honest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BacktestResult:
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
    """Backtest a spread strategy. We hold yesterday's signalled position
    into today's spread change (the position is lagged by one step, so the
    decision uses only past information), and charge `cost_per_trade`
    GBP/MWh each time the position changes.

    `cost_per_trade` stands in for fees plus half-spread; the guide adds a
    market-impact term that scales with clip size.
    """
    if spread.shape != signals.shape:
        raise ValueError("spread and signals must align")
    position = np.concatenate([[0], signals[:-1]])  # lag: no look-ahead
    spread_change = np.diff(spread, prepend=spread[0])
    gross = position * spread_change
    turnover = np.abs(np.diff(position, prepend=0))
    costs = turnover * cost_per_trade
    net = gross - costs

    equity = np.cumsum(net)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    sd = net.std()
    sharpe = float(net.mean() / sd * np.sqrt(len(net))) if sd > 0 else 0.0

    return BacktestResult(
        pnl=float(equity[-1]),
        n_trades=int((turnover > 0).sum()),
        sharpe=sharpe,
        max_drawdown=float(drawdown.min()),
        per_step_pnl=net,
    )
