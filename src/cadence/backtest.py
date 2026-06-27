"""Section 7: Would this strategy actually have made money?

Before risking real money, you replay a strategy over past data and see how
it would have done. That is a backtest. It sounds simple, and it is full of
traps that make a losing strategy look like a winner.

Three traps turn a losing strategy into an apparent winner, and this module
guards against each:

  - Peeking at the future. If your strategy ever uses information it could not
    have known at the time, even by accident, the backtest is a fantasy. We
    guard against this by always acting on *yesterday's* decision, never
    today's.
  - Pretending trading is free. Every trade costs something (fees, and the
    small loss from crossing the gap between buy and sell prices). Ignore
    that and any strategy looks better than it really is. We charge a cost
    for every trade, and you can watch the apparent profit shrink once you do.
  - Assuming you fill at the price on the screen. A big order moves the price
    against you, so `impact` adds a cost that grows with the size of the
    trade, not just the number of trades.

And it tests honestly: `walk_forward` scores the strategy on several
successive out-of-sample chunks rather than one split, the way live trading
only ever knows the past.
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
    gap: np.ndarray,
    signals: np.ndarray,
    cost_per_trade: float = 0.5,
    impact: float = 0.0,
) -> BacktestResult:
    """Replay a buy/sell/do-nothing strategy over past data and score it.

    `gap` is the price gap over time; `signals` is what the strategy decided
    at each step (+1 buy, -1 sell, 0 nothing). `cost_per_trade` is the flat
    cost per unit each time the position changes (fees plus half the bid-ask
    gap). `impact` adds market impact: a cost that grows with the *square* of
    the trade size, because a bigger order pushes the price further against
    you. Set it above zero and watch a busy strategy's apparent edge shrink.

    The important detail: we hold *yesterday's* decision into *today's* price
    move. That one-step delay is what stops the strategy from cheating by
    reacting to a move it should not yet have seen.
    """
    if gap.shape != signals.shape:
        raise ValueError("gap and signals must be the same length")
    # Yesterday's decision, applied to today: shift the signals by one step.
    position = np.concatenate([[0], signals[:-1]])
    todays_move = np.diff(gap, prepend=gap[0])
    profit_before_costs = position * todays_move

    # Charge each time the position changes: a flat cost, plus a market-impact
    # term that grows with the size of the change (a bigger trade hurts more).
    position_changes = np.abs(np.diff(position, prepend=0))
    costs = position_changes * cost_per_trade + position_changes**2 * impact
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


def walk_forward(
    gap: np.ndarray,
    signals: np.ndarray,
    n_folds: int = 4,
    cost_per_trade: float = 0.5,
    impact: float = 0.0,
) -> list[BacktestResult]:
    """Score the strategy on several successive out-of-sample chunks, instead
    of one split.

    A single train/test split gives you one number, and it is chosen knowing
    how the whole history played out. Walk-forward instead cuts the history
    into `n_folds` successive pieces and backtests each in turn, so you get
    many out-of-sample scores and only ever judge a piece on its own stretch
    of time, the way live trading only ever knows the past.

    Note this is the simple version: it only answers "was performance stable
    across time?". It does not re-tune the strategy on each window (full
    walk-forward optimisation), which is safe here only because this z-score
    signal already looks at nothing but the past.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    fold = gap.size // n_folds
    if fold == 0:
        raise ValueError("not enough data for that many folds")
    results = []
    for k in range(n_folds):
        chunk = slice(k * fold, (k + 1) * fold)
        results.append(
            backtest_signal(gap[chunk], signals[chunk], cost_per_trade, impact)
        )
    return results
