"""Section 6: A bot that trades a price gap between two countries.

This is Dan's bot. Unlike Alice, Dan owns no wind farm and no power. He just
looks for a pattern and bets on it, and he can sit and do nothing when there
is no pattern worth trading.

His pattern: the price gap between two neighbouring countries (say Romania and
Hungary) usually hovers around some normal level, but every so often it
stretches unusually wide. When it does, it tends to snap back. So when the
gap looks abnormally large, Dan bets it will return to normal.

How do we measure "abnormally large"? We compare today's gap to how the gap
has behaved recently, its recent average and its recent wobbliness. If the
gap is more than two wobbles away from its recent average, that is unusual
enough to act on. (The "number of wobbles away from average" has a name, the
z-score, used below.)

Crucially, we only ever look at the *past* when judging today. Peeking at the
future, even by accident, is the single biggest way to fool yourself into
thinking a strategy works. The backtest module is all about avoiding that.
"""

from __future__ import annotations

import numpy as np


def rolling_zscore(series: np.ndarray, window: int) -> np.ndarray:
    """Measure how unusual each value is, compared with the recent past.

    For each point, we look back over the last `window` values (and only
    those, never the point itself or anything after it). We work out their
    average and their spread, then express the current value as "how many
    spreads away from that average" it is. This "spreads-away-from-average"
    number is called the z-score.

    A z-score near 0 means business as usual. A z-score of +2 means today is
    unusually high; -2 means unusually low. The first `window` points have no
    history to compare against, so they come back as NaN ("not a number").
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    n = series.size
    z = np.full(n, np.nan)
    for i in range(window, n):
        recent_past = series[i - window : i]
        average = recent_past.mean()
        spread = recent_past.std()
        if spread > 0:
            z[i] = (series[i] - average) / spread
    return z


def zscore_signals(
    series: np.ndarray, window: int = 14 * 24, threshold: float = 2.0
) -> np.ndarray:
    """Turn the "how unusual is it" measure into buy / sell / do-nothing.

    We act only when today is more than `threshold` spreads from its recent
    average:
      - gap unusually low  -> +1, bet it rises back to normal (buy the gap)
      - gap unusually high -> -1, bet it falls back to normal (sell the gap)
      - otherwise          ->  0, do nothing

    The default `window` is 14 days of hourly values, matching the worked
    example in the guide: long enough to know what "normal" is, short enough
    to keep up as conditions change.
    """
    z = rolling_zscore(series, window)
    signal = np.zeros(series.size, dtype=int)
    signal[z >= threshold] = -1
    signal[z <= -threshold] = 1
    return signal
