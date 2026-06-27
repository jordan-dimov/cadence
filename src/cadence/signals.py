"""Section 6: A worked signal-driven bot - cross-border stat-arb.

Dan's bot, the pure-financial archetype: it chooses its exposure and can
sit flat, unlike Alice's asset-backed pipeline. The strategy itself
(cross-border / DART spreads) is taught in the Power Trading Strategies
guide; here we build the *system* that runs it.

The signal, from the June 2026 OPCOM-HUPX example: a rolling mean and
standard deviation of the hourly spread (rolling, to track the solar
cycle), with a Z-score trigger - when the spread is more than `threshold`
standard deviations from its rolling mean, trade the expected snap-back.
"""

from __future__ import annotations

import numpy as np


def rolling_zscore(series: np.ndarray, window: int) -> np.ndarray:
    """Z-score of each point against the trailing `window` (excluding the
    point itself, so there is no look-ahead). Points before a full window
    are NaN. Section 7 attacks the temptation to fit thresholds on the same
    data you evaluate on; the trailing window is the first defence."""
    if window < 2:
        raise ValueError("window must be >= 2")
    n = series.size
    z = np.full(n, np.nan)
    for i in range(window, n):
        past = series[i - window : i]
        mu = past.mean()
        sd = past.std()
        if sd > 0:
            z[i] = (series[i] - mu) / sd
    return z


def zscore_signals(
    series: np.ndarray, window: int = 14 * 24, threshold: float = 2.0
) -> np.ndarray:
    """Entry signals from the rolling Z-score. +1 = spread is low, expect it
    to rise (buy the spread); -1 = spread is high, expect mean reversion
    (sell the spread); 0 = no position. Default window is 14 days of hourly
    spreads, matching the worked example."""
    z = rolling_zscore(series, window)
    sig = np.zeros(series.size, dtype=int)
    sig[z >= threshold] = -1
    sig[z <= -threshold] = 1
    return sig
