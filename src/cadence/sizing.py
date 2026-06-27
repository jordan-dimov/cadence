"""Section 4: Decision under uncertainty - sizing the trade.

The day-ahead bid is an optimisation, not a strategy. Given a predictive
distribution of generation (from Section 3) and the asymmetric cost of
being short versus long at delivery, how much should Alice sell into the
blind auction? This is structurally the newsvendor problem, and its answer
is a *fractile* of the distribution set by the cost ratio - not the mean.

Key insight made quantitative: the optimal volume depends on the cost
asymmetry. If being short costs more than being long, you sell less than
the mean; if symmetric, you sell the median. The guide derives this; the
code is the derivation made runnable.
"""

from __future__ import annotations

from .forecast import ForecastDistribution


def optimal_volume(
    dist: ForecastDistribution,
    short_cost: float,
    long_cost: float,
) -> float:
    """Newsvendor-optimal day-ahead volume.

    `short_cost` is the penalty per MWh of selling more than you produce
    (buy back at imbalance); `long_cost` is the penalty per MWh of selling
    less than you produce (spill / sell down at imbalance). Both > 0.

    Newsvendor: selling too little is the underage (cost `long_cost`, you
    dump the excess); selling too much is the overage (cost `short_cost`,
    you buy back). The optimal volume is the quantile at the critical
    fractile long_cost / (short_cost + long_cost). So the more painful it is
    to be short, the lower the fractile and the less you sell.
    """
    if short_cost <= 0 or long_cost <= 0:
        raise ValueError("costs must be positive")
    critical_fractile = long_cost / (short_cost + long_cost)
    return max(0.0, dist.quantile(critical_fractile))
