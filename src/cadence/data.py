"""Fake-but-realistic market data, so every example runs on your laptop.

A real trading bot plugs into live exchange feeds. We cannot ship those: they
cost money, need logins, and would make every example give different results
each time you run it. So instead this module *invents* data that has the
right shape, the daily rise and fall of prices, the gustiness of wind, the
way a cross-border price gap drifts and snaps back, without being real.

Everything here is reproducible: give the same `seed` (a starting number for
the random generator) and you get the exact same data every time, so the
examples and tests behave the same way for everyone.

Quantities are in MWh (megawatt-hours, a chunk of energy). Prices are in
GBP per MWh. The typical price level, around 80, follows the shared figures
the whole course uses (see content/guide-reference-data-2026.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A trading day is split into 96 quarter-hour slots. European power markets
# moved to 15-minute products, so a single day has 24 * 4 = 96 of them.
PERIODS_PER_DAY = 96


def _rng(seed: int) -> np.random.Generator:
    """A random-number generator pinned to `seed`, so results repeat."""
    return np.random.default_rng(seed)


def simulated_generation(
    capacity_mw: float = 100.0, seed: int = 0
) -> np.ndarray:
    """Invent one day of wind-farm output, slot by slot (in MWh).

    The shape mimics real wind: a slow underlying trend across the day (the
    weather), plus quick random gusts on top. The result is never more than
    the farm's size, and never below zero.
    """
    rng = _rng(seed)
    t = np.linspace(0, 2 * np.pi, PERIODS_PER_DAY)
    weather_trend = 0.45 + 0.25 * np.sin(t + rng.uniform(0, 2 * np.pi))
    gusts = rng.normal(0, 0.07, PERIODS_PER_DAY)
    fraction_of_capacity = np.clip(weather_trend + gusts, 0.0, 1.0)
    return fraction_of_capacity * capacity_mw * 0.25  # MWh in 15 minutes


def simulated_period_history(
    period: int, days: int = 30, capacity_mw: float = 100.0, seed: int = 0
) -> np.ndarray:
    """Invent recent output readings for the SAME delivery period across many
    past days (in MWh).

    This is the history a forecaster should actually fit on: not one day's 96
    different periods, but the same quarter-hour slot seen on each of the last
    `days` days. Wind output for a given slot varies day to day with the
    weather, around a typical level, which is exactly what these readings show.
    """
    if not 0 <= period < PERIODS_PER_DAY:
        raise ValueError(f"period must be in 0..{PERIODS_PER_DAY - 1}")
    rng = _rng(seed * PERIODS_PER_DAY + period)
    typical_load_factor = 0.45
    load_factors = np.clip(rng.normal(typical_load_factor, 0.12, days), 0.0, 1.0)
    return load_factors * capacity_mw * 0.25  # MWh per quarter-hour


def simulated_prices(seed: int = 0, base: float = 80.0) -> np.ndarray:
    """Invent one day of power prices (GBP per MWh), slot by slot.

    The shape mimics a real day: prices peak in the early evening when demand
    is high, and dip around midday when solar power floods the grid. Random
    noise is added so no two days look identical.
    """
    rng = _rng(seed)
    slot = np.arange(PERIODS_PER_DAY)
    hour = slot / 4.0
    # A bump up in the early evening (demand peak), a dip down around midday
    # (solar floods the grid). Both are shaped so the effect fades away from
    # its hour, which reads more intuitively than a single sine wave.
    evening_peak = 18 * np.exp(-((hour - 19) ** 2) / 10)
    midday_solar_dip = -22 * np.exp(-((hour - 13) ** 2) / 8)
    noise = rng.normal(0, 4, PERIODS_PER_DAY)
    return base + evening_peak + midday_solar_dip + noise


@dataclass(frozen=True)
class OrderBook:
    """A snapshot of what is on offer at an exchange right now.

    Buyers post "bids" (prices they will pay) and sellers post "asks" (prices
    they want). Each side is a ladder of prices with some volume waiting at
    each rung. Bids run from highest down, asks from lowest up. The execution
    module sells and buys against this; the Market Microstructure guide
    explains order books in full.
    """

    bid_prices: np.ndarray
    bid_sizes: np.ndarray
    ask_prices: np.ndarray
    ask_sizes: np.ndarray

    @property
    def best_bid(self) -> float:
        """The highest price a buyer is currently offering."""
        return float(self.bid_prices[0])

    @property
    def best_ask(self) -> float:
        """The lowest price a seller is currently asking."""
        return float(self.ask_prices[0])

    @property
    def mid(self) -> float:
        """The midpoint between the best bid and best ask, a rough "fair"
        price right now."""
        return (self.best_bid + self.best_ask) / 2


def simulated_orderbook(
    mid: float = 80.0, levels: int = 5, tick: float = 0.1, seed: int = 0
) -> OrderBook:
    """Invent a simple order book centred on the price `mid`.

    The further a price rung is from the middle, the less volume waits there,
    which is what real books look like: lots on offer near the fair price,
    thinning out as you go further away.
    """
    rng = _rng(seed)
    bid_ask_gap = tick * rng.integers(1, 4)
    sizes = np.abs(rng.normal(8, 3, levels)).round(1) + 1.0
    bid_prices = mid - bid_ask_gap / 2 - tick * np.arange(levels)
    ask_prices = mid + bid_ask_gap / 2 + tick * np.arange(levels)
    return OrderBook(bid_prices, sizes.copy(), ask_prices, sizes.copy())


def simulated_country_gap(days: int = 30, seed: int = 0) -> np.ndarray:
    """Invent the price gap between two neighbouring countries, hour by hour,
    over several days. (Traders call this gap a "spread".)

    Power often costs slightly more in one country than its neighbour. The gap
    wanders about but tends to drift back toward zero: when it gets too wide,
    traders pile in and close it. On top of that it has a daily swing as solar
    comes and goes. Dan's bot (in the signals module) tries to profit from the
    drift-back. One value per delivery hour.
    """
    rng = _rng(seed)
    n = days * 24
    pull_back, jitter = 0.15, 2.5
    gap = np.empty(n)
    gap[0] = 0.0
    hour_of_day = np.arange(n) % 24
    daily_swing = 1.5 * np.sin((hour_of_day - 13) / 24 * 2 * np.pi)
    for i in range(1, n):
        # Each hour: drift a little back toward zero, plus a random nudge.
        gap[i] = gap[i - 1] * (1 - pull_back) + rng.normal(0, jitter)
    return gap + daily_swing
