"""Simulated market data. Deterministic (seeded) so every example in the
guide is reproducible and runs with no exchange access.

Section 2 of the guide motivates why a teaching repo simulates rather than
replays a real feed: a real feed is non-reproducible, licence-encumbered,
and obscures the logic under reconnection and gap handling. These
generators give honest *shapes* (diurnal price, solar-driven spread,
forecast uncertainty) without pretending to be real prints.

Standard levels follow content/guide-reference-data-2026.md: power around
GBP 80/MWh. Quantities are MWh, prices GBP/MWh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PERIODS_PER_DAY = 96  # quarter-hourly products (ACER 15-min methodology, 2025)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def simulated_generation(
    capacity_mw: float = 100.0, seed: int = 0
) -> np.ndarray:
    """A wind farm's per-period output (MWh) over one day. Lumpy and
    forecastable-but-uncertain: a slow synoptic component plus gusts."""
    rng = _rng(seed)
    t = np.linspace(0, 2 * np.pi, PERIODS_PER_DAY)
    synoptic = 0.45 + 0.25 * np.sin(t + rng.uniform(0, 2 * np.pi))
    gusts = rng.normal(0, 0.07, PERIODS_PER_DAY)
    load_factor = np.clip(synoptic + gusts, 0.0, 1.0)
    return load_factor * capacity_mw * 0.25  # MWh per quarter-hour


def simulated_prices(seed: int = 0, base: float = 80.0) -> np.ndarray:
    """Day-ahead-like price path (GBP/MWh): diurnal shape plus noise, with
    an evening peak and a midday solar trough."""
    rng = _rng(seed)
    q = np.arange(PERIODS_PER_DAY)
    hour = q / 4.0
    diurnal = -18 * np.cos((hour - 19) / 24 * 2 * np.pi)  # peak ~19:00
    solar_trough = -22 * np.exp(-((hour - 13) ** 2) / 8)  # dip ~13:00
    noise = rng.normal(0, 4, PERIODS_PER_DAY)
    return base + diurnal + solar_trough + noise


@dataclass(frozen=True)
class OrderBook:
    """A one-sided-depth snapshot: price levels and the volume resting at
    each. Bids descending, asks ascending. Section 5 executes against this;
    the order-book *concepts* are taught in the Market Microstructure guide.
    """

    bid_prices: np.ndarray
    bid_sizes: np.ndarray
    ask_prices: np.ndarray
    ask_sizes: np.ndarray

    @property
    def best_bid(self) -> float:
        return float(self.bid_prices[0])

    @property
    def best_ask(self) -> float:
        return float(self.ask_prices[0])

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2


def simulated_orderbook(
    mid: float = 80.0, levels: int = 5, tick: float = 0.1, seed: int = 0
) -> OrderBook:
    """A simple book around `mid`, thinning with depth."""
    rng = _rng(seed)
    spread = tick * rng.integers(1, 4)
    sizes = np.abs(rng.normal(8, 3, levels)).round(1) + 1.0
    bid_prices = mid - spread / 2 - tick * np.arange(levels)
    ask_prices = mid + spread / 2 + tick * np.arange(levels)
    return OrderBook(bid_prices, sizes.copy(), ask_prices, sizes.copy())


def simulated_spread_series(
    days: int = 30, seed: int = 0
) -> np.ndarray:
    """A mean-reverting cross-border spread (e.g. OPCOM-HUPX), one value per
    delivery hour over `days`. Ornstein-Uhlenbeck around zero with a daily
    solar-driven wobble. Used by the Section 6 stat-arb bot."""
    rng = _rng(seed)
    n = days * 24
    theta, mu, sigma = 0.15, 0.0, 2.5
    x = np.empty(n)
    x[0] = 0.0
    hour_of_day = np.arange(n) % 24
    seasonal = 1.5 * np.sin((hour_of_day - 13) / 24 * 2 * np.pi)
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) + rng.normal(0, sigma)
    return x + seasonal
