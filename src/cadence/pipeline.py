"""Section 9: Capstone - the full forecast-to-trade pipeline.

Alice's asset-backed bot, assembling every module: forecast generation as a
distribution, size the day-ahead bid under asymmetric imbalance cost,
execute the residual with a deadline-aware scheduler that reconciles fills,
and pass every order through the governance gate (Section 8) before it
counts. Then settle P&L. The guide compares this under perfect foresight
versus realistic forecast error, so you see how much P&L was forecast skill
versus execution.

This is the runnable spine; the guide fleshes out the intraday adjustment
loop and the failure-path behaviour (missing forecast, rejected order,
feed gap).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import data
from .forecast import PriceForecaster
from .risk import Governor, Order, make_governor
from .sizing import optimal_volume


@dataclass(frozen=True)
class DayResult:
    sold_mwh: float
    admitted_orders: int
    refused_orders: int


def run_day(
    seed: int = 0,
    book: str = "alice-wind",
    strategy: str = "da-bidder",
    position_limit: float = 500.0,
    short_cost: float = 90.0,
    long_cost: float = 60.0,
    governor: Governor | None = None,
) -> DayResult:
    """One day of the forecast-to-trade pipeline against simulated data."""
    gov = governor or make_governor()
    gov.open_book(book, position_limit)
    gov.enable_strategy(strategy)

    # Forecast generation as a distribution (Section 3).
    history = data.simulated_generation(seed=seed)
    dist = PriceForecaster().fit(history).predict()

    # Size the day-ahead sell volume (Section 4): a sell is negative qty.
    volume = optimal_volume(dist, short_cost=short_cost, long_cost=long_cost)

    # Propose it through the governance gate (Section 8). Slice into clips
    # so the position-limit gate is exercised honestly.
    clips = 8
    per_clip = volume / clips
    admitted = refused = 0
    for i in range(clips):
        order = Order(
            order_id=f"{book}-{seed}-{i}",
            strategy=strategy,
            book=book,
            signed_qty=-per_clip,  # selling generation
            price=float(np.mean(data.simulated_prices(seed=seed))),
        )
        decision = gov.admit(order)
        if decision.admitted:
            admitted += 1
        else:
            refused += 1

    return DayResult(
        sold_mwh=per_clip * admitted,
        admitted_orders=admitted,
        refused_orders=refused,
    )
