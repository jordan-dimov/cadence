"""Section 9: Putting it all together, one delivery period at a time.

This is the finale: every piece so far, working as one bot for Alice. The
unit is one delivery period (one of the day's ninety-six quarter-hour blocks),
because that is where the sizing decision is actually made (see the sizing
module). For one period her bot:

  1. forecasts that period's output as a range, not a single number (forecast)
  2. decides how much to sell, erring on the safe side (sizing)
  3. sends the orders through the safety gate before they count (risk)
  4. tallies up how much got sold

`run_period` does one block; `run_day` is the same loop run for each block of
the day. Everything runs on invented data, so you can run it yourself with
nothing installed. The guide then extends it: reacting to updated forecasts,
and deciding what the bot should do when something goes wrong (a missing
forecast, a rejected order, a dropped data feed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import data
from .forecast import ForecastDistribution, Forecaster
from .risk import Governor, Order, make_governor
from .sizing import optimal_volume


@dataclass(frozen=True)
class PeriodResult:
    """What happened for one delivery period."""

    sold_mwh: float
    admitted_orders: int
    refused_orders: int


def run_period(
    seed: int = 0,
    book: str = "alice-wind",
    strategy: str = "da-bidder",
    position_limit: float = 500.0,
    short_cost: float = 90.0,
    long_cost: float = 60.0,
    governor: Governor | None = None,
) -> PeriodResult:
    """Run Alice's bot for one delivery period against invented data.

    `short_cost` and `long_cost` say how painful it is to sell too much
    versus too little (see the sizing module). `position_limit` is the most
    the safety gate will let her sell for this period.
    """
    gate = governor or make_governor()
    gate.open_book(book, position_limit)
    gate.enable_strategy(strategy)

    # 1. Forecast this period's output as a range, from recent readings.
    recent_readings = data.simulated_generation(seed=seed)
    forecast = Forecaster().fit(recent_readings).predict()

    # 2. Decide how much to sell (a sale is a negative quantity).
    volume = optimal_volume(forecast, short_cost=short_cost, long_cost=long_cost)

    # 3. Send it in several pieces, each through the safety gate. Splitting it
    #    up means the position limit is checked piece by piece, as it would be
    #    with real orders.
    pieces = 8
    per_piece = volume / pieces
    price = float(np.mean(data.simulated_prices(seed=seed)))
    admitted = refused = 0
    for i in range(pieces):
        order = Order(
            order_id=f"{book}-{seed}-{i}",
            strategy=strategy,
            book=book,
            signed_qty=-per_piece,  # selling
            price=price,
        )
        if gate.admit(order).admitted:
            admitted += 1
        else:
            refused += 1

    return PeriodResult(
        sold_mwh=per_piece * admitted,
        admitted_orders=admitted,
        refused_orders=refused,
    )


@dataclass(frozen=True)
class DayResult:
    """A whole day: the per-period loop, summed up."""

    periods: int
    sold_mwh: float
    admitted_orders: int
    refused_orders: int


def run_day(seed: int = 0, n_periods: int = data.PERIODS_PER_DAY) -> DayResult:
    """A whole day is just `run_period` run once for each block. Each block
    gets its own forecast and its own decision; this loop is the scaling
    argument from Section 1 made concrete."""
    results = [run_period(seed=seed * 1000 + i) for i in range(n_periods)]
    return DayResult(
        periods=len(results),
        sold_mwh=sum(r.sold_mwh for r in results),
        admitted_orders=sum(r.admitted_orders for r in results),
        refused_orders=sum(r.refused_orders for r in results),
    )


def settle(
    sold: float, actual: float, short_cost: float, long_cost: float
) -> float:
    """The imbalance penalty for selling `sold` MWh when output turned out to
    be `actual`. Selling too much means buying the shortfall back (at
    short_cost per MWh); selling too little means offloading the surplus late
    (at long_cost per MWh). Selling exactly the right amount costs nothing."""
    if sold > actual:
        return (sold - actual) * short_cost
    return (actual - sold) * long_cost


def compare_perfect_and_realistic(
    seed: int = 0, short_cost: float = 90.0, long_cost: float = 60.0
) -> dict[str, float]:
    """Run Alice's sizing decision for one period twice, and measure the cost
    of not being able to see the future. This is the experiment the capstone
    section of the guide describes.

    Once with the realistic forecast she would actually have had, and once
    with a perfect forecast that already knows the period's output. We then
    settle each against what really turned up and return the imbalance penalty
    of each. Perfect foresight sells exactly the right amount and pays almost
    nothing; the gap between the two is the price of uncertainty, and it is
    almost always larger than anything execution can win or lose.
    """
    recent_readings = data.simulated_generation(seed=seed)
    realistic = Forecaster().fit(recent_readings).predict()

    # What actually turns up: one draw from the true distribution.
    rng = np.random.default_rng(seed + 1)
    actual = float(rng.normal(realistic.mean, realistic.std))

    # Perfect foresight: a forecast centred exactly on the actual, no spread.
    perfect = ForecastDistribution(mean=actual, std=1e-9)

    sold_realistic = optimal_volume(realistic, short_cost, long_cost)
    sold_perfect = optimal_volume(perfect, short_cost, long_cost)

    penalty_realistic = settle(sold_realistic, actual, short_cost, long_cost)
    penalty_perfect = settle(sold_perfect, actual, short_cost, long_cost)
    return {
        "penalty_realistic": penalty_realistic,
        "penalty_perfect": penalty_perfect,
        "cost_of_not_knowing": penalty_realistic - penalty_perfect,
    }
