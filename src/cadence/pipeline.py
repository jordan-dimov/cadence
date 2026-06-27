"""Section 9: Putting it all together, Alice's full day.

This is the finale: every piece so far, working as one bot. Alice the wind
farm owner needs to sell her power, so her bot:

  1. guesses tomorrow's wind as a range, not a single number (forecast)
  2. decides how much to sell, erring on the safe side (sizing)
  3. sends the orders through the safety gate before they count (risk)
  4. tallies up how much got sold

It runs on the invented data, so you can run the whole thing yourself with
nothing installed. The guide then extends it: reacting to updated forecasts
through the day, and deciding what the bot should do when something goes
wrong (a missing forecast, a rejected order, a dropped data feed).
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
    """What happened over the simulated day."""

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
    """Run one full day of Alice's bot against invented data.

    `short_cost` and `long_cost` say how painful it is to sell too much
    versus too little (see the sizing module). `position_limit` is the most
    the safety gate will let her sell in total.
    """
    gate = governor or make_governor()
    gate.open_book(book, position_limit)
    gate.enable_strategy(strategy)

    # 1. Guess the day's wind output as a range of possibilities.
    past_output = data.simulated_generation(seed=seed)
    forecast = PriceForecaster().fit(past_output).predict()

    # 2. Decide how much to sell (a sale is a negative quantity).
    volume = optimal_volume(forecast, short_cost=short_cost, long_cost=long_cost)

    # 3. Send it in several pieces, each through the safety gate. Splitting it
    #    up means the position limit is checked piece by piece, as it would be
    #    with real orders.
    pieces = 8
    per_piece = volume / pieces
    average_price = float(np.mean(data.simulated_prices(seed=seed)))
    admitted = refused = 0
    for i in range(pieces):
        order = Order(
            order_id=f"{book}-{seed}-{i}",
            strategy=strategy,
            book=book,
            signed_qty=-per_piece,  # selling
            price=average_price,
        )
        decision = gate.admit(order)
        if decision.admitted:
            admitted += 1
        else:
            refused += 1

    return DayResult(
        sold_mwh=per_piece * admitted,
        admitted_orders=admitted,
        refused_orders=refused,
    )
