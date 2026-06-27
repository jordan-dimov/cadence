"""Section 3: Guessing tomorrow's numbers.

Before Alice can decide how much power to sell, she has to guess how much
the wind will make, and what prices will do. That guessing is forecasting.

The one idea this module is built to teach: a useful forecast is not a
single number. "The wind will make 40 MWh tomorrow" is almost useless on its
own, because it hides how sure you are. Compare two forecasts that both say
40:

  - Forecast A: "40, and I am very confident, it will land within a MWh or
    two."
  - Forecast B: "40, but honestly it could easily be 20 or 60."

Those should lead to very different decisions (the next module, sizing,
shows exactly how). So a forecast here is always a *range of possibilities*
with a most likely value in the middle, not a lone number.

To keep the example runnable and simple, we describe that range with just
two things: a middle value (the average) and a spread (how wide the range
is). Real trading desks use far richer models, and often buy this part
ready-made from specialist vendors such as Volue rather than build it. We
build a simple one anyway, so you understand what you would otherwise be
renting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForecastDistribution:
    """A forecast expressed as a range, not a single number.

    `mean` is the most likely value (the centre of the range). `std`, short
    for standard deviation, is how spread out the range is: small means
    confident, large means uncertain.
    """

    mean: float
    std: float

    def quantile(self, q: float) -> float:
        """Answer questions of the form "what value will the outcome stay
        below this fraction of the time?"

        For example, quantile(0.5) is the middle value (half the time the
        outcome is below it). quantile(0.9) is a high value the outcome only
        beats 10% of the time. The sizing module uses this to ask the
        forecast for a deliberately cautious number.
        """
        if not 0.0 < q < 1.0:
            raise ValueError("the fraction q must be between 0 and 1")
        from statistics import NormalDist

        return NormalDist(self.mean, self.std).inv_cdf(q)

    def scenarios(self, n: int, seed: int = 0) -> np.ndarray:
        """Draw `n` example outcomes from the range, for when you want to
        try a decision against many possible futures rather than reason
        about the range mathematically."""
        rng = np.random.default_rng(seed)
        return rng.normal(self.mean, self.std, n)


class PriceForecaster:
    """A deliberately simple forecaster: look at what happened in the past,
    and assume the future looks similar.

    It takes a history of past values and summarises them as a middle value
    and a spread. That is about the weakest honest forecast you can make, and
    that is the point: the guide starts here and then improves the forecast
    without changing how the rest of the bot uses it.
    """

    def __init__(self) -> None:
        self._mean: float | None = None
        self._std: float | None = None

    def fit(self, history: np.ndarray) -> "PriceForecaster":
        """Learn from past values: their average, and how much they varied."""
        if history.size == 0:
            raise ValueError("cannot learn from an empty history")
        self._mean = float(np.mean(history))
        self._std = float(np.std(history)) or 1.0
        return self

    def predict(self) -> ForecastDistribution:
        """Hand back the forecast as a range (a middle value and a spread)."""
        if self._mean is None or self._std is None:
            raise RuntimeError("call fit() with some history before predict()")
        return ForecastDistribution(self._mean, self._std)
