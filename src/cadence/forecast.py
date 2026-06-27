"""Section 3: Forecasting as code.

The load-bearing teaching point lives here: a forecast must be a
*distribution*, not a point, because the sizing decision in Section 4 is
asymmetric. A point forecast of 40 MWh hides whether the spread is +/-2 or
+/-20, and the optimal bid depends entirely on that spread.

So `PriceForecaster` returns a `ForecastDistribution`, not a float. The
baseline here is a deliberately simple climatology-plus-residual model;
the guide climbs the ladder from this to gradient-boosted trees and
sequence models (LSTM / TCN / Transformer), and discusses why desks keep
a parsimonious, explainable core for regulatory reasons (Section 8). Many
desks buy this layer (e.g. Volue Insight) rather than build it; we build it
so you understand what you would otherwise rent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForecastDistribution:
    """A predictive distribution over one quantity (price or generation),
    summarised as a mean and standard deviation of an assumed-normal law.
    Real forecasters return quantiles or scenario ensembles; the interface
    is what matters - `quantile` and `scenarios` are what Section 4 consumes.
    """

    mean: float
    std: float

    def quantile(self, q: float) -> float:
        """Inverse-CDF. The sizing optimiser asks for a fractile, not a
        mean - this is the seam between forecasting and decision."""
        if not 0.0 < q < 1.0:
            raise ValueError("quantile q must be in (0, 1)")
        from statistics import NormalDist

        return NormalDist(self.mean, self.std).inv_cdf(q)

    def scenarios(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(self.mean, self.std, n)


class PriceForecaster:
    """A baseline climatology forecaster: predict each period from the mean
    and dispersion of its historical analogues. Honest and weak on purpose;
    Section 3 replaces the internals, not the interface.
    """

    def __init__(self) -> None:
        self._mean: float | None = None
        self._std: float | None = None

    def fit(self, history: np.ndarray) -> "PriceForecaster":
        if history.size == 0:
            raise ValueError("cannot fit on empty history")
        self._mean = float(np.mean(history))
        self._std = float(np.std(history)) or 1.0
        return self

    def predict(self) -> ForecastDistribution:
        if self._mean is None or self._std is None:
            raise RuntimeError("call fit() before predict()")
        return ForecastDistribution(self._mean, self._std)
