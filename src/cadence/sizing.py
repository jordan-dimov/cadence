"""Section 4: How much should we actually sell?

Alice runs a wind farm. Tomorrow she can sell her power in advance, today,
at a known price. The problem: she does not yet know how much the wind will
produce. Her best guess might be 40 MWh, but it could come in at 30, or 50.

So how much should she sell today? This is the question this module answers.

Think about the two ways she can be wrong:

  - She sells 40, the wind only makes 30. Now she is "short": she promised
    10 MWh she does not have, and has to buy it back, usually at a bad price.
  - She sells 40, the wind makes 50. Now she is "long": she has 10 MWh
    spare she did not sell in advance, and has to sell it late, usually for
    a bit less than she could have got.

Both mistakes cost money, but not always the same amount. The key idea: if
being short hurts more than being long, she should sell a little LESS than
her best guess, to stay on the safer side. If the two hurt equally, she
sells exactly her middle estimate. The function below turns that intuition
into a number.
"""

from __future__ import annotations

from .forecast import ForecastDistribution


def optimal_volume(
    dist: ForecastDistribution,
    short_cost: float,
    long_cost: float,
) -> float:
    """Work out how much Alice should sell in advance.

    Inputs:
      - `dist` is her forecast, not as a single number but as a range of
        possible outcomes (see the forecast module for why that matters).
      - `short_cost` is how much it costs her, per MWh, to sell too much
        (she has to buy the shortfall back).
      - `long_cost` is how much it costs her, per MWh, to sell too little
        (she has to offload the surplus late).

    The rule that balances the two costs is an old and well known one (it is
    called the "newsvendor" problem, after a newsagent deciding how many
    papers to stock: too few and you miss sales, too many and they go to
    waste). It says: pick the sell amount that the forecast expects her to
    beat with this probability:

        long_cost / (short_cost + long_cost)

    A quick sanity check on that fraction. If the two costs are equal, it is
    1/2, so she sells her middle estimate. If selling too much is twice as
    painful (short_cost high), the fraction drops below 1/2, so she sells
    less. That is exactly the safer behaviour we wanted.
    """
    if short_cost <= 0 or long_cost <= 0:
        raise ValueError("costs must be positive")
    safety_level = long_cost / (short_cost + long_cost)
    # Ask the forecast: what amount will the wind beat with this probability?
    return max(0.0, dist.quantile(safety_level))
