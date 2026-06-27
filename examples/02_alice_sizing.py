"""Alice decides how much wind power to sell in advance for one period.

    uv run python examples/02_alice_sizing.py
"""

from cadence.forecast import ForecastDistribution
from cadence.sizing import optimal_volume

forecast = ForecastDistribution(mean=40.0, std=10.0)
short_cost, long_cost = 90.0, 60.0
volume = optimal_volume(forecast, short_cost, long_cost)

print("Alice forecasts one delivery period's wind output.\n")
print(f"Most likely output  : {forecast.mean:5.1f} MWh")
print(f"Uncertainty (spread): {forecast.std:5.1f} MWh")
print(f"Cost of selling too much (short): GBP {short_cost:.0f}/MWh")
print(f"Cost of selling too little (long): GBP {long_cost:.0f}/MWh")
print(f"Recommended sale    : {volume:5.1f} MWh")
print()
print("Why below the most likely 40 MWh? Being short hurts more than being")
print("long, so Alice leans to the safe side and sells a little less.")
