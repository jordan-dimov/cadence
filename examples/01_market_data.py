"""Invent one day of market data and look at its shape.

    uv run python examples/01_market_data.py
"""

from cadence import data

prices = data.simulated_prices(seed=1)
generation = data.simulated_generation(seed=1)
book = data.simulated_orderbook(seed=1)

print("One simulated day, 96 quarter-hour slots.\n")
print(f"Price (GBP/MWh): low {prices.min():5.1f}  mean {prices.mean():5.1f}  high {prices.max():5.1f}")
print(f"Wind  (MWh/slot): low {generation.min():5.1f}  mean {generation.mean():5.1f}  high {generation.max():5.1f}")
print(f"Order book mid: {book.mid:.2f}  (best bid {book.best_bid:.2f}, best ask {book.best_ask:.2f})")

peak = int(prices.argmax())
dip = int(prices.argmin())
print()
print(f"Most expensive slot: {peak // 4:02d}:{(peak % 4) * 15:02d}  ({prices[peak]:.1f})  -- the evening demand peak")
print(f"Cheapest slot:       {dip // 4:02d}:{(dip % 4) * 15:02d}  ({prices[dip]:.1f})  -- midday, when solar floods the grid")
