"""Dan's cross-border stat-arb: the lifecycle, then a backtest of that same
lifecycle.

    uv run python examples/04_dan_stat_arb.py
"""

from cadence import data
from cadence.backtest import backtest_signal
from cadence.signals import run_stat_arb, stat_arb_positions

gap = data.simulated_country_gap(days=90, seed=2)

print("Dan trades the price gap between two countries when it looks unusual.\n")

# A signal is not a strategy. The lifecycle goes in, holds, and comes out.
trades = run_stat_arb(gap, window=14 * 24)
print(f"Round trips taken by the bot: {len(trades)}")
if trades:
    first = trades[0]
    held = first.exit - first.entry
    way = "sold the gap" if first.direction == -1 else "bought the gap"
    print(f"  first round trip: {way}, held {held} hours, made {first.pnl:+.1f}")
print(f"  sum of round-trip P&L (before costs): {sum(t.pnl for t in trades):+.1f}")
print()

# Backtest the SAME lifecycle (its held positions), not the bare signal, so
# the numbers below describe the bot above.
positions = stat_arb_positions(gap, window=14 * 24)
free = backtest_signal(gap, positions, cost_per_trade=0.0)
real = backtest_signal(gap, positions, cost_per_trade=0.5, impact=0.2)
print(f"P&L if trading were free  : {free.pnl:8.1f}")
print(f"P&L with costs and impact : {real.pnl:8.1f}")
print(f"Worst drawdown (with costs): {real.max_drawdown:8.1f}")
print()
print("The honest number is the one with costs. 'It made money for free' is")
print("not an answer, and the busiest strategies suffer the most once you pay.")
