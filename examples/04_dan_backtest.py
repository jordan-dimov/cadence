"""Dan's cross-border stat-arb signal, backtested honestly.

    uv run python examples/04_dan_backtest.py
"""

from cadence import data
from cadence.backtest import backtest_signal
from cadence.signals import zscore_signals

gap = data.simulated_country_gap(days=90, seed=2)
signal = zscore_signals(gap, window=14 * 24, threshold=2.0)

free = backtest_signal(gap, signal, cost_per_trade=0.0)
real = backtest_signal(gap, signal, cost_per_trade=0.5, impact=0.2)

print("Dan trades the price gap between two countries when it looks unusual.\n")
print(f"Trades taken: {real.n_trades}\n")
print(f"P&L if trading were free  : {free.pnl:8.1f}")
print(f"P&L with costs and impact : {real.pnl:8.1f}")
print(f"Worst drawdown (with costs): {real.max_drawdown:8.1f}")
print()
print("The honest number is the one with costs. 'It made money for free' is")
print("not an answer, and the busiest strategies suffer the most once you pay.")
