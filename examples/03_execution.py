"""Work a 100 MWh order into the market in pieces, three ways.

    uv run python examples/03_execution.py
"""

import numpy as np

from cadence.execution import cost_of_waiting, twap_schedule, urgency_schedule

total, slices = 100.0, 4
print(f"Sell {total:.0f} MWh in {slices} pieces.\n")
print("Even (TWAP)        :", np.round(twap_schedule(total, slices), 1))
print("Front-loaded (a=0.5):", np.round(urgency_schedule(total, slices, 0.5), 1))
print("Back-loaded  (a=2.0):", np.round(urgency_schedule(total, slices, 2.0), 1))
print()
cow = cost_of_waiting(prob_unfilled_at_gate=0.2, expected_imbalance_price=150.0)
print("Cost of waiting: a 20% chance of still being unfilled at gate closure,")
print(f"at an imbalance price of 150, costs {cow:.0f} GBP/MWh on average. That")
print("is why power execution leans on the deadline, not just the best price.")
