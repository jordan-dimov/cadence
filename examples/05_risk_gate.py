"""The safety gate: what it lets through, and what it stops.

    uv run python examples/05_risk_gate.py
"""

from cadence.risk import InProcessGovernor, Order

gate = InProcessGovernor()
gate.open_book("alice-wind", limit=100.0)
gate.enable_strategy("da-bidder")


def show(order: Order) -> None:
    decision = gate.admit(order)
    if decision.admitted:
        outcome = "admitted"
    else:
        outcome = f"REFUSED ({decision.reason})"
    print(f"  sell {abs(order.signed_qty):5.1f} on {order.book} -> {outcome}")


print("Position limit on alice-wind: 100 MWh.\n")
show(Order("o1", "da-bidder", "alice-wind", -60, 80))   # within the limit
show(Order("o2", "da-bidder", "alice-wind", -50, 80))   # would take it to 110

print("\nSwitching the strategy off (kill switch):")
gate.engage_kill_switch("da-bidder")
show(Order("o3", "da-bidder", "alice-wind", -10, 80))   # refused at the source
