"""cadence: an automated power-trading bot, built component by component.

Companion project to the guide "Algorithmic Trading in Power Markets"
(Practitioner Track, Energy Trading & ETRM Skills). Each module is one
section of the guide; the capstone (`pipeline`) and the stat-arb bot
(`signals` + `backtest`) assemble them into the two archetypes of
automated trading the guide teaches:

  - Alice: asset-backed, forecast-driven (forecast -> sizing -> execution)
  - Dan:   pure-financial, signal-driven  (signals -> execution)

Everything runs on simulated data (`data`) with no exchange or database
access. The governance layer (`risk`) can be backed by Morpholog for a
provable, replayable order-of-record; absent it, an in-process governor
with the same interface keeps the repo fully runnable.
"""

__version__ = "0.1.0"
