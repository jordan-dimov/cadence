# cadence

An automated power-trading bot, built component by component on simulated
European power-market data. This is the companion project to the guide
**Algorithmic Trading in Power Markets** (Practitioner Track, Energy
Trading & ETRM Skills). Each module is one section of the guide; together
they assemble the two archetypes of automated trading the guide teaches.

The name is the theme: automated trading is a problem of *timing* - working
orders into a continuous market on a beat, the control loop running each
tick, every decision racing a gate closure.

## The control loop

```text
market data  ->  forecast  ->  signal  ->  sizing decision  ->  pre-trade gate
   ->  execution  ->  fills  ->  position / P&L state  ->  monitoring  (loop)
```

| Module | Guide section | What it does |
|---|---|---|
| `data` | 2 | Seeded simulated generation, prices, order books, cross-border spread |
| `forecast` | 3 | A *distributional* forecaster (not a point: sizing needs the spread) |
| `sizing` | 4 | Newsvendor-optimal bid volume under asymmetric imbalance cost |
| `execution` | 5 | TWAP / urgency / liquidity schedulers, fill reconciliation |
| `signals` | 6 | Cross-border Z-score stat-arb (Dan, the pure-financial bot) |
| `backtest` | 7 | Cost-aware backtest with risk-adjusted metrics, no look-ahead |
| `risk` | 8 | The governance gate: position limit, kill switch, attribution |
| `pipeline` | 9 | The capstone forecast-to-trade run (Alice, the asset-backed bot) |

Two archetypes, contrasted on purpose: **Alice** is asset-backed and
forecast-driven (she must trade her wind); **Dan** is pure-financial and
signal-driven (he chooses his exposure and can sit flat).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14+.

```bash
uv sync --dev
uv run pytest -q
```

Everything runs on simulated data with no exchange or database access.

## The governance layer and Morpholog

`risk.py` is the pre-trade gate every order passes before it reaches the
exchange. It enforces what 2026 regulation now requires (REMIT II
order-and-trade logging; ACER per-transaction algo data from 29 October
2027; MiFID II-style kill switches and pre-trade risk controls): a position
limit, a kill switch per strategy, a circuit breaker per book, and
structural attribution so every order is traceable to the strategy that
produced it.

It ships with two interchangeable governors:

- **`InProcessGovernor`** (default): pure Python, the same rules, runnable
  with no infrastructure.
- **`MorphologGovernor`**: drives the [Morpholog](https://github.com/jordan-dimov/morpholog)
  runtime over [`cadence.morph`](./cadence.morph). The position limit lives
  in a runtime *invariant*, so an order that would breach it is **refused
  by the runtime and cannot be committed** - by any path, not just the one
  the bot remembered to check. The order-of-record is tamper-evident and
  replayable to any prior instant. This is "audit-grade by construction":
  the guarantee is the runtime's, not a logging convention the trading code
  has to be trusted to honour.

Select the runtime-backed governor with `CADENCE_GOVERNOR=morpholog` (needs
the `morpholog` binary, a disposable PostgreSQL database, and the generated
client: `morpholog generate python-client cadence.morph --out
src/cadence/_morph_client`). Section 8 of the guide walks the wiring.

## Status

Scaffold. Each module has a runnable core; the guide fills in the depth
(market-impact modelling, walk-forward validation, the intraday adjustment
loop, the full Morpholog wiring) section by section.
