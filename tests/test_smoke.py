"""Smoke tests: every module's runnable core works on simulated data, with
no exchange or database. The Morpholog-backed path is exercised in a
separate, skipped-by-default integration test (it needs the binary + PG).
"""

import numpy as np

from cadence import data
from cadence.backtest import backtest_signal
from cadence.execution import (
    FillReconciler,
    cost_of_waiting,
    twap_schedule,
    urgency_schedule,
)
from cadence.forecast import ForecastDistribution, PriceForecaster
from cadence.pipeline import run_day
from cadence.risk import InProcessGovernor, Order
from cadence.signals import rolling_zscore, zscore_signals
from cadence.sizing import optimal_volume


def test_data_shapes():
    assert data.simulated_generation(seed=1).shape == (data.PERIODS_PER_DAY,)
    assert data.simulated_prices(seed=1).shape == (data.PERIODS_PER_DAY,)
    book = data.simulated_orderbook(seed=1)
    assert book.best_ask > book.best_bid


def test_forecast_returns_distribution():
    dist = PriceForecaster().fit(data.simulated_generation(seed=2)).predict()
    assert isinstance(dist, ForecastDistribution)
    assert dist.quantile(0.9) > dist.quantile(0.1)


def test_sizing_asymmetry():
    dist = ForecastDistribution(mean=40.0, std=10.0)
    # Symmetric costs -> sell the median (the mean here).
    assert optimal_volume(dist, 50, 50) == dist.quantile(0.5)
    # Short more costly than long -> sell less than the median.
    assert optimal_volume(dist, 90, 60) < dist.quantile(0.5)


def test_execution_schedulers_and_reconciliation():
    assert np.isclose(twap_schedule(100, 4).sum(), 100)
    assert np.isclose(urgency_schedule(100, 4, alpha=0.5).sum(), 100)
    # Front-loaded puts more in the first clip than the last.
    front = urgency_schedule(100, 4, alpha=0.5)
    assert front[0] > front[-1]
    assert cost_of_waiting(0.5, 200) == 100.0

    rec = FillReconciler(target=10.0)
    rec.record_fill(4.0)
    assert rec.residual == 6.0 and not rec.is_complete
    rec.record_fill(6.0)
    assert rec.is_complete


def test_signals_no_lookahead():
    spread = data.simulated_spread_series(days=30, seed=3)
    z = rolling_zscore(spread, window=48)
    assert np.isnan(z[:48]).all()  # no signal before a full window
    sig = zscore_signals(spread, window=48, threshold=2.0)
    assert set(np.unique(sig)).issubset({-1, 0, 1})


def test_backtest_charges_costs():
    spread = data.simulated_spread_series(days=60, seed=4)
    sig = zscore_signals(spread, window=14 * 24, threshold=2.0)
    free = backtest_signal(spread, sig, cost_per_trade=0.0)
    costed = backtest_signal(spread, sig, cost_per_trade=1.0)
    assert costed.pnl <= free.pnl  # honest costs never improve P&L
    assert costed.n_trades >= 0


def test_governor_enforces_rules():
    gov = InProcessGovernor()
    gov.open_book("b", limit=100.0)
    gov.enable_strategy("s")
    # Missing attribution is impossible: Order requires strategy + book.
    assert gov.admit(Order("o1", "s", "b", 50, 80)).admitted
    # Breach the limit -> refused.
    assert not gov.admit(Order("o2", "s", "b", 60, 80)).admitted
    # Kill switch -> refused.
    gov.engage_kill_switch("s")
    assert not gov.admit(Order("o3", "s", "b", 10, 80)).admitted
    # Only the admitted order is on the record.
    assert [o.order_id for o in gov.order_log] == ["o1"]


def test_pipeline_runs_and_respects_limit():
    result = run_day(seed=5, position_limit=500.0)
    assert result.admitted_orders + result.refused_orders == 8
    assert result.sold_mwh >= 0
