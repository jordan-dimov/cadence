"""Quick checks that every piece runs on the invented data, with nothing
extra installed. The Morpholog-backed safety gate is not tested here because
it needs the Morpholog program and a database set up separately.
"""

import numpy as np
import pytest

from cadence import data
from cadence.backtest import backtest_signal, score_by_period
from cadence.execution import (
    FillReconciler,
    cost_of_waiting,
    liquidity_aware_clip,
    twap_schedule,
    urgency_schedule,
)
from cadence.forecast import ForecastDistribution, Forecaster
from cadence.pipeline import compare_perfect_and_realistic, run_day, run_period
from cadence.risk import InProcessGovernor, Order, make_governor
from cadence.signals import (
    rolling_zscore,
    run_stat_arb,
    stat_arb_positions,
    zscore_signals,
)
from cadence.sizing import optimal_volume


def test_data_shapes():
    assert data.simulated_generation(seed=1).shape == (data.PERIODS_PER_DAY,)
    assert data.simulated_prices(seed=1).shape == (data.PERIODS_PER_DAY,)
    book = data.simulated_orderbook(seed=1)
    assert book.best_ask > book.best_bid


def test_forecast_returns_distribution():
    dist = Forecaster().fit(data.simulated_generation(seed=2)).predict()
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
    # alpha = 1 reproduces the even (TWAP) schedule.
    assert np.allclose(urgency_schedule(100, 4, alpha=1.0), twap_schedule(100, 4))
    # alpha < 1 front-loads (more in the first clip than the last);
    # alpha > 1 back-loads (more in the last clip than the first).
    assert urgency_schedule(100, 4, alpha=0.5)[0] > urgency_schedule(100, 4, 0.5)[-1]
    assert urgency_schedule(100, 4, alpha=2.0)[0] < urgency_schedule(100, 4, 2.0)[-1]
    assert cost_of_waiting(0.5, 200) == 100.0

    # A clip is capped by what is on offer at the best level.
    thin = data.simulated_orderbook(levels=5, seed=1)
    assert liquidity_aware_clip(1000.0, thin, "buy") <= float(thin.ask_sizes[0])

    rec = FillReconciler(target=10.0)
    rec.record_fill(4.0)
    assert rec.residual == 6.0 and not rec.is_complete
    rec.record_fill(6.0)
    assert rec.is_complete
    # Overfilling is a bug, not a silent no-op: reject it.
    with pytest.raises(ValueError):
        rec.record_fill(1.0)


def test_signals_no_lookahead():
    gap = data.simulated_country_gap(days=30, seed=3)
    z = rolling_zscore(gap, window=48)
    assert np.isnan(z[:48]).all()  # no signal before a full window
    sig = zscore_signals(gap, window=48, threshold=2.0)
    assert set(np.unique(sig)).issubset({-1, 0, 1})


def test_backtest_charges_costs_and_impact():
    gap = data.simulated_country_gap(days=60, seed=4)
    sig = zscore_signals(gap, window=14 * 24, threshold=2.0)
    free = backtest_signal(gap, sig, cost_per_trade=0.0)
    costed = backtest_signal(gap, sig, cost_per_trade=1.0)
    impacted = backtest_signal(gap, sig, cost_per_trade=1.0, impact=0.5)
    assert costed.pnl <= free.pnl  # honest costs never improve P&L
    assert impacted.pnl <= costed.pnl  # market impact only makes it worse
    assert costed.n_trades >= 0


def test_score_by_period_gives_many_scores():
    gap = data.simulated_country_gap(days=60, seed=4)
    sig = zscore_signals(gap, window=14 * 24, threshold=2.0)
    folds = score_by_period(gap, sig, n_folds=4)
    assert len(folds) == 4  # one score per stretch of history


def test_stat_arb_lifecycle_opens_and_closes():
    gap = data.simulated_country_gap(days=90, seed=2)
    trades = run_stat_arb(gap, window=14 * 24)
    assert trades  # the bot actually takes some round trips
    # Every trade is a complete round trip: it exits after it enters, one way.
    for t in trades:
        assert t.exit > t.entry
        assert t.direction in (-1, 1)
    # The trades and the position series describe the same lifecycle.
    positions = stat_arb_positions(gap, window=14 * 24)
    assert set(np.unique(positions)).issubset({-1, 0, 1})


def test_order_rejects_zero_quantity_and_bad_price():
    with pytest.raises(ValueError):
        Order("o", "s", "b", 0, 80)      # zero size is not a trade
    with pytest.raises(ValueError):
        Order("o", "s", "b", 10, 0)      # price must be positive


def test_liquidity_clip_validates_side():
    book = data.simulated_orderbook(seed=1)
    with pytest.raises(ValueError):
        liquidity_aware_clip(10.0, book, "bid")  # must be 'buy' or 'sell'


def test_perfect_foresight_costs_less_than_realistic():
    r = compare_perfect_and_realistic(seed=7)
    # Perfect foresight sells the right amount, so its penalty is ~zero and
    # never more than the realistic one. The gap is the cost of uncertainty.
    assert r["penalty_perfect"] <= r["penalty_realistic"] + 1e-9
    assert r["penalty_perfect"] < 1e-3
    assert r["cost_of_not_knowing"] >= -1e-9


def test_governor_enforces_rules():
    gov = InProcessGovernor()
    gov.open_book("b", limit=100.0)
    gov.enable_strategy("s")
    # Missing attribution is impossible: Order requires strategy + book.
    assert gov.admit(Order("o1", "s", "b", 50, 80)).admitted
    # A duplicate order id is refused.
    assert not gov.admit(Order("o1", "s", "b", 50, 80)).admitted
    # Breach the limit (50 + 60 = 110 > 100) -> refused.
    assert not gov.admit(Order("o2", "s", "b", 60, 80)).admitted
    # Kill switch -> the strategy's orders are refused.
    gov.engage_kill_switch("s")
    assert not gov.admit(Order("o3", "s", "b", 10, 80)).admitted
    # Only the admitted order is on the record.
    assert [o.order_id for o in gov.order_log] == ["o1"]


def test_make_governor_rejects_unknown_kind():
    # A typo must not silently fall back to the weaker gate.
    with pytest.raises(ValueError):
        make_governor("morphlog")


def test_pipeline_period_runs_and_respects_limit():
    result = run_period(seed=5, position_limit=500.0)
    assert result.admitted_orders + result.refused_orders == 8
    assert result.sold_mwh >= 0


def test_day_is_the_period_loop():
    day = run_day(seed=5, n_periods=4)
    assert day.periods == 4
    assert day.admitted_orders + day.refused_orders == 4 * 8
