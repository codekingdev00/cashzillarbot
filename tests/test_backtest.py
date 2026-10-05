import asyncio

from bot.__main__ import _load_data
from bot.backtest.engine import Backtester
from bot.messages import signal_alert, status
from bot.risk.manager import SizingResult


def test_backtest_runs_and_respects_limits(cfg):
    data = _load_data(cfg, "synthetic", "2025-01-01", "2025-03-01", seed=5)
    res = Backtester(cfg).run(data)
    m = res.metrics
    for key in ("trades", "win_rate_pct", "avg_r_multiple", "avg_planned_rr", "max_drawdown_pct",
                "sharpe_daily_annualized", "profit_factor"):
        assert key in m
    assert res.signal_counts["entered"] == len(res.trades)
    assert res.max_leverage_used <= cfg.risk.max_leverage + 1e-6
    for t in res.trades:
        # every trade had a stop: a stop-out costs ~1R (gaps can make it slightly worse)
        assert t.r_multiple > -1.5, t
        assert t.planned_rr >= cfg.risk.min_reward_risk - 1e-6
    assert len(res.equity) > 0


def test_daily_loss_limit_respected_in_backtest(cfg):
    cfg.risk.max_daily_loss_pct = 1.0
    cfg.risk.risk_per_trade_pct = 1.0
    data = _load_data(cfg, "synthetic", "2025-01-01", "2025-03-01", seed=9)
    res = Backtester(cfg).run(data)
    daily = res.equity.resample("1D").last()
    # a day can end worse than -1% only through trades already open when the halt hit
    worst = (daily.pct_change().dropna().min()) * 100 if len(daily) > 1 else 0
    assert worst > -(cfg.risk.max_daily_loss_pct + cfg.risk.max_open_positions * cfg.risk.risk_per_trade_pct + 0.5)


def test_alert_message_contains_required_fields(cfg):
    data = _load_data(cfg, "synthetic", "2025-01-01", "2025-03-01", seed=5)
    from bot.indicators import add_entry_indicators, add_trend_indicators
    from bot.signals.engine import SignalEngine
    import numpy as np

    htf, ltf = data["BTC/USDT:USDT"]
    htf, ltf = add_trend_indicators(htf, cfg.strategy), add_entry_indicators(ltf, cfg.strategy)
    idx = np.searchsorted(htf["timestamp"].to_numpy() + 4 * 3600_000, ltf["timestamp"].to_numpy() + 900_000,
                          side="right") - 1
    eng = SignalEngine(cfg.strategy, cfg.risk, "15m", "4h")
    sig = next(ev.signal for i in range(len(ltf)) if idx[i] >= 0
               for ev in [eng.evaluate("BTC/USDT:USDT", htf, ltf, i, int(idx[i]))] if ev.signal)
    text = signal_alert(sig, SizingResult(True, qty=0.1, notional=3000, risk_amount=100, leverage=0.3),
                        10_000, False, 1.0)
    for needle in ("Entry", "Stop-loss", "Take-profit", "Size", "R:R", "Why", "Confidence"):
        assert needle in text
    assert "Past results do not guarantee" in text
    assert "Status" in status(10_000, 10_000, 10_000, 0, 0, 0, [], {}, [], False)
