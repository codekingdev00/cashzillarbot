import pytest

from bot.execution.live import LiveExecutor, paper_gate_status
from bot.execution.paper import PaperExecutor
from bot.risk.trade_manager import TradeManager
from bot.signals.engine import Signal
from bot.storage import Storage


def make_signal(side="long", entry=100.0, stop=98.0, tp=105.0, partial=102.0):
    return Signal(symbol="BTC/USDT:USDT", side=side, timestamp=0, entry=entry, stop=stop, take_profit=tp,
                  partial_tp=partial, atr=1.0, reward_risk=abs(tp - entry) / abs(entry - stop), confidence=4,
                  categories=["trend", "momentum", "volume"], reasons=["test"], missing=[],
                  stop_basis="atr", target_basis="2.5R")


def executor(cfg, slippage=0.0, fee=0.0):
    cfg.execution.slippage_pct = slippage
    cfg.execution.fee_pct = fee
    return PaperExecutor(cfg.execution, TradeManager(cfg.risk.take_profit), 10_000)


def test_stop_loss_closes_at_minus_one_r(cfg):
    ex = executor(cfg)
    pos, _ = ex.open_position(make_signal(), 10, 100.0, 0, risk_amount=20.0)
    events = ex.on_bar("BTC/USDT:USDT", 99.5, 99.8, 97.5, 98.5, 1.0, 1)
    closed = [e for e in events if e.kind == "closed"]
    assert closed and closed[0].trade.exit_price == 98.0
    assert closed[0].trade.r_multiple == pytest.approx(-1.0)
    assert ex.balance == pytest.approx(10_000 - 20)


def test_gap_through_stop_fills_at_open(cfg):
    ex = executor(cfg)
    ex.open_position(make_signal(), 10, 100.0, 0, risk_amount=20.0)
    events = ex.on_bar("BTC/USDT:USDT", 97.0, 97.5, 96.0, 97.2, 1.0, 1)
    assert events[-1].trade.exit_price == 97.0  # worse than the stop: losses can exceed 1R


def test_same_bar_stop_and_target_assumes_stop(cfg):
    ex = executor(cfg)
    ex.open_position(make_signal(), 10, 100.0, 0, risk_amount=20.0)
    events = ex.on_bar("BTC/USDT:USDT", 100.0, 106.0, 97.0, 101.0, 1.0, 1)
    assert events[-1].trade.exit_reason == "stop-loss"


def test_partial_breakeven_then_trailing(cfg):
    ex = executor(cfg)
    pos, _ = ex.open_position(make_signal(), 10, 100.0, 0, risk_amount=20.0)
    events = ex.on_bar("BTC/USDT:USDT", 100.0, 102.5, 100.0, 102.2, 1.0, 1)
    kinds = [e.kind for e in events]
    assert "partial" in kinds and "breakeven" in kinds
    assert pos.qty == pytest.approx(10 * (1 - cfg.risk.take_profit.partial_pct / 100))
    assert pos.stop >= 100.0
    # price advances: the trailing stop ratchets up and never down
    ex.on_bar("BTC/USDT:USDT", 102.2, 104.0, 102.0, 103.8, 1.0, 2)
    s1 = pos.stop
    assert s1 == pytest.approx(104.0 - cfg.risk.take_profit.trail_atr_mult * 1.0)
    ex.on_bar("BTC/USDT:USDT", 103.8, 103.9, 102.5, 102.6, 1.0, 3)
    assert pos.stop == s1
    events = ex.on_bar("BTC/USDT:USDT", 102.6, 102.7, 101.5, 101.8, 1.0, 4)
    assert events[-1].kind == "closed" and events[-1].trade.exit_reason == "trailing stop"
    assert events[-1].trade.net_pnl > 0


def test_short_take_profit(cfg):
    ex = executor(cfg)
    sig = make_signal(side="short", entry=100.0, stop=102.0, tp=95.0, partial=None)
    ex.open_position(sig, 10, 100.0, 0, risk_amount=20.0)
    events = ex.on_bar("BTC/USDT:USDT", 99.0, 99.5, 94.5, 95.5, 1.0, 1)
    assert events[-1].trade.exit_reason == "take-profit"
    assert events[-1].trade.r_multiple == pytest.approx(2.5)


def test_fees_and_slippage_reduce_pnl(cfg):
    ex = executor(cfg, slippage=0.02, fee=0.05)
    pos, fill = ex.open_position(make_signal(), 10, 100.0, 0, risk_amount=20.0)
    assert fill > 100.0
    ev = ex.close_position(pos.id, 100.0, "manual", 1)
    assert ev.trade.net_pnl < 0


def test_refuses_position_without_valid_stop(cfg):
    ex = executor(cfg)
    with pytest.raises(ValueError):
        ex.open_position(make_signal(stop=101.0), 10, 100.0, 0, risk_amount=20.0)


def test_state_roundtrip(cfg):
    ex = executor(cfg)
    ex.open_position(make_signal(), 10, 100.0, 0, risk_amount=20.0)
    ex2 = PaperExecutor(cfg.execution, TradeManager(cfg.risk.take_profit), 10_000, ex.to_state())
    assert list(ex2.positions) == list(ex.positions) and ex2.balance == ex.balance


def test_live_executor_disabled_and_paper_gate(cfg, tmp_path):
    with pytest.raises(NotImplementedError):
        LiveExecutor()
    st = Storage(str(tmp_path / "t.db"))
    ok, msg = paper_gate_status(st, cfg.paper_gate)
    assert not ok and "NOT met" in msg
    st.close()
