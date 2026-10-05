import numpy as np

from bot.data.synthetic import generate_ohlcv, resample_ohlcv
from bot.indicators import add_entry_indicators, add_trend_indicators
from bot.signals.engine import LONG, SignalEngine
from bot.signals.filters import MarketConditionFilter
from bot.utils import parse_utc, timeframe_to_ms


def _data(cfg, n=9000, seed=11):
    ltf = generate_ohlcv(n, "15m", parse_utc("2025-01-01"), seed=seed)
    htf = resample_ohlcv(ltf, "15m", "4h")
    return htf, ltf


def _htf_index(htf, ltf):
    ltf_close = ltf["timestamp"].to_numpy() + timeframe_to_ms("15m")
    htf_close = htf["timestamp"].to_numpy() + timeframe_to_ms("4h")
    return np.searchsorted(htf_close, ltf_close, side="right") - 1


def _scan(cfg, htf_raw, ltf_raw):
    engine = SignalEngine(cfg.strategy, cfg.risk, "15m", "4h")
    htf = add_trend_indicators(htf_raw, cfg.strategy)
    ltf = add_entry_indicators(ltf_raw, cfg.strategy)
    idx = _htf_index(htf, ltf)
    out = []
    for i in range(len(ltf)):
        if idx[i] < 0:
            continue
        ev = engine.evaluate("BTC/USDT:USDT", htf, ltf, i, int(idx[i]))
        if ev.signal:
            out.append((i, ev.signal))
    return out


def test_signals_respect_confluence_rr_and_stop_side(cfg):
    htf, ltf = _data(cfg)
    sigs = _scan(cfg, htf, ltf)
    assert sigs, "expected at least one signal on synthetic trending data"
    for _, s in sigs:
        assert cfg.strategy.min_confidence <= s.confidence <= 5
        assert len(s.categories) >= 3
        assert "trend" in s.categories
        assert s.reward_risk >= cfg.risk.min_reward_risk - 1e-9
        if s.side == LONG:
            assert s.stop < s.entry < s.take_profit
        else:
            assert s.take_profit < s.entry < s.stop
        assert abs(s.entry - s.stop) >= cfg.risk.stop.min_stop_atr * s.atr - 1e-9
        assert s.reasons and s.reasons[0]


def test_no_lookahead(cfg):
    """A signal at bar i must be identical when computed with only bars <= i."""
    htf_raw, ltf_raw = _data(cfg)
    sigs = _scan(cfg, htf_raw, ltf_raw)
    for i, s in sigs[:8]:
        ltf_cut = ltf_raw.iloc[: i + 1].copy()
        bar_close = int(ltf_cut["timestamp"].iat[-1]) + timeframe_to_ms("15m")
        htf_cut = htf_raw[htf_raw["timestamp"] + timeframe_to_ms("4h") <= bar_close].copy()
        engine = SignalEngine(cfg.strategy, cfg.risk, "15m", "4h")
        ev = engine.evaluate("BTC/USDT:USDT", add_trend_indicators(htf_cut, cfg.strategy),
                             add_entry_indicators(ltf_cut, cfg.strategy))
        assert ev.signal is not None, f"signal at {i} disappeared without future data"
        assert ev.signal.side == s.side
        assert abs(ev.signal.stop - s.stop) < 1e-6 * s.entry
        assert abs(ev.signal.take_profit - s.take_profit) < 1e-6 * s.entry


def test_cooldown_blocks_repeat_signal(cfg):
    htf_raw, ltf_raw = _data(cfg)
    htf = add_trend_indicators(htf_raw, cfg.strategy)
    ltf = add_entry_indicators(ltf_raw, cfg.strategy)
    i, s = _scan(cfg, htf_raw, ltf_raw)[0]
    engine = SignalEngine(cfg.strategy, cfg.risk, "15m", "4h")
    hi = int(_htf_index(htf, ltf)[i])
    assert engine.evaluate("X", htf, ltf, i, hi).signal is not None
    again = engine.evaluate("X", htf, ltf, i, hi)
    assert again.signal is None and "cooldown" in again.rejection


def test_news_filter_blocks_window(cfg):
    cfg.filters.news_events = [{"time": "2025-03-12T12:30:00Z", "name": "US CPI"}]
    f = MarketConditionFilter(cfg.filters)
    assert f.news_block(parse_utc("2025-03-12T12:00:00Z")) == "news blackout: US CPI"
    assert f.news_block(parse_utc("2025-03-12T14:00:00Z")) is None
    _, ltf = _data(cfg, n=400)
    ltf = add_entry_indicators(ltf, cfg.strategy)
    ok, why = f.check(ltf, 300, parse_utc("2025-03-12T12:30:00Z"))
    assert not ok and "news" in why
    cfg.filters.allow_during_news = True
    ok, why = MarketConditionFilter(cfg.filters).check(ltf, 300, parse_utc("2025-03-12T12:30:00Z"))
    assert why is None or "news" not in why


def test_volatility_filter_blocks_spike(cfg):
    _, ltf = _data(cfg, n=400)
    ltf = add_entry_indicators(ltf, cfg.strategy)
    i = 350
    ltf.loc[i, "high"] = ltf.loc[i, "close"] * 1.08  # an 8% wick
    f = MarketConditionFilter(cfg.filters)
    reason = f.volatility_block(ltf, i)
    assert reason and "abnormal candle" in reason
