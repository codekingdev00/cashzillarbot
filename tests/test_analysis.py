from bot.__main__ import _load_data
from bot.indicators import add_entry_indicators, add_trend_indicators
from bot.messages import analysis as format_analysis
from bot.signals.analysis import analyze
from bot.signals.engine import SignalEngine
from bot.signals.filters import MarketConditionFilter


def _frames(cfg, end="2025-02-01"):
    htf, ltf = _load_data(cfg, "synthetic", "2025-01-01", end, seed=5)["BTC/USDT:USDT"]
    return add_trend_indicators(htf, cfg.strategy), add_entry_indicators(ltf, cfg.strategy)


def test_analysis_explains_all_checks_and_formats(cfg):
    htf, ltf = _frames(cfg)
    engine = SignalEngine(cfg.strategy, cfg.risk, "15m", "4h")
    a = analyze(engine, MarketConditionFilter(cfg.filters), "BTC/USDT:USDT", htf, ltf)
    assert len(a.checks) == 5 and 0 <= a.score <= 5
    assert a.verdict in {"signal", "wait", "none"} and a.verdict_text
    text = format_analysis(a, 1.23)
    for needle in ("analysis", "Checks", "Volume", "Momentum", "Market filter", "24h +1.23%"):
        assert needle in text


def test_analysis_does_not_change_engine_state(cfg):
    htf, ltf = _frames(cfg)
    engine = SignalEngine(cfg.strategy, cfg.risk, "15m", "4h")
    filt = MarketConditionFilter(cfg.filters)
    # analyse every bar of a month: the live engine must never record a cooldown
    for end in range(300, len(ltf), 97):
        analyze(engine, filt, "BTC/USDT:USDT", htf[htf["timestamp"] < ltf["timestamp"].iat[end]], ltf.iloc[:end])
    assert engine._last_signal == {}


def test_analysis_verdict_matches_engine(cfg):
    htf, ltf = _frames(cfg, end="2025-03-01")
    filt = MarketConditionFilter(cfg.filters)
    seen_signal = False
    for end in range(400, len(ltf), 7):
        l = ltf.iloc[:end]
        h = htf[htf["timestamp"] + 4 * 3600_000 <= l["timestamp"].iat[-1] + 900_000]
        a = analyze(SignalEngine(cfg.strategy, cfg.risk, "15m", "4h"), filt, "X", h, l)
        ev = SignalEngine(cfg.strategy, cfg.risk, "15m", "4h").evaluate("X", h, l)
        assert (a.verdict == "signal") == (ev.signal is not None and a.filter_block is None)
        seen_signal |= a.verdict == "signal"
    assert seen_signal
