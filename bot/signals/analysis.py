"""On-demand market analysis for the Telegram Analyze option.

Shows the same reasoning the signal engine uses, but for the current moment and
with every check explained (passed or not). It never changes engine state, so
asking for an analysis cannot trigger or block a real signal.

An analysis is a snapshot of indicators, not a prediction. A "signal" verdict
means the rules line up; it does not mean the trade will win.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bot.signals.engine import LONG, SHORT, SignalEngine, _row, HTF_REQUIRED, LTF_ROW
from bot.signals.filters import MarketConditionFilter


@dataclass
class CheckLine:
    name: str
    passed: bool
    detail: str


@dataclass
class Analysis:
    symbol: str
    bar_close: int
    price: float
    trend_tf: str
    entry_tf: str
    bias: str | None
    bias_text: str
    htf: dict[str, float]
    ltf: dict[str, float]
    checks: list[CheckLine] = field(default_factory=list)
    trigger: str | None = None
    support: float | None = None
    resistance: float | None = None
    setup: dict | None = None          # hypothetical entry/stop/target in the bias direction
    verdict: str = "none"              # signal | wait | none
    verdict_text: str = ""
    filter_block: str | None = None

    @property
    def score(self) -> int:
        return sum(c.passed for c in self.checks)


def analyze(engine: SignalEngine, filt: MarketConditionFilter, symbol: str,
            htf: pd.DataFrame, ltf: pd.DataFrame) -> Analysis:
    """Analyse the last closed bars. Frames must carry indicator columns."""
    s = engine.s
    i, hi = len(ltf) - 1, len(htf) - 1
    h = _row(htf, hi, HTF_REQUIRED)
    row = _row(ltf, i, LTF_ROW)
    bar_close = int(row["timestamp"]) + engine.ltf_ms
    a = Analysis(symbol=symbol, bar_close=bar_close, price=row["close"], trend_tf=engine.trend_tf,
                 entry_tf=engine.entry_tf, bias=None, bias_text="", htf=h, ltf=row)

    if any(np.isnan(v) for v in h.values()) or any(np.isnan(row[c]) for c in LTF_ROW if c != "timestamp"):
        a.verdict_text = "Not enough history yet for all indicators."
        return a

    a.bias, a.bias_text = engine._htf_bias(h)
    # with no higher-timeframe trend, still explain the checks for the side the 15m EMAs lean to
    side = a.bias or (LONG if row["ema_fast"] > row["ema_slow"] else SHORT)
    long = side == LONG

    a.checks.append(CheckLine(f"{engine.trend_tf} trend", a.bias is not None, a.bias_text))

    aligned = (row["ema_fast"] > row["ema_slow"]) if long else (row["ema_fast"] < row["ema_slow"])
    a.checks.append(CheckLine(
        f"{engine.entry_tf} trend", bool(aligned and row["adx"] >= s.adx_min),
        f"EMA{s.ema_fast} {'above' if row['ema_fast'] > row['ema_slow'] else 'below'} EMA{s.ema_slow}, "
        f"ADX {row['adx']:.1f} (needs {s.adx_min:g})"))

    lo, hi_z = (s.rsi_long_min, s.rsi_long_max) if long else (s.rsi_short_min, s.rsi_short_max)
    zone = lo <= row["rsi"] <= hi_z
    macd_ok = row["macd_hist"] > 0 if long else row["macd_hist"] < 0
    div = engine._divergence(ltf, i, side)
    mom_detail = (f"RSI {row['rsi']:.1f} (needs {lo:g}-{hi_z:g}), MACD histogram {row['macd_hist']:+.4g}"
                  + (f"; {div}" if div else ""))
    a.checks.append(CheckLine("Momentum", bool((zone and macd_ok) or div), mom_detail))

    if long:
        vol_ok = row["bb_mid"] < row["close"] < row["bb_upper"]
        pos = "above upper band" if row["close"] >= row["bb_upper"] else (
            "upper half" if row["close"] > row["bb_mid"] else "lower half")
    else:
        vol_ok = row["bb_lower"] < row["close"] < row["bb_mid"]
        pos = "below lower band" if row["close"] <= row["bb_lower"] else (
            "lower half" if row["close"] < row["bb_mid"] else "upper half")
    a.checks.append(CheckLine("Bollinger position", bool(vol_ok),
                              f"close in the {pos} (needs the {'upper' if long else 'lower'} half)"))

    ratio = row["volume"] / row["vol_sma"] if row["vol_sma"] > 0 else 0.0
    a.checks.append(CheckLine("Volume", bool(ratio >= s.volume_mult),
                              f"{ratio:.2f}x average (needs {s.volume_mult:g}x)"))

    a.trigger = engine._trigger(ltf, i, side)

    highs, lows = engine._swings(ltf, i, "ph"), engine._swings(ltf, i, "pl")
    above, below = highs[highs > row["close"]], lows[lows < row["close"]]
    a.resistance = float(above.min()) if len(above) else None
    a.support = float(below.max()) if len(below) else None

    if a.bias is not None:
        entry, atr_v = row["close"], row["atr"]
        stop, stop_basis = engine._stop(ltf, i, side, entry, atr_v)
        if stop is not None:
            risk = abs(entry - stop)
            tp, target_basis = engine._target(ltf, i, side, entry, risk, atr_v)
            rr = (tp - entry) * (1 if long else -1) / risk
            a.setup = {"side": side, "entry": entry, "stop": stop, "take_profit": tp, "rr": rr,
                       "stop_basis": stop_basis, "target_basis": target_basis}

    _, a.filter_block = filt.check(ltf, i, bar_close)

    # verdict: run the real rules on a fresh engine (no cooldown, no shared state)
    fresh = SignalEngine(engine.s, engine.r, engine.entry_tf, engine.trend_tf)
    ev = fresh.evaluate(symbol, htf, ltf)
    if ev.signal is not None and a.filter_block is None:
        a.verdict = "signal"
        a.verdict_text = f"{ev.signal.side.upper()} setup: all entry rules are met on the last closed candle."
    elif ev.signal is not None:
        a.verdict = "wait"
        a.verdict_text = f"Rules are met, but the market filter blocks entries: {a.filter_block}."
    elif a.bias is None:
        a.verdict = "none"
        a.verdict_text = f"No trade: {ev.rejection}."
    else:
        a.verdict = "wait"
        a.verdict_text = f"{side.capitalize()} bias, but no entry yet: {ev.rejection}."
    return a
