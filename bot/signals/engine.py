"""Signal engine: multi-timeframe confluence.

A signal requires ALL of:
  1. Trend-timeframe (e.g. 4h) bias: fast EMA above/below slow EMA, price on the
     same side of the slow EMA, and ADX >= adx_min (trending, not ranging).
  2. A fresh entry-timeframe (e.g. 15m) trigger in the same direction within the
     last `trigger_lookback` bars: MACD histogram sign flip or fast/slow EMA cross.
  3. Confluence: a confidence score (1-5) >= min_confidence, drawn from at least
     min_categories distinct categories (trend, momentum, volatility/structure, volume).
  4. A valid stop (ATR and/or swing structure) and reward:risk >= min_reward_risk
     after capping the target at the nearest opposing swing level.

The five scored checks (one point each):
  htf_trend   trend                 always true once step 1 passes
  ltf_trend   trend                 entry-TF EMAs aligned and ADX >= adx_min
  momentum    momentum              RSI in the directional zone with MACD histogram on the
                                    right side of zero, OR a regular RSI divergence
  volatility  volatility/structure  close between the Bollinger mid and outer band in the
                                    trade direction: moving, but not overextended
  volume      volume                bar volume >= volume_mult x its moving average

Confluence improves the odds of a setup; it does not make any trade a winner.
Expect a large share of signals to hit their stop. Backtest results do not
guarantee future results.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bot.config import RiskConfig, StrategyConfig
from bot.utils import timeframe_to_ms

LONG, SHORT = "long", "short"
TREND, MOMENTUM, VOLATILITY, VOLUME = "trend", "momentum", "volatility/structure", "volume"
CHECK_CATEGORY = {
    "htf_trend": TREND,
    "ltf_trend": TREND,
    "momentum": MOMENTUM,
    "volatility": VOLATILITY,
    "volume": VOLUME,
}
CHECK_LABEL = {
    "htf_trend": "higher-timeframe trend",
    "ltf_trend": "entry-timeframe trend",
    "momentum": "momentum (RSI/MACD)",
    "volatility": "Bollinger position",
    "volume": "volume",
}
LTF_REQUIRED = ["ema_fast", "ema_slow", "adx", "rsi", "macd_hist", "atr", "bb_mid", "bb_upper", "bb_lower", "vol_sma"]
HTF_REQUIRED = ["ema_fast", "ema_slow", "adx", "close"]
LTF_ROW = LTF_REQUIRED + ["timestamp", "close", "volume"]


def _row(df: pd.DataFrame, i: int, cols: list[str]) -> dict[str, float]:
    """Scalar values of bar i. Much faster than df.iloc[i] on mixed dtypes."""
    return {c: float(df[c].to_numpy()[i]) for c in cols}


@dataclass
class Signal:
    symbol: str
    side: str
    timestamp: int            # close time (ms) of the bar that produced the signal
    entry: float              # reference price: that bar's close
    stop: float
    take_profit: float
    partial_tp: float | None
    atr: float
    reward_risk: float
    confidence: int           # 1-5
    categories: list[str]
    reasons: list[str]
    missing: list[str]
    stop_basis: str
    target_basis: str

    @property
    def sign(self) -> int:
        return 1 if self.side == LONG else -1


@dataclass
class Evaluation:
    signal: Signal | None = None
    side: str | None = None
    rejection: str | None = None
    checks: dict[str, bool] = field(default_factory=dict)


class SignalEngine:
    def __init__(self, strategy: StrategyConfig, risk: RiskConfig, entry_timeframe: str, trend_timeframe: str):
        self.s = strategy
        self.r = risk
        self.entry_tf = entry_timeframe
        self.trend_tf = trend_timeframe
        self.ltf_ms = timeframe_to_ms(entry_timeframe)
        self._last_signal: dict[str, int] = {}

    def update_risk_config(self, risk: RiskConfig) -> None:
        self.r = risk

    # ------------------------------------------------------------------ main
    def evaluate(self, symbol: str, htf: pd.DataFrame, ltf: pd.DataFrame,
                 i: int | None = None, htf_i: int | None = None) -> Evaluation:
        """Evaluate the closed entry bar `i` (default: last) against the last
        CLOSED trend bar `htf_i` (default: last). Frames must already carry
        indicator columns (see bot.indicators)."""
        i = len(ltf) - 1 if i is None else i
        htf_i = len(htf) - 1 if htf_i is None else htf_i
        if i < max(2, self.s.trigger_lookback) or htf_i < 0:
            return Evaluation(rejection="not enough data")

        h = _row(htf, htf_i, HTF_REQUIRED)
        if any(np.isnan(v) for v in h.values()):
            return Evaluation(rejection="trend indicators warming up")
        side, bias_reason = self._htf_bias(h)
        if side is None:
            return Evaluation(rejection=bias_reason)

        row = _row(ltf, i, LTF_ROW)
        if any(np.isnan(row[c]) for c in LTF_REQUIRED):
            return Evaluation(side=side, rejection="entry indicators warming up")
        bar_close = int(row["timestamp"]) + self.ltf_ms
        last = self._last_signal.get(symbol)
        if last is not None and bar_close - last < self.s.signal_cooldown_bars * self.ltf_ms:
            return Evaluation(side=side, rejection="cooldown after previous signal")

        trigger = self._trigger(ltf, i, side)
        if trigger is None:
            return Evaluation(side=side, rejection="no fresh entry trigger")

        checks, reasons = self._score(ltf, i, row, side, bias_reason)
        reasons.insert(1, f"Trigger ({self.entry_tf}): {trigger}")
        score = sum(checks.values())
        categories = sorted({CHECK_CATEGORY[k] for k, ok in checks.items() if ok})
        if score < self.s.min_confidence or len(categories) < self.s.min_categories:
            return Evaluation(side=side, checks=checks,
                              rejection=f"confluence too low ({score}/5 checks, {len(categories)} categories)")

        entry, atr_v = float(row["close"]), float(row["atr"])
        stop, stop_basis = self._stop(ltf, i, side, entry, atr_v)
        if stop is None:
            return Evaluation(side=side, checks=checks, rejection=stop_basis)
        risk = abs(entry - stop)
        tp, target_basis = self._target(ltf, i, side, entry, risk, atr_v)
        rr = (tp - entry) * (1 if side == LONG else -1) / risk
        if rr < self.r.min_reward_risk:
            return Evaluation(side=side, checks=checks,
                              rejection=f"R:R {rr:.2f} below minimum {self.r.min_reward_risk} ({target_basis})")

        sign = 1 if side == LONG else -1
        partial_r = self.r.take_profit.partial_at_r
        partial_tp = entry + sign * partial_r * risk if partial_r < rr else None

        self._last_signal[symbol] = bar_close
        sig = Signal(
            symbol=symbol, side=side, timestamp=bar_close, entry=entry, stop=stop, take_profit=tp,
            partial_tp=partial_tp, atr=atr_v, reward_risk=rr, confidence=int(score),
            categories=categories, reasons=reasons,
            missing=[CHECK_LABEL[k] for k, ok in checks.items() if not ok],
            stop_basis=stop_basis, target_basis=target_basis,
        )
        return Evaluation(signal=sig, side=side, checks=checks)

    # ------------------------------------------------------------- components
    def _htf_bias(self, h: dict[str, float]) -> tuple[str | None, str]:
        tf, s = self.trend_tf, self.s
        if h["adx"] < s.adx_min:
            return None, f"{tf} ADX {h['adx']:.1f} < {s.adx_min}: no trend"
        if h["ema_fast"] > h["ema_slow"] and h["close"] > h["ema_slow"]:
            return LONG, (f"{tf} uptrend: EMA{s.htf_ema_fast} > EMA{s.htf_ema_slow}, "
                          f"price above EMA{s.htf_ema_slow}, ADX {h['adx']:.1f}")
        if h["ema_fast"] < h["ema_slow"] and h["close"] < h["ema_slow"]:
            return SHORT, (f"{tf} downtrend: EMA{s.htf_ema_fast} < EMA{s.htf_ema_slow}, "
                           f"price below EMA{s.htf_ema_slow}, ADX {h['adx']:.1f}")
        return None, f"{tf} EMAs and price disagree: no clear trend"

    def _trigger(self, ltf: pd.DataFrame, i: int, side: str) -> str | None:
        hist = ltf["macd_hist"].to_numpy()
        ef = ltf["ema_fast"].to_numpy()
        es = ltf["ema_slow"].to_numpy()
        lo = max(1, i - self.s.trigger_lookback + 1)
        for k in range(i, lo - 1, -1):
            ago = i - k
            if side == LONG:
                if hist[k - 1] <= 0 < hist[k]:
                    return f"MACD histogram turned positive ({ago} bar(s) ago)"
                if ef[k - 1] <= es[k - 1] and ef[k] > es[k]:
                    return f"EMA{self.s.ema_fast} crossed above EMA{self.s.ema_slow} ({ago} bar(s) ago)"
            else:
                if hist[k - 1] >= 0 > hist[k]:
                    return f"MACD histogram turned negative ({ago} bar(s) ago)"
                if ef[k - 1] >= es[k - 1] and ef[k] < es[k]:
                    return f"EMA{self.s.ema_fast} crossed below EMA{self.s.ema_slow} ({ago} bar(s) ago)"
        return None

    def _score(self, ltf: pd.DataFrame, i: int, row: dict[str, float], side: str,
               bias_reason: str) -> tuple[dict[str, bool], list[str]]:
        s, tf = self.s, self.entry_tf
        long = side == LONG
        checks = {"htf_trend": True}
        reasons = [bias_reason]

        aligned = (row["ema_fast"] > row["ema_slow"]) if long else (row["ema_fast"] < row["ema_slow"])
        checks["ltf_trend"] = bool(aligned and row["adx"] >= s.adx_min)
        if checks["ltf_trend"]:
            reasons.append(f"{tf} trend aligned: EMA{s.ema_fast} {'>' if long else '<'} EMA{s.ema_slow}, "
                           f"ADX {row['adx']:.1f}")

        if long:
            zone = s.rsi_long_min <= row["rsi"] <= s.rsi_long_max
            macd_ok = row["macd_hist"] > 0
        else:
            zone = s.rsi_short_min <= row["rsi"] <= s.rsi_short_max
            macd_ok = row["macd_hist"] < 0
        div = self._divergence(ltf, i, side)
        checks["momentum"] = bool((zone and macd_ok) or div)
        if zone and macd_ok:
            reasons.append(f"Momentum: RSI {row['rsi']:.1f} in {side} zone, MACD histogram "
                           f"{'positive' if long else 'negative'}")
        if div:
            reasons.append(f"Momentum: {div}")

        if long:
            checks["volatility"] = bool(row["bb_mid"] < row["close"] < row["bb_upper"])
        else:
            checks["volatility"] = bool(row["bb_lower"] < row["close"] < row["bb_mid"])
        if checks["volatility"]:
            reasons.append(f"Volatility: close inside the {'upper' if long else 'lower'} Bollinger half, "
                           f"not overextended (ATR {row['atr']:.6g})")

        ratio = row["volume"] / row["vol_sma"] if row["vol_sma"] > 0 else 0.0
        checks["volume"] = bool(ratio >= s.volume_mult)
        if checks["volume"]:
            reasons.append(f"Volume: {ratio:.2f}x the {s.volume_sma_period}-bar average")
        return checks, reasons

    def _divergence(self, ltf: pd.DataFrame, i: int, side: str) -> str | None:
        lo = max(0, i - self.s.divergence_lookback + 1)
        price_col, osc_col = ("pl", "pl_rsi") if side == LONG else ("ph", "ph_rsi")
        price = ltf[price_col].to_numpy()[lo:i + 1]
        osc = ltf[osc_col].to_numpy()[lo:i + 1]
        idx = np.flatnonzero(~np.isnan(price))
        if len(idx) < 2:
            return None
        a, b = idx[-2], idx[-1]
        if side == LONG and price[b] < price[a] and osc[b] > osc[a]:
            return "bullish RSI divergence (lower low in price, higher low in RSI)"
        if side == SHORT and price[b] > price[a] and osc[b] < osc[a]:
            return "bearish RSI divergence (higher high in price, lower high in RSI)"
        return None

    def _swings(self, ltf: pd.DataFrame, i: int, col: str) -> np.ndarray:
        lo = max(0, i - self.s.structure_lookback + 1)
        arr = ltf[col].to_numpy()[lo:i + 1]
        return arr[~np.isnan(arr)]

    def _stop(self, ltf: pd.DataFrame, i: int, side: str, entry: float, atr_v: float) -> tuple[float | None, str]:
        cfg = self.r.stop
        sign = 1 if side == LONG else -1
        atr_stop = entry - sign * cfg.atr_mult * atr_v
        min_d, max_d = cfg.min_stop_atr * atr_v, cfg.max_stop_atr * atr_v

        struct_stop = None
        swings = self._swings(ltf, i, "pl" if side == LONG else "ph")
        for level in swings[::-1]:  # most recent first
            if (entry - level) * sign > 0:
                candidate = level - sign * cfg.structure_buffer_atr * atr_v
                if min_d <= abs(entry - candidate) <= max_d:
                    struct_stop = float(candidate)
                break

        if cfg.method == "atr":
            stop, basis = atr_stop, f"{cfg.atr_mult} x ATR"
        elif cfg.method == "structure":
            if struct_stop is None:
                return None, "no usable swing level for a structure stop"
            stop, basis = struct_stop, "beyond last swing " + ("low" if side == LONG else "high")
        else:  # hybrid: the wider of the two, i.e. the more conservative stop
            stop, basis = atr_stop, f"{cfg.atr_mult} x ATR"
            if struct_stop is not None and (atr_stop - struct_stop) * sign > 0:
                stop, basis = struct_stop, "beyond last swing " + ("low" if side == LONG else "high")

        if abs(entry - stop) < min_d:
            stop, basis = entry - sign * min_d, f"minimum {cfg.min_stop_atr} x ATR"
        if (entry - stop) * sign <= 0:
            return None, "invalid stop"
        return float(stop), basis

    def _target(self, ltf: pd.DataFrame, i: int, side: str, entry: float, risk: float,
                atr_v: float) -> tuple[float, str]:
        sign = 1 if side == LONG else -1
        target = entry + sign * self.r.take_profit.target_r * risk
        swings = self._swings(ltf, i, "ph" if side == LONG else "pl")
        beyond = swings[(swings - entry) * sign > 0]
        if len(beyond):
            nearest = float(beyond.min() if side == LONG else beyond.max())
            capped = nearest - sign * self.r.stop.structure_buffer_atr * atr_v
            if (target - capped) * sign > 0:
                kind = "resistance" if side == LONG else "support"
                return capped, f"capped below swing {kind} at {nearest:.6g}"
        return float(target), f"{self.r.take_profit.target_r}R target"
