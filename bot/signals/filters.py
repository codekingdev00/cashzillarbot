"""Market-condition filters: news blackout windows and volatility spikes.

There is no reliable free real-time economic calendar API, so news events come
from the config file (filters.news_events). Keep that list current (CPI, FOMC,
NFP, major exchange events). The volatility filter is data-driven and catches
unscheduled shocks the calendar misses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot.config import FilterConfig
from bot.utils import parse_utc


@dataclass
class NewsWindow:
    name: str
    start_ms: int
    end_ms: int


class MarketConditionFilter:
    def __init__(self, cfg: FilterConfig):
        self.cfg = cfg
        self.windows: list[NewsWindow] = []
        for ev in cfg.news_events:
            t = parse_utc(ev["time"])
            before = float(ev.get("before_min", cfg.default_news_before_min))
            after = float(ev.get("after_min", cfg.default_news_after_min))
            self.windows.append(NewsWindow(str(ev["name"]), int(t - before * 60_000), int(t + after * 60_000)))

    def news_block(self, ts_ms: int) -> str | None:
        for w in self.windows:
            if w.start_ms <= ts_ms <= w.end_ms:
                return f"news blackout: {w.name}"
        return None

    def volatility_block(self, ltf: pd.DataFrame, i: int) -> str | None:
        if i < 1:
            return None
        atr = ltf["atr"].to_numpy()
        close = ltf["close"].to_numpy()
        atr_pct = atr / close
        lo = max(0, i - self.cfg.atr_spike_lookback)
        hist = atr_pct[lo:i]
        hist = hist[~np.isnan(hist)]
        if len(hist) >= self.cfg.atr_spike_lookback // 2 and not np.isnan(atr_pct[i]):
            med = float(np.median(hist))
            if med > 0 and atr_pct[i] > self.cfg.atr_spike_mult * med:
                return f"volatility spike: ATR {atr_pct[i] * 100:.2f}% vs median {med * 100:.2f}%"
        prev_atr = atr[i - 1]
        rng = ltf["high"].iat[i] - ltf["low"].iat[i]
        if prev_atr > 0 and rng > self.cfg.candle_range_atr_max * prev_atr:
            return f"abnormal candle: range {rng / prev_atr:.1f}x ATR"
        return None

    def check(self, ltf: pd.DataFrame, i: int, ts_ms: int) -> tuple[bool, str | None]:
        if not self.cfg.allow_during_news:
            reason = self.news_block(ts_ms)
            if reason:
                return False, reason
        if not self.cfg.allow_extreme_volatility:
            reason = self.volatility_block(ltf, i)
            if reason:
                return False, reason
        return True, None
