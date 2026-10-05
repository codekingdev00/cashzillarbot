"""Technical indicators in pandas/numpy, using Wilder's definitions.

TA-Lib is not required: it needs a C build that is awkward on Windows, and these
implementations are short and unit tested. Every function is causal: the value
at bar i uses only bars <= i, so the backtester cannot peek at the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.config import StrategyConfig

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def ohlcv_to_frame(rows: list[list[float]]) -> pd.DataFrame:
    """ccxt OHLCV rows -> DataFrame with int64 ms timestamps (bar open time)."""
    df = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    if df.empty:
        return df
    df["timestamp"] = df["timestamp"].astype("int64")
    for col in OHLCV_COLUMNS[1:]:
        df[col] = df[col].astype(float)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (used by RSI, ATR, ADX)."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    avg_gain = rma(delta.clip(lower=0.0), n)
    avg_loss = rma((-delta).clip(lower=0.0), n)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return out


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    return rma(true_range(high, low, close), n)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (ADX, +DI, -DI)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr_s = rma(true_range(high, low, close), n).replace(0.0, np.nan)
    plus_di = 100.0 * rma(plus_dm, n) / tr_s
    minus_di = 100.0 * rma(minus_dm, n) / tr_s
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return rma(dx, n), plus_di, minus_di


def macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def bollinger(close: pd.Series, n: int, k: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def confirmed_pivots(high: pd.Series, low: pd.Series, left: int, right: int) -> tuple[pd.Series, pd.Series]:
    """Swing highs/lows reported on the bar where they become *confirmed*.

    A swing high at bar j needs `right` later bars to be known, so its value is
    placed at bar j + right. This avoids the classic look-ahead bug where a pivot
    is visible at the bar it formed on.
    """
    w = left + right + 1
    cand_h = high.shift(right)
    cand_l = low.shift(right)
    ph = cand_h.where(cand_h == high.rolling(w, min_periods=w).max())
    pl = cand_l.where(cand_l == low.rolling(w, min_periods=w).min())
    return ph, pl


def add_trend_indicators(df: pd.DataFrame, s: StrategyConfig) -> pd.DataFrame:
    """Higher-timeframe columns: ema_fast, ema_slow, adx."""
    out = df.copy()
    out["ema_fast"] = ema(out["close"], s.htf_ema_fast)
    out["ema_slow"] = ema(out["close"], s.htf_ema_slow)
    out["adx"], out["plus_di"], out["minus_di"] = adx(out["high"], out["low"], out["close"], s.adx_period)
    return out


def add_entry_indicators(df: pd.DataFrame, s: StrategyConfig) -> pd.DataFrame:
    """Entry-timeframe columns used by the signal engine and filters."""
    out = df.copy()
    c, h, l = out["close"], out["high"], out["low"]
    out["ema_fast"] = ema(c, s.ema_fast)
    out["ema_slow"] = ema(c, s.ema_slow)
    out["adx"], out["plus_di"], out["minus_di"] = adx(h, l, c, s.adx_period)
    out["rsi"] = rsi(c, s.rsi_period)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(c, s.macd_fast, s.macd_slow, s.macd_signal)
    out["atr"] = atr(h, l, c, s.atr_period)
    out["bb_mid"], out["bb_upper"], out["bb_lower"] = bollinger(c, s.bb_period, s.bb_std)
    out["vol_sma"] = sma(out["volume"], s.volume_sma_period)
    ph, pl = confirmed_pivots(h, l, s.swing_left, s.swing_right)
    out["ph"], out["pl"] = ph, pl
    rsi_at_pivot = out["rsi"].shift(s.swing_right)
    out["ph_rsi"] = rsi_at_pivot.where(ph.notna())
    out["pl_rsi"] = rsi_at_pivot.where(pl.notna())
    return out
