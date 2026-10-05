import numpy as np
import pandas as pd

from bot.data.synthetic import generate_ohlcv
from bot.indicators import add_entry_indicators, adx, atr, confirmed_pivots, ema, rsi


def _df(n=600):
    return generate_ohlcv(n, "15m", 1_700_000_000_000, seed=3)


def test_rsi_bounds_and_extremes():
    df = _df()
    r = rsi(df["close"], 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    rising = pd.Series(np.arange(1.0, 60.0))
    assert rsi(rising, 14).dropna().eq(100.0).all()
    flat = pd.Series(np.ones(60))
    assert rsi(flat, 14).dropna().eq(50.0).all()


def test_atr_positive_and_adx_bounded():
    df = _df()
    a = atr(df["high"], df["low"], df["close"], 14).dropna()
    assert (a > 0).all()
    x, pdi, mdi = adx(df["high"], df["low"], df["close"], 14)
    x = x.dropna()
    assert ((x >= 0) & (x <= 100)).all()


def test_ema_matches_definition():
    s = pd.Series(np.arange(1.0, 30.0))
    e = ema(s, 5)
    alpha = 2 / 6
    manual = s.iloc[0]
    for v in s.iloc[1:]:
        manual = alpha * v + (1 - alpha) * manual
    assert abs(e.iloc[-1] - manual) < 1e-9


def test_pivots_are_confirmed_with_lag_and_causal():
    high = pd.Series([1, 2, 5, 2, 1, 1, 1], dtype=float)
    low = high - 0.5
    ph, _ = confirmed_pivots(high, low, 2, 2)
    # the swing high at index 2 is only known at index 4
    assert np.isnan(ph.iloc[2]) and ph.iloc[4] == 5


def test_entry_indicators_are_causal(cfg):
    """Values at bar i must not change when future bars are removed."""
    s = cfg.strategy
    df = _df(800)
    full = add_entry_indicators(df, s)
    cut = add_entry_indicators(df.iloc[:500].copy(), s)
    cols = ["ema_fast", "rsi", "macd_hist", "atr", "adx", "bb_upper", "vol_sma", "ph", "pl"]
    pd.testing.assert_frame_equal(full[cols].iloc[:500], cut[cols], check_exact=False, rtol=1e-9)
