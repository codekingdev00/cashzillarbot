"""Backtest performance metrics.

These numbers describe the past under simulated fills. They do not predict
future results, and a good-looking backtest is often overfit.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from bot.execution.models import ClosedTrade


def compute_metrics(trades: list[ClosedTrade], equity: pd.Series, initial_equity: float) -> dict:
    n = len(trades)
    pnl = np.array([t.net_pnl for t in trades]) if n else np.array([])
    r = np.array([t.r_multiple for t in trades]) if n else np.array([])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()

    final = float(equity.iloc[-1]) if len(equity) else initial_equity
    max_dd = 0.0
    if len(equity):
        dd = equity / equity.cummax() - 1.0
        max_dd = float(-dd.min() * 100)

    sharpe = 0.0
    if len(equity) > 2:
        daily = equity.resample("1D").last().dropna()
        rets = daily.pct_change().dropna()
        if len(rets) > 1 and rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(365))  # crypto trades every day

    return {
        "trades": n,
        "win_rate_pct": float(len(wins) / n * 100) if n else 0.0,
        "avg_r_multiple": float(r.mean()) if n else 0.0,
        "avg_win_r": float(r[r > 0].mean()) if (r > 0).any() else 0.0,
        "avg_loss_r": float(r[r <= 0].mean()) if (r <= 0).any() else 0.0,
        "avg_planned_rr": float(np.mean([t.planned_rr for t in trades])) if n else 0.0,
        "worst_trade_r": float(r.min()) if n else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0),
        "expectancy": float(pnl.mean()) if n else 0.0,
        "net_pnl": float(pnl.sum()) if n else 0.0,
        "fees": float(sum(t.fees for t in trades)),
        "total_return_pct": (final - initial_equity) / initial_equity * 100,
        "max_drawdown_pct": max_dd,
        "sharpe_daily_annualized": sharpe,
        "max_consecutive_losses": _max_streak(pnl <= 0) if n else 0,
        "final_equity": final,
    }


def _max_streak(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def format_metrics(m: dict) -> str:
    rows = [
        ("Trades", f"{m['trades']}"),
        ("Win rate", f"{m['win_rate_pct']:.1f}%"),
        ("Average R multiple (realized)", f"{m['avg_r_multiple']:+.2f}R"),
        ("Average win / loss", f"{m['avg_win_r']:+.2f}R / {m['avg_loss_r']:+.2f}R"),
        ("Worst trade", f"{m['worst_trade_r']:+.2f}R"),
        ("Average planned R:R", f"1:{m['avg_planned_rr']:.2f}"),
        ("Profit factor", f"{m['profit_factor']:.2f}"),
        ("Expectancy per trade", f"{m['expectancy']:+,.2f}"),
        ("Net P&L (after fees)", f"{m['net_pnl']:+,.2f}"),
        ("Fees paid", f"{m['fees']:,.2f}"),
        ("Total return", f"{m['total_return_pct']:+.2f}%"),
        ("Max drawdown", f"{m['max_drawdown_pct']:.2f}%"),
        ("Sharpe (daily, annualized)", f"{m['sharpe_daily_annualized']:.2f}"),
        ("Max consecutive losses", f"{m['max_consecutive_losses']}"),
        ("Final equity", f"{m['final_equity']:,.2f}"),
    ]
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"  {k:<{width}}  {v}" for k, v in rows)
