"""Event-driven backtester.

Uses the SAME SignalEngine, MarketConditionFilter, RiskManager, TradeManager and
PaperExecutor as the live paper bot, so backtest and paper logic cannot drift.

Look-ahead protections:
  * indicators are causal and swing pivots are only visible once confirmed
  * a trend-timeframe candle is used only after it has CLOSED
  * a signal on bar i is filled at the OPEN of bar i+1
  * if a bar touches both stop and target, the stop is assumed to hit first

PAST PERFORMANCE DOES NOT GUARANTEE FUTURE RESULTS. A backtest models fees and a
fixed slippage, but not funding rates, liquidity, outages or regime changes.
Treat good results with suspicion (overfitting) and expect live results to be worse.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bot.backtest.metrics import compute_metrics
from bot.config import Config
from bot.execution.models import ClosedTrade
from bot.execution.paper import PaperExecutor
from bot.indicators import add_entry_indicators, add_trend_indicators
from bot.risk.manager import MarketLimits, RiskManager
from bot.risk.trade_manager import TradeManager
from bot.signals.engine import Signal, SignalEngine
from bot.signals.filters import MarketConditionFilter
from bot.utils import fmt_ts, timeframe_to_ms

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    trades: list[ClosedTrade]
    equity: pd.Series
    metrics: dict
    signal_counts: dict = field(default_factory=dict)
    kill_events: list[str] = field(default_factory=list)
    max_leverage_used: float = 0.0


class Backtester:
    def __init__(self, cfg: Config, limits: dict[str, MarketLimits] | None = None):
        self.cfg = cfg
        self.limits = limits or {}

    def _limits_for(self, symbol: str) -> MarketLimits:
        return self.limits.get(symbol) or MarketLimits(min_qty=self.cfg.backtest.qty_step,
                                                       min_notional=self.cfg.backtest.min_notional,
                                                       qty_step=self.cfg.backtest.qty_step)

    def run(self, data: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> BacktestResult:
        """data: {symbol: (trend_df, entry_df)} raw OHLCV frames."""
        cfg = self.cfg
        engine = SignalEngine(cfg.strategy, cfg.risk, cfg.timeframes.entry, cfg.timeframes.trend)
        filt = MarketConditionFilter(cfg.filters)
        risk = RiskManager(cfg.risk, cfg.execution)
        ex = PaperExecutor(cfg.execution, TradeManager(cfg.risk.take_profit), cfg.backtest.initial_equity)
        ltf_ms = timeframe_to_ms(cfg.timeframes.entry)
        htf_ms = timeframe_to_ms(cfg.timeframes.trend)

        prepared = {}
        timeline = []
        for sym, (htf_raw, ltf_raw) in data.items():
            htf = add_trend_indicators(htf_raw, cfg.strategy)
            ltf = add_entry_indicators(ltf_raw, cfg.strategy)
            ltf_close = ltf["timestamp"].to_numpy() + ltf_ms
            htf_close = htf["timestamp"].to_numpy() + htf_ms
            htf_idx = np.searchsorted(htf_close, ltf_close, side="right") - 1  # last CLOSED trend bar
            prepared[sym] = (htf, ltf, htf_idx)
            timeline.extend((int(ts), sym, i) for i, ts in enumerate(ltf["timestamp"].to_numpy()))
        timeline.sort()

        pending: dict[str, Signal] = {}
        marks: dict[str, float] = {}
        trades: list[ClosedTrade] = []
        eq_ts, eq_vals = [], []
        counts = {"signals": 0, "suppressed_by_filter": 0, "blocked_by_risk": 0, "skipped_at_entry": 0, "entered": 0}
        kills: list[str] = []
        max_lev = 0.0

        for ts, sym, i in timeline:
            htf, ltf, htf_idx = prepared[sym]
            o, h, l, c = (float(ltf[k].iat[i]) for k in ("open", "high", "low", "close"))
            atr_v = float(ltf["atr"].iat[i])
            atr_v = atr_v if atr_v == atr_v else None

            # 1. fill a pending entry at this bar's open
            sig = pending.pop(sym, None)
            if sig is not None:
                entered = self._try_enter(sig, o, ts, ex, risk, marks, counts)
                if entered:
                    max_lev = max(max_lev, ex.notional(marks | {sym: o}) / max(ex.equity(marks | {sym: o}), 1e-9))

            # 2. manage open positions with this bar
            for ev in ex.on_bar(sym, o, h, l, c, atr_v, ts + ltf_ms):
                if ev.kind == "closed" and ev.trade:
                    trades.append(ev.trade)
            marks[sym] = c

            # 3. equity-based risk checks at bar close
            eq = ex.equity(marks)
            for rev in risk.on_equity(eq, ts + ltf_ms):
                if rev.kind == "kill":
                    kills.append(f"{fmt_ts(ts + ltf_ms)}: {rev.message}")
                    if cfg.risk.kill_switch.flatten_on_kill:
                        for ev in ex.close_all(marks, "kill switch", ts + ltf_ms):
                            if ev.trade:
                                trades.append(ev.trade)
            eq_ts.append(ts + ltf_ms)
            eq_vals.append(ex.equity(marks))

            # 4. evaluate a new signal on this closed bar
            hi = int(htf_idx[i])
            if hi < 0:
                continue
            evaluation = engine.evaluate(sym, htf, ltf, i, hi)
            if evaluation.signal is None:
                continue
            counts["signals"] += 1
            ok, _ = filt.check(ltf, i, evaluation.signal.timestamp)
            if not ok:
                counts["suppressed_by_filter"] += 1
                continue
            pending[sym] = evaluation.signal

        end_ts = eq_ts[-1] if eq_ts else 0
        for ev in ex.close_all(marks, "end of backtest", end_ts):
            if ev.trade:
                trades.append(ev.trade)
        if eq_ts:
            eq_vals[-1] = ex.equity(marks)

        equity = pd.Series(eq_vals, index=pd.to_datetime(eq_ts, unit="ms", utc=True), dtype=float)
        equity = equity[~equity.index.duplicated(keep="last")]
        metrics = compute_metrics(trades, equity, cfg.backtest.initial_equity)
        return BacktestResult(trades, equity, metrics, counts, kills, max_lev)

    def _try_enter(self, sig: Signal, open_px: float, ts: int, ex: PaperExecutor, risk: RiskManager,
                   marks: dict[str, float], counts: dict) -> bool:
        if (open_px - sig.stop) * sig.sign <= 0:
            counts["skipped_at_entry"] += 1  # gapped through the stop
            return False
        rr = (sig.take_profit - open_px) * sig.sign / abs(open_px - sig.stop)
        if rr < self.cfg.risk.min_reward_risk:
            counts["skipped_at_entry"] += 1  # the open moved too far toward the target
            return False
        ok, _ = risk.can_open(sig.symbol, ex.positions.values())
        if not ok:
            counts["blocked_by_risk"] += 1
            return False
        m = marks | {sig.symbol: open_px}
        sizing = risk.size_position(ex.equity(m), open_px, sig.stop, self._limits_for(sig.symbol),
                                    ex.notional(m), ex.notional(m, sig.symbol))
        if not sizing.ok:
            counts["blocked_by_risk"] += 1
            return False
        ex.open_position(sig, sizing.qty, open_px, ts, sizing.risk_amount)
        counts["entered"] += 1
        return True
