"""Risk manager: position sizing, exposure caps, daily loss halt and kill switch.

Assume losses. Sizing is built so that a stop-out costs at most
risk_per_trade_pct of equity *including* estimated fees and slippage. Gaps and
exchange outages can still produce larger losses; the daily loss halt and the
drawdown kill switch are the second and third lines of defence.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from decimal import ROUND_FLOOR, Decimal
from typing import Callable, Iterable, Protocol

from bot.config import ExecutionConfig, RiskConfig
from bot.utils import utc_day

log = logging.getLogger(__name__)


class _HasSymbol(Protocol):
    symbol: str


@dataclass
class MarketLimits:
    min_qty: float = 0.0
    min_notional: float = 0.0
    qty_step: float | None = None
    rounder: Callable[[float], float] | None = None  # exchange truncation (live feed)

    def round_down(self, qty: float) -> float:
        if qty <= 0:
            return 0.0
        if self.rounder is not None:
            try:
                return float(self.rounder(qty))
            except Exception:  # noqa: BLE001 - ccxt raises when below the lot precision
                return 0.0
        if self.qty_step:
            step = Decimal(str(self.qty_step))
            return float((Decimal(str(qty)) / step).to_integral_value(ROUND_FLOOR) * step)
        return qty


@dataclass
class SizingResult:
    ok: bool
    reason: str = ""
    qty: float = 0.0
    notional: float = 0.0
    risk_amount: float = 0.0
    leverage: float = 0.0
    capped_by: str | None = None


@dataclass
class RiskEvent:
    kind: str      # day_rolled | daily_loss_halt | kill
    message: str


class RiskManager:
    def __init__(self, cfg: RiskConfig, execution: ExecutionConfig, state: dict | None = None):
        self.cfg = cfg
        self.exec = execution
        self.day: str | None = None
        self.day_start_equity: float | None = None
        self.peak_equity: float | None = None
        self.halted_for_day = False
        self.paused = False
        self.kill_active = False
        self.kill_reason: str | None = None
        self.consecutive_api_errors = 0
        if state:
            self.load_state(state)

    # ---------------------------------------------------------------- state
    def to_state(self) -> dict:
        return {k: getattr(self, k) for k in ("day", "day_start_equity", "peak_equity", "halted_for_day",
                                               "paused", "kill_active", "kill_reason")}

    def load_state(self, state: dict) -> None:
        for k, v in state.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def update_config(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    # --------------------------------------------------------------- equity
    def on_equity(self, equity: float, ts_ms: int) -> list[RiskEvent]:
        """Call regularly with mark-to-market equity. Handles the UTC day roll,
        the daily loss halt and the drawdown kill switch."""
        events: list[RiskEvent] = []
        day = utc_day(ts_ms)
        if self.day != day:
            if self.day is not None and self.day_start_equity:
                pnl = equity - self.day_start_equity
                events.append(RiskEvent("day_rolled", f"UTC day {self.day} closed. Day P&L {pnl:+,.2f} "
                                                      f"({pnl / self.day_start_equity * 100:+.2f}%)"))
            self.day = day
            self.day_start_equity = equity
            self.halted_for_day = False
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity

        loss_pct = -self.daily_pnl_pct(equity)
        if not self.halted_for_day and loss_pct >= self.cfg.max_daily_loss_pct:
            self.halted_for_day = True
            events.append(RiskEvent("daily_loss_halt",
                                    f"Daily loss limit hit ({loss_pct:.2f}% >= {self.cfg.max_daily_loss_pct}%). "
                                    f"No new entries until the next UTC day."))
        dd = self.drawdown_pct(equity)
        if not self.kill_active and dd >= self.cfg.kill_switch.max_drawdown_pct:
            self.trigger_kill(f"drawdown {dd:.2f}% >= {self.cfg.kill_switch.max_drawdown_pct}%")
            events.append(RiskEvent("kill", self.kill_reason or "drawdown"))
        return events

    def on_deposit(self, amount: float) -> None:
        """A top-up raises the day-start and peak baselines by the same amount, so it
        does not count as profit for the daily loss limit or the drawdown kill switch."""
        if self.day_start_equity is not None:
            self.day_start_equity += amount
        if self.peak_equity is not None:
            self.peak_equity += amount

    def daily_pnl(self, equity: float) -> float:
        return equity - (self.day_start_equity or equity)

    def daily_pnl_pct(self, equity: float) -> float:
        base = self.day_start_equity or equity
        return (equity - base) / base * 100 if base else 0.0

    def drawdown_pct(self, equity: float) -> float:
        peak = self.peak_equity or equity
        return (peak - equity) / peak * 100 if peak else 0.0

    # ------------------------------------------------------- gating entries
    def trading_allowed(self) -> tuple[bool, str]:
        if self.kill_active:
            return False, f"kill switch active ({self.kill_reason})"
        if self.paused:
            return False, "trading paused"
        if self.halted_for_day:
            return False, "daily loss limit reached"
        return True, ""

    def can_open(self, symbol: str, open_positions: Iterable[_HasSymbol]) -> tuple[bool, str]:
        ok, why = self.trading_allowed()
        if not ok:
            return False, why
        positions = list(open_positions)
        if len(positions) >= self.cfg.max_open_positions:
            return False, f"max open positions ({self.cfg.max_open_positions}) reached"
        if sum(1 for p in positions if p.symbol == symbol) >= self.cfg.max_positions_per_symbol:
            return False, f"max positions per symbol ({self.cfg.max_positions_per_symbol}) reached"
        return True, ""

    def size_position(self, equity: float, entry: float, stop: float, limits: MarketLimits,
                      total_notional: float = 0.0, asset_notional: float = 0.0,
                      max_notional: float | None = None) -> SizingResult:
        """Quantity such that hitting the stop loses <= risk_per_trade_pct of equity,
        then capped by the leverage cap and the per-asset exposure cap, then rounded
        DOWN to the exchange lot step."""
        if equity <= 0:
            return SizingResult(False, "equity is not positive")
        stop_dist = abs(entry - stop)
        if stop_dist <= 0 or entry <= 0:
            return SizingResult(False, "invalid stop distance")
        budget = equity * self.cfg.risk_per_trade_pct / 100
        cost_rate = (self.exec.fee_pct + self.exec.slippage_pct) / 100
        per_unit_risk = stop_dist + (entry + stop) * cost_rate  # entry + exit costs
        qty = budget / per_unit_risk
        risk_capped = False
        if max_notional is not None:  # user's trade amount x leverage; the risk limit still caps it
            if max_notional <= 0:
                return SizingResult(False, "trade amount is not positive")
            if max_notional / entry <= qty:
                qty = max_notional / entry
            else:
                risk_capped = True

        lev_room = equity * self.cfg.max_leverage - total_notional
        asset_room = equity * self.cfg.max_exposure_per_asset_pct / 100 - asset_notional
        room = min(lev_room, asset_room)
        if room <= 0:
            return SizingResult(False, "exposure limit reached")
        capped_by = None
        if risk_capped:
            capped_by = f"the {self.cfg.risk_per_trade_pct:g}% risk-per-trade limit"
        if qty * entry > room:
            qty = room / entry
            capped_by = "leverage cap" if lev_room <= asset_room else "per-asset exposure cap"

        qty = limits.round_down(qty)
        if qty <= 0 or qty < limits.min_qty:
            return SizingResult(False, f"size below exchange minimum quantity ({limits.min_qty})")
        notional = qty * entry
        if notional < limits.min_notional:
            return SizingResult(False, f"order value {notional:.2f} below exchange minimum {limits.min_notional}")
        risk_amount = qty * per_unit_risk
        return SizingResult(True, "", qty=qty, notional=notional, risk_amount=risk_amount,
                            leverage=notional / equity, capped_by=capped_by)

    # ---------------------------------------------------------- kill switch
    def trigger_kill(self, reason: str) -> None:
        if not self.kill_active:
            log.critical("KILL SWITCH: %s", reason)
        self.kill_active = True
        self.kill_reason = reason

    def reset_kill(self, equity: float) -> None:
        """Manual reset only. Peak equity restarts from here so the drawdown
        check does not immediately re-trigger."""
        self.kill_active = False
        self.kill_reason = None
        self.consecutive_api_errors = 0
        self.peak_equity = equity

    def record_api_error(self, where: str) -> bool:
        """Returns True if this error tripped the kill switch."""
        self.consecutive_api_errors += 1
        limit = self.cfg.kill_switch.max_consecutive_api_errors
        if not self.kill_active and self.consecutive_api_errors >= limit:
            self.trigger_kill(f"{self.consecutive_api_errors} consecutive API errors (last: {where})")
            return True
        return False

    def record_api_success(self) -> None:
        self.consecutive_api_errors = 0

    def check_slippage(self, expected: float, fill: float) -> tuple[float, bool]:
        """Returns (slippage %, tripped kill switch)."""
        pct = abs(fill - expected) / expected * 100 if expected else 0.0
        if pct > self.cfg.kill_switch.max_slippage_pct and not self.kill_active:
            self.trigger_kill(f"abnormal slippage {pct:.3f}% > {self.cfg.kill_switch.max_slippage_pct}%")
            return pct, True
        return pct, False

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def snapshot(self) -> dict:
        return asdict(self.cfg) | self.to_state()
