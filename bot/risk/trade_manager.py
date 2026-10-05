"""In-trade management: stop-loss, partial take-profit, breakeven, ATR trailing stop.

Evaluation order within one bar is deliberately pessimistic:
  1. stop-loss first (if a bar touches both stop and target, assume the stop hit)
  2. partial take-profit, then move the stop to breakeven
  3. final take-profit
  4. trailing stop update, which only takes effect from the NEXT bar
Live paper trading feeds single prices (open = high = low = close).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from bot.config import TakeProfitConfig
from bot.execution.models import LONG, Position


@dataclass
class ExitOrder:
    price: float
    qty: float
    reason: str


@dataclass
class BarUpdate:
    exits: list[ExitOrder] = field(default_factory=list)
    notes: list[tuple[str, str]] = field(default_factory=list)  # (kind, text)


class TradeManager:
    def __init__(self, cfg: TakeProfitConfig):
        self.cfg = cfg

    def on_bar(self, pos: Position, o: float, h: float, l: float, c: float, atr: float | None) -> BarUpdate:
        upd = BarUpdate()
        long = pos.side == LONG
        sign = pos.sign

        # 1. stop-loss (gap through the stop fills at the open, which is worse)
        if (long and l <= pos.stop) or (not long and h >= pos.stop):
            px = min(pos.stop, o) if long else max(pos.stop, o)
            upd.exits.append(ExitOrder(px, pos.qty, pos.stop_label))
            return upd

        remaining = pos.qty
        # 2. partial take-profit
        if pos.partial_tp is not None and not pos.partial_done and \
                ((long and h >= pos.partial_tp) or (not long and l <= pos.partial_tp)):
            q = min(pos.initial_qty * self.cfg.partial_pct / 100, remaining)
            px = max(pos.partial_tp, o) if long else min(pos.partial_tp, o)
            upd.exits.append(ExitOrder(px, q, "partial take-profit"))
            remaining -= q
            pos.partial_done = True
            if self.cfg.move_stop_to_breakeven and (pos.entry - pos.stop) * sign > 0:
                pos.stop = pos.entry
                pos.stop_label = "breakeven stop"
                upd.notes.append(("breakeven", "stop moved to breakeven"))

        # 3. final take-profit
        if remaining > 0 and ((long and h >= pos.take_profit) or (not long and l <= pos.take_profit)):
            px = max(pos.take_profit, o) if long else min(pos.take_profit, o)
            upd.exits.append(ExitOrder(px, remaining, "take-profit"))
            return upd

        # 4. trailing stop (applies from the next bar)
        pos.best_price = max(pos.best_price, h) if long else min(pos.best_price, l)
        if self.cfg.trailing and atr and atr > 0 and remaining > 0:
            init_risk = abs(pos.entry - pos.initial_stop)
            moved = (pos.best_price - pos.entry) * sign
            active = pos.partial_done or (pos.partial_tp is None and moved >= self.cfg.partial_at_r * init_risk)
            if active:
                trail = pos.best_price - sign * self.cfg.trail_atr_mult * atr
                if (trail - pos.stop) * sign > 0:
                    pos.stop = trail
                    pos.stop_label = "trailing stop"
                    upd.notes.append(("trailing", f"trailing stop raised to {trail:.6g}" if long
                                      else f"trailing stop lowered to {trail:.6g}"))
        return upd
