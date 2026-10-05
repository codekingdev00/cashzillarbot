"""Paper (simulated) futures account.

Used by both the live paper bot and the backtester, so both share identical fill,
fee, slippage and trade-management logic. Fills are simulated as taker market
orders with configured slippage applied *against* the trader on every fill.
Paper results are optimistic compared to real trading: real slippage, partial
fills, funding payments and outages are not fully modelled.
"""
from __future__ import annotations

import itertools

from bot.config import ExecutionConfig
from bot.execution.models import LONG, ClosedTrade, Event, Position
from bot.risk.trade_manager import TradeManager
from bot.signals.engine import Signal

DUST = 1e-9


class PaperExecutor:
    mode = "paper"

    def __init__(self, cfg: ExecutionConfig, trade_manager: TradeManager, starting_equity: float,
                 state: dict | None = None, id_prefix: str = ""):
        self.cfg = cfg
        self.tm = trade_manager
        self.id_prefix = id_prefix              # keeps position ids unique across accounts
        self.starting_equity = starting_equity
        self.balance = starting_equity          # realized equity (fees already deducted)
        self.total_deposits = 0.0               # paper top-ups after the start
        self.positions: dict[str, Position] = {}
        self._seq = itertools.count(1)
        if state:
            self.load_state(state)

    # ---------------------------------------------------------------- state
    def to_state(self) -> dict:
        return {"balance": self.balance, "starting_equity": self.starting_equity,
                "total_deposits": self.total_deposits,
                "positions": [p.to_dict() for p in self.positions.values()]}

    def load_state(self, state: dict) -> None:
        self.balance = float(state["balance"])
        self.starting_equity = float(state.get("starting_equity", self.starting_equity))
        self.total_deposits = float(state.get("total_deposits", 0.0))
        self.positions = {d["id"]: Position.from_dict(d) for d in state.get("positions", [])}

    @property
    def capital(self) -> float:
        """Money put in: starting equity plus top-ups. P&L is measured against this."""
        return self.starting_equity + self.total_deposits

    def deposit(self, amount: float) -> None:
        """Paper top-up. Adds capital, not profit."""
        if amount <= 0:
            raise ValueError("top-up amount must be positive")
        self.balance += amount
        self.total_deposits += amount

    # --------------------------------------------------------------- pricing
    def _fill_price(self, price: float, buying: bool) -> float:
        slip = self.cfg.slippage_pct / 100
        return price * (1 + slip) if buying else price * (1 - slip)

    def _fee(self, notional: float) -> float:
        return abs(notional) * self.cfg.fee_pct / 100

    def equity(self, marks: dict[str, float]) -> float:
        return self.balance + sum(p.unrealized(marks.get(p.symbol, p.entry)) for p in self.positions.values())

    def notional(self, marks: dict[str, float], symbol: str | None = None) -> float:
        return sum(p.notional(marks.get(p.symbol, p.entry)) for p in self.positions.values()
                   if symbol is None or p.symbol == symbol)

    # ---------------------------------------------------------------- orders
    def used_margin(self) -> float:
        return sum(p.margin for p in self.positions.values())

    def available(self, marks: dict[str, float]) -> float:
        """Equity not committed to open trades: what a new trade amount can come from."""
        return self.equity(marks) - self.used_margin()

    def open_position(self, signal: Signal, qty: float, ref_price: float, ts: int,
                      risk_amount: float, margin: float = 0.0) -> tuple[Position, float]:
        """Market entry at ref_price (+ slippage). Returns (position, fill price).
        The stop is attached immediately: no position ever exists without one."""
        if signal.stop is None or qty <= 0:
            raise ValueError("refusing to open a position without a stop or with zero size")
        fill = self._fill_price(ref_price, buying=signal.side == LONG)
        if (fill - signal.stop) * signal.sign <= 0:
            raise ValueError("fill price is already beyond the stop")
        fee = self._fee(fill * qty)
        self.balance -= fee
        pid = f"{self.id_prefix}{signal.symbol.split('/')[0]}-{ts}-{next(self._seq)}"
        rr = abs(signal.take_profit - fill) / abs(fill - signal.stop)
        pos = Position(
            id=pid, symbol=signal.symbol, side=signal.side, entry=fill, qty=qty, initial_qty=qty,
            stop=signal.stop, initial_stop=signal.stop, take_profit=signal.take_profit,
            partial_tp=signal.partial_tp, risk_amount=risk_amount, opened_at=ts,
            confidence=signal.confidence, planned_rr=rr, reasons=list(signal.reasons),
            best_price=fill, fees=fee, margin=margin,
        )
        self.positions[pid] = pos
        return pos, fill

    def on_bar(self, symbol: str, o: float, h: float, l: float, c: float, atr: float | None,
               ts: int) -> list[Event]:
        """Apply one price bar (or one tick) to every open position on `symbol`."""
        events: list[Event] = []
        for pos in [p for p in self.positions.values() if p.symbol == symbol]:
            upd = self.tm.on_bar(pos, o, h, l, c, atr)
            for kind, text in upd.notes:
                events.append(Event(kind, pos, text=text))
            for ex in upd.exits:
                events.append(self._fill_exit(pos, ex.price, ex.qty, ex.reason, ts))
                if pos.id not in self.positions:
                    break
        return events

    def close_position(self, pos_id: str, price: float, reason: str, ts: int) -> Event:
        pos = self.positions[pos_id]
        return self._fill_exit(pos, price, pos.qty, reason, ts)

    def close_all(self, marks: dict[str, float], reason: str, ts: int) -> list[Event]:
        return [self.close_position(pid, marks.get(p.symbol, p.entry), reason, ts)
                for pid, p in list(self.positions.items())]

    def _fill_exit(self, pos: Position, price: float, qty: float, reason: str, ts: int) -> Event:
        fill = self._fill_price(price, buying=pos.side != LONG)
        qty = min(qty, pos.qty)
        gross = (fill - pos.entry) * qty * pos.sign
        fee = self._fee(fill * qty)
        self.balance += gross - fee
        pos.qty -= qty
        pos.realized_gross += gross
        pos.fees += fee
        pos.exit_value += fill * qty
        pos.exited_qty += qty
        if pos.qty <= pos.initial_qty * DUST:
            trade = self._finalize(pos, reason, ts)
            return Event("closed", pos, price=fill, qty=qty, pnl=trade.net_pnl, trade=trade, text=reason)
        return Event("partial", pos, price=fill, qty=qty, pnl=gross - fee, text=reason)

    def _finalize(self, pos: Position, reason: str, ts: int) -> ClosedTrade:
        del self.positions[pos.id]
        net = pos.realized_gross - pos.fees
        return ClosedTrade(
            id=pos.id, symbol=pos.symbol, side=pos.side, opened_at=pos.opened_at, closed_at=ts,
            entry=pos.entry, exit_price=pos.exit_value / pos.exited_qty if pos.exited_qty else pos.entry,
            qty=pos.initial_qty, gross_pnl=pos.realized_gross, fees=pos.fees, net_pnl=net,
            r_multiple=net / pos.risk_amount if pos.risk_amount else 0.0, exit_reason=reason,
            confidence=pos.confidence, planned_rr=pos.planned_rr, reasons=pos.reasons, margin=pos.margin,
        )
