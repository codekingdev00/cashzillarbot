"""Position and trade records shared by the paper executor and the backtester."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

LONG, SHORT = "long", "short"


@dataclass
class Position:
    id: str
    symbol: str
    side: str
    entry: float
    qty: float                 # remaining quantity (base units)
    initial_qty: float
    stop: float
    initial_stop: float
    take_profit: float
    partial_tp: float | None
    risk_amount: float         # money lost at the initial stop incl. estimated costs
    opened_at: int
    confidence: int
    planned_rr: float
    reasons: list[str] = field(default_factory=list)
    partial_done: bool = False
    best_price: float = 0.0
    realized_gross: float = 0.0
    fees: float = 0.0
    exit_value: float = 0.0
    exited_qty: float = 0.0
    stop_label: str = "stop-loss"
    margin: float = 0.0        # trade amount taken from the available balance while open
    exchange_stop_id: str | None = None   # stop-loss order held on the exchange (demo mode)

    @property
    def sign(self) -> int:
        return 1 if self.side == LONG else -1

    def unrealized(self, price: float) -> float:
        return (price - self.entry) * self.qty * self.sign

    def notional(self, price: float) -> float:
        return abs(self.qty * price)

    def r_now(self, price: float) -> float:
        init_risk = abs(self.entry - self.initial_stop)
        return (price - self.entry) * self.sign / init_risk if init_risk else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(**d)


@dataclass
class ClosedTrade:
    id: str
    symbol: str
    side: str
    opened_at: int
    closed_at: int
    entry: float
    exit_price: float          # quantity-weighted average exit
    qty: float
    gross_pnl: float
    fees: float
    net_pnl: float
    r_multiple: float          # net P&L / risk at entry
    exit_reason: str
    confidence: int
    planned_rr: float
    reasons: list[str] = field(default_factory=list)
    margin: float = 0.0        # trade amount that was committed; returned with the P&L on close


@dataclass
class Event:
    kind: str                  # opened | partial | closed | breakeven | trailing
    position: Position
    price: float | None = None
    qty: float | None = None
    pnl: float | None = None
    trade: ClosedTrade | None = None
    text: str = ""
