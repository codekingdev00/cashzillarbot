"""Bitget DEMO execution with a pooled exchange account (fake money).

The Bitget demo account is ONE shared trading pool. Every user's trades are sent to it,
but each user trades against their OWN funded balance (a ledger kept by the bot):

  * sizing, the trade amount (% of balance) and "can I afford it" use the user's balance
  * before an order, the pool's free margin on Bitget must also cover it
  * each user's position has its own reduce-only stop-loss trigger order on Bitget, for
    that user's quantity only, so it protects them even if the bot is down
  * fills come from Bitget's executed orders and settle into the user's balance

Netting: one-way mode merges every user's same-coin position into one Bitget position.
Same-direction trades add up safely (each user closes only their own quantity). Opposite
directions on the same coin would cancel each other out, so the bot refuses them (see
DemoPool.conflicting_side, enforced by the app).

Safety: the client is ALWAYS in demo (sandbox) mode; it refuses to start otherwise.
"""
from __future__ import annotations

import logging
import os
import time

from bot.config import Config
from bot.execution.models import LONG, Event, Position
from bot.execution.paper import PaperExecutor
from bot.risk.trade_manager import TradeManager

log = logging.getLogger(__name__)

MARGIN_COIN = "USDT"
TRIGGER_CACHE_SECONDS = 5.0


class DemoExecutionError(RuntimeError):
    pass


def make_demo_client(cfg: Config):
    """Authenticated async ccxt client locked to the exchange's demo environment."""
    import ccxt.async_support as ccxt_async

    ex_cfg = cfg.exchange
    key, secret = os.environ.get(ex_cfg.api_key_env, ""), os.environ.get(ex_cfg.api_secret_env, "")
    password = os.environ.get(ex_cfg.api_password_env, "")
    missing = [n for n, v in ((ex_cfg.api_key_env, key), (ex_cfg.api_secret_env, secret),
                              (ex_cfg.api_password_env, password)) if not v]
    if missing:
        raise DemoExecutionError(f"demo trading needs {', '.join(missing)} in .env")
    client = getattr(ccxt_async, ex_cfg.id)({
        "apiKey": key.strip(), "secret": secret.strip(), "password": password.strip(),
        "enableRateLimit": True, "timeout": ex_cfg.request_timeout_ms,
        "options": {"defaultType": "swap", "fetchMarkets": {"types": ["swap"]}},
    })
    client.has = dict(client.has, fetchCurrencies=False)
    client.set_sandbox_mode(True)
    if not client.options.get("sandboxMode"):
        raise DemoExecutionError("refusing to trade: demo mode could not be enabled on the exchange client")
    return client


class DemoPool:
    """The shared Bitget demo account: one client, its balance, and its open stop orders."""

    def __init__(self, cfg: Config, client=None):
        self.cfg = cfg
        self.client = client
        self.leverage = int(cfg.risk.max_leverage)
        self.connected = False
        self.equity = 0.0
        self.available = 0.0
        self._trigger_ids: dict[str, tuple[float, set[str]]] = {}
        self._prepared: set[str] = set()

    async def connect(self, symbols: list[str], markets: dict | None = None) -> None:
        if self.client is None:
            self.client = make_demo_client(self.cfg)
        if markets:
            self.client.set_markets(markets)
        else:
            await self.client.load_markets()
        for sym in symbols:
            await self.prepare(sym)
        await self.refresh()
        self.connected = True
        log.info("Bitget DEMO pool connected: equity %.2f USDT, available %.2f", self.equity, self.available)

    async def prepare(self, sym: str) -> None:
        """One-way mode, isolated margin, leverage cap. Mode changes fail harmlessly while positions are open."""
        if sym in self._prepared:
            return
        for name, call in (("position mode", lambda: self.client.set_position_mode(False, sym)),
                           ("margin mode", lambda: self.client.set_margin_mode("isolated", sym,
                                                                               {"marginCoin": MARGIN_COIN})),
                           ("leverage", lambda: self.client.set_leverage(self.leverage, sym,
                                                                         {"marginCoin": MARGIN_COIN}))):
            try:
                await call()
            except Exception as exc:  # noqa: BLE001
                log.info("demo %s for %s not changed: %s", name, sym, str(exc)[:120])
        self._prepared.add(sym)

    async def refresh(self) -> None:
        accounts = (await self.client.privateMixGetV2MixAccountAccounts({"productType": "USDT-FUTURES"}))["data"]
        acc = next((a for a in accounts if a.get("marginCoin") == MARGIN_COIN), None)
        self.equity = float(acc.get("accountEquity") or 0) if acc else 0.0
        self.available = float(acc.get("available") or acc.get("crossedMaxAvailable") or 0) if acc else 0.0

    async def open_trigger_ids(self, symbol: str, fresh: bool = False) -> set[str]:
        """Ids of stop trigger orders still waiting on Bitget (short cache: one call per heartbeat)."""
        cached = self._trigger_ids.get(symbol)
        if cached and not fresh and time.monotonic() - cached[0] < TRIGGER_CACHE_SECONDS:
            return cached[1]
        orders = await self.client.fetch_open_orders(symbol, params={"trigger": True})
        ids = {str(o.get("id")) for o in orders if o.get("id")}
        self._trigger_ids[symbol] = (time.monotonic(), ids)
        return ids

    def forget(self, symbol: str) -> None:
        self._trigger_ids.pop(symbol, None)

    async def close(self) -> None:
        if self.client is not None:
            try:
                await self.client.close()
            except Exception:  # noqa: BLE001
                log.debug("error closing demo client", exc_info=True)


def conflicting_side(executors, symbol: str, side: str) -> bool:
    """True if any user holds `symbol` in the opposite direction (one-way mode would net them)."""
    return any(p.symbol == symbol and p.side != side for ex in executors for p in ex.positions.values())


class DemoExecutor(PaperExecutor):
    """One user's ledger whose trades are executed on the shared Bitget demo pool."""

    mode = "demo"

    def __init__(self, cfg: Config, trade_manager: TradeManager, pool: DemoPool, starting_equity: float,
                 state: dict | None = None, id_prefix: str = ""):
        super().__init__(cfg.execution, trade_manager, starting_equity, state, id_prefix=id_prefix)
        self.pool = pool

    @property
    def client(self):
        return self.pool.client

    @property
    def connected(self) -> bool:
        return self.pool.connected

    def to_state(self) -> dict:
        return super().to_state() | {"ledger": True}

    # ------------------------------------------------------------ orders
    async def _avg_fill(self, order: dict, symbol: str, fallback: float) -> float:
        price = order.get("average") or order.get("price") if order else None
        if not price and order and order.get("id"):
            try:
                fetched = await self.client.fetch_order(order["id"], symbol)
                price = fetched.get("average") or fetched.get("price")
            except Exception:  # noqa: BLE001
                log.debug("could not fetch demo order %s", order.get("id"), exc_info=True)
        return float(price) if price else fallback

    async def _place_stop(self, pos: Position) -> None:
        side = "sell" if pos.side == LONG else "buy"
        order = await self.client.create_order(pos.symbol, "market", side, pos.qty, None,
                                               {"triggerPrice": pos.stop, "reduceOnly": True,
                                                "marginMode": "isolated", "marginCoin": MARGIN_COIN})
        pos.exchange_stop_id = str(order.get("id")) if order.get("id") else None
        self.pool.forget(pos.symbol)

    async def _stop_still_open(self, pos: Position) -> bool:
        if not pos.exchange_stop_id:
            return False
        return pos.exchange_stop_id in await self.pool.open_trigger_ids(pos.symbol, fresh=True)

    async def _cancel_stop(self, pos: Position) -> bool:
        """Cancel this user's stop. True if it was still waiting (so the user's quantity is still open)."""
        if not pos.exchange_stop_id:
            return False
        try:
            await self.client.cancel_order(pos.exchange_stop_id, pos.symbol, {"trigger": True})
            ok = True
        except Exception as exc:  # noqa: BLE001 - already triggered, or a network error: check the order book
            log.info("demo stop %s not cancelled: %s", pos.exchange_stop_id, str(exc)[:120])
            ok = await self._stop_still_open(pos)
            if ok:  # still there: the cancel failed for another reason, try once more
                await self.client.cancel_order(pos.exchange_stop_id, pos.symbol, {"trigger": True})
        pos.exchange_stop_id = None
        self.pool.forget(pos.symbol)
        return ok

    async def place_open(self, pos: Position, expected: float) -> float:
        """Send the entry and this user's exchange-held stop. Returns the real fill price.
        On failure the local position is removed and the error re-raised."""
        await self.pool.prepare(pos.symbol)
        side = "buy" if pos.side == LONG else "sell"
        try:
            order = await self.client.create_order(pos.symbol, "market", side, pos.qty, None,
                                                   {"marginMode": "isolated", "marginCoin": MARGIN_COIN})
        except Exception:
            self.positions.pop(pos.id, None)
            self.balance += pos.fees  # undo the simulated entry fee
            raise
        fill = await self._avg_fill(order, pos.symbol, pos.entry)
        pos.entry = pos.best_price = fill
        try:
            await self._place_stop(pos)
        except Exception as exc:
            # never leave a position without a stop: close this user's quantity straight away
            log.error("demo stop-loss placement failed (%s); closing the position", exc)
            close = await self._market_close(pos.symbol, pos.side, pos.qty)
            self.close_position(pos.id, await self._avg_fill(close, pos.symbol, fill), "stop-loss not placed",
                                int(time.time() * 1000))
            raise DemoExecutionError(f"stop-loss could not be placed on Bitget, position closed: {exc}") from exc
        return fill

    async def _market_close(self, symbol: str, side: str, qty: float) -> dict | None:
        close_side = "sell" if side == LONG else "buy"
        return await self.client.create_order(symbol, "market", close_side, qty, None,
                                              {"reduceOnly": True, "marginMode": "isolated",
                                               "marginCoin": MARGIN_COIN})

    def _adjust_exit(self, ev: Event, real_price: float) -> None:
        """Replace the simulated exit price of a partial/close with Bitget's fill."""
        pos, qty = ev.position, ev.qty or 0.0
        diff = (real_price - (ev.price or real_price)) * qty * pos.sign
        if not diff:
            return
        pos.realized_gross += diff
        pos.exit_value += (real_price - (ev.price or real_price)) * qty
        self.balance += diff
        ev.price = real_price
        if ev.pnl is not None:
            ev.pnl += diff
        if ev.trade is not None:
            t = ev.trade
            t.gross_pnl += diff
            t.net_pnl += diff
            t.exit_price = pos.exit_value / pos.exited_qty if pos.exited_qty else real_price
            t.r_multiple = t.net_pnl / pos.risk_amount if pos.risk_amount else 0.0

    async def sync(self, events: list[Event]) -> list[str]:
        """Mirror locally generated events to Bitget. Returns warnings for the user.
        A user's quantity is only ever closed while their own stop is still waiting on Bitget,
        so a close can never eat into another user's share of a pooled position."""
        warnings: list[str] = []
        for ev in events:
            pos = ev.position
            try:
                if ev.kind in ("breakeven", "trailing"):
                    if pos.id in self.positions and await self._cancel_stop(pos):
                        await self._place_stop(pos)
                elif ev.kind == "partial":
                    if await self._cancel_stop(pos):
                        order = await self._market_close(pos.symbol, pos.side, ev.qty)
                        self._adjust_exit(ev, await self._avg_fill(order, pos.symbol, ev.price))
                        await self._place_stop(pos)  # re-place for the remaining quantity
                elif ev.kind == "closed":
                    if await self._cancel_stop(pos):
                        order = await self._market_close(pos.symbol, pos.side, ev.qty)
                        self._adjust_exit(ev, await self._avg_fill(order, pos.symbol, ev.price))
                    # else: Bitget's stop already closed this user's quantity at the stop price
            except Exception as exc:  # noqa: BLE001
                log.exception("demo sync of %s %s failed", ev.kind, pos.id)
                warnings.append(f"Bitget demo {ev.kind} for {pos.symbol} failed: {exc}")
        return warnings

    async def reconcile(self, ts: int) -> list[Event]:
        """Close local positions whose stop already triggered on Bitget (between price ticks)."""
        events: list[Event] = []
        for pid, pos in list(self.positions.items()):
            if pos.exchange_stop_id and pos.exchange_stop_id in await self.pool.open_trigger_ids(pos.symbol):
                continue
            pos.exchange_stop_id = None
            events.append(self.close_position(pid, pos.stop, f"{pos.stop_label} (on Bitget)", ts))
        return events
