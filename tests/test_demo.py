"""Bitget DEMO execution through a shared pool, against a simulated exchange that records every order."""
import asyncio

import pytest

from bot.config import ConfigError, config_from_dict
from bot.execution.demo import DemoExecutionError, DemoExecutor, DemoPool, conflicting_side, make_demo_client
from bot.risk.trade_manager import TradeManager
from bot.signals.engine import Signal

SYM = "BTC/USDT:USDT"


class FakeBitget:
    """Minimal stand-in for ccxt's async bitget client in demo mode (one-way mode, pooled position)."""

    def __init__(self, balance=1_000_000.0, fill_offset=0.5, fail_stop=False):
        self.options = {"sandboxMode": True}
        self.orders: list[dict] = []
        self.open_stops: dict[str, dict] = {}
        self.cancelled: list[str] = []
        self.position = 0.0          # signed base qty, all users netted together
        self.balance = balance
        self.fill_offset = fill_offset
        self.fail_stop = fail_stop
        self.last_price = 100.0
        self._id = 0

    def set_markets(self, markets):
        pass

    async def load_markets(self):
        pass

    async def set_position_mode(self, hedged, symbol):
        assert hedged is False

    async def set_margin_mode(self, mode, symbol, params):
        assert mode == "isolated"

    async def set_leverage(self, lev, symbol, params):
        assert lev == 3

    async def privateMixGetV2MixAccountAccounts(self, params):
        return {"data": [{"marginCoin": "USDT", "accountEquity": str(self.balance), "available": str(self.balance)}]}

    async def fetch_open_orders(self, symbol, params=None):
        assert params == {"trigger": True}
        return [{"id": oid} for oid in self.open_stops]

    async def create_order(self, symbol, type_, side, amount, price=None, params=None):
        params = params or {}
        self._id += 1
        oid = f"o{self._id}"
        is_stop = "triggerPrice" in params
        if is_stop and self.fail_stop:
            raise RuntimeError("trigger order rejected")
        rec = {"id": oid, "side": side, "amount": amount, "stop": is_stop,
               "reduceOnly": params.get("reduceOnly", False), "trigger": params.get("triggerPrice")}
        self.orders.append(rec)
        if is_stop:
            self.open_stops[oid] = rec
            return {"id": oid}
        sign = 1 if side == "buy" else -1
        if params.get("reduceOnly"):
            assert abs(self.position) >= amount - 1e-12, "reduce-only larger than the pooled position"
        self.position += sign * amount
        return {"id": oid, "average": self.last_price + sign * self.fill_offset}

    def trigger(self, oid):
        """Bitget fires a stop: that user's quantity is closed."""
        rec = self.open_stops.pop(oid)
        self.position += (1 if rec["side"] == "buy" else -1) * rec["amount"]

    async def fetch_order(self, oid, symbol):
        return {"id": oid}

    async def cancel_order(self, oid, symbol, params=None):
        assert params == {"trigger": True}
        if oid not in self.open_stops:
            raise RuntimeError("order does not exist")
        self.open_stops.pop(oid)
        self.cancelled.append(oid)

    async def close(self):
        pass


def signal(entry=100.0, stop=98.0, tp=105.0, partial=102.0, side="long"):
    return Signal(symbol=SYM, side=side, timestamp=0, entry=entry, stop=stop, take_profit=tp, partial_tp=partial,
                  atr=1.0, reward_risk=2.5, confidence=4, categories=["trend", "momentum", "volume"],
                  reasons=["test"], missing=[], stop_basis="atr", target_basis="2.5R")


def demo_cfg(raw_cfg):
    raw_cfg["mode"]["trading_mode"] = "demo"
    raw_cfg["exchange"]["id"] = "bitget"
    return config_from_dict(raw_cfg)


def pool_and_users(raw_cfg, n=1, balance=10_000.0, **fake_kw):
    cfg = demo_cfg(raw_cfg)
    cfg.execution.slippage_pct = cfg.execution.fee_pct = 0.0
    fake = FakeBitget(**fake_kw)
    pool = DemoPool(cfg, client=fake)
    asyncio.run(pool.connect([SYM], markets={}))
    tm = TradeManager(cfg.risk.take_profit)
    users = [DemoExecutor(cfg, tm, pool, balance, id_prefix=f"U{i}-") for i in range(n)]
    return users, fake, pool


def test_user_balance_is_own_ledger_and_pool_is_bitget(raw_cfg):
    (ex,), fake, pool = pool_and_users(raw_cfg, balance=7_500.0)
    assert ex.balance == 7_500.0 and pool.equity == 1_000_000.0      # user's funds vs shared Bitget funds
    ex.deposit(500)                                                  # users can still be funded
    assert ex.balance == 8_000.0


def test_open_places_entry_and_users_own_stop(raw_cfg):
    (ex,), fake, _ = pool_and_users(raw_cfg)
    pos, _ = ex.open_position(signal(), 2.0, 100.0, 0, risk_amount=4.0, margin=66.7)
    fill = asyncio.run(ex.place_open(pos, 100.0))
    assert fill == 100.5 and pos.entry == 100.5
    entry, stop = fake.orders
    assert entry["side"] == "buy" and not entry["reduceOnly"] and entry["amount"] == 2.0
    assert stop["stop"] and stop["reduceOnly"] and stop["amount"] == 2.0 and stop["trigger"] == 98.0
    assert pos.exchange_stop_id == stop["id"]


def test_partial_breakeven_trailing_and_close_are_mirrored(raw_cfg):
    (ex,), fake, _ = pool_and_users(raw_cfg, fill_offset=0.0)
    pos, _ = ex.open_position(signal(), 2.0, 100.0, 0, risk_amount=4.0)
    asyncio.run(ex.place_open(pos, 100.0))

    fake.last_price = 102.5
    asyncio.run(ex.sync(ex.on_bar(SYM, 100.0, 102.5, 100.0, 102.2, 1.0, 1)))   # partial + breakeven
    assert fake.position == pytest.approx(1.0)
    stops = list(fake.open_stops.values())
    assert len(stops) == 1 and stops[0]["amount"] == pytest.approx(1.0) and stops[0]["trigger"] >= 100.0

    fake.last_price = 104.0
    asyncio.run(ex.sync(ex.on_bar(SYM, 102.2, 104.0, 102.0, 103.8, 1.0, 2)))   # trailing stop raised
    assert list(fake.open_stops.values())[0]["trigger"] == pytest.approx(102.0)

    fake.last_price = 101.5
    events = ex.on_bar(SYM, 103.8, 103.9, 101.5, 101.8, 1.0, 3)
    asyncio.run(ex.sync(events))
    assert fake.position == pytest.approx(0.0) and not fake.open_stops and not ex.positions
    assert [e for e in events if e.kind == "closed"][0].trade.net_pnl > 0


def test_pooled_users_close_only_their_own_quantity(raw_cfg):
    """Two users long the same coin share one Bitget position; one's stop fires on Bitget,
    the other's close must not touch the first user's (already closed) quantity, nor vice versa."""
    (a, b), fake, _ = pool_and_users(raw_cfg, n=2, fill_offset=0.0)
    pa, _ = a.open_position(signal(stop=98.0), 2.0, 100.0, 0, risk_amount=4.0)
    pb, _ = b.open_position(signal(stop=97.0), 3.0, 100.0, 0, risk_amount=9.0)
    asyncio.run(a.place_open(pa, 100.0))
    asyncio.run(b.place_open(pb, 100.0))
    assert fake.position == pytest.approx(5.0) and len(fake.open_stops) == 2

    fake.trigger(pa.exchange_stop_id)                     # A's stop fires on Bitget between ticks
    assert fake.position == pytest.approx(3.0)
    events = asyncio.run(a.reconcile(5))
    assert len(events) == 1 and "on Bitget" in events[0].trade.exit_reason and not a.positions
    asyncio.run(a.sync(events))                           # no extra close for A
    assert fake.position == pytest.approx(3.0)
    assert asyncio.run(b.reconcile(5)) == []              # B is untouched

    events = [b.close_position(pb.id, 99.0, "manual", 6)]
    asyncio.run(b.sync(events))
    assert fake.position == pytest.approx(0.0) and not fake.open_stops


def test_local_stop_after_bitget_stop_does_not_double_close(raw_cfg):
    (a, b), fake, _ = pool_and_users(raw_cfg, n=2, fill_offset=0.0)
    pa, _ = a.open_position(signal(), 2.0, 100.0, 0, risk_amount=4.0)
    pb, _ = b.open_position(signal(), 2.0, 100.0, 0, risk_amount=4.0)
    asyncio.run(a.place_open(pa, 100.0))
    asyncio.run(b.place_open(pb, 100.0))
    fake.trigger(pa.exchange_stop_id)                     # Bitget closed A
    events = a.on_bar(SYM, 97.9, 97.9, 97.9, 97.9, 1.0, 7)  # local tick also sees A's stop
    asyncio.run(a.sync(events))
    assert fake.position == pytest.approx(2.0)            # B's quantity is still fully open


def test_opposite_side_is_detected(raw_cfg):
    (a, b), _, _ = pool_and_users(raw_cfg, n=2)
    a.open_position(signal(), 1.0, 100.0, 0, risk_amount=2.0)
    assert conflicting_side([a, b], SYM, "short") and not conflicting_side([a, b], SYM, "long")


def test_failed_stop_placement_closes_the_position(raw_cfg):
    (ex,), fake, _ = pool_and_users(raw_cfg, fail_stop=True)
    pos, _ = ex.open_position(signal(), 2.0, 100.0, 0, risk_amount=4.0)
    with pytest.raises(DemoExecutionError, match="stop-loss"):
        asyncio.run(ex.place_open(pos, 100.0))
    assert fake.position == 0.0 and not ex.positions


def test_failed_entry_removes_local_position(raw_cfg):
    (ex,), fake, _ = pool_and_users(raw_cfg)

    async def reject(*a, **k):
        raise RuntimeError("Insufficient margin")
    fake.create_order = reject
    pos, _ = ex.open_position(signal(), 2.0, 100.0, 0, risk_amount=4.0)
    with pytest.raises(RuntimeError, match="Insufficient"):
        asyncio.run(ex.place_open(pos, 100.0))
    assert not ex.positions and ex.balance == 10_000.0


def test_config_modes(raw_cfg):
    import copy
    assert demo_cfg(copy.deepcopy(raw_cfg)).mode.trading_mode == "demo"
    bad = copy.deepcopy(raw_cfg)
    bad["mode"]["trading_mode"] = "live"
    with pytest.raises(ConfigError, match="paper' or 'demo"):
        config_from_dict(bad)
    other = copy.deepcopy(raw_cfg)
    other["mode"]["trading_mode"] = "demo"
    other["exchange"]["id"] = "bybit"
    with pytest.raises(ConfigError, match="bitget"):
        config_from_dict(other)


def test_demo_client_is_locked_to_sandbox_and_needs_passphrase(raw_cfg, monkeypatch):
    cfg = demo_cfg(raw_cfg)
    monkeypatch.setenv(cfg.exchange.api_key_env, "k")
    monkeypatch.setenv(cfg.exchange.api_secret_env, "s")
    monkeypatch.delenv(cfg.exchange.api_password_env, raising=False)
    with pytest.raises(DemoExecutionError, match=cfg.exchange.api_password_env):
        make_demo_client(cfg)
    monkeypatch.setenv(cfg.exchange.api_password_env, "p")
    client = make_demo_client(cfg)
    assert client.options["sandboxMode"] is True
    asyncio.run(client.close())
