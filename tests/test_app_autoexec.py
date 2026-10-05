"""End-to-end checks of the live orchestrator with per-user paper accounts.

A fake feed replays synthetic candles up to a bar where the engine fires a
signal; each account must be alerted in its own chat, sized on its own equity,
and (if auto-execute is on) get its own paper position, history and charts.
"""
import asyncio

import pytest

import numpy as np

from bot.__main__ import _load_data
from bot.app import TradingBot
from bot.indicators import add_entry_indicators, add_trend_indicators
from bot.risk.manager import MarketLimits
from bot.signals.engine import SignalEngine
from bot.storage import Storage
from bot.telegram_ui import TelegramUser, current_user

SYM = "BTC/USDT:USDT"
OWNER, ALICE, BOB = 111, 222, 333
PNG_MAGIC = bytes([0x89, 0x50, 0x4E, 0x47])  # first bytes of every PNG file


class FakeFeed:
    def __init__(self, htf, ltf):
        self.htf, self.ltf = htf, ltf

    async def fetch_closed_ohlcv(self, symbol, timeframe, limit):
        return (self.ltf if timeframe == "15m" else self.htf).tail(limit).reset_index(drop=True)

    def market_limits(self, symbol):
        return MarketLimits(min_qty=0.0001, min_notional=5, qty_step=0.0001)  # Bitget's BTC lot size

    def has_symbol(self, symbol):
        return True

    async def close(self):
        pass


class Capture:
    """Records every message per chat id."""

    def __init__(self):
        self.by_chat: dict[int, list[str]] = {}
        self.photos: dict[int, int] = {}

    async def send(self, text, keyboard=False, chat_id=None):
        self.by_chat.setdefault(chat_id, []).append(text)

    async def send_photo(self, photo, caption, chat_id=None):
        assert photo[:4] == PNG_MAGIC
        self.photos[chat_id] = self.photos.get(chat_id, 0) + 1
        self.by_chat.setdefault(chat_id, []).append(caption)

    async def profile_photo(self, user_id):
        return None

    def text(self, chat_id):
        return "\n".join(self.by_chat.get(chat_id, []))


def _signal_cut(cfg):
    htf_raw, ltf_raw = _load_data(cfg, "synthetic", "2025-01-01", "2025-03-01", seed=5)[SYM]
    htf, ltf = add_trend_indicators(htf_raw, cfg.strategy), add_entry_indicators(ltf_raw, cfg.strategy)
    idx = np.searchsorted(htf["timestamp"].to_numpy() + 4 * 3600_000, ltf["timestamp"].to_numpy() + 900_000,
                          side="right") - 1
    eng = SignalEngine(cfg.strategy, cfg.risk, "15m", "4h")
    for i in range(len(ltf)):
        if idx[i] >= 0:
            ev = eng.evaluate(SYM, htf, ltf, i, int(idx[i]))
            if ev.signal and ev.signal.side == "long":
                close_ms = int(ltf_raw["timestamp"].iat[i]) + 900_000
                return (htf_raw[htf_raw["timestamp"] + 4 * 3600_000 <= close_ms].reset_index(drop=True),
                        ltf_raw.iloc[: i + 1].reset_index(drop=True), ev.signal)
    raise AssertionError("no long signal in synthetic data")


def _bot(cfg, tmp_path, auto=True):
    cfg.logging.db = str(tmp_path / "bot.db")
    cfg.telegram.allowed_chat_ids = [OWNER]
    cfg.mode.auto_execute = auto
    cfg.pairs = [SYM]
    cfg.filters.allow_extreme_volatility = True  # synthetic candles are not what this test is about
    htf, ltf, _ = _signal_cut(cfg)
    bot = TradingBot(cfg)
    bot.feed = FakeFeed(htf, ltf)
    bot.notifier = Capture()
    return bot


def _as(user_id, name, owner=False):
    current_user.set(TelegramUser(id=user_id, name=name, username=name.lower(), is_owner=owner, chat_id=user_id))


def test_each_user_trades_their_own_account(cfg, tmp_path):
    bot = _bot(cfg, tmp_path)
    cap = bot.notifier

    async def scenario():
        _as(ALICE, "Alice")
        await bot.cmd_account([])                 # Alice's first contact creates her account
        await bot.cmd_topup(["10000"])            # Alice has 20,000; the owner keeps 10,000
        await bot.cmd_amount(["1%"])              # Alice: 1% of 20,000 = 200; owner: default 5% of 10,000 = 500
        _as(BOB, "Bob")
        await bot.cmd_autoexec(["off"])           # Bob only wants alerts
        await bot._process_symbol(SYM, act=True)

        owner, alice, bob = (bot.accounts.get(u) for u in (OWNER, ALICE, BOB))
        assert len(owner.executor.positions) == 1 and len(alice.executor.positions) == 1
        assert not bob.executor.positions
        p_owner = next(iter(owner.executor.positions.values()))
        p_alice = next(iter(alice.executor.positions.values()))
        assert p_owner.id != p_alice.id
        lev = cfg.risk.max_leverage
        one_lot = 0.0001 * p_owner.entry / lev                     # rounding can trim at most one lot
        assert 500 - one_lot <= p_owner.margin <= 500 and 200 - one_lot <= p_alice.margin <= 200
        assert abs(p_owner.qty * p_owner.entry / (p_owner.margin * lev) - 1) < 0.001  # fill incl. slippage
        assert abs(p_alice.qty / p_owner.qty - 0.4) < 0.05
        assert abs(owner.executor.available(bot.marks) - (owner.executor.equity(bot.marks) - p_owner.margin)) < 1e-9
        assert p_alice.risk_amount <= 20_000 * cfg.risk.risk_per_trade_pct / 100 + 1e-6

        # price runs to the partial target, higher, then falls back through the trailing stop
        risk = p_owner.entry - p_owner.initial_stop
        for price in (p_owner.entry + 0.2 * risk, p_owner.partial_tp + 1e-6, p_owner.entry + 1.8 * risk,
                      p_owner.entry + 0.1 * risk):
            bot.marks[SYM] = price
            for acc in bot.accounts.all():
                await bot._handle_events(acc, acc.executor.on_bar(SYM, price, price, price, price, risk / 1.5, 0))
        return owner, alice, bob

    owner, alice, bob = asyncio.run(scenario())
    # every account got its own alert in its own chat
    for chat in (OWNER, ALICE, BOB):
        assert "LONG BTC/USDT:USDT" in cap.text(chat)
    assert "PAPER FILL" in cap.text(ALICE) and "CLOSED" in cap.text(ALICE)
    assert "SIGNAL ONLY" in cap.text(BOB) and "PAPER FILL" not in cap.text(BOB)
    # fill chart, close chart and profit chart
    assert cap.photos[OWNER] == 3 and cap.photos[ALICE] == 3 and BOB not in cap.photos
    assert "returned to your balance" in cap.text(ALICE) and "Profit added to your balance" in cap.text(ALICE)
    assert "taken from your balance" in cap.text(ALICE)
    assert "Welcome" in cap.text(ALICE)

    # histories and balances are separate
    st = bot.storage
    assert len(st.recent_trades(OWNER, 10)) == 1 and len(st.recent_trades(ALICE, 10)) == 1
    assert st.recent_trades(BOB, 10) == []
    assert alice.executor.capital == 20_000 and owner.executor.capital == 10_000
    assert alice.executor.balance > 20_000 and owner.executor.balance > 10_000
    assert len(st.deposits(ALICE)) == 1 and st.deposits(OWNER) == []
    st.close()


def test_account_card_is_per_user(cfg, tmp_path):
    bot = _bot(cfg, tmp_path)
    _as(ALICE, "Alice")
    asyncio.run(bot.cmd_topup(["5000"]))
    card_alice = asyncio.run(bot.cmd_account([]))
    _as(BOB, "Bob")
    card_bob = asyncio.run(bot.cmd_account([]))
    assert "Alice (@alice)" in card_alice.text and "15,000.00" in card_alice.text
    assert "Bob (@bob)" in card_bob.text and "10,000.00" in card_bob.text and "15,000" not in card_bob.text
    assert card_alice.photo[:4] == PNG_MAGIC and card_alice.buttons
    alice, bob = bot.accounts.get(ALICE), bot.accounts.get(BOB)
    assert alice.code != bob.code and alice.code in card_alice.text
    bot.storage.close()


def test_account_limits_are_independent(cfg, tmp_path):
    bot = _bot(cfg, tmp_path)
    _as(ALICE, "Alice")
    asyncio.run(bot.cmd_pause([]))
    asyncio.run(bot._process_symbol(SYM, act=True))
    assert not bot.accounts.get(ALICE).executor.positions          # Alice paused herself
    assert bot.accounts.get(OWNER).executor.positions              # the owner still trades
    assert "LONG" not in bot.notifier.text(ALICE)                  # paused: no alert either
    bot.storage.close()


def test_topup_is_capital_not_profit(cfg, tmp_path):
    bot = _bot(cfg, tmp_path)
    _as(ALICE, "Alice")
    asyncio.run(bot.cmd_account([]))
    alice = bot.accounts.get(ALICE)
    alice.risk.on_equity(10_000, 1_790_000_000_000)
    asyncio.run(bot.cmd_topup(["5,000"]))
    eq = bot._equity(alice)
    assert alice.risk.daily_pnl(eq) == 0 and alice.risk.drawdown_pct(eq) == 0
    assert "between" in asyncio.run(bot.cmd_topup([str(cfg.mode.max_topup * 2)]))
    assert "number" in asyncio.run(bot.cmd_topup(["lots"]))
    bot.storage.close()


def test_accounts_survive_restart(cfg, tmp_path):
    bot = _bot(cfg, tmp_path)
    _as(ALICE, "Alice")
    asyncio.run(bot.cmd_topup(["2500"]))
    code = bot.accounts.get(ALICE).code
    bot._persist()
    bot.storage.close()
    again = TradingBot(cfg)
    alice = again.accounts.get(ALICE)
    assert alice.code == code and alice.executor.capital == 12_500
    again.storage.close()


def test_single_account_data_migrates_to_owner(cfg, tmp_path):
    """Data from before per-user accounts (one shared account) becomes the owner's account."""
    db = str(tmp_path / "bot.db")
    old = Storage(db)
    old.set_kv("paper_state", {"balance": 10_500.0, "starting_equity": 10_000.0, "positions": []})
    old.set_kv("account_id", "PAPER-OLD123")
    old.conn.execute("INSERT INTO trades (id, mode, symbol, side, opened_at, closed_at, net_pnl, r_multiple) "
                     "VALUES ('t1', 'paper', 'BTC/USDT:USDT', 'long', 1, 2, 500, 1.0)")
    old.conn.execute("INSERT INTO equity VALUES (5, 10500, 10500)")
    old.conn.commit()
    old.close()

    cfg.logging.db = db
    cfg.telegram.allowed_chat_ids = [OWNER]
    bot = TradingBot(cfg)
    owner = bot.accounts.get(OWNER)
    assert owner.code == "PAPER-OLD123" and owner.executor.balance == 10_500
    assert len(bot.storage.recent_trades(OWNER, 5)) == 1
    assert len(bot.storage.equity_history(OWNER)) == 1
    bot.storage.close()


def test_amount_selection_and_limits(cfg, tmp_path):
    bot = _bot(cfg, tmp_path)
    _as(ALICE, "Alice")
    menu = asyncio.run(bot.cmd_amount([]))
    labels = [label for row in menu.buttons for label, _ in row]
    assert len(labels) == 3 and labels[1].startswith("✅ 5%") and "500 USDT" in labels[1]   # 5% of 10,000
    assert "Choose one of" in asyncio.run(bot.cmd_amount(["7"]))       # only the offered percentages
    assert "10% of your balance" in asyncio.run(bot.cmd_amount(["10%"]))
    alice = bot.accounts.get(ALICE)
    assert alice.stake_pct == 10 and bot._stake_amount(alice) == 1_000
    asyncio.run(bot.cmd_topup(["10000"]))
    assert bot._stake_amount(alice) == 2_000                            # grows with the balance
    bot.storage.close()


def test_insufficient_balance_skips_with_message(cfg, tmp_path):
    bot = _bot(cfg, tmp_path)
    _as(ALICE, "Alice")
    asyncio.run(bot.cmd_amount(["10"]))
    alice = bot.accounts.get(ALICE)
    alice.executor.balance = 0.0                                         # nothing left to trade with
    asyncio.run(bot._process_symbol(SYM, act=True))
    assert not alice.executor.positions
    assert "below your trade amount" in bot.notifier.text(ALICE)
    bot.storage.close()


def test_risk_limit_still_caps_a_large_amount(cfg, tmp_path):
    """10% x 3x = 30% of the balance in a position; with a ~3.5% stop that risks over 1%: it shrinks."""
    bot = _bot(cfg, tmp_path)
    _as(ALICE, "Alice")
    asyncio.run(bot.cmd_amount(["10"]))
    asyncio.run(bot._process_symbol(SYM, act=True))
    alice = bot.accounts.get(ALICE)
    pos = next(iter(alice.executor.positions.values()))
    assert pos.risk_amount <= 10_000 * cfg.risk.risk_per_trade_pct / 100 + 1e-6
    assert pos.margin < 1_000 * 0.97 and "reduced from 1,000.00" in bot.notifier.text(ALICE)
    bot.storage.close()


def test_profit_view(cfg, tmp_path):
    bot = _bot(cfg, tmp_path)
    _as(ALICE, "Alice")
    reply = asyncio.run(bot.cmd_profit([]))
    assert reply.photo[:4] == PNG_MAGIC and "Total realized profit" in reply.text
    bot.storage.close()


def test_scanner_shows_every_scan(cfg, tmp_path):
    bot = _bot(cfg, tmp_path)
    _as(ALICE, "Alice")
    asyncio.run(bot._process_symbol(SYM, act=True))   # the scheduled scan that also trades
    bot._finish_scan()
    assert bot.board[SYM].verdict == "signal" and bot.last_scan_ms
    board = asyncio.run(bot.cmd_scanner([]))
    assert "Live scanner" in board.text and "🟢 <b>BTC</b>" in board.text and "all entry rules met" in board.text
    refreshed = asyncio.run(bot.cmd_scanner(["now"]))   # display-only refresh
    assert "BTC" in refreshed.text
    assert len(bot.accounts.get(OWNER).executor.positions) == 1   # the refresh did not trade again
    bot.storage.close()


def test_demo_mode_every_user_trades_on_the_bitget_pool_with_own_balance(cfg, tmp_path):
    from test_demo import FakeBitget

    cfg.mode.trading_mode = "demo"
    cfg.exchange.id = "bitget"
    bot = _bot(cfg, tmp_path)
    _as(ALICE, "Alice")
    asyncio.run(bot.cmd_account([]))
    asyncio.run(bot.cmd_topup(["10000"]))                        # Alice funds her own balance: 20,000
    owner, alice = bot.accounts.get(OWNER), bot.accounts.get(ALICE)
    assert owner.executor.mode == alice.executor.mode == "demo"
    assert alice.executor.balance == 20_000 and owner.executor.balance == 10_000

    fake = FakeBitget(balance=1_000_000.0, fill_offset=0.0)
    bot.accounts.pool.client = fake
    bot.feed.rest = type("R", (), {"markets": {}})()
    asyncio.run(bot._connect_demo())
    fake.last_price = float(bot.feed.ltf["close"].iat[-1])
    asyncio.run(bot._process_symbol(SYM, act=True))

    entries = [o for o in fake.orders if not o["stop"]]
    assert len(entries) == 2 and len(fake.open_stops) == 2         # one entry + one own stop per user
    p_owner = next(iter(owner.executor.positions.values()))
    p_alice = next(iter(alice.executor.positions.values()))
    assert abs(p_alice.margin / p_owner.margin - 2) < 0.05          # 5% of each user's OWN balance
    assert fake.position == pytest.approx(p_owner.qty + p_alice.qty)
    assert "Bitget DEMO trading connected" in bot.notifier.text(OWNER)
    assert "1,000,000.00" in bot.notifier.text(OWNER)
    bot.storage.close()


def test_demo_pool_without_free_margin_skips_with_message(cfg, tmp_path):
    from test_demo import FakeBitget

    cfg.mode.trading_mode = "demo"
    cfg.exchange.id = "bitget"
    bot = _bot(cfg, tmp_path)
    fake = FakeBitget(balance=0.0)
    bot.accounts.pool.client = fake
    bot.feed.rest = type("R", (), {"markets": {}})()
    asyncio.run(bot._connect_demo())
    asyncio.run(bot._process_symbol(SYM, act=True))
    assert not fake.orders and not bot.accounts.get(OWNER).executor.positions
    assert "free margin" in bot.notifier.text(OWNER) and "balance is 0" in bot.notifier.text(OWNER)
    bot.storage.close()
