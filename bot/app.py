"""Live paper-trading orchestrator with per-user paper accounts.

Loops (all asyncio, single thread):
  * candle loop   - on every entry-timeframe close: fetch candles (REST), compute
                    indicators, evaluate signals once, then for EVERY account: check
                    that account's risk, size against its own equity, alert its chat,
                    and open a paper position if that account has auto-execute on
  * price loop    - WebSocket/REST prices: manage stops, partials, trailing stops
                    for every account
  * housekeeping  - per-account equity checks: UTC day roll, daily loss halt,
                    drawdown kill switch
  * P&L updates   - periodic status message to each account with open positions

Bot-wide (owner) controls: strategy/risk settings and the global kill switch
(manual /kill, repeated API errors, abnormal slippage).
No real orders are ever sent: this build is paper-trading only.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from html import escape
from pathlib import Path

if __package__ in (None, ""):  # started as a script (e.g. VS Code "Run Python File"): make `bot` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from bot import charts, messages  # noqa: E402
from bot.accounts import Account, AccountManager  # noqa: E402
from bot.config import (SETTABLE, Config, ConfigError, _parse_bool, apply_overrides, apply_setting,  # noqa: E402
                        get_path, normalize_symbol)
from bot.data.feed import MarketDataFeed  # noqa: E402
from bot.execution.live import paper_gate_status  # noqa: E402
from bot.execution.models import Event  # noqa: E402
from bot.indicators import add_entry_indicators, add_trend_indicators  # noqa: E402
from bot.risk.manager import MarketLimits, RiskManager  # noqa: E402
from bot.risk.trade_manager import TradeManager  # noqa: E402
from bot.signals.analysis import analyze  # noqa: E402
from bot.signals.engine import Signal, SignalEngine  # noqa: E402
from bot.signals.filters import MarketConditionFilter  # noqa: E402
from bot.storage import Storage  # noqa: E402
from bot.telegram_ui import ConsoleNotifier, Notifier, Reply, TelegramUI, current_user  # noqa: E402
from bot.utils import fmt_ts, now_ms, timeframe_to_ms  # noqa: E402

log = logging.getLogger(__name__)

STARTUP_ATTEMPTS = 4          # connection attempts before giving up at start-up
CHART_BARS_BEFORE_ENTRY = 60  # candles of context before the entry
CHART_MAX_BARS = 300
CHART_MIN_BARS = 10
TOPUP_BUTTONS = (1_000, 5_000, 10_000)


class TradingBot:
    def __init__(self, cfg: Config):
        self.storage = Storage(cfg.logging.db)
        self.cfg, notes = apply_overrides(cfg, self.storage.get_kv("settings_overrides", {}))
        for n in notes:
            log.info(n)
        self.trade_manager = TradeManager(self.cfg.risk.take_profit)
        # bot-wide guard: API-error and slippage kill switch, manual /kill
        self.guard = RiskManager(self.cfg.risk, self.cfg.execution, self.storage.get_kv("guard_state"))
        self.accounts = AccountManager(self.storage, self.cfg, self.trade_manager)
        self.engine = SignalEngine(self.cfg.strategy, self.cfg.risk, self.cfg.timeframes.entry,
                                   self.cfg.timeframes.trend)
        self.filters = MarketConditionFilter(self.cfg.filters)
        self.feed = MarketDataFeed(self.cfg.exchange, demo=self.cfg.mode.trading_mode == "demo")
        self.marks: dict[str, float] = {}
        self.latest_atr: dict[str, float] = {}
        self.notifier: Notifier = ConsoleNotifier()
        self.telegram: TelegramUI | None = None
        self.bot_name = "Trading bot"  # replaced by the Telegram bot's own name
        self._last_equity_snapshot = 0
        self.board: dict = {}                # latest analysis per pair (the scanner)
        self.last_scan_ms: int | None = None

    # ================================================================ run
    async def run(self) -> None:
        log.info("Starting bot in PAPER mode on %s: %s (%d account(s))", self.cfg.exchange.id,
                 ", ".join(self.cfg.pairs), len(self.accounts.all()))
        tasks: list[asyncio.Task] = []
        started = False
        try:
            await self._load_markets_with_retry()
            missing = [p for p in self.cfg.pairs if not self.feed.has_symbol(p)]
            if missing:
                raise ConfigError(f"pairs not listed on {self.cfg.exchange.id}: {missing}")

            if self.cfg.telegram.enabled:
                token = os.environ.get(self.cfg.telegram.token_env)
                if not token:
                    raise ConfigError(f"telegram is enabled but ${self.cfg.telegram.token_env} is not set")
                self.telegram = TelegramUI(self.cfg.telegram, token, self._commands())
                await self.telegram.start()
                self.notifier = self.telegram
                self.bot_name = self.telegram.bot_name

            await self._connect_demo()
            await self._warmup()
            started = True
            await self.notifier.send(
                f"🤖 <b>{escape(self.bot_name)} started</b> "
                f"({'BITGET DEMO for the owner, paper for others' if self.cfg.mode.trading_mode == 'demo' else 'PAPER'}"
                f" on {escape(self.cfg.exchange.id)})\n"
                f"Pairs: {escape(', '.join(self.cfg.pairs))}\n"
                f"Accounts: {len(self.accounts.all())} | risk {self.cfg.risk.risk_per_trade_pct}%/trade | "
                f"leverage cap {self.cfg.risk.max_leverage}x\n"
                f"Each user trades their own paper account. Tap 💼 Account to see yours.\n"
                f"<i>{messages.DISCLAIMER}</i>", keyboard=True)

            tasks = [asyncio.create_task(coro, name=name) for name, coro in (
                ("candles", self._candle_loop()), ("prices", self._price_loop()),
                ("housekeeping", self._housekeeping_loop()), ("pnl", self._pnl_loop()))]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for t in done:
                if t.exception():
                    raise t.exception()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._persist()
            if started:
                await self.notifier.send(f"🛑 {escape(self.bot_name)} stopped. Open paper positions are saved "
                                         f"and resume on restart.")
            if self.telegram:
                await self.telegram.stop()
            if self.accounts.pool is not None:
                await self.accounts.pool.close()
            await self.feed.close()
            self.storage.close()

    async def _load_markets_with_retry(self) -> None:
        delay = self.cfg.exchange.rest_poll_seconds
        for attempt in range(1, STARTUP_ATTEMPTS + 1):
            try:
                await self.feed.load_markets()
                return
            except Exception as exc:  # noqa: BLE001
                if attempt == STARTUP_ATTEMPTS:
                    raise ConnectionError(
                        f"cannot reach {self.cfg.exchange.id} after {attempt} attempts ({exc!r}). "
                        f"Check your network, or whether the exchange is blocked in your region.") from exc
                log.warning("loading markets failed (attempt %d/%d): %s; retrying in %.0fs",
                            attempt, STARTUP_ATTEMPTS, exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)

    # ============================================================ helpers
    def _persist(self, acc: Account | None = None) -> None:
        if acc is None:
            self.accounts.persist_all()
        else:
            self.accounts.persist(acc)
        self.storage.set_kv("guard_state", self.guard.to_state())

    def _equity(self, acc: Account) -> float:
        return acc.executor.equity(self.marks)

    def _watched_symbols(self) -> list[str]:
        held = {p.symbol for a in self.accounts.all() for p in a.executor.positions.values()}
        return sorted(set(self.cfg.pairs) | held)

    def _limits(self, symbol: str) -> MarketLimits:
        return self.feed.market_limits(symbol)

    async def _send(self, acc: Account, text: str) -> None:
        await self.notifier.send(text, chat_id=acc.chat_id)

    async def _caller(self) -> Account:
        """The account of whoever sent the current Telegram update (owner when run from code)."""
        user = current_user.get()
        if user is None or user.id is None:
            owner = self.accounts.owner()
            if owner is None:
                raise RuntimeError("no owner account configured")
            return owner
        acc, created = self.accounts.get_or_create(user.id, user.chat_id or user.id, user.name, user.username)
        if created:
            self.storage.log_event("INFO", "account_created", f"{acc.code} for {acc.display}")
            await self._send(acc, messages.welcome(acc, self.bot_name, self.cfg.mode.starting_equity))
        return acc

    async def _on_api_error(self, exc: Exception, where: str) -> None:
        log.warning("API error in %s: %s", where, exc)
        self.storage.log_event("WARNING", "api_error", f"{where}: {exc}")
        if self.guard.record_api_error(where):
            await self._global_kill()

    async def _global_kill(self) -> None:
        """Bot-wide kill switch: no entries for anyone; flatten every account if configured."""
        reason = self.guard.kill_reason or "unknown"
        self.storage.log_event("CRITICAL", "kill_switch", reason)
        for acc in self.accounts.all():
            closed = 0
            if self.cfg.risk.kill_switch.flatten_on_kill and acc.executor.positions:
                events = acc.executor.close_all(self.marks, "kill switch", now_ms())
                closed = len(events)
                await self._handle_events(acc, events, notify=False)
            if closed or acc.is_owner:
                await self._send(acc, f"🚨 <b>KILL SWITCH</b> (bot-wide): {escape(reason)}\n"
                                      + (f"Closed {closed} paper position(s).\n" if closed else "")
                                      + "No new entries until the owner clears it.")
        self._persist()

    async def _account_kill(self, acc: Account) -> None:
        """One account hit its own drawdown limit: flatten only that account."""
        reason = acc.risk.kill_reason or "drawdown"
        self.storage.log_event("CRITICAL", "account_kill", f"{acc.code}: {reason}")
        text = f"🚨 <b>Your account's kill switch tripped</b>: {escape(reason)}"
        if self.cfg.risk.kill_switch.flatten_on_kill and acc.executor.positions:
            events = acc.executor.close_all(self.marks, "kill switch", now_ms())
            await self._handle_events(acc, events, notify=False)
            text += f"\nClosed {len(events)} paper position(s)."
        text += "\nNo new entries on your account until you send /resetkill CONFIRM."
        self._persist(acc)
        await self._send(acc, text)

    def _is_demo(self, acc: Account) -> bool:
        return getattr(acc.executor, "mode", "paper") == "demo"

    def _pool_text(self) -> str:
        pool = self.accounts.pool
        ledgers = sum(a.executor.balance for a in self.accounts.all())
        return (f"Bitget demo pool: equity <b>{pool.equity:,.2f} USDT</b>, free margin {pool.available:,.2f} · "
                f"users' funded balances total {ledgers:,.2f} USDT")

    async def _connect_demo(self) -> None:
        """Demo mode: connect the shared Bitget demo pool and reconcile every user's ledger.
        Refuses to start without it, so the bot never silently falls back to simulation."""
        pool = self.accounts.pool
        if pool is None:
            return
        held = {p.symbol for a in self.accounts.all() for p in a.executor.positions.values()}
        await pool.connect(sorted(set(self.cfg.pairs) | held), self.feed.rest.markets)
        for acc in self.accounts.all():
            events = await acc.executor.reconcile(now_ms())
            if events:
                await self._handle_events(acc, events)
        self._persist()
        note = "" if pool.equity > 0 else (
            "\n⚠️ The Bitget demo futures balance is 0: add demo USDT in Bitget (Demo Trading) before trades can open.")
        owner = self.accounts.owner()
        if owner is not None:
            await self._send(owner, f"🧪 <b>Bitget DEMO trading connected</b> for all users\n{self._pool_text()}\n"
                                    f"Each user trades from their own funded balance; orders go to the shared "
                                    f"Bitget demo account (fake money). Each user's stop-loss is held on Bitget.{note}")

    async def _check_pool(self, acc: Account, sym: str, side: str, margin: float) -> str | None:
        """Pool-level checks before a demo order. Returns the reason to skip, or None."""
        from bot.execution.demo import conflicting_side

        if conflicting_side([a.executor for a in self.accounts.all()], sym, side):
            return (f"another user holds {sym} in the opposite direction; Bitget would net the two positions "
                    f"on the shared demo account")
        try:
            await self.accounts.pool.refresh()
        except Exception as exc:  # noqa: BLE001
            await self._on_api_error(exc, "demo pool balance")
            return "could not read the Bitget demo balance"
        if self.accounts.pool.available < margin * 1.02:  # small buffer for fees
            return (f"the shared Bitget demo account has only {self.accounts.pool.available:,.2f} USDT free margin "
                    f"(needs {margin:,.2f})")
        return None

    async def _handle_events(self, acc: Account, events: list[Event], notify: bool = True) -> None:
        if self._is_demo(acc) and events:
            for warning in await acc.executor.sync(events):
                await self._send(acc, f"⚠️ {escape(warning)}")
                await self._on_api_error(RuntimeError(warning), "demo order")
        for ev in events:
            if ev.kind == "trailing":
                log.debug("%s %s %s", acc.code, ev.position.symbol, ev.text)
                continue
            if ev.kind in ("partial", "closed"):
                self.storage.log_fill(ev, acc.user_id)
            if ev.kind == "closed" and ev.trade:
                self.storage.log_trade(ev.trade, acc.executor.mode, acc.user_id)
                log.info("%s closed %s %s net %.2f (%+.2fR) %s", acc.code, ev.trade.symbol, ev.trade.side,
                         ev.trade.net_pnl, ev.trade.r_multiple, ev.trade.exit_reason)
            text = messages.position_event(ev)
            if notify and text:
                if ev.kind == "closed" and ev.trade:
                    text += "\n" + messages.settlement(ev.trade, acc.executor.balance,
                                                       acc.executor.available(self.marks))
                    await self._send_chart(acc, ev.position, text, trade=ev.trade)
                    await self._send_profit_chart(acc, ev.trade.net_pnl)
                else:
                    await self._send(acc, text)
        if events:
            self._persist(acc)

    # ============================================================== loops
    async def _warmup(self) -> None:
        """Compute ATR/marks and prime signal cooldowns on the last closed candle
        so a stale signal is not fired at startup."""
        for sym in self.cfg.pairs:
            try:
                await self._process_symbol(sym, act=False)
            except Exception as exc:  # noqa: BLE001
                await self._on_api_error(exc, f"warmup {sym}")
        self._finish_scan()

    async def _candle_loop(self) -> None:
        tf_ms = timeframe_to_ms(self.cfg.timeframes.entry)
        while True:
            now = now_ms()
            next_close = (now // tf_ms + 1) * tf_ms
            await asyncio.sleep((next_close - now) / 1000 + self.cfg.timeframes.candle_close_delay_seconds)
            for sym in list(self.cfg.pairs):
                try:
                    await self._process_symbol(sym, act=True)
                except Exception as exc:  # noqa: BLE001
                    log.exception("processing %s failed", sym)
                    await self._on_api_error(exc, f"candles {sym}")
            self._finish_scan()

    def _next_scan_ms(self) -> int:
        tf_ms = timeframe_to_ms(self.cfg.timeframes.entry)
        return (now_ms() // tf_ms + 1) * tf_ms + int(self.cfg.timeframes.candle_close_delay_seconds * 1000)

    def _finish_scan(self) -> None:
        """Record the scan and log a one-line market summary so the console shows the bot is working."""
        self.last_scan_ms = now_ms()
        parts = []
        for sym in self.cfg.pairs:
            a = self.board.get(sym)
            if a is not None:
                parts.append(f"{sym.split('/')[0]} {a.verdict.upper()} {a.score}/5")
        log.info("scan done: %s | next scan %s", " | ".join(parts) or "no data",
                 fmt_ts(self._next_scan_ms()))

    async def _process_symbol(self, sym: str, act: bool) -> None:
        tfc = self.cfg.timeframes
        ltf_raw = await self.feed.fetch_closed_ohlcv(sym, tfc.entry, tfc.history_bars)
        htf_raw = await self.feed.fetch_closed_ohlcv(sym, tfc.trend, tfc.history_bars)
        self.guard.record_api_success()
        if ltf_raw.empty or htf_raw.empty:
            return
        ltf = add_entry_indicators(ltf_raw, self.cfg.strategy)
        htf = add_trend_indicators(htf_raw, self.cfg.strategy)
        atr_v = ltf["atr"].iat[-1]
        if atr_v == atr_v:  # not NaN
            self.latest_atr[sym] = float(atr_v)
        self.marks.setdefault(sym, float(ltf["close"].iat[-1]))

        try:  # scanner board: full explanation of every pair on every scan (never changes trading state)
            self.board[sym] = analyze(self.engine, self.filters, sym, htf, ltf)
        except Exception:  # noqa: BLE001
            log.debug("scanner analysis failed for %s", sym, exc_info=True)

        ev = self.engine.evaluate(sym, htf, ltf)
        if ev.signal is None:
            log.debug("%s: no signal (%s)", sym, ev.rejection)
            return
        sig = ev.signal
        if not act:
            log.info("%s: startup signal ignored (stale candle)", sym)
            return

        ok, why = self.filters.check(ltf, len(ltf) - 1, sig.timestamp)
        if not ok:
            log.info("%s %s signal suppressed: %s", sym, sig.side, why)
            self.storage.log_signal(sig, "suppressed", why or "")
            return
        ok, why = self.guard.trading_allowed()
        if not ok:
            log.info("%s %s signal blocked bot-wide: %s", sym, sig.side, why)
            self.storage.log_signal(sig, "blocked", why)
            return
        self.storage.log_signal(sig, "delivered", f"{len(self.accounts.all())} account(s)")
        for acc in self.accounts.all():
            try:
                await self._deliver_signal(acc, sig, ltf_raw)
            except Exception:  # noqa: BLE001 - one account's problem must not affect the others
                log.exception("delivering %s signal to %s failed", sym, acc.code)

    async def _deliver_signal(self, acc: Account, sig: Signal, candles: pd.DataFrame) -> None:
        """Size, alert and (if enabled) execute one signal on one account."""
        sym = sig.symbol
        ok, why = acc.risk.can_open(sym, acc.executor.positions.values())
        if not ok:
            log.info("%s: %s signal skipped (%s)", acc.code, sym, why)
            return
        ref = self.marks.get(sym, sig.entry)
        if (ref - sig.stop) * sig.sign <= 0:
            return
        equity = self._equity(acc)
        available = acc.executor.available(self.marks)
        lev = self.cfg.risk.max_leverage
        amount = self._stake_amount(acc)
        if amount <= 0 or available < amount:
            if acc.auto_execute:
                await self._send(acc, f"⚠️ Skipped {escape(sym)} {sig.side} signal: your available balance "
                                      f"{available:,.2f} USDT is below your trade amount {amount:,.2f} USDT "
                                      f"({acc.stake_pct:g}% of your balance; the rest is in open trades).")
            return
        # the trade amount (% of balance) is the margin; position = amount x leverage, still capped by risk limits
        sizing = acc.risk.size_position(equity, ref, sig.stop, self._limits(sym),
                                        acc.executor.notional(self.marks), acc.executor.notional(self.marks, sym),
                                        max_notional=amount * lev)
        if not sizing.ok:
            log.info("%s: %s signal not sized (%s)", acc.code, sym, sizing.reason)
            if acc.auto_execute:
                await self._send(acc, f"⚠️ Skipped {escape(sym)} signal on your account: {escape(sizing.reason)}")
            return
        margin = sizing.notional / lev
        stake_line = (f"Trade amount: <b>{margin:,.2f} USDT</b> ({acc.stake_pct:g}% of your balance) × {lev:g}x "
                      f"leverage = {sizing.notional:,.2f} USDT position")
        if margin < amount * 0.97:  # more than lot rounding: the risk limit shrank it
            stake_line += f" (reduced from {amount:,.2f} by {escape(sizing.capped_by or 'the risk limits')})"
        await self._send(acc, messages.signal_alert(sig, sizing, equity, acc.auto_execute,
                                                    self.cfg.risk.risk_per_trade_pct, stake_line))
        if not acc.auto_execute:
            return
        if self._is_demo(acc):
            why = await self._check_pool(acc, sym, sig.side, margin)
            if why:
                log.info("%s: demo order skipped (%s)", acc.code, why)
                await self._send(acc, f"⚠️ Skipped {escape(sym)} trade: {escape(why)}.")
                return
        pos, fill = acc.executor.open_position(sig, sizing.qty, ref, now_ms(), sizing.risk_amount, margin=margin)
        if self._is_demo(acc):
            try:
                fill = await acc.executor.place_open(pos, ref)
            except Exception as exc:  # noqa: BLE001
                log.exception("demo entry failed for %s", acc.code)
                await self._send(acc, f"⚠️ Bitget demo order for {escape(sym)} failed: {escape(str(exc)[:200])}")
                await self._on_api_error(exc, "demo entry")
                self._persist(acc)
                return
        self.storage.log_fill(Event("opened", pos, price=fill, qty=pos.qty), acc.user_id)
        slip_pct, killed = self.guard.check_slippage(ref, fill)
        self._persist(acc)
        caption = (messages.fill_confirmation(pos, fill, ref, slip_pct)
                   + f"\n💵 {margin:,.2f} USDT taken from your balance · available now "
                     f"{acc.executor.available(self.marks):,.2f} USDT")
        await self._send_chart(acc, pos, caption, candles=candles)
        if killed:
            await self._global_kill()

    async def _price_loop(self) -> None:
        async for sym, price, ts in self.feed.stream_prices(self._watched_symbols, self._on_api_error,
                                                             self.guard.record_api_success):
            self.marks[sym] = price
            for acc in self.accounts.all():
                if any(p.symbol == sym for p in acc.executor.positions.values()):
                    events = acc.executor.on_bar(sym, price, price, price, price, self.latest_atr.get(sym), ts)
                    if events:
                        await self._handle_events(acc, events)

    async def _housekeeping_loop(self) -> None:
        snap_ms = self.cfg.logging.equity_snapshot_minutes * 60_000
        while True:
            await asyncio.sleep(self.cfg.mode.heartbeat_seconds)
            try:
                ts = now_ms()
                snapshot = ts - self._last_equity_snapshot >= snap_ms
                for acc in self.accounts.all():
                    if self._is_demo(acc) and acc.executor.connected:
                        try:  # positions whose stop already triggered on Bitget (between price ticks)
                            closed = await acc.executor.reconcile(ts)
                            if closed:
                                await self._handle_events(acc, closed)
                        except Exception as exc:  # noqa: BLE001
                            await self._on_api_error(exc, "demo account refresh")
                    eq = self._equity(acc)
                    for rev in acc.risk.on_equity(eq, ts):
                        self.storage.log_event("INFO" if rev.kind == "day_rolled" else "WARNING", rev.kind,
                                               f"{acc.code}: {rev.message}")
                        if rev.kind == "kill":
                            await self._account_kill(acc)
                        elif rev.kind == "daily_loss_halt":
                            await self._send(acc, f"⛔ {escape(rev.message)}")
                        elif acc.executor.positions or acc.is_owner:
                            await self._send(acc, f"📅 {escape(rev.message)}")
                    if snapshot:
                        self.storage.log_equity(acc.user_id, ts, eq, acc.executor.balance)
                if snapshot:
                    self._last_equity_snapshot = ts
                self._persist()
            except Exception:  # noqa: BLE001
                log.exception("housekeeping failed")

    async def _pnl_loop(self) -> None:
        minutes = self.cfg.telegram.pnl_update_minutes
        if minutes <= 0:
            return
        while True:
            await asyncio.sleep(minutes * 60)
            for acc in self.accounts.all():
                if acc.executor.positions or acc.is_owner:
                    await self._send(acc, "⏱ " + self._status_text(acc))

    # =========================================================== commands
    def _commands(self) -> dict:
        return {"start": self.cmd_help, "help": self.cmd_help, "status": self.cmd_status,
                "history": self.cmd_history, "pause": self.cmd_pause, "resume": self.cmd_resume,
                "settings": self.cmd_settings, "kill": self.cmd_kill, "resetkill": self.cmd_resetkill,
                "papergate": self.cmd_papergate, "analyze": self.cmd_analyze, "account": self.cmd_account,
                "topup": self.cmd_topup, "autoexec": self.cmd_autoexec, "amount": self.cmd_amount,
                "profit": self.cmd_profit, "scanner": self.cmd_scanner}

    async def cmd_help(self, args: list[str]) -> str:
        await self._caller()  # creates the account on first contact
        return messages.help_text(self.bot_name)

    def _flags(self, acc: Account) -> list[str]:
        flags = []
        if self.guard.kill_active:
            flags.append(f"BOT-WIDE KILL SWITCH ({self.guard.kill_reason})")
        if acc.risk.kill_active:
            flags.append(f"account kill switch ({acc.risk.kill_reason})")
        if acc.risk.paused:
            flags.append("paused")
        if acc.risk.halted_for_day:
            flags.append("halted for the day (daily loss limit)")
        return flags

    def _status_text(self, acc: Account) -> str:
        eq = self._equity(acc)
        text = messages.status(eq, acc.executor.balance, acc.executor.capital,
                               acc.risk.daily_pnl(eq), acc.risk.daily_pnl_pct(eq), acc.risk.drawdown_pct(eq),
                               list(acc.executor.positions.values()), self.marks, self._flags(acc),
                               acc.auto_execute, owner=f"{acc.display} · {acc.code}")
        return text + "\n\n" + self._amount_line(acc)

    def _stake_amount(self, acc: Account) -> float:
        """Trade amount in USDT right now: the chosen % of the account's balance."""
        return max(acc.executor.balance, 0.0) * acc.stake_pct / 100

    def _amount_line(self, acc: Account) -> str:
        lev = self.cfg.risk.max_leverage
        amount = self._stake_amount(acc)
        return (f"💵 Trade amount {acc.stake_pct:g}% of balance = {amount:,.2f} USDT "
                f"(× {lev:g}x = {amount * lev:,.2f} USDT position) | available "
                f"{acc.executor.available(self.marks):,.2f} | in trades {acc.executor.used_margin():,.2f}")

    async def cmd_amount(self, args: list[str]):
        """Pick the trade amount (% of balance) used for every auto-executed trade on the caller's account."""
        acc = await self._caller()
        options = [float(x) for x in self.cfg.mode.stake_pct_options]
        lev = self.cfg.risk.max_leverage
        if args:
            try:
                value = float(args[0].replace("%", "").strip())
            except ValueError:
                value = -1.0
            if value not in options:
                return f"Choose one of: {', '.join(f'{o:g}%' for o in options)}"
            acc.stake_pct = value
            self.accounts.save_profile(acc)
            self.storage.log_event("INFO", "amount", f"{acc.code}: {value:g}%")
            amount = self._stake_amount(acc)
            return (f"✅ Trade amount set to <b>{value:g}% of your balance</b>, "
                    f"{amount:,.2f} USDT right now.\n"
                    f"Each trade takes {value:g}% of your balance as margin and opens a position {lev:g}x that size "
                    f"({amount * lev:,.2f} USDT now). The amount is recalculated for every trade, so it grows "
                    f"with profits and shrinks after losses. When a trade closes you get the amount back plus "
                    f"the profit, or minus the loss.\n"
                    f"<i>Safety: a position is made smaller if it would risk more than "
                    f"{self.cfg.risk.risk_per_trade_pct:g}% of your balance at the stop-loss.</i>")
        balance = acc.executor.balance
        buttons = [[((("✅ " if abs(o - acc.stake_pct) < 1e-9 else "") + f"{o:g}% · {balance * o / 100:,.0f} USDT"),
                     f"amount:{o:g}") for o in options]]
        return Reply(f"💵 <b>Trade amount</b> (% of your balance per trade)\n"
                     f"Current: <b>{acc.stake_pct:g}%</b> = {self._stake_amount(acc):,.2f} USDT × {lev:g}x = "
                     f"{self._stake_amount(acc) * lev:,.2f} USDT position\n"
                     f"Balance {balance:,.2f} USDT · available {acc.executor.available(self.marks):,.2f} USDT\n"
                     f"Tap a percentage:", buttons)

    async def _profit_png(self, acc: Account) -> bytes:
        trades = self.storage.all_trades(acc.user_id, acc.executor.mode)
        avatar = await self.notifier.profile_photo(acc.user_id)
        return await asyncio.to_thread(charts.profit_chart, [t["closed_at"] for t in trades],
                                       [t["net_pnl"] for t in trades],
                                       f"{acc.code} · trade amount {acc.stake_pct:g}% of balance", acc.display, avatar)

    async def _send_profit_chart(self, acc: Account, last_pnl: float) -> None:
        try:
            trades = self.storage.all_trades(acc.user_id, acc.executor.mode)
            total = sum(t["net_pnl"] for t in trades)
            head = ("💰 Profit added to your balance" if last_pnl > 0 else "📉 Loss taken from your balance")
            caption = (f"{head}: <b>{last_pnl:+,.2f} USDT</b>\n"
                       f"Total profit so far: <b>{total:+,.2f} USDT</b> over {len(trades)} trade(s)\n"
                       f"Balance: {acc.executor.balance:,.2f} USDT")
            await self.notifier.send_photo(await self._profit_png(acc), caption, chat_id=acc.chat_id)
        except Exception:  # noqa: BLE001 - the trade itself is already settled and reported
            log.exception("profit chart failed for %s", acc.code)

    async def cmd_profit(self, args: list[str]):
        acc = await self._caller()
        trades = self.storage.all_trades(acc.user_id, acc.executor.mode)
        pnls = [t["net_pnl"] for t in trades]
        text = messages.profit_summary(acc, pnls, acc.executor.balance, acc.executor.capital)
        return Reply(text, [[("💵 Trade amount", "amount"), ("💼 Account", "account")]], await self._profit_png(acc))

    async def cmd_status(self, args: list[str]) -> str:
        return self._status_text(await self._caller())

    async def cmd_history(self, args: list[str]) -> str:
        acc = await self._caller()
        n = int(args[0]) if args and args[0].isdigit() else 10
        return messages.history(self.storage.recent_trades(acc.user_id, max(1, min(n, 50))))

    async def cmd_pause(self, args: list[str]) -> str:
        acc = await self._caller()
        acc.risk.pause()
        self._persist(acc)
        return "⏸ Your account is paused: no new alerts or entries. Open positions keep their stops. /resume to continue."

    async def cmd_resume(self, args: list[str]) -> str:
        acc = await self._caller()
        acc.risk.resume()
        self._persist(acc)
        flags = [f for f in self._flags(acc) if f != "paused"]
        return "▶️ Resumed." if not flags else f"▶️ Pause cleared, but entries are still blocked: {escape('; '.join(flags))}"

    async def cmd_autoexec(self, args: list[str]):
        acc = await self._caller()
        if args:
            try:
                acc.auto_execute = _parse_bool(args[0])
            except ValueError:
                return "Use /autoexec on or /autoexec off"
            self.accounts.save_profile(acc)
            return ("🤖 Auto-execute <b>ON</b>: signals open paper trades on your account automatically."
                    if acc.auto_execute else "🔔 Auto-execute <b>OFF</b>: you get signal alerts only.")
        state = "ON" if acc.auto_execute else "OFF"
        return Reply(f"Auto-execute on your account is <b>{state}</b>.",
                     [[("Turn on", "autoexec:on"), ("Turn off", "autoexec:off")]])

    async def cmd_settings(self, args: list[str]) -> str:
        if len(args) >= 2:
            key, value = args[0].lower(), " ".join(args[1:])
            if key == "autoexec":  # per-account now
                return await self.cmd_autoexec([value])
            new = apply_setting(self.cfg, key, value)  # raises ConfigError with a clear message
            if key == "pairs":
                missing = [p for p in new.pairs if not self.feed.has_symbol(p)]
                if missing:
                    return f"Unknown on {self.cfg.exchange.id}: {escape(', '.join(missing))}"
            self.cfg = new
            self.guard.update_config(new.risk)
            self.accounts.update_risk_config(new)
            self.engine.update_risk_config(new.risk)
            overrides = self.storage.get_kv("settings_overrides", {})
            overrides[key] = value
            self.storage.set_kv("settings_overrides", overrides)
            self.storage.log_event("INFO", "settings", f"{key} = {value}")
            return f"✅ {key} set to {escape(str(get_path(new, SETTABLE[key][0])))} for all accounts"
        acc = await self._caller()
        lines = ["<b>Settings</b> (bot-wide; the owner changes them with /settings &lt;key&gt; &lt;value&gt;)"]
        for key, (path, _, desc) in SETTABLE.items():
            if key == "autoexec":
                lines.append(f"• autoexec = <code>{'on' if acc.auto_execute else 'off'}</code> - "
                             f"your account (change with /autoexec on|off)")
                continue
            val = get_path(self.cfg, path)
            shown = ", ".join(val) if isinstance(val, list) else str(val)
            lines.append(f"• {key} = <code>{escape(shown)}</code> - {escape(desc)}")
        lines.append("Hard limits from bot/risk/limits.py always apply.")
        return "\n".join(lines)

    async def cmd_kill(self, args: list[str]) -> str:
        reason = "manual: " + (" ".join(args) or "Telegram /kill")
        self.guard.trigger_kill(reason)
        await self._global_kill()
        return "Bot-wide kill switch engaged."

    async def cmd_resetkill(self, args: list[str]) -> str:
        acc = await self._caller()
        can_global = acc.is_owner and self.guard.kill_active
        if not acc.risk.kill_active and not can_global:
            return "No kill switch you can clear is active."
        if not args or args[0] != "CONFIRM":
            reasons = [r for r in (self.guard.kill_reason if can_global else None,
                                   acc.risk.kill_reason if acc.risk.kill_active else None) if r]
            return f"Active: {escape('; '.join(reasons))}\nSend /resetkill CONFIRM to clear it."
        cleared = []
        if acc.risk.kill_active:
            acc.risk.reset_kill(self._equity(acc))
            cleared.append("your account's kill switch")
        if can_global:
            self.guard.reset_kill(0.0)
            cleared.append("the bot-wide kill switch")
        self._persist(acc)
        self.storage.log_event("WARNING", "kill_reset", f"{acc.code}: {', '.join(cleared)}")
        return f"Cleared {', '.join(cleared)}. Trading resumes on the next signal."

    async def cmd_papergate(self, args: list[str]) -> str:
        acc = await self._caller()
        _, msg = paper_gate_status(self.storage, self.cfg.paper_gate, acc.user_id)
        return f"Paper-trading requirement for your account: {escape(msg)}\nLive execution is not built in this version."

    async def cmd_scanner(self, args: list[str]):
        """Live board of what the continuous scan sees on every pair. 'now' refreshes it immediately
        (display only: entries are only taken on a closed candle by the scheduled scan)."""
        acc = await self._caller()
        if args and args[0].lower() == "now":
            tfc = self.cfg.timeframes
            for sym in self.cfg.pairs:
                ltf = add_entry_indicators(await self.feed.fetch_closed_ohlcv(sym, tfc.entry, tfc.history_bars),
                                           self.cfg.strategy)
                htf = add_trend_indicators(await self.feed.fetch_closed_ohlcv(sym, tfc.trend, tfc.history_bars),
                                           self.cfg.strategy)
                if not ltf.empty and not htf.empty:
                    self.board[sym] = analyze(self.engine, self.filters, sym, htf, ltf)
        text = messages.scanner_board(self.cfg.pairs, self.board, self.marks, self.last_scan_ms,
                                      self._next_scan_ms(), acc.auto_execute, len(acc.executor.positions),
                                      self.cfg.timeframes.entry)
        return Reply(text, [[("🔄 Refresh now", "scanner:now"), ("📊 Analyze a pair", "analyze")]])

    async def cmd_analyze(self, args: list[str]):
        """No argument: offer pair buttons. 'all', a pair, or a coin name: analyse it now."""
        if not args:
            pairs = self.cfg.pairs
            rows = [[(p.split("/")[0], f"analyze:{p}") for p in pairs[k:k + 2]] for k in range(0, len(pairs), 2)]
            rows.append([("All pairs", "analyze:all")])
            return Reply("📊 Which market should I analyse?\nTap a pair, or type /analyze SOL for any coin "
                         f"on {escape(self.cfg.exchange.id)}.", rows)
        arg = args[0].strip()
        symbols = list(self.cfg.pairs) if arg.lower() == "all" else [self._resolve_symbol(arg)]
        return [await self._analysis_text(sym) for sym in symbols]

    def _resolve_symbol(self, text: str) -> str:
        t = text.strip().upper()
        sym = normalize_symbol(t if "/" in t else f"{t}/USDT", self.cfg.exchange.market_type)
        if not self.feed.has_symbol(sym):
            raise ValueError(f"{sym} is not listed on {self.cfg.exchange.id}")
        return sym

    async def _analysis_text(self, sym: str):
        tfc = self.cfg.timeframes
        ltf = add_entry_indicators(await self.feed.fetch_closed_ohlcv(sym, tfc.entry, tfc.history_bars),
                                   self.cfg.strategy)
        htf = add_trend_indicators(await self.feed.fetch_closed_ohlcv(sym, tfc.trend, tfc.history_bars),
                                   self.cfg.strategy)
        if ltf.empty or htf.empty:
            return f"No candle data for {escape(sym)}."
        change = None
        try:
            change = (await self.feed.rest.fetch_ticker(sym)).get("percentage")
        except Exception:  # noqa: BLE001 - the 24h change is optional
            log.debug("ticker fetch failed for %s", sym, exc_info=True)
        result = analyze(self.engine, self.filters, sym, htf, ltf)
        text = messages.analysis(result, float(change) if change is not None else None)
        verdict = {"signal": "SIGNAL", "wait": "WAIT", "none": "NO TRADE"}[result.verdict]
        try:
            png = await asyncio.to_thread(charts.analysis_chart, ltf, sym, verdict, self.cfg.strategy.ema_fast,
                                          self.cfg.strategy.ema_slow, result.support, result.resistance,
                                          result.setup)
        except Exception:  # noqa: BLE001 - the text analysis still goes out
            log.exception("analysis chart failed for %s", sym)
            return text
        return Reply(text, photo=png)

    # ======================================================= account & charts
    async def _send_chart(self, acc: Account, pos, caption: str, candles=None, trade=None) -> None:
        """Execution chart (entry, SL, TP, and the exit for a closed trade) to the account's chat."""
        try:
            if candles is None:
                bars = self.cfg.timeframes.history_bars
                candles = await self.feed.fetch_closed_ohlcv(pos.symbol, self.cfg.timeframes.entry, bars)
                tf_ms = timeframe_to_ms(self.cfg.timeframes.entry)
                start = pos.opened_at - CHART_BARS_BEFORE_ENTRY * tf_ms
                window = candles[candles["timestamp"] >= start]
                candles = window if len(window) >= CHART_MIN_BARS else candles.tail(CHART_BARS_BEFORE_ENTRY * 2)
            candles = candles.tail(CHART_MAX_BARS)
            result = None
            if trade is not None:
                result = f"{trade.r_multiple:+.2f}R  {trade.net_pnl:+,.2f} USDT"
            png = await asyncio.to_thread(
                charts.trade_chart, candles, pos.side, pos.entry, pos.initial_stop, pos.take_profit, pos.partial_tp,
                pos.opened_at, pos.symbol, trade.closed_at if trade else None,
                trade.exit_price if trade else None, result)
            await self.notifier.send_photo(png, caption, chat_id=acc.chat_id)
        except Exception:  # noqa: BLE001 - a chart must never block a trade notification
            log.exception("trade chart failed; sending text only")
            await self._send(acc, caption)

    async def cmd_account(self, args: list[str]):
        acc = await self._caller()
        eq = self._equity(acc)
        trades = self.storage.all_trades(acc.user_id, acc.executor.mode)
        deposits = self.storage.deposits(acc.user_id)
        text = messages.account_card(
            user=acc, account_id=acc.code, exchange=self.cfg.exchange.id, bot_name=self.bot_name,
            equity=eq, balance=acc.executor.balance, capital=acc.executor.capital,
            starting=acc.executor.starting_equity, deposits=acc.executor.total_deposits,
            open_positions=len(acc.executor.positions), open_pnl=eq - acc.executor.balance,
            daily_pnl=acc.risk.daily_pnl(eq), drawdown=acc.risk.drawdown_pct(eq),
            r_multiples=[t["r_multiple"] for t in trades], pnls=[t["net_pnl"] for t in trades],
            auto_execute=acc.auto_execute)
        hist = self.storage.equity_history(acc.user_id)
        equity = pd.Series([r["equity"] for r in hist], dtype=float,
                           index=pd.to_datetime([r["ts"] for r in hist], unit="ms", utc=True))
        if len(equity):
            equity.loc[pd.Timestamp.now(tz="UTC")] = eq
        dep = pd.Series([d["amount"] for d in deposits], dtype=float,
                        index=pd.to_datetime([d["ts"] for d in deposits], unit="ms", utc=True))
        avatar = await self.notifier.profile_photo(acc.user_id)
        png = await asyncio.to_thread(charts.account_chart, equity, [t["r_multiple"] for t in trades],
                                      acc.executor.capital, dep, f"{acc.code} · {self.bot_name} {'BITGET DEMO' if self._is_demo(acc) else 'paper'} account",
                                      acc.display, avatar)
        if self._is_demo(acc):
            text = ("🧪 <b>BITGET DEMO trading</b>: your trades are real orders on the shared Bitget demo "
                    "account (fake money), sized from your own funded balance below\n"
                    + text.replace("Paper account: simulated trades on real prices. No real money.",
                                   "Fills come from Bitget demo. No real money."))
        buttons = [[(f"➕ {a:,}", f"topup:{a}") for a in TOPUP_BUTTONS],
                   [(f"💵 Amount: {acc.stake_pct:g}%", "amount"), ("📈 Profit", "profit")],
                   [("🤖 Auto-execute: " + ("ON" if acc.auto_execute else "OFF"), "autoexec")]]
        buttons = [row for row in buttons if row]
        return Reply(text + "\n" + self._amount_line(acc), buttons, png)

    async def cmd_topup(self, args: list[str]) -> str:
        acc = await self._caller()
        if not args:
            return "Usage: /topup &lt;amount&gt;, e.g. /topup 5000 (paper USDT, your account)"
        try:
            amount = float(args[0].replace(",", ""))
        except ValueError:
            return "Amount must be a number, e.g. /topup 5000"
        if not 0 < amount <= self.cfg.mode.max_topup:
            return f"Top-up must be between 0 and {self.cfg.mode.max_topup:,.0f} USDT."
        ts = now_ms()
        try:
            acc.executor.deposit(amount)
        except ValueError as exc:
            return f"ℹ️ {escape(str(exc))}"
        acc.risk.on_deposit(amount)
        self.storage.log_deposit(acc.user_id, ts, amount, acc.display)
        self.storage.log_equity(acc.user_id, ts, self._equity(acc), acc.executor.balance)
        self.storage.log_event("INFO", "topup", f"{acc.code}: {amount:,.2f}")
        self._persist(acc)
        return (f"✅ Topped up <b>{amount:,.2f} USDT</b> (paper) on {escape(acc.code)}.\n"
                f"Balance {acc.executor.balance:,.2f} | equity {self._equity(acc):,.2f} | "
                f"capital in {acc.executor.capital:,.2f}\n"
                "Top-ups count as capital, not profit, so P&amp;L and risk limits are unaffected.")


if __name__ == "__main__":
    # Started directly (e.g. VS Code "Run Python File" on this file): hand over to the launcher,
    # which also switches to the project's .venv interpreter if needed.
    import runpy

    sys.argv[0] = str(Path(__file__).resolve().parents[1] / "run_bot.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
