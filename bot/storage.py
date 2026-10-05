"""SQLite persistence: accounts, signals, fills, closed trades, events, equity and bot state.

Every per-user record (trades, fills, deposits, equity) carries an account_id, the
Telegram user id of the account holder. Signals and events are bot-wide.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from bot.execution.models import ClosedTrade, Event
from bot.signals.engine import Signal
from bot.utils import now_ms

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, symbol TEXT, side TEXT, confidence INTEGER,
    entry REAL, stop REAL, take_profit REAL, reward_risk REAL, categories TEXT, reasons TEXT,
    status TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, position_id TEXT, symbol TEXT, action TEXT,
    price REAL, qty REAL, pnl REAL, note TEXT);
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY, mode TEXT, symbol TEXT, side TEXT, opened_at INTEGER, closed_at INTEGER,
    entry REAL, exit_price REAL, qty REAL, gross_pnl REAL, fees REAL, net_pnl REAL, r_multiple REAL,
    exit_reason TEXT, confidence INTEGER, planned_rr REAL, reasons TEXT);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, level TEXT, kind TEXT, message TEXT);
CREATE TABLE IF NOT EXISTS equity (ts INTEGER PRIMARY KEY, equity REAL, balance REAL);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS deposits (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, amount REAL, by_user TEXT);
CREATE TABLE IF NOT EXISTS accounts (
    user_id INTEGER PRIMARY KEY, chat_id INTEGER, name TEXT, username TEXT, code TEXT UNIQUE,
    created_at INTEGER, auto_execute INTEGER);
CREATE TABLE IF NOT EXISTS account_equity (
    account_id INTEGER, ts INTEGER, equity REAL, balance REAL, PRIMARY KEY (account_id, ts));
"""

ACCOUNT_TABLES = ("trades", "fills", "deposits")


class Storage:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        for table in ACCOUNT_TABLES:
            cols = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            if "account_id" not in cols:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN account_id INTEGER")
        acct_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(accounts)")}
        if "stake_pct" not in acct_cols:  # trade amount as % of balance (older 'stake' USDT column is unused)
            self.conn.execute("ALTER TABLE accounts ADD COLUMN stake_pct REAL")
        self.conn.commit()

    def migrate_to_accounts(self, owner_id: int) -> bool:
        """One-time: records written before per-user accounts existed belong to the owner.
        Returns True if anything was migrated."""
        if self.get_kv("migrated_accounts_v1"):
            return False
        for table in ACCOUNT_TABLES:
            self.conn.execute(f"UPDATE {table} SET account_id = ? WHERE account_id IS NULL", (owner_id,))
        self.conn.execute("INSERT OR IGNORE INTO account_equity SELECT ?, ts, equity, balance FROM equity",
                          (owner_id,))
        self.conn.commit()
        self.set_kv("migrated_accounts_v1", True)
        return True

    def _exec(self, sql: str, params: tuple = ()) -> None:
        self.conn.execute(sql, params)
        self.conn.commit()

    # ---------------------------------------------------------------- accounts
    def upsert_account(self, user_id: int, chat_id: int, name: str, username: str | None, code: str,
                       created_at: int, auto_execute: bool, stake_pct: float | None) -> None:
        self._exec("INSERT INTO accounts (user_id, chat_id, name, username, code, created_at, auto_execute, stake_pct) "
                   "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET "
                   "chat_id=excluded.chat_id, name=excluded.name, username=excluded.username, "
                   "auto_execute=excluded.auto_execute, stake_pct=excluded.stake_pct",
                   (user_id, chat_id, name, username, code, created_at, int(auto_execute), stake_pct))

    def accounts(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM accounts ORDER BY created_at").fetchall()

    # ----------------------------------------------------------------- logging
    def log_signal(self, sig: Signal, status: str, note: str = "") -> None:
        self._exec("INSERT INTO signals (ts, symbol, side, confidence, entry, stop, take_profit, reward_risk, "
                   "categories, reasons, status, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (sig.timestamp, sig.symbol, sig.side, sig.confidence, sig.entry, sig.stop, sig.take_profit,
                    sig.reward_risk, json.dumps(sig.categories), json.dumps(sig.reasons), status, note))

    def log_fill(self, ev: Event, account_id: int, ts: int | None = None) -> None:
        self._exec("INSERT INTO fills (ts, position_id, symbol, action, price, qty, pnl, note, account_id) "
                   "VALUES (?,?,?,?,?,?,?,?,?)",
                   (ts or now_ms(), ev.position.id, ev.position.symbol, ev.kind, ev.price, ev.qty, ev.pnl, ev.text,
                    account_id))

    def log_trade(self, t: ClosedTrade, mode: str, account_id: int) -> None:
        self._exec("INSERT OR REPLACE INTO trades (id, mode, symbol, side, opened_at, closed_at, entry, exit_price, "
                   "qty, gross_pnl, fees, net_pnl, r_multiple, exit_reason, confidence, planned_rr, reasons, "
                   "account_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (t.id, mode, t.symbol, t.side, t.opened_at, t.closed_at, t.entry, t.exit_price, t.qty,
                    t.gross_pnl, t.fees, t.net_pnl, t.r_multiple, t.exit_reason, t.confidence, t.planned_rr,
                    json.dumps(t.reasons), account_id))

    def log_event(self, level: str, kind: str, message: str) -> None:
        self._exec("INSERT INTO events (ts, level, kind, message) VALUES (?,?,?,?)",
                   (now_ms(), level, kind, message))

    def log_equity(self, account_id: int, ts: int, equity: float, balance: float) -> None:
        self._exec("INSERT OR REPLACE INTO account_equity VALUES (?,?,?,?)", (account_id, ts, equity, balance))

    def log_deposit(self, account_id: int, ts: int, amount: float, by_user: str) -> None:
        self._exec("INSERT INTO deposits (ts, amount, by_user, account_id) VALUES (?,?,?,?)",
                   (ts, amount, by_user, account_id))

    # ----------------------------------------------------------------- queries
    def deposits(self, account_id: int) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM deposits WHERE account_id = ? ORDER BY ts", (account_id,)).fetchall()

    def equity_history(self, account_id: int) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT ts, equity FROM account_equity WHERE account_id = ? ORDER BY ts",
                                 (account_id,)).fetchall()

    def all_trades(self, account_id: int, mode: str = "paper") -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM trades WHERE account_id = ? AND mode = ? ORDER BY closed_at",
                                 (account_id, mode)).fetchall()

    def recent_trades(self, account_id: int, n: int) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM trades WHERE account_id = ? ORDER BY closed_at DESC LIMIT ?",
                                 (account_id, n)).fetchall()

    def paper_stats(self, account_id: int | None = None) -> dict[str, Any]:
        where, params = "WHERE mode = 'paper'", ()
        if account_id is not None:
            where, params = where + " AND account_id = ?", (account_id,)
        row = self.conn.execute("SELECT COUNT(*) AS n, MIN(opened_at) AS first, "
                                "SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) AS wins, "
                                f"SUM(net_pnl) AS pnl FROM trades {where}", params).fetchone()
        return {"trades": row["n"] or 0, "first_opened_at": row["first"], "wins": row["wins"] or 0,
                "net_pnl": row["pnl"] or 0.0}

    # ---------------------------------------------------------------- kv state
    def get_kv(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_kv(self, key: str, value: Any) -> None:
        self._exec("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, json.dumps(value)))

    def close(self) -> None:
        self.conn.close()
