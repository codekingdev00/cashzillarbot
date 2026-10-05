"""Per-user paper accounts.

Every Telegram user gets their own simulated account: balance, positions, trade
history, top-ups, daily loss limit, drawdown kill switch, pause and auto-execute.
The strategy (signals) is shared; sizing and execution happen per account against
that account's own equity.
"""
from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass

from bot.config import Config
from bot.execution.paper import PaperExecutor
from bot.risk.manager import RiskManager
from bot.risk.trade_manager import TradeManager
from bot.storage import Storage
from bot.utils import now_ms

log = logging.getLogger(__name__)


@dataclass
class Account:
    user_id: int
    chat_id: int
    name: str
    username: str | None
    code: str
    created_at: int
    auto_execute: bool
    executor: PaperExecutor
    risk: RiskManager
    stake_pct: float            # trade amount, % of the balance, used as margin for each auto-executed trade
    is_owner: bool = False

    @property
    def display(self) -> str:
        return self.name + (f" (@{self.username})" if self.username else "")


class AccountManager:
    def __init__(self, storage: Storage, cfg: Config, trade_manager: TradeManager):
        self.storage = storage
        self.cfg = cfg
        self.tm = trade_manager
        self.owner_ids = {int(x) for x in cfg.telegram.allowed_chat_ids}
        self._accounts: dict[int, Account] = {}
        self.pool = None
        if cfg.mode.trading_mode == "demo":
            from bot.execution.demo import DemoPool

            self.pool = DemoPool(cfg)  # the shared Bitget demo account every user trades through
        self._load()

    # ------------------------------------------------------------------ load
    def _state_key(self, user_id: int, kind: str) -> str:
        return f"acct:{user_id}:{kind}"

    def _kinds(self, uid: int) -> tuple[str, str]:
        """State keys: in demo mode each ledger keeps its own positions and limits, apart from paper."""
        return ("demo", "demo_risk") if self.pool is not None else ("paper", "risk")

    def _initial_demo_state(self, uid: int) -> dict | None:
        """A user's demo ledger starts from their funded paper balance (positions are not carried over)."""
        state = self.storage.get_kv(self._state_key(uid, "demo"))
        if state and state.get("ledger"):
            return state
        paper = self.storage.get_kv(self._state_key(uid, "paper"))
        if not paper:
            return None
        return {"balance": paper["balance"], "starting_equity": paper.get("starting_equity",
                                                                          self.cfg.mode.starting_equity),
                "total_deposits": paper.get("total_deposits", 0.0), "positions": [], "ledger": True}

    def _build(self, row) -> Account:
        uid = int(row["user_id"])
        code = row["code"]
        exec_kind, risk_kind = self._kinds(uid)
        if exec_kind == "demo":
            from bot.execution.demo import DemoExecutor

            executor = DemoExecutor(self.cfg, self.tm, self.pool, self.cfg.mode.starting_equity,
                                    self._initial_demo_state(uid), id_prefix=f"{code}-D-")
        else:
            executor = PaperExecutor(self.cfg.execution, self.tm, self.cfg.mode.starting_equity,
                                     self.storage.get_kv(self._state_key(uid, exec_kind)), id_prefix=f"{code}-")
        risk = RiskManager(self.cfg.risk, self.cfg.execution, self.storage.get_kv(self._state_key(uid, risk_kind)))
        return Account(user_id=uid, chat_id=int(row["chat_id"]), name=row["name"], username=row["username"],
                       code=code, created_at=int(row["created_at"]), auto_execute=bool(row["auto_execute"]),
                       executor=executor, risk=risk, stake_pct=self._valid_pct(row["stake_pct"]),
                       is_owner=uid in self.owner_ids)

    def _valid_pct(self, value) -> float:
        opts = [float(x) for x in self.cfg.mode.stake_pct_options]
        return float(value) if value is not None and float(value) in opts else float(self.cfg.mode.default_stake_pct)

    def _load(self) -> None:
        owner = next(iter(self.cfg.telegram.allowed_chat_ids), None)
        if owner is not None:
            self._migrate_single_account(int(owner))
        for row in self.storage.accounts():
            acc = self._build(row)
            self._accounts[acc.user_id] = acc

    def _migrate_single_account(self, owner_id: int) -> None:
        """Before per-user accounts, there was one bot account. It becomes the owner's."""
        self.storage.migrate_to_accounts(owner_id)
        if any(int(r["user_id"]) == owner_id for r in self.storage.accounts()):
            return
        old_paper, old_risk = self.storage.get_kv("paper_state"), self.storage.get_kv("risk_state")
        code = self.storage.get_kv("account_id") or self._new_code()
        if old_paper:
            self.storage.set_kv(self._state_key(owner_id, "paper"), old_paper)
        if old_risk:
            self.storage.set_kv(self._state_key(owner_id, "risk"), old_risk)
        self.storage.upsert_account(owner_id, owner_id, "Owner", None, code, now_ms(), self.cfg.mode.auto_execute,
                                    self.cfg.mode.default_stake_pct)
        if old_paper:
            log.info("moved the existing paper account to the owner's account %s", code)

    def _new_code(self) -> str:
        existing = {a.code for a in self._accounts.values()}
        while True:
            code = f"PAPER-{secrets.token_hex(3).upper()}"
            if code not in existing:
                return code

    # ---------------------------------------------------------------- access
    def get(self, user_id: int) -> Account | None:
        return self._accounts.get(int(user_id))

    def get_or_create(self, user_id: int, chat_id: int, name: str, username: str | None) -> tuple[Account, bool]:
        """Returns (account, created). Keeps name/username/chat current."""
        acc = self._accounts.get(int(user_id))
        created = acc is None
        if created:
            code = self._new_code()
            self.storage.upsert_account(user_id, chat_id, name, username, code, now_ms(), self.cfg.mode.auto_execute,
                                        self.cfg.mode.default_stake_pct)
            acc = self._build(next(r for r in self.storage.accounts() if int(r["user_id"]) == int(user_id)))
            self._accounts[acc.user_id] = acc
            log.info("created paper account %s for %s", acc.code, acc.display)
        elif (acc.name, acc.username, acc.chat_id) != (name, username, chat_id):
            acc.name, acc.username, acc.chat_id = name, username, chat_id
            self.save_profile(acc)
        return acc, created

    def all(self) -> list[Account]:
        return list(self._accounts.values())

    def owner(self) -> Account | None:
        return next((a for a in self._accounts.values() if a.is_owner), None)

    # ------------------------------------------------------------------ save
    def save_profile(self, acc: Account) -> None:
        self.storage.upsert_account(acc.user_id, acc.chat_id, acc.name, acc.username, acc.code, acc.created_at,
                                    acc.auto_execute, acc.stake_pct)

    def persist(self, acc: Account) -> None:
        exec_kind, risk_kind = self._kinds(acc.user_id)
        self.storage.set_kv(self._state_key(acc.user_id, exec_kind), acc.executor.to_state())
        self.storage.set_kv(self._state_key(acc.user_id, risk_kind), acc.risk.to_state())

    def persist_all(self) -> None:
        for acc in self._accounts.values():
            self.persist(acc)

    def update_risk_config(self, cfg: Config) -> None:
        self.cfg = cfg
        for acc in self._accounts.values():
            acc.risk.update_config(cfg.risk)
