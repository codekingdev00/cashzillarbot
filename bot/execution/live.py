"""Live execution: intentionally NOT implemented yet.

The bot starts in paper-trading only, by design. Config validation rejects
trading_mode: live, and LiveExecutor refuses to construct.

Before building live execution, the paper gate below must pass, and the live
executor must at minimum:
  * read API keys from the environment only (never from config or code)
  * set isolated margin and the configured leverage cap per symbol
  * place the entry and a reduce-only STOP_MARKET stop-loss on the exchange
    itself, so the stop survives a bot crash or network loss
  * verify the stop order was accepted; if not, immediately close the position
  * reconcile local positions with exchange positions on every start
  * measure real slippage on fills and feed RiskManager.check_slippage
  * count every exchange error toward the API-error kill switch
  * start on the exchange testnet before any real funds
"""
from __future__ import annotations

from datetime import datetime, timezone

from bot.config import PaperGateConfig
from bot.storage import Storage


def paper_gate_status(storage: Storage, gate: PaperGateConfig, account_id: int | None = None) -> tuple[bool, str]:
    """Has the paper account met the minimum track record for live trading?

    Meeting it does NOT mean the strategy is profitable or safe; it only means
    there is enough paper history to judge. Review the results before going live.
    """
    stats = storage.paper_stats(account_id)
    trades, first = stats["trades"], stats["first_opened_at"]
    days = 0.0
    if first:
        days = (datetime.now(timezone.utc).timestamp() * 1000 - first) / 86_400_000
    ok = trades >= gate.min_paper_trades and days >= gate.min_paper_days
    msg = (f"paper trades {trades}/{gate.min_paper_trades}, paper days {days:.1f}/{gate.min_paper_days}"
           + (" - requirement met" if ok else " - requirement NOT met"))
    return ok, msg


class LiveExecutor:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Live execution is disabled. This build is paper-trading only. "
            "See bot/execution/live.py for the requirements before enabling it."
        )
