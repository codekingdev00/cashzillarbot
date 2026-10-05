"""One-shot end-to-end check of Bitget DEMO execution (fake money).

Opens the smallest BTC long the exchange allows, verifies the stop-loss trigger order
is held on Bitget, closes the position and verifies the account is flat again.

    .\\.venv\\Scripts\\python scripts\\demo_smoke_test.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot.config import load_config, load_dotenv  # noqa: E402
from bot.execution.demo import DemoExecutor, DemoPool, make_demo_client  # noqa: E402
from bot.risk.trade_manager import TradeManager  # noqa: E402
from bot.signals.engine import Signal  # noqa: E402
from bot.utils import now_ms  # noqa: E402

SYM = "BTC/USDT:USDT"


async def main() -> int:
    load_dotenv(ROOT / ".env")
    cfg = load_config(ROOT / "config" / "config.yaml")
    cfg.exchange.id = "bitget"
    cfg.mode.trading_mode = "demo"
    client = make_demo_client(cfg)
    pool = DemoPool(cfg, client=client)
    ex = DemoExecutor(cfg, TradeManager(cfg.risk.take_profit), pool, starting_equity=10_000.0)
    try:
        await pool.connect([SYM])
        print(f"1. connected to Bitget DEMO, pool equity {pool.equity:,.2f} USDT, free {pool.available:,.2f}")
        if pool.equity <= 0:
            print("   STOP: the demo futures balance is 0. Add demo USDT in Bitget (Demo Trading) and rerun.")
            return 1
        price = float((await client.fetch_ticker(SYM))["last"])
        market = client.market(SYM)
        qty = float(market["limits"]["amount"]["min"] or 0.0001)
        while qty * price < max(float(market["limits"]["cost"]["min"] or 5), 6):
            qty += float(market["precision"]["amount"])
        stop = round(price * 0.98, 1)
        sig = Signal(symbol=SYM, side="long", timestamp=now_ms(), entry=price, stop=stop, take_profit=price * 1.05,
                     partial_tp=None, atr=price * 0.005, reward_risk=2.5, confidence=0, categories=[],
                     reasons=["demo smoke test"], missing=[], stop_basis="2% test stop", target_basis="test")
        pos, _ = ex.open_position(sig, qty, price, now_ms(), risk_amount=qty * (price - stop),
                                  margin=qty * price / cfg.risk.max_leverage)
        fill = await ex.place_open(pos, price)
        print(f"2. opened {qty} BTC long at {fill:,.2f} (ticker {price:,.2f})")

        ours = pos.exchange_stop_id in await pool.open_trigger_ids(SYM, fresh=True)
        print(f"3. stop-loss order {pos.exchange_stop_id} on Bitget at {stop:,.1f}: {'FOUND' if ours else 'NOT FOUND'}")

        await asyncio.sleep(3)
        events = [ex.close_position(pos.id, float((await client.fetch_ticker(SYM))["last"]), "smoke test close",
                                    now_ms())]
        warnings = await ex.sync(events)
        trade = events[0].trade
        await pool.refresh()
        print(f"4. closed at {trade.exit_price:,.2f}: net {trade.net_pnl:+.4f} USDT | stop removed: "
              f"{pos.exchange_stop_id is None} | pool equity {pool.equity:,.2f}")
        for w in warnings:
            print("   warning:", w)
        ok = bool(ours) and pos.exchange_stop_id is None and not warnings
        print("RESULT:", "PASS" if ok else "CHECK THE LINES ABOVE")
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}")
        try:  # never leave a test position behind
            for p in list(ex.positions.values()):
                await ex.sync([ex.close_position(p.id, p.entry, "cleanup", now_ms())])
                print("   cleaned up the open test position")
        except Exception:  # noqa: BLE001
            pass
        return 1
    finally:
        await pool.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
