"""Command line entry point.

  python -m bot check-config [--config PATH]
  python -m bot run          [--config PATH]          # live PAPER trading
  python -m bot telegram-chatid                      # find your chat id
  python -m bot backtest     [--config PATH] [--source exchange|csv|synthetic] [--start] [--end]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):  # started as a script: make `bot` importable
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.config import ConfigError, load_config, load_dotenv  # noqa: E402
from bot.logging_setup import setup_logging  # noqa: E402
from bot.utils import parse_utc, timeframe_to_ms  # noqa: E402

log = logging.getLogger("bot")

BACKTEST_WARNING = (
    "PAST PERFORMANCE DOES NOT GUARANTEE FUTURE RESULTS. This backtest uses simulated fills with fixed\n"
    "fees and slippage and ignores funding, liquidity and outages. Expect live results to be worse.\n"
    "Paper trade before considering real money."
)


def _load_data(cfg, source: str, start: str, end: str, seed: int) -> dict:
    tf_e, tf_t = cfg.timeframes.entry, cfg.timeframes.trend
    start_ms, end_ms = parse_utc(start), parse_utc(end)
    data = {}
    if source == "synthetic":
        from bot.data.synthetic import generate_ohlcv, resample_ohlcv

        warm = cfg.timeframes.history_bars * timeframe_to_ms(tf_t)
        n = int((end_ms - start_ms + warm) // timeframe_to_ms(tf_e))
        for k, sym in enumerate(cfg.pairs):
            ltf = generate_ohlcv(n, tf_e, start_ms - warm, seed=seed + k, start_price=30000.0 / (k + 1))
            data[sym] = (resample_ohlcv(ltf, tf_e, tf_t), ltf)
        return data
    from bot.data.history import load_csv, load_or_download, symbol_slug

    for sym in cfg.pairs:
        if source == "csv":
            d = Path(cfg.backtest.data_dir)
            ltf = load_csv(d / f"{symbol_slug(sym)}_{tf_e}.csv")
            htf = load_csv(d / f"{symbol_slug(sym)}_{tf_t}.csv")
            ltf = ltf[(ltf["timestamp"] >= start_ms) & (ltf["timestamp"] < end_ms)].reset_index(drop=True)
            htf = htf[htf["timestamp"] < end_ms].reset_index(drop=True)
        else:
            # trend data starts earlier so the slow EMA is warm on day one
            warm = cfg.timeframes.history_bars * timeframe_to_ms(tf_t)
            htf = load_or_download(cfg.backtest.data_dir, cfg.exchange.id, cfg.exchange.market_type, sym, tf_t,
                                   start_ms - warm, end_ms, cfg.exchange.hostname)
            ltf = load_or_download(cfg.backtest.data_dir, cfg.exchange.id, cfg.exchange.market_type, sym, tf_e,
                                   start_ms, end_ms, cfg.exchange.hostname)
        log.info("%s: %d %s candles, %d %s candles", sym, len(ltf), tf_e, len(htf), tf_t)
        data[sym] = (htf, ltf)
    return data


def cmd_backtest(cfg, args) -> int:
    from bot.backtest.engine import Backtester
    from bot.backtest.metrics import format_metrics

    start = args.start or cfg.backtest.start
    end = args.end or cfg.backtest.end
    data = _load_data(cfg, args.source, start, end, args.seed)
    result = Backtester(cfg).run(data)

    out = Path(cfg.backtest.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = f"{args.source}_{start}_{end}".replace(":", "")
    trades_path = out / f"trades_{stamp}.csv"
    import pandas as pd

    pd.DataFrame([asdict(t) for t in result.trades]).to_csv(trades_path, index=False)
    result.equity.to_csv(out / f"equity_{stamp}.csv", header=["equity"])
    (out / f"metrics_{stamp}.json").write_text(json.dumps(result.metrics | {"counts": result.signal_counts},
                                                          indent=2, default=str))

    print(f"\nBacktest {start} -> {end} | source: {args.source} | pairs: {', '.join(cfg.pairs)}")
    if args.source == "synthetic":
        print("NOTE: synthetic random-walk data. These numbers only prove the code runs; they mean nothing.")
    print(format_metrics(result.metrics))
    print(f"  Signal funnel: {result.signal_counts}")
    print(f"  Max leverage used: {result.max_leverage_used:.2f}x (cap {cfg.risk.max_leverage}x)")
    for k in result.kill_events:
        print(f"  KILL SWITCH: {k}")
    print(f"  Reports written to {out.resolve()}")
    print("\n" + BACKTEST_WARNING)
    return 0


def cmd_telegram_chatid(cfg) -> int:
    """Print chat ids of everyone who has messaged the bot, to fill telegram.allowed_chat_ids."""
    import os
    import urllib.request

    token = os.environ.get(cfg.telegram.token_env)
    if not token:
        print(f"${cfg.telegram.token_env} is not set (put it in .env)")
        return 2
    with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/getUpdates", timeout=20) as resp:
        updates = json.loads(resp.read().decode())["result"]
    chats = {}
    for u in updates:
        msg = u.get("message") or u.get("channel_post") or u.get("my_chat_member") or {}
        chat = msg.get("chat")
        if chat:
            chats[chat["id"]] = chat.get("username") or chat.get("title") or chat.get("first_name") or ""
    if not chats:
        print("No messages found. Open your bot in Telegram, press Start or send any message, then run this again.")
        return 1
    for cid, name in chats.items():
        print(f"chat id {cid}  ({name})")
    print("Add the id to telegram.allowed_chat_ids in config/config.yaml and set telegram.enabled: true")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bot", description="Paper-trading crypto futures bot")
    parser.add_argument("command", choices=["run", "backtest", "check-config", "telegram-chatid"])
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--source", choices=["exchange", "csv", "synthetic"], default="exchange",
                        help="backtest data source")
    parser.add_argument("--start", help="backtest start date (default from config)")
    parser.add_argument("--end", help="backtest end date (default from config)")
    parser.add_argument("--seed", type=int, default=7, help="seed for synthetic data")
    args = parser.parse_args(argv)
    # config/, .env, data/ and logs/ are relative to the project folder, wherever the bot is started from
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute() and not cfg_path.exists():
        cfg_path = PROJECT_ROOT / cfg_path
    args.config = str(cfg_path.resolve())
    os.chdir(PROJECT_ROOT)

    load_dotenv()
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2
    setup_logging(cfg.logging)

    if args.command == "check-config":
        print(f"Config OK. Mode: {cfg.mode.trading_mode} | pairs: {', '.join(cfg.pairs)} | risk {cfg.risk.risk_per_trade_pct}% "
              f"| leverage cap {cfg.risk.max_leverage}x | daily loss halt {cfg.risk.max_daily_loss_pct}%")
        return 0
    if args.command == "backtest":
        return cmd_backtest(cfg, args)
    if args.command == "telegram-chatid":
        return cmd_telegram_chatid(cfg)

    from bot.app import TradingBot

    try:
        asyncio.run(TradingBot(cfg).run())
    except KeyboardInterrupt:
        log.info("stopped by user")
    except ConfigError as e:
        log.error("Config error: %s", e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
