"""Telegram/console message formatting (Telegram HTML parse mode)."""
from __future__ import annotations

from html import escape

from bot.execution.models import LONG, ClosedTrade, Event, Position
from bot.risk.manager import SizingResult
from bot.signals.engine import Signal
from bot.utils import fmt_money, fmt_price, fmt_ts

DISCLAIMER = "Signals are probabilistic. Many will hit the stop. Past results do not guarantee future results."


def _stars(n: int) -> str:
    return "★" * n + "☆" * (5 - n)


def _pct(a: float, b: float) -> str:
    return f"{(a - b) / b * 100:+.2f}%" if b else "-"


def signal_alert(sig: Signal, sizing: SizingResult, equity: float, auto_execute: bool, risk_pct: float,
                 stake_line: str | None = None) -> str:
    icon = "🟢" if sig.side == LONG else "🔴"
    mode = "PAPER AUTO-EXECUTE" if auto_execute else "SIGNAL ONLY"
    base = sig.symbol.split("/")[0]
    lines = [
        f"{icon} <b>{sig.side.upper()} {escape(sig.symbol)}</b>",
        f"Confidence: {_stars(sig.confidence)} ({sig.confidence}/5) | Mode: {mode}",
        "",
        f"Entry: <code>{fmt_price(sig.entry)}</code>",
        f"Stop-loss: <code>{fmt_price(sig.stop)}</code> ({_pct(sig.stop, sig.entry)}, {escape(sig.stop_basis)})",
        f"Take-profit: <code>{fmt_price(sig.take_profit)}</code> ({_pct(sig.take_profit, sig.entry)}, "
        f"{escape(sig.target_basis)})",
    ]
    if sig.partial_tp is not None:
        lines.append(f"Partial TP: <code>{fmt_price(sig.partial_tp)}</code>, then stop to breakeven + trailing")
    lines += [
        f"R:R: <b>1:{sig.reward_risk:.2f}</b>",
        f"Size: <code>{sizing.qty:g}</code> {escape(base)} (notional {fmt_money(sizing.notional)}, "
        f"{sizing.leverage:.2f}x equity)",
        f"Risk if stopped: {fmt_money(sizing.risk_amount)} (≤ {risk_pct:g}% of {fmt_money(equity)})",
    ]
    if sizing.capped_by:
        lines.append(f"Size reduced by {escape(sizing.capped_by)}")
    if stake_line:
        lines.append(f"💵 {stake_line}")
    lines += ["", "<b>Why:</b>"] + [f"• {escape(r)}" for r in sig.reasons]
    if sig.missing:
        lines.append(f"Not confirmed: {escape(', '.join(sig.missing))}")
    lines += ["", f"<i>{DISCLAIMER}</i>"]
    return "\n".join(lines)


def fill_confirmation(pos: Position, fill: float, expected: float, slippage_pct: float) -> str:
    return (f"✅ <b>PAPER FILL</b> {pos.side.upper()} {escape(pos.symbol)}\n"
            f"Filled {pos.qty:g} @ <code>{fmt_price(fill)}</code> (expected {fmt_price(expected)}, "
            f"slippage {slippage_pct:.3f}%)\n"
            f"Stop <code>{fmt_price(pos.stop)}</code> | TP <code>{fmt_price(pos.take_profit)}</code> | "
            f"risk {fmt_money(pos.risk_amount)}\nID: <code>{escape(pos.id)}</code>")


def position_event(ev: Event) -> str | None:
    p = ev.position
    if ev.kind == "partial":
        return (f"💰 <b>Partial take-profit</b> {escape(p.symbol)} {p.side}\n"
                f"Closed {ev.qty:g} @ <code>{fmt_price(ev.price)}</code>, P&amp;L {fmt_money(ev.pnl or 0)}\n"
                f"Remaining {p.qty:g}, stop now <code>{fmt_price(p.stop)}</code>")
    if ev.kind == "breakeven":
        return f"🛡 {escape(p.symbol)}: stop moved to breakeven <code>{fmt_price(p.stop)}</code>"
    if ev.kind == "closed" and ev.trade:
        return trade_closed(ev.trade)
    return None


def trade_closed(t: ClosedTrade) -> str:
    icon = "✅" if t.net_pnl > 0 else "❌"
    return (f"{icon} <b>CLOSED</b> {escape(t.symbol)} {t.side} ({escape(t.exit_reason)})\n"
            f"Entry {fmt_price(t.entry)} → exit {fmt_price(t.exit_price)}\n"
            f"Net P&amp;L <b>{fmt_money(t.net_pnl)}</b> ({t.r_multiple:+.2f}R), fees {fmt_money(t.fees)}")


def status(equity: float, balance: float, start_equity: float, daily_pnl: float, daily_pct: float,
           drawdown: float, positions: list[Position], marks: dict[str, float], flags: list[str],
           auto_execute: bool, owner: str | None = None) -> str:
    lines = [
        "<b>Status</b> (paper account)" + (f"\n👤 {escape(owner)}" if owner else ""),
        f"Mode: {'PAPER AUTO-EXECUTE' if auto_execute else 'SIGNAL ONLY'}",
        f"State: {escape(', '.join(flags)) if flags else 'active'}",
        f"Equity: <b>{fmt_money(equity)}</b> (realized {fmt_money(balance)}, "
        f"total {_pct(equity, start_equity)})",
        f"Today: {fmt_money(daily_pnl)} ({daily_pct:+.2f}%) | drawdown from peak {drawdown:.2f}%",
        "",
        f"<b>Open positions ({len(positions)})</b>",
    ]
    if not positions:
        lines.append("none")
    for p in positions:
        mark = marks.get(p.symbol, p.entry)
        lines.append(f"• {escape(p.symbol)} {p.side} {p.qty:g} @ {fmt_price(p.entry)} | mark {fmt_price(mark)} | "
                     f"uPnL {fmt_money(p.unrealized(mark))} ({p.r_now(mark):+.2f}R) | "
                     f"SL {fmt_price(p.stop)} TP {fmt_price(p.take_profit)}")
    return "\n".join(lines)


def history(rows: list) -> str:
    if not rows:
        return "No closed trades yet."
    lines = [f"<b>Last {len(rows)} trades</b>"]
    for r in rows:
        icon = "✅" if r["net_pnl"] > 0 else "❌"
        lines.append(f"{icon} {fmt_ts(r['closed_at'])} {escape(r['symbol'])} {r['side']} "
                     f"{fmt_money(r['net_pnl'])} ({r['r_multiple']:+.2f}R) - {escape(r['exit_reason'])}")
    wins = sum(1 for r in rows if r["net_pnl"] > 0)
    lines.append(f"\nWin rate (these {len(rows)}): {wins / len(rows) * 100:.0f}% | "
                 f"total {fmt_money(sum(r['net_pnl'] for r in rows))}")
    return "\n".join(lines)


HELP_BODY = """<b>Your own paper account</b>
/account - your account card, profile picture and charts
/status - your equity, daily P&amp;L, open positions
/history [n] - your recent closed trades
/amount - pick your trade amount: 1%, 5% or 10% of your balance
/profit - your profit chart and totals
/topup &lt;amount&gt; - add paper USDT to your account
/autoexec on|off - auto-trade signals on your account
/pause, /resume - pause or resume your account
/resetkill CONFIRM - clear your account's kill switch
/papergate - your progress toward the paper-trading requirement

<b>Market</b>
/scanner - live board: what the continuous scan sees on every pair
/analyze [coin|all] - live analysis with chart (or tap 📊 Analyze)
/settings - bot-wide strategy and risk settings

<b>Owner only</b>
/settings &lt;key&gt; &lt;value&gt; - change a bot-wide setting
/kill [reason] - bot-wide kill switch (also cleared with /resetkill CONFIRM)"""


def account_card(user, account_id: str, exchange: str, bot_name: str, equity: float, balance: float,
                 capital: float, starting: float, deposits: float, open_positions: int, open_pnl: float,
                 daily_pnl: float, drawdown: float, r_multiples: list[float], pnls: list[float],
                 auto_execute: bool) -> str:
    total = equity - capital
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    who = "Unknown"
    role = ""
    if user is not None:
        who = escape(user.display)
        role = " · owner" if user.is_owner else " · viewer"
    lines = [
        f"💼 <b>{escape(bot_name)} account</b>",
        f"👤 {who}{role}",
        f"🆔 <code>{escape(account_id)}</code> · PAPER · {escape(exchange)} prices",
        f"Mode: {'auto-execute' if auto_execute else 'signal only'}",
        "",
        f"Equity: <b>{fmt_money(equity)}</b>",
        f"Balance (realized): {fmt_money(balance)}",
        f"Open P&amp;L: {fmt_money(open_pnl)} on {open_positions} position(s)",
        f"Capital in: {fmt_money(capital)} (start {fmt_money(starting)} + top-ups {fmt_money(deposits)})",
        f"Total P&amp;L: <b>{fmt_money(total)}</b> ({total / capital * 100:+.2f}%)" if capital else "",
        f"Today: {fmt_money(daily_pnl)} | drawdown from peak {drawdown:.2f}%",
        "",
        f"Closed trades: {n}"
        + (f" | win rate {wins / n * 100:.0f}% | avg {sum(r_multiples) / n:+.2f}R" if n else ""),
        "",
        "<i>Paper account: simulated trades on real prices. No real money.</i>",
    ]
    return "\n".join(line for line in lines if line is not None)


def settlement(t: ClosedTrade, balance: float, available: float) -> str:
    """How the closed trade settles against the balance: amount back plus profit (or minus loss)."""
    lines = []
    if t.margin:
        back = t.margin + t.net_pnl
        verb = "profit" if t.net_pnl > 0 else "loss"
        lines.append(f"💵 Trade amount {t.margin:,.2f} {'+' if t.net_pnl >= 0 else '−'} {verb} "
                     f"{abs(t.net_pnl):,.2f} = <b>{back:,.2f} USDT</b> returned to your balance")
    lines.append(f"💼 Balance {balance:,.2f} USDT · available {available:,.2f} USDT")
    return "\n".join(lines)


def profit_summary(acc, pnls: list[float], balance: float, capital: float) -> str:
    n = len(pnls)
    total = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    lines = [f"📈 <b>Profit</b> · {escape(acc.display)} · <code>{escape(acc.code)}</code>",
             f"Total realized profit: <b>{fmt_money(total)}</b> over {n} trade(s)"]
    if n:
        lines += [f"Winners {len(wins)} (+{sum(wins):,.2f}) · losers {len(losses)} ({sum(losses):,.2f})",
                  f"Best {max(pnls):+,.2f} · worst {min(pnls):+,.2f} USDT"]
    lines += [f"Balance {fmt_money(balance)} · capital in {fmt_money(capital)}",
              f"Trade amount: {acc.stake_pct:g}% of your balance per trade",
              "", "<i>Paper trading: simulated on real prices. Past results do not guarantee future results.</i>"]
    return "\n".join(lines)


def welcome(acc, bot_name: str, starting_equity: float) -> str:
    return (f"👋 Welcome to {escape(bot_name)}, {escape(acc.name)}!\n"
            f"Your own paper account <code>{escape(acc.code)}</code> was created with "
            f"<b>{fmt_money(starting_equity)}</b> of simulated USDT.\n"
            f"Auto-execute is <b>{'ON' if acc.auto_execute else 'OFF'}</b>: "
            + ("signals open paper trades on your account automatically." if acc.auto_execute
               else "you receive signal alerts only.")
            + "\nTap 💼 Account to see it, /autoexec on|off to change it, /topup to add paper funds.\n"
            f"<i>No real money is involved. {DISCLAIMER}</i>")


def help_text(bot_name: str) -> str:
    return f"<b>{escape(bot_name)}</b> (paper trading)\n" + HELP_BODY


def analysis(a, change_24h: float | None = None) -> str:
    """Format a bot.signals.analysis.Analysis for Telegram."""
    icon = {"signal": "🟢" if (a.setup or {}).get("side") == LONG else "🔴", "wait": "⏳", "none": "⚪"}[a.verdict]
    if a.verdict == "signal" and a.setup is None:
        icon = "🟢"
    trend = {"long": "UPTREND", "short": "DOWNTREND", None: "NO CLEAR TREND"}[a.bias]
    chg = f" | 24h {change_24h:+.2f}%" if change_24h is not None else ""
    l, h = a.ltf, a.htf
    lines = [
        f"📊 <b>{escape(a.symbol)}</b> analysis",
        f"Price <code>{fmt_price(a.price)}</code>{chg}",
        f"Last closed {a.entry_tf} candle: {fmt_ts(a.bar_close)}",
        "",
        f"{icon} <b>{escape(a.verdict_text)}</b>",
    ]
    if not a.checks:
        lines += ["", f"<i>{DISCLAIMER}</i>"]
        return "\n".join(lines)
    lines += [
        "",
        f"<b>{a.trend_tf} trend:</b> {trend} (ADX {h['adx']:.1f})",
        f"<b>{a.entry_tf} readings:</b> RSI {l['rsi']:.1f} | MACD hist {l['macd_hist']:+.4g} | "
        f"ADX {l['adx']:.1f} | ATR {fmt_price(l['atr'])} ({l['atr'] / l['close'] * 100:.2f}%)",
        "",
        f"<b>Checks: {a.score}/5</b>",
    ]
    lines += [f"{'✅' if c.passed else '❌'} {escape(c.name)}: {escape(c.detail)}" for c in a.checks]
    lines.append(f"Entry trigger: {escape(a.trigger) if a.trigger else 'none in the last few candles'}")
    lines.append(f"Support <code>{fmt_price(a.support)}</code> | resistance <code>{fmt_price(a.resistance)}</code>")
    if a.setup:
        s = a.setup
        lines += [
            "",
            f"<b>{'Signal levels' if a.verdict == 'signal' else 'If a ' + s['side'] + ' entry came now'}:</b>",
            f"Entry <code>{fmt_price(s['entry'])}</code> | SL <code>{fmt_price(s['stop'])}</code> | "
            f"TP <code>{fmt_price(s['take_profit'])}</code> | R:R 1:{s['rr']:.2f}",
        ]
    lines.append(f"Market filter: {'OK' if not a.filter_block else escape('blocked, ' + a.filter_block)}")
    lines += ["", f"<i>Analysis is a snapshot, not a prediction. {DISCLAIMER}</i>"]
    return "\n".join(lines)


def _missing_hint(a) -> str:
    """What is still needed before an entry, in plain words."""
    if not a.checks:  # analysis stopped early (e.g. not enough candle history)
        return a.verdict_text
    if a.verdict == "signal":
        return "all entry rules met on the last closed candle"
    if a.bias is None:
        return f"waiting for a clear {a.trend_tf} trend"
    need = [c.name for c in a.checks if not c.passed]
    parts = []
    if not a.trigger:
        parts.append(f"a fresh {a.entry_tf} trigger (MACD flip or EMA cross)")
    if need:
        parts.append("confirmation from " + ", ".join(need))
    if a.setup and a.setup["rr"] < 1.5:
        parts.append(f"more room to the target (R:R {a.setup['rr']:.2f})")
    if a.filter_block:
        parts.append(f"market filter to clear ({a.filter_block})")
    return "needs " + "; ".join(parts) if parts else a.verdict_text


def scanner_board(pairs: list[str], board: dict, marks: dict[str, float], last_scan_ms: int | None,
                  next_scan_ms: int, auto_execute: bool, open_positions: int, entry_tf: str) -> str:
    icons = {"signal": "🟢", "wait": "⏳", "none": "⚪"}
    lines = ["📡 <b>Live scanner</b>",
             f"Scans every closed {entry_tf} candle on all pairs and trades automatically when every rule is met.",
             f"Last scan: {fmt_ts(last_scan_ms) if last_scan_ms else 'starting…'} · next: {fmt_ts(next_scan_ms)}",
             f"Your account: auto-execute <b>{'ON' if auto_execute else 'OFF'}</b> · open trades {open_positions} "
             f"(managed on every live price tick)", ""]
    for sym in pairs:
        a = board.get(sym)
        base = escape(sym.split("/")[0])
        if a is None:
            lines.append(f"⚪ <b>{base}</b>: no data yet")
            continue
        price = marks.get(sym, a.price)
        trend = {"long": "uptrend", "short": "downtrend", None: "no trend"}[a.bias]
        score = f"checks {a.score}/5" if a.checks else "not enough data"
        lines.append(f"{icons[a.verdict]} <b>{base}</b> <code>{fmt_price(price)}</code> · {a.trend_tf} {trend} · "
                     f"{score}")
        lines.append(f"    {escape(_missing_hint(a))}")
    lines += ["", "🟢 entry · ⏳ trend present, waiting for the entry · ⚪ no trade conditions",
              "<i>Entries are only taken on closed candles, so a half-formed candle cannot trigger a trade.</i>"]
    return "\n".join(lines)
