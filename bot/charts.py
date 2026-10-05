"""PNG charts for Telegram: trade execution charts and the account dashboard.

Colours come from the reference data-viz palette (dark surface). Green/red are the
reserved status colours (good/critical) and always appear with a text label, so
meaning never depends on colour alone. One y-axis per panel, thin marks, muted grid.
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#1a1a19"
TEXT = "#ffffff"
TEXT_2 = "#c3c2b7"
MUTED = "#6e6d68"
GRID = "#2c2c2a"
SERIES = "#3987e5"      # categorical slot 1, dark step
GOOD = "#0ca30c"        # status good
CRITICAL = "#d03b3b"    # status critical
NEUTRAL = "#8f8e88"


def _fig(rows: int = 1, height: float = 4.6, ratios=None):
    fig, axes = plt.subplots(rows, 1, figsize=(8, height), dpi=130, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": ratios} if ratios else None, squeeze=False)
    for ax in axes[:, 0]:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT_2, labelsize=8, length=0)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    return fig, axes[:, 0]


def _time_axis(ax) -> None:
    loc = mdates.AutoDateLocator(minticks=3, maxticks=6, tz="UTC")
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc, tz="UTC"))
    ax.xaxis.get_offset_text().set_color(MUTED)
    ax.xaxis.get_offset_text().set_fontsize(7.5)


def _right_labels(ax, items: list[tuple[str, float]], min_gap: float = 0.05) -> None:
    """Label horizontal levels at the right edge, nudged apart so none overlap.
    Call after the y-limits are final."""
    lo, hi = ax.get_ylim()
    pts = sorted(((y - lo) / (hi - lo), text) for text, y in items)
    placed: list[float] = []
    for frac, _ in pts:
        placed.append(max(frac, placed[-1] + min_gap) if placed else frac)
    overflow = (placed[-1] - 0.98) if placed else 0
    if overflow > 0:  # shift the stack down if it ran off the top
        placed = [f - overflow for f in placed]
    for (frac, text), pos in zip(pts, placed):
        ax.annotate(text, xy=(1.0, pos), xycoords="axes fraction", xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=TEXT_2)


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return buf.getvalue()


def _price_fmt(x: float) -> str:
    return f"{x:,.2f}" if abs(x) >= 1 else f"{x:.5g}"


def trade_chart(candles: pd.DataFrame, side: str, entry: float, stop: float, take_profit: float,
                partial_tp: float | None, entry_ms: int, title: str,
                exit_ms: int | None = None, exit_price: float | None = None, result: str | None = None) -> bytes:
    """Candles with entry, stop-loss and take-profit levels, plus entry/exit markers."""
    df = candles.copy()
    t = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    fig, (ax,) = _fig()
    width = (t.iloc[1] - t.iloc[0]) * 0.6 if len(t) > 1 else pd.Timedelta(minutes=9)
    up = df["close"] >= df["open"]
    for mask, color in ((up, GOOD), (~up, CRITICAL)):
        ax.vlines(t[mask], df["low"][mask], df["high"][mask], color=color, linewidth=0.8, alpha=0.75)
        ax.bar(t[mask], (df["close"] - df["open"])[mask].abs().clip(lower=1e-12), bottom=df[["open", "close"]].min(axis=1)[mask],
               width=width, color=color, alpha=0.75, linewidth=0)

    x_end = t.iloc[-1]
    levels = [("Entry", entry, SERIES, "-"), ("Stop-loss", stop, CRITICAL, "--"), ("Take-profit", take_profit, GOOD, "--")]
    if partial_tp is not None:
        levels.append(("Partial TP", partial_tp, NEUTRAL, ":"))
    for label, y, color, style in levels:
        ax.axhline(y, color=color, linewidth=1.6, linestyle=style)

    ax.axhspan(min(entry, stop), max(entry, stop), color=CRITICAL, alpha=0.07, linewidth=0)
    ax.axhspan(min(entry, take_profit), max(entry, take_profit), color=GOOD, alpha=0.07, linewidth=0)

    t_entry = pd.to_datetime(entry_ms, unit="ms", utc=True)
    ax.scatter([t_entry], [entry], s=90, marker="^" if side == "long" else "v", color=SERIES,
               edgecolors=SURFACE, linewidths=2, zorder=5)
    if exit_ms is not None and exit_price is not None:
        ax.scatter([pd.to_datetime(exit_ms, unit="ms", utc=True)], [exit_price], s=80, marker="X",
                   color=TEXT, edgecolors=SURFACE, linewidths=2, zorder=5)
        ax.annotate(f"Exit {_price_fmt(exit_price)}", (pd.to_datetime(exit_ms, unit="ms", utc=True), exit_price),
                    xytext=(-8, 10), textcoords="offset points", ha="right", fontsize=8, color=TEXT)

    lo = min(df["low"].min(), stop, take_profit)
    hi = max(df["high"].max(), stop, take_profit)
    pad = (hi - lo) * 0.06
    ax.set_ylim(lo - pad, hi + pad)
    _right_labels(ax, [(f"{label} {_price_fmt(y)}", y) for label, y, _, _ in levels])
    ax.set_xlim(t.iloc[0], x_end + width * 4)
    _time_axis(ax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _price_fmt(v)))
    head = f"{title}  ·  {side.upper()}" + (f"  ·  {result}" if result else "")
    ax.set_title(head, color=TEXT, fontsize=11, loc="left", pad=10)
    ax.text(0, -0.13, "15m candles, UTC  ·  ▲/▼ entry  ·  X exit  ·  paper trade", transform=ax.transAxes,
            fontsize=7.5, color=MUTED)
    return _png(fig)


def _avatar_image(avatar: bytes | None, name: str, size: int = 256):
    """Round profile picture as an RGBA array; initials on a coloured disc if no picture."""
    from PIL import Image, ImageDraw, ImageFont

    if avatar:
        try:
            img = Image.open(io.BytesIO(avatar)).convert("RGB")
            side = min(img.size)
            left, top = (img.width - side) // 2, (img.height - side) // 2
            img = img.crop((left, top, left + side, top + side)).resize((size, size))
        except Exception:  # noqa: BLE001 - unreadable picture: fall back to initials
            img = None
    else:
        img = None
    if img is None:
        img = Image.new("RGB", (size, size), SERIES)
        initials = "".join(w[0] for w in name.split()[:2]).upper() or "?"
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(size * 0.4))
        except OSError:
            try:
                font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"),
                                          int(size * 0.4))
            except OSError:
                font = ImageFont.load_default()
        draw.text((size / 2, size / 2), initials, fill="#ffffff", font=font, anchor="mm")
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
    out = img.convert("RGBA")
    out.putalpha(mask)
    return np.asarray(out)


def account_chart(equity: pd.Series, r_multiples: list[float], capital: float, deposits: pd.Series | None,
                  title: str, user_name: str | None = None, avatar: bytes | None = None) -> bytes:
    """Header with the holder's picture and name, then two panels: equity over time and R per trade."""
    fig, (ax1, ax2) = _fig(rows=2, height=7.0, ratios=[3, 2])

    if len(equity) >= 2:
        ax1.plot(equity.index, equity.values, color=SERIES, linewidth=2)
        ax1.fill_between(equity.index, equity.values, equity.min(), color=SERIES, alpha=0.10, linewidth=0)
        ax1.axhline(capital, color=MUTED, linewidth=1, linestyle="--")
        ax1.annotate(f"Capital in {capital:,.0f}", xy=(0.0, capital), xycoords=("axes fraction", "data"),
                     xytext=(4, 4), textcoords="offset points", fontsize=8, color=TEXT_2)
        last = equity.iloc[-1]
        ax1.scatter([equity.index[-1]], [last], s=40, color=SERIES, edgecolors=SURFACE, linewidths=2, zorder=5)
        ax1.annotate(f"{last:,.2f}", (equity.index[-1], last), xytext=(-6, 8), textcoords="offset points",
                     ha="right", fontsize=9, color=TEXT)
        if deposits is not None and len(deposits):
            for ts in deposits.index:
                ax1.axvline(ts, color=NEUTRAL, linewidth=1, linestyle=":")
        _time_axis(ax1)
        ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    else:
        ax1.text(0.5, 0.5, "Equity history appears after the bot has run for a few minutes",
                 transform=ax1.transAxes, ha="center", va="center", color=MUTED, fontsize=9)
        ax1.set_xticks([])
        ax1.set_yticks([])
    note = ""
    if deposits is not None and len(deposits):
        note = f"   ·   dotted lines = top-ups ({len(deposits)}, +{deposits.sum():,.0f} USDT)"
    ax1.set_title("Equity (USDT)" + note, color=TEXT_2, fontsize=9, loc="left", pad=6)

    if r_multiples:
        r = np.array(r_multiples[-30:])
        x = np.arange(1, len(r) + 1)
        colors = [GOOD if v > 0 else CRITICAL for v in r]
        ax2.bar(x, r, color=colors, width=0.6, linewidth=0)
        ax2.axhline(0, color=MUTED, linewidth=1)
        ax2.set_xticks(x if len(x) <= 15 else x[::max(1, len(x) // 10)])
        ax2.set_xlim(0.3, len(r) + 0.7)
        ax2.set_title(f"Result per trade, last {len(r)}  (green = win, red = loss)", color=TEXT_2,
                      fontsize=9, loc="left", pad=6)
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.1f}R"))
    else:
        ax2.text(0.5, 0.5, "No closed trades yet", transform=ax2.transAxes, ha="center", va="center",
                 color=MUTED, fontsize=9)
        ax2.set_xticks([])
        ax2.set_yticks([])
    ax2.text(0, -0.28, "Paper account. Past results do not guarantee future results.", transform=ax2.transAxes,
             fontsize=7.5, color=MUTED)
    fig.tight_layout(h_pad=2.2, rect=(0, 0, 1, 0.86))
    # header: round avatar + name + account code
    av = fig.add_axes((0.035, 0.885, 0.09, 0.1))
    av.imshow(_avatar_image(avatar, user_name or "?"))
    av.set_axis_off()
    fig.text(0.14, 0.95, user_name or "Paper account", color=TEXT, fontsize=14, va="center", weight="bold")
    fig.text(0.14, 0.905, title, color=TEXT_2, fontsize=9.5, va="center")
    return _png(fig)


EMA_FAST = SERIES        # categorical slot 1 (dark)
EMA_SLOW = "#d95926"     # categorical slot 2 (dark)


def analysis_chart(ltf: pd.DataFrame, symbol: str, verdict: str, ema_fast: int, ema_slow: int,
                   support: float | None, resistance: float | None, setup: dict | None) -> bytes:
    """Recent candles with the two EMAs, nearest swing support/resistance and, when a
    trend bias exists, the entry/stop/target the rules would use."""
    df = ltf.tail(96)
    t = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    fig, (ax,) = _fig()
    width = (t.iloc[1] - t.iloc[0]) * 0.6 if len(t) > 1 else pd.Timedelta(minutes=9)
    up = df["close"] >= df["open"]
    for mask, color in ((up, GOOD), (~up, CRITICAL)):
        ax.vlines(t[mask], df["low"][mask], df["high"][mask], color=color, linewidth=0.8, alpha=0.6)
        ax.bar(t[mask], (df["close"] - df["open"])[mask].abs().clip(lower=1e-12),
               bottom=df[["open", "close"]].min(axis=1)[mask], width=width, color=color, alpha=0.6, linewidth=0)
    ax.plot(t, df["ema_fast"], color=EMA_FAST, linewidth=2, label=f"EMA{ema_fast}")
    ax.plot(t, df["ema_slow"], color=EMA_SLOW, linewidth=2, label=f"EMA{ema_slow}")

    ys = [df["low"].min(), df["high"].max()]
    labels: list[tuple[str, float]] = []
    for label, y, color, style in (("Resistance", resistance, TEXT_2, ":"), ("Support", support, TEXT_2, ":")):
        if y is not None:
            ax.axhline(y, color=color, linewidth=1.2, linestyle=style)
            labels.append((f"{label} {_price_fmt(y)}", y))
            ys.append(y)
    if setup:
        for label, key, color, style in (("Stop-loss", "stop", CRITICAL, "--"), ("Take-profit", "take_profit", GOOD, "--")):
            y = setup[key]
            ax.axhline(y, color=color, linewidth=1.4, linestyle=style)
            labels.append((f"{label} {_price_fmt(y)}", y))
            ys.append(y)
    last = float(df["close"].iloc[-1])
    ax.scatter([t.iloc[-1]], [last], s=40, color=TEXT, edgecolors=SURFACE, linewidths=2, zorder=5)
    ax.annotate(f"{_price_fmt(last)}", (t.iloc[-1], last), xytext=(-6, 9), textcoords="offset points",
                ha="right", fontsize=9, color=TEXT)

    pad = (max(ys) - min(ys)) * 0.06
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    if labels:
        _right_labels(ax, labels)
    ax.set_xlim(t.iloc[0], t.iloc[-1] + width * 3)
    _time_axis(ax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _price_fmt(v)))
    leg = ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=TEXT_2)
    for h in leg.legend_handles:
        h.set_linewidth(2)
    ax.set_title(f"{symbol}  ·  {verdict}", color=TEXT, fontsize=11, loc="left", pad=10)
    ax.text(0, -0.13, "15m candles, UTC  ·  live exchange data  ·  levels are what the rules would use, not advice",
            transform=ax.transAxes, fontsize=7.5, color=MUTED)
    return _png(fig)


def _header(fig, user_name: str | None, subtitle: str, avatar: bytes | None) -> None:
    av = fig.add_axes((0.035, 0.885, 0.09, 0.1))
    av.imshow(_avatar_image(avatar, user_name or "?"))
    av.set_axis_off()
    fig.text(0.14, 0.95, user_name or "Paper account", color=TEXT, fontsize=14, va="center", weight="bold")
    fig.text(0.14, 0.905, subtitle, color=TEXT_2, fontsize=9.5, va="center")


def profit_chart(closed_at: list[int], pnls: list[float], subtitle: str, user_name: str | None = None,
                 avatar: bytes | None = None) -> bytes:
    """Cumulative realized profit (USDT) over time, and profit/loss per trade. The newest
    trade is outlined so a user can find the one that just closed."""
    fig, (ax1, ax2) = _fig(rows=2, height=7.0, ratios=[3, 2])
    if pnls:
        t = pd.to_datetime(closed_at, unit="ms", utc=True)
        cum = np.cumsum(pnls)
        xs = [t[0] - pd.Timedelta(minutes=1)] + list(t)
        ys = [0.0] + list(cum)
        ax1.step(xs, ys, where="post", color=SERIES, linewidth=2)
        ax1.axhline(0, color=MUTED, linewidth=1, linestyle="--")
        total = float(cum[-1])
        ax1.scatter([xs[-1]], [total], s=40, color=SERIES, edgecolors=SURFACE, linewidths=2, zorder=5)
        ax1.annotate(f"{total:+,.2f} USDT", (xs[-1], total), xytext=(-6, 9), textcoords="offset points",
                     ha="right", fontsize=10, color=TEXT, weight="bold")
        _time_axis(ax1)
        ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+,.0f}"))

        recent = np.array(pnls[-30:])
        x = np.arange(1, len(recent) + 1)
        colors = [GOOD if v > 0 else CRITICAL for v in recent]
        bars = ax2.bar(x, recent, color=colors, width=0.6, linewidth=0)
        bars[-1].set_edgecolor(TEXT)
        bars[-1].set_linewidth(1.5)
        last = recent[-1]
        ax2.annotate(f"latest {last:+,.2f}", (x[-1], last), xytext=(0, 5 if last >= 0 else -12),
                     textcoords="offset points", ha="center", fontsize=8, color=TEXT)
        ax2.axhline(0, color=MUTED, linewidth=1)
        ax2.set_xlim(0.3, len(recent) + 0.7)
        ax2.set_xticks(x if len(x) <= 15 else x[::max(1, len(x) // 10)])
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+,.0f}"))
        wins = int((np.array(pnls) > 0).sum())
        ax2.set_title(f"Profit per trade, USDT (last {len(recent)})  ·  {wins}/{len(pnls)} winners  "
                      f"(green = profit, red = loss)", color=TEXT_2, fontsize=9, loc="left", pad=6)
    else:
        for ax, msg in ((ax1, "Profit appears here after your first closed trade"), (ax2, "No closed trades yet")):
            ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    ax1.set_title("Total realized profit (USDT, after fees)", color=TEXT_2, fontsize=9, loc="left", pad=6)
    ax2.text(0, -0.28, "Paper account: simulated trades on real prices. Past results do not guarantee future results.",
             transform=ax2.transAxes, fontsize=7.5, color=MUTED)
    fig.tight_layout(h_pad=2.2, rect=(0, 0, 1, 0.86))
    _header(fig, user_name, subtitle, avatar)
    return _png(fig)
