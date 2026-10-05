# jamesbot

A crypto futures signal bot with multi-timeframe confluence, hard risk limits, a
Telegram interface, a backtester and paper trading. **This build trades on paper
only.** No real orders can be sent.

> **Disclaimer.** No strategy here is profitable by design, and none is claimed to be.
> Backtest and paper results do not guarantee future results. Every strategy loses
> trades, often several in a row. The risk module exists because losses are expected.
> With default settings, the 2025 backtest on Bybit BTC/ETH **lost money** (see below).

## Quick start (Windows, Python 3.10+)

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
copy config\config.example.yaml config\config.yaml
copy .env.example .env
.\.venv\Scripts\python -m bot check-config
.\.venv\Scripts\python -m bot backtest --source exchange      # real history, cached in data/history
.\.venv\Scripts\python -m bot run                             # live PAPER trading
.\.venv\Scripts\python -m pytest -q                           # 44 tests
```

The simplest way to start the bot is `python run_bot.py`, or VS Code's Run button on
`run_bot.py`. It switches to the project's `.venv` automatically. `python -m bot ...`
must be run from inside the project folder.

Paper trading uses public market data only, so no exchange API keys are needed.
The example config uses **Bybit** because Binance Futures was unreachable from the
development machine. Set `exchange.id: binanceusdm` to use Binance USDT-M futures.

## Telegram setup

1. Create a bot with [@BotFather](https://t.me/BotFather) and put the token in `.env`
   as `TELEGRAM_BOT_TOKEN=...`.
2. Send your bot a message, then run `.\.venv\Scripts\python -m bot telegram-chatid`
   to print your chat id.
3. In `config/config.yaml` set `telegram.enabled: true` and add that id to
   `allowed_chat_ids`. Commands from any other chat are ignored and logged.
4. Optional: to post alerts to a channel, add the bot as a channel admin and set
   `alert_chat_id` to the channel id (`-100...`).

| Command | What it does |
|---|---|
| `/analyze [coin\|all]` | Live analysis with every check explained (or tap 📊 Analyze) |
| `/account` | Your Telegram name, the paper account card, an equity chart and per-trade results |
| `/topup <amount>` | Add paper USDT (owner only). Counts as capital, not profit |
| `/status` | Equity, daily P&L, drawdown, open positions with stops and targets |
| `/history [n]` | Last n closed trades with R multiples |
| `/pause` / `/resume` | Stop / restart new alerts and entries. Open positions keep their stops |
| `/settings` | Show settings. `/settings risk 0.5`, `leverage 3`, `maxdaily 2`, `maxpos 2`, `minrr 2`, `autoexec on`, `pairs BTC/USDT,SOL/USDT` |
| `/kill [reason]` | Manual kill switch. Closes paper positions if `flatten_on_kill` is true |
| `/resetkill CONFIRM` | Clear the kill switch |
| `/papergate` | Progress toward the paper-trading requirement |

Settings changed in Telegram are validated against the same limits as the config
file and are saved, so they survive a restart.

## Per-user paper accounts

Every Telegram user who talks to the bot gets their own paper account, created on
first contact with `mode.starting_equity`. Each account has its own balance,
positions, trade history, top-ups, daily loss limit, drawdown kill switch, pause
and auto-execute switch. Signals are shared: each one is sized against each
account's own equity and delivered to that user's chat. The owner's earlier
single account is migrated automatically on the first start.

**Trade amount.** Each user picks a percentage of their balance per trade with
`/amount`: 1%, 5% or 10% (`mode.stake_pct_options`, default `mode.default_stake_pct`).
On a signal, auto-execute takes that percentage of the user's balance, recalculated for
every trade, as margin and opens a position of amount x leverage cap. On close the amount comes back plus the
profit or minus the loss, and the user gets the settlement and a profit chart
(`/profit`). The risk rules still apply: a position is made smaller if the amount would
risk more than `risk.risk_per_trade_pct` of the balance at the stop, and a trade is
skipped with a message if the available balance is below the amount.

The account card shows the user's Telegram profile picture when their privacy
settings allow bots to see it, and their initials otherwise.

Owner-only, bot-wide controls are `/settings <key> <value>` and `/kill`. A bot-wide
kill switch also trips on repeated API errors or abnormal slippage.

## Modes

- **Signal only** (`mode.auto_execute: false`, default): alerts with entry, stop,
  target, size, R:R, confidence and reasons. You decide what to do.
- **Paper auto-execute** (`autoexec on`): the bot opens and manages simulated
  positions with fees and slippage, and reports fills, partials and closes.
- **Live**: not implemented. Config validation rejects it. See
  `bot/execution/live.py` for what must be built first, and the paper gate
  (default 50 paper trades over 30 days) that must be met before it is enabled.

## How a signal is formed

A signal needs **all** of the following on a closed 15m candle:

1. **4h trend bias.** EMA50 above EMA200 with price above EMA200 (or the reverse
   for shorts), and ADX at least 20.
2. **A fresh 15m trigger** within 3 bars: MACD histogram flip or EMA21/55 cross.
3. **Confluence.** A 1-5 confidence score from five checks covering at least three
   categories:

   | Check | Category |
   |---|---|
   | 4h trend (always true at this point) | trend |
   | 15m EMAs aligned and ADX at least 20 | trend |
   | RSI in the directional zone with MACD agreeing, or RSI divergence | momentum |
   | Close between the Bollinger mid and the outer band, not beyond it | volatility/structure |
   | Volume at least 1.2x its 20-bar average | volume |

4. **A valid stop and R:R.** The stop is the wider of 1.5x ATR or just beyond the
   last swing low/high. The target is 2.5R, capped below the nearest opposing swing
   level. Signals under 1:1.5 are rejected.
5. **Market filters.** No signal inside a news blackout window (`filters.news_events`,
   which you must keep up to date), during an ATR spike, or on an abnormal candle.

## Risk rules

| Rule | Default | Hard ceiling in code |
|---|---|---|
| Risk per trade, fees and slippage included | 1% | 2% |
| Leverage cap on total notional | 3x | 10x |
| Daily loss halt, mark-to-market, until next UTC day | 3% | 10% |
| Max open positions / per symbol | 3 / 1 | 10 |
| Max notional per asset | 150% of equity | leverage x 100% |
| Minimum R:R | 1.5 | at least 1.0 |
| Drawdown kill switch | 15% | 50% |
| Kill switch on API errors / slippage | 5 in a row / 0.5% | |

The ceilings live in `bot/risk/limits.py`. Every position gets a stop at entry.
Size comes from stop distance and risk %, then is capped by leverage and exposure,
then rounded down to the exchange lot step. After +1R the bot closes 50%, moves
the stop to breakeven, and trails the rest by 2x ATR. The kill switch survives
restarts and needs `/resetkill CONFIRM`.

## Backtesting

```powershell
.\.venv\Scripts\python -m bot backtest --source exchange --start 2025-01-01 --end 2026-01-01
.\.venv\Scripts\python -m bot backtest --source csv        # data/history/<SYMBOL>_<tf>.csv
.\.venv\Scripts\python -m bot backtest --source synthetic  # random data: code check only
```

The backtester uses the same signal engine, risk manager and paper executor as the
live bot. Signals fill at the next bar's open. A 4h candle is used only after it
closes. Swing levels appear only once confirmed. When a bar touches both stop and
target, the stop is assumed to hit first. A test verifies that removing future
bars never changes a signal. Reports go to `reports/`.

**Default-settings result, Bybit BTC + ETH, 2025:** 87 trades, 50.6% win rate,
-0.10R average, profit factor 0.82, -7.4% return, Sharpe -0.88. The 15% drawdown
kill switch tripped on 2025-08-27 and the bot stopped trading for the rest of the
year. Fees were larger than the net loss. Treat this as a starting point to study,
not a strategy to run with money. If you tune parameters, check the result on a
period you did not tune on, or you will be fitting noise.

## Layout

```
bot/
  config.py            strict YAML loader + validation + /settings changes
  indicators.py        EMA, RSI, MACD, ATR, ADX, Bollinger, confirmed swing pivots
  data/feed.py         ccxt REST candles + ccxt.pro WebSocket prices (REST fallback)
  data/history.py      historical download with CSV cache
  signals/engine.py    multi-timeframe confluence, stops, targets, confidence
  signals/filters.py   news blackout windows, volatility spike filter
  risk/limits.py       absolute safety ceilings
  risk/manager.py      sizing, exposure, daily loss halt, kill switch
  risk/trade_manager.py  stop, partial TP, breakeven, trailing stop
  execution/paper.py   simulated account (shared with the backtester)
  execution/live.py    disabled live executor + paper gate
  telegram_ui.py       commands and alerts
  app.py               live paper-trading orchestrator
  backtest/            event-driven backtester and metrics
  storage.py           SQLite: signals, fills, trades, events, equity, state
```

Logs go to `logs/jamesbot.log`. Signals, fills, trades, events and equity go to
`data/jamesbot.db`.

## Known limitations

- Funding payments, order-book depth and partial fills are not simulated.
- Paper stops are checked on ticks, so a wick between ticks can be missed. A real
  exchange-side stop would catch it.
- The news filter only knows the events you list in the config.
