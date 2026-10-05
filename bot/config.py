"""Configuration loading and validation.

Every tunable value lives in the YAML file. This module only knows the *shape*
of the config and the absolute ceilings in bot.risk.limits. Unknown or missing
keys are errors, so a typo can never silently fall back to a default.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional, get_type_hints

import yaml

from bot.risk import limits
from bot.utils import parse_utc, timeframe_to_ms


# trading modes that place orders on an exchange's DEMO environment (fake money), per exchange
DEMO_EXCHANGES = {"demo": ("bitget",)}


class ConfigError(ValueError):
    """Raised when the configuration is missing, malformed or unsafe."""


@dataclass
class ExchangeConfig:
    id: str
    hostname: Optional[str]
    market_type: str
    use_websocket: bool
    rest_poll_seconds: float
    request_timeout_ms: int
    api_key_env: str
    api_secret_env: str
    api_password_env: str


@dataclass
class ModeConfig:
    trading_mode: str
    auto_execute: bool
    starting_equity: float
    heartbeat_seconds: float
    max_topup: float
    stake_pct_options: list
    default_stake_pct: float


@dataclass
class TimeframeConfig:
    trend: str
    entry: str
    history_bars: int
    candle_close_delay_seconds: float


@dataclass
class StrategyConfig:
    ema_fast: int
    ema_slow: int
    htf_ema_fast: int
    htf_ema_slow: int
    adx_period: int
    adx_min: float
    rsi_period: int
    rsi_long_min: float
    rsi_long_max: float
    rsi_short_min: float
    rsi_short_max: float
    macd_fast: int
    macd_slow: int
    macd_signal: int
    bb_period: int
    bb_std: float
    atr_period: int
    volume_sma_period: int
    volume_mult: float
    swing_left: int
    swing_right: int
    structure_lookback: int
    divergence_lookback: int
    trigger_lookback: int
    min_confidence: int
    min_categories: int
    signal_cooldown_bars: int


@dataclass
class FilterConfig:
    allow_during_news: bool
    news_events: list
    default_news_before_min: float
    default_news_after_min: float
    allow_extreme_volatility: bool
    atr_spike_mult: float
    atr_spike_lookback: int
    candle_range_atr_max: float


@dataclass
class StopConfig:
    method: str
    atr_mult: float
    structure_buffer_atr: float
    min_stop_atr: float
    max_stop_atr: float


@dataclass
class TakeProfitConfig:
    target_r: float
    partial_at_r: float
    partial_pct: float
    move_stop_to_breakeven: bool
    trailing: bool
    trail_atr_mult: float


@dataclass
class KillSwitchConfig:
    max_consecutive_api_errors: int
    max_slippage_pct: float
    max_drawdown_pct: float
    flatten_on_kill: bool


@dataclass
class RiskConfig:
    risk_per_trade_pct: float
    max_leverage: float
    max_daily_loss_pct: float
    max_open_positions: int
    max_positions_per_symbol: int
    max_exposure_per_asset_pct: float
    min_reward_risk: float
    stop: StopConfig
    take_profit: TakeProfitConfig
    kill_switch: KillSwitchConfig


@dataclass
class ExecutionConfig:
    fee_pct: float
    slippage_pct: float


@dataclass
class PaperGateConfig:
    min_paper_trades: int
    min_paper_days: int


@dataclass
class TelegramConfig:
    enabled: bool
    token_env: str
    allowed_chat_ids: list
    alert_chat_id: Optional[int]
    pnl_update_minutes: float
    public_access: bool
    public_rate_limit_seconds: float


@dataclass
class LoggingConfig:
    level: str
    file: str
    db: str
    max_bytes: int
    backup_count: int
    equity_snapshot_minutes: float


@dataclass
class BacktestConfig:
    start: str
    end: str
    initial_equity: float
    data_dir: str
    report_dir: str
    qty_step: float
    min_notional: float


@dataclass
class Config:
    exchange: ExchangeConfig
    mode: ModeConfig
    pairs: list
    timeframes: TimeframeConfig
    strategy: StrategyConfig
    filters: FilterConfig
    risk: RiskConfig
    execution: ExecutionConfig
    paper_gate: PaperGateConfig
    telegram: TelegramConfig
    logging: LoggingConfig
    backtest: BacktestConfig


def _build(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    unknown = sorted(set(data) - set(names))
    missing = [n for n in names if n not in data]
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}")
    if missing:
        raise ConfigError(f"{where}: missing keys {missing}")
    kwargs = {}
    for name in names:
        value = data[name]
        hint = hints[name]
        if is_dataclass(hint):
            value = _build(hint, value, f"{where}.{name}")
        kwargs[name] = value
    return cls(**kwargs)


def normalize_symbol(symbol: str, market_type: str) -> str:
    """BTC/USDT -> BTC/USDT:USDT for perpetual swaps (ccxt unified symbol)."""
    s = str(symbol).strip().upper()
    if "/" not in s:
        raise ConfigError(f"pair '{symbol}' must look like BTC/USDT")
    if market_type == "swap" and ":" not in s:
        s = f"{s}:{s.split('/')[1]}"
    return s


def validate(cfg: Config) -> None:
    """Raise ConfigError listing every problem. Enforces the hard limits."""
    p: list[str] = []
    r, s, f = cfg.risk, cfg.strategy, cfg.filters

    if cfg.mode.trading_mode not in DEMO_EXCHANGES and cfg.mode.trading_mode != "paper":
        p.append("mode.trading_mode must be 'paper' or 'demo'. Real-money execution is intentionally not "
                 "implemented; complete the paper/demo testing first (see README).")
    if cfg.mode.trading_mode == "demo" and cfg.exchange.id not in DEMO_EXCHANGES["demo"]:
        p.append(f"demo trading is implemented for {', '.join(DEMO_EXCHANGES['demo'])} only "
                 f"(exchange.id is '{cfg.exchange.id}')")
    if cfg.mode.starting_equity <= 0:
        p.append("mode.starting_equity must be > 0")
    opts = cfg.mode.stake_pct_options
    if not isinstance(opts, list) or not opts or any(not isinstance(x, (int, float)) or not 0 < x <= 100
                                                     for x in opts):
        p.append("mode.stake_pct_options must be a list of percentages in (0, 100]")
    elif cfg.mode.default_stake_pct not in opts:
        p.append("mode.default_stake_pct must be one of mode.stake_pct_options")
    if cfg.mode.max_topup <= 0:
        p.append("mode.max_topup must be > 0")
    if cfg.mode.heartbeat_seconds <= 0:
        p.append("mode.heartbeat_seconds must be > 0")
    if cfg.exchange.market_type not in {"swap", "future"}:
        p.append("exchange.market_type must be 'swap' or 'future'")
    if cfg.exchange.rest_poll_seconds <= 0:
        p.append("exchange.rest_poll_seconds must be > 0")
    if not cfg.pairs:
        p.append("pairs must not be empty")
    if len(set(cfg.pairs)) != len(cfg.pairs):
        p.append("pairs contains duplicates")

    try:
        trend_ms = timeframe_to_ms(cfg.timeframes.trend)
        entry_ms = timeframe_to_ms(cfg.timeframes.entry)
        if trend_ms <= entry_ms:
            p.append("timeframes.trend must be a longer timeframe than timeframes.entry")
    except ValueError as e:
        p.append(str(e))
    warmup = max(s.htf_ema_slow, s.ema_slow, s.structure_lookback, s.macd_slow + s.macd_signal) + 20
    if cfg.timeframes.history_bars < warmup:
        p.append(f"timeframes.history_bars must be >= {warmup} so indicators can warm up")

    # --- risk: hard limits -------------------------------------------------
    if not 0 < r.risk_per_trade_pct <= limits.MAX_RISK_PER_TRADE_PCT:
        p.append(f"risk.risk_per_trade_pct must be in (0, {limits.MAX_RISK_PER_TRADE_PCT}]")
    if not 1 <= r.max_leverage <= limits.MAX_LEVERAGE:
        p.append(f"risk.max_leverage must be in [1, {limits.MAX_LEVERAGE}]")
    if not 0 < r.max_daily_loss_pct <= limits.MAX_DAILY_LOSS_PCT:
        p.append(f"risk.max_daily_loss_pct must be in (0, {limits.MAX_DAILY_LOSS_PCT}]")
    if r.max_daily_loss_pct < r.risk_per_trade_pct:
        p.append("risk.max_daily_loss_pct must be >= risk.risk_per_trade_pct")
    if not 1 <= r.max_open_positions <= limits.MAX_OPEN_POSITIONS:
        p.append(f"risk.max_open_positions must be in [1, {limits.MAX_OPEN_POSITIONS}]")
    if not 1 <= r.max_positions_per_symbol <= r.max_open_positions:
        p.append("risk.max_positions_per_symbol must be in [1, max_open_positions]")
    if not 0 < r.max_exposure_per_asset_pct <= r.max_leverage * 100:
        p.append("risk.max_exposure_per_asset_pct must be in (0, max_leverage * 100]")
    if r.min_reward_risk < limits.MIN_REWARD_RISK:
        p.append(f"risk.min_reward_risk must be >= {limits.MIN_REWARD_RISK}")

    st = r.stop
    if st.method not in {"atr", "structure", "hybrid"}:
        p.append("risk.stop.method must be atr, structure or hybrid")
    if st.atr_mult <= 0 or st.structure_buffer_atr < 0:
        p.append("risk.stop.atr_mult must be > 0 and structure_buffer_atr >= 0")
    if not 0 < st.min_stop_atr <= st.max_stop_atr:
        p.append("risk.stop requires 0 < min_stop_atr <= max_stop_atr")

    tp = r.take_profit
    if tp.target_r < r.min_reward_risk:
        p.append("risk.take_profit.target_r must be >= risk.min_reward_risk")
    if not 0 < tp.partial_pct < 100:
        p.append("risk.take_profit.partial_pct must be in (0, 100)")
    if not 0 < tp.partial_at_r < tp.target_r:
        p.append("risk.take_profit.partial_at_r must be in (0, target_r)")
    if tp.trail_atr_mult <= 0:
        p.append("risk.take_profit.trail_atr_mult must be > 0")

    ks = r.kill_switch
    if ks.max_consecutive_api_errors < 1:
        p.append("risk.kill_switch.max_consecutive_api_errors must be >= 1")
    if ks.max_slippage_pct <= 0:
        p.append("risk.kill_switch.max_slippage_pct must be > 0")
    if not 0 < ks.max_drawdown_pct <= limits.MAX_DRAWDOWN_KILL_PCT:
        p.append(f"risk.kill_switch.max_drawdown_pct must be in (0, {limits.MAX_DRAWDOWN_KILL_PCT}]")

    if cfg.execution.fee_pct < 0 or cfg.execution.slippage_pct < 0:
        p.append("execution.fee_pct and slippage_pct must be >= 0")
    if cfg.execution.slippage_pct >= ks.max_slippage_pct:
        p.append("execution.slippage_pct must be below risk.kill_switch.max_slippage_pct")

    # --- strategy -----------------------------------------------------------
    if not 0 < s.ema_fast < s.ema_slow:
        p.append("strategy.ema_fast must be < ema_slow")
    if not 0 < s.htf_ema_fast < s.htf_ema_slow:
        p.append("strategy.htf_ema_fast must be < htf_ema_slow")
    if not 0 < s.macd_fast < s.macd_slow:
        p.append("strategy.macd_fast must be < macd_slow")
    if not (0 <= s.rsi_long_min < s.rsi_long_max <= 100 and 0 <= s.rsi_short_min < s.rsi_short_max <= 100):
        p.append("strategy RSI zones must satisfy 0 <= min < max <= 100")
    if not limits.MIN_CONFIDENCE <= s.min_confidence <= 5:
        p.append(f"strategy.min_confidence must be in [{limits.MIN_CONFIDENCE}, 5]")
    if not limits.MIN_SIGNAL_CATEGORIES <= s.min_categories <= 4:
        p.append(f"strategy.min_categories must be in [{limits.MIN_SIGNAL_CATEGORIES}, 4] (confluence is required)")
    for name in ("adx_period", "rsi_period", "macd_signal", "bb_period", "atr_period", "volume_sma_period",
                 "swing_left", "swing_right", "structure_lookback", "divergence_lookback", "trigger_lookback"):
        if getattr(s, name) < 1:
            p.append(f"strategy.{name} must be >= 1")
    if s.signal_cooldown_bars < 0 or s.volume_mult <= 0 or s.bb_std <= 0:
        p.append("strategy.signal_cooldown_bars >= 0, volume_mult > 0 and bb_std > 0 are required")

    # --- filters ------------------------------------------------------------
    if f.atr_spike_mult <= 1 or f.candle_range_atr_max <= 0 or f.atr_spike_lookback < 10:
        p.append("filters: atr_spike_mult > 1, candle_range_atr_max > 0, atr_spike_lookback >= 10")
    if not isinstance(f.news_events, list):
        p.append("filters.news_events must be a list")
    else:
        for i, ev in enumerate(f.news_events):
            if not isinstance(ev, dict) or "time" not in ev or "name" not in ev:
                p.append(f"filters.news_events[{i}] needs 'time' and 'name'")
                continue
            try:
                parse_utc(ev["time"])
            except ValueError:
                p.append(f"filters.news_events[{i}].time is not an ISO datetime")

    # --- telegram -----------------------------------------------------------
    t = cfg.telegram
    if t.enabled and not t.allowed_chat_ids:
        p.append("telegram.allowed_chat_ids must list at least one chat id when telegram is enabled "
                 "(otherwise anyone could control the bot)")
    if t.pnl_update_minutes < 0:
        p.append("telegram.pnl_update_minutes must be >= 0 (0 disables)")
    if t.public_rate_limit_seconds < 0:
        p.append("telegram.public_rate_limit_seconds must be >= 0")

    # --- backtest / paper gate ---------------------------------------------
    b = cfg.backtest
    try:
        if parse_utc(b.start) >= parse_utc(b.end):
            p.append("backtest.start must be before backtest.end")
    except ValueError:
        p.append("backtest.start/end must be ISO dates")
    if b.initial_equity <= 0 or b.qty_step <= 0 or b.min_notional < 0:
        p.append("backtest: initial_equity > 0, qty_step > 0, min_notional >= 0 required")
    if cfg.paper_gate.min_paper_trades < 1 or cfg.paper_gate.min_paper_days < 1:
        p.append("paper_gate values must be >= 1")

    if p:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(p))


def config_from_dict(raw: Any) -> Config:
    cfg = _build(Config, raw, "config")
    if not isinstance(cfg.pairs, list):
        raise ConfigError("pairs must be a list")
    cfg.pairs = [normalize_symbol(sym, cfg.exchange.market_type) for sym in cfg.pairs]
    cfg.telegram.allowed_chat_ids = [int(x) for x in (cfg.telegram.allowed_chat_ids or [])]
    validate(cfg)
    return cfg


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file {p} not found. Copy config/config.example.yaml to {p} and edit it.")
    with p.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return config_from_dict(raw)


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Existing environment variables win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# --- runtime settings (Telegram /settings) -----------------------------------

def _parse_bool(v: str) -> bool:
    v = str(v).strip().lower()
    if v in {"on", "true", "yes", "1"}:
        return True
    if v in {"off", "false", "no", "0"}:
        return False
    raise ValueError(f"expected on/off, got '{v}'")


def _parse_pairs(v: str) -> list[str]:
    items = [x for x in str(v).replace(",", " ").split() if x]
    if not items:
        raise ValueError("no pairs given")
    return items


SETTABLE: dict[str, tuple[str, Callable[[str], Any], str]] = {
    "risk": ("risk.risk_per_trade_pct", float, "risk per trade, % of equity"),
    "leverage": ("risk.max_leverage", float, "leverage cap (x)"),
    "maxdaily": ("risk.max_daily_loss_pct", float, "daily loss halt, % of equity"),
    "maxpos": ("risk.max_open_positions", int, "max concurrent positions"),
    "minrr": ("risk.min_reward_risk", float, "minimum reward:risk"),
    "autoexec": ("mode.auto_execute", _parse_bool, "paper auto-execution on/off"),
    "pairs": ("pairs", _parse_pairs, "pairs, comma or space separated"),
}


def get_path(cfg: Config, path: str) -> Any:
    obj: Any = cfg
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set_path(cfg: Config, path: str, value: Any) -> None:
    parts = path.split(".")
    obj: Any = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def apply_setting(cfg: Config, key: str, raw_value: str) -> Config:
    """Return a NEW validated config with one setting changed. Never mutates cfg."""
    key = key.lower().strip()
    if key not in SETTABLE:
        raise ConfigError(f"unknown setting '{key}'. Settable: {', '.join(SETTABLE)}")
    path, parser, _ = SETTABLE[key]
    try:
        value = parser(raw_value)
    except (ValueError, TypeError) as e:
        raise ConfigError(f"invalid value for {key}: {raw_value} ({e})") from e
    if key == "pairs":
        value = [normalize_symbol(sym, cfg.exchange.market_type) for sym in value]
    new = copy.deepcopy(cfg)
    _set_path(new, path, value)
    validate(new)
    return new


def apply_overrides(cfg: Config, overrides: dict[str, str]) -> tuple[Config, list[str]]:
    """Re-apply persisted /settings changes at startup. Invalid ones are skipped."""
    notes = []
    for key, raw in (overrides or {}).items():
        try:
            cfg = apply_setting(cfg, key, raw)
            notes.append(f"applied saved setting {key}={raw}")
        except ConfigError as e:
            notes.append(f"skipped saved setting {key}={raw}: {e}")
    return cfg, notes
