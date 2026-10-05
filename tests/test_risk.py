from types import SimpleNamespace

import pytest

from bot.config import ConfigError, apply_setting, config_from_dict
from bot.risk.manager import MarketLimits, RiskManager
from bot.utils import parse_utc

LIMITS = MarketLimits(min_qty=0.001, min_notional=5, qty_step=0.001)
T0 = parse_utc("2025-05-01T10:00:00Z")


def rm(cfg, state=None):
    return RiskManager(cfg.risk, cfg.execution, state)


def test_size_risks_at_most_configured_pct_including_costs(cfg):
    r = rm(cfg)
    res = r.size_position(10_000, entry=100.0, stop=98.0, limits=LIMITS)
    assert res.ok
    loss_at_stop = res.qty * (100 - 98) + res.qty * (100 + 98) * (cfg.execution.fee_pct + cfg.execution.slippage_pct) / 100
    assert loss_at_stop <= 10_000 * cfg.risk.risk_per_trade_pct / 100 + 1e-9
    assert loss_at_stop > 10_000 * cfg.risk.risk_per_trade_pct / 100 * 0.99  # not needlessly small


def test_size_rounds_down_to_lot_step(cfg):
    res = rm(cfg).size_position(10_000, 100.0, 98.0, LIMITS)
    assert abs(res.qty / 0.001 - round(res.qty / 0.001)) < 1e-6


def test_leverage_cap_limits_tight_stops(cfg):
    # a very tight stop would need huge size; leverage cap must bind
    res = rm(cfg).size_position(10_000, 100.0, 99.95, LIMITS)
    assert res.ok and res.notional <= 10_000 * cfg.risk.max_leverage + 1e-6
    assert res.capped_by is not None


def test_exposure_already_used_blocks(cfg):
    res = rm(cfg).size_position(10_000, 100.0, 98.0, LIMITS, total_notional=30_000)
    assert not res.ok and "exposure" in res.reason


def test_below_exchange_minimum_rejected(cfg):
    res = rm(cfg).size_position(10, 100.0, 50.0, MarketLimits(min_qty=1.0, min_notional=5, qty_step=1.0))
    assert not res.ok


def test_daily_loss_halt_and_reset_next_day(cfg):
    r = rm(cfg)
    r.on_equity(10_000, T0)
    events = r.on_equity(10_000 * (1 - cfg.risk.max_daily_loss_pct / 100) - 1, T0 + 60_000)
    assert any(e.kind == "daily_loss_halt" for e in events)
    assert not r.trading_allowed()[0]
    assert not r.can_open("BTC/USDT:USDT", [])[0]
    r.on_equity(9_690, T0 + 86_400_000)  # next UTC day
    assert r.trading_allowed()[0]


def test_drawdown_kill_switch_requires_manual_reset(cfg):
    r = rm(cfg)
    r.on_equity(10_000, T0)
    r.on_equity(10_000, T0 + 86_400_000)
    events = r.on_equity(10_000 * (1 - cfg.risk.kill_switch.max_drawdown_pct / 100), T0 + 2 * 86_400_000)
    assert any(e.kind == "kill" for e in events)
    r.on_equity(9_000, T0 + 3 * 86_400_000)  # a new day does not clear the kill switch
    assert not r.trading_allowed()[0]
    r.reset_kill(8_500)
    assert r.trading_allowed()[0]


def test_api_error_kill_switch(cfg):
    r = rm(cfg)
    n = cfg.risk.kill_switch.max_consecutive_api_errors
    for _ in range(n - 1):
        assert not r.record_api_error("x")
    r.record_api_success()  # a success resets the counter
    for _ in range(n - 1):
        r.record_api_error("x")
    assert r.record_api_error("x") and r.kill_active


def test_slippage_kill_switch(cfg):
    r = rm(cfg)
    _, killed = r.check_slippage(100.0, 100.0 * (1 + cfg.risk.kill_switch.max_slippage_pct / 100 * 2))
    assert killed and not r.trading_allowed()[0]


def test_position_count_limits(cfg):
    r = rm(cfg)
    open_pos = [SimpleNamespace(symbol="BTC/USDT:USDT")]
    assert not r.can_open("BTC/USDT:USDT", open_pos)[0]  # one per symbol
    assert r.can_open("ETH/USDT:USDT", open_pos)[0]
    many = [SimpleNamespace(symbol=f"X{i}/USDT:USDT") for i in range(cfg.risk.max_open_positions)]
    assert not r.can_open("ETH/USDT:USDT", many)[0]


def test_pause_blocks_entries(cfg):
    r = rm(cfg)
    r.pause()
    assert not r.can_open("BTC/USDT:USDT", [])[0]
    r.resume()
    assert r.can_open("BTC/USDT:USDT", [])[0]


def test_state_roundtrip(cfg):
    r = rm(cfg)
    r.on_equity(10_000, T0)
    r.trigger_kill("test")
    r2 = rm(cfg, r.to_state())
    assert r2.kill_active and r2.kill_reason == "test" and r2.day_start_equity == 10_000


# ---------------------------------------------------------------- config limits

@pytest.mark.parametrize("path,value", [
    (("risk", "risk_per_trade_pct"), 5.0),
    (("risk", "max_leverage"), 50),
    (("risk", "max_daily_loss_pct"), 25),
    (("risk", "min_reward_risk"), 0.5),
    (("strategy", "min_categories"), 2),
    (("strategy", "min_confidence"), 2),
    (("mode", "trading_mode"), "live"),
])
def test_config_rejects_unsafe_values(raw_cfg, path, value):
    raw_cfg[path[0]][path[1]] = value
    with pytest.raises(ConfigError):
        config_from_dict(raw_cfg)


def test_config_rejects_unknown_and_missing_keys(raw_cfg):
    raw_cfg["risk"]["risk_per_trad_pct"] = 1.0
    with pytest.raises(ConfigError, match="unknown keys"):
        config_from_dict(raw_cfg)
    del raw_cfg["risk"]["risk_per_trad_pct"]
    del raw_cfg["risk"]["max_leverage"]
    with pytest.raises(ConfigError, match="missing keys"):
        config_from_dict(raw_cfg)


def test_telegram_requires_allowed_chats(raw_cfg):
    raw_cfg["telegram"]["enabled"] = True
    with pytest.raises(ConfigError, match="allowed_chat_ids"):
        config_from_dict(raw_cfg)


def test_runtime_settings_validated(cfg):
    new = apply_setting(cfg, "risk", "0.5")
    assert new.risk.risk_per_trade_pct == 0.5 and cfg.risk.risk_per_trade_pct == 1.0
    with pytest.raises(ConfigError):
        apply_setting(cfg, "risk", "3")
    with pytest.raises(ConfigError):
        apply_setting(cfg, "leverage", "25")
    assert apply_setting(cfg, "pairs", "sol/usdt, BTC/USDT").pairs == ["SOL/USDT:USDT", "BTC/USDT:USDT"]
    assert apply_setting(cfg, "autoexec", "on").mode.auto_execute is True
