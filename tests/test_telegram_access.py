from types import SimpleNamespace

from bot.telegram_ui import TelegramUI

OWNER, STRANGER = 111, 222
FAKE_TOKEN = "123456:TEST-token-not-real"


def ui(cfg, public: bool, rate: float = 0.0) -> TelegramUI:
    cfg.telegram.allowed_chat_ids = [OWNER]
    cfg.telegram.public_access = public
    cfg.telegram.public_rate_limit_seconds = rate
    return TelegramUI(cfg.telegram, FAKE_TOKEN, {})


def upd(chat_id: int):
    return SimpleNamespace(effective_chat=SimpleNamespace(id=chat_id))


def test_owner_can_do_everything(cfg):
    t = ui(cfg, public=False)
    for name in ("analyze", "kill", "pause", "settings"):
        assert t._check_access(upd(OWNER), name, ["x"])[0]


def test_private_mode_ignores_strangers_silently(cfg):
    t = ui(cfg, public=False)
    assert t._check_access(upd(STRANGER), "analyze", []) == (False, None)


def test_public_mode_allows_read_only_and_blocks_controls(cfg):
    t = ui(cfg, public=True)
    # everything that only touches the caller's own account is allowed
    for name in ("analyze", "status", "history", "help", "start", "settings", "account", "topup", "autoexec",
                 "pause", "resume", "resetkill"):
        assert t._check_access(upd(STRANGER), name, [])[0], name
    # bot-wide controls are owner-only
    for name, args in (("kill", []), ("settings", ["risk", "2"])):
        ok, msg = t._check_access(upd(STRANGER), name, args)
        assert not ok and "owner" in msg, name


def test_public_rate_limit(cfg):
    t = ui(cfg, public=True, rate=60)
    assert t._check_access(upd(STRANGER), "analyze", [])[0]
    ok, msg = t._check_access(upd(STRANGER), "analyze", [])
    assert not ok and "wait" in msg
    assert t._check_access(upd(OWNER), "analyze", [])[0]  # owners are never rate limited
