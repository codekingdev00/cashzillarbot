"""Telegram interface (python-telegram-bot v21).

Only chats listed in telegram.allowed_chat_ids may use the bot; everything else
is ignored and logged. A failed Telegram send never interrupts trading.

Three ways in, all routed to the same command functions:
  * slash commands (/analyze BTC), listed in Telegram's menu button
  * a persistent keyboard under the chat (📊 Analyze, 📈 Status, ...)
  * inline buttons attached to replies (callback data "command:argument")
"""
from __future__ import annotations

import contextvars
import logging
import time
from dataclasses import dataclass
from html import escape
from typing import Awaitable, Callable, Protocol, Union

from bot.config import TelegramConfig

log = logging.getLogger(__name__)
MAX_LEN = 4000      # Telegram text limit is 4096 characters
MAX_CAPTION = 1024  # Telegram photo caption limit


@dataclass
class Reply:
    text: str
    buttons: list[list[tuple[str, str]]] | None = None  # rows of (label, callback data)
    photo: bytes | None = None                          # PNG sent with the text as caption


@dataclass
class TelegramUser:
    id: int | None
    name: str
    username: str | None
    is_owner: bool
    chat_id: int | None = None

    @property
    def display(self) -> str:
        return self.name + (f" (@{self.username})" if self.username else "")


# who sent the update being handled (set per update; updates are processed one at a time)
current_user: contextvars.ContextVar[TelegramUser | None] = contextvars.ContextVar("current_user", default=None)

ReplyLike = Union[str, Reply, list]
CommandFn = Callable[[list[str]], Awaitable[ReplyLike]]

# persistent keyboard label -> command
KEYBOARD = [["📡 Scanner", "📊 Analyze"], ["💼 Account", "💵 Amount"], ["📈 Profit", "📋 Status"],
            ["📜 History", "⚙️ Settings"]]
KEYBOARD_COMMANDS = {"📡 Scanner": "scanner", "📊 Analyze": "analyze", "💼 Account": "account", "💵 Amount": "amount", "📈 Profit": "profit",
                     "📋 Status": "status", "📈 Status": "status", "📜 History": "history", "⚙️ Settings": "settings"}
OWNER_ONLY = {"kill"}  # bot-wide kill; /settings with a value is owner-only too
MENU_DESCRIPTIONS = {
    "scanner": "Live scan of all pairs: what the bot sees",
    "analyze": "Analyse a pair now",
    "account": "Your own paper account and charts",
    "topup": "Add paper funds to your account",
    "autoexec": "Auto-trade signals on your account: on/off",
    "amount": "Pick your trade amount (% of balance)",
    "profit": "Your profit chart and totals",
    "status": "Equity, daily P&L, open positions",
    "history": "Recent closed trades",
    "settings": "Show or change settings",
    "pause": "Pause alerts and entries on your account",
    "resume": "Resume your account",
    "kill": "Emergency stop (kill switch)",
    "resetkill": "Clear the kill switch",
    "papergate": "Paper-trading progress",
    "help": "All commands",
}


class Notifier(Protocol):
    async def send(self, text: str) -> None: ...


class ConsoleNotifier:
    """Used when Telegram is disabled: alerts go to the log."""

    async def send(self, text: str, keyboard: bool = False, chat_id: int | None = None) -> None:
        log.info("ALERT to %s\n%s", chat_id or "default", text)

    async def send_photo(self, photo: bytes, caption: str, chat_id: int | None = None) -> None:
        log.info("ALERT to %s (chart, %d bytes)\n%s", chat_id or "default", len(photo), caption)

    async def profile_photo(self, user_id: int) -> bytes | None:
        return None


def _as_replies(result: ReplyLike) -> list[Reply]:
    items = result if isinstance(result, list) else [result]
    return [r if isinstance(r, Reply) else Reply(str(r)) for r in items]


class TelegramUI:
    def __init__(self, cfg: TelegramConfig, token: str, commands: dict[str, CommandFn]):
        from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters

        self.cfg = cfg
        self.commands = commands
        self.allowed = {int(x) for x in cfg.allowed_chat_ids}
        self.alert_chat = int(cfg.alert_chat_id) if cfg.alert_chat_id is not None else next(iter(cfg.allowed_chat_ids))
        # concurrent updates: one slow request (e.g. an exchange timeout) must not hold up every other button
        self.app = Application.builder().token(token).concurrent_updates(True).build()
        self.bot_name = "Trading bot"
        self._last_public: dict[int, float] = {}
        for name in commands:
            self.app.add_handler(CommandHandler(name, self._on_command))
        self.app.add_handler(CallbackQueryHandler(self._on_button))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text))

    # ------------------------------------------------------------- routing
    def _check_access(self, update, name: str | None, args: list[str]) -> tuple[bool, str | None]:
        """(allowed, message for the user if refused). Owners can do everything.
        With public_access, other chats may use read-only commands, rate limited."""
        chat = update.effective_chat
        if chat is None:
            return False, None
        if chat.id in self.allowed:
            return True, None
        if not self.cfg.public_access:
            log.warning("ignored update from unauthorized chat %s", chat.id)
            return False, None
        if name in OWNER_ONLY or (name == "settings" and args):
            log.info("refused owner-only /%s from chat %s", name, chat.id)
            return False, "🔒 Only the bot owner can use this command."
        now = time.monotonic()
        if now - self._last_public.get(chat.id, -1e9) < self.cfg.public_rate_limit_seconds:
            return False, "⏳ Please wait a few seconds between requests."
        self._last_public[chat.id] = now
        return True, None

    def _user(self, update) -> TelegramUser:
        u, chat = update.effective_user, update.effective_chat
        name = (u.full_name if u else None) or (chat.title if chat else None) or "Unknown"
        return TelegramUser(id=u.id if u else None, name=name, username=u.username if u else None,
                            is_owner=bool(chat and chat.id in self.allowed), chat_id=chat.id if chat else None)

    async def _run(self, name: str, args: list[str], update=None) -> list[Reply]:
        if update is not None:
            current_user.set(self._user(update))
        fn = self.commands.get(name)
        if fn is None:
            return [Reply(f"Unknown command /{escape(name)}. Try /help.")]
        try:
            return _as_replies(await fn(args))
        except Exception as e:  # noqa: BLE001
            log.exception("command /%s failed", name)
            return [Reply(f"Error: {escape(str(e))}")]

    def _markup(self, reply: Reply, with_keyboard: bool):
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup

        if reply.buttons:
            return InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=data) for label, data in row]
                                         for row in reply.buttons])
        if with_keyboard:
            return ReplyKeyboardMarkup(KEYBOARD, resize_keyboard=True, is_persistent=True)
        return None

    async def _reply(self, message, replies: list[Reply], with_keyboard: bool = False) -> None:
        from telegram.constants import ParseMode

        for r in replies:
            markup = self._markup(r, with_keyboard)
            if r.photo is not None:
                fits = len(r.text) <= MAX_CAPTION
                await message.reply_photo(r.photo, caption=r.text if fits else None, parse_mode=ParseMode.HTML,
                                          reply_markup=markup if fits else None)
                if fits:
                    continue
            await message.reply_text(r.text[:MAX_LEN], parse_mode=ParseMode.HTML, disable_web_page_preview=True,
                                     reply_markup=markup)

    async def _on_command(self, update, context) -> None:
        name = update.effective_message.text.split()[0].lstrip("/").split("@")[0].lower()
        args = list(context.args or [])
        ok, msg = self._check_access(update, name, args)
        if not ok:
            if msg:
                await self._reply(update.effective_message, [Reply(msg)])
            return
        await self._reply(update.effective_message, await self._run(name, args, update),
                          with_keyboard=name in ("start", "help"))

    async def _on_text(self, update, context) -> None:
        name = KEYBOARD_COMMANDS.get((update.effective_message.text or "").strip())
        ok, msg = self._check_access(update, name, [])
        if not ok:
            if msg:
                await self._reply(update.effective_message, [Reply(msg)])
            return
        if name is None:
            await self._reply(update.effective_message, [Reply("Use the buttons below or /help.")],
                              with_keyboard=True)
            return
        await self._reply(update.effective_message, await self._run(name, [], update))

    async def _on_button(self, update, context) -> None:
        query = update.callback_query
        name, _, arg = (query.data or "").partition(":")
        ok, msg = self._check_access(update, name, [arg] if arg else [])
        if not ok:
            await self._answer(query, msg or "", show_alert=bool(msg))
            return
        await self._answer(query, "Working…")
        await self._reply(query.message, await self._run(name, [arg] if arg else [], update))

    @staticmethod
    async def _answer(query, text: str, show_alert: bool = False) -> None:
        """Acknowledge a button press. Telegram rejects answers after ~15 s; that is harmless."""
        try:
            await query.answer(text, show_alert=show_alert)
        except Exception as exc:  # noqa: BLE001
            log.debug("button answer skipped: %s", exc)

    # ----------------------------------------------------------- lifecycle
    async def start(self) -> None:
        from telegram import BotCommand

        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        self.bot_name = self.app.bot.first_name or self.app.bot.username or "Trading bot"
        try:
            await self.app.bot.set_my_commands([BotCommand(n, d) for n, d in MENU_DESCRIPTIONS.items()
                                                if n in self.commands])
        except Exception:  # noqa: BLE001
            log.warning("could not set the Telegram command menu", exc_info=True)
        log.info("Telegram bot started; alerts go to chat %s", self.alert_chat)

    async def stop(self) -> None:
        try:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
        except Exception:  # noqa: BLE001
            log.debug("telegram shutdown error", exc_info=True)

    async def send(self, text: str, keyboard: bool = False, chat_id: int | None = None) -> None:
        """Message to one chat (default: the owner's alert chat). Never raises."""
        from telegram.constants import ParseMode

        try:
            await self.app.bot.send_message(chat_id=chat_id or self.alert_chat, text=text[:MAX_LEN],
                                            parse_mode=ParseMode.HTML, disable_web_page_preview=True,
                                            reply_markup=self._markup(Reply(text), keyboard))
        except Exception:  # noqa: BLE001 - never let a notification failure stop trading
            log.exception("Telegram send to %s failed", chat_id or self.alert_chat)

    async def send_photo(self, photo: bytes, caption: str, chat_id: int | None = None) -> None:
        """Chart to one chat; a long caption is sent as a separate message. Never raises."""
        from telegram.constants import ParseMode

        target = chat_id or self.alert_chat
        try:
            fits = len(caption) <= MAX_CAPTION
            await self.app.bot.send_photo(chat_id=target, photo=photo, caption=caption if fits else None,
                                          parse_mode=ParseMode.HTML)
            if not fits:
                await self.send(caption, chat_id=target)
        except Exception:  # noqa: BLE001
            log.exception("Telegram photo send to %s failed", target)
            await self.send(caption, chat_id=target)

    async def profile_photo(self, user_id: int) -> bytes | None:
        """The user's current Telegram profile picture (smallest size >= 160px), or None
        if they have none or their privacy settings hide it from bots."""
        try:
            photos = await self.app.bot.get_user_profile_photos(user_id, limit=1)
            if not photos.total_count or not photos.photos:
                return None
            sizes = sorted(photos.photos[0], key=lambda p: p.width)
            pick = next((p for p in sizes if p.width >= 160), sizes[-1])
            file = await self.app.bot.get_file(pick.file_id)
            return bytes(await file.download_as_bytearray())
        except Exception:  # noqa: BLE001 - the account card works without a picture
            log.debug("profile photo unavailable for %s", user_id, exc_info=True)
            return None
