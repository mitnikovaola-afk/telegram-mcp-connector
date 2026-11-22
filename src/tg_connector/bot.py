"""Team bot for tracked group chats: /status, /tasks, /help (an extension beyond the test task).

The bot is a separate Telegram bot account (token from @BotFather), not the user: it can never send
messages as the user. It only reads the connector's own data (SQLite + the Notion tasks database).

Access rules, enforced in BotLogic (so they are unit-tested without Telegram):
* privacy mode (BotFather default) means Telegram delivers only commands, not the conversation;
* it answers only in TRACKED chats and only with the data of the chat the command came from;
* in an untracked group it leaves without answering; in private messages it shows help only;
* one command per chat per COOLDOWN_S seconds — extra calls are ignored (no spam, no load).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .config import Settings
from .db import Database
from .notion_sink import NotionSink
from .sync import TEAM_BOT_KEY
from .team_status import (
    ERROR_TEXT,
    HELP_TEXT,
    PRIVATE_TEXT,
    collect,
    order_tasks,
    render_status,
    render_tasks,
    synced_label,
)

log = logging.getLogger(__name__)

COOLDOWN_S = 60
COMMAND_RE = re.compile(r"^/(status|tasks|help|start)(?:@(\w+))?(?:\s|$)", re.IGNORECASE)
MENU = [
    ("status", "Статус проєкту за 7 днів"),
    ("tasks", "Відкриті задачі, прострочені зверху"),
    ("help", "Що вміє бот"),
]


@dataclass(frozen=True)
class Reply:
    text: str | None = None
    leave: bool = False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BotLogic:
    """Everything the bot decides, independent of Telethon."""

    def __init__(self, settings: Settings, db: Database, notion: NotionSink | None,
                 bot_username: str | None = None, clock: Callable[[], datetime] = _utcnow):
        self.settings = settings
        self.db = db
        self.notion = notion
        self.bot_username = (bot_username or "").lower()
        self.clock = clock
        self._last: dict[tuple[int, str], float] = {}

    def _cooled_down(self, chat_id: int, command: str) -> bool:
        now = self.clock().timestamp()
        key = (chat_id, command)
        if now - self._last.get(key, float("-inf")) < COOLDOWN_S:
            return False
        self._last[key] = now
        return True

    async def handle(self, chat_id: int, is_private: bool, text: str) -> Reply | None:
        match = COMMAND_RE.match((text or "").strip())
        if not match:
            return None
        command, addressee = match[1].lower(), (match[2] or "").lower()
        if addressee and self.bot_username and addressee != self.bot_username:
            return None  # "/status@another_bot" is not for us
        if is_private:
            return Reply(PRIVATE_TEXT)
        if chat_id not in self.db.tracked_ids():
            log.warning("Bot used in an untracked chat %s — leaving it", chat_id)
            return Reply(leave=True)
        if not self._cooled_down(chat_id, command):
            return None
        if command in ("help", "start"):
            return Reply(HELP_TEXT)
        try:
            return Reply(await self._report(chat_id, command))
        except Exception:
            log.exception("Bot could not build /%s for chat %s", command, chat_id)
            return Reply(ERROR_TEXT)

    async def _report(self, chat_id: int, command: str) -> str:
        chat = self.db.get_chat(chat_id)
        title = chat["title"] if chat else str(chat_id)
        today = self.clock().astimezone(self.settings.timezone).date()
        data = collect(self.db, chat_id, today)
        notion_tasks = await self.notion.open_tasks(title) if self.notion else None
        from_notion = notion_tasks is not None
        tasks = order_tasks(notion_tasks if from_notion else data["summary_tasks"], today)
        synced = synced_label(self.db, chat_id, self.settings)
        if command == "tasks":
            return render_tasks(title, tasks, from_notion, today, synced)
        return render_status(title, data, tasks, from_notion, today, synced)


# ------------------------------------------------------------------------ Telegram


def database_ready(path: Any) -> bool:
    """An uploaded database is ready when it exists and SQLite accepts it (not a half-uploaded file)."""
    import sqlite3

    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            ok = conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            has_state = conn.execute("SELECT count(*) FROM sqlite_master WHERE name = 'chats'").fetchone()[0] == 1
            return ok and has_state
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return False


async def wait_for_state(settings: Settings, poll_s: float = 60, settle_s: float = 10) -> None:
    """Hosted first start: wait for the uploaded database instead of starting from zero — a fresh state
    would create SECOND Notion databases and re-summarize history. The file must also stop growing
    (an upload in progress) and pass SQLite's own check."""
    path = settings.db_path
    last_size = -1
    while True:
        if database_ready(path):
            size = path.stat().st_size
            if size == last_size:
                log.info("Database found at %s (%d bytes) — starting", path, size)
                return
            last_size = size
            await asyncio.sleep(settle_s)
            continue
        log.warning("Waiting for the database upload to %s (WAIT_FOR_STATE=true) …", path)
        await asyncio.sleep(poll_s)



def _save_bot_session(settings: Settings, session: str) -> None:
    path = settings.bot_session_path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(session)
    os.chmod(path, 0o600)


async def run_bot(settings: Settings, with_sync: bool = False) -> None:
    """Run the bot until stopped. with_sync=True also runs the periodic sync in this process
    (for 24/7 hosting: the bot and the sync must share one SQLite database)."""
    from telethon import TelegramClient, events, functions, types
    from telethon.sessions import StringSession

    from .pipeline import build_app

    if not settings.tg_bot_token:
        raise SystemExit("TG_BOT_TOKEN is not set in .env (create a bot with @BotFather → /newbot).")
    if not (settings.tg_api_id and settings.tg_api_hash):
        raise SystemExit("TG_API_ID / TG_API_HASH are not set in .env.")
    if settings.wait_for_state:
        await wait_for_state(settings)

    app = build_app(settings)
    saved = settings.bot_session_path.read_text().strip() if settings.bot_session_path.exists() else ""
    client = TelegramClient(StringSession(saved), settings.tg_api_id, settings.tg_api_hash,
                            device_model="tg-mcp-connector-bot", app_version="0.1.0")
    await client.start(bot_token=settings.tg_bot_token)
    _save_bot_session(settings, client.session.save())  # next start reuses it instead of logging in again
    me = await client.get_me()
    app.db.kv_set(TEAM_BOT_KEY, str(me.id))  # the sync treats this bot's own replies as noise
    logic = BotLogic(settings, app.db, app.sink, bot_username=me.username)
    await client(functions.bots.SetBotCommandsRequest(
        scope=types.BotCommandScopeDefault(), lang_code="",
        commands=[types.BotCommand(command=c, description=d) for c, d in MENU],
    ))

    @client.on(events.NewMessage(pattern=r"^/"))
    async def on_command(event: Any) -> None:
        reply = await logic.handle(event.chat_id, event.is_private, event.raw_text or "")
        if reply is None:
            return
        if reply.leave:
            try:
                await client.kick_participant(event.chat_id, "me")
            except Exception as exc:  # leaving is best effort; the chat gets no data either way
                log.warning("Could not leave chat %s: %s", event.chat_id, exc)
            return
        await event.reply(reply.text, parse_mode="html", link_preview=False)

    sync_task = None
    if with_sync and settings.sync_interval_minutes > 0:
        from .server import periodic_sync

        sync_task = asyncio.create_task(periodic_sync(app))
    log.info("Bot @%s is running (sync in this process: %s)", me.username, bool(sync_task))
    try:
        await client.run_until_disconnected()
    finally:
        if sync_task:
            sync_task.cancel()
        await app.close()
