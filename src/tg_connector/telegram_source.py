"""Telegram access via Telethon (MTProto, logged in as the user — needed to read DMs and groups).

* The session is a Telethon StringSession stored in data/telegram.session (chmod 600).
  It grants full access to the account, so it is treated like a password: never committed,
  never logged (it is registered as a secret in the log redaction filter).
* Updates are disabled (receive_updates=False): we pull history on demand, which is what
  makes repeated/missed runs safe — nothing depends on a live update stream.
* FLOOD_WAIT up to 60s is slept through by Telethon; longer waits are surfaced as
  SourceRateLimited so the pipeline stops politely and resumes from the stored cursor.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from datetime import datetime, timezone, tzinfo
from pathlib import Path

from telethon import TelegramClient, errors, types, utils
from telethon.sessions import StringSession

from .config import Settings
from .models import ChatInfo, RawMessage, SourceRateLimited

log = logging.getLogger(__name__)


class NotAuthorized(Exception):
    pass


def _kind_of(msg) -> str:  # noqa: ANN001 - telethon Message
    if getattr(msg, "action", None) is not None:
        return "service"
    if msg.sticker:
        return "sticker"
    if msg.gif:
        return "gif"
    if msg.voice:
        return "voice"
    if msg.video_note:
        return "video_note"
    if msg.poll:
        return "poll"
    if msg.photo:
        return "photo"
    if msg.video:
        return "video"
    if msg.document:
        return "document"
    if msg.media is not None and not msg.message:
        return "other"
    return "text"


def _forwarded_from(msg, tz: tzinfo | None = None) -> str | None:  # noqa: ANN001
    """'X, originally 20.11 17:55' — the original date matters: "завтра" in a forwarded message
    is relative to when it was written, not to when it was forwarded (found in the live demo)."""
    fwd = getattr(msg, "fwd_from", None)
    if fwd is None:
        return None
    origin = getattr(msg, "forward", None)
    name = None
    for entity in (getattr(origin, "sender", None), getattr(origin, "chat", None)):
        if entity is not None and utils.get_display_name(entity):
            name = utils.get_display_name(entity)
            break
    name = name or getattr(fwd, "from_name", None) or getattr(fwd, "post_author", None) or "hidden user"
    sent = getattr(fwd, "date", None)
    if isinstance(sent, datetime):
        sent = sent if sent.tzinfo else sent.replace(tzinfo=timezone.utc)
        name += f", originally {sent.astimezone(tz or timezone.utc):%d.%m %H:%M}"
    return name


def _text_of(msg, kind: str, tz: tzinfo | None = None) -> str:  # noqa: ANN001
    text = msg.message or ""
    if kind == "document":
        name = getattr(msg.file, "name", None) or "file"
        text = f"[file: {name}] {text}".strip()
    elif kind == "poll":
        question = msg.poll.poll.question
        question = getattr(question, "text", question)
        text = f"[poll] {question}"
    origin = _forwarded_from(msg, tz)
    if origin and text:
        # The forwarder is the sender; who actually said it matters for owners and decisions.
        text = f"[forwarded from {origin}] {text}"
    return text


def _migrated_to(msg) -> int | None:  # noqa: ANN001
    action = getattr(msg, "action", None)
    if isinstance(action, types.MessageActionChatMigrateTo):
        return utils.get_peer_id(types.PeerChannel(action.channel_id))
    return None


def _sender_name(msg) -> str:  # noqa: ANN001
    sender = msg.sender
    if sender is not None:
        name = utils.get_display_name(sender)
        if name:
            return name
    return getattr(msg, "post_author", None) or "Unknown"


def to_raw(chat_id: int, msg, tz: tzinfo | None = None) -> RawMessage:  # noqa: ANN001
    kind = _kind_of(msg)
    reply_to = getattr(msg, "reply_to", None)
    sent_at = msg.date if msg.date.tzinfo else msg.date.replace(tzinfo=timezone.utc)
    return RawMessage(
        chat_id=chat_id,
        message_id=msg.id,
        sent_at=sent_at.astimezone(timezone.utc),
        sender_id=msg.sender_id,
        sender_name=_sender_name(msg),
        kind=kind,
        text=_text_of(msg, kind, tz),
        reply_to_id=getattr(reply_to, "reply_to_msg_id", None),
        edit_date=msg.edit_date,
        migrated_to=_migrated_to(msg),
    )


def _chat_kind(dialog) -> str:  # noqa: ANN001
    if dialog.is_user:
        return "bot" if getattr(dialog.entity, "bot", False) else "dm"
    if dialog.is_group:
        return "group"
    return "channel"


class TelethonSource:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: TelegramClient | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------- session
    def _new_client(self, session: str, device_model: str = "tg-mcp-connector") -> TelegramClient:
        if not (self.settings.tg_api_id and self.settings.tg_api_hash):
            raise NotAuthorized("TG_API_ID / TG_API_HASH are not set in .env")
        return TelegramClient(
            StringSession(session),
            self.settings.tg_api_id,
            self.settings.tg_api_hash,
            flood_sleep_threshold=60,
            receive_updates=False,
            device_model=device_model,
            app_version="0.1.0",
        )

    def save_session(self, session: str, path: Path | None = None) -> None:
        path = path or self.settings.session_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(session)
        os.chmod(path, 0o600)

    async def login_interactive(self, cloud: bool = False) -> str:
        """Prompts for phone, login code and (if enabled) 2FA password in the terminal.

        cloud=True creates a SEPARATE session (a new device in Telegram → Devices) for a hosted worker and
        saves it to data/cloud.session without printing it; the local session is not touched. One session
        must never run in two places at once: Telegram may revoke it (AUTH_KEY_DUPLICATED)."""
        client = self._new_client("", device_model="tg-mcp-connector-cloud" if cloud else "tg-mcp-connector")
        await client.start()
        me = await client.get_me()
        self.save_session(client.session.save(), self.settings.cloud_session_path if cloud else None)
        await client.disconnect()
        return utils.get_display_name(me)

    async def ensure_connected(self) -> TelegramClient:
        async with self._lock:
            if self._client is not None and self._client.is_connected():
                return self._client
            path = self.settings.session_path
            from_env = bool(self.settings.tg_session)  # hosted worker: the session is a (sealed) variable
            session = self.settings.tg_session if from_env else (path.read_text().strip() if path.exists() else "")
            if not session:
                raise NotAuthorized("Telegram is not logged in yet: run `uv run tg-connector login` in a terminal")
            client = self._new_client(session)
            await client.connect()
            if not await client.is_user_authorized():
                await client.disconnect()
                hint = "create a new one with `tg-connector login --cloud` and update TG_SESSION" if from_env else \
                    "run `uv run tg-connector login` again"
                raise NotAuthorized(f"Telegram session is invalid or revoked: {hint}")
            self._client = client
            return client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.disconnect()
            self._client = None

    async def _input_entity(self, chat_id: int):  # noqa: ANN202
        client = await self.ensure_connected()
        try:
            return await client.get_input_entity(chat_id)
        except ValueError:
            # A StringSession keeps the entity cache in memory only: warm it from the dialog list.
            await client.get_dialogs(limit=None)
            return await client.get_input_entity(chat_id)

    # ---------------------------------------------------------------- reads
    async def list_chats(self, limit: int = 100) -> list[ChatInfo]:
        client = await self.ensure_connected()
        chats: list[ChatInfo] = []
        try:
            async for d in client.iter_dialogs(limit=limit):
                chats.append(ChatInfo(chat_id=d.id, title=d.name or str(d.id), kind=_chat_kind(d), unread=d.unread_count))
        except errors.FloodWaitError as exc:
            raise SourceRateLimited(exc.seconds) from exc
        return chats

    async def get_chat_info(self, chat_id: int) -> ChatInfo:
        client = await self.ensure_connected()
        await self._input_entity(chat_id)  # warms the cache if needed
        entity = await client.get_entity(chat_id)
        if entity.__class__.__name__ == "User":
            kind = "bot" if getattr(entity, "bot", False) else "dm"
        elif entity.__class__.__name__ == "Channel" and not getattr(entity, "megagroup", False):
            kind = "channel"
        else:
            kind = "group"
        return ChatInfo(chat_id=chat_id, title=utils.get_display_name(entity) or str(chat_id), kind=kind)

    async def iter_new_messages(
        self, chat_id: int, min_id: int, since: datetime | None, batch_size: int
    ) -> AsyncIterator[list[RawMessage]]:
        client = await self.ensure_connected()
        entity = await self._input_entity(chat_id)
        kwargs: dict = {"reverse": True}  # oldest → newest, so the cursor only moves forward
        if min_id > 0:
            kwargs["min_id"] = min_id  # exclusive: only id > cursor
        elif since is not None:
            kwargs["offset_date"] = since  # first sync: limited backfill window
        batch: list[RawMessage] = []
        try:
            async for msg in client.iter_messages(entity, **kwargs):
                batch.append(to_raw(chat_id, msg, self.settings.timezone))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        except errors.FloodWaitError as exc:
            if batch:
                yield batch
            raise SourceRateLimited(exc.seconds) from exc
        if batch:
            yield batch

    async def messages_in_window(
        self, chat_id: int, since: datetime, max_id: int, min_id: int = 0
    ) -> list[RawMessage]:
        """Already-synced messages (min_id < id <= max_id) newer than `since` — used to catch edits.

        Walks newest → oldest and stops at the window start or at min_id (the "track from now" floor),
        so history the user chose to skip is never requested page after page.
        """
        client = await self.ensure_connected()
        entity = await self._input_entity(chat_id)
        found: list[RawMessage] = []
        try:
            async for m in client.iter_messages(entity, max_id=max_id + 1, min_id=min_id):
                if m.date < since:
                    break
                found.append(to_raw(chat_id, m, self.settings.timezone))
        except errors.FloodWaitError as exc:
            raise SourceRateLimited(exc.seconds) from exc
        found.reverse()
        return found

    async def latest_message_id(self, chat_id: int) -> int:
        """Id of the newest message (0 for an empty chat). Only the id is used — the text is discarded."""
        client = await self.ensure_connected()
        entity = await self._input_entity(chat_id)
        latest = await client.get_messages(entity, limit=1)
        return latest[0].id if latest else 0

    async def recent_messages(self, chat_id: int, limit: int) -> list[RawMessage]:
        client = await self.ensure_connected()
        entity = await self._input_entity(chat_id)
        messages = [to_raw(chat_id, m, self.settings.timezone) async for m in client.iter_messages(entity, limit=limit)]
        return list(reversed(messages))

    # --------------------------------------------------------------- writes
    async def send_message(self, chat_id: int, text: str) -> int:
        client = await self.ensure_connected()
        entity = await self._input_entity(chat_id)
        sent = await client.send_message(entity, text)
        return sent.id
