"""Stage 1 of the pipeline: pull NEW messages from Telegram into SQLite.

Per tracked chat:
  1. fetch messages with id > cursor (oldest first) in batches;
     on the very first sync (cursor = 0) only the last BACKFILL_DAYS are fetched;
  2. store each batch + advance the cursor in one transaction (db.store_batch);
  3. optionally re-read the last EDIT_LOOKBACK_HOURS to pick up edited messages and to notice
     messages deleted in Telegram (they are removed locally, so they also leave the summary).

A chat tracked "from now" (include_history=False) gets cursor = floor = its latest message id:
nothing at or below the floor is ever stored or sent to the LLM, not even by the edit lookback.

No LLM work happens here — sync is cheap and safe to repeat.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from datetime import datetime, time, timedelta
from typing import Any, Protocol

from .config import Settings
from .db import Database
from .models import ChatInfo, RawMessage, SourceRateLimited
from .noise import prepare

log = logging.getLogger(__name__)


class MessageSource(Protocol):
    async def list_chats(self, limit: int = 100) -> list[ChatInfo]: ...

    async def get_chat_info(self, chat_id: int) -> ChatInfo: ...

    def iter_new_messages(
        self, chat_id: int, min_id: int, since: datetime | None, batch_size: int
    ) -> AsyncIterator[list[RawMessage]]: ...

    async def messages_in_window(
        self, chat_id: int, since: datetime, max_id: int, min_id: int = 0
    ) -> list[RawMessage]: ...

    async def latest_message_id(self, chat_id: int) -> int: ...

    async def recent_messages(self, chat_id: int, limit: int) -> list[RawMessage]: ...

    async def send_message(self, chat_id: int, text: str) -> int: ...


async def sync_chat(
    db: Database, source: MessageSource, settings: Settings, chat_id: int, now: datetime, heartbeat: Callable[[], None]
) -> dict[str, Any]:
    chat = db.get_chat(chat_id)
    start_cursor = int(chat["cursor_message_id"]) if chat else 0
    floor = int(chat["floor_message_id"]) if chat else 0
    ignore = team_bot_ids(db)
    since = None if start_cursor > 0 else backfill_start(now, settings)
    stats: dict[str, Any] = {"fetched": 0, "inserted": 0, "updated": 0, "deleted": 0, "dirty_days": set()}

    async for batch in source.iter_new_messages(chat_id, start_cursor, since, settings.batch_size):
        if not batch:
            continue
        prepared = [prepare(m, settings.timezone, ignore) for m in batch]
        result = db.store_batch(chat_id, prepared, new_cursor=max(m.message_id for m in batch))
        for m in batch:
            if m.migrated_to:
                stats["migrated_to"] = m.migrated_to
        stats["fetched"] += len(batch)
        stats["inserted"] += result.inserted
        stats["updated"] += result.updated
        stats["dirty_days"] |= result.dirty_days
        heartbeat()

    if start_cursor > 0 and settings.edit_lookback_hours > 0:
        window_start = (now - timedelta(hours=settings.edit_lookback_hours)).replace(microsecond=0)
        recent = await source.messages_in_window(chat_id, window_start, start_cursor, min_id=floor)
        recent = [m for m in recent if m.message_id > floor]  # never touch history skipped by "track from now"
        if recent:
            result = db.store_batch(chat_id, [prepare(m, settings.timezone, ignore) for m in recent], new_cursor=None)
            stats["updated"] += result.updated + result.inserted
            stats["dirty_days"] |= result.dirty_days
        # Stored in the same window but no longer returned by Telegram = deleted there.
        stored = db.message_ids_in_window(chat_id, window_start.isoformat(), floor, start_cursor)
        deleted = stored - {m.message_id for m in recent}
        if deleted:
            stats["deleted"] += len(deleted)
            stats["dirty_days"] |= db.delete_messages(chat_id, deleted)

    db.mark_chat_synced(chat_id, error=None)
    return stats


TEAM_BOT_KEY = "team_bot_id"


def team_bot_ids(db: Database) -> frozenset[int]:
    raw = db.kv_get(TEAM_BOT_KEY)
    return frozenset({int(raw)}) if raw else frozenset()


def backfill_start(now: datetime, settings: Settings) -> datetime:
    """First sync of a chat starts at local midnight, so the oldest day is never summarized half-read."""
    first_day = (now.astimezone(settings.timezone) - timedelta(days=settings.backfill_days)).date()
    return datetime.combine(first_day, time.min, tzinfo=settings.timezone)


async def sync_tracked_chats(
    db: Database, source: MessageSource, settings: Settings, now: datetime, heartbeat: Callable[[], None]
) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "chats": 0, "fetched": 0, "inserted": 0, "updated": 0, "deleted": 0, "dirty_days": 0, "errors": [],
    }
    for chat in db.tracked_chats():
        chat_id = int(chat["chat_id"])
        try:
            stats = await sync_chat(db, source, settings, chat_id, now, heartbeat)
        except SourceRateLimited as exc:
            # Everything stored so far is committed; the next run resumes from the cursor.
            log.warning("Telegram rate limit while syncing %s: %s", chat["title"], exc)
            db.mark_chat_synced(chat_id, error=str(exc))
            totals["errors"].append({"chat": chat["title"], "error": str(exc)})
            totals["rate_limited"] = True
            break
        except Exception as exc:  # one broken chat must not stop the others
            log.exception("Sync failed for chat %s", chat["title"])
            db.mark_chat_synced(chat_id, error=f"{type(exc).__name__}: {exc}")
            totals["errors"].append({"chat": chat["title"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        totals["chats"] += 1
        totals["fetched"] += stats["fetched"]
        totals["inserted"] += stats["inserted"]
        totals["updated"] += stats["updated"]
        totals["deleted"] += stats["deleted"]
        totals["dirty_days"] += len(stats["dirty_days"])
        log.info(
            "Synced %s: fetched=%d new=%d updated=%d deleted=%d days_changed=%d",
            chat["title"], stats["fetched"], stats["inserted"], stats["updated"], stats["deleted"],
            len(stats["dirty_days"]),
        )
        if stats.get("migrated_to"):
            await _follow_migration(db, source, chat, int(stats["migrated_to"]), totals)
    return totals


async def _follow_migration(db: Database, source: MessageSource, chat: Any, new_id: int, totals: dict[str, Any]) -> None:
    """A basic group upgraded to a supergroup gets a NEW chat id; the old one never receives messages again.
    Track the new id automatically (it starts empty, so no history is pulled twice) and stop tracking the old one."""
    try:
        info = await source.get_chat_info(new_id)
    except Exception as exc:  # keep the old chat tracked and make the problem visible
        msg = f"group was upgraded to supergroup {new_id}; run `track {new_id}` ({type(exc).__name__}: {exc})"
        db.mark_chat_synced(int(chat["chat_id"]), error=msg)
        totals["errors"].append({"chat": chat["title"], "error": msg})
        return
    db.upsert_chat(info)
    db.set_tracked(new_id, True)
    db.set_tracked(int(chat["chat_id"]), False)
    totals.setdefault("migrated", []).append({"from": int(chat["chat_id"]), "to": new_id, "title": info.title})
    log.warning("Chat %s was upgraded to supergroup %s — now tracking the new id", chat["title"], new_id)
