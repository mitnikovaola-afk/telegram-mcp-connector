"""The pipeline: sync → summarize → export, guarded by a single lease lock.

Every stage is idempotent on its own, so a run can be repeated, skipped or interrupted at any
point: the next run picks up exactly what is still pending.
"""

from __future__ import annotations

import logging
import os
import socket
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

from .config import Settings
from .db import Database
from .llm import LLMClient, build_llm
from .notion_sink import NotionSink, RENDER_SUFFIX
from .summarizer import llm_paused_until, summarize_pending
from .sync import MessageSource, sync_tracked_chats

log = logging.getLogger(__name__)

LOCK_NAME = "pipeline"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        source: MessageSource,
        llm: LLMClient | None,
        sink: NotionSink | None,
        clock: Callable[[], datetime] = utcnow,
    ):
        self.settings = settings
        self.db = db
        self.source = source
        self.llm = llm
        self.sink = sink
        self.clock = clock

    async def run(self, trigger: str, *, summarize: bool = True, export: bool = True) -> dict[str, Any]:
        owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        ttl = self.settings.lock_ttl_seconds
        if not self.db.try_acquire_lock(LOCK_NAME, owner, ttl):
            log.info("Run skipped (%s): another run holds the lock", trigger)
            return {"status": "skipped", "trigger": trigger, "reason": "another sync run is in progress"}

        started = time.monotonic()
        run_id = self.db.start_run(trigger)
        report: dict[str, Any] = {"run_id": run_id, "trigger": trigger, "status": "ok"}
        error: str | None = None

        def heartbeat() -> None:
            self.db.refresh_lock(LOCK_NAME, owner, ttl)

        try:
            now = self.clock()
            if not self.db.tracked_chats():
                report["sync"] = {"note": "no tracked chats yet — add some with tg_track_chat / `tg-connector track`"}
            else:
                ensure = getattr(self.source, "ensure_connected", None)
                if ensure is not None:
                    await ensure()
                report["sync"] = await sync_tracked_chats(self.db, self.source, self.settings, now, heartbeat)

            if summarize:
                if self.llm is None:
                    report["summaries"] = {"skipped": "GEMINI_API_KEY is not set"}
                else:
                    report["summaries"] = await summarize_pending(self.db, self.llm, self.settings, now, heartbeat)

            if export:
                if self.sink is None:
                    report["notion"] = {"skipped": "NOTION_TOKEN / NOTION_PARENT_PAGE_ID are not set"}
                else:
                    report["notion"] = await self.sink.export_pending()

            if self.settings.raw_retention_days > 0:
                cutoff = (now.astimezone(self.settings.timezone).date() - timedelta(days=self.settings.raw_retention_days))
                report["retention"] = {"deleted_messages": self.db.purge_old_messages(cutoff.isoformat())}

            if any(isinstance(v, dict) and v.get("errors") for v in report.values()):
                report["status"] = "partial"
        except Exception as exc:
            log.exception("Run %s failed", run_id)
            report["status"] = "failed"
            error = f"{type(exc).__name__}: {exc}"
            report["error"] = error
        finally:
            report["duration_s"] = round(time.monotonic() - started, 2)
            self.db.finish_run(run_id, report["status"], report, error)
            self.db.release_lock(LOCK_NAME, owner)
        return report


@dataclass
class App:
    settings: Settings
    db: Database
    source: Any
    llm: LLMClient | None
    sink: NotionSink | None
    pipeline: Pipeline
    next_run_at: datetime | None = field(default=None)

    async def close(self) -> None:
        close = getattr(self.source, "close", None)
        if close is not None:
            await close()
        self.db.close()


def build_app(settings: Settings) -> App:
    from .telegram_source import TelethonSource

    db = Database(settings.db_path)
    source = TelethonSource(settings)
    llm = build_llm(settings.gemini_api_key, settings.gemini_model, settings.gemini_thinking_level)
    sink = None
    if settings.notion_token and settings.notion_parent_page_id:
        sink = NotionSink(settings.notion_token, settings.notion_parent_page_id, db)
    pipeline = Pipeline(settings, db, source, llm, sink)
    return App(settings, db, source, llm, sink, pipeline)


async def track_chat(app: App, chat_id: int, include_history: bool = True) -> dict[str, Any]:
    """Start syncing a chat. include_history=False skips everything already in the chat:
    only messages written after this call are synced (useful for personal chats such as Saved Messages)."""
    info = await app.source.get_chat_info(chat_id)
    app.db.upsert_chat(info)
    if not include_history:
        app.db.start_from(chat_id, await app.source.latest_message_id(chat_id))
    app.db.set_tracked(chat_id, True)
    row = app.db.get_chat(chat_id)
    return {
        "tracked": True,
        "chat_id": chat_id,
        "title": info.title,
        "kind": info.kind,
        "history": "included (last BACKFILL_DAYS days)" if include_history and not row["floor_message_id"]
        else f"skipped: only messages with id > {row['floor_message_id']}",
    }


def status_report(app: App) -> dict[str, Any]:
    db, settings = app.db, app.settings
    chats = []
    for c in db.tracked_chats():
        chats.append(
            {
                "chat_id": c["chat_id"],
                "title": c["title"],
                "kind": c["kind"],
                "cursor_message_id": c["cursor_message_id"],
                "history_skipped_up_to": c["floor_message_id"] or None,
                "stored_messages": db.count_messages(c["chat_id"]),
                "last_synced_at": c["last_synced_at"],
                "last_error": c["last_error"],
            }
        )
    runs = []
    for r in db.recent_runs(5):
        runs.append(
            {
                "run_id": r["run_id"],
                "trigger": r["trigger"],
                "status": r["status"],
                "started_at": r["started_at"],
                "finished_at": r["finished_at"],
                "error": r["error"],
            }
        )
    holder = db.lock_holder(LOCK_NAME)
    return {
        "tracked_chats": chats,
        "pending_days": len(db.pending_days(settings.max_llm_attempts)),
        "failed_days": [dict(r) for r in db.failed_days(settings.max_llm_attempts)],
        "summaries_waiting_for_notion": len(db.summaries_to_export(RENDER_SUFFIX)),
        "llm_paused_until": (lambda t: t.isoformat(timespec="seconds") if t else None)(llm_paused_until(db, utcnow())),
        "run_in_progress": holder is not None,
        "recent_runs": runs,
        "schedule": {
            "interval_minutes": settings.sync_interval_minutes,
            "next_run_at": app.next_run_at.isoformat() if app.next_run_at else None,
        },
        "configured": {
            "telegram": bool(settings.tg_api_id and settings.tg_api_hash),
            "telegram_logged_in": settings.session_path.exists(),
            "llm": f"gemini:{settings.gemini_model}" if settings.gemini_api_key else None,
            "notion": bool(settings.notion_token and settings.notion_parent_page_id),
            "read_only": settings.read_only,
        },
    }
