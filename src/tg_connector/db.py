"""SQLite storage — the single source of truth for sync state.

Reliability rules implemented here:
* messages are keyed by (chat_id, message_id): re-fetching a message never creates a duplicate;
* the per-chat cursor is advanced in the SAME transaction that stores the messages
  (atomic checkpoint), and only ever moves forward;
* every change to a (chat, day) bumps `days.rev`; a day is "dirty" while rev > summarized_rev,
  so a summary written for an older revision never hides newer messages;
* a lease lock (with expiry) makes concurrent runs from the MCP server, the scheduler
  and the CLI mutually exclusive, and a crashed process cannot block sync forever.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import ChatInfo, PreparedMessage

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    chat_id           INTEGER PRIMARY KEY,
    title             TEXT    NOT NULL,
    kind              TEXT    NOT NULL,
    tracked           INTEGER NOT NULL DEFAULT 0,
    cursor_message_id INTEGER NOT NULL DEFAULT 0,
    floor_message_id  INTEGER NOT NULL DEFAULT 0,  -- messages with id <= floor are never read ("track from now")
    last_synced_at    TEXT,
    last_error        TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    chat_id      INTEGER NOT NULL,
    message_id   INTEGER NOT NULL,
    sent_at      TEXT    NOT NULL,
    local_date   TEXT    NOT NULL,
    sender_id    INTEGER,
    sender_name  TEXT    NOT NULL,
    kind         TEXT    NOT NULL,
    text         TEXT    NOT NULL DEFAULT '',
    reply_to_id  INTEGER,
    edit_date    TEXT,
    is_noise     INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT    NOT NULL,
    stored_at    TEXT    NOT NULL,
    PRIMARY KEY (chat_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_day ON messages (chat_id, local_date);

CREATE TABLE IF NOT EXISTS days (
    chat_id        INTEGER NOT NULL,
    local_date     TEXT    NOT NULL,
    rev            INTEGER NOT NULL DEFAULT 1,
    summarized_rev INTEGER NOT NULL DEFAULT 0,
    attempts       INTEGER NOT NULL DEFAULT 0,
    last_error     TEXT,
    updated_at     TEXT    NOT NULL,
    PRIMARY KEY (chat_id, local_date)
);

CREATE TABLE IF NOT EXISTS summaries (
    chat_id            INTEGER NOT NULL,
    local_date         TEXT    NOT NULL,
    status             TEXT    NOT NULL,      -- ok | no_signal
    is_final           INTEGER NOT NULL,      -- 0 while the day is still in progress
    summary_json       TEXT    NOT NULL,
    input_hash         TEXT    NOT NULL,      -- hash of the messages the summary was built from
    export_hash        TEXT    NOT NULL,      -- hash of what is shown in Notion
    model              TEXT    NOT NULL,
    prompt_version     TEXT    NOT NULL,
    signal_messages    INTEGER NOT NULL,
    noise_messages     INTEGER NOT NULL,
    created_at         TEXT    NOT NULL,
    updated_at         TEXT    NOT NULL,
    notion_page_id     TEXT,
    notion_synced_hash TEXT,
    PRIMARY KEY (chat_id, local_date)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger     TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL,
    stats_json  TEXT,
    error       TEXT
);

CREATE TABLE IF NOT EXISTS locks (
    name       TEXT PRIMARY KEY,
    owner      TEXT NOT NULL,
    expires_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS notion_tasks (
    chat_id    INTEGER NOT NULL,
    local_date TEXT    NOT NULL,
    task_key   TEXT    NOT NULL,      -- chat_id:date:hash(task, owner)
    page_id    TEXT    NOT NULL,
    PRIMARY KEY (chat_id, local_date, task_key)
);

CREATE TABLE IF NOT EXISTS notion_decisions (
    chat_id      INTEGER NOT NULL,
    local_date   TEXT    NOT NULL,
    decision_key TEXT    NOT NULL,
    page_id      TEXT    NOT NULL,
    PRIMARY KEY (chat_id, local_date, decision_key)
);

CREATE TABLE IF NOT EXISTS project_briefs (
    project_page_id TEXT PRIMARY KEY,
    brief_json      TEXT NOT NULL,
    updated_on      TEXT NOT NULL,       -- local date of the last daily update
    input_hash      TEXT NOT NULL,       -- JSON {chat_id:date -> hash}: the day versions the brief contains
    model           TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class StoreResult:
    inserted: int
    updated: int
    unchanged: int
    dirty_days: frozenset[str]


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout=30000")
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._migrate()
        if self.path != ":memory:":
            os.chmod(self.path, 0o600)  # chat contents are personal data

    def _migrate(self) -> None:
        columns = {r["name"] for r in self.conn.execute("PRAGMA table_info(chats)")}
        if "floor_message_id" not in columns:
            self.conn.execute("ALTER TABLE chats ADD COLUMN floor_message_id INTEGER NOT NULL DEFAULT 0")

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Write transaction. BEGIN IMMEDIATE takes the write lock up front, so two
        processes can never interleave a read-modify-write on the same rows."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    # ------------------------------------------------------------------ chats
    def upsert_chat(self, info: ChatInfo) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO chats (chat_id, title, kind) VALUES (?, ?, ?)
                   ON CONFLICT(chat_id) DO UPDATE SET title = excluded.title, kind = excluded.kind""",
                (info.chat_id, info.title, info.kind),
            )

    def set_tracked(self, chat_id: int, tracked: bool) -> bool:
        with self.tx() as c:
            cur = c.execute("UPDATE chats SET tracked = ? WHERE chat_id = ?", (int(tracked), chat_id))
            return cur.rowcount > 0

    def start_from(self, chat_id: int, message_id: int) -> None:
        """Skip the chat's history: only messages with id > message_id will ever be read."""
        with self.tx() as c:
            c.execute(
                """UPDATE chats SET cursor_message_id = MAX(cursor_message_id, ?),
                                     floor_message_id  = MAX(floor_message_id, ?) WHERE chat_id = ?""",
                (message_id, message_id, chat_id),
            )

    def get_chat(self, chat_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM chats WHERE chat_id = ?", (chat_id,)).fetchone()

    def all_chats(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM chats ORDER BY title").fetchall()

    def tracked_chats(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM chats WHERE tracked = 1 ORDER BY title").fetchall()

    def tracked_ids(self) -> set[int]:
        return {r["chat_id"] for r in self.conn.execute("SELECT chat_id FROM chats WHERE tracked = 1")}

    def mark_chat_synced(self, chat_id: int, error: str | None) -> None:
        with self.tx() as c:
            if error is None:
                c.execute(
                    "UPDATE chats SET last_synced_at = ?, last_error = NULL WHERE chat_id = ?",
                    (utcnow_iso(), chat_id),
                )
            else:
                c.execute("UPDATE chats SET last_error = ? WHERE chat_id = ?", (error[:500], chat_id))

    # --------------------------------------------------------------- messages
    def store_batch(self, chat_id: int, batch: list[PreparedMessage], new_cursor: int | None) -> StoreResult:
        """Idempotently store a batch and (optionally) advance the cursor — one transaction."""
        inserted = updated = unchanged = 0
        dirty: set[str] = set()
        now = utcnow_iso()
        with self.tx() as c:
            for pm in batch:
                m = pm.raw
                row = c.execute(
                    "SELECT content_hash, local_date FROM messages WHERE chat_id = ? AND message_id = ?",
                    (chat_id, m.message_id),
                ).fetchone()
                values = (
                    m.sent_at.isoformat(),
                    pm.local_date,
                    m.sender_id,
                    m.sender_name,
                    m.kind,
                    m.text,
                    m.reply_to_id,
                    m.edit_date.isoformat() if m.edit_date else None,
                    int(pm.is_noise),
                    pm.content_hash,
                )
                if row is None:
                    c.execute(
                        """INSERT INTO messages (chat_id, message_id, sent_at, local_date, sender_id, sender_name,
                               kind, text, reply_to_id, edit_date, is_noise, content_hash, stored_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (chat_id, m.message_id, *values, now),
                    )
                    inserted += 1
                    dirty.add(pm.local_date)
                elif row["content_hash"] != pm.content_hash:
                    c.execute(
                        """UPDATE messages SET sent_at = ?, local_date = ?, sender_id = ?, sender_name = ?, kind = ?,
                               text = ?, reply_to_id = ?, edit_date = ?, is_noise = ?, content_hash = ?
                           WHERE chat_id = ? AND message_id = ?""",
                        (*values, chat_id, m.message_id),
                    )
                    updated += 1
                    dirty.add(pm.local_date)
                    dirty.add(row["local_date"])
                else:
                    unchanged += 1
            for day in dirty:
                c.execute(
                    """INSERT INTO days (chat_id, local_date, rev, summarized_rev, attempts, updated_at)
                       VALUES (?, ?, 1, 0, 0, ?)
                       ON CONFLICT(chat_id, local_date) DO UPDATE SET
                           rev = days.rev + 1, attempts = 0, last_error = NULL, updated_at = excluded.updated_at""",
                    (chat_id, day, now),
                )
            if new_cursor is not None:
                c.execute(
                    "UPDATE chats SET cursor_message_id = MAX(cursor_message_id, ?) WHERE chat_id = ?",
                    (new_cursor, chat_id),
                )
        return StoreResult(inserted, updated, unchanged, frozenset(dirty))

    def delete_messages(self, chat_id: int, message_ids: set[int]) -> frozenset[str]:
        """Remove messages deleted in Telegram and mark their days dirty — one transaction."""
        if not message_ids:
            return frozenset()
        now = utcnow_iso()
        ids = sorted(message_ids)
        marks = ",".join("?" * len(ids))
        with self.tx() as c:
            days = {
                r["local_date"]
                for r in c.execute(
                    f"SELECT DISTINCT local_date FROM messages WHERE chat_id = ? AND message_id IN ({marks})",
                    (chat_id, *ids),
                )
            }
            c.execute(f"DELETE FROM messages WHERE chat_id = ? AND message_id IN ({marks})", (chat_id, *ids))
            for day in days:
                c.execute(
                    """UPDATE days SET rev = rev + 1, attempts = 0, last_error = NULL, updated_at = ?
                       WHERE chat_id = ? AND local_date = ?""",
                    (now, chat_id, day),
                )
        return frozenset(days)

    def message_ids_in_window(self, chat_id: int, since_iso: str, min_id: int, max_id: int) -> set[int]:
        rows = self.conn.execute(
            """SELECT message_id FROM messages
               WHERE chat_id = ? AND sent_at >= ? AND message_id > ? AND message_id <= ?""",
            (chat_id, since_iso, min_id, max_id),
        )
        return {r["message_id"] for r in rows}

    def messages_by_ids(self, chat_id: int, message_ids: set[int]) -> dict[int, sqlite3.Row]:
        if not message_ids:
            return {}
        ids = sorted(message_ids)
        rows = self.conn.execute(
            f"SELECT * FROM messages WHERE chat_id = ? AND message_id IN ({','.join('?' * len(ids))})",
            (chat_id, *ids),
        )
        return {r["message_id"]: r for r in rows}

    def purge_old_messages(self, before_date: str) -> int:
        """Data minimization for hosted installs: drop raw messages of days that are finished AND summarized.
        Summaries stay; the cursor keeps the sync incremental; edits older than the edit window are
        not tracked anyway, so nothing that the pipeline still needs is removed."""
        with self.tx() as c:
            cur = c.execute(
                """DELETE FROM messages WHERE local_date < ? AND EXISTS (
                       SELECT 1 FROM days d JOIN summaries s ON s.chat_id = d.chat_id AND s.local_date = d.local_date
                       WHERE d.chat_id = messages.chat_id AND d.local_date = messages.local_date
                         AND d.summarized_rev >= d.rev AND s.is_final = 1)""",
                (before_date,),
            )
            return cur.rowcount

    def snapshot(self, target: str) -> None:
        """A consistent single-file copy of the database (the live one runs in WAL mode)."""
        self.conn.execute("VACUUM INTO ?", (target,))

    def recent_signal_texts(self, chat_id: int, limit: int = 200) -> list[str]:
        rows = self.conn.execute(
            "SELECT text FROM messages WHERE chat_id = ? AND is_noise = 0 ORDER BY message_id DESC LIMIT ?",
            (chat_id, limit),
        )
        return [r["text"] for r in rows]

    def messages_for_day(self, chat_id: int, local_date: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM messages WHERE chat_id = ? AND local_date = ? ORDER BY sent_at, message_id",
            (chat_id, local_date),
        ).fetchall()

    def count_messages(self, chat_id: int | None = None) -> int:
        if chat_id is None:
            return self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,)).fetchone()[0]

    # ------------------------------------------------------------------- days
    def pending_days(self, max_attempts: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT d.chat_id, d.local_date, d.rev, d.attempts, c.title, c.kind
               FROM days d JOIN chats c ON c.chat_id = d.chat_id
               WHERE c.tracked = 1 AND d.rev > d.summarized_rev AND d.attempts < ?
               ORDER BY d.local_date, d.chat_id""",
            (max_attempts,),
        ).fetchall()

    def failed_days(self, max_attempts: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT chat_id, local_date, attempts, last_error FROM days
               WHERE rev > summarized_rev AND attempts >= ? ORDER BY local_date""",
            (max_attempts,),
        ).fetchall()

    def days_to_finalize(self, today: str) -> list[sqlite3.Row]:
        """Summaries written while the day was in progress, for days that are now over
        and have no new messages — they only need the 'final' flag, not a new LLM call."""
        return self.conn.execute(
            """SELECT s.chat_id, s.local_date FROM summaries s
               JOIN days d ON d.chat_id = s.chat_id AND d.local_date = s.local_date
               WHERE s.is_final = 0 AND s.local_date < ? AND d.rev = d.summarized_rev""",
            (today,),
        ).fetchall()

    def mark_day_done(self, chat_id: int, local_date: str, rev: int) -> None:
        with self.tx() as c:
            c.execute(
                """UPDATE days SET summarized_rev = MAX(summarized_rev, ?), attempts = 0, last_error = NULL,
                       updated_at = ? WHERE chat_id = ? AND local_date = ?""",
                (rev, utcnow_iso(), chat_id, local_date),
            )

    def mark_day_failed(self, chat_id: int, local_date: str, error: str, count_attempt: bool = True) -> None:
        """count_attempt=False for temporary errors (quota, overload, timeout): a day-long outage must not
        exhaust MAX_LLM_ATTEMPTS — only errors that repeat with the same input (bad output) give up."""
        with self.tx() as c:
            c.execute(
                """UPDATE days SET attempts = attempts + ?, last_error = ?, updated_at = ?
                   WHERE chat_id = ? AND local_date = ?""",
                (int(count_attempt), error[:500], utcnow_iso(), chat_id, local_date),
            )

    # -------------------------------------------------------------- summaries
    def get_summary(self, chat_id: int, local_date: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM summaries WHERE chat_id = ? AND local_date = ?", (chat_id, local_date)
        ).fetchone()

    def save_summary(
        self,
        *,
        chat_id: int,
        local_date: str,
        rev: int,
        status: str,
        is_final: bool,
        summary: dict[str, Any],
        input_hash: str,
        export_hash: str,
        model: str,
        prompt_version: str,
        signal_messages: int,
        noise_messages: int,
    ) -> None:
        """Upsert the summary and mark the day's revision as summarized — atomically."""
        now = utcnow_iso()
        with self.tx() as c:
            c.execute(
                """INSERT INTO summaries (chat_id, local_date, status, is_final, summary_json, input_hash, export_hash,
                       model, prompt_version, signal_messages, noise_messages, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(chat_id, local_date) DO UPDATE SET
                       status = excluded.status, is_final = excluded.is_final, summary_json = excluded.summary_json,
                       input_hash = excluded.input_hash, export_hash = excluded.export_hash, model = excluded.model,
                       prompt_version = excluded.prompt_version, signal_messages = excluded.signal_messages,
                       noise_messages = excluded.noise_messages, updated_at = excluded.updated_at""",
                (
                    chat_id,
                    local_date,
                    status,
                    int(is_final),
                    json.dumps(summary, ensure_ascii=False, sort_keys=True),
                    input_hash,
                    export_hash,
                    model,
                    prompt_version,
                    signal_messages,
                    noise_messages,
                    now,
                    now,
                ),
            )
            c.execute(
                """UPDATE days SET summarized_rev = MAX(summarized_rev, ?), attempts = 0, last_error = NULL,
                       updated_at = ? WHERE chat_id = ? AND local_date = ?""",
                (rev, now, chat_id, local_date),
            )

    def finalize_summary(self, chat_id: int, local_date: str, export_hash: str) -> None:
        with self.tx() as c:
            c.execute(
                """UPDATE summaries SET is_final = 1, export_hash = ?, updated_at = ?
                   WHERE chat_id = ? AND local_date = ?""",
                (export_hash, utcnow_iso(), chat_id, local_date),
            )

    def list_summaries(
        self, chat_id: int | None = None, date_from: str | None = None, date_to: str | None = None, limit: int = 50
    ) -> list[sqlite3.Row]:
        sql = """SELECT s.*, c.title, c.kind FROM summaries s JOIN chats c ON c.chat_id = s.chat_id WHERE 1 = 1"""
        params: list[Any] = []
        if chat_id is not None:
            sql += " AND s.chat_id = ?"
            params.append(chat_id)
        if date_from:
            sql += " AND s.local_date >= ?"
            params.append(date_from)
        if date_to:
            sql += " AND s.local_date <= ?"
            params.append(date_to)
        sql += " ORDER BY s.local_date DESC, c.title LIMIT ?"
        params.append(limit)
        return self.conn.execute(sql, params).fetchall()

    def summaries_to_export(self, *accepted: str) -> list[sqlite3.Row]:
        """Summaries whose Notion page is missing or outdated: the content changed (export_hash) or the page
        was made by a layout version that is not among the `accepted` suffixes."""
        accepted = accepted or ("",)
        marks = ", ".join(["s.export_hash || ?"] * len(accepted))
        return self.conn.execute(
            f"""SELECT s.*, c.title, c.kind FROM summaries s JOIN chats c ON c.chat_id = s.chat_id
               WHERE s.notion_synced_hash IS NULL OR s.notion_synced_hash NOT IN ({marks})
               ORDER BY s.local_date, s.chat_id""",
            accepted,
        ).fetchall()

    def set_notion_page(self, chat_id: int, local_date: str, page_id: str, synced_hash: str | None) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE summaries SET notion_page_id = ?, notion_synced_hash = ? WHERE chat_id = ? AND local_date = ?",
                (page_id, synced_hash, chat_id, local_date),
            )

    def requeue_export(self, chat_id: int) -> int:
        """Export all days of a chat again (e.g. its project changed); the pages are updated in place."""
        with self.tx() as c:
            return c.execute(
                "UPDATE summaries SET notion_synced_hash = NULL WHERE chat_id = ? AND notion_synced_hash IS NOT NULL", (chat_id,)
            ).rowcount

    def notion_tasks_for_day(self, chat_id: int, local_date: str) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT task_key, page_id FROM notion_tasks WHERE chat_id = ? AND local_date = ?", (chat_id, local_date)
        )
        return {r["task_key"]: r["page_id"] for r in rows}

    def save_notion_task(self, chat_id: int, local_date: str, task_key: str, page_id: str) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO notion_tasks (chat_id, local_date, task_key, page_id) VALUES (?, ?, ?, ?)
                   ON CONFLICT(chat_id, local_date, task_key) DO UPDATE SET page_id = excluded.page_id""",
                (chat_id, local_date, task_key, page_id),
            )

    def forget_notion_task(self, chat_id: int, local_date: str, task_key: str) -> None:
        with self.tx() as c:
            c.execute(
                "DELETE FROM notion_tasks WHERE chat_id = ? AND local_date = ? AND task_key = ?",
                (chat_id, local_date, task_key),
            )

    def notion_decisions_for_day(self, chat_id: int, local_date: str) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT decision_key, page_id FROM notion_decisions WHERE chat_id = ? AND local_date = ?", (chat_id, local_date)
        )
        return {r["decision_key"]: r["page_id"] for r in rows}

    def save_notion_decision(self, chat_id: int, local_date: str, key: str, page_id: str) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO notion_decisions (chat_id, local_date, decision_key, page_id) VALUES (?, ?, ?, ?)
                   ON CONFLICT(chat_id, local_date, decision_key) DO UPDATE SET page_id = excluded.page_id""",
                (chat_id, local_date, key, page_id),
            )

    def forget_notion_decision(self, chat_id: int, local_date: str, key: str) -> None:
        with self.tx() as c:
            c.execute(
                "DELETE FROM notion_decisions WHERE chat_id = ? AND local_date = ? AND decision_key = ?",
                (chat_id, local_date, key),
            )

    def count_notion_decisions(self, chat_ids: set[int]) -> int:
        if not chat_ids:
            return 0
        ids = sorted(chat_ids)
        return self.conn.execute(
            f"SELECT count(*) FROM notion_decisions WHERE chat_id IN ({','.join('?' * len(ids))})", ids
        ).fetchone()[0]

    def get_project_brief(self, page_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM project_briefs WHERE project_page_id = ?", (page_id,)).fetchone()

    def save_project_brief(self, page_id: str, brief: dict[str, Any], updated_on: str, input_hash: str, model: str) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO project_briefs (project_page_id, brief_json, updated_on, input_hash, model, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(project_page_id) DO UPDATE SET brief_json = excluded.brief_json,
                       updated_on = excluded.updated_on, input_hash = excluded.input_hash, model = excluded.model,
                       updated_at = excluded.updated_at""",
                (page_id, json.dumps(brief, ensure_ascii=False, sort_keys=True), updated_on, input_hash, model, utcnow_iso()),
            )

    def mark_project_brief_checked(self, page_id: str, updated_on: str) -> None:
        """Nothing new in the project's chats today: remember that the day was checked (no LLM call)."""
        with self.tx() as c:
            c.execute("UPDATE project_briefs SET updated_on = ? WHERE project_page_id = ?", (updated_on, page_id))

    # ------------------------------------------------------------------- runs
    def start_run(self, trigger: str) -> int:
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO runs (trigger, started_at, status) VALUES (?, ?, 'running')", (trigger, utcnow_iso())
            )
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str, stats: dict[str, Any], error: str | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE runs SET finished_at = ?, status = ?, stats_json = ?, error = ? WHERE run_id = ?",
                (utcnow_iso(), status, json.dumps(stats, ensure_ascii=False), error, run_id),
            )

    def recent_runs(self, limit: int = 5) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM runs ORDER BY run_id DESC LIMIT ?", (limit,)).fetchall()

    # ------------------------------------------------------------------ locks
    def try_acquire_lock(self, name: str, owner: str, ttl_seconds: int) -> bool:
        now = time.time()
        with self.tx() as c:
            row = c.execute("SELECT owner, expires_at FROM locks WHERE name = ?", (name,)).fetchone()
            if row is not None and row["owner"] != owner and row["expires_at"] > now:
                return False
            c.execute(
                "INSERT OR REPLACE INTO locks (name, owner, expires_at) VALUES (?, ?, ?)",
                (name, owner, now + ttl_seconds),
            )
            return True

    def refresh_lock(self, name: str, owner: str, ttl_seconds: int) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE locks SET expires_at = ? WHERE name = ? AND owner = ?",
                (time.time() + ttl_seconds, name, owner),
            )

    def release_lock(self, name: str, owner: str) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM locks WHERE name = ? AND owner = ?", (name, owner))

    def lock_holder(self, name: str) -> sqlite3.Row | None:
        row = self.conn.execute("SELECT owner, expires_at FROM locks WHERE name = ?", (name,)).fetchone()
        if row is None or row["expires_at"] <= time.time():
            return None
        return row

    # --------------------------------------------------------------------- kv
    def kv_get(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def kv_set(self, key: str, value: str) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, value))
