"""Stage 3 of the pipeline: publish summaries to Notion (the human-readable view).

Two linked databases (see notion_layout.py) plus ready-made views (notion_views.py).
SQLite stays the source of truth; Notion is a projection of it. Idempotency:
* a summary is exported only when its export_hash (or the page layout version) changed;
* Notion page ids of days and tasks are stored locally, so an update edits the same pages;
* if the process died after Notion created a page but before its id was stored, the next run
  finds it by the unique "Key" property instead of creating a duplicate;
* the task state ("Нова → В роботі → Готово") belongs to the user: it is set only when a task is
  created, and a task that disappeared from the summary is removed only if nobody touched it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from notion_client import AsyncClient
from notion_client.errors import APIResponseError

from .db import Database
from .notion_layout import (
    DAY_PROPERTIES,
    DAYS_TITLE,
    STATUS_EMPTY,
    STATUS_FINAL,
    STATUS_PROGRESS,
    TASK_DONE,
    TASK_NEW,
    TASKS_TITLE,
    build_day_blocks,
    build_day_properties,
    build_task_properties,
    day_icon,
    select_name,
    task_key,
    task_properties_schema,
)
from .notion_views import VIEWS_API_VERSION, day_views, ensure_views, property_ids, task_views

log = logging.getLogger(__name__)

KV_DATA_SOURCE = "notion_data_source_id"
KV_DATABASE = "notion_database_id"
KV_TASKS_DATA_SOURCE = "notion_tasks_data_source_id"
KV_TASKS_DATABASE = "notion_tasks_database_id"
KV_LAYOUT = "notion_layout_version"
KV_DAY_PROP_IDS = "notion_day_property_ids"
KV_TASK_PROP_IDS = "notion_task_property_ids"
MIN_INTERVAL_S = 0.35  # Notion allows ~3 requests/s per integration

# Schema of both databases + their views. Bump to migrate existing installs once.
LAYOUT_VERSION = "1"
# Bump when the page layout changes: every page is re-rendered once on the next run, without LLM calls.
# v2: pages of days with voice notes lost their content; v3: new layout + tasks database;
# v4: deadline shown as "📅 вівторок, 24.11", weekday-only deadlines get a date;
# v5: page title = topics of the day (chat and date are already columns).
RENDER_VERSION = "5"
RENDER_SUFFIX = f":r{RENDER_VERSION}"

# Kept for callers/tests that import them from here.
build_blocks = build_day_blocks
build_properties = build_day_properties
__all__ = ["NotionSink", "build_blocks", "build_properties", "STATUS_EMPTY", "STATUS_FINAL", "STATUS_PROGRESS"]


def _renamed(props: dict[str, Any], names: dict[str, str]) -> dict[str, Any]:
    """Our property names → their current names in Notion (the user may have renamed columns)."""
    return {names.get(name, name): value for name, value in props.items()}


def _title(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": {"content": text}}]


class NotionSink:
    def __init__(self, token: str, parent_page_id: str, db: Database):
        self.client = AsyncClient(auth=token)
        self.views_client = AsyncClient(auth=token, notion_version=VIEWS_API_VERSION)
        self.parent_page_id = parent_page_id
        self.db = db
        self._last_call = 0.0

    async def _throttle(self) -> None:
        loop = asyncio.get_running_loop()
        wait = self._last_call + MIN_INTERVAL_S - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_call = loop.time()

    # ------------------------------------------------------------------ setup
    async def _create_database(self, title: str, emoji: str, properties: dict[str, Any]) -> tuple[str, str]:
        await self._throttle()
        database = await self.client.databases.create(
            parent={"type": "page_id", "page_id": self.parent_page_id},
            title=_title(title),
            icon={"type": "emoji", "emoji": emoji},
            initial_data_source={"properties": properties},
        )
        log.info("Created Notion database %s", database.get("url", database["id"]))
        return database["id"], database["data_sources"][0]["id"]

    async def ensure_data_source(self) -> str:
        cached = self.db.kv_get(KV_DATA_SOURCE)
        if cached:
            return cached
        database_id, data_source_id = await self._create_database(DAYS_TITLE, "📰", DAY_PROPERTIES)
        self.db.kv_set(KV_DATABASE, database_id)
        self.db.kv_set(KV_DATA_SOURCE, data_source_id)
        return data_source_id

    async def ensure_tasks_data_source(self, days_data_source_id: str) -> str:
        cached = self.db.kv_get(KV_TASKS_DATA_SOURCE)
        if cached:
            return cached
        database_id, data_source_id = await self._create_database(
            TASKS_TITLE, "✅", task_properties_schema(days_data_source_id)
        )
        self.db.kv_set(KV_TASKS_DATABASE, database_id)
        self.db.kv_set(KV_TASKS_DATA_SOURCE, data_source_id)
        return data_source_id

    async def ensure_layout(self) -> None:
        """Both databases with the current schema, plus the ready-made views. Runs once per LAYOUT_VERSION;
        older installs are migrated in place (new properties are added, nothing is removed)."""
        days_ds = await self.ensure_data_source()
        await self.property_names(days_ds, KV_DAY_PROP_IDS, DAY_PROPERTIES)
        await self._throttle()
        await self.client.databases.update(self.db.kv_get(KV_DATABASE), icon={"type": "emoji", "emoji": "📰"})
        tasks_ds = await self.ensure_tasks_data_source(days_ds)
        await self.property_names(tasks_ds, KV_TASK_PROP_IDS, task_properties_schema(days_ds))
        try:
            await self._throttle()
            days = await self.client.data_sources.retrieve(days_ds)
            await self._throttle()
            tasks = await self.client.data_sources.retrieve(tasks_ds)
            made = await ensure_views(
                self.views_client, self.db.kv_get(KV_DATABASE), days_ds, day_views(property_ids(days)), self._throttle
            )
            made += await ensure_views(
                self.views_client, self.db.kv_get(KV_TASKS_DATABASE), tasks_ds, task_views(property_ids(tasks)),
                self._throttle,
            )
            if made:
                log.info("Notion views ready: %s", ", ".join(made))
        except APIResponseError as exc:
            # Views are a convenience: summaries are still exported; the setup is retried next run.
            log.warning("Could not set up Notion views (%s); will retry next run", exc)
            return
        self.db.kv_set(KV_LAYOUT, LAYOUT_VERSION)

    async def property_names(self, data_source_id: str, kv_key: str, expected: dict[str, Any]) -> dict[str, str]:
        """Map our property names to their CURRENT names in Notion, following renames through property ids.

        Columns belong to the user: renaming one must not break the sync (seen live: «Чат» → «Категорія»).
        A deleted column is added back; columns the user added themselves are never touched. Installs made
        before ids were stored are bootstrapped once: a missing column matched by exactly one unknown
        column of the same type is treated as renamed."""
        await self._throttle()
        props = (await self.client.data_sources.retrieve(data_source_id))["properties"]
        name_by_id = {p["id"]: name for name, p in props.items()}
        stored: dict[str, str] = json.loads(self.db.kv_get(kv_key) or "{}")
        names: dict[str, str] = {}
        ids: dict[str, str] = {}
        missing: dict[str, Any] = {}
        for ours, spec in expected.items():
            pid = stored.get(ours)
            if pid and pid in name_by_id:
                names[ours], ids[ours] = name_by_id[pid], pid
            elif not pid and ours in props:
                names[ours], ids[ours] = ours, props[ours]["id"]
            else:
                missing[ours] = spec
        for ours in list(missing):
            if stored.get(ours):
                continue  # we knew its id and it is gone: the user deleted it → add it back
            kind = next(iter(missing[ours]))
            taken = set(ids.values()) | set(stored.values())
            candidates = [n for n, p in props.items() if p["type"] == kind and n not in expected and p["id"] not in taken]
            if len(candidates) == 1:
                names[ours], ids[ours] = candidates[0], props[candidates[0]]["id"]
                log.info("Notion column «%s» was renamed to «%s» — following the rename", ours, candidates[0])
                del missing[ours]
        if missing:
            await self._throttle()
            await self.client.data_sources.update(data_source_id, properties=missing)
            log.info("Added Notion properties: %s", ", ".join(missing))
            await self._throttle()
            props = (await self.client.data_sources.retrieve(data_source_id))["properties"]
            for ours in missing:
                names[ours], ids[ours] = ours, props[ours]["id"]
        if ids != stored:
            self.db.kv_set(kv_key, json.dumps(ids, ensure_ascii=False))
        return names

    # ------------------------------------------------------------------ pages
    async def _find_page(self, data_source_id: str, key: str, key_property: str = "Key") -> str | None:
        await self._throttle()
        result = await self.client.data_sources.query(
            data_source_id, filter={"property": key_property, "rich_text": {"equals": key}}, page_size=1
        )
        pages = result.get("results", [])
        return pages[0]["id"] if pages else None

    async def _replace_content(self, page_id: str, blocks: list[dict[str, Any]]) -> None:
        cursor = None
        old_ids: list[str] = []
        while True:
            await self._throttle()
            kwargs = {"start_cursor": cursor} if cursor else {}
            children = await self.client.blocks.children.list(page_id, **kwargs)
            old_ids.extend(b["id"] for b in children.get("results", []))
            if not children.get("has_more"):
                break
            cursor = children.get("next_cursor")
        for block_id in old_ids:
            await self._throttle()
            await self.client.blocks.delete(block_id)
        await self._throttle()
        await self.client.blocks.children.append(page_id, children=blocks)

    async def _sync_tasks(
        self, row: Any, day_page_id: str, tasks_ds: str, stats: dict[str, Any], names: dict[str, str]
    ) -> list[tuple[dict[str, Any], str]]:
        """One row per action item in the tasks database; returns (item, task page id) for the day page."""
        chat_id, local_date = row["chat_id"], row["local_date"]
        summary = json.loads(row["summary_json"])
        items = summary.get("action_items", []) if row["status"] == "ok" else []
        known = self.db.notion_tasks_for_day(chat_id, local_date)
        linked: list[tuple[dict[str, Any], str]] = []
        seen: dict[str, str] = {}
        for item in items:
            key = task_key(chat_id, local_date, item)
            if key in seen:
                linked.append((item, seen[key]))
                continue
            page_id = known.get(key) or await self._find_page(tasks_ds, key, names["Key"])
            await self._throttle()
            if page_id:
                await self.client.pages.update(
                    page_id, properties=_renamed(build_task_properties(row, item, key, day_page_id, new=False), names)
                )
                stats["tasks_updated"] += 1
            else:
                page = await self.client.pages.create(
                    parent={"type": "data_source_id", "data_source_id": tasks_ds},
                    properties=_renamed(build_task_properties(row, item, key, day_page_id, new=True), names),
                )
                page_id = page["id"]
                stats["tasks_created"] += 1
            self.db.save_notion_task(chat_id, local_date, key, page_id)
            seen[key] = page_id
            linked.append((item, page_id))
        for key, page_id in known.items():
            if key in seen:
                continue
            await self._throttle()
            page = await self.client.pages.retrieve(page_id)
            state = ((page.get("properties", {}).get(names["Стан"]) or {}).get("select") or {}).get("name")
            if state in (None, TASK_NEW) and not page.get("in_trash"):
                await self._throttle()
                await self.client.pages.update(page_id, in_trash=True)
                stats["tasks_removed"] += 1
            # A task someone already moved forward stays in Notion even if the summary no longer lists it.
            self.db.forget_notion_task(chat_id, local_date, key)
        return linked

    async def export_pending(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "created": 0, "updated": 0, "skipped_empty": 0, "failed": 0,
            "tasks_created": 0, "tasks_updated": 0, "tasks_removed": 0, "errors": [],
        }
        if self.db.kv_get(KV_LAYOUT) != LAYOUT_VERSION:
            await self.ensure_layout()
        rows = self.db.summaries_to_export(RENDER_SUFFIX)
        if not rows:
            return stats
        days_ds = await self.ensure_data_source()
        tasks_ds = await self.ensure_tasks_data_source(days_ds)
        day_names = await self.property_names(days_ds, KV_DAY_PROP_IDS, DAY_PROPERTIES)
        task_names = await self.property_names(tasks_ds, KV_TASK_PROP_IDS, task_properties_schema(days_ds))
        for row in rows:
            key = f"{row['chat_id']}:{row['local_date']}"
            try:
                page_id = row["notion_page_id"] or await self._find_page(days_ds, key, day_names["Key"])
                summary = json.loads(row["summary_json"])
                if page_id is None and row["status"] != "ok" and not summary.get("not_transcribed"):
                    # Don't clutter Notion with empty days (a day of only voice messages still gets a page).
                    self.db.set_notion_page(row["chat_id"], row["local_date"], "", row["export_hash"] + RENDER_SUFFIX)
                    stats["skipped_empty"] += 1
                    continue
                props, icon = _renamed(build_day_properties(row), day_names), day_icon(row)
                await self._throttle()
                if page_id:
                    await self.client.pages.update(page_id, properties=props, icon=icon)
                    stats["updated"] += 1
                else:
                    page = await self.client.pages.create(
                        parent={"type": "data_source_id", "data_source_id": days_ds}, properties=props, icon=icon
                    )
                    page_id = page["id"]
                    stats["created"] += 1
                    self.db.set_notion_page(row["chat_id"], row["local_date"], page_id, None)  # before tasks: crash-safe
                tasks = await self._sync_tasks(row, page_id, tasks_ds, stats, task_names)
                await self._replace_content(page_id, build_day_blocks(row, tasks))
                self.db.set_notion_page(row["chat_id"], row["local_date"], page_id, row["export_hash"] + RENDER_SUFFIX)
            except APIResponseError as exc:
                stats["failed"] += 1
                stats["errors"].append({"key": key, "error": f"{exc.code}: {exc}"})
                log.warning("Notion export failed for %s: %s", key, exc)
        return stats

    async def open_tasks(self, chat_name: str) -> list[dict[str, Any]] | None:
        """Tasks of one chat that are not «Готово», as the user sees them in Notion (the state lives there).
        None if Notion is not set up or not reachable — the caller then falls back to local summaries."""
        tasks_ds = self.db.kv_get(KV_TASKS_DATA_SOURCE)
        days_ds = self.db.kv_get(KV_DATA_SOURCE)
        if not tasks_ds or not days_ds:
            return None
        try:
            n = await self.property_names(tasks_ds, KV_TASK_PROP_IDS, task_properties_schema(days_ds))
            await self._throttle()
            result = await self.client.data_sources.query(
                tasks_ds,
                filter={"and": [
                    {"property": n["Чат"], "select": {"equals": select_name(chat_name)}},
                    {"property": n["Стан"], "select": {"does_not_equal": TASK_DONE}},
                ]},
                sorts=[{"property": n["Дедлайн"], "direction": "ascending"}],
                page_size=50,
            )
        except APIResponseError as exc:
            log.warning("Could not read tasks from Notion: %s", exc)
            return None
        tasks = []
        for page in result.get("results", []):
            props = page.get("properties", {})

            def text(ours: str) -> str:
                prop = props.get(n[ours]) or {}
                return "".join(t.get("plain_text") or t.get("text", {}).get("content", "") for t in prop.get("rich_text") or prop.get("title") or [])

            def select(ours: str) -> str | None:
                return ((props.get(n[ours]) or {}).get("select") or {}).get("name")

            tasks.append({
                "task": text("Задача"),
                "owner": select("Виконавець"),
                "due_date": ((props.get(n["Дедлайн"]) or {}).get("date") or {}).get("start"),
                "due": text("Термін") or None,
                "state": select("Стан"),
                "url": page.get("url"),
            })
        return tasks

    async def check_access(self) -> str:
        await self._throttle()
        page = await self.client.pages.retrieve(self.parent_page_id)
        return page.get("url", self.parent_page_id)
