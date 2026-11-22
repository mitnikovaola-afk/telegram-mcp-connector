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
from urllib.parse import unquote

from notion_client import AsyncClient
from notion_client.errors import APIResponseError

from .db import Database
from .notion_layout import (
    LOG_AI_COLUMNS,
    LOG_EFFECTS,
    LOG_TITLE,
    LOG_TYPES,
    PROJECTS_TITLE,
    SERVICE_TITLE,
    build_brief_children,
    build_brief_header,
    build_log_properties,
    build_new_project_properties,
    build_service_page,
    build_team_section,
    checklist_view_spec,
    log_properties_schema,
    plain_value,
    project_properties_schema,
    project_relation_spec,
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
from .notion_views import (
    VIEWS_API_VERSION,
    day_views,
    ensure_views,
    log_views,
    project_views,
    property_ids,
    task_views,
    view_body,
)

log = logging.getLogger(__name__)

KV_DATA_SOURCE = "notion_data_source_id"
KV_DATABASE = "notion_database_id"
KV_TASKS_DATA_SOURCE = "notion_tasks_data_source_id"
KV_TASKS_DATABASE = "notion_tasks_database_id"
KV_LAYOUT = "notion_layout_version"
KV_DAY_PROP_IDS = "notion_day_property_ids"
KV_TASK_PROP_IDS = "notion_task_property_ids"
KV_PROJECTS_DATABASE = "notion_projects_database_id"
KV_PROJECTS_DATA_SOURCE = "notion_projects_data_source_id"
KV_LOG_DATABASE = "notion_decisions_database_id"
KV_LOG_DATA_SOURCE = "notion_decisions_data_source_id"
KV_PROJECT_PROP_IDS = "notion_project_property_ids"
KV_LOG_PROP_IDS = "notion_decision_property_ids"
KV_PROJECTS_LAYOUT = "notion_projects_layout_version"
KV_DEFAULT_PROJECT = "project_default_page"
# v3: the log keeps important entries only and lives on the service page.
PROJECTS_LAYOUT_VERSION = "3"
KV_SERVICE_PAGE = "notion_service_page"
KV_TASK_PROJECT_READY = "notion_task_project_column_ready"  # the «Проєкт» column exists on tasks
MIN_INTERVAL_S = 0.35  # Notion allows ~3 requests/s per integration

# Schema of both databases + their views. Bump to migrate existing installs once.
LAYOUT_VERSION = "1"
# Bump when the page layout changes: every page is re-rendered once on the next run, without LLM calls.
# v2: pages of days with voice notes lost their content; v3: new layout + tasks database;
# v4: deadline shown as "📅 вівторок, 24.11", weekday-only deadlines get a date;
# v5: page title = topics of the day (chat and date are already columns);
# v6: the «Проєкт» relation of tasks for days exported before projects existed.
RENDER_VERSION = "6"
RENDER_SUFFIX = f":r{RENDER_VERSION}"
# Added once the tasks of a day are linked to their project. If the project databases were not available at
# that moment, the day is exported once more when they are.
PROJECTS_SUFFIX = "p"

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
    def __init__(self, token: str, parent_page_id: str, db: Database, *, projects: bool = True,
                 projects_mode: str = "per_chat", project_name: str = "Мій проєкт"):
        self.client = AsyncClient(auth=token)
        self.views_client = AsyncClient(auth=token, notion_version=VIEWS_API_VERSION)
        self.parent_page_id = parent_page_id
        self.db = db
        self.projects_enabled = projects
        self.projects_mode = projects_mode
        self.project_name = project_name
        self._last_call = 0.0

    async def _throttle(self) -> None:
        loop = asyncio.get_running_loop()
        wait = self._last_call + MIN_INTERVAL_S - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_call = loop.time()

    # ------------------------------------------------------------------ setup
    async def _create_database(self, title: str, emoji: str, properties: dict[str, Any],
                               parent_page_id: str | None = None) -> tuple[str, str]:
        await self._throttle()
        database = await self.client.databases.create(
            parent={"type": "page_id", "page_id": parent_page_id or self.parent_page_id},
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

    def _task_schema(self, days_ds: str) -> dict[str, Any]:
        """Expected task columns; «Проєкт» joins once the projects database exists."""
        schema = task_properties_schema(days_ds)
        projects_ds = self.db.kv_get(KV_PROJECTS_DATA_SOURCE)
        if self.projects_enabled and projects_ds and self.db.kv_get(KV_TASK_PROJECT_READY):
            schema["Проєкт"] = project_relation_spec(projects_ds)
        return schema

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
        await self.property_names(tasks_ds, KV_TASK_PROP_IDS, self._task_schema(days_ds))
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

    # ------------------------------------------------------------- projects (beyond the task)
    async def ensure_projects_layout(self) -> tuple[str, str] | None:
        """Projects + project-log databases, the «Проєкт» relation on tasks, and their views (once)."""
        if not self.projects_enabled:
            return None
        days_ds = await self.ensure_data_source()
        tasks_ds = await self.ensure_tasks_data_source(days_ds)
        projects_ds = self.db.kv_get(KV_PROJECTS_DATA_SOURCE)
        if not projects_ds:
            database_id, projects_ds = await self._create_database(PROJECTS_TITLE, "📁", project_properties_schema())
            self.db.kv_set(KV_PROJECTS_DATABASE, database_id)
            self.db.kv_set(KV_PROJECTS_DATA_SOURCE, projects_ds)
        log_ds = self.db.kv_get(KV_LOG_DATA_SOURCE)
        if not log_ds:
            database_id, log_ds = await self._create_database(LOG_TITLE, "📒", log_properties_schema(projects_ds),
                                                              await self._service_page())
            self.db.kv_set(KV_LOG_DATABASE, database_id)
            self.db.kv_set(KV_LOG_DATA_SOURCE, log_ds)
        if self.db.kv_get(KV_PROJECTS_LAYOUT) == PROJECTS_LAYOUT_VERSION:
            return projects_ds, log_ds
        await self._migrate_to_v3(log_ds)
        with_project = task_properties_schema(days_ds) | {"Проєкт": project_relation_spec(projects_ds)}
        await self.property_names(tasks_ds, KV_TASK_PROP_IDS, with_project)  # adds «Проєкт» to tasks
        self.db.kv_set(KV_TASK_PROJECT_READY, "1")
        await self.property_names(projects_ds, KV_PROJECT_PROP_IDS, project_properties_schema())
        await self.property_names(log_ds, KV_LOG_PROP_IDS, log_properties_schema(projects_ds))
        try:
            await ensure_views(self.views_client, self.db.kv_get(KV_PROJECTS_DATABASE), projects_ds,
                               project_views(self._view_ids(KV_PROJECT_PROP_IDS)), self._throttle)
            await ensure_views(self.views_client, self.db.kv_get(KV_LOG_DATABASE), log_ds,
                               log_views(self._view_ids(KV_LOG_PROP_IDS)), self._throttle)
        except APIResponseError as exc:
            log.warning("Could not set up project views (%s); will retry next run", exc)
            return projects_ds, log_ds
        self.db.kv_set(KV_PROJECTS_LAYOUT, PROJECTS_LAYOUT_VERSION)
        return projects_ds, log_ds

    def _view_ids(self, kv_key: str) -> dict[str, str]:
        """Our column name → property id for view filters and columns (a renamed column keeps its id)."""
        return {name: unquote(pid) for name, pid in json.loads(self.db.kv_get(kv_key) or "{}").items()}

    async def _service_page(self) -> str:
        """Page that holds the log: people see it through the project pages, so it does not clutter the main page."""
        stored = self.db.kv_get(KV_SERVICE_PAGE)
        if stored:
            return stored
        cursor, page_id = None, None
        while page_id is None:  # created by a run that died before remembering it?
            await self._throttle()
            result = await self.client.blocks.children.list(self.parent_page_id, **({"start_cursor": cursor} if cursor else {}))
            page_id = next((b["id"] for b in result.get("results", [])
                            if (b.get("child_page") or {}).get("title") == SERVICE_TITLE), None)
            if not result.get("has_more"):
                break
            cursor = result.get("next_cursor")
        if page_id is None:
            await self._throttle()
            page = await self.client.pages.create(
                parent={"type": "page_id", "page_id": self.parent_page_id}, icon={"type": "emoji", "emoji": "⚙️"},
                properties={"title": {"title": _title(SERVICE_TITLE)}}, children=build_service_page())
            page_id = page["id"]
        self.db.kv_set(KV_SERVICE_PAGE, page_id)
        return page_id

    # --- one-off: the log of the previous version (all decisions + «Важливе» checkbox, database on the main page)
    async def _migrate_to_v3(self, log_ds: str) -> None:
        database_id = self.db.kv_get(KV_LOG_DATABASE)
        ids = json.loads(self.db.kv_get(KV_LOG_PROP_IDS) or "{}")
        if "Важливе" in ids:
            await self._throttle()
            props = (await self.client.data_sources.retrieve(log_ds))["properties"]
            by_id = {p["id"]: name for name, p in props.items()}
            important, kind, key = by_id.get(ids["Важливе"]), by_id.get(ids["Тип"]), by_id.get(ids["Key"])
            while important and kind and key:  # routine decisions leave the log (they stay on the day pages)
                await self._throttle()
                result = await self.client.data_sources.query(log_ds, page_size=100, filter={"and": [
                    {"property": important, "checkbox": {"equals": False}},
                    {"property": kind, "select": {"equals": LOG_TYPES["decision"]}},
                    {"property": key, "rich_text": {"is_not_empty": True}},
                ]})
                for page in result.get("results", []):
                    await self._throttle()
                    await self.client.pages.update(page["id"], in_trash=True)
                if not result.get("has_more"):
                    break
            view_ids = {name: unquote(pid) for name, pid in ids.items() if name != "Важливе"}
            await self._throttle()
            listing = await self.views_client.request(path=f"views?database_id={database_id}", method="GET")
            specs = {s["name"]: s for s in log_views(view_ids)}
            for ref in listing.get("results", []):
                await self._throttle()
                view = await self.views_client.request(path=f"views/{ref['id']}", method="GET")
                await self._throttle()
                if view.get("name") in specs:
                    body = {"filter": None} | view_body(specs[view["name"]])
                    body.pop("type")
                    await self.views_client.request(path=f"views/{view['id']}", method="PATCH", body=body)
                elif view.get("name") == "🗂 Усі записи":
                    await self.views_client.request(path=f"views/{view['id']}", method="DELETE")
            for row in self.db.conn.execute("SELECT key, value FROM kv WHERE key LIKE 'project_views:%'").fetchall():
                page_id, stored = row["key"].partition(":")[2], json.loads(row["value"])
                for spec in log_views(view_ids, page_id):
                    if spec["key"] in stored:
                        body = view_body(spec)
                        body.pop("type")
                        await self._throttle()
                        await self.views_client.request(path=f"views/{stored[spec['key']]['id']}", method="PATCH", body=body)
            if important:
                await self._throttle()
                await self.client.data_sources.update(log_ds, properties={important: None})
            ids.pop("Важливе")
            self.db.kv_set(KV_LOG_PROP_IDS, json.dumps(ids, ensure_ascii=False))
        await self._throttle()
        parent = (await self.client.databases.retrieve(database_id)).get("parent") or {}
        service = await self._service_page()
        if (parent.get("page_id") or "").replace("-", "") == self.parent_page_id.replace("-", ""):
            await self._throttle()
            await self.client.databases.update(database_id, parent={"type": "page_id", "page_id": service})

    async def load_projects(self) -> list[dict[str, Any]]:
        """Projects as the user sees them in Notion (their «Чати» column is the chat → project mapping)."""
        projects_ds = self.db.kv_get(KV_PROJECTS_DATA_SOURCE)
        if not projects_ds:
            return []
        n = await self.property_names(projects_ds, KV_PROJECT_PROP_IDS, project_properties_schema())
        projects, cursor = [], None
        while True:
            await self._throttle()
            kwargs: dict[str, Any] = {"page_size": 100}
            if cursor:
                kwargs["start_cursor"] = cursor
            result = await self.client.data_sources.query(projects_ds, **kwargs)
            for page in result.get("results", []):
                if page.get("in_trash"):
                    continue
                props = page.get("properties", {})
                title = "".join(t.get("plain_text") or t.get("text", {}).get("content", "")
                                for t in (props.get(n["Проєкт"]) or {}).get("title", []))
                projects.append({
                    "page_id": page["id"],
                    "title": title or "Проєкт",
                    "chats": [o["name"] for o in (props.get(n["Чати"]) or {}).get("multi_select", [])],
                    "status": ((props.get(n["Статус"]) or {}).get("select") or {}).get("name"),
                    "release_date": ((props.get(n["Дата релізу"]) or {}).get("date") or {}).get("start"),
                })
            if not result.get("has_more"):
                return projects
            cursor = result.get("next_cursor")

    async def assign_chats(self, projects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Every tracked group chat gets a project ONCE (per_chat: its own; single: the default project).
        After that the «Чати» column is the user's: removing a chat there is respected."""
        projects_ds = self.db.kv_get(KV_PROJECTS_DATA_SOURCE)
        if not projects_ds:
            return projects
        n = await self.property_names(projects_ds, KV_PROJECT_PROP_IDS, project_properties_schema())
        listed = {chat for p in projects for chat in p["chats"]}
        for chat in self.db.tracked_chats():
            if chat["kind"] not in ("group", "channel") or self.db.kv_get(f"project_auto:{chat['chat_id']}"):
                continue
            name = select_name(chat["title"] or str(chat["chat_id"]))
            if name not in listed:
                default = self.db.kv_get(KV_DEFAULT_PROJECT) if self.projects_mode == "single" else None
                target = next((p for p in projects if p["page_id"] == default), None)
                await self._throttle()
                if target is not None:
                    target["chats"].append(name)
                    await self.client.pages.update(target["page_id"], properties={
                        n["Чати"]: {"multi_select": [{"name": c} for c in target["chats"]]}})
                    log.info("Chat «%s» added to project «%s»", name, target["title"])
                else:
                    title = self.project_name if self.projects_mode == "single" else name
                    page = await self.client.pages.create(
                        parent={"type": "data_source_id", "data_source_id": projects_ds},
                        properties=_renamed(build_new_project_properties(title), n)
                        | {n["Чати"]: {"multi_select": [{"name": name}]}},
                        icon={"type": "emoji", "emoji": "📁"},
                    )
                    projects.append({"page_id": page["id"], "title": title, "chats": [name],
                                     "status": None, "release_date": None})
                    if self.projects_mode == "single":
                        self.db.kv_set(KV_DEFAULT_PROJECT, page["id"])
                    log.info("Created project «%s» for chat «%s»", title, name)
                listed.add(name)
            self.db.kv_set(f"project_auto:{chat['chat_id']}", "1")
        return projects

    @staticmethod
    def projects_by_chat(projects: list[dict[str, Any]]) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for p in projects:
            for chat in p["chats"]:
                mapping.setdefault(chat, []).append(p["page_id"])
        return mapping

    def requeue_moved_chats(self, by_chat: dict[str, list[str]]) -> None:
        """The user changed a «Чати» column: days of that chat are exported again, so that the «Проєкт» links
        of its old tasks follow (no LLM calls)."""
        for chat in self.db.all_chats():
            name = chat["title"] or str(chat["chat_id"])
            links, key = ",".join(sorted(by_chat.get(select_name(name), []))), f"project_links:{chat['chat_id']}"
            before = self.db.kv_get(key)
            if before == links:
                continue
            if before is not None and (days := self.db.requeue_export(chat["chat_id"])):
                log.info("Project of chat «%s» changed: %d day(s) will be linked again", name, days)
            self.db.kv_set(key, links)

    async def _update_row(self, page_id: str, properties: dict[str, Any]) -> bool:
        """Update a row; False if the user moved it to the trash (their decision: it is not brought back)."""
        try:
            await self._throttle()
            await self.client.pages.update(page_id, properties=properties)
            return True
        except APIResponseError:
            await self._throttle()
            if not (await self.client.pages.retrieve(page_id)).get("in_trash"):
                raise
            return False

    # ------------------------------------------------------------- project log (beyond the task)
    async def _log_names(self) -> dict[str, str]:
        return await self.property_names(self.db.kv_get(KV_LOG_DATA_SOURCE), KV_LOG_PROP_IDS,
                                         log_properties_schema(self.db.kv_get(KV_PROJECTS_DATA_SOURCE)))

    async def load_log(self, project_page_id: str) -> list[dict[str, Any]]:
        """Log rows of a project as the user sees them now (with their edits and their own rows)."""
        n, ds = await self._log_names(), self.db.kv_get(KV_LOG_DATA_SOURCE)
        types = {label: key for key, label in LOG_TYPES.items()}
        effects = {label: key for key, label in LOG_EFFECTS.items()}
        rows: list[dict[str, Any]] = []
        cursor = None
        while True:
            await self._throttle()
            result = await self.client.data_sources.query(
                ds, filter={"property": n["Проєкт"], "relation": {"contains": project_page_id}},
                sorts=[{"property": n["Дата"], "direction": "ascending"}], page_size=100,
                **({"start_cursor": cursor} if cursor else {}))
            for page in result.get("results", []):
                if page.get("in_trash"):
                    continue
                v = {ours: plain_value(page.get("properties", {}).get(n[ours])) for ours in LOG_AI_COLUMNS}
                rows.append({
                    "page_id": page["id"], "date": v["Дата"], "title": v["Запис"],
                    "type": types.get(v["Тип"], v["Тип"]),
                    "result": v["Результат"] or None, "effect": effects.get(v["Ефект"], v["Ефект"]),
                    "takeaway": v["Висновок"] or None, "metric": v["Метрика"], "value": v["Значення"],
                })
            if not result.get("has_more"):
                return rows
            cursor = result.get("next_cursor")

    async def sync_log(self, project_page_id: str) -> int:
        """Write new/changed log entries to Notion. A value the user changed in Notion is never overwritten, a row
        the user removed is never created again."""
        pending = self.db.project_log(project_page_id, "dirty")
        obsolete = self.db.project_log(project_page_id, "obsolete")
        if not pending and not obsolete:
            return 0
        n, ds = await self._log_names(), self.db.kv_get(KV_LOG_DATA_SOURCE)
        done = 0
        for row in obsolete:
            done += await self._retire_row(row, n)
        for row in pending:
            entry = json.loads(row["entry_json"])
            wanted = build_log_properties(entry)
            key = f"log:{project_page_id}:{row['entry_key']}"
            page_id = row["page_id"] or await self._find_page(ds, key, n["Key"])
            written = json.loads(row["written_json"] or "{}")
            if page_id is None:
                extra = {"Проєкт": {"relation": [{"id": project_page_id}]},
                         "Key": {"rich_text": [{"type": "text", "text": {"content": key}}]}}
                await self._throttle()
                page = await self.client.pages.create(
                    parent={"type": "data_source_id", "data_source_id": ds}, properties=_renamed(wanted | extra, n))
                page_id, written = page["id"], {ours: plain_value(value) for ours, value in wanted.items()}
                self.db.log_row_created(row["id"], page_id)
            else:
                await self._throttle()
                page = await self.client.pages.retrieve(page_id)
                if page.get("in_trash"):
                    self.db.set_log_state(row["id"], "dismissed")
                    continue
                current = {ours: plain_value(page["properties"].get(n[ours])) for ours in wanted}
                changed = {ours: value for ours, value in wanted.items()
                           if plain_value(value) != current[ours] and current[ours] == written.get(ours, current[ours])}
                if changed:
                    await self._throttle()
                    await self.client.pages.update(page_id, properties=_renamed(changed, n))
                written = written | {ours: plain_value(value) for ours, value in changed.items()}
            self.db.log_row_synced(row["id"], page_id, written)
            done += 1
        return done

    async def _retire_row(self, row: Any, n: dict[str, str]) -> int:
        """The decision is no longer in its day summary: its row goes to the trash, unless the user worked on it."""
        removed = False
        if row["page_id"]:
            await self._throttle()
            page = await self.client.pages.retrieve(row["page_id"])
            written = json.loads(row["written_json"]) if row["written_json"] else None
            removed = not page.get("in_trash") and written is not None and all(
                plain_value(page["properties"].get(n[ours])) == value for ours, value in written.items())
            if removed:
                await self._throttle()
                await self.client.pages.update(row["page_id"], in_trash=True)
        self.db.retire_log_row(row["id"], row_removed=removed)
        return 1

    # ------------------------------------------------------------- project page
    async def render_project(self, project: dict[str, Any], brief: dict[str, Any] | None, updated_on: str | None,
                             model: str | None) -> None:
        """Passport block (the only part of a project page the connector rewrites), the team's template and the
        linked log table (both created once)."""
        page_id = project["page_id"]
        if brief is not None and updated_on:
            n = await self.property_names(self.db.kv_get(KV_PROJECTS_DATA_SOURCE), KV_PROJECT_PROP_IDS,
                                          project_properties_schema())
            await self._throttle()
            await self.client.pages.update(page_id, properties={n["Оновлено"]: {"date": {"start": updated_on}}})
            await self._write_brief(page_id, brief, updated_on, model or "")
        await self._ensure_team_section(page_id)
        await self._ensure_project_views(project)

    async def _write_brief(self, page_id: str, brief: dict[str, Any], updated_on: str, model: str) -> None:
        header = build_brief_header(updated_on, model)
        children = build_brief_children(brief, updated_on)
        block_id = self.db.kv_get(f"project_brief_block:{page_id}")
        if block_id:
            try:
                await self._throttle()
                await self.client.blocks.update(block_id, callout={"rich_text": header})
                await self._replace_content(block_id, children)
                return
            except APIResponseError as exc:  # the user deleted the block: create it again
                log.info("Passport block of project %s is gone (%s) — creating a new one", page_id, exc.code)
        await self._throttle()
        created = await self.client.blocks.children.append(page_id, position={"type": "start"}, children=[{
            "object": "block", "type": "callout",
            "callout": {"rich_text": header, "icon": {"type": "emoji", "emoji": "🧭"}, "color": "blue_background",
                        "children": children},
        }])
        self.db.kv_set(f"project_brief_block:{page_id}", created["results"][0]["id"])

    async def _ensure_team_section(self, page_id: str) -> None:
        """Template for what chats do not contain. Created once (right under the passport), then the team's own."""
        kv_key = f"project_team_section:{page_id}"
        if self.db.kv_get(kv_key):
            return
        passport = self.db.kv_get(f"project_brief_block:{page_id}")
        place = {"position": {"type": "after_block", "after_block": {"id": passport}}} if passport else {}
        await self._throttle()
        created = await self.client.blocks.children.append(page_id, children=build_team_section(), **place)
        self.db.kv_set(kv_key, created["results"][0]["id"])

    async def _ensure_project_views(self, project: dict[str, Any]) -> None:
        """The log table on the project page (one block, three tabs) and the pre-release checklist among the views
        of the tasks database. Created once; the checklist follows «Дата релізу» and the project's name."""
        page_id = project["page_id"]
        kv_key = f"project_views:{page_id}"
        stored = json.loads(self.db.kv_get(kv_key) or "{}")
        log_ds, tasks_ds = self.db.kv_get(KV_LOG_DATA_SOURCE), self.db.kv_get(KV_TASKS_DATA_SOURCE)
        t = await self.property_names(tasks_ds, KV_TASK_PROP_IDS, self._task_schema(self.db.kv_get(KV_DATA_SOURCE)))
        await self._log_names()
        try:
            for spec in log_views(self._view_ids(KV_LOG_PROP_IDS), page_id):
                if spec["key"] in stored:
                    continue
                body = {"data_source_id": log_ds, **view_body(spec)}
                if spec["key"] == "important":
                    body["create_database"] = {"parent": {"type": "page_id", "page_id": page_id},
                                               "show_data_source_title": False}
                elif "important" in stored:
                    body["database_id"] = stored["important"]["database"]
                else:
                    continue
                await self._throttle()
                view = await self.views_client.request(path="views", method="POST", body=body)
                stored[spec["key"]] = {"id": view["id"], "database": (view.get("parent") or {}).get("database_id")}
                self.db.kv_set(kv_key, json.dumps(stored))
                if spec["key"] == "important":
                    await self._name_table(view, "📒 Журнал проєкту")
            await self._ensure_checklist(project, stored, tasks_ds, t)
        except APIResponseError as exc:
            log.warning("Could not set up views on project «%s» (%s); will retry", project["title"], exc)
        self.db.kv_set(kv_key, json.dumps(stored))

    async def _ensure_checklist(self, project: dict[str, Any], stored: dict[str, Any], tasks_ds: str, t: dict[str, str]) -> None:
        spec = checklist_view_spec(project["page_id"], project["title"], project.get("release_date"), t)
        version = {"release": project.get("release_date"), "name": spec["name"]}
        known = stored.get("release")
        if known and {k: known.get(k) for k in version} == version:
            return
        await self._throttle()
        if known:
            await self.views_client.request(path=f"views/{known['id']}", method="PATCH",
                                            body={"name": spec["name"], "filter": spec["filter"]})
            stored["release"] = {"id": known["id"], **version}
        else:
            view = await self.views_client.request(path="views", method="POST", body={
                "database_id": self.db.kv_get(KV_TASKS_DATABASE), "data_source_id": tasks_ds, **view_body(spec)})
            stored["release"] = {"id": view["id"], **version}

    async def _name_table(self, view: dict[str, Any], title: str) -> None:
        """A linked table is born «Untitled»; its title is its own, the source database keeps its name."""
        try:
            await self._throttle()
            await self.client.databases.update(view["parent"]["database_id"], title=_title(title))
        except (APIResponseError, KeyError) as exc:
            log.info("Linked table «%s» keeps its default title (%s)", title, exc)

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
        for ours in list(missing) if not stored else []:  # with known ids a missing column was deleted or is new: add it
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
        self, row: Any, day_page_id: str, tasks_ds: str, stats: dict[str, Any], names: dict[str, str],
        project_ids: list[str] | None = None,
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
            extra = {"Проєкт": {"relation": [{"id": pid} for pid in project_ids]}} if project_ids is not None else {}
            if page_id:
                await self._update_row(
                    page_id, _renamed(build_task_properties(row, item, key, day_page_id, new=False) | extra, names)
                )
                stats["tasks_updated"] += 1
            else:
                await self._throttle()
                page = await self.client.pages.create(
                    parent={"type": "data_source_id", "data_source_id": tasks_ds},
                    properties=_renamed(build_task_properties(row, item, key, day_page_id, new=True) | extra, names),
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
        projects_setup: tuple[str, str] | None = None
        by_chat: dict[str, list[str]] | None = None
        if self.projects_enabled:
            try:
                projects_setup = await self.ensure_projects_layout()
                by_chat = self.projects_by_chat(await self.assign_chats(await self.load_projects()))
                self.requeue_moved_chats(by_chat)
            except APIResponseError as exc:  # projects are an extra: summaries and tasks go on without them
                projects_setup, by_chat = None, None
                log.warning("Project databases are not ready (%s); will retry next run", exc)
                stats["errors"].append({"key": "projects", "error": f"{exc.code}: {exc}"})
        with_projects = RENDER_SUFFIX + PROJECTS_SUFFIX
        rows = self.db.summaries_to_export(*((with_projects,) if projects_setup else (RENDER_SUFFIX, with_projects)))
        if not rows:
            return stats
        days_ds = await self.ensure_data_source()
        tasks_ds = await self.ensure_tasks_data_source(days_ds)
        day_names = await self.property_names(days_ds, KV_DAY_PROP_IDS, DAY_PROPERTIES)
        task_names = await self.property_names(tasks_ds, KV_TASK_PROP_IDS, self._task_schema(days_ds))
        suffix = with_projects if projects_setup else RENDER_SUFFIX
        for row in rows:
            key = f"{row['chat_id']}:{row['local_date']}"
            try:
                page_id = row["notion_page_id"] or await self._find_page(days_ds, key, day_names["Key"])
                summary = json.loads(row["summary_json"])
                if page_id is None and row["status"] != "ok" and not summary.get("not_transcribed"):
                    # Don't clutter Notion with empty days (a day of only voice messages still gets a page).
                    self.db.set_notion_page(row["chat_id"], row["local_date"], "", row["export_hash"] + suffix)
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
                project_ids = by_chat.get(select_name(row["title"] or str(row["chat_id"])), []) if by_chat is not None else None
                tasks = await self._sync_tasks(row, page_id, tasks_ds, stats, task_names, project_ids)
                await self._replace_content(page_id, build_day_blocks(row, tasks))
                self.db.set_notion_page(row["chat_id"], row["local_date"], page_id, row["export_hash"] + suffix)
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
            n = await self.property_names(tasks_ds, KV_TASK_PROP_IDS, self._task_schema(days_ds))
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
