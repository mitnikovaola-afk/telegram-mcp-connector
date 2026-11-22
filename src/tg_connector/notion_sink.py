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
    DECISIONS_TITLE,
    PROJECTS_TITLE,
    build_brief_children,
    build_brief_header,
    build_decision_properties,
    build_new_project_properties,
    build_project_counters,
    decision_key,
    decision_properties_schema,
    project_properties_schema,
    project_relation_spec,
    project_view_specs,
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
    decision_views,
    ensure_views,
    project_views,
    property_ids,
    task_views,
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
KV_DECISIONS_DATABASE = "notion_decisions_database_id"
KV_DECISIONS_DATA_SOURCE = "notion_decisions_data_source_id"
KV_PROJECT_PROP_IDS = "notion_project_property_ids"
KV_DECISION_PROP_IDS = "notion_decision_property_ids"
KV_PROJECTS_LAYOUT = "notion_projects_layout_version"
KV_DEFAULT_PROJECT = "project_default_page"
PROJECTS_LAYOUT_VERSION = "1"
KV_TASK_PROJECT_READY = "notion_task_project_column_ready"  # the «Проєкт» column exists on tasks
MIN_INTERVAL_S = 0.35  # Notion allows ~3 requests/s per integration

# Schema of both databases + their views. Bump to migrate existing installs once.
LAYOUT_VERSION = "1"
# Bump when the page layout changes: every page is re-rendered once on the next run, without LLM calls.
# v2: pages of days with voice notes lost their content; v3: new layout + tasks database;
# v4: deadline shown as "📅 вівторок, 24.11", weekday-only deadlines get a date;
# v5: page title = topics of the day (chat and date are already columns);
# v6: decision log rows and the «Проєкт» relation of tasks for days exported before projects existed.
RENDER_VERSION = "6"
RENDER_SUFFIX = f":r{RENDER_VERSION}"
# Added once a day is exported together with its decision-log rows. If the project databases were not
# available at that moment, the day is exported once more when they are (no decision is lost).
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
        """Projects + decision-log databases, the «Проєкт» relation on tasks, and their views (once)."""
        if not self.projects_enabled:
            return None
        days_ds = await self.ensure_data_source()
        tasks_ds = await self.ensure_tasks_data_source(days_ds)
        projects_ds = self.db.kv_get(KV_PROJECTS_DATA_SOURCE)
        if not projects_ds:
            database_id, projects_ds = await self._create_database(PROJECTS_TITLE, "📁", project_properties_schema())
            self.db.kv_set(KV_PROJECTS_DATABASE, database_id)
            self.db.kv_set(KV_PROJECTS_DATA_SOURCE, projects_ds)
        decisions_ds = self.db.kv_get(KV_DECISIONS_DATA_SOURCE)
        if not decisions_ds:
            database_id, decisions_ds = await self._create_database(
                DECISIONS_TITLE, "🎯", decision_properties_schema(projects_ds)
            )
            self.db.kv_set(KV_DECISIONS_DATABASE, database_id)
            self.db.kv_set(KV_DECISIONS_DATA_SOURCE, decisions_ds)
        if self.db.kv_get(KV_PROJECTS_LAYOUT) == PROJECTS_LAYOUT_VERSION:
            return projects_ds, decisions_ds
        with_project = task_properties_schema(days_ds) | {"Проєкт": project_relation_spec(projects_ds)}
        await self.property_names(tasks_ds, KV_TASK_PROP_IDS, with_project)  # adds «Проєкт» to tasks
        self.db.kv_set(KV_TASK_PROJECT_READY, "1")
        await self.property_names(projects_ds, KV_PROJECT_PROP_IDS, project_properties_schema())
        await self.property_names(decisions_ds, KV_DECISION_PROP_IDS, decision_properties_schema(projects_ds))
        try:
            await self._throttle()
            projects = await self.client.data_sources.retrieve(projects_ds)
            await self._throttle()
            decisions = await self.client.data_sources.retrieve(decisions_ds)
            await ensure_views(self.views_client, self.db.kv_get(KV_PROJECTS_DATABASE), projects_ds,
                               project_views(property_ids(projects)), self._throttle)
            await ensure_views(self.views_client, self.db.kv_get(KV_DECISIONS_DATABASE), decisions_ds,
                               decision_views(property_ids(decisions)), self._throttle)
        except APIResponseError as exc:
            log.warning("Could not set up project views (%s); will retry next run", exc)
            return projects_ds, decisions_ds
        self.db.kv_set(KV_PROJECTS_LAYOUT, PROJECTS_LAYOUT_VERSION)
        return projects_ds, decisions_ds

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
                        properties=_renamed(build_new_project_properties(title, f"auto:{chat['chat_id']}"), n)
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
        of its old tasks and decisions follow (no LLM calls)."""
        for chat in self.db.all_chats():
            name = chat["title"] or str(chat["chat_id"])
            links, key = ",".join(sorted(by_chat.get(select_name(name), []))), f"project_links:{chat['chat_id']}"
            before = self.db.kv_get(key)
            if before == links:
                continue
            if before is not None and (days := self.db.requeue_export(chat["chat_id"])):
                log.info("Project of chat «%s» changed: %d day(s) will be linked again", name, days)
            self.db.kv_set(key, links)

    async def _sync_decisions(self, row: Any, decisions_ds: str, names: dict[str, str], project_ids: list[str],
                              stats: dict[str, Any]) -> None:
        """Decision log rows for one chat-day. The log belongs to the connector: rows of decisions that are
        no longer in the summary are removed; no duplicates on re-runs (key = chat + date + text hash).
        A private chat stays out of the log until the user puts it into a project."""
        chat_id, local_date = row["chat_id"], row["local_date"]
        summary = json.loads(row["summary_json"])
        belongs = bool(project_ids) or row["kind"] in ("group", "channel")
        texts = summary.get("decisions", []) if row["status"] == "ok" and belongs else []
        known = self.db.notion_decisions_for_day(chat_id, local_date)
        seen: set[str] = set()
        for text in texts:
            key = decision_key(chat_id, local_date, text)
            if key in seen:
                continue
            seen.add(key)
            props = _renamed(build_decision_properties(row, text, key, project_ids), names)
            page_id = known.get(key) or await self._find_page(decisions_ds, key, names["Key"])
            await self._throttle()
            if page_id:
                await self.client.pages.update(page_id, properties=props)
            else:
                page = await self.client.pages.create(
                    parent={"type": "data_source_id", "data_source_id": decisions_ds}, properties=props)
                page_id = page["id"]
                stats["decisions_created"] += 1
            self.db.save_notion_decision(chat_id, local_date, key, page_id)
        for key, page_id in known.items():
            if key not in seen:
                await self._throttle()
                await self.client.pages.update(page_id, in_trash=True)
                self.db.forget_notion_decision(chat_id, local_date, key)
                stats["decisions_removed"] += 1

    async def project_open_tasks(self, page_id: str) -> int | None:
        tasks_ds, days_ds = self.db.kv_get(KV_TASKS_DATA_SOURCE), self.db.kv_get(KV_DATA_SOURCE)
        if not tasks_ds or not days_ds:
            return None
        n = await self.property_names(tasks_ds, KV_TASK_PROP_IDS, self._task_schema(days_ds))
        await self._throttle()
        result = await self.client.data_sources.query(tasks_ds, filter={"and": [
            {"property": n["Проєкт"], "relation": {"contains": page_id}},
            {"property": n["Стан"], "select": {"does_not_equal": TASK_DONE}},
        ]}, page_size=100)
        return len(result.get("results", []))

    async def render_project(self, project: dict[str, Any], brief: dict[str, Any] | None, updated_on: str | None,
                             model: str | None, chat_ids: set[int]) -> None:
        """Counters + the brief block (the only part of a project page the connector rewrites) + live views."""
        projects_ds = self.db.kv_get(KV_PROJECTS_DATA_SOURCE)
        n = await self.property_names(projects_ds, KV_PROJECT_PROP_IDS, project_properties_schema())
        counters = build_project_counters(updated_on, await self.project_open_tasks(project["page_id"]),
                                          self.db.count_notion_decisions(chat_ids))
        await self._throttle()
        await self.client.pages.update(project["page_id"], properties=_renamed(counters, n))
        if brief is not None and updated_on:
            await self._write_brief(project["page_id"], brief, updated_on, model or "")
        await self._ensure_project_views(project)

    async def _write_brief(self, page_id: str, brief: dict[str, Any], updated_on: str, model: str) -> None:
        header = build_brief_header(updated_on, model)
        children = build_brief_children(brief)
        block_id = self.db.kv_get(f"project_brief_block:{page_id}")
        if block_id:
            try:
                await self._throttle()
                await self.client.blocks.update(block_id, callout={"rich_text": header})
                await self._replace_content(block_id, children)
                return
            except APIResponseError as exc:  # the user deleted the block: create it again
                log.info("Brief block of project %s is gone (%s) — creating a new one", page_id, exc.code)
        await self._throttle()
        created = await self.client.blocks.children.append(page_id, children=[{
            "object": "block", "type": "callout",
            "callout": {"rich_text": header, "icon": {"type": "emoji", "emoji": "🧭"}, "color": "blue_background",
                        "children": children},
        }])
        self.db.kv_set(f"project_brief_block:{page_id}", created["results"][0]["id"])

    async def _ensure_project_views(self, project: dict[str, Any]) -> None:
        """Live tables on the project page, created once; the checklist follows «Дата релізу»."""
        stored = json.loads(self.db.kv_get(f"project_views:{project['page_id']}") or "{}")
        sources = {"decisions": self.db.kv_get(KV_DECISIONS_DATA_SOURCE), "tasks": self.db.kv_get(KV_TASKS_DATA_SOURCE)}
        days_ds, projects_ds = self.db.kv_get(KV_DATA_SOURCE), self.db.kv_get(KV_PROJECTS_DATA_SOURCE)
        t = await self.property_names(sources["tasks"], KV_TASK_PROP_IDS, self._task_schema(days_ds))
        d = await self.property_names(sources["decisions"], KV_DECISION_PROP_IDS, decision_properties_schema(projects_ds))
        changed = False
        try:
            for spec in project_view_specs(project["page_id"], project.get("release_date"), t, d):
                body = {k: spec[k] for k in ("name", "type", "filter", "sorts")}
                if spec["key"] not in stored:
                    await self._throttle()
                    view = await self.views_client.request(path="views", method="POST", body={
                        "create_database": {"parent": {"type": "page_id", "page_id": project["page_id"]},
                                            "show_data_source_title": False},
                        "data_source_id": sources[spec["db"]], **body})
                    stored[spec["key"]] = {"id": view["id"], "release": project.get("release_date")}
                    changed = True
                    await self._name_table(view, spec["name"])
                elif spec["key"] == "checklist" and stored[spec["key"]].get("release") != project.get("release_date"):
                    await self._throttle()
                    await self.views_client.request(path=f"views/{stored[spec['key']]['id']}", method="PATCH",
                                                    body={"filter": spec["filter"]})
                    stored[spec["key"]]["release"] = project.get("release_date")
                    changed = True
        except APIResponseError as exc:
            log.warning("Could not set up views on project «%s» (%s); will retry", project["title"], exc)
        if changed:
            self.db.kv_set(f"project_views:{project['page_id']}", json.dumps(stored))

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
            await self._throttle()
            if page_id:
                await self.client.pages.update(
                    page_id,
                    properties=_renamed(build_task_properties(row, item, key, day_page_id, new=False) | extra, names),
                )
                stats["tasks_updated"] += 1
            else:
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
        decision_names: dict[str, str] = {}
        if self.projects_enabled:
            try:
                projects_ds, decisions_ds = await self.ensure_projects_layout()
                by_chat = self.projects_by_chat(await self.assign_chats(await self.load_projects()))
                decision_names = await self.property_names(decisions_ds, KV_DECISION_PROP_IDS,
                                                           decision_properties_schema(projects_ds))
                projects_setup = (projects_ds, decisions_ds)
                self.requeue_moved_chats(by_chat)
            except APIResponseError as exc:  # projects are an extra: summaries and tasks go on without them
                by_chat = None
                log.warning("Project databases are not ready (%s); will retry next run", exc)
                stats["errors"].append({"key": "projects", "error": f"{exc.code}: {exc}"})
        with_projects = RENDER_SUFFIX + PROJECTS_SUFFIX
        rows = self.db.summaries_to_export(*((with_projects,) if projects_setup else (RENDER_SUFFIX, with_projects)))
        if not rows:
            return stats
        stats |= {"decisions_created": 0, "decisions_removed": 0}
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
                if projects_setup:  # last: if it fails, the day itself is complete and is retried next run
                    await self._sync_decisions(row, projects_setup[1], decision_names, project_ids or [], stats)
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
