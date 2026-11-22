"""Test doubles for Telegram, the LLM and Notion — the pipeline is tested end-to-end without network."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from tg_connector import notion_sink
from tg_connector.config import Settings
from tg_connector.db import Database
from tg_connector.models import ChatInfo, RawMessage
from tg_connector.notion_sink import NotionSink
from tg_connector.pipeline import Pipeline

KYIV = ZoneInfo("Europe/Kyiv")
# "Now" for tests: 2026-11-20 15:00 Kyiv time.
NOW = datetime(2026, 11, 20, 15, 0, tzinfo=KYIV).astimezone(timezone.utc)


def at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 11, day, hour, minute, tzinfo=KYIV).astimezone(timezone.utc)


class FakeSource:
    def __init__(self) -> None:
        self.chats: dict[int, ChatInfo] = {}
        self.messages: dict[int, dict[int, RawMessage]] = {}
        self.fetch_calls: list[dict] = []
        self.window_calls: list[dict] = []
        self.fail_chats: dict[int, Exception] = {}
        self.crash_after_batches: int | None = None

    def add_chat(self, chat_id: int, title: str, kind: str = "group") -> None:
        self.chats[chat_id] = ChatInfo(chat_id, title, kind)
        self.messages.setdefault(chat_id, {})

    def add(self, chat_id: int, msg_id: int, text: str, when: datetime, sender: str = "Аня",
            kind: str = "text", reply_to: int | None = None, migrated_to: int | None = None) -> None:
        self.messages[chat_id][msg_id] = RawMessage(
            chat_id, msg_id, when, hash(sender) % 1000, sender, kind, text, reply_to, migrated_to=migrated_to
        )

    def delete(self, chat_id: int, msg_id: int) -> None:
        del self.messages[chat_id][msg_id]

    def edit(self, chat_id: int, msg_id: int, text: str) -> None:
        old = self.messages[chat_id][msg_id]
        self.messages[chat_id][msg_id] = RawMessage(
            old.chat_id, old.message_id, old.sent_at, old.sender_id, old.sender_name, old.kind, text, old.reply_to_id, NOW
        )

    async def list_chats(self, limit: int = 100) -> list[ChatInfo]:
        return list(self.chats.values())[:limit]

    async def get_chat_info(self, chat_id: int) -> ChatInfo:
        return self.chats[chat_id]

    async def iter_new_messages(self, chat_id, min_id, since, batch_size):
        self.fetch_calls.append({"chat_id": chat_id, "min_id": min_id, "since": since})
        if chat_id in self.fail_chats:
            raise self.fail_chats[chat_id]
        ordered = sorted(self.messages[chat_id].values(), key=lambda m: m.message_id)
        if min_id > 0:
            selected = [m for m in ordered if m.message_id > min_id]
        else:
            selected = [m for m in ordered if since is None or m.sent_at > since]
        batches = 0
        for i in range(0, len(selected), batch_size):
            if self.crash_after_batches is not None and batches >= self.crash_after_batches:
                raise ConnectionError("connection to Telegram lost")
            yield selected[i : i + batch_size]
            batches += 1

    async def messages_in_window(self, chat_id, since, max_id, min_id=0):
        self.window_calls.append({"chat_id": chat_id, "min_id": min_id})
        return [
            m for m in sorted(self.messages[chat_id].values(), key=lambda m: m.message_id)
            if m.sent_at >= since and min_id < m.message_id <= max_id
        ]

    async def latest_message_id(self, chat_id):
        return max(self.messages[chat_id], default=0)

    async def recent_messages(self, chat_id, limit):
        return sorted(self.messages[chat_id].values(), key=lambda m: m.message_id)[-limit:]

    async def send_message(self, chat_id, text):
        return 999


GOOD_SUMMARY = {
    "has_signal": True,
    "tldr": "Домовились запустити креатив B у пʼятницю.",
    "facts": ["CPI креативу B — $1.65"],
    "topics": ["Запуск креативу B"],
    "decisions": ["Бюджет $500 погоджено"],
    "action_items": [{"task": "Запустити креатив B", "owner": "Аня", "due": "пʼятниця"}],
    "open_questions": [],
}


GOOD_BRIEF = {
    "summary": "Запуск креативу B у TikTok.", "status": "Бюджет погоджено, старт у понеділок.",
    "owners": [{"person": "Аня", "area": "запуск креативу"}], "risks": ["Апрув клієнта ще не отримано"],
    "next_steps": ["Запустити креатив B до 23.11"], "open_questions": ["Хто готує звіт?"],
    "milestones": [{"date": "2026-11-18", "text": "Старт кампанії"}], "key_decisions": [], "log": [],
}


class FakeLLM:
    model_name = "fake-llm"

    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.queue: list[object] = []
        self.brief_calls = 0
        self.brief_prompts: list[str] = []
        self.brief_queue: list[object] = []

    async def generate_json(self, system, user, schema):
        if schema.__name__ == "ProjectBrief":
            self.brief_calls += 1
            self.brief_prompts.append(user)
            item = self.brief_queue.pop(0) if self.brief_queue else json.dumps(GOOD_BRIEF, ensure_ascii=False)
            if isinstance(item, Exception):
                raise item
            return item
        self.calls += 1
        self.prompts.append(user)
        self.systems.append(system)
        if self.queue:
            item = self.queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return json.dumps(GOOD_SUMMARY, ensure_ascii=False)


class FakeNotion:
    """In-memory imitation of the notion_client.AsyncClient surface we use: databases (days, tasks, projects,
    decisions), data sources, pages, nested blocks and the Views API (incl. linked views on pages).
    `pages` = day pages, `tasks` = task pages, `projects` = project pages, `decisions` = decision rows."""

    def __init__(self) -> None:
        self.databases_created: list[dict] = []
        self.data_sources: dict[str, dict] = {}
        self.store: dict[str, dict] = {}  # page id -> {"ds", "key", "props", "children", "icon", "in_trash"}
        self.blocks_by_id: dict[str, dict] = {}
        self.parent_of: dict[str, str] = {}
        self.views: dict[str, dict] = {}
        self.database_titles: dict[str, str] = {}
        self.fail_queries: Exception | None = None
        self.created = 0  # day pages only
        self.updated = 0
        self._next = 0

        def new_id(prefix: str) -> str:
            self._next += 1
            return f"{prefix}{self._next}"

        def column(pid: str, spec: dict) -> dict:
            kind = next(iter(spec))
            extra = {"relation": {"type": spec["relation"]["type"], "data_source_id": spec["relation"]["data_source_id"],
                                  "dual_property": spec["relation"].get("dual_property")}} if kind == "relation" else {}
            return {"id": pid, "type": kind, **extra}

        def add_dual(spec_props: dict) -> None:
            for spec in spec_props.values():
                rel = spec.get("relation") if isinstance(spec, dict) else None
                if rel and rel.get("dual_property"):
                    target = self.data_sources.get(rel["data_source_id"])
                    name = rel["dual_property"].get("synced_property_name")
                    if target is not None and name:
                        target["properties"].setdefault(name, {"id": new_id("dual-"), "type": "relation"})

        def register(blocks: list, parent: str) -> None:
            for block in blocks:
                block.setdefault("id", new_id("blk-"))
                self.blocks_by_id[block["id"]] = block
                self.parent_of[block["id"]] = parent
                register(block.get(block["type"], {}).get("children", []), block["id"])

        def children_of(parent: str) -> list:
            if parent in self.store:
                return self.store[parent]["children"]
            block = self.blocks_by_id[parent]
            return block[block["type"]].setdefault("children", [])

        async def db_create(parent, title, initial_data_source, icon=None, **kw):
            n = len(self.databases_created) + 1
            db_id, ds_id = f"db-{n}", f"ds-{n}"
            props = {name: column(f"p{n}-{i}", spec)
                     for i, (name, spec) in enumerate(initial_data_source["properties"].items())}
            self.database_titles[db_id] = title[0]["text"]["content"]
            self.data_sources[ds_id] = {"properties": props}
            self.databases_created.append({"id": db_id, "title": title[0]["text"]["content"], "ds": ds_id})
            self.views[f"view-{db_id}-default"] = {"id": f"view-{db_id}-default", "name": "Default view", "database_id": db_id}
            add_dual(initial_data_source["properties"])
            return {"id": db_id, "url": f"https://notion.so/{db_id}", "data_sources": [{"id": ds_id}]}

        async def db_update(database_id, **kw):
            if "title" in kw:
                self.database_titles[database_id] = kw["title"][0]["text"]["content"]
            return {"id": database_id}

        async def db_retrieve(database_id):
            return {"id": database_id, "title": [{"plain_text": self.database_titles.get(database_id, "")}]}

        async def ds_retrieve(data_source_id):
            return {"id": data_source_id, "properties": dict(self.data_sources[data_source_id]["properties"])}

        async def ds_update(data_source_id, properties=None, title=None):
            props = self.data_sources[data_source_id]["properties"]
            rows = [p for p in self.store.values() if p["ds"] == data_source_id]
            for name, spec in (properties or {}).items():
                if spec is None:  # delete the column
                    props.pop(name)
                    for row in rows:
                        row["props"].pop(name, None)
                elif "name" in spec:  # rename: the id stays
                    props[spec["name"]] = props.pop(name)
                    for row in rows:
                        if name in row["props"]:
                            row["props"][spec["name"]] = row["props"].pop(name)
                else:
                    old = props.get(name, {}).get("relation") or {}
                    props[name] = column(props[name]["id"] if name in props else new_id("added-"), spec)
                    if old.get("type") == "dual_property" and spec["relation"]["type"] == "single_property":
                        # the mirror column on the other side disappears, the links of the rows stay (checked live)
                        self.data_sources[old["data_source_id"]]["properties"].pop(old["dual_property"]["synced_property_name"], None)
            add_dual({k: v for k, v in (properties or {}).items() if v and "relation" in v})
            return {"id": data_source_id}

        def matches(page: dict, flt: dict | None) -> bool:
            if not flt:
                return True
            if "and" in flt:
                return all(matches(page, f) for f in flt["and"])
            prop = page["props"].get(flt["property"]) or {}
            if "rich_text" in flt:
                value = "".join(t["text"]["content"] for t in prop.get("rich_text", []))
                return bool(value) if "is_not_empty" in flt["rich_text"] else value == flt["rich_text"]["equals"]
            if "relation" in flt:
                return flt["relation"]["contains"] in [r["id"] for r in prop.get("relation", [])]
            name = (prop.get("select") or {}).get("name")
            cond = flt["select"]
            if "is_empty" in cond:
                return not name
            return name == cond["equals"] if "equals" in cond else name != cond["does_not_equal"]

        async def ds_query(data_source_id, filter=None, page_size=100, sorts=None, start_cursor=None):
            if self.fail_queries:
                raise self.fail_queries
            hits = [
                {"id": pid, "properties": p["props"], "url": f"https://notion.so/{pid}", "in_trash": False}
                for pid, p in self.store.items()
                if p["ds"] == data_source_id and not p["in_trash"] and matches(p, filter)
            ]
            return {"results": hits[:page_size], "has_more": False}

        async def page_create(parent, properties, children=(), icon=None):
            pid = new_id("page-")
            key = "".join(t["text"]["content"] for t in properties.get("Key", {}).get("rich_text", []))
            ds = parent["data_source_id"]
            self.store[pid] = {"ds": ds, "key": key, "props": dict(properties), "children": [], "icon": icon,
                               "in_trash": False}
            self.store[pid]["children"].extend(children)
            register(list(children), pid)
            if ds == "ds-1":
                self.created += 1
            return {"id": pid}

        async def page_update(page_id, properties=None, icon=None, in_trash=None):
            page = self.store[page_id]
            if page["in_trash"] and properties:
                import httpx
                from notion_client.errors import APIResponseError

                raise APIResponseError("validation_error", 400, "Can't edit block that is archived", httpx.Headers(), "{}")
            if properties:
                page["props"] = {**page["props"], **properties}
            if icon:
                page["icon"] = icon
            if in_trash is not None:
                page["in_trash"] = in_trash
            if page["ds"] == "ds-1" and properties:
                self.updated += 1
            return {"id": page_id}

        async def page_retrieve(page_id):
            if page_id not in self.store:
                return {"id": page_id, "url": f"https://notion.so/{page_id}"}
            page = self.store[page_id]
            return {"id": page_id, "properties": page["props"], "in_trash": page["in_trash"], "url": f"https://notion.so/{page_id}"}

        async def children_list(block_id, start_cursor=None):
            return {"results": [{"id": b["id"], "type": b["type"]} for b in children_of(block_id)], "has_more": False}

        async def block_delete(block_id):
            parent = self.parent_of.pop(block_id)
            siblings = children_of(parent)
            siblings[:] = [b for b in siblings if b.get("id") != block_id]
            self.blocks_by_id.pop(block_id, None)
            for vid in [v["id"] for v in self.views.values() if v.get("database_id") == block_id]:
                del self.views[vid]

        async def block_update(block_id, **fields):
            if block_id not in self.blocks_by_id:
                import httpx
                from notion_client.errors import APIResponseError

                raise APIResponseError("object_not_found", 404, "block not found", httpx.Headers(), "{}")
            block = self.blocks_by_id[block_id]
            for kind, body in fields.items():
                block[kind] = {**block.get(kind, {}), **body}
            return block

        async def children_append(block_id, children, position=None):
            created = [dict(c) for c in children]
            register(created, block_id)
            siblings = children_of(block_id)
            at = len(siblings)
            if (position or {}).get("type") == "start":
                at = 0
            elif (position or {}).get("type") == "after_block":
                at = 1 + [b["id"] for b in siblings].index(position["after_block"]["id"])
            siblings[at:at] = created
            return {"results": created}

        async def request(path, method, body=None, **kw):
            if method == "GET" and path.startswith("views?database_id="):
                db_id = path.split("=", 1)[1]
                return {"results": [{"id": v["id"]} for v in self.views.values() if v.get("database_id") == db_id]}
            if method == "GET" and path.startswith("views/"):
                return self.views[path.split("/", 1)[1]]
            if method == "PATCH" and path.startswith("views/"):
                self.views[path.split("/", 1)[1]].update(body)
                return self.views[path.split("/", 1)[1]]
            if method == "POST" and path == "views":
                vid = new_id("view-")
                self.views[vid] = {"id": vid, **body}
                container = body.get("database_id")
                if "create_database" in body:  # a linked table on a page has its own (untitled) container block
                    container = f"linked-{vid}"
                    page_id = body["create_database"]["parent"]["page_id"]
                    block = {"id": container, "type": "child_database", "child_database": {"title": "Untitled"}}
                    register([block], page_id)
                    children_of(page_id).append(block)
                self.views[vid] |= {"parent": {"type": "database_id", "database_id": container}, "database_id": container}
                return self.views[vid]
            raise AssertionError(f"unexpected request {method} {path}")

        self.databases = SimpleNamespace(create=db_create, update=db_update, retrieve=db_retrieve)
        self.data_sources_api = SimpleNamespace(retrieve=ds_retrieve, update=ds_update, query=ds_query)
        self.pages_api = SimpleNamespace(create=page_create, update=page_update, retrieve=page_retrieve)
        self.blocks = SimpleNamespace(delete=block_delete, update=block_update,
                                      children=SimpleNamespace(list=children_list, append=children_append))
        self.request = request

    def _live(self, ds: str) -> dict[str, dict]:
        return {pid: p for pid, p in self.store.items() if p["ds"] == ds and not p["in_trash"]}

    @property
    def pages(self) -> dict[str, dict]:
        """Day pages (live, not trashed)."""
        return self._live("ds-1")

    @property
    def tasks(self) -> dict[str, dict]:
        return self._live("ds-2")

    @property
    def projects(self) -> dict[str, dict]:
        return self._live("ds-3")

    @property
    def decisions(self) -> dict[str, dict]:
        return self._live("ds-4")

    def view_names(self, db_id: str) -> list[str]:
        return [v["name"] for v in self.views.values() if v.get("database_id") == db_id]

    def page_view_names(self, page_id: str) -> list[str]:
        """Tabs of the linked tables on a page."""
        tables = [b["id"] for b in self.store[page_id]["children"] if b["type"] == "child_database"]
        return [v["name"] for v in self.views.values() if v.get("database_id") in tables]

    def as_client(self) -> SimpleNamespace:
        return SimpleNamespace(
            databases=self.databases, data_sources=self.data_sources_api, pages=self.pages_api,
            blocks=self.blocks, request=self.request,
        )


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(
        tg_api_id=1, tg_api_hash="hash", gemini_api_key="key", gemini_model="fake-llm", gemini_thinking_level="",
        notion_token="token", notion_parent_page_id="0" * 32, data_dir=tmp_path,
        timezone=KYIV, sync_interval_minutes=0, backfill_days=7, edit_lookback_hours=24,
        summarize_today=True, summary_language="auto", read_only=False, send_only_tracked=True,
        max_llm_attempts=5, max_chars_per_llm_call=60_000, batch_size=200, lock_ttl_seconds=900,
        projects_enabled=False,  # the daily brief adds LLM calls; tests/test_projects.py turns it on
    )
    values.update(overrides)
    return Settings(**values)


class Env:
    def __init__(self, tmp_path: Path, **settings_overrides) -> None:
        self.settings = make_settings(tmp_path, **settings_overrides)
        self.db = Database(self.settings.db_path)
        self.source = FakeSource()
        self.llm = FakeLLM()
        self.notion = FakeNotion()
        self.sink = NotionSink("token", "0" * 32, self.db)
        self.sink.client = self.notion.as_client()
        self.sink.views_client = self.sink.client
        self.now = NOW
        self.pipeline = Pipeline(self.settings, self.db, self.source, self.llm, self.sink, clock=lambda: self.now)

    def track(self, chat_id: int, title: str, kind: str = "group") -> None:
        self.source.add_chat(chat_id, title, kind)
        self.db.upsert_chat(self.source.chats[chat_id])
        self.db.set_tracked(chat_id, True)

    async def run(self, **kw):
        return await self.pipeline.run("test", **kw)


@pytest.fixture
def env(tmp_path, monkeypatch) -> Env:
    monkeypatch.setattr(notion_sink, "MIN_INTERVAL_S", 0)
    return Env(tmp_path)


@pytest.fixture
def make_env(tmp_path, monkeypatch):
    monkeypatch.setattr(notion_sink, "MIN_INTERVAL_S", 0)
    return lambda **kw: Env(tmp_path, **kw)


__all__ = ["Env", "FakeLLM", "FakeNotion", "FakeSource", "GOOD_SUMMARY", "NOW", "at", "timedelta"]
