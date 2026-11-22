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


class FakeLLM:
    model_name = "fake-llm"

    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.queue: list[object] = []

    async def generate_json(self, system, user, schema):
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
    """In-memory imitation of the notion_client.AsyncClient surface we use: two databases (days, tasks),
    data sources, pages, blocks and the Views API. `pages` holds the day pages, `tasks` the task pages."""

    def __init__(self) -> None:
        self.databases_created: list[dict] = []
        self.data_sources: dict[str, dict] = {}  # ds id -> {"properties": {...}}
        self.store: dict[str, dict] = {}  # every page: {"ds", "key", "props", "children", "in_trash"}
        self.views: dict[str, dict] = {}  # view id -> view (with "database_id")
        self.fail_queries: Exception | None = None
        self.created = 0  # day pages only
        self.updated = 0
        self._next = 0

        async def db_create(parent, title, initial_data_source, icon=None, **kw):
            n = len(self.databases_created) + 1
            db_id, ds_id = f"db-{n}", f"ds-{n}"
            props = {name: {"id": f"p{n}-{i}", "type": next(iter(spec))} for i, (name, spec) in enumerate(initial_data_source["properties"].items())}
            self.data_sources[ds_id] = {"properties": props}
            self.databases_created.append({"id": db_id, "title": title[0]["text"]["content"], "ds": ds_id})
            self.views[f"view-{db_id}-default"] = {"id": f"view-{db_id}-default", "name": "Default view", "database_id": db_id}
            if n == 2:  # the tasks database relates to the days one: Notion adds the synced property there
                self.data_sources["ds-1"]["properties"]["Задачі"] = {"id": "p1-rel", "type": "relation"}
            return {"id": db_id, "url": f"https://notion.so/{db_id}", "data_sources": [{"id": ds_id}]}

        async def db_update(database_id, **kw):
            return {"id": database_id}

        async def ds_retrieve(data_source_id):
            return {"id": data_source_id, "properties": dict(self.data_sources[data_source_id]["properties"])}

        async def ds_update(data_source_id, properties):
            props = self.data_sources[data_source_id]["properties"]
            for i, (name, spec) in enumerate(properties.items()):
                props[name] = {"id": f"added-{i}", "type": next(iter(spec))}
            return {"id": data_source_id}

        def matches(page: dict, flt: dict) -> bool:
            if "and" in flt:
                return all(matches(page, f) for f in flt["and"])
            prop = page["props"].get(flt["property"]) or {}
            if "rich_text" in flt:
                value = "".join(t["text"]["content"] for t in prop.get("rich_text", []))
                return value == flt["rich_text"]["equals"]
            name = (prop.get("select") or {}).get("name")
            cond = flt["select"]
            return name == cond["equals"] if "equals" in cond else name != cond["does_not_equal"]

        async def ds_query(data_source_id, filter, page_size=1, sorts=None):
            if self.fail_queries:
                raise self.fail_queries
            hits = [
                {"id": pid, "properties": p["props"], "url": f"https://notion.so/{pid}"}
                for pid, p in self.store.items()
                if p["ds"] == data_source_id and not p["in_trash"] and matches(p, filter)
            ]
            return {"results": hits[:page_size]}

        async def page_create(parent, properties, children=(), icon=None):
            self._next += 1
            pid = f"page-{self._next}"
            key = properties["Key"]["rich_text"][0]["text"]["content"]
            ds = parent["data_source_id"]
            self.store[pid] = {"ds": ds, "key": key, "props": properties, "children": list(children), "icon": icon, "in_trash": False}
            if ds == "ds-1":
                self.created += 1
            return {"id": pid}

        async def page_update(page_id, properties=None, icon=None, in_trash=None):
            page = self.store[page_id]
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

        async def children_list(page_id, start_cursor=None):
            return {"results": [{"id": f"{page_id}-b{i}"} for i, _ in enumerate(self.store[page_id]["children"])], "has_more": False}

        async def block_delete(block_id):
            page_id = block_id.rsplit("-b", 1)[0]
            if self.store[page_id]["children"]:
                self.store[page_id]["children"].pop()

        async def children_append(page_id, children):
            self.store[page_id]["children"].extend(children)

        async def request(path, method, body=None, **kw):
            if method == "GET" and path.startswith("views?database_id="):
                db_id = path.split("=", 1)[1]
                return {"results": [{"id": v["id"]} for v in self.views.values() if v["database_id"] == db_id]}
            if method == "GET" and path.startswith("views/"):
                return self.views[path.split("/", 1)[1]]
            if method == "PATCH" and path.startswith("views/"):
                self.views[path.split("/", 1)[1]].update(body)
                return self.views[path.split("/", 1)[1]]
            if method == "POST" and path == "views":
                vid = f"view-{len(self.views) + 1}"
                self.views[vid] = {"id": vid, **body}
                return self.views[vid]
            raise AssertionError(f"unexpected request {method} {path}")

        self.databases = SimpleNamespace(create=db_create, update=db_update)
        self.data_sources_api = SimpleNamespace(retrieve=ds_retrieve, update=ds_update, query=ds_query)
        self.pages_api = SimpleNamespace(create=page_create, update=page_update, retrieve=page_retrieve)
        self.blocks = SimpleNamespace(delete=block_delete, children=SimpleNamespace(list=children_list, append=children_append))
        self.request = request

    @property
    def pages(self) -> dict[str, dict]:
        """Day pages (live, not trashed)."""
        return {pid: p for pid, p in self.store.items() if p["ds"] == "ds-1" and not p["in_trash"]}

    @property
    def tasks(self) -> dict[str, dict]:
        """Task pages (live, not trashed)."""
        return {pid: p for pid, p in self.store.items() if p["ds"] == "ds-2" and not p["in_trash"]}

    def view_names(self, db_id: str) -> list[str]:
        return [v["name"] for v in self.views.values() if v["database_id"] == db_id]

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
