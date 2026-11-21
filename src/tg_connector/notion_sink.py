"""Stage 3 of the pipeline: publish summaries to a Notion database (the human-readable view).

SQLite stays the source of truth; Notion is a projection of it. Idempotency:
* a summary is exported only when its export_hash differs from the last exported one;
* the Notion page id is stored locally, so an update edits the same page;
* if the process died after Notion created a page but before we stored its id, the next run
  finds the page by its unique "Key" property (chat_id:date) instead of creating a duplicate.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from notion_client import AsyncClient
from notion_client.errors import APIResponseError

from .db import Database

log = logging.getLogger(__name__)

DB_TITLE = "Telegram — денні самарі"
KV_DATA_SOURCE = "notion_data_source_id"
KV_DATABASE = "notion_database_id"
MIN_INTERVAL_S = 0.35  # Notion allows ~3 requests/s per integration

STATUS_FINAL = "Підсумок дня"
STATUS_PROGRESS = "День триває"
STATUS_EMPTY = "Нічого важливого"

PROPERTIES_SCHEMA: dict[str, Any] = {
    "Назва": {"title": {}},
    "Чат": {"select": {}},
    "Дата": {"date": {}},
    "Статус": {
        "select": {
            "options": [
                {"name": STATUS_FINAL, "color": "green"},
                {"name": STATUS_PROGRESS, "color": "yellow"},
                {"name": STATUS_EMPTY, "color": "gray"},
            ]
        }
    },
    "Повідомлень": {"number": {"format": "number"}},
    "Key": {"rich_text": {}},
}


def _text(content: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": {"content": content[:1900]}}]


def _block(kind: str, content: str, **extra: Any) -> dict[str, Any]:
    return {"object": "block", "type": kind, kind: {"rich_text": _text(content), **extra}}


# Bump when build_blocks/build_properties change: every page is re-rendered once on the next run,
# even though its summary did not change (v2: fixed pages of days with voice notes losing their content).
RENDER_VERSION = "2"
RENDER_SUFFIX = f":r{RENDER_VERSION}"


def build_blocks(row: Any) -> list[dict[str, Any]]:
    s = json.loads(row["summary_json"])
    blocks: list[dict[str, Any]] = []
    voice_note = None
    if s.get("not_transcribed"):
        voice_note = _block(
            "callout",
            f"Голосових і відеоповідомлень без розшифровки: {s['not_transcribed']}. Їхній зміст у самарі не врахований.",
            icon={"type": "emoji", "emoji": "🎙️"},
        )
    if row["status"] != "ok":
        blocks.append(_block("paragraph", "За цей день у чаті не було робочих обговорень."))
        if voice_note:
            blocks.append(voice_note)
    else:
        blocks.append(_block("callout", s.get("tldr") or "—", icon={"type": "emoji", "emoji": "🧭"}))
        if voice_note:
            blocks.append(voice_note)
        sections = [
            ("Рішення", s.get("decisions", [])),
            ("Ключові факти й цифри", s.get("facts", [])),
            ("Відкриті питання", s.get("open_questions", [])),
            ("Теми", s.get("topics", [])),
        ]
        if s.get("action_items"):
            blocks.append(_block("heading_3", "Задачі"))
            for a in s["action_items"]:
                label = a["task"]
                if a.get("owner"):
                    label = f"{a['owner']}: {label}"
                if a.get("due"):
                    label += f" — термін: {a['due']}"
                blocks.append(_block("to_do", label, checked=False))
        for title, items in sections:
            if items:
                blocks.append(_block("heading_3", title))
                blocks.extend(_block("bulleted_list_item", item) for item in items)
        if s.get("links"):
            blocks.append(_block("heading_3", "Посилання"))
            for url in s["links"]:
                blocks.append(
                    {
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [{"type": "text", "text": {"content": url[:1900], "link": {"url": url}}}]
                        },
                    }
                )
    footer = (
        f"Враховано повідомлень: {row['signal_messages']}, відфільтровано шуму: {row['noise_messages']} · "
        f"модель: {row['model']} · оновлено: {row['updated_at']}"
    )
    blocks.append({"object": "block", "type": "divider", "divider": {}})
    blocks.append(_block("paragraph", footer))
    return blocks


def build_properties(row: Any) -> dict[str, Any]:
    if row["status"] != "ok":
        status = STATUS_EMPTY
    else:
        status = STATUS_FINAL if row["is_final"] else STATUS_PROGRESS
    chat_title = (row["title"] or str(row["chat_id"])).replace(",", " ")[:100]
    return {
        "Назва": {"title": _text(f"{row['title']} · {row['local_date']}")},
        "Чат": {"select": {"name": chat_title}},
        "Дата": {"date": {"start": row["local_date"]}},
        "Статус": {"select": {"name": status}},
        "Повідомлень": {"number": row["signal_messages"]},
        "Key": {"rich_text": _text(f"{row['chat_id']}:{row['local_date']}")},
    }


class NotionSink:
    def __init__(self, token: str, parent_page_id: str, db: Database):
        self.client = AsyncClient(auth=token)
        self.parent_page_id = parent_page_id
        self.db = db
        self._last_call = 0.0

    async def _throttle(self) -> None:
        loop = asyncio.get_running_loop()
        wait = self._last_call + MIN_INTERVAL_S - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_call = loop.time()

    async def ensure_data_source(self) -> str:
        cached = self.db.kv_get(KV_DATA_SOURCE)
        if cached:
            return cached
        await self._throttle()
        database = await self.client.databases.create(
            parent={"type": "page_id", "page_id": self.parent_page_id},
            title=_text(DB_TITLE),
            initial_data_source={"properties": PROPERTIES_SCHEMA},
        )
        data_source_id = database["data_sources"][0]["id"]
        self.db.kv_set(KV_DATABASE, database["id"])
        self.db.kv_set(KV_DATA_SOURCE, data_source_id)
        log.info("Created Notion database %s", database.get("url", database["id"]))
        return data_source_id

    async def _find_page(self, data_source_id: str, key: str) -> str | None:
        await self._throttle()
        result = await self.client.data_sources.query(
            data_source_id, filter={"property": "Key", "rich_text": {"equals": key}}, page_size=1
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

    async def export_pending(self) -> dict[str, Any]:
        stats: dict[str, Any] = {"created": 0, "updated": 0, "skipped_empty": 0, "failed": 0, "errors": []}
        rows = self.db.summaries_to_export(RENDER_SUFFIX)
        if not rows:
            return stats
        data_source_id = await self.ensure_data_source()
        for row in rows:
            key = f"{row['chat_id']}:{row['local_date']}"
            try:
                page_id = row["notion_page_id"] or await self._find_page(data_source_id, key)
                if page_id is None and row["status"] != "ok" and not json.loads(row["summary_json"]).get("not_transcribed"):
                    # Don't clutter Notion with empty days (a day of only voice messages still gets a page).
                    self.db.set_notion_page(row["chat_id"], row["local_date"], "", row["export_hash"] + RENDER_SUFFIX)
                    stats["skipped_empty"] += 1
                    continue
                props = build_properties(row)
                blocks = build_blocks(row)
                if page_id:
                    await self._throttle()
                    await self.client.pages.update(page_id, properties=props)
                    await self._replace_content(page_id, blocks)
                    stats["updated"] += 1
                else:
                    await self._throttle()
                    page = await self.client.pages.create(
                        parent={"type": "data_source_id", "data_source_id": data_source_id},
                        properties=props,
                        children=blocks,
                    )
                    page_id = page["id"]
                    stats["created"] += 1
                self.db.set_notion_page(row["chat_id"], row["local_date"], page_id, row["export_hash"] + RENDER_SUFFIX)
            except APIResponseError as exc:
                stats["failed"] += 1
                stats["errors"].append({"key": key, "error": f"{exc.code}: {exc}"})
                log.warning("Notion export failed for %s: %s", key, exc)
        return stats

    async def check_access(self) -> str:
        await self._throttle()
        page = await self.client.pages.retrieve(self.parent_page_id)
        return page.get("url", self.parent_page_id)
