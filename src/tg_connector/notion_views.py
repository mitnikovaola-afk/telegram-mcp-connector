"""Ready-made views of the two Notion databases (Notion Views API, version 2026-03-11).

Created once by the connector, so a fresh install looks the same for everyone. Views the user
created or renamed are never touched: a view is created only if no view with its name exists,
and the default view is customized only while it is still called "Default view".
"""

from __future__ import annotations

from typing import Any
from urllib.parse import unquote

from .notion_layout import TASK_DONE

VIEWS_API_VERSION = "2026-03-11"
DEFAULT_VIEW_NAME = "Default view"


def _cols(ids: dict[str, str], visible: list[str], hidden: list[str] = ()) -> list[dict[str, Any]]:
    cols = [{"property_id": ids[name], "visible": True} for name in visible if name in ids]
    cols += [{"property_id": ids[name], "visible": False} for name in hidden if name in ids]
    return cols


def _group_by(ids: dict[str, str], name: str) -> dict[str, Any]:
    return {"type": "select", "property_id": ids[name], "sort": {"type": "manual"}, "hide_empty_groups": True}


def day_views(ids: dict[str, str]) -> list[dict[str, Any]]:
    newest_first = [{"property": "Дата", "direction": "descending"}]
    return [
        {
            "name": "📰 Стрічка",
            "type": "table",
            "default": True,
            "sorts": newest_first,
            "quick_filters": {"Чат": {}, "Статус": {}, "Тип": {}},
            "configuration": {
                "type": "table",
                "wrap_cells": True,
                "properties": _cols(
                    ids,
                    ["Назва", "Коротко", "Чат", "Дата", "Статус", "Задач", "Відкритих питань", "Виконавці"],
                    ["Тип", "Рішень", "Повідомлень", "Задачі", "Key"],
                ),
            },
        },
        {
            "name": "🗂 За чатами",
            "type": "board",
            "sorts": newest_first,
            "configuration": {
                "type": "board",
                "group_by": _group_by(ids, "Чат"),
                "properties": _cols(ids, ["Коротко", "Дата", "Статус", "Задач"]),
            },
        },
        {
            "name": "🗓 Календар",
            "type": "calendar",
            "configuration": {
                "type": "calendar",
                "date_property_id": ids["Дата"],
                "view_range": "month",
                "properties": _cols(ids, ["Чат", "Статус"]),
            },
        },
        {
            "name": "❓ Є відкриті питання",
            "type": "table",
            "filter": {"property": "Відкритих питань", "number": {"greater_than": 0}},
            "sorts": newest_first,
            "configuration": {
                "type": "table",
                "wrap_cells": True,
                "properties": _cols(ids, ["Назва", "Коротко", "Чат", "Дата", "Відкритих питань"]),
            },
        },
    ]


def task_views(ids: dict[str, str]) -> list[dict[str, Any]]:
    not_done = {"property": "Стан", "select": {"does_not_equal": TASK_DONE}}
    by_deadline = [{"property": "Дедлайн", "direction": "ascending"}]
    return [
        {
            "name": "🔥 Відкриті задачі",
            "type": "table",
            "default": True,
            "filter": not_done,
            "sorts": by_deadline,
            "quick_filters": {"Виконавець": {}, "Чат": {}},
            "configuration": {
                "type": "table",
                "wrap_cells": True,
                "properties": _cols(
                    ids, ["Задача", "Виконавець", "Дедлайн", "Стан", "Чат", "День"], ["Термін", "Обговорено", "Key"]
                ),
            },
        },
        {
            "name": "📋 Дошка",
            "type": "board",
            "sorts": by_deadline,
            "configuration": {
                "type": "board",
                "group_by": _group_by(ids, "Стан"),
                "properties": _cols(ids, ["Виконавець", "Дедлайн", "Чат"]),
            },
        },
        {
            "name": "👤 За людьми",
            "type": "board",
            "filter": not_done,
            "sorts": by_deadline,
            "configuration": {
                "type": "board",
                "group_by": _group_by(ids, "Виконавець"),
                "properties": _cols(ids, ["Дедлайн", "Стан", "Чат"]),
            },
        },
        {
            "name": "🗓 Дедлайни",
            "type": "calendar",
            "configuration": {
                "type": "calendar",
                "date_property_id": ids["Дедлайн"],
                "view_range": "month",
                "properties": _cols(ids, ["Виконавець", "Стан"]),
            },
        },
    ]


def property_ids(data_source: dict[str, Any]) -> dict[str, str]:
    """Property name → id as the Views API expects it (data sources return ids URL-encoded)."""
    return {name: unquote(prop["id"]) for name, prop in data_source["properties"].items()}


async def ensure_views(client: Any, database_id: str, data_source_id: str, specs: list[dict[str, Any]], throttle) -> list[str]:  # noqa: ANN001
    """Create the missing views; returns the names created or updated."""
    await throttle()
    listing = await client.request(path=f"views?database_id={database_id}", method="GET")
    existing: dict[str, str] = {}
    for ref in listing.get("results", []):
        await throttle()
        view = await client.request(path=f"views/{ref['id']}", method="GET")
        existing[view.get("name") or ""] = view["id"]
    done: list[str] = []
    for spec in specs:
        body = {k: v for k, v in spec.items() if k != "default"}
        if spec["name"] in existing:
            continue
        if spec.get("default") and DEFAULT_VIEW_NAME in existing:
            await throttle()
            await client.request(path=f"views/{existing.pop(DEFAULT_VIEW_NAME)}", method="PATCH", body={
                k: body[k] for k in ("name", "filter", "sorts", "quick_filters", "configuration") if k in body
            })
        else:
            await throttle()
            await client.request(path="views", method="POST", body={
                "database_id": database_id, "data_source_id": data_source_id, **body,
            })
        done.append(spec["name"])
    return done


def project_views(ids: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "name": "📁 Проєкти",
            "type": "table",
            "default": True,
            "sorts": [{"property": "Дата релізу", "direction": "ascending"}],
            "quick_filters": {"Статус": {}},
            "configuration": {
                "type": "table",
                "wrap_cells": True,
                "properties": _cols(
                    ids,
                    ["Проєкт", "Статус", "Дата релізу", "Відкритих задач", "Рішень", "Бриф оновлено", "Чати"],
                    ["Key", "Задачі", "Рішення"],
                ),
            },
        },
    ]


def decision_views(ids: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "name": "🎯 Журнал рішень",
            "type": "table",
            "default": True,
            "sorts": [{"property": "Дата", "direction": "descending"}],
            "quick_filters": {"Проєкт": {}, "Чат": {}},
            "configuration": {
                "type": "table",
                "wrap_cells": True,
                "properties": _cols(ids, ["Рішення", "Дата", "Проєкт", "Чат"], ["Key"]),
            },
        },
    ]
