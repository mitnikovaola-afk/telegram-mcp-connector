"""Notion structure: two linked databases, ready-made views, tasks as rows, user-owned task state."""

from __future__ import annotations

import json

import httpx
import pytest
from conftest import GOOD_SUMMARY, at
from notion_client.errors import APIResponseError

from tg_connector.notion_layout import TASK_DONE, TASK_DOING, parse_due
from tg_connector.notion_sink import KV_LAYOUT, LAYOUT_VERSION

GROUP = -1001
TWO_TASKS = dict(GOOD_SUMMARY, action_items=[
    {"task": "Запустити креатив B", "owner": "Аня", "due": "понеділок, 23.11"},
    {"task": "Підготувати звіт", "owner": "Олег", "due": None},
])
NO_TASKS = dict(GOOD_SUMMARY, action_items=[])


def task_by_title(env, title: str) -> dict:
    return next(p for p in env.notion.tasks.values() if p["props"]["Задача"]["title"][0]["text"]["content"] == title)


@pytest.mark.parametrize(
    ("due", "expected"),
    [
        ("понеділок, 23.11", "2026-11-23"),
        ("четвер, 26.11 10:00", "2026-11-26"),
        ("5 грудня", "2026-12-05"),
        ("12.01", "2027-01-12"),  # a January deadline discussed in November is next year
        ("Wednesday, 2026-11-25", "2026-11-25"),
        ("пʼятниця", "2026-11-27"),  # said on Friday 20.11 → the next Friday
        ("до п'ятниці", "2026-11-27"),
        ("завтра", "2026-11-21"),
        ("після релізу 2.5", None),  # a version number is not a date
        ("до кінця кварталу", None),
        (None, None),
    ],
)
def test_deadline_text_becomes_a_date(due, expected):
    assert parse_due(due, "2026-11-20") == expected


async def test_two_linked_databases_and_views_are_created_once(env):
    env.track(GROUP, "Маркетинг")
    env.source.add(GROUP, 1, "Запускаємо креатив B у пʼятницю", at(20, 10))
    await env.run()
    env.source.add(GROUP, 2, "Нове повідомлення", at(20, 11))
    await env.run()

    assert [d["title"] for d in env.notion.databases_created] == [
        "Telegram — денні самарі", "Telegram — задачі", "Telegram — проєкти", "Telegram — журнал проєкту"]
    assert env.notion.view_names("db-1") == ["📰 Стрічка", "🗂 За чатами", "🗓 Календар", "❓ Є відкриті питання"]
    assert env.notion.view_names("db-2") == ["🔥 Відкриті задачі", "📋 Дошка", "👤 За людьми", "🗓 Дедлайни"]
    assert env.db.kv_get(KV_LAYOUT) == LAYOUT_VERSION


async def test_each_task_is_a_row_with_owner_deadline_state_and_link_to_its_day(env):
    env.track(GROUP, "Маркетинг")
    env.source.add(GROUP, 1, "Аня, запусти креатив B до пʼятниці", at(20, 10))

    await env.run()

    (day_id, day), = env.notion.pages.items()
    (task_id, task), = env.notion.tasks.items()
    props = task["props"]
    assert props["Виконавець"]["select"]["name"] == "Аня"
    assert props["Дедлайн"]["date"]["start"] == "2026-11-27"  # "пʼятниця" said on Friday 20.11
    assert props["Стан"]["select"]["name"] == "Нова"
    assert props["День"]["relation"] == [{"id": day_id}]
    assert day["props"]["Виконавці"]["multi_select"] == [{"name": "Аня"}]
    assert day["props"]["Задач"]["number"] == 1
    assert day["props"]["Коротко"]["rich_text"][0]["text"]["content"] == GOOD_SUMMARY["tldr"]
    assert day["icon"] == {"type": "emoji", "emoji": "👥"}
    mention = {"type": "mention", "mention": {"type": "page", "page": {"id": task_id}}}
    assert json.dumps(mention) in json.dumps(day["children"])  # the task on the day page links to its row


async def test_user_task_state_is_never_overwritten(env):
    env.track(GROUP, "Маркетинг")
    env.source.add(GROUP, 1, "Аня, запусти креатив B", at(20, 10))
    await env.run()
    (task_id, task), = env.notion.tasks.items()
    task["props"]["Стан"] = {"select": {"name": TASK_DONE}}  # ticked in Notion by the user

    env.source.add(GROUP, 2, "До речі, бюджет $500 погоджено", at(20, 11))
    report = await env.run()

    assert report["notion"]["tasks_updated"] == 1
    assert env.notion.tasks[task_id]["props"]["Стан"]["select"]["name"] == TASK_DONE


async def test_task_dropped_from_summary_is_removed_only_if_nobody_touched_it(env):
    env.track(GROUP, "Маркетинг")
    env.source.add(GROUP, 1, "Аня запускає креатив B, Олег готує звіт", at(20, 10))
    env.llm.queue = [json.dumps(TWO_TASKS, ensure_ascii=False)]
    await env.run()
    assert len(env.notion.tasks) == 2
    task_by_title(env, "Підготувати звіт")["props"]["Стан"] = {"select": {"name": TASK_DOING}}

    env.source.add(GROUP, 2, "Скасовуємо обидві задачі", at(20, 12))
    env.llm.queue = [json.dumps(NO_TASKS, ensure_ascii=False)]
    report = await env.run()

    assert report["notion"]["tasks_removed"] == 1
    assert [p["props"]["Задача"]["title"][0]["text"]["content"] for p in env.notion.tasks.values()] == ["Підготувати звіт"]


async def test_old_install_is_migrated_in_place(env):
    """A database created by an older version (6 properties, no tasks base, no views) is upgraded, not recreated."""
    old_schema = {name: {"rich_text": {}} for name in ("Назва", "Чат", "Дата", "Статус", "Повідомлень", "Key")}
    created = await env.notion.databases.create(parent={}, title=[{"text": {"content": "Telegram — денні самарі"}}],
                                                initial_data_source={"properties": old_schema})
    env.db.kv_set("notion_database_id", created["id"])
    env.db.kv_set("notion_data_source_id", "ds-1")
    env.track(GROUP, "Маркетинг")
    env.source.add(GROUP, 1, "Аня, запусти креатив B", at(20, 10))

    await env.run()

    props = env.notion.data_sources["ds-1"]["properties"]
    assert {"Коротко", "Тип", "Виконавці", "Задач", "Рішень", "Відкритих питань", "Задачі"} <= set(props)
    assert len(env.notion.databases_created) == 4  # the old one + tasks, projects, decisions
    assert "📰 Стрічка" in env.notion.view_names("db-1")


async def test_views_failure_does_not_block_summaries_and_is_retried(env):
    env.track(GROUP, "Маркетинг")
    env.source.add(GROUP, 1, "Запускаємо креатив B", at(20, 10))
    real_request = env.notion.request
    calls = {"n": 0}

    async def flaky(path, method, body=None, **kw):
        if method == "POST":
            calls["n"] += 1
            if calls["n"] == 1:
                raise APIResponseError("validation_error", 400, "bad view", httpx.Headers(), "{}")
        return await real_request(path, method, body, **kw)

    env.sink.client.request = flaky
    first = await env.run()
    assert first["notion"]["created"] == 1  # the summary is in Notion anyway
    assert env.db.kv_get(KV_LAYOUT) is None

    await env.run()
    assert env.db.kv_get(KV_LAYOUT) == LAYOUT_VERSION
    assert len(env.notion.view_names("db-1")) == 4


def _rename_column(env, ds: str, old: str, new: str) -> None:
    """What a user does in Notion: the column keeps its id, only the name changes (also on every page)."""
    props = env.notion.data_sources[ds]["properties"]
    props[new] = props.pop(old)
    for page in env.notion.store.values():
        if page["ds"] == ds and old in page["props"]:
            page["props"][new] = page["props"].pop(old)


async def test_renamed_column_is_followed_not_recreated(env):
    """Seen live: «Чат» was renamed to «Категорія» in Notion and the next export failed."""
    env.track(GROUP, "Маркетинг")
    env.source.add(GROUP, 1, "Аня, запусти креатив B", at(20, 10))
    await env.run()
    _rename_column(env, "ds-1", "Чат", "Категорія")

    env.source.add(GROUP, 2, "Бюджет $500 погоджено", at(20, 11))
    report = await env.run()

    assert report["notion"]["failed"] == 0 and report["notion"]["updated"] == 1
    (day,) = env.notion.pages.values()
    assert day["props"]["Категорія"]["select"]["name"] == "Маркетинг"
    assert "Чат" not in env.notion.data_sources["ds-1"]["properties"]  # no duplicate column


async def test_rename_is_followed_even_for_installs_without_stored_ids(env):
    env.track(GROUP, "Маркетинг")
    env.source.add(GROUP, 1, "Аня, запусти креатив B", at(20, 10))
    await env.run()
    env.db.kv_set("notion_day_property_ids", "{}")  # an install made before ids were stored
    _rename_column(env, "ds-1", "Чат", "Категорія")

    env.source.add(GROUP, 2, "Бюджет $500 погоджено", at(20, 11))
    report = await env.run()

    assert report["notion"]["failed"] == 0
    assert "Чат" not in env.notion.data_sources["ds-1"]["properties"]


async def test_deleted_column_is_added_back_and_own_columns_are_untouched(env):
    env.track(GROUP, "Маркетинг")
    env.source.add(GROUP, 1, "Аня, запусти креатив B", at(20, 10))
    await env.run()
    props = env.notion.data_sources["ds-1"]["properties"]
    del props["Коротко"]  # the user deleted one of our columns
    props["Пріоритет"] = {"id": "own-1", "type": "select"}  # and added their own
    (day,) = env.notion.pages.values()
    day["props"]["Пріоритет"] = {"select": {"name": "Високий"}}

    env.source.add(GROUP, 2, "Бюджет $500 погоджено", at(20, 11))
    await env.run()

    assert "Коротко" in env.notion.data_sources["ds-1"]["properties"]
    assert day["props"]["Пріоритет"] == {"select": {"name": "Високий"}}


async def test_bot_reads_tasks_after_a_tasks_column_was_renamed(env):
    from tg_connector.bot import BotLogic

    env.track(GROUP, "Маркетинг")
    env.source.add(GROUP, 1, "Аня, запусти креатив B", at(20, 10))
    await env.run()
    _rename_column(env, "ds-2", "Стан", "Статус задачі")
    (task,) = env.notion.tasks.values()
    task["props"]["Статус задачі"] = {"select": {"name": TASK_DONE}}

    text = (await BotLogic(env.settings, env.db, env.sink, clock=lambda: env.now).handle(GROUP, False, "/tasks")).text

    assert "Запустити креатив B" not in text  # ticked «Готово» in the renamed column → hidden
    assert "відкритих задач немає" in text


def test_day_title_says_what_the_day_was_about_not_chat_and_date():
    from tg_connector.notion_layout import day_headline

    summary = {"topics": ["Реліз 2.4", "Реліз 2.5 та новий пейвол", "Найм дизайнерки"], "tldr": "Вийшов реліз."}
    assert day_headline(summary, "ok") == "Реліз 2.4 · Реліз 2.5 та новий пейвол · Найм дизайнерки"
    long_topics = {"topics": ["Дуже довга тема " * 4, "Друга тема", "Третя"]}
    assert len(day_headline(long_topics, "ok")) <= 90
    assert day_headline({"topics": [], "tldr": "Погодили бюджет. Інше."}, "ok") == "Погодили бюджет"
    assert day_headline({"not_transcribed": 2}, "no_signal") == "Лише голосові (2) — без розшифровки"
    assert day_headline({}, "no_signal") == "Нічого важливого"
