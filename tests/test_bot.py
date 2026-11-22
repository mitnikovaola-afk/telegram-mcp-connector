"""Team bot: what /status and /tasks show, and who gets what (access rules, cooldown, no cross-chat leaks)."""

from __future__ import annotations

import json
import re
from datetime import timedelta

import httpx
from conftest import GOOD_SUMMARY, at, make_settings
from notion_client.errors import APIResponseError

from tg_connector.bot import BotLogic
from tg_connector.notion_layout import TASK_DONE
from tg_connector.team_status import PRIVATE_TEXT

GROUP, OTHER = -1001, -1002
PROJECT = dict(
    GOOD_SUMMARY,
    decisions=["Бюджет $500 погоджено"],
    action_items=[
        {"task": "Підготувати звіт", "owner": "Марія", "due": "четвер, 19.11"},  # overdue on 20.11
        {"task": "Запустити креатив B", "owner": "Аня", "due": "понеділок, 23.11"},
    ],
    open_questions=["Коли робимо ретро?"],
)
SECRET = dict(GOOD_SUMMARY, decisions=["Секретний проєкт Х затверджено"], action_items=[], open_questions=[])


def summaries_by_chat(mapping: dict[str, dict]):
    async def generate(system, user, schema):
        title = next(t for t in mapping if f"Chat: {t}" in user)
        return json.dumps(mapping[title], ensure_ascii=False)

    return generate


async def setup(env) -> BotLogic:
    env.track(GROUP, "Маркетинг")
    env.track(OTHER, "Інший відділ")
    env.source.add(GROUP, 1, "Бюджет $500 погоджено, Марія готує звіт, Аня запускає креатив B", at(20, 10))
    env.source.add(OTHER, 1, "Секретний проєкт Х затверджено", at(20, 11))
    env.llm.generate_json = summaries_by_chat({"Маркетинг": PROJECT, "Інший відділ": SECRET})
    await env.run()
    return BotLogic(env.settings, env.db, env.sink, bot_username="status_bot", clock=lambda: env.now)


async def test_status_shows_decisions_tasks_questions_and_freshness(env):
    bot = await setup(env)

    text = (await bot.handle(GROUP, False, "/status")).text

    assert "📊 Статус: Маркетинг" in text
    assert "20.11 — Бюджет $500 погоджено" in text
    assert "⚠️ <b>Марія</b>" in text and "(прострочено)" in text
    assert "<b>Аня</b>" in text and "до 23.11" in text
    assert text.index("Марія") < text.index("Аня")  # overdue first
    assert "20.11 — Коли робимо ретро?" in text
    assert re.search(r"дані станом на \d\d\.\d\d \d\d:\d\d", text)  # time of the last real sync
    assert "notion.so/" in text


async def test_tasks_ticked_done_in_notion_are_not_shown(env):
    bot = await setup(env)
    task = next(p for p in env.notion.tasks.values() if p["props"]["Задача"]["title"][0]["text"]["content"] == "Підготувати звіт")
    task["props"]["Стан"] = {"select": {"name": TASK_DONE}}

    text = (await bot.handle(GROUP, False, "/status")).text

    assert "Підготувати звіт" not in text
    assert "Запустити креатив B" in text


async def test_tasks_command_lists_only_open_tasks(env):
    bot = await setup(env)

    text = (await bot.handle(GROUP, False, "/tasks")).text

    assert "📌 Відкриті задачі: Маркетинг" in text
    assert "Запустити креатив B" in text and "Підготувати звіт" in text
    assert "Рішення" not in text


async def test_no_data_from_another_chat_ever(env):
    bot = await setup(env)

    for command in ("/status", "/tasks"):
        text = (await bot.handle(GROUP, False, command)).text
        assert "Секретний" not in text and "Інший відділ" not in text


async def test_untracked_group_gets_nothing_and_the_bot_leaves(env):
    bot = await setup(env)

    reply = await bot.handle(-999, False, "/status")

    assert reply.leave and reply.text is None


async def test_private_messages_get_help_but_no_project_data(env):
    bot = await setup(env)

    reply = await bot.handle(555, True, "/status")

    assert reply.text == PRIVATE_TEXT
    assert "Бюджет" not in reply.text


async def test_cooldown_ignores_repeated_commands(env):
    bot = await setup(env)

    assert await bot.handle(GROUP, False, "/status") is not None
    assert await bot.handle(GROUP, False, "/status") is None  # within 60 s: ignored, no spam
    assert await bot.handle(GROUP, False, "/tasks") is not None  # another command is fine
    env.now += timedelta(seconds=61)
    assert await bot.handle(GROUP, False, "/status") is not None


async def test_only_its_own_commands(env):
    bot = await setup(env)

    assert await bot.handle(GROUP, False, "привіт, як справи") is None
    assert await bot.handle(GROUP, False, "/status@another_bot") is None
    assert await bot.handle(GROUP, False, "/status@status_bot") is not None
    assert "Я показую статус" in (await bot.handle(GROUP, False, "/help")).text


async def test_notion_down_falls_back_to_tasks_from_summaries(env):
    bot = await setup(env)
    env.notion.fail_queries = APIResponseError("service_unavailable", 503, "down", httpx.Headers(), "{}")

    text = (await bot.handle(GROUP, False, "/status")).text

    assert "Стан задач у Notion зараз недоступний" in text
    assert "Запустити креатив B" in text


async def test_chat_text_is_escaped_in_html_replies(env):
    env.track(GROUP, "Маркетинг <b>")
    env.source.add(GROUP, 1, "Погодили бюджет на кампанію", at(20, 10))  # Ukrainian, so no language re-ask
    env.llm.queue = [json.dumps(dict(GOOD_SUMMARY, decisions=["<script>alert(1)</script>"]), ensure_ascii=False)]
    await env.run()
    bot = BotLogic(env.settings, env.db, env.sink, clock=lambda: env.now)

    text = (await bot.handle(GROUP, False, "/status")).text

    assert "&lt;script&gt;" in text and "<script>" not in text
    assert "Маркетинг &lt;b&gt;" in text


async def test_empty_chat_explains_instead_of_failing(env):
    env.track(GROUP, "Новий чат")
    bot = BotLogic(env.settings, env.db, env.sink, clock=lambda: env.now)

    text = (await bot.handle(GROUP, False, "/status")).text

    assert "самарі ще немає" in text


def test_bot_token_is_masked_in_logs(tmp_path):
    settings = make_settings(tmp_path, tg_bot_token="123456:ABC-secret-token")
    assert "123456:ABC-secret-token" in settings.secret_values()


async def test_old_open_questions_are_not_shown_once_a_newer_day_exists(env):
    """«пейвол у 2.4 чи 2.5?» asked on 19.11 and decided on 20.11 must not show up as still open."""
    env.track(GROUP, "Продукт")
    env.source.add(GROUP, 1, "Вмикаємо пейвол у 2.4 чи 2.5?", at(19, 10))
    env.source.add(GROUP, 2, "Вирішили: пейвол у 2.5", at(20, 10))
    env.llm.queue = [
        json.dumps(dict(GOOD_SUMMARY, open_questions=["Пейвол у 2.4 чи 2.5?"]), ensure_ascii=False),
        json.dumps(dict(GOOD_SUMMARY, decisions=["Пейвол у 2.5"], open_questions=[]), ensure_ascii=False),
    ]
    await env.run()
    bot = BotLogic(env.settings, env.db, env.sink, clock=lambda: env.now)

    text = (await bot.handle(GROUP, False, "/status")).text

    assert "Пейвол у 2.5" in text
    assert "2.4 чи 2.5" not in text
