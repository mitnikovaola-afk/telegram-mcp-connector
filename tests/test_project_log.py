"""Project log: only what matters gets in, events are one row each, metrics keep their dynamics,
the user's edits and removals are respected."""

from __future__ import annotations

import json
from datetime import timedelta

import httpx
import pytest
from conftest import GOOD_BRIEF, GOOD_SUMMARY, at
from notion_client.errors import APIResponseError

from tg_connector.notion_layout import LOG_TYPES
from tg_connector.notion_sink import KV_LOG_PROP_IDS, KV_PROJECTS_LAYOUT, KV_SERVICE_PAGE
from tg_connector.project_brief import ProjectBrief
from tg_connector.project_log import LogContext

MARKETING, PRODUCT = -1001, -1002
TEST = {"id": None, "date": "2026-11-20", "type": "experiment", "title": "Тест Reels $200 на тиждень", "result": None,
        "effect": "in_progress", "takeaway": None, "metric": None, "value": None}
CPI = {"id": None, "date": "2026-11-20", "type": "analytics", "title": "CPI Android", "result": "$1.40", "effect": None,
       "takeaway": None, "metric": "CPI Android ($)", "value": 1.4}


@pytest.fixture
def env(make_env):
    e = make_env(projects_enabled=True)
    e.sink.projects_mode, e.sink.project_name = "single", "Тестовий проєкт"
    e.track(MARKETING, "Маркетинг")
    e.track(PRODUCT, "Продукт")
    e.source.add(MARKETING, 1, "Погодили бюджет, запускаємо Reels", at(20, 10))
    e.source.add(PRODUCT, 1, "Реліз переносимо", at(20, 11))
    return e


def answer(env, log=(), key_decisions=(), **extra) -> None:
    env.llm.brief_queue.append(json.dumps(
        GOOD_BRIEF | {"log": list(log), "key_decisions": list(key_decisions)} | extra, ensure_ascii=False))


async def next_day(env, day: int, text: str = "Новина") -> dict:
    env.now += timedelta(days=1)
    env.source.add(MARKETING, 100 + day, text, at(day, 9))
    return await env.run()


def rows(env, kind: str | None = None) -> list[dict]:
    """Live log rows as {column: plain value}, optionally of one type."""
    def plain(prop):  # noqa: ANN001, ANN202
        if "title" in prop or "rich_text" in prop:
            return "".join(t["text"]["content"] for t in prop.get("title", prop.get("rich_text")) or [])
        value = next(iter(prop.values()))
        return value.get("name") if isinstance(value, dict) and "name" in value else value

    out = [{k: plain(v) for k, v in p["props"].items()} | {"_id": pid} for pid, p in env.notion.decisions.items()]
    return [r for r in out if kind is None or r["Тип"] == LOG_TYPES[kind]]


# ------------------------------------------------------------------ importance


async def test_only_decisions_the_model_calls_important_reach_the_log(env):
    # days are shown oldest first, then by chat id: D1 — «Продукт», D2 — «Маркетинг»
    answer(env, key_decisions=[{"id": "D1", "takeaway": "Щоб дотестувати"}, {"id": "D99", "takeaway": None}])
    await env.run()

    prompt = env.llm.brief_prompts[0]
    assert '{"id": "D1", "text":' in prompt and '{"id": "D2", "text":' in prompt
    (row,) = rows(env, "decision")  # routine stays on the day pages; an id the model invented (D99) is ignored
    assert (row["Чат"], row["Запис"], row["Висновок"]) == ("Продукт", GOOD_SUMMARY["decisions"][0], "Щоб дотестувати")
    assert row["Дата"]["start"] == "2026-11-20" and [r["id"] for r in row["Проєкт"]] == list(env.notion.projects)


async def test_private_chats_never_reach_the_log(env):
    env.track(42, "Олег", kind="dm")
    env.source.add(42, 1, "Вирішили між собою", at(20, 9))
    answer(env, key_decisions=[{"id": f"D{i}", "takeaway": None} for i in range(1, 6)])
    await env.run()
    assert {r["Чат"] for r in rows(env, "decision")} == {"Маркетинг", "Продукт"}


async def test_a_decision_removed_by_the_user_never_comes_back(env):
    answer(env, key_decisions=[{"id": "D1", "takeaway": None}])
    await env.run()
    next(iter(env.notion.decisions.values()))["in_trash"] = True

    answer(env, key_decisions=[{"id": "D1", "takeaway": "знову важливе"}])
    env.db.conn.execute("UPDATE project_briefs SET input_hash = '{\"_v\": 2}'")  # the model sees the same days again
    await next_day(env, 21)

    assert "D1" in env.llm.brief_prompts[1] and rows(env, "decision") == []


async def test_a_decision_already_in_the_log_is_not_rewritten_on_every_run(env):
    answer(env, key_decisions=[{"id": "D1", "takeaway": "Щоб дотестувати"}])
    await env.run()
    answer(env, key_decisions=[{"id": "D1", "takeaway": "Інші слова про те саме"}])
    env.db.conn.execute("UPDATE project_briefs SET input_hash = '{\"_v\": 2}'")  # the model sees the same days again
    await next_day(env, 21)
    assert [r["Висновок"] for r in rows(env, "decision")] == ["Щоб дотестувати"]


async def test_a_new_passport_format_rebuilds_every_project_once_the_same_day(env):
    await env.run()
    env.db.conn.execute("UPDATE project_briefs SET input_hash = replace(input_hash, '\"_v\": 2', '\"_v\": 1')")
    env.db.kv_set("project_briefs_done_on", "2026-11-20 v1")
    for _ in range(2):
        await env.run()
    assert env.llm.brief_calls == 2 and '"date": "2026-11-20"' in env.llm.brief_prompts[1]


async def test_decision_that_left_the_summary_leaves_the_log(env):
    answer(env, key_decisions=[{"id": "D2", "takeaway": None}])
    await env.run()
    env.llm.queue = [json.dumps(GOOD_SUMMARY | {"decisions": ["Бюджет $700 погоджено"]}, ensure_ascii=False)]
    env.source.add(MARKETING, 2, "Бюджет підняли до 700", at(20, 23))
    env.now = env.now.replace(hour=21, minute=30)  # still 20.11 in Kyiv: the summary of the day changes
    await env.run()
    answer(env, key_decisions=[{"id": "D1", "takeaway": None}])
    await next_day(env, 21)

    assert [r["Запис"] for r in rows(env, "decision")] == ["Бюджет $700 погоджено"]
    assert "REMOVED BY THE USER: none" in env.llm.brief_prompts[1]  # it is not the user who removed it


async def test_decision_the_user_worked_on_stays_even_if_the_summary_changed(env):
    answer(env, key_decisions=[{"id": "D2", "takeaway": None}])
    await env.run()
    page = next(iter(env.notion.decisions.values()))
    page["props"]["Висновок"] = {"rich_text": [{"text": {"content": "мій висновок"}}]}
    env.llm.queue = [json.dumps(GOOD_SUMMARY | {"decisions": []}, ensure_ascii=False)]
    env.source.add(MARKETING, 2, "Скасували", at(20, 23))
    env.now = env.now.replace(hour=21, minute=30)
    await env.run()
    await next_day(env, 21)
    assert [r["Висновок"] for r in rows(env, "decision")] == ["мій висновок"]


async def test_decisions_of_a_day_that_became_empty_leave_the_log(env):
    answer(env, key_decisions=[{"id": "D1", "takeaway": None}, {"id": "D2", "takeaway": None}])
    await env.run()
    env.db.conn.execute("UPDATE summaries SET status = 'no_signal' WHERE chat_id = ?", (MARKETING,))
    env.now += timedelta(days=1)
    await env.run()  # no new summaries: no LLM call is needed for this
    assert [r["Чат"] for r in rows(env, "decision")] == ["Продукт"] and env.llm.brief_calls == 1


# ------------------------------------------------------------- one event = one row


async def test_result_of_a_test_updates_its_row_instead_of_adding_a_new_one(env):
    answer(env, log=[TEST])
    await env.run()
    (row,) = rows(env, "experiment")
    assert (row["Ефект"], row["Результат"]) == ("🔄 У процесі", "")

    answer(env, log=[TEST | {"id": "L1", "date": "2026-11-21", "result": "CTR 0.4%, дорожче TikTok удвічі",
                             "effect": "failed", "takeaway": "Не повторювати Reels без нового креативу"}])
    await next_day(env, 21)

    assert '"id": "L1"' in env.llm.brief_prompts[1] and "Тест Reels" in env.llm.brief_prompts[1]
    (row,) = rows(env, "experiment")
    assert (row["Ефект"], row["Дата"]["start"]) == ("❌ Не спрацювало", "2026-11-20")
    assert row["Результат"].startswith("CTR 0.4%") and row["Висновок"] == "Не повторювати Reels без нового креативу"


async def test_a_metric_gets_one_row_per_day_so_its_dynamics_stay(env):
    answer(env, log=[CPI, CPI | {"title": "CPI по Android"}])  # the same metric twice in one answer
    await env.run()
    answer(env, log=[CPI | {"id": "L1", "date": "2026-11-21", "result": "$1.25", "value": 1.25}])
    await next_day(env, 21)

    assert "KNOWN METRICS: CPI Android ($)" in env.llm.brief_prompts[1]
    metric = sorted((r["Дата"]["start"], r["Значення"]) for r in rows(env, "analytics"))
    assert metric == [("2026-11-20", 1.4), ("2026-11-21", 1.25)]
    assert {r["Метрика"] for r in rows(env, "analytics")} == {"CPI Android ($)"}


async def test_repeated_answers_create_no_duplicates(env):
    for day in range(3):
        answer(env, log=[TEST, CPI])
    await env.run()
    await next_day(env, 21)
    await next_day(env, 22)
    assert len(rows(env, "experiment")) == 1 and len(rows(env, "analytics")) == 1


async def test_a_poorer_repeat_does_not_erase_what_is_already_known(env):
    answer(env, log=[CPI, TEST | {"result": "CTR 0.4%", "effect": "failed"}])
    await env.run()
    answer(env, log=[CPI | {"result": None, "value": None}, TEST | {"effect": None}])
    await next_day(env, 21)
    (metric,), (test,) = rows(env, "analytics"), rows(env, "experiment")
    assert (metric["Результат"], metric["Значення"]) == ("$1.40", 1.4)
    assert (test["Результат"], test["Ефект"]) == ("CTR 0.4%", "❌ Не спрацювало")


async def test_crash_between_creating_a_row_and_remembering_it_makes_no_second_row(env):
    answer(env, log=[TEST])
    real = env.db.log_row_created

    def crash(row_id, page_id):  # noqa: ANN001, ANN202
        raise APIResponseError("internal_server_error", 500, "connection lost", httpx.Headers(), "{}")

    env.db.log_row_created = crash
    report = await env.run()
    assert report["status"] == "partial" and len(rows(env, "experiment")) == 1
    env.db.log_row_created = real
    await env.run()
    assert len(rows(env, "experiment")) == 1 and env.llm.brief_calls == 1
    assert [r["state"] for r in env.db.project_log(next(iter(env.notion.projects)))] == ["synced"]


# --------------------------------------------------------------- the user decides


async def test_values_edited_by_the_user_are_not_overwritten(env):
    answer(env, log=[TEST])
    await env.run()
    page = next(p for p in env.notion.decisions.values() if p["props"]["Тип"]["select"]["name"] == LOG_TYPES["experiment"])
    page["props"]["Висновок"] = {"rich_text": [{"text": {"content": "Повторити з новим відео"}}]}

    answer(env, log=[TEST | {"id": "L1", "effect": "failed", "takeaway": "Не повторювати"}])
    await next_day(env, 21)

    (row,) = rows(env, "experiment")
    assert row["Висновок"] == "Повторити з новим відео"  # the user's words win
    assert row["Ефект"] == "❌ Не спрацювало"  # columns the user did not touch are still updated
    assert "Повторити з новим відео" in env.llm.brief_prompts[1]  # and the model sees them


async def test_a_removed_row_is_never_created_again(env):
    answer(env, log=[TEST])
    await env.run()
    page = next(p for p in env.notion.decisions.values() if p["props"]["Тип"]["select"]["name"] == LOG_TYPES["experiment"])
    page["in_trash"] = True

    answer(env, log=[TEST, TEST | {"date": "2026-11-21"}])
    await next_day(env, 21)

    assert "REMOVED BY THE USER: Тест Reels $200 на тиждень" in env.llm.brief_prompts[1]
    assert rows(env, "experiment") == []


async def test_rows_added_by_the_user_are_shown_to_the_model_but_never_changed(env):
    await env.run()
    (project_id,) = env.notion.projects
    own = {"Запис": {"title": [{"text": {"content": "A/B пейволу: річна підписка"}}]}, "Дата": {"date": {"start": "2026-11-21"}},
           "Тип": {"select": {"name": LOG_TYPES["experiment"]}},
           "Проєкт": {"relation": [{"id": project_id}]}}
    env.notion.store["mine"] = {"ds": "ds-4", "key": "", "props": dict(own), "children": [], "icon": None, "in_trash": False}

    answer(env, log=[TEST | {"id": "-", "title": "A/B пейволу: річна підписка", "date": "2026-11-21", "effect": "failed"}])
    await next_day(env, 21)

    assert '"id": "-"' in env.llm.brief_prompts[1] and "A/B пейволу" in env.llm.brief_prompts[1]
    assert env.notion.store["mine"]["props"] == own and len(rows(env, "experiment")) == 1


async def test_task_in_the_trash_does_not_block_the_export(env):
    await env.run()
    next(iter(env.notion.tasks.values()))["in_trash"] = True
    env.db.conn.execute("UPDATE summaries SET notion_synced_hash = NULL")  # every day is exported again
    report = await env.run()
    assert report["notion"]["failed"] == 0 and len(env.notion.tasks) == 1


async def test_team_section_is_created_once_and_never_touched_again(env):
    await env.run()
    (project_id,) = env.notion.projects
    children = env.notion.store[project_id]["children"]
    assert [b["type"] for b in children][:2] == ["callout", "heading_2"]  # passport, then the team's section
    assert children[-1]["type"] == "child_database"  # the log is at the bottom
    del children[3:6]  # the team reworked the template
    before = json.dumps(children[1:], ensure_ascii=False)
    await next_day(env, 21)
    assert json.dumps(children[1:], ensure_ascii=False) == before


# ----------------------------------------------------------------- bad model output


def test_bad_items_are_cleaned_or_dropped():
    brief = ProjectBrief.model_validate(GOOD_BRIEF | {"log": [
        TEST | {"title": "   "},
        TEST | {"title": "Дата з майбутнього", "date": "2031-01-01"},
        TEST | {"title": "Не дата", "date": "вчора"},
        CPI | {"effect": "worked"},
        CPI | {"metric": "CPI, Android", "type": "release"},
    ]})
    out = LogContext().apply([], brief.log, "2026-11-08", "2026-11-22")
    assert [(e["title"], e["date"], e["effect"], e["metric"], e["value"]) for e in out.values()] == [
        ("Дата з майбутнього", "2026-11-22", "in_progress", None, None), ("Не дата", "2026-11-22", "in_progress", None, None),
        ("CPI Android", "2026-11-20", None, "CPI Android ($)", 1.4),  # an effect belongs to tests only
        ("CPI Android", "2026-11-20", None, None, None)]


def test_one_answer_cannot_flood_the_log():
    items = [TEST | {"title": f"Подія {i}"} for i in range(200)]
    brief = ProjectBrief.model_validate(GOOD_BRIEF | {"log": items})
    assert len(LogContext().apply([], brief.log, "2026-11-08", "2026-11-22")) == 25


async def test_milestones_are_sorted_and_future_ones_are_marked_as_plans(env):
    answer(env, milestones=[{"date": "2026-12-05", "text": "Реліз 2.5"}, {"date": "2026-11-19", "text": "Реліз 2.4"},
                            {"date": "колись", "text": "Без дати"}])
    await env.run()
    (project_id,) = env.notion.projects
    (callout,) = [b for b in env.notion.store[project_id]["children"] if b["type"] == "callout"]
    text = json.dumps(callout, ensure_ascii=False)
    assert text.index("19.11.2026 — Реліз 2.4") < text.index("05.12.2026 — Реліз 2.5 (план)") and "Без дати" not in text


# ------------------------------------------------------------ where the log lives


async def test_log_database_lives_on_the_service_page_not_on_the_main_one(env):
    await env.run()
    ((page_id, page),) = env.notion.child_pages.items()
    assert page["title"] == "Службове" and page["parent"] == "0" * 32
    assert env.notion.database_parents["db-4"] == page_id
    assert [env.notion.database_parents[f"db-{i}"] for i in (1, 2, 3)] == ["0" * 32] * 3
    await next_day(env, 21)
    assert len(env.notion.child_pages) == 1


async def test_service_page_is_found_again_after_a_crash(env):
    env.notion.child_pages["half-done"] = {"parent": "0" * 32, "title": "Службове", "children": []}
    await env.run()
    assert list(env.notion.child_pages) == ["half-done"] and env.db.kv_get(KV_SERVICE_PAGE) == "half-done"


async def test_previous_version_of_the_log_is_migrated_in_place(env):
    """Before: ALL decisions as rows, a «Важливе» checkbox, the «Усі записи» view, the database on the main page."""
    answer(env, key_decisions=[{"id": "D1", "takeaway": None}], log=[TEST])
    await env.run()
    n, db = env.notion, env.db
    (project_id,) = n.projects
    await n.data_sources_api.update("ds-4", {"Важливе": {"checkbox": {}}})
    ids = json.loads(db.kv_get(KV_LOG_PROP_IDS))
    important = n.data_sources["ds-4"]["properties"]["Важливе"]["id"]
    db.kv_set(KV_LOG_PROP_IDS, json.dumps(ids | {"Важливе": important}))
    for page in n.decisions.values():
        page["props"]["Важливе"] = {"checkbox": True}
    routine = {"Запис": {"title": [{"text": {"content": "Планерка о 10:00"}}]}, "Тип": {"select": {"name": LOG_TYPES["decision"]}},
               "Важливе": {"checkbox": False}, "Проєкт": {"relation": [{"id": project_id}]},
               "Key": {"rich_text": [{"text": {"content": "-1001:2026-11-20:abc"}}]}}
    users = {k: v for k, v in routine.items() if k != "Key"}  # added by hand: no Key
    for pid, props in (("routine", routine), ("users", users)):
        n.store[pid] = {"ds": "ds-4", "key": "", "props": dict(props), "children": [], "icon": None, "in_trash": False}
    await n.request("views", "POST", {"database_id": "db-4", "data_source_id": "ds-4", "name": "🗂 Усі записи", "type": "table"})
    for view in n.views.values():
        if view["name"] in ("📒 Важливе", "🧪 Тести", "📊 Динаміка"):
            view["filter"] = {"and": [{"property": important, "checkbox": {"equals": True}}]}
    n.database_parents["db-4"] = "0" * 32
    n.child_pages.clear()
    db.conn.execute("DELETE FROM kv WHERE key = ?", (KV_SERVICE_PAGE,))
    db.kv_set(KV_PROJECTS_LAYOUT, "2")
    before = {r["_id"] for r in rows(env)} - {"routine"}

    report = await env.run()

    assert report["status"] == "ok" and {r["_id"] for r in rows(env)} == before and "users" in before
    assert "Важливе" not in n.data_sources["ds-4"]["properties"]
    assert n.view_names("db-4") == ["📒 Важливе", "🧪 Тести", "📊 Динаміка"]
    assert n.page_view_names(project_id) == ["📒 Важливе", "🧪 Тести", "📊 Динаміка"]
    assert not any(important in json.dumps(v.get("filter") or {}) for v in n.views.values())
    (service,) = n.child_pages
    assert n.database_parents["db-4"] == service
    again = await env.run()
    assert again["projects"] == {"skipped": "already updated today"} and len(n.child_pages) == 1


async def test_columns_renamed_by_the_user_are_followed_in_the_log(env):
    await env.run()
    await env.notion.data_sources_api.update("ds-4", {"Висновок": {"name": "Урок"}})
    answer(env, log=[TEST | {"takeaway": "Спершу креатив"}])
    await next_day(env, 21)
    (row,) = rows(env, "experiment")
    assert row["Урок"] == "Спершу креатив" and "Висновок" not in row
