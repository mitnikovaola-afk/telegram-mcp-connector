"""Project log: what matters is marked once, events are one row each, metrics keep their dynamics,
the user's edits and removals are respected, the first version of the databases is migrated in place."""

from __future__ import annotations

import json
from datetime import timedelta

import httpx
import pytest
from conftest import GOOD_BRIEF, at
from notion_client.errors import APIResponseError

from tg_connector.notion_layout import LOG_TYPES
from tg_connector.notion_sink import KV_DECISION_PROP_IDS, KV_PROJECT_PROP_IDS, KV_PROJECTS_LAYOUT, KV_TASK_PROP_IDS
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


async def test_only_decisions_the_model_calls_important_are_marked(env):
    # days are shown oldest first, then by chat id: D1 — «Продукт», D2 — «Маркетинг»
    answer(env, key_decisions=[{"id": "D1", "takeaway": "Реліз зсунуто, щоб дотестувати"}, {"id": "D99", "takeaway": None}])
    await env.run()

    prompt = env.llm.brief_prompts[0]
    assert '{"id": "D1", "text":' in prompt and '{"id": "D2", "text":' in prompt
    decisions = {r["Чат"]: r for r in rows(env, "decision")}
    assert set(decisions) == {"Маркетинг", "Продукт"}  # the database keeps ALL decisions
    assert decisions["Продукт"]["Важливе"] is True and decisions["Продукт"]["Висновок"] == "Реліз зсунуто, щоб дотестувати"
    assert "Важливе" not in decisions["Маркетинг"]  # an id the model invented (D99) is ignored


async def test_unticked_by_the_user_stays_unticked(env):
    answer(env, key_decisions=[{"id": "D1", "takeaway": None}])
    await env.run()
    page = next(p for p in env.notion.decisions.values() if p["props"].get("Важливе"))
    page["props"]["Важливе"] = {"checkbox": False}

    env.llm.queue = []
    answer(env, key_decisions=[{"id": "D1", "takeaway": "знову важливе"}])
    env.db.conn.execute("UPDATE project_briefs SET input_hash = '{}'")  # the same decision is shown to the model again
    await next_day(env, 21)

    assert page["props"]["Важливе"] == {"checkbox": False}


async def test_users_own_conclusion_is_kept_when_a_decision_is_marked(env):
    env.now = env.now.replace(hour=2)  # 04:00 in Kyiv: decisions are exported, the daily update has not run yet
    await env.run()
    for page in env.notion.decisions.values():
        page["props"]["Висновок"] = {"rich_text": [{"text": {"content": "мій висновок"}}]}
    env.now = env.now.replace(hour=10)
    answer(env, key_decisions=[{"id": "D1", "takeaway": "висновок моделі"}])
    await env.run()
    marked = [r for r in rows(env, "decision") if r.get("Важливе")]
    assert [r["Висновок"] for r in marked] == ["мій висновок"]


# ------------------------------------------------------------- one event = one row


async def test_result_of_a_test_updates_its_row_instead_of_adding_a_new_one(env):
    answer(env, log=[TEST])
    await env.run()
    (row,) = rows(env, "experiment")
    assert (row["Ефект"], row["Важливе"], row["Результат"]) == ("🔄 У процесі", True, "")

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
           "Тип": {"select": {"name": LOG_TYPES["experiment"]}}, "Важливе": {"checkbox": True},
           "Проєкт": {"relation": [{"id": project_id}]}}
    env.notion.store["mine"] = {"ds": "ds-4", "key": "", "props": dict(own), "children": [], "icon": None, "in_trash": False}

    answer(env, log=[TEST | {"id": "-", "title": "A/B пейволу: річна підписка", "date": "2026-11-21", "effect": "failed"}])
    await next_day(env, 21)

    assert '"id": "-"' in env.llm.brief_prompts[1] and "A/B пейволу" in env.llm.brief_prompts[1]
    assert env.notion.store["mine"]["props"] == own and len(rows(env, "experiment")) == 1


async def test_row_of_a_decision_or_task_in_the_trash_does_not_block_the_export(env):
    await env.run()
    next(iter(env.notion.decisions.values()))["in_trash"] = True
    next(iter(env.notion.tasks.values()))["in_trash"] = True
    env.db.conn.execute("UPDATE summaries SET notion_synced_hash = NULL")  # every day is exported again
    report = await env.run()
    assert report["notion"]["failed"] == 0 and len(env.notion.decisions) == 1 and len(env.notion.tasks) == 1


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


# --------------------------------------------------------------------- migration


async def test_first_version_of_the_databases_is_migrated_in_place(env):
    """Databases as the first release made them: two-way links, counters, «Рішення» as a title, three tables."""
    await env.run()
    n, db = env.notion, env.db
    (project_id, project), = n.projects.items()
    for ds, mirror in (("ds-2", "Задачі"), ("ds-4", "Рішення")):
        rel = n.data_sources[ds]["properties"]["Проєкт"]["relation"]
        rel |= {"type": "dual_property", "dual_property": {"synced_property_name": mirror}}
        n.data_sources["ds-3"]["properties"][mirror] = {"id": f"mirror-{mirror}", "type": "relation"}
    await n.data_sources_api.update("ds-4", {"Запис": {"name": "Рішення"}})
    await n.data_sources_api.update("ds-3", {"Оновлено": {"name": "Бриф оновлено"}})
    for name in ("Тип", "Важливе", "Результат", "Ефект", "Висновок", "Метрика", "Значення"):
        await n.data_sources_api.update("ds-4", {name: None})
    ids = json.loads(db.kv_get(KV_DECISION_PROP_IDS))
    db.kv_set(KV_DECISION_PROP_IDS, json.dumps({"Рішення": ids["Запис"]} | {k: ids[k] for k in ("Дата", "Чат", "Проєкт", "Key")}))
    ids = json.loads(db.kv_get(KV_PROJECT_PROP_IDS))
    ids["Бриф оновлено"] = ids.pop("Оновлено")
    for name in ("Відкритих задач", "Рішень", "Key"):
        n.data_sources["ds-3"]["properties"][name] = {"id": f"old-{name}", "type": "number"}
        ids[name] = f"old-{name}"
    db.kv_set(KV_PROJECT_PROP_IDS, json.dumps(ids))
    n.database_titles["db-4"] = "Telegram — рішення"
    children = n.store[project_id]["children"]
    del children[1:]  # the first page had the brief and three tables
    stored = {}
    for key, ds in (("decisions", "ds-4"), ("tasks", "ds-2"), ("checklist", "ds-2")):
        view = await n.request("views", "POST", {"create_database": {"parent": {"type": "page_id", "page_id": project_id}},
                                                 "data_source_id": ds, "name": key, "type": "table"})
        stored[key] = {"id": view["id"], "release": None}
    children.append({"id": "users-note", "type": "paragraph", "paragraph": {"rich_text": []}})
    n.parent_of["users-note"] = project_id
    db.kv_set(f"project_views:{project_id}", json.dumps(stored))
    db.conn.execute("DELETE FROM kv WHERE key LIKE 'project_team_section:%'")
    db.kv_set(KV_PROJECTS_LAYOUT, "1")
    db.kv_set("project_briefs_done_on", "2026-11-20")  # the first version already ran today
    db.kv_set(f"project_brief_rendered:{project_id}", "2026-11-20")
    db.conn.execute("UPDATE project_briefs SET input_hash = replace(input_hash, '\"_v\": 2', '\"_v\": 1')")
    rows_before = (len(n.decisions), len(n.tasks), [t["props"]["Проєкт"] for t in n.tasks.values()])
    answer(env, log=[TEST])

    report = await env.run()

    assert report["status"] == "ok" and env.llm.brief_calls == 2  # rebuilt once in the new format, the same day
    assert set(n.data_sources["ds-3"]["properties"]) == {"Проєкт", "Чати", "Статус", "Дата релізу", "Оновлено"}
    for ds in ("ds-2", "ds-4"):
        assert n.data_sources[ds]["properties"]["Проєкт"]["relation"]["type"] == "single_property"
    assert len(rows(env, "decision")) == 2  # old decision rows got their type
    assert {"Запис", "Тип", "Важливе", "Ефект"} <= set(n.data_sources["ds-4"]["properties"])
    assert "Рішення" not in n.data_sources["ds-4"]["properties"] and n.database_titles["db-4"] == "Telegram — журнал проєкту"
    assert (len(n.decisions) - 1, len(n.tasks), [t["props"]["Проєкт"] for t in n.tasks.values()]) == rows_before
    assert n.page_view_names(project_id) == ["📒 Важливе", "🧪 Тести", "📊 Динаміка"]
    assert "users-note" in [b["id"] for b in children]  # the user's own block stays
    assert json.loads(db.kv_get(KV_TASK_PROP_IDS))["Проєкт"] == n.data_sources["ds-2"]["properties"]["Проєкт"]["id"]
    again = await env.run()
    assert again["projects"] == {"skipped": "already updated today"} and again["notion"]["updated"] == 0


async def test_columns_renamed_by_the_user_are_followed_in_the_log(env):
    await env.run()
    await env.notion.data_sources_api.update("ds-4", {"Висновок": {"name": "Урок"}})
    answer(env, log=[TEST | {"takeaway": "Спершу креатив"}])
    await next_day(env, 21)
    (row,) = rows(env, "experiment")
    assert row["Урок"] == "Спершу креатив" and "Висновок" not in row
