"""How summaries look in Notion: database schemas, page properties and page content.

Two databases, the pattern used by meeting-notes tools (Notion AI Meeting Notes, Fireflies, n8n
templates): one row per chat-day with a readable page, and a separate tasks database where every
action item is a row with owner, real due date and state — so tasks can be filtered, sorted,
put on a board or a calendar. Days and tasks are linked by a two-way relation.

Everything here is pure (no API calls), so it is unit-tested directly.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, timedelta
from typing import Any

DAYS_TITLE = "Telegram — денні самарі"
TASKS_TITLE = "Telegram — задачі"

STATUS_FINAL = "Підсумок дня"
STATUS_PROGRESS = "День триває"
STATUS_EMPTY = "Нічого важливого"

TASK_NEW, TASK_DOING, TASK_DONE = "Нова", "В роботі", "Готово"

CHAT_TYPES = {"group": ("Група", "👥"), "dm": ("Особистий", "💬"), "channel": ("Канал", "📣"), "bot": ("Бот", "🤖")}

DAY_PROPERTIES: dict[str, Any] = {
    "Назва": {"title": {}},
    "Коротко": {"rich_text": {}},
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
    "Тип": {
        "select": {
            "options": [
                {"name": "Група", "color": "blue"},
                {"name": "Особистий", "color": "purple"},
                {"name": "Канал", "color": "orange"},
                {"name": "Бот", "color": "gray"},
            ]
        }
    },
    "Виконавці": {"multi_select": {}},
    "Задач": {"number": {"format": "number"}},
    "Рішень": {"number": {"format": "number"}},
    "Відкритих питань": {"number": {"format": "number"}},
    "Повідомлень": {"number": {"format": "number"}},
    "Key": {"rich_text": {}},
}


def task_properties_schema(days_data_source_id: str) -> dict[str, Any]:
    return {
        "Задача": {"title": {}},
        "Стан": {
            "select": {
                "options": [
                    {"name": TASK_NEW, "color": "blue"},
                    {"name": TASK_DOING, "color": "yellow"},
                    {"name": TASK_DONE, "color": "green"},
                ]
            }
        },
        "Виконавець": {"select": {}},
        "Дедлайн": {"date": {}},
        "Термін": {"rich_text": {}},
        "Чат": {"select": {}},
        "Обговорено": {"date": {}},
        "День": {
            "relation": {
                "data_source_id": days_data_source_id,
                "type": "dual_property",
                "dual_property": {"synced_property_name": "Задачі"},
            }
        },
        "Key": {"rich_text": {}},
    }


# ----------------------------------------------------------------------------- dates

_MONTHS = {
    name: i
    for names in (
        "січня лютого березня квітня травня червня липня серпня вересня жовтня листопада грудня",
        "января февраля марта апреля мая июня июля августа сентября октября ноября декабря",
        "january february march april may june july august september october november december",
    )
    for i, name in enumerate(names.split(), 1)
}
_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
# "23.11" / "5.12.2026" — the month has two digits, so release numbers like "2.5" are not dates.
_DOT = re.compile(r"(?<![\d.])(\d{1,2})\.(\d{2})(?:\.(\d{4}))?(?![\d])")
_WORD = re.compile(r"(?<!\d)(\d{1,2})\s+(" + "|".join(_MONTHS) + r")\b", re.IGNORECASE)
_WEEKDAYS = {
    0: "понеділок понеділка понедельник понедельника monday",
    1: "вівторок вівторка вторник вторника tuesday",
    2: "середа середи среда среды wednesday",
    3: "четвер четверга четверг четверга thursday",
    4: "пʼятниця пʼятниці п'ятниця п'ятниці пятница пятницы friday",
    5: "субота суботи суббота субботы saturday",
    6: "неділя неділі воскресенье воскресенья sunday",
}
_WEEKDAY = {word: day for day, words in _WEEKDAYS.items() for word in words.split()}
_WEEKDAY_RE = re.compile(r"(?<!\w)(" + "|".join(sorted(map(re.escape, _WEEKDAY), key=len, reverse=True)) + r")(?!\w)", re.IGNORECASE)
_TODAY = re.compile(r"\b(сьогодні|сегодня|today)\b", re.IGNORECASE)
_TOMORROW = re.compile(r"\b(завтра|tomorrow)\b", re.IGNORECASE)


def parse_due(due: str | None, discussed_on: str) -> str | None:
    """A Notion date for a deadline the model wrote as text ("понеділок, 23.11", "5 грудня").

    The prompt asks the model to add a calendar date to relative deadlines, so the date is
    read deterministically here (no extra LLM call). A missing year is the one closest to the day
    the task was discussed (a deadline in January discussed in December is next year)."""
    if not due:
        return None
    day = date.fromisoformat(discussed_on)
    if m := _ISO.search(due):
        return _safe(int(m[1]), int(m[2]), int(m[3]))
    if m := _DOT.search(due):
        return _with_year(day, int(m[1]), int(m[2]), int(m[3]) if m[3] else None)
    if m := _WORD.search(due):
        return _with_year(day, int(m[1]), _MONTHS[m[2].lower()], None)
    if _TODAY.search(due):
        return day.isoformat()
    if _TOMORROW.search(due):
        return (day + timedelta(days=1)).isoformat()
    if m := _WEEKDAY_RE.search(due.replace("’", "ʼ")):
        # "до пʼятниці" said on a Friday means the next one: always strictly after the discussion day.
        ahead = (_WEEKDAY[m[1].lower()] - day.weekday() - 1) % 7 + 1
        return (day + timedelta(days=ahead)).isoformat()
    return None


def _safe(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _with_year(discussed: date, day: int, month: int, year: int | None) -> str | None:
    if year is not None:
        return _safe(year, month, day)
    best = None
    for y in (discussed.year - 1, discussed.year, discussed.year + 1):
        iso = _safe(y, month, day)
        if iso and (best is None or abs((date.fromisoformat(iso) - discussed).days) < abs((date.fromisoformat(best) - discussed).days)):
            best = iso
    return best


def human_date(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{d}.{m}.{y}"


# ------------------------------------------------------------------------ rich text


def _rt(content: str, *, bold: bool = False, italic: bool = False, color: str = "default", url: str | None = None) -> dict[str, Any]:
    text: dict[str, Any] = {"content": content[:1900]}
    if url:
        text["link"] = {"url": url}
    return {
        "type": "text",
        "text": text,
        "annotations": {"bold": bold, "italic": italic, "strikethrough": False, "underline": False, "code": False, "color": color},
    }


def _mention(page_id: str) -> dict[str, Any]:
    return {"type": "mention", "mention": {"type": "page", "page": {"id": page_id}}}


def _block(kind: str, rich: list[dict[str, Any]] | str, **extra: Any) -> dict[str, Any]:
    if isinstance(rich, str):
        rich = [_rt(rich)]
    return {"object": "block", "type": kind, kind: {"rich_text": rich, **extra}}


def _heading(text: str) -> dict[str, Any]:
    return _block("heading_3", text)


def _bullets(items: list[str]) -> list[dict[str, Any]]:
    return [_block("bulleted_list_item", item) for item in items]


def _plain(rich: list[dict[str, Any]]) -> str:
    return "".join(r.get("text", {}).get("content", "") for r in rich)


# ----------------------------------------------------------------------- day pages


def day_status(row: Any) -> str:
    if row["status"] != "ok":
        return STATUS_EMPTY
    return STATUS_FINAL if row["is_final"] else STATUS_PROGRESS


def chat_type(row: Any) -> tuple[str, str]:
    kind = row["kind"] if "kind" in row.keys() else "group"
    return CHAT_TYPES.get(kind or "group", CHAT_TYPES["group"])


def _select_name(value: str) -> str:
    return value.replace(",", " ").strip()[:100]


select_name = _select_name  # public: the bot filters tasks by the same chat name


def day_icon(row: Any) -> dict[str, Any]:
    return {"type": "emoji", "emoji": chat_type(row)[1]}


HEADLINE_CHARS = 90


def day_headline(summary: dict[str, Any], status: str) -> str:
    """Page title = what the day was about. The chat and the date already have their own columns,
    so repeating them in the title was duplicate information (feedback from the first user)."""
    if status != "ok":
        voice = summary.get("not_transcribed")
        return f"Лише голосові ({voice}) — без розшифровки" if voice else "Нічого важливого"
    headline = ""
    for topic in summary.get("topics", [])[:3]:
        candidate = f"{headline} · {topic}" if headline else topic
        if len(candidate) > HEADLINE_CHARS and headline:
            break
        headline = candidate
    if not headline:
        headline = (summary.get("tldr") or "Підсумок дня").split(". ")[0]
    return headline if len(headline) <= HEADLINE_CHARS else headline[: HEADLINE_CHARS - 1] + "…"


def build_day_properties(row: Any) -> dict[str, Any]:
    s = json.loads(row["summary_json"])
    owners: list[str] = []
    for a in s.get("action_items", []):
        name = _select_name(a.get("owner") or "")
        if name and name not in owners:
            owners.append(name)
    title = row["title"] or str(row["chat_id"])
    return {
        "Назва": {"title": [_rt(day_headline(s, row["status"]))]},
        "Коротко": {"rich_text": [_rt(s.get("tldr") or "")] if s.get("tldr") else []},
        "Чат": {"select": {"name": _select_name(title)}},
        "Дата": {"date": {"start": row["local_date"]}},
        "Статус": {"select": {"name": day_status(row)}},
        "Тип": {"select": {"name": chat_type(row)[0]}},
        "Виконавці": {"multi_select": [{"name": o} for o in owners[:20]]},
        "Задач": {"number": len(s.get("action_items", []))},
        "Рішень": {"number": len(s.get("decisions", []))},
        "Відкритих питань": {"number": len(s.get("open_questions", []))},
        "Повідомлень": {"number": row["signal_messages"]},
        "Key": {"rich_text": [_rt(f"{row['chat_id']}:{row['local_date']}")]},
    }


def _task_line(item: dict[str, Any], page_id: str | None) -> list[dict[str, Any]]:
    rich = [_mention(page_id)] if page_id else [_rt(item["task"])]
    if item.get("owner"):
        rich.append(_rt(" · "))
        rich.append(_rt(item["owner"], bold=True))
    if item.get("due"):
        rich.append(_rt(f" · 📅 {item['due']}", color="gray"))
    return rich


def build_day_blocks(row: Any, tasks: list[tuple[dict[str, Any], str | None]] | None = None) -> list[dict[str, Any]]:
    """Page body. `tasks` pairs each action item with its page in the tasks database (a clickable mention)."""
    s = json.loads(row["summary_json"])
    blocks: list[dict[str, Any]] = []
    voice = None
    if s.get("not_transcribed"):
        voice = _block(
            "callout",
            f"Голосових і відеоповідомлень без розшифровки: {s['not_transcribed']}. Їхній зміст у самарі не врахований.",
            icon={"type": "emoji", "emoji": "🎙️"},
            color="yellow_background",
        )
    if row["status"] != "ok":
        blocks.append(_block("paragraph", [_rt("За цей день у чаті не було робочих обговорень.", color="gray")]))
        if voice:
            blocks.append(voice)
    else:
        blocks.append(_block("callout", [_rt(s.get("tldr") or "—")], icon={"type": "emoji", "emoji": "🧭"}, color="blue_background"))
        if voice:
            blocks.append(voice)

        tasks = tasks if tasks is not None else [(a, None) for a in s.get("action_items", [])]
        task_section = [_heading("✅ Задачі")] + [_block("bulleted_list_item", _task_line(a, pid)) for a, pid in tasks] if tasks else []
        decisions = s.get("decisions", [])
        decision_section = [_heading("🎯 Рішення"), *_bullets(decisions)] if decisions else []
        if task_section and decision_section:
            blocks.append({
                "object": "block",
                "type": "column_list",
                "column_list": {"children": [
                    {"object": "block", "type": "column", "column": {"children": task_section}},
                    {"object": "block", "type": "column", "column": {"children": decision_section}},
                ]},
            })
        else:
            blocks += task_section + decision_section

        if s.get("facts"):
            blocks += [_heading("📊 Ключові факти й цифри"), *_bullets(s["facts"])]
        if s.get("open_questions"):
            blocks += [_heading("❓ Відкриті питання"), *_bullets(s["open_questions"])]
        if s.get("links"):
            blocks.append(_heading("🔗 Посилання"))
            blocks += [_block("bulleted_list_item", [_rt(url, url=url)]) for url in s["links"]]
        if s.get("topics"):
            blocks.append(_block("paragraph", [_rt("Теми: ", color="gray"), _rt(" · ".join(s["topics"]), color="gray")]))

    footer = (
        f"Враховано повідомлень: {row['signal_messages']} · шум: {row['noise_messages']} · "
        f"модель: {row['model']} · оновлено: {row['updated_at'][:16].replace('T', ' ')} UTC"
    )
    blocks.append({"object": "block", "type": "divider", "divider": {}})
    blocks.append(_block("paragraph", [_rt(footer, color="gray", italic=True)]))
    return blocks


# --------------------------------------------------------------------------- tasks


def _norm(text: str | None) -> str:
    return re.sub(r"[\W_]+", " ", (text or "").lower()).strip()


def task_key(chat_id: int, local_date: str, item: dict[str, Any]) -> str:
    """Stable id of a task: same text and owner on the same chat-day = same Notion row.
    A reassigned task (new owner) is a new row; the old one is removed unless someone already worked on it."""
    digest = hashlib.sha1(f"{_norm(item['task'])}|{_norm(item.get('owner'))}".encode()).hexdigest()[:12]
    return f"{chat_id}:{local_date}:{digest}"


def build_task_properties(row: Any, item: dict[str, Any], key: str, day_page_id: str | None, *, new: bool) -> dict[str, Any]:
    owner = _select_name(item.get("owner") or "")
    due_iso = parse_due(item.get("due"), row["local_date"])
    props: dict[str, Any] = {
        "Задача": {"title": [_rt(item["task"])]},
        "Виконавець": {"select": {"name": owner} if owner else None},
        "Дедлайн": {"date": {"start": due_iso} if due_iso else None},
        "Термін": {"rich_text": [_rt(item["due"])] if item.get("due") else []},
        "Чат": {"select": {"name": _select_name(row["title"] or str(row["chat_id"]))}},
        "Обговорено": {"date": {"start": row["local_date"]}},
        "Key": {"rich_text": [_rt(key)]},
    }
    if day_page_id:
        props["День"] = {"relation": [{"id": day_page_id}]}
    if new:
        props["Стан"] = {"select": {"name": TASK_NEW}}  # afterwards the state belongs to the user
    return props


# Back-compat names used by older tests / callers.
build_blocks = build_day_blocks
build_properties = build_day_properties


# ---------------------------------------------------------------- projects (beyond the task)

PROJECTS_TITLE = "Telegram — проєкти"
LOG_TITLE = "Telegram — журнал проєкту"
OLD_LOG_TITLE = "Telegram — рішення"  # the first version of this database held decisions only
PROJECT_ACTIVE, PROJECT_PAUSED, PROJECT_DONE = "Активний", "На паузі", "Завершено"
# Columns the USER owns on a project (set once on creation, never overwritten by the connector).
PROJECT_USER_COLUMNS = ("Проєкт", "Чати", "Статус", "Дата релізу")

LOG_DECISION = "🎯 Рішення"
LOG_TYPES = {
    "decision": LOG_DECISION,
    "experiment": "🧪 Тест",
    "analytics": "📊 Аналітика",
    "release": "🚀 Реліз",
    "incident": "⚠️ Проблема",
}
LOG_EFFECTS = {"worked": "✅ Спрацювало", "failed": "❌ Не спрацювало", "in_progress": "🔄 У процесі"}
# Columns the connector fills on log rows; whatever the user changes there afterwards stays the user's.
LOG_AI_COLUMNS = ("Запис", "Дата", "Тип", "Результат", "Ефект", "Висновок", "Метрика", "Значення")


def project_properties_schema() -> dict[str, Any]:
    return {
        "Проєкт": {"title": {}},
        "Чати": {"multi_select": {}},
        "Статус": {
            "select": {
                "options": [
                    {"name": PROJECT_ACTIVE, "color": "green"},
                    {"name": PROJECT_PAUSED, "color": "yellow"},
                    {"name": PROJECT_DONE, "color": "gray"},
                ]
            }
        },
        "Дата релізу": {"date": {}},
        "Оновлено": {"date": {}},
    }


def project_relation_spec(projects_data_source_id: str) -> dict[str, Any]:
    """One-way «Проєкт» link: rows know their project, the project page is not flooded with lists of rows."""
    return {"relation": {"data_source_id": projects_data_source_id, "type": "single_property", "single_property": {}}}


def log_properties_schema(projects_data_source_id: str) -> dict[str, Any]:
    return {
        "Запис": {"title": {}},
        "Дата": {"date": {}},
        "Тип": {
            "select": {
                "options": [
                    {"name": LOG_TYPES["decision"], "color": "blue"},
                    {"name": LOG_TYPES["experiment"], "color": "purple"},
                    {"name": LOG_TYPES["analytics"], "color": "green"},
                    {"name": LOG_TYPES["release"], "color": "orange"},
                    {"name": LOG_TYPES["incident"], "color": "red"},
                ]
            }
        },
        "Важливе": {"checkbox": {}},
        "Результат": {"rich_text": {}},
        "Ефект": {
            "select": {
                "options": [
                    {"name": LOG_EFFECTS["worked"], "color": "green"},
                    {"name": LOG_EFFECTS["failed"], "color": "red"},
                    {"name": LOG_EFFECTS["in_progress"], "color": "yellow"},
                ]
            }
        },
        "Висновок": {"rich_text": {}},
        "Метрика": {"select": {}},
        "Значення": {"number": {"format": "number"}},
        "Чат": {"select": {}},
        "Проєкт": project_relation_spec(projects_data_source_id),
        "Key": {"rich_text": {}},
    }


def decision_key(chat_id: int, local_date: str, text: str) -> str:
    digest = hashlib.sha1(_norm(text).encode()).hexdigest()[:12]
    return f"{chat_id}:{local_date}:{digest}"


def build_decision_properties(row: Any, text: str, key: str, project_ids: list[str], *, new: bool) -> dict[str, Any]:
    """A decision as it stands in the day summary. Type, importance and conclusion are set later (by the daily
    update) or by the user, so an update of the row never touches them."""
    props = {
        "Запис": {"title": [_rt(text)]},
        "Дата": {"date": {"start": row["local_date"]}},
        "Чат": {"select": {"name": _select_name(row["title"] or str(row["chat_id"]))}},
        "Проєкт": {"relation": [{"id": pid} for pid in project_ids]},
        "Key": {"rich_text": [_rt(key)]},
    }
    return props | {"Тип": {"select": {"name": LOG_DECISION}}} if new else props


def build_log_properties(entry: dict[str, Any]) -> dict[str, Any]:
    """Columns of a log row written by the daily update (tests, analytics, releases, problems)."""
    metric = _select_name(entry.get("metric") or "")
    effect = LOG_EFFECTS.get(entry.get("effect") or "")
    return {
        "Запис": {"title": [_rt(entry["title"])]},
        "Дата": {"date": {"start": entry["date"]}},
        "Тип": {"select": {"name": LOG_TYPES[entry["type"]]}},
        "Результат": {"rich_text": [_rt(entry["result"])] if entry.get("result") else []},
        "Ефект": {"select": {"name": effect} if effect else None},
        "Висновок": {"rich_text": [_rt(entry["takeaway"])] if entry.get("takeaway") else []},
        "Метрика": {"select": {"name": metric} if metric else None},
        "Значення": {"number": entry.get("value")},
    }


def plain_value(prop: dict[str, Any] | None) -> Any:
    """Comparable value of a Notion property, both as sent by us and as returned by the API."""
    prop = prop or {}
    for kind in ("title", "rich_text"):
        if kind in prop:
            return "".join(t.get("plain_text") or t.get("text", {}).get("content", "") for t in prop[kind] or []).strip()
    if "select" in prop:
        return (prop["select"] or {}).get("name")
    if "date" in prop:
        return (prop["date"] or {}).get("start")
    if "number" in prop:
        return prop["number"]
    return bool(prop.get("checkbox"))


def build_new_project_properties(chat_name: str) -> dict[str, Any]:
    return {
        "Проєкт": {"title": [_rt(chat_name)]},
        "Чати": {"multi_select": [{"name": _select_name(chat_name)}]},
        "Статус": {"select": {"name": PROJECT_ACTIVE}},
    }


PASSPORT_TITLE = "🧭 Паспорт проєкту"


def build_brief_header(updated_on: str, model: str) -> list[dict[str, Any]]:
    return [_rt(f"{PASSPORT_TITLE} · оновлено {human_date(updated_on)}", bold=True), _rt(f" · {model}", color="gray")]


def build_brief_children(brief: dict[str, Any], today: str) -> list[dict[str, Any]]:
    """Children of the passport callout: the only part of a project page the connector rewrites."""
    blocks: list[dict[str, Any]] = []
    if brief.get("summary"):
        blocks += [_heading("📌 Про проєкт"), _block("paragraph", brief["summary"])]
    if brief.get("status"):
        blocks += [_heading("📍 Поточний стан"), _block("paragraph", brief["status"])]
    owners = [f"{o['person']} — {o['area']}" for o in brief.get("owners", []) if o.get("person")]
    milestones = [f"{human_date(m['date'])} — {m['text']}" + (" (план)" if m["date"] > today else "")
                  for m in brief.get("milestones", [])]
    sections = [
        ("👥 Команда: хто за що відповідає", owners),
        ("🗓 Віхи проєкту", milestones),
        ("⚠️ Ризики й блокери", brief.get("risks", [])),
        ("➡️ Наступні кроки", brief.get("next_steps", [])),
        ("❓ Відкриті питання", brief.get("open_questions", [])),
    ]
    for title, items in sections:
        if items:
            blocks += [_heading(title), *_bullets(items)]
    return blocks or [_block("paragraph", [_rt("Поки що немає даних для паспорта.", color="gray")])]


def build_team_section() -> list[dict[str, Any]]:
    """Template created once; from then on it belongs to the team."""

    def hint(text: str) -> dict[str, Any]:
        return _block("bulleted_list_item", [_rt(text, color="gray", italic=True)])

    return [
        _block("heading_2", "✍️ Від команди"),
        _block("paragraph", [_rt("Те, чого немає в чатах. Цей розділ пишете ви: конектор його не змінює.", color="gray")]),
        _heading("🚦 З чого почати новачку"),
        hint("перші кроки, кого спитати, що прочитати"),
        _heading("🔑 Доступи"),
        hint("назва сервісу і хто видає доступ; паролі й ключі тут не зберігаємо"),
        _heading("☎️ Контакти"),
        hint("імʼя, роль, як звʼязатися"),
        _heading("📚 Документи й посилання"),
        hint("дизайн, аналітика, таск-трекер, регламенти"),
    ]


def checklist_view_spec(project_page_id: str, title: str, release_date: str | None, t: dict[str, str]) -> dict[str, Any]:
    """Pre-release checklist: a view of the TASKS database. `t`: current column names there."""
    date_filter = (
        {"property": t["Дедлайн"], "date": {"on_or_before": release_date}}
        if release_date else {"property": t["Дедлайн"], "date": {"is_not_empty": True}}
    )
    return {
        "name": f"🚀 До релізу · {title}"[:100],
        "type": "table",
        "filter": {"and": [{"property": t["Проєкт"], "relation": {"contains": project_page_id}}, date_filter]},
        "sorts": [{"property": t["Дедлайн"], "direction": "ascending"}],
    }
