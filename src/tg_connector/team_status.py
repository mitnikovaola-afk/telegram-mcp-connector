"""Replies of the team bot (/status, /tasks) for ONE chat, built from stored data — no LLM call.

Why no LLM: the answer is instant, costs nothing (the Gemini free tier is ~20 requests a day and a
team calling /status would use it up by noon), and it can only contain what is already in the
summaries. Task states come from the Notion tasks database, because that is where people tick them.
"""

from __future__ import annotations

import html
import json
from datetime import date, datetime, timedelta
from typing import Any

from .config import Settings
from .db import Database
from .notion_layout import TASK_DONE, parse_due

STATUS_DAYS = 7
# Open questions only from the latest summarized day: an older one may have been answered since
# (seen live: «пейвол у 2.4 чи 2.5?» from 20.11 was decided on 21.11). Missing beats misleading.
MAX_DECISIONS = 6
MAX_TASKS = 10
MAX_QUESTIONS = 5
TELEGRAM_LIMIT = 4000

HELP_TEXT = (
    "<b>Я показую статус проєкту цього чату</b> — із самарі, які збирає конектор.\n\n"
    "/status — рішення, задачі в роботі та відкриті питання за 7 днів\n"
    "/tasks — лише відкриті задачі, прострочені зверху\n"
    "/help — ця довідка\n\n"
    "<i>Я бачу лише команди, а не листування. Дані оновлюються з кожною синхронізацією.</i>"
)
PRIVATE_TEXT = (
    "Я працюю в робочих групах, до яких мене додали: напишіть там /status або /tasks.\n"
    "В особистих повідомленнях я не показую дані проєктів."
)
ERROR_TEXT = "Не вдалося зібрати статус. Спробуйте трохи пізніше."


def _e(text: str | None) -> str:
    return html.escape(text or "", quote=False)


def notion_url(page_id: str | None) -> str | None:
    return f"https://www.notion.so/{page_id.replace('-', '')}" if page_id else None


def _ddmm(iso: str) -> str:
    return f"{date.fromisoformat(iso):%d.%m}"


def synced_label(db: Database, chat_id: int, settings: Settings) -> str:
    chat = db.get_chat(chat_id)
    raw = chat["last_synced_at"] if chat else None
    if not raw:
        return "ще не синхронізовано"
    return f"{datetime.fromisoformat(raw).astimezone(settings.timezone):%d.%m %H:%M}"


def collect(db: Database, chat_id: int, today: date) -> dict[str, Any]:
    """Decisions, open questions and tasks of the chat's last STATUS_DAYS days (newest first)."""
    since = (today - timedelta(days=STATUS_DAYS - 1)).isoformat()
    rows = [r for r in db.list_summaries(chat_id, since, today.isoformat(), limit=STATUS_DAYS) if r["status"] == "ok"]
    decisions: list[tuple[str, str]] = []
    questions: list[tuple[str, str]] = []
    seen_questions: set[str] = set()
    tasks: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        summary = json.loads(row["summary_json"])
        decisions += [(row["local_date"], d) for d in summary.get("decisions", [])]
        if i == 0:
            for q in summary.get("open_questions", []):
                if q.lower() not in seen_questions:
                    seen_questions.add(q.lower())
                    questions.append((row["local_date"], q))
        for a in summary.get("action_items", []):
            tasks.append({
                "task": a["task"], "owner": a.get("owner"), "due": a.get("due"),
                "due_date": parse_due(a.get("due"), row["local_date"]), "state": None, "url": None,
            })
    latest_page = next((r["notion_page_id"] for r in rows if r["notion_page_id"]), None)
    return {
        "since": since,
        "decisions": decisions[:MAX_DECISIONS],
        "questions": questions[:MAX_QUESTIONS],
        "summary_tasks": tasks,
        "latest_page": latest_page,
        "has_data": bool(rows),
    }


def order_tasks(tasks: list[dict[str, Any]], today: date) -> list[dict[str, Any]]:
    """Overdue first, then by deadline, tasks without a deadline last; finished ones dropped."""
    iso = today.isoformat()
    open_tasks = [t for t in tasks if t.get("state") != TASK_DONE]

    def key(t: dict[str, Any]) -> tuple[int, str]:
        due = t.get("due_date")
        if due and due < iso:
            return (0, due)
        return (1, due) if due else (2, "")

    return sorted(open_tasks, key=key)[:MAX_TASKS]


def _task_line(t: dict[str, Any], today: date) -> str:
    due = t.get("due_date")
    overdue = bool(due and due < today.isoformat())
    owner = f"<b>{_e(t['owner'])}</b>" if t.get("owner") else "<i>без виконавця</i>"
    title = f'<a href="{_e(t["url"])}">{_e(t["task"])}</a>' if t.get("url") else _e(t["task"])
    when = f" · до {_ddmm(due)}" if due else (f" · {_e(t['due'])}" if t.get("due") else "")
    if overdue:
        return f"⚠️ {owner} — {title}{when} <b>(прострочено)</b>"
    return f"• {owner} — {title}{when}"


def _tasks_block(tasks: list[dict[str, Any]], from_notion: bool, today: date) -> list[str]:
    lines = [_task_line(t, today) for t in tasks] or ["• відкритих задач немає"]
    if not from_notion:
        lines.append("<i>Стан задач у Notion зараз недоступний — показано задачі із самарі (без позначок «Готово»).</i>")
    return lines


def _fit(text: str) -> str:
    return text if len(text) <= TELEGRAM_LIMIT else text[: TELEGRAM_LIMIT - 1] + "…"


def render_status(title: str, data: dict[str, Any], tasks: list[dict[str, Any]], from_notion: bool,
                  today: date, synced: str) -> str:
    header = [f"<b>📊 Статус: {_e(title)}</b>",
              f"<i>за {_ddmm(data['since'])}–{today:%d.%m} · дані станом на {synced}</i>"]
    if not data["has_data"] and not tasks:
        return "\n".join(header + ["", "За останні 7 днів самарі ще немає: вони зʼявляться після найближчої синхронізації."])
    lines = header + ["", "<b>✅ Рішення</b>"]
    lines += [f"• {_ddmm(d)} — {_e(text)}" for d, text in data["decisions"]] or ["• рішень за цей період немає"]
    lines += ["", "<b>📌 Задачі в роботі</b>", *_tasks_block(tasks, from_notion, today)]
    if data["questions"]:
        lines += ["", "<b>❓ Відкриті питання</b>"]
        lines += [f"• {_ddmm(d)} — {_e(q)}" for d, q in data["questions"]]
    url = notion_url(data["latest_page"])
    if url:
        lines += ["", f'🔗 <a href="{url}">Останній день у Notion</a>']
    return _fit("\n".join(lines))


def render_tasks(title: str, tasks: list[dict[str, Any]], from_notion: bool, today: date, synced: str) -> str:
    lines = [f"<b>📌 Відкриті задачі: {_e(title)}</b>", f"<i>дані станом на {synced}</i>", ""]
    lines += _tasks_block(tasks, from_notion, today)
    return _fit("\n".join(lines))
