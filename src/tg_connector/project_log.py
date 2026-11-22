"""Project log: the memory of a project — what was decided, tested, measured, released, what broke.

Only what matters for the product: decisions (the model picks them by id from the day summaries, their text is
never rewritten), tests, analytics, releases and problems. Everything comes from the same daily LLM call as the
passport. One event is one row: a later result updates the row of the test instead of adding a new one, a metric
has one row per day, so its dynamics stay readable. Rows the user edited, added or removed are respected.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

from .notion_layout import _norm, decision_key, select_name

MAX_NEW_ITEMS = 25  # per call: the log is a digest, not a second chat
MAX_CONTEXT_ROWS = 60  # log rows shown to the model
MAX_REMOVED_SHOWN = 30


class KeyDecision(BaseModel):
    id: str = Field(description='Id of a decision from DAILY SUMMARIES, e.g. "D3"')
    takeaway: str | None = Field(description="The reason or consequence that is NOT already in the text of the decision; null if there is none")


class LogItem(BaseModel):
    id: str | None = Field(description='Id of the PROJECT LOG entry this item continues, e.g. "L2"; null for a new entry')
    date: str = Field(description="YYYY-MM-DD, the day it happened or was reported")
    type: Literal["experiment", "analytics", "release", "incident"]
    title: str = Field(description="Short and self-contained, up to 12 words")
    result: str | None = Field(description="What came out, with figures; null if unknown")
    effect: Literal["worked", "failed", "in_progress"] | None = Field(description="experiment only, and only if the data says so")
    takeaway: str | None = Field(description="Cause or lesson stated in the data, one sentence; null if there is none")
    metric: str | None = Field(description='analytics only: stable metric name with its unit, e.g. "CPI Android ($)"')
    value: float | None = Field(description="analytics only: the measured number")


LOG_RULES = """LOG — "log" and "key_decisions" are the project's memory: newcomers and managers use it to follow the dynamics and not to repeat what did not work. Only what affects the product, its users, metrics, money or the release plan. Never logistics: meetings, calls, vacations, interviews, who prepares a report or a document.
- key_decisions: ids of decisions (D…) that changed the product, its scope, release plan, budget or strategy.
- log, one item per event:
  - experiment: a test, campaign or hypothesis — what was tried, the result in figures, effect; "in_progress" until the data shows a result. A comparison of variants (A/B, creative A vs creative B, one channel vs another) is an experiment, and so is a test that was agreed and scheduled;
  - analytics: ONE measured figure per item with "metric" and "value". EVERY figure about the product or its users that can be tracked over time (CPI, CTR, conversion, crash-free, revenue, retention) gets its own item, also when it is mentioned inside another entry. Reuse names from KNOWN METRICS exactly. If the data gives the earlier value, put it into "result" (was → is);
  - release: a version shipped or moved;
  - incident: a bug, crash or failure and how it was solved.
- Other decisions and tasks are NOT log items (decisions are referenced by id).
- "takeaway" is a cause or lesson that PEOPLE stated in the data (what to do, what not to repeat) and that the title and the result do not say yet. Never rephrase the title, never draw your own conclusions. If there is none, null.
- If an event already stands in PROJECT LOG, do not repeat it. If the new summaries bring its result or continuation, return it once with its id (L…) and the new fields. Entries with id "-" belong to the user: do not repeat and do not change them.
- Never return anything listed under REMOVED BY THE USER, in any wording.
- Figures exactly as in the data; nothing the data does not support."""


def _iso(value: str) -> str | None:
    try:
        return date.fromisoformat(value.strip()[:10]).isoformat()
    except ValueError:
        return None


def _clip(value: str | None, limit: int) -> str | None:
    text = " ".join((value or "").split())
    return text[:limit] or None


def entry_key(entry: dict[str, Any]) -> str:
    """Identity of an event: the same metric on the same day, or the same title on the same day, is one row."""
    if entry["type"] == "analytics" and entry.get("metric"):
        return f"analytics:{entry['date']}:{_norm(entry['metric'])}"
    return f"{entry['type']}:{entry['date']}:{hashlib.sha1(_norm(entry['title']).encode()).hexdigest()[:12]}"


def clean_item(item: LogItem, first_day: str, last_day: str) -> dict[str, Any] | None:
    title = _clip(item.title, 150)
    if not title:
        return None
    day = _iso(item.date) or last_day
    analytics = item.type == "analytics"
    return {
        "type": item.type,
        "date": min(max(day, first_day), last_day),
        "title": title,
        "result": _clip(item.result, 600),
        "effect": item.effect if item.type == "experiment" else None,
        "takeaway": _clip(item.takeaway, 400),
        "metric": (select_name(_clip(item.metric, 80) or "") or None) if analytics else None,
        "value": item.value if analytics else None,
    }


@dataclass
class LogContext:
    """What the model sees of the log, and how its answer is turned into rows."""

    own: dict[str, Any] = field(default_factory=dict)  # "L1" → project_log row
    stored: dict[str, Any] = field(default_factory=dict)  # entry key → project_log row (also rows not shown)
    decisions: dict[str, dict[str, Any]] = field(default_factory=dict)  # "D1" → its log entry + "key"
    lines: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    removed_ids: list[int] = field(default_factory=list)
    known_decisions: set[str] = field(default_factory=set)
    foreign: set[tuple[str, str]] = field(default_factory=set)  # (date, title) of rows the connector does not own

    @classmethod
    def build(cls, stored: list[Any], in_notion: list[dict[str, Any]], today: str) -> LogContext:
        """`stored`: project_log rows of the project; `in_notion`: its rows as the user sees them now."""
        ctx = cls()
        by_page = {row["page_id"]: row for row in stored if row["page_id"]}
        visible = {r["page_id"] for r in in_notion}
        ctx.known_decisions = {row["entry_key"] for row in stored
                               if row["entry_key"].startswith("decision:") and row["state"] not in ("obsolete", "retired")}
        for row in stored:
            ctx.stored[row["entry_key"]] = row
            gone = row["page_id"] and row["page_id"] not in visible and row["created_on"] < today
            if row["state"] == "dismissed" or gone:
                ctx.removed.append(json.loads(row["entry_json"]).get("title") or "")
                if row["state"] != "dismissed":
                    ctx.removed_ids.append(row["id"])
        ctx.removed = [title for title in ctx.removed if title][-MAX_REMOVED_SHOWN:]
        ctx.metrics = sorted({r["metric"] for r in in_notion if r.get("metric")})
        for r in in_notion[-MAX_CONTEXT_ROWS:]:
            row = by_page.get(r["page_id"])
            mine = row is not None and not row["entry_key"].startswith("decision:")
            ref = f"L{len(ctx.own) + 1}" if mine else "-"
            if mine:
                ctx.own[ref] = row
            fields = ("date", "type", "title", "result", "effect", "takeaway", "metric", "value")
            ctx.lines.append(json.dumps({"id": ref} | {k: r[k] for k in fields if r.get(k) is not None}, ensure_ascii=False))
        ctx.foreign = {(r["date"] or "", _norm(r["title"])) for r in in_notion if r["page_id"] not in by_page}
        return ctx

    def label_decisions(self, day: dict[str, Any], chat_id: int) -> dict[str, Any]:
        """Give the decisions of a day view short ids the model can refer to."""
        labelled = []
        for text in day["decisions"]:
            ref = f"D{len(self.decisions) + 1}"
            self.decisions[ref] = {"key": decision_key(chat_id, day["date"], text), "date": day["date"],
                                   "title": text, "chat": day["chat"]}
            labelled.append({"id": ref, "text": text})
        return day | {"decisions": labelled}

    def apply(self, key_decisions: list[KeyDecision], items: list[LogItem], first_day: str, last_day: str) -> dict[str, dict[str, Any]]:
        """The model's answer → {entry_key: entry} to store. Ids it invented, repeats and empty items are dropped."""
        out: dict[str, dict[str, Any]] = {}
        removed = {_norm(title) for title in self.removed}
        for d in key_decisions:
            found = self.decisions.get(d.id.strip())
            if found is None or f"decision:{found['key']}" in self.known_decisions:
                continue
            if _norm(found["title"]) not in removed and (found["date"], _norm(found["title"])) not in self.foreign:
                out[f"decision:{found['key']}"] = {
                    "type": "decision", "date": found["date"], "title": found["title"][:1900], "chat": found["chat"],
                    "result": None, "effect": None, "takeaway": _clip(d.takeaway, 400), "metric": None, "value": None}
        new = 0
        for item in items:
            entry = clean_item(item, first_day, last_day)
            if entry is None or _norm(entry["title"]) in removed:
                continue
            row = self.own.get((item.id or "").strip())  # decisions are not among them: their text is fixed
            if row is not None and entry["type"] == "analytics" and json.loads(row["entry_json"])["date"] != entry["date"]:
                row = None  # a new measurement of a known metric is a new row: that is the dynamics
            row = row or self.stored.get(entry_key(entry))
            if row is not None:
                before = json.loads(row["entry_json"])
                merged = before | {k: entry[k] for k in ("result", "effect", "takeaway", "metric", "value") if entry[k] is not None}
                if merged != before:
                    out[row["entry_key"]] = merged
                continue
            if (entry["date"], _norm(entry["title"])) in self.foreign or new >= MAX_NEW_ITEMS:
                continue
            key = entry_key(entry)
            if key not in out:
                out[key] = entry
                new += 1
        return out
