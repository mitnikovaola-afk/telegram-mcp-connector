"""Run the dialogue scenarios through the REAL pipeline (noise filter → masking → prompt → LLM → validation)
and check the summaries automatically. Telegram and Notion are not involved; the LLM is real.

    uv run python evals/run_eval.py                          # model from .env
    uv run python evals/run_eval.py --model gemini-3.7-flash # another model (free tier quota is per model)
    uv run python evals/run_eval.py --only secrets_in_chat,english_chat

Every scenario costs 1 LLM call (the cross-midnight one costs 2). Results: evals/results/<date>-<model>.md
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenarios import KYIV, SCENARIOS, Scenario, when  # noqa: E402

from tg_connector.config import load_settings  # noqa: E402
from tg_connector.db import Database  # noqa: E402
from tg_connector.llm import LLMError, build_llm  # noqa: E402
from tg_connector.models import ChatInfo, RawMessage  # noqa: E402
from tg_connector.noise import prepare  # noqa: E402
from tg_connector.summarizer import summarize_pending  # noqa: E402

CHAT_ID = -100500
API_ATTEMPTS = 3  # free-tier models often answer 503 "high demand" for a minute or two
API_RETRY_PAUSE_S = 30
NOW = datetime(2026, 11, 21, 12, 0, tzinfo=KYIV)  # every scenario day is already closed


class RecordingLLM:
    def __init__(self, inner) -> None:  # noqa: ANN001
        self.inner = inner
        self.model_name = inner.model_name
        self.prompts: list[str] = []

    async def generate_json(self, system: str, user: str, schema) -> str:  # noqa: ANN001
        self.prompts.append(user)
        return await self.inner.generate_json(system, user, schema)


async def run_scenario(sc: Scenario, settings, llm) -> dict:  # noqa: ANN001
    with tempfile.TemporaryDirectory() as tmp:
        cfg = dataclasses.replace(settings, data_dir=Path(tmp), summarize_today=True)
        db = Database(cfg.db_path)
        db.upsert_chat(ChatInfo(CHAT_ID, sc.chat_title, sc.kind))
        db.set_tracked(CHAT_ID, True)
        raws = [
            RawMessage(CHAT_ID, i, when(m.time), abs(hash(m.sender)) % 10_000, m.sender, m.kind, m.text, m.reply_to)
            for i, m in enumerate(sc.messages, 1)
        ]
        db.store_batch(CHAT_ID, [prepare(r, cfg.timezone) for r in raws], new_cursor=len(raws))
        rec = RecordingLLM(llm)
        stats = await summarize_pending(db, rec, cfg, NOW)
        row = db.get_summary(CHAT_ID, sc.date)
        summary = json.loads(row["summary_json"]) if row else {}
        db.close()
    prompt = "\n".join(rec.prompts)
    api_error = None if summary else next((e["error"] for e in stats.get("errors", [])), "no summary")
    results = []
    for label, check in sc.checks:
        ok, detail = check(summary, prompt) if summary else (False, f"no summary: {stats.get('errors')}")
        results.append({"check": label, "ok": ok, "detail": detail})
    return {"scenario": sc, "summary": summary, "checks": results, "stats": stats, "prompt": prompt, "api_error": api_error}


def _counted(outcomes: list[dict]) -> list[dict]:
    """Scenarios the model actually answered (API outages are reported separately, not as quality failures)."""
    return [o for o in outcomes if not o["api_error"]]


def write_report(path: Path, model: str, outcomes: list[dict]) -> None:
    passed = sum(c["ok"] for o in _counted(outcomes) for c in o["checks"])
    total = sum(len(o["checks"]) for o in _counted(outcomes))
    not_run = [o["scenario"].name for o in outcomes if o["api_error"]]
    lines = [
        f"# Eval: діалогові edge cases — `{model}`",
        "",
        f"Запуск: {datetime.now(KYIV):%Y-%m-%d %H:%M} (Київ). Перевірок пройдено: **{passed}/{total}**.",
        "",
        *([f"Не виконано через недоступність API (не рахується як провал): {', '.join(not_run)}.", ""] if not_run else []),
        "| Сценарій | Ситуація | Результат |",
        "|---|---|---|",
    ]
    for o in outcomes:
        sc, ok = o["scenario"], sum(c["ok"] for c in o["checks"])
        if o["api_error"]:
            lines.append(f"| `{sc.name}` | {sc.title} | ⚠️ API недоступне |")
            continue
        mark = "✅" if ok == len(o["checks"]) else "❌"
        lines.append(f"| `{sc.name}` | {sc.title} | {mark} {ok}/{len(o['checks'])} |")
    for o in outcomes:
        sc = o["scenario"]
        lines += ["", f"## `{sc.name}` — {sc.title}", "", "**Діалог:**", "```"]
        lines += [f"{m.time} {m.sender}: {m.text or '[' + m.kind + ']'}" for m in sc.messages]
        lines += ["```", ""]
        if o["api_error"]:
            lines += [f"⚠️ Не виконано, API недоступне: {o['api_error'][:300]}"]
            continue
        lines += ["**Перевірки:**"]
        lines += [f"- {'✅' if c['ok'] else '❌'} {c['check']} — {c['detail']}" for c in o["checks"]]
        lines += ["", "**Самарі:**", "```json", json.dumps(o["summary"], ensure_ascii=False, indent=1), "```"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--only", help="comma-separated scenario names")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    settings = load_settings()
    model = args.model or settings.gemini_model
    llm = build_llm(settings.gemini_api_key, model, settings.gemini_thinking_level)
    if llm is None:
        print("GEMINI_API_KEY is not set in .env")
        return 2
    chosen = [s for s in SCENARIOS if not args.only or s.name in args.only.split(",")]

    outcomes = []
    quota_hit = False
    for sc in chosen:
        for attempt in range(1, API_ATTEMPTS + 1):
            try:
                outcome = await run_scenario(sc, settings, llm)
            except LLMError as exc:
                outcome = {"scenario": sc, "summary": {}, "checks": [], "stats": {}, "prompt": "", "api_error": str(exc)}
            error = outcome["api_error"] or ""
            if not error or "429" in error or attempt == API_ATTEMPTS:
                break
            print(f"   {sc.name}: API unavailable ({error[:80]}…), retrying in {API_RETRY_PAUSE_S}s")
            await asyncio.sleep(API_RETRY_PAUSE_S)
        outcomes.append(outcome)
        if outcome["api_error"]:
            print(f"SKIP      {sc.name}: {outcome['api_error'][:160]}")
            if "429" in outcome["api_error"]:
                quota_hit = True
                break
            continue
        ok = sum(c["ok"] for c in outcome["checks"])
        print(f"{'PASS' if ok == len(outcome['checks']) else 'FAIL'} {ok}/{len(outcome['checks'])}  {sc.name}")
        for c in outcome["checks"]:
            if not c["ok"]:
                print(f"     ✗ {c['check']} — {c['detail'][:220]}")
    if quota_hit:
        print("Daily quota of this model is exhausted — the remaining scenarios were not run.")

    out = args.out or Path(__file__).parent / "results" / f"{datetime.now(KYIV):%Y-%m-%d}-{model}.md"
    write_report(out, model, outcomes)
    counted = _counted(outcomes)
    passed = sum(c["ok"] for o in counted for c in o["checks"])
    total = sum(len(o["checks"]) for o in counted)
    print(f"\n{passed}/{total} checks passed in {len(counted)}/{len(chosen)} scenarios · report: {out}")
    return 0 if passed == total and len(counted) == len(chosen) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
