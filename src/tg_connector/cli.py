"""Command line: setup helpers, one-off runs (cron/launchd/demo) and the MCP server entry point.

    uv run tg-connector login            # one-time Telegram login (phone + code)
    uv run tg-connector check            # verify Telegram / Gemini / Notion configuration
    uv run tg-connector chats [-q text]  # list chats with ids
    uv run tg-connector track <id>...    # choose chats to sync (--from-now: skip history)
    uv run tg-connector sync             # one pipeline run (what the scheduler does)
    uv run tg-connector status           # cursors, pending work, recent runs
    uv run tg-connector summaries        # print stored summaries
    uv run tg-connector serve            # MCP server over stdio (used by Claude Desktop / Claude Code)
    uv run tg-connector bot              # team bot for group chats: /status, /tasks (--sync: + periodic sync)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

from .config import load_settings
from .logging_setup import setup_logging
from .pipeline import build_app, status_report, track_chat


def _print_report(report: dict[str, Any]) -> None:
    if report.get("status") == "skipped":
        print(f"Run skipped: {report.get('reason')}")
        return
    print(f"=== Run #{report['run_id']} ({report['trigger']}) — {report['status']}, {report['duration_s']}s")
    sync = report.get("sync", {})
    if "note" in sync:
        print(f"Telegram : {sync['note']}")
    elif sync:
        print(
            f"Telegram : chats={sync['chats']} fetched={sync['fetched']} new={sync['inserted']} "
            f"updated={sync['updated']} deleted={sync.get('deleted', 0)} changed_days={sync['dirty_days']}"
        )
        for m in sync.get("migrated", []):
            print(f"           group upgraded to supergroup: {m['from']} → {m['to']} ({m['title']}), now tracked")
    s = report.get("summaries", {})
    if "skipped" in s:
        print(f"Summaries: skipped — {s['skipped']}")
    elif s:
        print(
            f"Summaries: llm_calls={s['llm_calls']} summarized={s['summarized']} unchanged={s['unchanged']} "
            f"no_signal={s['no_signal']} finalized={s['finalized']} failed={s['failed']} deferred_today={s['deferred_today']}"
        )
        if s.get("llm_paused_until"):
            print(f"           LLM quota exhausted until {s['llm_paused_until']}: {s['deferred_quota']} day(s) wait in the queue")
    n = report.get("notion", {})
    if "skipped" in n:
        print(f"Notion   : skipped — {n['skipped']}")
    elif n:
        print(f"Notion   : created={n['created']} updated={n['updated']} skipped_empty={n['skipped_empty']} failed={n['failed']}")
        if any(n.get(k) for k in ("tasks_created", "tasks_updated", "tasks_removed")):
            print(
                f"           tasks: created={n.get('tasks_created', 0)} updated={n.get('tasks_updated', 0)} "
                f"removed={n.get('tasks_removed', 0)}"
            )
    b = report.get("projects", {})
    if "skipped" in b:
        print(f"Projects : skipped — {b['skipped']}")
    elif b:
        print(f"Projects : briefs updated={b['updated']} unchanged={b['unchanged']} llm_calls={b['llm_calls']} failed={b['failed']}")
    for section in ("sync", "summaries", "notion", "projects"):
        for err in report.get(section, {}).get("errors", []) if isinstance(report.get(section), dict) else []:
            print(f"  ! {section}: {err}")
    if report.get("error"):
        print(f"  ! {report['error']}")


async def _cmd_login(args: argparse.Namespace) -> int:
    from .telegram_source import TelethonSource

    settings = load_settings()
    name = await TelethonSource(settings).login_interactive(cloud=args.cloud)
    if args.cloud:
        print(
            f"Logged in to Telegram as {name} — a SEPARATE session for the cloud (a new device in Telegram → Devices).\n"
            f"Saved to {settings.cloud_session_path} (chmod 600, not printed).\n"
            f"Next: pbcopy < {settings.cloud_session_path}  → paste into the Railway sealed variable TG_SESSION →  "
            f"rm {settings.cloud_session_path}"
        )
    else:
        print(f"Logged in to Telegram as {name}. Session saved to {settings.session_path} (keep it private).")
    return 0


async def _cmd_snapshot(args: argparse.Namespace) -> int:
    """A consistent copy of the local database to move the state to a hosted worker."""
    from pathlib import Path

    from .db import Database

    settings = load_settings()
    target = Path(args.path)
    if target.exists():
        print(f"{target} already exists — choose another path or delete it first.")
        return 1
    db = Database(settings.db_path)
    db.snapshot(str(target))
    os.chmod(target, 0o600)
    tracked = len(db.tracked_chats())
    db.close()
    print(f"Snapshot written to {target} ({target.stat().st_size} bytes, chmod 600): {tracked} tracked chat(s).")
    print("It contains chat messages — upload it only to your own volume and delete it afterwards.")
    return 0


async def _cmd_check(args: argparse.Namespace) -> int:
    settings = load_settings()
    app = build_app(settings)
    ok = True
    try:
        try:
            client = await app.source.ensure_connected()
            me = await client.get_me()
            print(f"[ok] Telegram: logged in as {me.first_name or ''} {me.last_name or ''}".rstrip())
        except Exception as exc:
            ok = False
            print(f"[!!] Telegram: {exc}")
        if app.llm is None:
            ok = False
            print("[!!] Gemini: GEMINI_API_KEY is not set")
        else:
            try:
                models = await app.llm.list_models()  # type: ignore[attr-defined]
                wanted = f"models/{settings.gemini_model}"
                if wanted in models:
                    print(f"[ok] Gemini: key works, model {settings.gemini_model} is available")
                else:
                    ok = False
                    flash = [m.removeprefix("models/") for m in models if "flash" in m][:8]
                    print(f"[!!] Gemini: key works, but model {settings.gemini_model} not found. Flash models: {flash}")
            except Exception as exc:
                ok = False
                print(f"[!!] Gemini: {exc}")
        if app.sink is None:
            ok = False
            print("[!!] Notion: NOTION_TOKEN / NOTION_PARENT_PAGE_ID are not set")
        else:
            try:
                url = await app.sink.check_access()
                print(f"[ok] Notion: parent page is accessible ({url})")
            except Exception as exc:
                ok = False
                print(f"[!!] Notion: {exc} — is the page shared with the integration (••• → Connections)?")
    finally:
        await app.close()
    return 0 if ok else 1


async def _cmd_chats(args: argparse.Namespace) -> int:
    app = build_app(load_settings())
    try:
        chats = await app.source.list_chats(limit=300 if args.query else args.limit)
        tracked = app.db.tracked_ids()
        if args.query:
            chats = [c for c in chats if args.query.lower() in c.title.lower()]
        for c in chats[: args.limit]:
            mark = "*" if c.chat_id in tracked else " "
            print(f"{mark} {c.chat_id:>16}  {c.kind:<7}  {c.title}")
        print("\n* = tracked")
    finally:
        await app.close()
    return 0


async def _cmd_track(args: argparse.Namespace, tracked: bool) -> int:
    app = build_app(load_settings())
    try:
        for chat_id in args.chat_ids:
            if tracked:
                result = await track_chat(app, chat_id, include_history=not args.from_now)
                print(f"Tracking {chat_id}: {result['title']} ({result['kind']}), history {result['history']}")
            else:
                app.db.set_tracked(chat_id, False)
                print(f"Stopped tracking {chat_id}")
    finally:
        await app.close()
    return 0


async def _cmd_sync(args: argparse.Namespace) -> int:
    app = build_app(load_settings())
    try:
        report = await app.pipeline.run("cli", summarize=not args.no_summarize, export=not args.no_export)
    finally:
        await app.close()
    _print_report(report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") in {"ok", "skipped"} else 1


async def _cmd_status(args: argparse.Namespace) -> int:
    app = build_app(load_settings())
    try:
        print(json.dumps(status_report(app), ensure_ascii=False, indent=2))
    finally:
        await app.close()
    return 0


async def _cmd_summaries(args: argparse.Namespace) -> int:
    app = build_app(load_settings())
    try:
        for r in app.db.list_summaries(args.chat, args.date_from, args.date_to, args.limit):
            s = json.loads(r["summary_json"])
            state = "final" if r["is_final"] else "in progress"
            print(f"\n## {r['title']} · {r['local_date']} ({r['status']}, {state})")
            if s.get("tldr"):
                print(s["tldr"])
            for a in s.get("action_items", []):
                owner = f"{a['owner']}: " if a.get("owner") else ""
                due = f" — due: {a['due']}" if a.get("due") else ""
                print(f"  [ ] {owner}{a['task']}{due}")
            for label in ("decisions", "facts", "open_questions", "topics", "links"):
                for item in s.get(label, []):
                    print(f"  - {label}: {item}")
            if s.get("not_transcribed"):
                print(f"  ! voice/video messages not transcribed: {s['not_transcribed']}")
    finally:
        await app.close()
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tg-connector", description="Telegram → daily summaries MCP connector")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("login", help="log in to Telegram (interactive)")
    p.add_argument("--cloud", action="store_true", help="separate session for a hosted worker (Railway)")
    sub.add_parser("check", help="verify Telegram / Gemini / Notion configuration")
    p = sub.add_parser("chats", help="list chats with ids")
    p.add_argument("-q", "--query")
    p.add_argument("-n", "--limit", type=int, default=40)
    p = sub.add_parser("track", help="start syncing chats")
    p.add_argument("chat_ids", type=int, nargs="+")
    p.add_argument("--from-now", action="store_true",
                   help="skip existing history: only messages written after this command are synced")
    p = sub.add_parser("untrack", help="stop syncing chats")
    p.add_argument("chat_ids", type=int, nargs="+")
    p = sub.add_parser("sync", help="run sync → summarize → Notion once")
    p.add_argument("--no-summarize", action="store_true")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--json", action="store_true", help="also print the full JSON report")
    sub.add_parser("status", help="show sync state")
    p = sub.add_parser("summaries", help="print stored summaries")
    p.add_argument("--chat", type=int)
    p.add_argument("--from", dest="date_from")
    p.add_argument("--to", dest="date_to")
    p.add_argument("-n", "--limit", type=int, default=20)
    sub.add_parser("serve", help="run the MCP server over stdio")
    p = sub.add_parser("snapshot", help="consistent copy of the local database (to move the state to Railway)")
    p.add_argument("path")
    p = sub.add_parser("bot", help="run the team bot (/status, /tasks) for tracked group chats")
    p.add_argument("--sync", action="store_true",
                   help="also run the periodic sync in this process (for 24/7 hosting)")
    args = parser.parse_args(argv)

    settings = load_settings()
    setup_logging(settings, logging.DEBUG if args.verbose else logging.INFO)

    if args.command == "serve":
        from .server import create_server

        create_server(settings).run("stdio")
        return

    if args.command == "bot":
        from .bot import run_bot

        try:
            asyncio.run(run_bot(settings, with_sync=args.sync))
        except KeyboardInterrupt:
            pass
        return

    handlers = {
        "login": _cmd_login,
        "check": _cmd_check,
        "chats": _cmd_chats,
        "track": lambda a: _cmd_track(a, True),
        "untrack": lambda a: _cmd_track(a, False),
        "sync": _cmd_sync,
        "status": _cmd_status,
        "summaries": _cmd_summaries,
        "snapshot": _cmd_snapshot,
    }
    sys.exit(asyncio.run(handlers[args.command](args)))
