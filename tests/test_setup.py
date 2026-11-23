"""Setup and hygiene: the Notion page link a user pastes, secrets in logs, long messages, the run lock."""

from __future__ import annotations

import io
import logging

import pytest
from conftest import at

from tg_connector.config import parse_notion_id
from tg_connector.logging_setup import RedactingFormatter
from tg_connector.summarizer import MAX_MESSAGE_CHARS

PAGE = "4207451896e381caa73ec1c2556eef5f"
DASHED = "42074518-96e3-81ca-a73e-c1c2556eef5f"


@pytest.mark.parametrize("pasted", [
    PAGE, DASHED, f"https://www.notion.so/{PAGE}",
    f"https://www.notion.so/My-page-{PAGE}",  # a title ending in a hex letter shifted the id before
    f"https://www.notion.so/Knowledge-base-{PAGE}?pvs=4",
    f"https://www.notion.so/workspace/Plan-2026-{PAGE}#section",
    f"https://app.notion.com/p/{PAGE}/",
    f"https://www.notion.so/{DASHED}",
])
def test_any_form_of_a_notion_page_link_gives_the_right_id(pasted):
    assert parse_notion_id(pasted) == PAGE


def test_text_without_a_page_id_is_rejected():
    assert parse_notion_id("https://www.notion.so/My-page") is None and parse_notion_id("") is None


def test_secret_is_masked_in_tracebacks_too():
    out = io.StringIO()
    handler = logging.StreamHandler(out)
    handler.setFormatter(RedactingFormatter("%(message)s", "%H:%M", ["SECRET-TOKEN-123456"]))
    log = logging.getLogger("test-secrets")
    log.addHandler(handler)
    log.propagate = False
    try:
        raise RuntimeError("auth failed for SECRET-TOKEN-123456")
    except RuntimeError:
        log.exception("call with %s failed", "SECRET-TOKEN-123456")
    text = out.getvalue()
    assert "SECRET-TOKEN-123456" not in text and "Traceback" in text and "auth failed for ***" in text


async def test_long_message_is_cut_with_a_marker_the_model_sees(make_env):
    env = make_env()
    env.track(-1001, "Маркетинг")
    env.source.add(-1001, 1, "План: " + "дуже довгий текст " * 400, at(20, 10))
    await env.run()
    (prompt,) = env.llm.prompts
    assert "…[truncated]" in prompt and MAX_MESSAGE_CHARS >= 4000


async def test_notion_export_keeps_the_run_lock_alive(make_env):
    env = make_env()
    for i in range(3):
        env.track(-1001 - i, f"Чат {i}")
        env.source.add(-1001 - i, 1, "Запускаємо креатив B у пʼятницю", at(20, 10))
    env.now = env.now.replace(hour=2)
    await env.run()  # summaries exist, now export them again with a fresh layout
    env.db.conn.execute("UPDATE summaries SET notion_synced_hash = NULL")
    beats = []
    await env.sink.export_pending(lambda: beats.append(1))
    assert len(beats) >= 3
