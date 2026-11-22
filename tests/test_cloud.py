"""Hosted (Railway) pieces: session from a variable, separate cloud login, waiting for the uploaded state,
retention of raw texts, consistent snapshot."""

from __future__ import annotations

import asyncio
import os
import stat
from types import SimpleNamespace

from conftest import at, make_settings

from tg_connector import telegram_source
from tg_connector.bot import database_ready, wait_for_state
from tg_connector.db import Database
from tg_connector.models import ChatInfo

GROUP = -1001


class FakeTelethon:
    def __init__(self, session: str, device_model: str) -> None:
        self.session_string = session
        self.device_model = device_model
        self.session = SimpleNamespace(save=lambda: "NEW-CLOUD-SESSION-STRING-0123456789")

    async def connect(self):  # noqa: ANN201
        return None

    async def is_user_authorized(self) -> bool:
        return True

    async def start(self):  # noqa: ANN201
        return self

    async def get_me(self):  # noqa: ANN201
        return SimpleNamespace(first_name="Olha", last_name=None, title=None)

    async def disconnect(self):  # noqa: ANN201
        return None

    def is_connected(self) -> bool:
        return True


def fake_source(settings, created: list):  # noqa: ANN001
    src = telegram_source.TelethonSource(settings)

    def new_client(session: str, device_model: str = "tg-mcp-connector"):  # noqa: ANN202
        client = FakeTelethon(session, device_model)
        created.append(client)
        return client

    src._new_client = new_client
    return src


async def test_hosted_worker_takes_the_session_from_a_variable_not_a_file(tmp_path):
    settings = make_settings(tmp_path, tg_session="SESSION-FROM-SEALED-VARIABLE")
    settings.session_path.write_text("LOCAL-FILE-SESSION")
    created: list = []

    await fake_source(settings, created).ensure_connected()

    assert created[0].session_string == "SESSION-FROM-SEALED-VARIABLE"
    assert "SESSION-FROM-SEALED-VARIABLE" in settings.secret_values()  # masked in logs


async def test_cloud_login_creates_a_separate_session_file_and_keeps_the_local_one(tmp_path):
    settings = make_settings(tmp_path)
    settings.session_path.write_text("LOCAL-FILE-SESSION")
    created: list = []

    await fake_source(settings, created).login_interactive(cloud=True)

    assert created[0].device_model == "tg-mcp-connector-cloud"  # a distinct row in Telegram → Devices
    assert settings.cloud_session_path.read_text() == "NEW-CLOUD-SESSION-STRING-0123456789"
    assert stat.S_IMODE(os.stat(settings.cloud_session_path).st_mode) == 0o600
    assert settings.session_path.read_text() == "LOCAL-FILE-SESSION"
    assert "NEW-CLOUD-SESSION-STRING-0123456789" in settings.secret_values()


def test_database_ready_rejects_missing_empty_and_garbage_files(tmp_path):
    path = tmp_path / "connector.db"
    assert not database_ready(path)
    path.write_bytes(b"")
    assert not database_ready(path)
    path.write_bytes(b"this is not sqlite at all" * 100)
    assert not database_ready(path)
    path.unlink()
    Database(path).close()
    assert database_ready(path)


async def test_worker_waits_for_the_uploaded_database_instead_of_starting_from_zero(tmp_path):
    settings = make_settings(tmp_path, wait_for_state=True)
    waiting = asyncio.create_task(wait_for_state(settings, poll_s=0.01, settle_s=0.01))
    await asyncio.sleep(0.05)
    assert not waiting.done()  # nothing uploaded yet: no fresh state, no second Notion databases
    assert not settings.db_path.exists()

    snapshot_source = Database(tmp_path / "local.db")
    snapshot_source.snapshot(str(settings.db_path))  # "railway volume files upload"
    snapshot_source.close()

    await asyncio.wait_for(waiting, timeout=2)


async def test_old_raw_texts_are_purged_after_their_day_is_final_and_summarized(make_env):
    env = make_env(backfill_days=30, raw_retention_days=14)
    env.track(GROUP, "Маркетинг")
    env.source.add(GROUP, 1, "Старе рішення про бюджет", at(1, 10))
    env.source.add(GROUP, 2, "Свіже обговорення релізу", at(20, 10))

    report = await env.run()

    assert report["retention"] == {"deleted_messages": 1}
    assert [r["message_id"] for r in env.db.messages_for_day(GROUP, "2026-11-20")] == [2]
    assert env.db.get_summary(GROUP, "2026-11-01") is not None  # the summary of the old day stays
    second = await env.run()
    assert second["sync"]["fetched"] == 0 and env.llm.calls == 2  # still incremental, nothing re-summarized


def test_snapshot_is_a_consistent_copy_with_the_state(tmp_path):
    db = Database(tmp_path / "connector.db")
    db.upsert_chat(ChatInfo(GROUP, "Маркетинг", "group"))
    db.set_tracked(GROUP, True)
    db.kv_set("notion_database_id", "db-1")

    db.snapshot(str(tmp_path / "transfer.db"))

    copy = Database(tmp_path / "transfer.db")
    assert copy.tracked_ids() == {GROUP}
    assert copy.kv_get("notion_database_id") == "db-1"
