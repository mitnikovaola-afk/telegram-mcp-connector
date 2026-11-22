"""Settings loaded from environment variables / the project's .env file.

Secrets never get default values and are never logged (see logging_setup.py).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw is not None else default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _thinking_level(raw: str | None) -> str:
    """"" = the model's default. Gemini 3.x also accepts low / medium / high."""
    value = (raw or "").lower()
    return "" if value in {"default", "off", "none", "0"} else value


_NOTION_ID_RE = re.compile(r"([0-9a-f]{32})", re.IGNORECASE)


def parse_notion_id(value: str | None) -> str | None:
    """Accepts a bare page id or a full Notion page URL and returns the 32-hex id."""
    if not value:
        return None
    compact = value.replace("-", "")
    matches = _NOTION_ID_RE.findall(compact)
    return matches[-1].lower() if matches else None


@dataclass(frozen=True)
class Settings:
    # Telegram (my.telegram.org)
    tg_api_id: int | None
    tg_api_hash: str | None
    # LLM (Gemini free tier by default)
    gemini_api_key: str | None
    gemini_model: str
    gemini_thinking_level: str
    # Notion
    notion_token: str | None
    notion_parent_page_id: str | None
    # Behaviour
    data_dir: Path
    timezone: ZoneInfo
    sync_interval_minutes: int
    backfill_days: int
    edit_lookback_hours: int
    summarize_today: bool
    summary_language: str
    read_only: bool
    send_only_tracked: bool
    max_llm_attempts: int
    max_chars_per_llm_call: int
    batch_size: int
    lock_ttl_seconds: int
    # Team bot (optional, beyond the test task): token from @BotFather.
    tg_bot_token: str | None = None
    # Cloud (Railway): the Telegram session comes from a variable instead of a file; the worker waits
    # for the uploaded database instead of starting from zero; raw texts are kept only N days (0 = forever).
    tg_session: str | None = None
    wait_for_state: bool = False
    raw_retention_days: int = 0
    # Project knowledge base (beyond the task): daily project brief after this local hour.
    projects_enabled: bool = True
    project_brief_hour: int = 7
    # per_chat: every tracked group chat becomes its own project; single: all of them go into one project.
    projects_mode: str = "per_chat"
    project_name: str = "Мій проєкт"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "connector.db"

    @property
    def session_path(self) -> Path:
        return self.data_dir / "telegram.session"

    @property
    def cloud_session_path(self) -> Path:
        return self.data_dir / "cloud.session"

    @property
    def bot_session_path(self) -> Path:
        return self.data_dir / "bot.session"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "logs" / "connector.log"

    def secret_values(self) -> list[str]:
        """Values that must never appear in logs or tool output."""
        values = [self.tg_api_hash, self.gemini_api_key, self.notion_token, self.tg_bot_token, self.tg_session]
        for path in (self.session_path, self.bot_session_path, self.cloud_session_path):
            if path.exists():
                try:
                    values.append(path.read_text().strip())
                except OSError:
                    pass
        return [v for v in values if v and len(v) >= 6]


def load_settings(env_file: Path | None = None) -> Settings:
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    api_id = _env("TG_API_ID")
    data_dir = Path(_env("DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser()
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    return Settings(
        tg_api_id=int(api_id) if api_id else None,
        tg_api_hash=_env("TG_API_HASH"),
        gemini_api_key=_env("GEMINI_API_KEY"),
        gemini_model=_env("GEMINI_MODEL", "gemini-3.5-flash") or "gemini-3.5-flash",
        gemini_thinking_level=_thinking_level(_env("GEMINI_THINKING_LEVEL")),
        notion_token=_env("NOTION_TOKEN"),
        notion_parent_page_id=parse_notion_id(_env("NOTION_PARENT_PAGE_ID")),
        data_dir=data_dir,
        timezone=ZoneInfo(_env("TIMEZONE", "Europe/Kyiv") or "Europe/Kyiv"),
        sync_interval_minutes=_env_int("SYNC_INTERVAL_MINUTES", 30),
        backfill_days=_env_int("BACKFILL_DAYS", 7),
        edit_lookback_hours=_env_int("EDIT_LOOKBACK_HOURS", 24),
        summarize_today=_env_bool("SUMMARIZE_TODAY", True),
        summary_language=_env("SUMMARY_LANGUAGE", "auto") or "auto",
        read_only=_env_bool("READ_ONLY", False),
        send_only_tracked=_env_bool("SEND_ONLY_TRACKED", True),
        max_llm_attempts=_env_int("MAX_LLM_ATTEMPTS", 5),
        max_chars_per_llm_call=_env_int("MAX_CHARS_PER_LLM_CALL", 60_000),
        batch_size=_env_int("SYNC_BATCH_SIZE", 200),
        lock_ttl_seconds=_env_int("LOCK_TTL_SECONDS", 900),
        tg_bot_token=_env("TG_BOT_TOKEN"),
        tg_session=_env("TG_SESSION"),
        wait_for_state=_env_bool("WAIT_FOR_STATE", False),
        raw_retention_days=_env_int("RAW_RETENTION_DAYS", 0),
        projects_enabled=_env_bool("PROJECTS_ENABLED", True),
        project_brief_hour=_env_int("PROJECT_BRIEF_HOUR", 7),
        projects_mode=(_env("PROJECTS_MODE", "per_chat") or "per_chat").lower(),
        project_name=_env("PROJECT_NAME", "Мій проєкт") or "Мій проєкт",
    )
