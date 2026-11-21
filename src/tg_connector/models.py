"""Plain data objects shared between the Telegram source, storage and pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ChatInfo:
    chat_id: int  # Telegram "marked" peer id (users > 0, groups/channels < 0)
    title: str
    kind: str  # dm | bot | group | channel
    unread: int = 0


@dataclass(frozen=True)
class RawMessage:
    """A message as delivered by the source, before classification."""

    chat_id: int
    message_id: int
    sent_at: datetime  # timezone-aware, UTC
    sender_id: int | None
    sender_name: str
    kind: str  # text | photo | video | document | sticker | gif | voice | video_note | poll | service | other
    text: str
    reply_to_id: int | None = None
    edit_date: datetime | None = None
    migrated_to: int | None = None  # set on the service message "group upgraded to supergroup <id>"


@dataclass(frozen=True)
class PreparedMessage:
    """A classified message ready to be stored."""

    raw: RawMessage
    local_date: str  # YYYY-MM-DD in the configured timezone
    is_noise: bool
    content_hash: str


class SourceRateLimited(Exception):
    """The source asked us to back off (e.g. Telegram FLOOD_WAIT)."""

    def __init__(self, seconds: int):
        super().__init__(f"rate limited by source, retry in {seconds}s")
        self.seconds = seconds
