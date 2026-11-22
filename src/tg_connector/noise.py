"""Cheap, deterministic noise filter applied BEFORE the LLM.

The filter is intentionally conservative: it is better to let a little noise reach the
LLM than to drop a short "ok" / "+" that was actually agreement to a decision.
Noise messages are still stored (so the sync stays complete) — they are only excluded
from the summary input.
"""

from __future__ import annotations

import hashlib
import re
from zoneinfo import ZoneInfo

from .models import PreparedMessage, RawMessage

# Kinds that never carry summarizable text (voice/video notes are not transcribed — see README limitations).
ALWAYS_NOISE_KINDS = {"service", "sticker", "gif", "voice", "video_note"}
# Media that is noise unless it has a caption.
CAPTION_ONLY_KINDS = {"photo", "video", "other"}
# Agreement markers that can be meaningful even without letters.
AGREEMENT_MARKS = {"+", "++", "+++", "-", "--"}
# Pure gratitude / laughter: no work signal whatever the context.
THANKS_AND_LAUGHS = {
    "спс", "спасибо", "спасибки", "пасиб", "благодарю", "дякую", "дяк", "дякс", "спасибі",
    "thx", "thanks", "thank you", "ty", "tnx",
    "лол", "lol", "кек", "хах", "хаха", "хахаха", "ахах", "ахаха", "ахахах", "haha", "hahaha", "rofl", "lmao",
}
_TRAILING = re.compile(r"[\s.!?,…)(:;]+$")
# "/status", "/tasks@some_bot": commands to bots, not conversation.
_BOT_COMMAND = re.compile(r"^/[A-Za-z0-9_]+(@\w+)?(\s|$)")
URL_RE = re.compile(r"https?://[^\s<>()\"'«»]+")


def _norm(text: str) -> str:
    return _TRAILING.sub("", text.strip().lower())


def is_noise(m: RawMessage) -> bool:
    if m.kind in ALWAYS_NOISE_KINDS:
        return True
    text = m.text.strip()
    if not text:
        return True  # media without caption (CAPTION_ONLY_KINDS), empty message
    if _BOT_COMMAND.match(text):
        return True
    if text in AGREEMENT_MARKS:
        return False
    if not any(ch.isalnum() for ch in text):
        return True  # emoji-only, "!!!", "..."
    if _norm(text) in THANKS_AND_LAUGHS:
        return True
    return False


def content_hash(m: RawMessage) -> str:
    payload = "\x1f".join([m.kind, m.sender_name, m.text, str(m.reply_to_id or "")])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare(m: RawMessage, tz: ZoneInfo, ignore_senders: frozenset[int] = frozenset()) -> PreparedMessage:
    """ignore_senders: our own team bot — its /status replies retell earlier summaries, and summarizing
    them again would echo old decisions into new days (a loop). Other bots (CI, reminders) stay."""
    return PreparedMessage(
        raw=m,
        local_date=m.sent_at.astimezone(tz).date().isoformat(),
        is_noise=is_noise(m) or (m.sender_id is not None and m.sender_id in ignore_senders),
        content_hash=content_hash(m),
    )


def extract_links(texts: list[str], limit: int = 10) -> list[str]:
    """Links are extracted deterministically (never by the LLM), so they cannot be hallucinated."""
    seen: list[str] = []
    for text in texts:
        for url in URL_RE.findall(text):
            url = url.rstrip(".,;:!?")
            if url not in seen:
                seen.append(url)
            if len(seen) >= limit:
                return seen
    return seen
