"""Deterministic masking of secrets BEFORE chat text leaves the machine (LLM prompt → summary → Notion).

The local database keeps the original text: it is the source of truth, local and chmod 600.
Only what is sent out is masked. Patterns are deliberately narrow, so normal work content
(prices, phone numbers, dates, links) is not hidden.
"""

from __future__ import annotations

import re

HIDDEN = "[hidden]"

# "пароль від адмінки: Qw3rty!" / "password = x" → the rest of the line is hidden.
_PASSWORD = re.compile(
    r"(?im)\b(парол\w*|password|passwd|pwd|pass|passcode)\b([^\n:=]{0,40}?)\s*[:=]\s*[^\n]+"
)
# One-time / PIN codes: "код 482913", "OTP: 1234", "код підтвердження 5521".
_OTP = re.compile(
    r"(?i)\b(код|code|otp|pin|пін|пин)\b(\s+(?:підтвердження|подтверждения|verification|доступу|доступа))?"
    r"(\s*[:\-–]?\s*)\d{4,8}\b"
)
# Well-known API key / token formats and JWTs.
_TOKENS = re.compile(
    r"\b(?:sk|pk|rk)[-_](?:live|test|proj)[-_][A-Za-z0-9_\-]{12,}"
    r"|\bgh[pousr]_[A-Za-z0-9]{20,}"
    r"|\bxox[abprs]-[A-Za-z0-9-]{10,}"
    r"|\bAIza[0-9A-Za-z_\-]{30,}"
    r"|\bntn_[A-Za-z0-9]{20,}"
    r"|\bsecret_[A-Za-z0-9]{20,}"
    r"|\bglpat-[A-Za-z0-9_\-]{20,}"
    r"|\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"
    r"|(?i:\bbearer\s+)[A-Za-z0-9._\-]{16,}"
)
# Secrets passed in links: ?token=…&api_key=…
_URL_SECRET = re.compile(
    r"(?i)([?&](?:token|access_token|api_key|apikey|key|password|pass|secret|sig|signature|auth)=)[^&#\s]+"
)
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_CARD = re.compile(r"(?<![\d.,])(?:\d[ \-]?){12,18}\d(?![\d.,]*\d)")
_URL = re.compile(r"https?://\S+")
# Long random-looking strings (keys, hashes) that are not part of a link.
_LONG_TOKEN = re.compile(r"^[A-Za-z0-9_\-+/=]{32,}$")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return "[card number hidden]" if 13 <= len(digits) <= 19 and _luhn_ok(digits) else match.group(0)


def _iban(match: re.Match[str]) -> str:
    return "[IBAN hidden]" if sum(ch.isdigit() for ch in match.group(0)) >= 10 else match.group(0)


def _long_tokens(text: str) -> str:
    def fix(match: re.Match[str]) -> str:
        word = match.group(0)
        core = word.strip(".,;:!?()[]«»\"'")
        if (
            core
            and not _URL.match(core)
            and _LONG_TOKEN.match(core)
            and any(c.isdigit() for c in core)
            and any(c.isalpha() for c in core)
        ):
            return word.replace(core, HIDDEN)
        return word

    return re.sub(r"\S+", fix, text)


def redact(text: str) -> str:
    if not text:
        return text
    text = _PASSWORD.sub(lambda m: f"{m.group(1)}{m.group(2)}: {HIDDEN}", text)
    text = _OTP.sub(lambda m: f"{m.group(1)}{m.group(2) or ''}{m.group(3)}{HIDDEN}", text)
    text = _TOKENS.sub(HIDDEN, text)
    text = _URL_SECRET.sub(lambda m: m.group(1) + HIDDEN, text)
    text = _IBAN.sub(_iban, text)
    text = _CARD.sub(_card, text)
    return _long_tokens(text)
