"""LLM provider layer. The pipeline only depends on the `LLMClient` protocol, so switching
Gemini → Claude / a local model means adding one class here and one setting.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Protocol

import httpx
from pydantic import BaseModel

RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504}
# Free-tier latency varies a lot (5–80s per summary in the demo); a hung call must not block the run forever.
LLM_TIMEOUT_S = 180


class LLMError(Exception):
    def __init__(self, message: str, *, retryable: bool, retry_after: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after
        self.calls = 0  # API calls spent before giving up (set by the summarizer)


class LLMClient(Protocol):
    model_name: str

    async def generate_json(self, system: str, user: str, schema: type[BaseModel]) -> str:
        """Return the raw JSON text produced by the model for `schema`."""
        ...


def _retry_after(details: Any) -> float | None:
    """Gemini puts RetryInfo.retryDelay (e.g. "17s") into the error details."""
    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(details))
    return float(match.group(1)) if match else None


class GeminiClient:
    def __init__(self, api_key: str, model: str, thinking_level: str = ""):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model_name = model
        # Gemini 3.x "thinks" before answering. We keep the model's default: with "low" the demo summaries
        # started slipping single Russian words into Ukrainian text, and the latency gain was small because
        # free-tier latency is dominated by queueing on Google's side. Configurable via GEMINI_THINKING_LEVEL.
        self.thinking_level = thinking_level if model.startswith("gemini-3") else ""

    async def generate_json(self, system: str, user: str, schema: type[BaseModel]) -> str:
        from google.genai import errors

        try:
            response = await asyncio.wait_for(self._generate(system, user, schema), timeout=LLM_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            raise LLMError(f"Gemini did not answer within {LLM_TIMEOUT_S}s", retryable=True) from exc
        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            raise LLMError(
                f"Gemini API error {code}: {getattr(exc, 'message', None) or exc}",
                retryable=code in RETRYABLE_HTTP,
                retry_after=_retry_after(getattr(exc, "details", None)),
            ) from exc
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise LLMError(f"Gemini network error: {exc}", retryable=True) from exc

        text = response.text
        if not text:
            reason = None
            if response.candidates:
                reason = response.candidates[0].finish_reason
            raise LLMError(f"Gemini returned no text (finish_reason={reason})", retryable=False)
        return text

    async def _generate(self, system: str, user: str, schema: type[BaseModel]):  # noqa: ANN202
        from google.genai import types

        return await self._client.aio.models.generate_content(
            model=self.model_name,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=schema,
                max_output_tokens=8192,
                thinking_config=(
                    types.ThinkingConfig(thinking_level=self.thinking_level) if self.thinking_level else None
                ),
            ),
        )

    async def list_models(self) -> list[str]:
        names: list[str] = []
        async for model in await self._client.aio.models.list():
            names.append(model.name)
        return names


def build_llm(api_key: str | None, model: str, thinking_level: str = "") -> LLMClient | None:
    if not api_key:
        return None
    return GeminiClient(api_key, model, thinking_level)
