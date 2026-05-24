from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Optional

import httpx

from .base import BaseProvider, ProviderCallResult, BaseProviderDelta
from ..core.sse_repair import heuristic_json_repair, extract_error_from_payload

logger = logging.getLogger("auto-gateway.openai_compatible")


class OpenAICompatibleProvider(BaseProvider):
    """Async OpenAI-compatible provider.

    Supports both non-stream (`call`) and streaming (`call_stream`) for
    `/v1/chat/completions`.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        keys: list[str] | None,
        model_configs: dict[str, list[str]],
        extra: Optional[dict[str, Any]] = None,
        key_aliases: dict[str, str] | None = None,
    ):
        super().__init__(name=name, keys=keys, models=model_configs, key_aliases=key_aliases)
        self.base_url = base_url.rstrip("/")
        self.extra = extra or {}

    async def call(
        self,
        *,
        key: str | None,
        model: str,
        messages: list[dict[str, Any]],
        timeout: float,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        extra_body: dict[str, Any] | None = None,
    ) -> ProviderCallResult:
        headers: dict[str, str] = {}
        if key:
            headers["Authorization"] = f"Bearer {key}"

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        payload.update(self.extra.get("extra_body", {}))
        if "temperature" in self.extra:
            payload["temperature"] = self.extra["temperature"]
        if extra_body:
            payload.update(extra_body)

        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            resp = await client.post(f"{self.base_url}/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

        # Normalize response to ProviderCallResult
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}

        return {
            "text": msg.get("content") or "",
            "reasoning": msg.get("reasoning") or None,
            "tool_calls": (msg.get("tool_calls") if msg.get("tool_calls") else None),
            "usage": data.get("usage"),
        }

    async def call_stream(
        self,
        *,
        key: str | None,
        model: str,
        messages: list[dict[str, Any]],
        timeout: float,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncIterator[BaseProviderDelta]:
        """Yield OpenAI-compatible structured delta events from upstream SSE."""

        headers: dict[str, str] = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        payload.update(self.extra.get("extra_body", {}))
        if "temperature" in self.extra:
            payload["temperature"] = self.extra["temperature"]
        if extra_body:
            payload.update(extra_body)

        # Enforce proper timeouts to prevent infinite stalling
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=timeout,
            ) as resp:

                resp.raise_for_status()

                async for raw_line in resp.aiter_lines():
                    if not raw_line:
                        continue
                    if not raw_line.startswith("data:"):
                        continue

                    data_str = raw_line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        return

                    # data_str is JSON for the chunk
                    import json as _json

                    try:
                        chunk = _json.loads(data_str)
                    except Exception:
                        # ----- heuristic repair path -----
                        # Third-party servers sometimes emit malformed
                        # JavaScript-style objects (unquoted keys,
                        # e.g. {error: "overloaded"}) instead of proper SSE.
                        # We attempt to repair and extract structured info
                        # rather than silently dropping the chunk.
                        repaired = heuristic_json_repair(data_str)

                        if repaired is not None:
                            error_info = extract_error_from_payload(repaired)

                            if error_info is not None:
                                # Provider emitted an error in a malformed
                                # format — surface it as a structured delta
                                # so the downstream SDK can react properly.
                                logger.warning(
                                    "Repaired malformed SSE error from upstream: "
                                    "%s (raw: %.120s)",
                                    error_info,
                                    data_str,
                                )
                                yield {
                                    "type": "error",
                                    "message": error_info["message"],
                                    "error_type": error_info.get("type", "provider_error"),
                                    "code": error_info.get("code"),
                                }
                                continue

                            # Repaired payload looks like a normal chunk
                            # (has choices, delta, etc.) — process it below.
                            logger.debug(
                                "Repaired malformed SSE chunk as normal data "
                                "(first 120 chars: %.120s)",
                                data_str,
                            )
                            chunk = repaired
                        else:
                            # Unrecoverable — preserve existing behaviour of
                            # silently dropping the undecodable line.
                            logger.debug(
                                "Skipping unparseable SSE data line: %.120s",
                                data_str,
                            )
                            continue

                    # ---- process the chunk (original or repaired) ----
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue

                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")

                    # Must use `is not None` to avoid skipping empty string chunks (`""`)
                    if content is not None:
                        yield {"type": "content", "content": content}

                    tool_calls = delta.get("tool_calls")
                    if tool_calls:
                        # Forward OpenAI tool_calls deltas as-is.
                        for tc in tool_calls:
                            yield {
                                "type": "tool_calls",
                                "index": tc.get("index", 0),
                                "id": tc.get("id"),
                                "function": tc.get("function") or {},
                            }

                    finish_reason = choices[0].get("finish_reason")
                    if finish_reason:
                        yield {"type": "finish", "finish_reason": finish_reason}


class OpenAIProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        keys: list[str] | None,
        model_configs: dict[str, list[str]],
        base_url: str = "https://api.openai.com/v1",
        extra: Optional[dict[str, Any]] = None,
        key_aliases: dict[str, str] | None = None,
    ):
        super().__init__(
            name="openai",
            base_url=base_url,
            keys=keys,
            model_configs=model_configs,
            extra=extra,
            key_aliases=key_aliases,
        )
