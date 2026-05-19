from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator, Optional


import asyncio

from ..providers.base import BaseProvider, ProviderCallResult
from ..strategies.base import BaseStrategy
from .router_tool_calls_helpers import chunk_bytes_tool_calls



@dataclass(frozen=True)
class RouteRequest:
    strategy: BaseStrategy
    provider: str | None
    models: list[str] | None
    timeout: float
    shuffle: bool
    tools: list[dict[str, Any]] | None
    tool_choice: Any
    extra_body: dict[str, Any]
    messages: list[dict[str, Any]]
    context_id: str | None = None


class ProviderRouter:
    """Async provider fallback router."""

    def __init__(self, providers: dict[str, BaseProvider]):
        self.providers = providers

    async def route(self, req: RouteRequest) -> ProviderCallResult:
        last: ProviderCallResult | None = None

        for pname, model, key, features in req.strategy.generate_targets(
            req.provider,
            req.models,
            req.shuffle,
            message_hash=req.context_id,
            is_new_session=True,
        ):
            prov = self.providers.get(pname)
            if not prov:
                continue

            filtered_messages = self._filter_messages(req.messages, features)

            try:
                t0 = time.perf_counter()
                res = await prov.call(
                    key=key,
                    model=model,
                    messages=filtered_messages,
                    timeout=req.timeout,
                    tools=req.tools,
                    tool_choice=req.tool_choice,
                    extra_body=req.extra_body,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                req.strategy.record_success(key, req.models, pname)
                req.strategy.record_latency(key, pname, model, latency_ms)
                return res
            except Exception as e:
                req.strategy.record_failure(key, req.models, pname, str(e), message_hash=req.context_id)
                last = {
                    "text": None,
                    "reasoning": None,
                    "tool_calls": None,
                    "usage": None,
                }

        if last is None:
            raise RuntimeError("No providers available")
        return last

    async def route_stream(
        self,
        req: RouteRequest,
        *,
        chatcmpl_id: str,
    ) -> AsyncIterator[bytes]:
        """Stream OpenAI-compatible SSE bytes.

        Router failover behavior: tries provider candidates sequentially.
        If a provider fails mid-stream, the exception is caught and the next
        provider is attempted.
        """

        start = time.perf_counter()

        for pname, model, key, features in req.strategy.generate_targets(
            req.provider,
            req.models,
            req.shuffle,
            message_hash=req.context_id,
            is_new_session=True,
        ):
            prov = self.providers.get(pname)
            if not prov:
                continue

            filtered_messages = self._filter_messages(req.messages, features)

            try:
                t0 = time.perf_counter()

                # initial empty delta (matches OpenAI-ish behavior)
                yield self._chunk_bytes(
                    chatcmpl_id=chatcmpl_id,
                    created=int(time.time()),
                    model=req.models[0] if req.models else model,
                    content_delta="",
                    finish_reason=None,
                )

                role_emitted = False


                stream = prov.call_stream(
                    key=key,
                    model=model,
                    messages=filtered_messages,
                    timeout=req.timeout,
                    tools=req.tools,
                    tool_choice=req.tool_choice,
                    extra_body=req.extra_body,
                )

                # Some providers may accidentally implement call_stream as a coroutine.
                if hasattr(stream, "__await__") and not hasattr(stream, "__aiter__"):
                    stream = await stream  # type: ignore[assignment]

                if not hasattr(stream, "__aiter__"):
                    raise TypeError("Provider call_stream() must return an async iterator")

                async for ev in stream:
                    if not ev:
                        continue

                    ev_type = ev.get("type")
                    if ev_type == "content":
                        content_delta = ev.get("content") or ""
                        if not role_emitted:
                            yield self._chunk_bytes(
                                chatcmpl_id=chatcmpl_id,
                                created=int(time.time()),
                                model=req.models[0] if req.models else model,
                                content_delta="",
                                finish_reason=None,
                                role_delta=True,
                            )
                            role_emitted = True

                        if content_delta:
                            yield self._chunk_bytes(
                                chatcmpl_id=chatcmpl_id,
                                created=int(time.time()),
                                model=req.models[0] if req.models else model,
                                content_delta=content_delta,
                                finish_reason=None,
                                role_delta=False,
                            )

                    elif ev_type == "tool_calls":
                        yield chunk_bytes_tool_calls(
                            chatcmpl_id=chatcmpl_id,
                            created=int(time.time()),
                            model=req.models[0] if req.models else model,
                            tool_calls_delta=[
                                {
                                    "index": ev.get("index", 0),
                                    "id": ev.get("id"),
                                    "function": ev.get("function") or {},
                                }
                            ],
                            finish_reason=None,
                        )

                    elif ev_type == "finish":
                        finish_reason = ev.get("finish_reason") or "stop"
                        yield self._chunk_bytes(
                            chatcmpl_id=chatcmpl_id,
                            created=int(time.time()),
                            model=req.models[0] if req.models else model,
                            content_delta="",
                            finish_reason=finish_reason,
                            role_delta=False,
                        )
                        yield b"data: [DONE]\n\n"
                        return


                # final stop chunk
                yield self._chunk_bytes(
                    chatcmpl_id=chatcmpl_id,
                    created=int(time.time()),
                    model=req.models[0] if req.models else model,
                    content_delta="",
                    finish_reason="stop",
                )
                yield b"data: [DONE]\n\n"

                latency_ms = (time.perf_counter() - t0) * 1000
                req.strategy.record_success(key, req.models, pname)
                req.strategy.record_latency(key, pname, model, latency_ms)
                return

            except Exception as e:
                req.strategy.record_failure(key, req.models, pname, str(e), message_hash=req.context_id)
                # try next provider
                continue

        _ = time.perf_counter() - start
        raise RuntimeError("All providers exhausted in stream mode")

    def _filter_messages(self, messages: list[dict[str, Any]], features: list[str]) -> list[dict[str, Any]]:
        supports_img = any(f.lower() in {"vision", "img2img"} for f in features)
        supports_media = any(f.lower() in {"media"} for f in features)
        supports_video = any(f.lower() in {"video_vision"} for f in features)

        # If supports everything, avoid copying
        if supports_img and supports_media and supports_video:
            return messages

        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                new_content = [
                    item
                    for item in msg["content"]
                    if not (
                        ("image" in str(item.get("type", "")) and not supports_img)
                        or ("media" in str(item.get("type", "")) and not supports_media)
                        or ("video" in str(item.get("type", "")) and not supports_video)
                    )
                ]
                out.append({**msg, "content": new_content})
            else:
                out.append(msg)
        return out

    def _chunk_bytes(
        self,
        *,
        chatcmpl_id: str,
        created: int,
        model: str,
        content_delta: str,
        finish_reason: str | None,
        role_delta: bool = False,
    ) -> bytes:

        payload = {
            "id": chatcmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant" if role_delta else None,
                        "content": content_delta,
                    },

                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")
