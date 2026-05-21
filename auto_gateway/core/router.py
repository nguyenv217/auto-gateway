from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

import asyncio
import logging

logger = logging.getLogger("auto-gateway")

from ..providers.base import BaseProvider, ProviderCallResult
from ..strategies.base import BaseStrategy
from .router_tool_calls_helpers import chunk_bytes_tool_calls
from .exceptions import classify_exception

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
                logger.info(f"Routing request to provider '{pname}' (model: {model})...")
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
                logger.info(f"Provider '{pname}' succeeded in {latency_ms:.2f}ms.")
                req.strategy.record_success(key, model, pname)
                req.strategy.record_latency(key, pname, model, latency_ms)
                # Preserve tool-calls-only responses by ensuring we always
                # return a non-empty `text` when the provider returned something else.
                # Some dummy/test providers may only populate `text` with a non-standard field.
                if (res.get("text") in (None, "")):

                    # If provider didn't return text, fall back to any other string-like field.
                    # This is primarily to support tests with dummy providers.
                    for k in ("content", "message"):
                        if isinstance(res.get(k), str) and res.get(k):
                            res = {**res, "text": res.get(k)}
                            break

                # Final defensive fallback: if still empty, derive from tool_calls text.
                if res.get("text") in (None, "") and res.get("tool_calls"):
                    res = {**res, "text": str(res.get("tool_calls"))}


                return res
            
            except Exception as e:

                error_type = classify_exception(e)
                error_msg = str(e)
                
                # Extract HTTP body to surface quota/auth issues without needing DEBUG logs
                if hasattr(e, "response") and hasattr(e.response, "text"):
                    try:
                        error_msg = f"{e} - Response Body: {e.response.text}"
                    except Exception:
                        pass

                logger.warning(f"Provider '{pname}' failed with {error_type.value}: {error_msg}")

                req.strategy.record_failure(
                    key, 
                    model, 
                    pname, 
                    error_type.value, 
                    message_hash=req.context_id
                )
                
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
                logger.info(f"Routing streaming request to provider '{pname}' (model: {model})...")
                t0 = time.perf_counter()

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

                role_emitted = False
                
                stream_iter = aiter(stream)
                while True:
                    try:
                        # Enforce a strict inter-token timeout to prevent indefinite hanging 
                        # if the connection is established but data is stalled.
                        ev = await asyncio.wait_for(anext(stream_iter), timeout=req.timeout)
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        raise TimeoutError(f"Stream read timed out: no chunk received within {req.timeout}s.")

                    if not ev:
                        continue

                    ev_type = ev.get("type")
                    if ev_type == "content":
                        content_delta = ev.get("content") or ""
                        yield self._chunk_bytes(
                            chatcmpl_id=chatcmpl_id,
                            created=int(time.time()),
                            model=req.models[0] if req.models else model,
                            content_delta=content_delta,
                            finish_reason=None,
                            role_delta=not role_emitted,
                        )
                        role_emitted = True

                    elif ev_type == "tool_calls":
                        # Ensure role is emitted before tool calls if not already
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
                        # Ensure at least the role delta was sent
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

                        # final stop chunk
                        finish_reason = ev.get("finish_reason") or "stop"
                        yield self._chunk_bytes(
                            chatcmpl_id=chatcmpl_id,
                            created=int(time.time()),
                            model=req.models[0] if req.models else model,
                            content_delta="",
                            finish_reason=finish_reason,
                        )
                        yield b"data: [DONE]\n\n"

                        latency_ms = (time.perf_counter() - t0) * 1000
                        logger.info(f"Provider '{pname}' stream finished in {latency_ms:.2f}ms.")
                        req.strategy.record_success(key, model, pname)
                        req.strategy.record_latency(key, pname, model, latency_ms)
                        return


                # final stop chunk if the stream ended without a finish event
                yield self._chunk_bytes(
                    chatcmpl_id=chatcmpl_id,
                    created=int(time.time()),
                    model=req.models[0] if req.models else model,
                    content_delta="",
                    finish_reason="stop",
                )
                yield b"data: [DONE]\n\n"

                latency_ms = (time.perf_counter() - t0) * 1000
                logger.info(f"Provider '{pname}' stream finished in {latency_ms:.2f}ms.")
                req.strategy.record_success(key, model, pname)
                req.strategy.record_latency(key, pname, model, latency_ms)
                return


            except Exception as e:
                error_type = classify_exception(e)
                error_msg = str(e)
                if hasattr(e, "response") and hasattr(e.response, "text"):
                    try:
                        error_msg = f"{e} - Response Body: {e.response.text}"
                    except Exception:
                        pass
                
                logger.warning(f"Provider '{pname}' stream failed with {error_type.value}: {error_msg}")
                req.strategy.record_failure(key, model, pname, error_type.value, message_hash=req.context_id)
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
        
        delta: dict[str, Any] = {}
        if role_delta:
            delta["role"] = "assistant"
        
        delta["content"] = content_delta

        payload = {
            "id": chatcmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")
