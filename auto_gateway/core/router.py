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
from .exceptions import classify_exception, AllProvidersExhaustedError

@dataclass(frozen=True)
class RouteRequest:
    strategy: BaseStrategy
    provider: str | None
    alias: str | None
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
            alias=req.alias,
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
                
                # Extract HTTP body safely.
                # NOTE: accessing httpx.Response.text/content on streaming responses
                # can raise httpx.ResponseNotRead. Wrap defensively.
                error_body: str | None = None
                resp = getattr(e, "response", None)
                if resp is not None:
                    try:
                        # Best-effort; may raise ResponseNotRead for streaming responses.
                        error_body = getattr(resp, "text", None)
                    except Exception:
                        error_body = None

                if error_body:
                    error_msg = f"{e} - Response Body: {error_body}"


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
            raise AllProvidersExhaustedError(
                message="All providers exhausted",
                error_type="rate_limit_error",
                code="rate_limit_exceeded",
            )
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

        any_chunk_emitted = False
        last_error_type: str | None = None
        last_error_msg: str | None = None

        for pname, model, key, features in req.strategy.generate_targets(
            req.provider,
            req.models,
            req.shuffle,
            alias=req.alias,
            message_hash=req.context_id,
            is_new_session=True,
        ):
            prov = self.providers.get(pname)
            if not prov:
                continue

            filtered_messages = self._filter_messages(req.messages, features)

            provider_emitted = False

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

                    # ---- handle provider-reported errors (e.g. from repaired SSE) ----
                    if ev_type == "error":
                        error_info = {
                            "message": ev.get("message", "Unknown provider error"),
                            "type": ev.get("error_type", "provider_error"),
                            "code": ev.get("code"),
                        }
                        logger.warning(
                            "Provider '%s' reported a stream error: %s",
                            pname,
                            error_info,
                        )

                        if provider_emitted:
                            # Content has already been sent to the client;
                            # we cannot failover cleanly. Emit a structured
                            # error chunk and terminate with [DONE].
                            err_payload = {
                                "error": error_info,
                            }
                            yield f"data: {json.dumps(err_payload, separators=(',', ':'))}\n\n".encode("utf-8")
                            yield b"data: [DONE]\n\n"
                            return
                        else:
                            # No content sent yet — treat as a provider
                            # failure and try the next candidate.
                            last_error_type = error_info.get("type", "provider_error")
                            last_error_msg = error_info.get("message", "Unknown error")
                            req.strategy.record_failure(
                                key, model, pname,
                                last_error_type,
                                message_hash=req.context_id,
                            )
                            # Break out of this provider's inner loop to
                            # trigger failover to the next provider.
                            break

                    elif ev_type == "content":
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
                        provider_emitted = True
                        any_chunk_emitted = True


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
                            provider_emitted = True
                            any_chunk_emitted = True

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
                        provider_emitted = True
                        any_chunk_emitted = True


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


                # If we broke out of the inner loop via `break` (error with no
                # emitted content), check whether `provider_emitted` is still
                # False. If so, try the next provider.
                if not provider_emitted:
                    continue

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
                
                # Extract HTTP body safely.
                # NOTE: accessing httpx.Response.text/content on streaming responses
                # can raise httpx.ResponseNotRead. Wrap defensively.
                error_body: str | None = None
                resp = getattr(e, "response", None)
                if resp is not None:
                    try:
                        # Best-effort; may raise ResponseNotRead for streaming responses.
                        error_body = getattr(resp, "text", None)
                    except Exception:
                        error_body = None

                if error_body:
                    error_msg = f"{e} - Response Body: {error_body}"

                last_error_type = error_type.value
                last_error_msg = error_msg


                logger.warning(f"Provider '{pname}' stream failed with {error_type.value}: {error_msg}")
                req.strategy.record_failure(key, model, pname, error_type.value, message_hash=req.context_id)
                # try next provider
                continue

        _ = time.perf_counter() - start

        # No provider succeeded: raise AllProvidersExhaustedError so the server
        # can return a proper OpenAI-compatible HTTP 429 error response
        raise AllProvidersExhaustedError(
            message=last_error_msg or "All providers exhausted",
            error_type="rate_limit_error",
            code="rate_limit_exceeded",
        )


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
