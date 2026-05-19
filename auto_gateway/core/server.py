from __future__ import annotations

import time
import uuid
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatMessage,
    sse_pack,
)
from .router import ProviderRouter, RouteRequest


def create_app(*, router: ProviderRouter, strategy, model_name_default: str = "gateway") -> FastAPI:
    app = FastAPI()
    state = {"router": router, "strategy": strategy, "model": model_name_default}

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest, request: Request):
        del request

        route_req = RouteRequest(
            strategy=state["strategy"],
            provider=None,
            models=None,
            timeout=15.0,
            shuffle=False,
            tools=payload.tools,
            tool_choice=payload.tool_choice,
            extra_body=payload.extra_body,
            messages=[m.model_dump() for m in payload.messages],
            context_id=None,
        )

        if not payload.stream:
            res = await state["router"].route(route_req)
            now = int(time.time())

            msg = ChatMessage(
                role="assistant",
                content=res.get("text") or "",
                tool_calls=res.get("tool_calls"),
            )
            out = ChatCompletionResponse(
                id=f"chatcmpl_{uuid.uuid4().hex}",
                created=now,
                model=payload.model,
                choices=[
                    {
                        "index": 0,
                        "message": msg,
                        "finish_reason": "stop",
                    }
                ],
                usage=res.get("usage"),
            )
            return out.model_dump()

        async def gen() -> AsyncIterator[bytes]:
            # OpenAI-compatible SSE stream (router yields already-packed SSE bytes).
            chatcmpl_id = f"chatcmpl_{uuid.uuid4().hex}"
            async for chunk_bytes in state["router"].route_stream(route_req, chatcmpl_id=chatcmpl_id):
                yield chunk_bytes

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
