from __future__ import annotations

import time
import uuid
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state = {"router": router, "strategy": strategy, "model": model_name_default}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": state["model"], "object": "model", "created": int(time.time()), "owned_by": "auto-gateway"}]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest, request: Request):

        del request

        # Extract extra fields like max_tokens, top_p, etc. dynamically
        payload_dict = payload.model_dump(exclude_unset=True)
        extra_body = {
            k: v for k, v in payload_dict.items() 
            if k not in {"model", "messages", "tools", "tool_choice", "stream"}
        }

        route_req = RouteRequest(
            strategy=state["strategy"],
            provider=None,
            models=None,
            timeout=15.0,
            shuffle=False,
            tools=payload.tools,
            tool_choice=payload.tool_choice,
            extra_body=extra_body,
            messages=[m.model_dump(exclude_none=True) for m in payload.messages],
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
