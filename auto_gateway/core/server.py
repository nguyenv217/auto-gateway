from __future__ import annotations

import time
import uuid
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)
from .router import ProviderRouter, RouteRequest

security = HTTPBearer(auto_error=False)

def verify_api_key(api_key: str | None):
    """Dependency that verifies the Authorization header if an api_key is configured."""
    async def _verify(credentials: HTTPAuthorizationCredentials = Depends(security)):
        if api_key:
            if not credentials or credentials.credentials != api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or missing API key",
                    headers={"WWW-Authenticate": "Bearer"},
                )
    return _verify

def create_app(*, router: ProviderRouter, strategy, model_name_default: str = "gateway", api_key: str | None = None, timeout: float = 60.0) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state = {"router": router, "strategy": strategy, "model": model_name_default, "timeout": timeout}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": state["model"], "object": "model", "created": int(time.time()), "owned_by": "auto-gateway"}]
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key(api_key))])
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
            # If the requested model isn't supported by any provider, we still
            # want failover to try available providers.
            models=[payload.model] if payload.model else None,
            timeout=state["timeout"],

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
