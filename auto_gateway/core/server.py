from __future__ import annotations

import time
import uuid
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)
from .router import ProviderRouter, RouteRequest
from .exceptions import AllProvidersExhaustedError

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

def create_app(*, router: ProviderRouter, strategy, model_name_default: str = "gateway", api_key: str | None = None, timeout: float = 60.0, all_models: dict[str, dict[str, list[str]]] | None = None) -> FastAPI:

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state = {"router": router, "strategy": strategy, "model": model_name_default, "timeout": timeout, "all_models": all_models or {}}


    @app.get("/health")
    async def health():
        return {"status": "ok", "providers": len(state["all_models"])}

    @app.get("/v1/models", dependencies=[Depends(verify_api_key(api_key))])
    async def list_models():
        models_data = []
        seen_models: set[str] = set()
        for provider_name, provider_models in state["all_models"].items():
            for model_id in provider_models:
                if model_id not in seen_models:
                    seen_models.add(model_id)
                    models_data.append({
                        "id": model_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": provider_name,
                    })
        return {
            "object": "list",
            "data": models_data,
        }


    @app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key(api_key))])
    async def chat_completions(payload: ChatCompletionRequest, request: Request):
        del request

        # Extract extra fields like max_tokens, top_p, etc. dynamically
        payload_dict = payload.model_dump(exclude_unset=True)
        extra_body = {
            k: v for k, v in payload_dict.items()
            if k not in {"model", "messages", "tools", "tool_choice", "stream", "provider", "alias", "strict_alias"}
        }

        raw_model = payload.model.strip() if payload.model else None
        requested_models: list[str] | None = None

        if raw_model.lower() in ("", "none", "any", "auto"):
            requested_models = None
        elif raw_model.startswith("[") and raw_model.endswith("]"):
            # Parse "[modelA, modelB]"
            requested_models = [m.strip().strip("'\"") for m in raw_model[1:-1].split(",") if m.strip()]
        elif "," in raw_model:
            # Parse "modelA, modelB"
            requested_models = [m.strip() for m in raw_model.split(",") if m.strip()]
        else:
            requested_models = [raw_model]

        route_req = RouteRequest(
            strategy=state["strategy"],
            provider=payload.provider,
            alias=payload.alias,
            strict_alias=payload.strict_alias,
            # If the requested model isn't supported by any provider, we still
            # want failover to try available providers.
            models=requested_models,
            timeout=state["timeout"],

            shuffle=False,
            tools=payload.tools,
            tool_choice=payload.tool_choice,
            extra_body=extra_body,
            messages=[m.model_dump(exclude_none=True) for m in payload.messages],
            context_id=None,
        )


        if not payload.stream:
            try:
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
            except AllProvidersExhaustedError as e:
                # Return proper OpenAI-compatible error response with HTTP 429
                # This is compatible with openai.RateLimitError exception
                return JSONResponse(
                    status_code=429,
                    content=e.to_openai_error_response(),
                )

        async def gen() -> AsyncIterator[bytes]:
            # OpenAI-compatible SSE stream (router yields already-packed SSE bytes).
            chatcmpl_id = f"chatcmpl_{uuid.uuid4().hex}"
            
            try:
                async for chunk_bytes in state["router"].route_stream(route_req, chatcmpl_id=chatcmpl_id):
                    yield chunk_bytes
            except AllProvidersExhaustedError as e:
                import json
                # For streaming, emit error as a structured error chunk before [DONE]
                # This provides visibility into the failure while maintaining SSE format
                err_payload = {
                    "error": {
                        "message": e.message,
                        "type": e.error_type,
                        "param": e.param,
                        "code": e.code,
                    }
                }
                yield f"data: {json.dumps(err_payload)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
            except Exception as e:
                # Catch-all for any other unexpected errors - ensure they also 
                # fall through as proper SSE error format
                error_response = {
                    "error": {
                        "message": str(e),
                        "type": "internal_error",
                        "param": None,
                        "code": "internal_error",
                    }
                }
                yield f"data: {json.dumps(error_response)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
