import json

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from auto_gateway.core.server import create_app
from auto_gateway.core.router import ProviderRouter, RouteRequest
from auto_gateway.providers.base import BaseProvider, ProviderCallResult
from auto_gateway.strategies.sequential import SequentialStrategy


class ToolingStreamingProvider(BaseProvider):
    def __init__(self):
        super().__init__(name="tool_ok", keys=[None], models={"m": []})

    async def call(
        self,
        *,
        key,
        model,
        messages,
        timeout,
        tools,
        tool_choice,
        extra_body=None,
    ) -> ProviderCallResult:
        return {"text": "", "reasoning": None, "tool_calls": None, "usage": None}

    async def call_stream(
        self,
        *,
        key,
        model,
        messages,
        timeout,
        tools,
        tool_choice,
        extra_body=None,
    ):
        # Simulate OpenAI SSE semantics at provider-layer:
        # - content role empty initial delta
        # - then tool_calls deltas for function_call streaming
        # - then finish_reason
        yield {
            "type": "content",
            "content": "",
        }
        yield {
            "type": "tool_calls",
            "index": 0,
            "id": "call_1",
            "function": {
                "name": "get_weather",
                "arguments": "{\"city\":\"Pari",
            },
        }
        yield {
            "type": "tool_calls",
            "index": 0,
            "function": {
                "arguments": "s\"}",
            },
        }
        yield {
            "type": "content",
            "content": "", 
        }
        yield {
            "type": "finish",
            "finish_reason": "stop",
        }


@pytest_asyncio.fixture
async def app() -> FastAPI:

    providers = {"tool_ok": ToolingStreamingProvider()}
    all_models = {p.name: p.get_all_models() for p in providers.values()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)
    return create_app(router=router, strategy=strategy)


@pytest.mark.asyncio
async def test_openai_stream_chunk_delta_and_finish_shapes(app: FastAPI):
    transport = ASGITransport(app=app)

    async with AsyncClient(base_url="http://test", transport=transport) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        assert resp.status_code == 200
        lines = [ln for ln in resp.text.splitlines() if ln.startswith("data: {")]
        assert lines, "expected at least one streamed chunk"

        payloads = [json.loads(ln[len("data: ") :]) for ln in lines]

        # Basic OpenAI invariants
        for p in payloads:
            assert p["object"] == "chat.completion.chunk"
            assert p["choices"][0]["index"] == 0
            assert "delta" in p["choices"][0]

        # finish_reason must appear (at least once) and last streamed chunk must be stop
        finish_reasons = [p["choices"][0].get("finish_reason") for p in payloads]
        assert "stop" in finish_reasons

        last = payloads[-1]
        assert last["choices"][0].get("finish_reason") == "stop"

        # Tool/function_call deltas are tested separately at router/provider level.
        # This test primarily enforces OpenAI chunk invariants (object/name/finish_reason).



