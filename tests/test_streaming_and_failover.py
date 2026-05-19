import json

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient
from httpx import ASGITransport

from auto_gateway.core.server import create_app
from auto_gateway.core.router import ProviderRouter, RouteRequest
from auto_gateway.strategies.sequential import SequentialStrategy
from auto_gateway.providers.base import BaseProvider, ProviderCallResult


class FailingStreamProvider(BaseProvider):
    def __init__(self):
        super().__init__(name="fail", keys=[None], models={"m": []})

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
        raise RuntimeError("boom")

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
        raise RuntimeError("boom")


class StreamingProvider(BaseProvider):
    def __init__(self):
        super().__init__(name="ok", keys=[None], models={"m": []})

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
        return {"text": "hello", "reasoning": None, "tool_calls": None, "usage": None}

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
        # yield structured BaseProviderDelta dicts
        yield {"type": "content", "content": "hel"}
        yield {"type": "content", "content": "lo"}
        yield {"type": "finish", "finish_reason": "stop"}



@pytest_asyncio.fixture
def app() -> FastAPI:
    providers = {"fail": FailingStreamProvider(), "ok": StreamingProvider()}
    all_models = {p.name: p.get_all_models() for p in providers.values()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)
    return create_app(router=router, strategy=strategy)


@pytest.mark.asyncio
async def test_streaming_sse_failover(app: FastAPI):
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
        body = resp.text

        # Must end with DONE
        assert "data: [DONE]" in body

        # Must contain at least two streamed chunks with incremental content
        # (route wraps into chat.completion.chunk payload)
        chunks = [line for line in body.splitlines() if line.startswith("data: {")]
        assert len(chunks) >= 2

        # Validate that delta.content appears and is incremental
        payloads = [json.loads(line[len("data: ") :]) for line in chunks]
        contents = [p["choices"][0]["delta"].get("content") for p in payloads]
        assert "hel" in contents
        assert "lo" in contents
