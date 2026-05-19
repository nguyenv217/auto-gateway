import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient

from auto_gateway.cli.main import DummyProvider
from auto_gateway.core.server import create_app
from auto_gateway.core.router import ProviderRouter
from auto_gateway.strategies.sequential import SequentialStrategy


@pytest_asyncio.fixture
def app() -> FastAPI:
    providers = {"dummy": DummyProvider()}
    all_models = {p.name: p.get_all_models() for p in providers.values()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)
    return create_app(router=router, strategy=strategy)


@pytest.mark.asyncio
async def test_chat_completion_non_stream(app: FastAPI):
    from httpx import ASGITransport

    transport = ASGITransport(app=app)

    async with AsyncClient(base_url="http://test", transport=transport) as client:
        resp = await client.post(
            "/v1/chat/completions",

            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"].startswith("dummy: ")

