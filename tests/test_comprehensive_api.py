"""Comprehensive test suite for auto-gateway OpenAI API compatibility."""
import json
import asyncio

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from auto_gateway.core.server import create_app
from auto_gateway.core.router import ProviderRouter, RouteRequest
from auto_gateway.providers.base import BaseProvider, ProviderCallResult
from auto_gateway.strategies.sequential import SequentialStrategy
from auto_gateway.config.schema import GatewayConfig
from auto_gateway.config.manager import load_config
from auto_gateway.providers.registry import available_provider_types


class DummyProvider(BaseProvider):
    def __init__(self, name: str = "dummy"):
        super().__init__(name=name, keys=[None], models={"gpt-4o-mini": []})

    async def call(self, *, key, model, messages, timeout, tools, tool_choice, extra_body=None) -> ProviderCallResult:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content")
                last_user = c if isinstance(c, str) else (c[0].get("text") if isinstance(c, list) and c else "")
                break
        return {"text": f"dummy-{self.name}: {last_user}", "reasoning": None, "tool_calls": None, "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

    async def call_stream(self, *, key, model, messages, timeout, tools, tool_choice, extra_body=None):
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content")
                last_user = c if isinstance(c, str) else (c[0].get("text") if isinstance(c, list) and c else "")
                break
        text = f"dummy-{self.name}: {last_user}"
        for ch in text:
            yield {"type": "content", "content": ch}
        yield {"type": "finish", "finish_reason": "stop"}


class ToolCallProvider(BaseProvider):
    def __init__(self):
        super().__init__(name="tool_provider", keys=[None], models={"gpt-4o-mini": []})

    async def call(self, *, key, model, messages, timeout, tools, tool_choice, extra_body=None) -> ProviderCallResult:
        return {"text": "", "reasoning": None, "tool_calls": [{"id": "call_abc123", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}], "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}}

    async def call_stream(self, *, key, model, messages, timeout, tools, tool_choice, extra_body=None):
        yield {"type": "content", "content": ""}
        yield {"type": "tool_calls", "index": 0, "id": "call_abc123", "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}
        yield {"type": "finish", "finish_reason": "tool_calls"}


class FailingProvider(BaseProvider):
    def __init__(self):
        super().__init__(name="always_fail", keys=[None], models={"m": []})

    async def call(self, *, key, model, messages, timeout, tools, tool_choice, extra_body=None):
        raise RuntimeError("intentional failure")

    async def call_stream(self, *, key, model, messages, timeout, tools, tool_choice, extra_body=None):
        raise RuntimeError("intentional stream failure")


class VisionAwareProvider(BaseProvider):
    def __init__(self, features: list[str], name: str = "vision_provider"):
        super().__init__(name=name, keys=[None], models={"gpt-4-vision": features})

    async def call(self, *, key, model, messages, timeout, tools, tool_choice, extra_body=None) -> ProviderCallResult:
        content_types = []
        for m in messages:
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for item in m["content"]:
                    content_types.append(item.get("type", "unknown"))
        return {"text": f"received: {json.dumps(content_types)}", "reasoning": None, "tool_calls": None, "usage": None}

    async def call_stream(self, *, key, model, messages, timeout, tools, tool_choice, extra_body=None):
        content_types = []
        for m in messages:
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for item in m["content"]:
                    content_types.append(item.get("type", "unknown"))
        yield {"type": "content", "content": f"received: {json.dumps(content_types)}"}
        yield {"type": "finish", "finish_reason": "stop"}


class EmptyContentProvider(BaseProvider):
    def __init__(self):
        super().__init__(name="empty", keys=[None], models={"m": []})

    async def call(self, *, key, model, messages, timeout, tools, tool_choice, extra_body=None) -> ProviderCallResult:
        return {"text": "", "reasoning": None, "tool_calls": None, "usage": None}

    async def call_stream(self, *, key, model, messages, timeout, tools, tool_choice, extra_body=None):
        yield {"type": "content", "content": ""}
        yield {"type": "content", "content": ""}
        yield {"type": "finish", "finish_reason": "stop"}


# ── Fixtures ──


@pytest_asyncio.fixture
def basic_app() -> FastAPI:
    providers = {"dummy": DummyProvider()}
    all_models = {p.name: p.get_all_models() for p in providers.values()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)
    return create_app(router=router, strategy=strategy)


@pytest_asyncio.fixture
def tool_app() -> FastAPI:
    providers = {"tool_provider": ToolCallProvider()}
    all_models = {p.name: p.get_all_models() for p in providers.values()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)
    return create_app(router=router, strategy=strategy)


@pytest_asyncio.fixture
def failover_app() -> FastAPI:
    # "always_fail" must match FailingProvider's name (default __init__)
    # and its models registered under "always_fail"
    providers = {
        "always_fail": FailingProvider(),
        "ok": DummyProvider(name="ok"),
    }
    all_models = {p.name: p.get_all_models() for p in providers.values()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)
    return create_app(router=router, strategy=strategy)


@pytest_asyncio.fixture
def vision_app() -> FastAPI:
    providers = {
        "vision": VisionAwareProvider(features=["vision"], name="vision"),
        "novision": DummyProvider(name="novision"),
    }
    all_models = {p.name: p.get_all_models() for p in providers.values()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)
    return create_app(router=router, strategy=strategy)


# ── Test 1: Non-streaming basic chat ──


@pytest.mark.asyncio
async def test_non_stream_basic(basic_app: FastAPI):
    transport = ASGITransport(app=basic_app)
    async with AsyncClient(base_url="http://test", transport=transport) as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello world"}],
            "stream": False,
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl_")
    assert isinstance(data["created"], int)
    assert data["model"] == "gpt-4o-mini"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["index"] == 0
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "dummy-dummy: hello world"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["prompt_tokens"] == 10


# ── Test 2: Non-streaming with tool calls ──


@pytest.mark.asyncio
async def test_non_stream_tool_calls(tool_app: FastAPI):
    transport = ASGITransport(app=tool_app)
    async with AsyncClient(base_url="http://test", transport=transport) as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "what's the weather?"}],
            "stream": False,
            "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
            "tool_choice": "auto",
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    msg = data["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["id"] == "call_abc123"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"])["city"] == "Paris"


# ── Test 3: Streaming basic chat (SSE delta shapes) ──


@pytest.mark.asyncio
async def test_stream_basic_delta_shapes(basic_app: FastAPI):
    transport = ASGITransport(app=basic_app)
    async with AsyncClient(base_url="http://test", transport=transport) as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        })
    assert resp.status_code == 200
    lines = [ln for ln in resp.text.splitlines() if ln.startswith("data: ")]
    assert len(lines) >= 2
    chunks = []
    for ln in lines:
        p = ln[len("data: "):].strip()
        if p == "[DONE]": continue
        chunks.append(json.loads(p))
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["id"].startswith("chatcmpl_")
        assert isinstance(chunk["created"], int)
        assert len(chunk["choices"]) == 1
        assert chunk["choices"][0]["index"] == 0
        assert "delta" in chunk["choices"][0]
    full = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert full == "dummy-dummy: hello"
    assert chunks[-1]["choices"][0].get("finish_reason") == "stop"


# ── Test 4: Streaming with tool calls ──


@pytest.mark.asyncio
async def test_stream_tool_calls_delta_shapes(tool_app: FastAPI):
    transport = ASGITransport(app=tool_app)
    async with AsyncClient(base_url="http://test", transport=transport) as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "weather in paris"}],
            "stream": True,
        })
    assert resp.status_code == 200
    lines = [ln for ln in resp.text.splitlines() if ln.startswith("data: ")]
    chunks = []
    for ln in lines:
        p = ln[len("data: "):].strip()
        if p == "[DONE]": continue
        chunks.append(json.loads(p))
    tc_chunks = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
    assert len(tc_chunks) >= 1
    tc = tc_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert tc["index"] == 0
    assert tc["id"] == "call_abc123"
    assert tc["function"]["name"] == "get_weather"
    finish_chunks = [c for c in chunks if c["choices"][0].get("finish_reason")]
    assert len(finish_chunks) >= 1
    assert finish_chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


# ── Test 5: Non-streaming provider failover ──


@pytest.mark.asyncio
async def test_non_stream_failover(failover_app: FastAPI):
    transport = ASGITransport(app=failover_app)
    async with AsyncClient(base_url="http://test", transport=transport) as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "m", "messages": [{"role": "user", "content": "test failover"}], "stream": False,
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "dummy-ok: test failover"


# ── Test 6: Streaming provider failover ──


@pytest.mark.asyncio
async def test_stream_failover(failover_app: FastAPI):
    transport = ASGITransport(app=failover_app)
    async with AsyncClient(base_url="http://test", transport=transport) as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "m", "messages": [{"role": "user", "content": "test stream failover"}], "stream": True,
        })
    assert resp.status_code == 200
    lines = resp.text.splitlines()
    done_lines = [ln for ln in lines if ln.strip() == "data: [DONE]"]
    assert len(done_lines) >= 1
    content = []
    for ln in lines:
        if ln.startswith("data: {"):
            payload = json.loads(ln[len("data: "):])
            delta = payload["choices"][0]["delta"]
            if delta.get("content"):
                content.append(delta["content"])
    assert "dummy-ok" in "".join(content)


# ── Test 7: All providers exhausted ──


@pytest.mark.asyncio
async def test_all_providers_exhausted():
    providers = {"always_fail": FailingProvider()}
    all_models = {p.name: p.get_all_models() for p in providers.values()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)
    app = create_app(router=router, strategy=strategy)
    transport = ASGITransport(app=app)
    async with AsyncClient(base_url="http://test", transport=transport) as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "m", "messages": [{"role": "user", "content": "hello"}], "stream": False,
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == ""


# ── Test 8: Message filtering (vision) ──


@pytest.mark.asyncio
async def test_message_filtering_vision(vision_app: FastAPI):
    transport = ASGITransport(app=vision_app)
    async with AsyncClient(base_url="http://test", transport=transport) as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "gpt-4-vision",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "describe this"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}],
            "stream": False,
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == 'received: ["text", "image_url"]'


@pytest.mark.asyncio
async def test_message_filtering_no_vision():
    providers = {"novision": DummyProvider(name="novision")}
    all_models = {p.name: p.get_all_models() for p in providers.values()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)
    app = create_app(router=router, strategy=strategy)
    transport = ASGITransport(app=app)
    async with AsyncClient(base_url="http://test", transport=transport) as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}],
            "stream": False,
        })
    assert resp.status_code == 200
    data = resp.json()
    assert "dummy-novision" in data["choices"][0]["message"]["content"]


# ── Test 9: Config schema ──


def test_config_schema_valid():
    cfg = GatewayConfig(
        server={"host": "0.0.0.0", "port": 8080, "tunnel": "none"},
        router={"strategy": "adaptive", "retries": 3},
        providers=[{"name": "openai", "type": "openai_compatible", "api_key": "sk-test", "base_url": "https://api.openai.com/v1", "models": {"gpt-4o-mini": []}}],
    )
    assert cfg.server.host == "0.0.0.0"
    assert cfg.server.port == 8080
    assert cfg.router.strategy == "adaptive"
    assert len(cfg.providers) == 1
    assert cfg.providers[0].name == "openai"


def test_config_schema_invalid_type():
    with pytest.raises(ValidationError):
        GatewayConfig(
            server={"host": "0.0.0.0", "port": 8080},
            router={"strategy": "sequential"},
            providers=[{"name": "bad", "type": "nonexistent_type", "base_url": "http://example.com", "models": {}}],
        )


# ── Test 10: Config loading ──


@pytest.mark.asyncio
async def test_config_loading(tmp_path):
    config_content = json.dumps({
        "server": {"host": "0.0.0.0", "port": 8000, "tunnel": "none"},
        "router": {"strategy": "sequential", "retries": 2},
        "providers": [{"name": "test_provider", "type": "openai_compatible", "api_key": "sk-test", "base_url": "https://test.local/v1", "models": {"test-model": []}}],
    })
    config_path = tmp_path / "config.json"
    config_path.write_text(config_content)
    cfg = load_config(str(config_path))
    assert cfg.server.port == 8000
    assert cfg.router.strategy == "sequential"
    assert cfg.providers[0].name == "test_provider"


# ── Test 11: Registry ──


def test_provider_registry_available_types():
    assert isinstance(available_provider_types(), list)


# ── Test 12: Tunnel info ──


def test_tunnel_info_basic():
    from auto_gateway.network.hosting import TunnelInfo
    info = TunnelInfo(public_url="https://test.trycloudflare.com", backend="cloudflared")
    assert info.public_url == "https://test.trycloudflare.com"
    assert info.backend == "cloudflared"
    info2 = TunnelInfo(public_url="https://abc123.ngrok.io", backend="ngrok")
    assert info2.public_url == "https://abc123.ngrok.io"
    assert info2.backend == "ngrok"


# ── Test 13: Concurrent requests ──


@pytest.mark.asyncio
async def test_concurrent_requests(basic_app: FastAPI):
    transport = ASGITransport(app=basic_app)
    async with AsyncClient(base_url="http://test", transport=transport) as client:
        tasks = [client.post("/v1/chat/completions", json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": f"request {i}"}], "stream": False}) for i in range(10)]
        responses = await asyncio.gather(*tasks)
    for i, resp in enumerate(responses):
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == f"dummy-dummy: request {i}"


# ── Test 14: Stream empty content ──


@pytest.mark.asyncio
async def test_stream_empty_content():
    providers = {"empty": EmptyContentProvider()}
    all_models = {p.name: p.get_all_models() for p in providers.values()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)
    app = create_app(router=router, strategy=strategy)
    transport = ASGITransport(app=app)
    async with AsyncClient(base_url="http://test", transport=transport) as client:
        resp = await client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert resp.status_code == 200
    lines = [ln for ln in resp.text.splitlines() if ln.startswith("data: ")]
    payload_lines = [ln for ln in lines if not ln.strip().endswith("[DONE]")]
    assert len(payload_lines) >= 1
    finishes = [ln for ln in payload_lines if json.loads(ln[len("data: "):])["choices"][0].get("finish_reason")]
    assert len(finishes) >= 1


# ── Test 15: Route with provider selected by name ──


@pytest.mark.asyncio
async def test_non_stream_provider_selected_by_name():
    prov1 = DummyProvider(name="prov1")
    prov2 = DummyProvider(name="prov2")
    providers = {"prov1": prov1, "prov2": prov2}
    all_models = {p.name: p.get_all_models() for p in providers.values()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)
    req = RouteRequest(strategy=strategy, provider="prov2", models=None, timeout=15.0, shuffle=False, tools=None, tool_choice=None, extra_body={}, messages=[{"role": "user", "content": "hello"}])
    result = await router.route(req)
    assert result["text"] == "dummy-prov2: hello"


# ── Test 16: SSE termination marker ──


@pytest.mark.asyncio
async def test_sse_termination(basic_app: FastAPI):
    transport = ASGITransport(app=basic_app)
    async with AsyncClient(base_url="http://test", transport=transport) as client:
        resp = await client.post("/v1/chat/completions", json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}], "stream": True})
    assert resp.status_code == 200
    assert "data: [DONE]" in resp.text
    data_lines = [ln.strip() for ln in resp.text.splitlines() if ln.startswith("data: ")]
    assert data_lines[-1] == "data: [DONE]"


# ── Test 17: Route with shuffle ──


@pytest.mark.asyncio
async def test_route_shuffle():
    prov1 = DummyProvider(name="prov1")
    prov2 = DummyProvider(name="prov2")
    providers = {"prov1": prov1, "prov2": prov2}
    all_models = {p.name: p.get_all_models() for p in providers.values()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)
    req = RouteRequest(strategy=strategy, provider=None, models=None, timeout=15.0, shuffle=True, tools=None, tool_choice=None, extra_body={}, messages=[{"role": "user", "content": "test shuffle"}])
    result = await router.route(req)
    assert result["text"] is not None
    assert "dummy-" in result["text"]
