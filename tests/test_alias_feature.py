"""Tests for the provider+alias feature — pinning requests to specific keys."""
import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from auto_gateway.core.server import create_app
from auto_gateway.core.router import ProviderRouter, RouteRequest
from auto_gateway.core.exceptions import AllProvidersExhaustedError
from auto_gateway.providers.base import BaseProvider, ProviderCallResult
from auto_gateway.strategies.sequential import SequentialStrategy
from auto_gateway.strategies.adaptive import AdaptiveStrategy
from auto_gateway.strategies.bandit import UCBBanditStrategy
from auto_gateway.config.schema import GatewayConfig


# ── Helper: provider that echoes which key was used ──
class EchoKeyProvider(BaseProvider):
    def __init__(self, name: str = "echo", keys: list[str] | None = None, models: dict[str, list[str]] | None = None, key_aliases: dict[str, str] | None = None):
        super().__init__(name=name, keys=keys, models=models or {"m": []}, key_aliases=key_aliases)

    async def call(self, *, key, model, messages, timeout, tools, tool_choice, extra_body=None) -> ProviderCallResult:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content")
                last_user = c if isinstance(c, str) else (c[0].get("text") if isinstance(c, list) and c else "")
                break
        key_display = key[:12] + "..." if key else "no-key"
        return {"text": f"{self.name}:{key_display}:{last_user}", "reasoning": None, "tool_calls": None, "usage": None}

    async def call_stream(self, *, key, model, messages, timeout, tools, tool_choice, extra_body=None):
        result = await self.call(key=key, model=model, messages=messages, timeout=timeout, tools=tools, tool_choice=tool_choice, extra_body=extra_body)
        for ch in result["text"]:
            yield {"type": "content", "content": ch}
        yield {"type": "finish", "finish_reason": "stop"}


# ── Config schema tests ──

def test_config_api_key_dict_valid():
    """api_key can be a dict for openai_compatible."""
    cfg = GatewayConfig(
        server={"host": "0.0.0.0", "port": 8080},
        router={"strategy": "sequential"},
        providers=[{
            "name": "test",
            "type": "openai_compatible",
            "api_key": {"us-east": "sk-east", "us-west": "sk-west"},
            "base_url": "https://test.local/v1",
            "models": {"m": []},
        }],
    )
    assert isinstance(cfg.providers[0].api_key, dict)
    assert cfg.providers[0].api_key["us-east"] == "sk-east"


def test_config_api_key_dict_google():
    """api_key dict also works for google provider."""
    cfg = GatewayConfig(
        server={"host": "0.0.0.0", "port": 8080},
        router={"strategy": "sequential"},
        providers=[{
            "name": "g",
            "type": "google",
            "api_key": {"primary": "g-key-1", "secondary": "g-key-2"},
            "models": {"gemini-1.5-flash": []},
        }],
    )
    assert isinstance(cfg.providers[0].api_key, dict)
    assert cfg.providers[0].api_key["secondary"] == "g-key-2"


def test_config_api_key_old_formats_still_work():
    """String and list api_key formats remain valid."""
    # string
    cfg1 = GatewayConfig(
        server={"host": "0.0.0.0", "port": 8080},
        router={"strategy": "sequential"},
        providers=[{
            "name": "t1", "type": "openai_compatible",
            "api_key": "sk-abc", "base_url": "https://x.com/v1", "models": {},
        }],
    )
    assert cfg1.providers[0].api_key == "sk-abc"

    # list
    cfg2 = GatewayConfig(
        server={"host": "0.0.0.0", "port": 8080},
        router={"strategy": "sequential"},
        providers=[{
            "name": "t2", "type": "openai_compatible",
            "api_key": ["sk-1", "sk-2"], "base_url": "https://x.com/v1", "models": {},
        }],
    )
    assert cfg2.providers[0].api_key == ["sk-1", "sk-2"]


# ── BaseProvider.get_keys_for_alias ──

def test_get_keys_for_alias_none_returns_all_keys():
    prov = EchoKeyProvider(
        name="p",
        keys=["k1", "k2", "k3"],
        models={"m": []},
        key_aliases={"a": "k1", "b": "k2"},
    )
    assert set(prov.get_keys_for_alias(None)) == {"k1", "k2", "k3"}


def test_get_keys_for_alias_specific():
    prov = EchoKeyProvider(
        name="p",
        keys=["k1", "k2", "k3"],
        models={"m": []},
        key_aliases={"a": "k1", "b": "k2"},
    )
    assert prov.get_keys_for_alias("a") == ["k1"]
    assert prov.get_keys_for_alias("b") == ["k2"]


def test_get_keys_for_alias_not_found():
    prov = EchoKeyProvider(
        name="p",
        keys=["k1", "k2"],
        models={"m": []},
        key_aliases={"a": "k1"},
    )
    assert prov.get_keys_for_alias("nonexistent") == []


def test_get_keys_for_alias_no_aliases_configured():
    prov = EchoKeyProvider(name="p", keys=["k1"], models={"m": []})
    assert prov.get_keys_for_alias(None) == ["k1"]
    assert prov.get_keys_for_alias("any") == []


# ── RouteRequest alias propagation (router-level) ──

@pytest.mark.asyncio
async def test_router_uses_aliased_key():
    prov = EchoKeyProvider(
        name="echoer",
        keys=["k-east", "k-west"],
        models={"m": []},
        key_aliases={"east": "k-east", "west": "k-west"},
    )
    providers = {"echoer": prov}
    all_models = {pn: p.get_all_models() for pn, p in providers.items()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)

    req = RouteRequest(
        strategy=strategy,
        provider="echoer",
        alias="west",
        models=["m"],
        timeout=15.0,
        shuffle=False,
        tools=None,
        tool_choice=None,
        extra_body={},
        messages=[{"role": "user", "content": "hi"}],
    )
    result = await router.route(req)
    assert "k-west" in result["text"]
    assert "k-east" not in result["text"]


@pytest.mark.asyncio
async def test_router_with_alias_not_found_skips_provider():
    """When alias doesn't match, get_keys_for_alias returns [] and provider is skipped,
    resulting in AllProvidersExhaustedError."""
    prov = EchoKeyProvider(
        name="only",
        keys=["k1"],
        models={"m": []},
        key_aliases={"good": "k1"},
    )
    providers = {"only": prov}
    all_models = {pn: p.get_all_models() for pn, p in providers.items()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)

    req = RouteRequest(
        strategy=strategy,
        provider=None,
        alias="bad_alias",
        models=["m"],
        timeout=15.0,
        shuffle=False,
        tools=None,
        tool_choice=None,
        extra_body={},
        messages=[{"role": "user", "content": "hi"}],
    )
    # No targets match the alias → AllProvidersExhaustedError
    with pytest.raises(AllProvidersExhaustedError):
        await router.route(req)


@pytest.mark.asyncio
async def test_router_unrecognized_provider_falls_through_to_all():
    """When provider name is unrecognized, the strategy falls through to all providers
    (pre-existing behavior — no error raised)."""
    prov = EchoKeyProvider(name="real", keys=["k1"], models={"m": []})
    providers = {"real": prov}
    all_models = {pn: p.get_all_models() for pn, p in providers.items()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)

    req = RouteRequest(
        strategy=strategy,
        provider="nonexistent_provider",
        alias=None,
        models=["m"],
        timeout=15.0,
        shuffle=False,
        tools=None,
        tool_choice=None,
        extra_body={},
        messages=[{"role": "user", "content": "hi"}],
    )
    # Unrecognized provider → falls through to all providers (existing behavior)
    result = await router.route(req)
    assert "real" in result["text"]


@pytest.mark.asyncio
async def test_router_without_alias_rotates_all_keys():
    """Without alias, all keys are available for rotation."""
    prov = EchoKeyProvider(
        name="echoer",
        keys=["k1", "k2"],
        models={"m": []},
        key_aliases={"a": "k1", "b": "k2"},
    )
    providers = {"echoer": prov}
    all_models = {pn: p.get_all_models() for pn, p in providers.items()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)

    req = RouteRequest(
        strategy=strategy,
        provider=None,
        alias=None,
        models=["m"],
        timeout=15.0,
        shuffle=False,
        tools=None,
        tool_choice=None,
        extra_body={},
        messages=[{"role": "user", "content": "hi"}],
    )
    result = await router.route(req)
    # First key in order should be used
    assert "k1" in result["text"]


# ── API-level tests (non-streaming) ──

@pytest.mark.asyncio
async def test_api_with_provider_and_alias():
    prov = EchoKeyProvider(
        name="apiprov",
        keys=["kk-east", "kk-west"],
        models={"m": []},
        key_aliases={"east": "kk-east", "west": "kk-west"},
    )
    providers = {"apiprov": prov}
    all_models = {pn: p.get_all_models() for pn, p in providers.items()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)
    app = create_app(router=router, strategy=strategy, all_models=all_models)
    transport = ASGITransport(app=app)
    async with AsyncClient(base_url="http://test", transport=transport) as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "provider": "apiprov",
            "alias": "west",
        })
    assert resp.status_code == 200
    data = resp.json()
    assert "kk-west" in data["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_api_with_provider_only():
    prov = EchoKeyProvider(
        name="apiprov2",
        keys=["kA", "kB"],
        models={"m": []},
        key_aliases={"a": "kA", "b": "kB"},
    )
    providers = {"apiprov2": prov}
    all_models = {pn: p.get_all_models() for pn, p in providers.items()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)
    app = create_app(router=router, strategy=strategy, all_models=all_models)
    transport = ASGITransport(app=app)
    async with AsyncClient(base_url="http://test", transport=transport) as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "provider": "apiprov2",
        })
    assert resp.status_code == 200
    data = resp.json()
    # Without alias, the first key kA is used (sequential order)
    assert "kA" in data["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_api_with_alias_without_provider():
    """Alias without provider: the provider that has the alias will be matched."""
    prov1 = EchoKeyProvider(
        name="prov1",
        keys=["bad-key"],
        models={"m": []},
        key_aliases={"target": "good-key"},
    )
    prov2 = EchoKeyProvider(
        name="prov2",
        keys=["other"],
        models={"m": []},
        key_aliases={},
    )
    providers = {"prov1": prov1, "prov2": prov2}
    all_models = {pn: p.get_all_models() for pn, p in providers.items()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)
    app = create_app(router=router, strategy=strategy, all_models=all_models)
    transport = ASGITransport(app=app)
    async with AsyncClient(base_url="http://test", transport=transport) as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "alias": "target",
        })
    assert resp.status_code == 200
    data = resp.json()
    assert "good-key" in data["choices"][0]["message"]["content"]


# ── Streaming with alias ──

@pytest.mark.asyncio
async def test_streaming_with_alias():
    prov = EchoKeyProvider(
        name="streamprov",
        keys=["sk-east", "sk-west"],
        models={"m": []},
        key_aliases={"east": "sk-east", "west": "sk-west"},
    )
    providers = {"streamprov": prov}
    all_models = {pn: p.get_all_models() for pn, p in providers.items()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)
    app = create_app(router=router, strategy=strategy, all_models=all_models)
    transport = ASGITransport(app=app)
    async with AsyncClient(base_url="http://test", transport=transport) as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "data"}],
            "stream": True,
            "provider": "streamprov",
            "alias": "east",
        })
    assert resp.status_code == 200
    full_text = ""
    for ln in resp.text.splitlines():
        if ln.startswith("data: {"):
            chunk = json.loads(ln[len("data: "):])
            delta = chunk["choices"][0]["delta"]
            if delta.get("content"):
                full_text += delta["content"]
    assert "sk-east" in full_text
    assert "sk-west" not in full_text


# ── All 3 strategies with alias ──

@pytest.mark.asyncio
async def test_adaptive_strategy_respects_alias():
    prov = EchoKeyProvider(
        name="adaptiveprov",
        keys=["ak-east", "ak-west"],
        models={"m": []},
        key_aliases={"east": "ak-east", "west": "ak-west"},
    )
    providers = {"adaptiveprov": prov}
    all_models = {pn: p.get_all_models() for pn, p in providers.items()}
    strategy = AdaptiveStrategy(providers, all_models)
    router = ProviderRouter(providers)
    req = RouteRequest(
        strategy=strategy,
        provider="adaptiveprov",
        alias="east",
        models=["m"],
        timeout=15.0,
        shuffle=False,
        tools=None,
        tool_choice=None,
        extra_body={},
        messages=[{"role": "user", "content": "adaptive"}],
    )
    result = await router.route(req)
    assert "ak-east" in result["text"]
    assert "ak-west" not in result["text"]


@pytest.mark.asyncio
async def test_bandit_strategy_respects_alias():
    prov = EchoKeyProvider(
        name="banditprov",
        keys=["bk-east", "bk-west"],
        models={"m": []},
        key_aliases={"east": "bk-east", "west": "bk-west"},
    )
    providers = {"banditprov": prov}
    all_models = {pn: p.get_all_models() for pn, p in providers.items()}
    strategy = UCBBanditStrategy(providers, all_models)
    router = ProviderRouter(providers)
    req = RouteRequest(
        strategy=strategy,
        provider="banditprov",
        alias="east",
        models=["m"],
        timeout=15.0,
        shuffle=False,
        tools=None,
        tool_choice=None,
        extra_body={},
        messages=[{"role": "user", "content": "bandit"}],
    )
    result = await router.route(req)
    assert "bk-east" in result["text"]
    assert "bk-west" not in result["text"]


@pytest.mark.asyncio
async def test_sequential_strategy_respects_alias():
    prov = EchoKeyProvider(
        name="seqprov",
        keys=["sk-east", "sk-west"],
        models={"m": []},
        key_aliases={"east": "sk-east", "west": "sk-west"},
    )
    providers = {"seqprov": prov}
    all_models = {pn: p.get_all_models() for pn, p in providers.items()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)
    req = RouteRequest(
        strategy=strategy,
        provider="seqprov",
        alias="west",
        models=["m"],
        timeout=15.0,
        shuffle=False,
        tools=None,
        tool_choice=None,
        extra_body={},
        messages=[{"role": "user", "content": "seq"}],
    )
    result = await router.route(req)
    assert "sk-west" in result["text"]
    assert "sk-east" not in result["text"]


# ── Edge cases ──

@pytest.mark.asyncio
async def test_dict_key_with_none_values():
    """Keys list may contain None entries from old configs; aliases still work."""
    prov = EchoKeyProvider(
        name="nprov",
        keys=["real-key", None],
        models={"m": []},
        key_aliases={"valid": "real-key"},
    )
    providers = {"nprov": prov}
    all_models = {pn: p.get_all_models() for pn, p in providers.items()}
    strategy = SequentialStrategy(providers, all_models)
    router = ProviderRouter(providers)
    req = RouteRequest(
        strategy=strategy,
        provider=None,
        alias="valid",
        models=["m"],
        timeout=15.0,
        shuffle=False,
        tools=None,
        tool_choice=None,
        extra_body={},
        messages=[{"role": "user", "content": "edge"}],
    )
    result = await router.route(req)
    assert "real-key" in result["text"]


def test_provider_with_empty_aliases_dict():
    """Empty key_aliases dict means alias lookups always return []."""
    prov = EchoKeyProvider(
        name="emptyaliases",
        keys=["k1"],
        models={"m": []},
        key_aliases={},
    )
    assert prov.get_keys_for_alias("anything") == []
    assert prov.get_keys_for_alias(None) == ["k1"]


def test_provider_with_no_aliases_param():
    """key_aliases defaults to None, which becomes {}."""
    prov = EchoKeyProvider(name="noalias", keys=["k1"], models={"m": []})
    assert prov.get_keys_for_alias("anything") == []
    assert prov.get_keys_for_alias(None) == ["k1"]
