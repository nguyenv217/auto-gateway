from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from ..config.manager import load_config
from ..core.router import ProviderRouter
from ..core.server import create_app
from ..providers.base import BaseProvider, ProviderCallResult
from ..providers.openai_compatible import OpenAICompatibleProvider
from ..providers.google import GoogleProvider
from ..strategies.adaptive import AdaptiveStrategy
from ..strategies.sequential import SequentialStrategy

app = typer.Typer(add_completion=False)


class DummyProvider(BaseProvider):
    """Deterministic provider for tests."""

    def __init__(self):
        super().__init__(name="dummy", keys=[None], models={"gpt-4o-mini": []})

    async def call(
        self,
        *,
        key: str | None,
        model: str,
        messages: list[dict[str, Any]],
        timeout: float,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        extra_body: dict[str, Any] | None = None,
    ) -> ProviderCallResult:
        del key, timeout, tools, tool_choice, extra_body
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content")
                last_user = c if isinstance(c, str) else (c[0].get("text") if isinstance(c, list) and c else "")
                break
        return {
            "text": f"dummy: {last_user}",
            "reasoning": None,
            "tool_calls": None,
            "usage": None,
        }


def _build_providers(config) -> tuple[dict[str, BaseProvider], dict[str, dict[str, list[str]]]]:
    providers: dict[str, BaseProvider] = {}
    all_models: dict[str, dict[str, list[str]]] = {}

    for p in config.providers:
        if p.type == "openai_compatible":
            prov = OpenAICompatibleProvider(
                name=p.name,
                base_url=p.base_url,
                keys=[p.api_key] if p.api_key else [None],
                model_configs=p.models,
                extra={"extra_body": getattr(p, "extra_body", {})},
            )
        elif p.type == "google":
            prov = GoogleProvider(
                name=p.name,
                keys=[p.api_key],
                model_configs=p.models,
            )
        else:
            raise ValueError(f"Unsupported provider type: {p.type}")

        providers[p.name] = prov
        all_models[p.name] = prov.get_all_models()

    return providers, all_models


@app.command()
def start(

    config: str = typer.Option(..., "--config", help="Path to config.json"),

    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    tunnel: str = typer.Option("none", "--tunnel", help="none|ngrok|cloudflared (public URL optional)"),
):
    """Start config-driven gateway."""

    cfg = load_config(config)

    # Allow CLI overrides
    cfg.server.host = host
    cfg.server.port = port
    cfg.server.tunnel = tunnel

    providers, all_models = _build_providers(cfg)

    if cfg.router.strategy == "adaptive":
        # persistence omitted for now; can be wired to config
        strategy = AdaptiveStrategy(
            providers=providers,
            all_models=all_models,
            persistence_path=None,
        )
    else:
        strategy = SequentialStrategy(providers, all_models)

    router = ProviderRouter(providers)
    application = create_app(router=router, strategy=strategy)

    from ..network.hosting import start_tunnel

    tunnel_info = None
    if cfg.server.tunnel and cfg.server.tunnel != "none":
        tunnel_info = start_tunnel(
            cfg.server.tunnel,
            port=cfg.server.port,
            config={"ngrok_authtoken": cfg.extra.get("tunnels", {}).get("ngrok_authtoken"), "cloudflared_binary": cfg.extra.get("tunnels", {}).get("cloudflared_binary")},
        )

        # typer command is sync; run tunnel startup in event loop used by uvicorn via asyncio.run
        import asyncio

        tunnel_info = asyncio.run(tunnel_info)  # type: ignore[assignment]
        typer.echo(f"Public URL ({cfg.server.tunnel}): {tunnel_info.public_url}")

    from ..network.uvicorn_runner import run_uvicorn_app

    run_uvicorn_app(
        app=application,
        host=cfg.server.host,
        port=cfg.server.port,
        socket_path=cfg.server.socket_path,
    )


@app.command()
def check(
    config: str = typer.Option(..., "--config", help="Path to config.json"),
):
    """Validate config schema and print summary."""

    cfg = load_config(config)
    typer.echo(f"OK: providers={len(cfg.providers)} strategy={cfg.router.strategy} tunnel={cfg.server.tunnel}")
    for p in cfg.providers:
        typer.echo(f"- {p.name}: type={p.type}, models={list(p.models.keys())}")


@app.command()
def version():
    typer.echo("auto-gateway 0.1.0")

