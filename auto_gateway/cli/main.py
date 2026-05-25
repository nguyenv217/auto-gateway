from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import logging

import typer

from ..config.manager import load_config, save_global_config, load_global_config

from ..core.router import ProviderRouter
from ..core.server import create_app
from ..providers.base import BaseProvider, ProviderCallResult
from ..providers.openai_compatible import OpenAICompatibleProvider
from ..providers.google import GoogleProvider
from ..strategies.adaptive import AdaptiveStrategy
from ..strategies.sequential import SequentialStrategy
from ..strategies.bandit import UCBBanditStrategy

logger = logging.getLogger("auto-gateway")

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
        # Normalize api_key: it can be str, list[str], dict[str, str], or None
        raw_keys = p.api_key
        if isinstance(raw_keys, dict):
            # New format: {"alias1": "key1", "alias2": "key2"}
            keys_list = list(raw_keys.values())
            key_aliases = raw_keys
        elif isinstance(raw_keys, list):
            keys_list = raw_keys
            key_aliases = None
        elif raw_keys is not None:
            keys_list = [raw_keys]
            key_aliases = None
        else:
            keys_list = [None]
            key_aliases = None

        if p.type == "openai_compatible":
            prov = OpenAICompatibleProvider(
                name=p.name,
                base_url=p.base_url,
                keys=keys_list,
                model_configs=p.models,
                extra={"extra_body": getattr(p, "extra_body", {})},
                key_aliases=key_aliases,
            )
        elif p.type == "google":
            prov = GoogleProvider(
                name=p.name,
                keys=keys_list,
                model_configs=p.models,
                key_aliases=key_aliases,
            )
        else:
            raise ValueError(f"Unsupported provider type: {p.type}")

        providers[p.name] = prov
        all_models[p.name] = prov.get_all_models()

    return providers, all_models

@app.command()
def start(
    config: str | None = typer.Option(None, "--config", help="Path to config.json (uses global config if omitted)"),
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
    tunnel: str | None = typer.Option(None, "--tunnel", help="none|ngrok|cloudflared (public URL optional)"),
    name: str | None = typer.Option(None, "--name", "-n", help="Alias of a global config defined with `auto-gateway save_global --name\-n`"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
):
    """Start config-driven gateway."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(levelname)s:%(name)s: %(message)s"
    )
    
    # Silence chatty third-party HTTP loggers unless we explicitly ask for DEBUG
    if numeric_level >= logging.INFO:
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    logger.info("Initializing configuration...")
    if config is None:
        try:
            cfg = load_global_config(name)
            logger.info("Loaded global config from ~/.auto-gateway/config.json")
        except FileNotFoundError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
    else:
        cfg = load_config(config)


    if host is not None:
        cfg.server.host = host
    if port is not None:
        cfg.server.port = port
    if tunnel is not None:
        cfg.server.tunnel = tunnel

    providers, all_models = _build_providers(cfg)

    if cfg.router.strategy == "adaptive":
        strategy = AdaptiveStrategy(
            providers=providers,
            all_models=all_models,
            persistence_path=None,
        )
    elif cfg.router.strategy == "bandit":
        strategy = UCBBanditStrategy(providers, all_models)
    else:
        strategy = SequentialStrategy(providers, all_models)

    router = ProviderRouter(providers)
    application = create_app(
        router=router, 
        strategy=strategy,
        api_key=cfg.server.api_key,
        timeout=cfg.router.timeout,
        all_models=all_models,
    )


    from ..network.hosting import start_tunnel

    tunnel_info = None
    if cfg.server.tunnel and cfg.server.tunnel != "none":
        tunnel_info = start_tunnel(
            cfg.server.tunnel,
            port=cfg.server.port,
            config={"ngrok_authtoken": cfg.extra.get("tunnels", {}).get("ngrok_authtoken"), "cloudflared_binary": cfg.extra.get("tunnels", {}).get("cloudflared_binary")},
        )
        import asyncio
        tunnel_info = asyncio.run(tunnel_info)
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
def save_global(
    config: str = typer.Option(..., "--config", help="Path to config.json to save as global"),
    alias: str | None = typer.Option(None, "--name", "-n", help="Alias to be used globally with --name/-n"),
):
    """Save the specified config as the global default config."""
    save_global_config(config, alias)
    typer.echo(f"Global config saved to ~/.auto-gateway/config.json")


@app.command()
def version():

    typer.echo("auto-gateway 0.1.0")

if __name__ == "__main__":
    app()
