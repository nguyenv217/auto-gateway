from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    socket_path: str | None = None
    tunnel: Literal["none", "ngrok", "cloudflared"] = "none"


class RouterConfig(BaseModel):
    strategy: Literal["sequential", "adaptive"] = "sequential"
    retries: int = 1


class ProviderBaseConfig(BaseModel):
    type: Literal["openai_compatible", "google"]
    name: str
    models: dict[str, list[str]] = Field(default_factory=dict)


class OpenAICompatibleProviderConfig(ProviderBaseConfig):
    type: Literal["openai_compatible"] = "openai_compatible"
    base_url: str
    api_key: str | list[str] | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)


class GoogleProviderConfig(ProviderBaseConfig):
    type: Literal["google"] = "google"
    api_key: str | list[str]


ProviderConfig = OpenAICompatibleProviderConfig | GoogleProviderConfig


class GatewayConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    providers: list[ProviderConfig] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

