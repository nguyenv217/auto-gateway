from __future__ import annotations

from typing import Any, Callable

from .base import BaseProvider


_PROVIDERS: dict[str, Callable[..., BaseProvider]] = {}


def register_provider(name: str):
    def deco(fn: Callable[..., BaseProvider]):
        _PROVIDERS[name] = fn
        return fn

    return deco


def get_provider_factory(name: str) -> Callable[..., BaseProvider]:
    if name not in _PROVIDERS:
        raise KeyError(f"Unknown provider type: {name}")
    return _PROVIDERS[name]


def available_provider_types() -> list[str]:
    return sorted(_PROVIDERS.keys())

