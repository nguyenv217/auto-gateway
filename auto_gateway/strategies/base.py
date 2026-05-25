from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator, Optional


class BaseStrategy(ABC):
    """Select an ordered list of provider candidates."""

    @abstractmethod
    def generate_targets(
        self,
        provider: str | None,
        models: list[str] | None,
        shuffle: bool,
        alias: str | None = None,
        strict_alias: bool = True,
        message_hash: str | None = None,
        is_new_session: bool = False,
    ) -> Iterator[tuple[str, str, str | None, list[str]]]:
        """Yield: (provider_name, model_name, api_key, features)."""
        raise NotImplementedError

    def record_failure(self, *args: Any, **kwargs: Any) -> None:
        return

    def record_success(self, *args: Any, **kwargs: Any) -> None:
        return

    def record_latency(self, *args: Any, **kwargs: Any) -> None:
        return

    @staticmethod
    def normalize_model_name(name: str) -> str:
        """Strips provider prefixes (e.g., 'fireworks/llama-3' -> 'llama-3') and lowercases."""
        if "/" in name:
            name = name.split("/", 1)[-1]
        return name.lower().strip()

    @staticmethod
    def models_match(requested_models: list[str] | None, provider_model: str) -> bool:
        """Checks if the provider's model maps to any of the requested models."""
        if not requested_models:
            return True  # If no specific models requested, everything matches
        
        norm_provider = BaseStrategy.normalize_model_name(provider_model)
        for req_m in requested_models:
            # Match normalized (e.g. 'gpt-4o' == 'gpt-4o') or exact (in case prefixes were explicitly requested)
            if BaseStrategy.normalize_model_name(req_m) == norm_provider or req_m.lower().strip() == provider_model.lower().strip():
                return True
        return False
