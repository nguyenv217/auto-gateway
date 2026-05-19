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

