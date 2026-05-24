from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypedDict, AsyncIterator, Literal, Optional


class ProviderCallResult(TypedDict):
    text: str | None
    reasoning: str | None
    tool_calls: list[dict[str, Any]] | None
    usage: dict[str, Any] | None


class BaseProviderDelta(TypedDict, total=False):
    # Structured streaming delta events to be translated into OpenAI SSE chunks by the router.
    # type:
    # - content: incremental assistant text
    # - tool_calls: incremental function_call arguments / tool call deltas
    # - finish: indicates completion
    type: Literal["content", "tool_calls", "finish"]
    content: str
    finish_reason: Optional[str]

    # tool_calls event fields (OpenAI compatible)
    index: int
    id: Optional[str]
    function: dict[str, Any]


class BaseProvider(ABC):
    def __init__(
        self,
        name: str,
        keys: list[str] | None,
        models: dict[str, list[str]],
        key_aliases: dict[str, str] | None = None,
    ):
        self.name = name
        self._keys = keys or []
        self._models = models or {}
        self._key_aliases = key_aliases or {}

    def get_keys(self) -> list[str]:
        return list(self._keys)

    def get_all_models(self) -> dict[str, list[str]]:
        return self._models

    def get_model_features(self, model: str) -> list[str]:
        return self._models.get(model, [])

    def get_sticky_id(self, key: str | None) -> str:
        return (key or "")[0:8] or "default"

    def get_keys_for_alias(self, alias: str | None) -> list[str]:
        """Resolve keys for a given alias.
        
        If alias is None, returns all keys (existing rotation behavior).
        If alias is specified, looks up the matching key in key_aliases.
        Returns [key] if found, else [] (provider skipped).
        """
        if alias is None:
            return list(self._keys)
        key = self._key_aliases.get(alias)
        if key is not None:
            return [key]
        return []

    @abstractmethod
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
        raise NotImplementedError

    async def call_stream(
        self,
        *,
        key: str | None,
        model: str,
        messages: list[dict[str, Any]],
        timeout: float,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncIterator[BaseProviderDelta]:
        """Yield structured streaming delta events.

        Default implementation falls back to `call()` and yields the whole text once.
        """
        res = await self.call(
            key=key,
            model=model,
            messages=messages,
            timeout=timeout,
            tools=tools,
            tool_choice=tool_choice,
            extra_body=extra_body,
        )
        text = res.get("text") or ""
        if text:
            yield {"type": "content", "content": text}
        yield {"type": "finish", "finish_reason": "stop"}
