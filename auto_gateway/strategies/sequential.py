from __future__ import annotations

import random
from typing import Iterator

from .base import BaseStrategy
from ..providers.base import BaseProvider


class SequentialStrategy(BaseStrategy):
    """Simple ordered rotation."""

    def __init__(self, providers: dict[str, BaseProvider], all_models: dict[str, dict[str, list[str]]]):
        self.providers = providers
        self.all_models = all_models

    def generate_targets(
        self,
        provider: str | None,
        models: list[str] | None,
        shuffle: bool,
        message_hash: str | None = None,
        is_new_session: bool = False,
    ) -> Iterator[tuple[str, str, str | None, list[str]]]:
        del message_hash, is_new_session

        error_container: list[str] = []

        target_providers = self._select_providers(provider, models, shuffle, error_container)
        if not target_providers:
            return iter([])

        for pname in target_providers:
            prov = self.providers.get(pname)
            if not prov:
                continue
            keys = prov.get_keys().copy()
            if shuffle:
                random.shuffle(keys)
            if not keys:
                keys = [None]

            targeted_models = self._prepare_models(pname, models, shuffle)
            for mname in targeted_models:
                if mname not in self.all_models.get(pname, {}):
                    continue
                features = self.all_models[pname].get(mname, [])
                for key in keys:
                    yield pname, mname, key, features

    def _select_providers(self, provider: str | None, models: list[str] | None, shuffle: bool, error_container: list[str]):
        if provider:
            if provider in self.providers:
                return [provider]
            error_container.append(f"Provider '{provider}' unrecognized")

        if models:
            matched = [
                pname
                for pname, p_models in self.all_models.items()
                if any(m in p_models for m in models)
            ]
            if matched:
                if shuffle:
                    random.shuffle(matched)
                return matched

        all_providers = list(self.providers.keys())
        if shuffle:
            random.shuffle(all_providers)
        return all_providers

    def _prepare_models(self, pname: str, models: list[str] | None, shuffle: bool) -> list[str]:
        if models:
            allowed = [m for m in models if m in self.all_models.get(pname, {})]
            if allowed:
                if shuffle:
                    random.shuffle(allowed)
                return allowed

        available = list(self.all_models.get(pname, {}).keys())
        if shuffle:
            random.shuffle(available)
        return available

