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
        alias: str | None = None,
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
            keys = prov.get_keys_for_alias(alias).copy()
            if shuffle:
                random.shuffle(keys)
            # If alias was specified and returned empty, skip this provider entirely.
            # The [None] fallback only makes sense when no alias was requested.
            if alias is not None and not keys:
                continue
            if not keys:
                keys = [None]

            targeted_models = self._prepare_models(pname, models, shuffle)
            for mname in targeted_models:
                # Failover should still attempt the originally requested model name
                # even if the provider does not advertise it.
                # In that case we fall back to an empty feature set.
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
                if any(self.models_match(models, avail_m) for avail_m in p_models)
            ]
            all_providers = list(self.providers.keys())

            # Failover-friendly behavior:
            # - Prefer providers that match the requested model.
            # - But if they all fail, also try the remaining providers.
            # This is required for tests where the first provider doesn't advertise the
            # requested model but still needs to be followed by a provider that can
            # handle it.
            if matched:
                remainder = [p for p in all_providers if p not in set(matched)]
                ordered = matched + remainder
                if shuffle:
                    # Shuffle within the two groups to avoid bias while still keeping
                    # matched providers first.
                    random.shuffle(matched)
                    random.shuffle(remainder)
                    ordered = matched + remainder
                return ordered

            # If no provider advertises the requested model, try everyone.
            if shuffle:
                random.shuffle(all_providers)
            return all_providers

        # No models specified: try all providers.
        all_providers = list(self.providers.keys())
        if shuffle:
            random.shuffle(all_providers)
        return all_providers



    def _prepare_models(self, pname: str, models: list[str] | None, shuffle: bool) -> list[str]:
        # If explicit models are requested, only keep the intersection when possible.
        # Otherwise (failover case), fall back to any model the provider supports.
        # This enables provider failover even when the originally requested model
        # isn't advertised by the next provider.
        if models:
            allowed = []
            for req_m in models:
                for avail_m in self.all_models.get(pname, {}):
                    if self.models_match([req_m], avail_m) and avail_m not in allowed:
                        allowed.append(avail_m)
                        
            if allowed:
                if shuffle:
                    random.shuffle(allowed)
                return allowed
            # Failover: provider doesn't advertise the requested model; use any
            # model it does support (if any).

        available = list(self.all_models.get(pname, {}).keys())

        if shuffle:
            random.shuffle(available)
        # As a final fallback, if the provider advertises no models at all,
        # still yield the requested model(s) (or a single None).
        if not available:
            return models[:] if models else [""]

        return available
