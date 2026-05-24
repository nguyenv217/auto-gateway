from __future__ import annotations

import math
import threading
from typing import Iterator

from .base import BaseStrategy
from ..providers.base import BaseProvider


class UCBBanditStrategy(BaseStrategy):
    """
    Multi-Armed Bandit routing using the UCB1 algorithm.
    Mathematically balances Exploration (trying unknown/recovered providers) 
    and Exploitation (sticking to the fastest, most reliable ones).
    """

    def __init__(self, providers: dict[str, BaseProvider], all_models: dict[str, dict[str, list[str]]]):
        self.providers = providers
        self.all_models = all_models
        self._lock = threading.RLock()
        
        # UCB State
        self.total_requests = 0
        self.arm_counts: dict[str, int] = {}       
        self.arm_rewards: dict[str, float] = {}      
        self.arm_failures: dict[str, int] = {}     

    def _hash_arm(self, pname: str, mname: str, key: str | None) -> str:
        return f"{pname}:{mname}:{str(key)[:20]}"

    def generate_targets(
        self,
        provider: str | None,
        models: list[str] | None,
        shuffle: bool,
        alias: str | None = None,
        message_hash: str | None = None,
        is_new_session: bool = False,
    ) -> Iterator[tuple[str, str, str | None, list[str]]]:
        
        candidates = []

        with self._lock:
            for pname, prov in self.providers.items():
                if provider and pname != provider:
                    continue
                
                keys = prov.get_keys_for_alias(alias)
                # If alias was specified and returned empty, skip this provider
                if alias is not None and not keys:
                    continue
                if not keys:
                    keys = [None]

                provider_models = self.all_models.get(pname, {})
                for mname, features in provider_models.items():
                    if models and not self.models_match(models, mname):
                        continue
                    
                    for key in keys:
                        arm_hash = self._hash_arm(pname, mname, key)
                        
                        # Soft circuit breaker: heavily penalize if recent failures
                        if self.arm_failures.get(arm_hash, 0) >= 3:
                            ucb_score = -999.0
                        else:
                            count = self.arm_counts.get(arm_hash, 0)
                            if count == 0:
                                ucb_score = float('inf')  # Always try completely untried arms first
                            else:
                                avg_reward = self.arm_rewards.get(arm_hash, 0.0) / count
                                
                                # C is the exploration factor. 1.5 balances well against our reward scale
                                exploration_term = 1.5 * math.sqrt(math.log(self.total_requests) / count)
                                ucb_score = avg_reward + exploration_term
                        
                        candidates.append({
                            "score": ucb_score,
                            "pname": pname,
                            "mname": mname,
                            "key": key,
                            "features": features
                        })

        # Sort by highest UCB score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        for c in candidates:
            yield c["pname"], c["mname"], c["key"], c["features"]

    def record_success(self, key, model: str, provider: str):
        with self._lock:
            arm_hash = self._hash_arm(provider, model, key)
            self.total_requests += 1
            self.arm_counts[arm_hash] = self.arm_counts.get(arm_hash, 0) + 1
            self.arm_failures[arm_hash] = 0
            
            # Base reward for success is +1.0 (Latency bonus added later)
            self.arm_rewards[arm_hash] = self.arm_rewards.get(arm_hash, 0.0) + 1.0

    def record_failure(self, key, model: str, provider: str, error_type: str = "unknown", message_hash: str | None = None):
        with self._lock:
            arm_hash = self._hash_arm(provider, model, key)
            self.total_requests += 1
            self.arm_counts[arm_hash] = self.arm_counts.get(arm_hash, 0) + 1
            self.arm_failures[arm_hash] = self.arm_failures.get(arm_hash, 0) + 1
            
            # Heavy penalty for failure to force the router to explore other arms
            self.arm_rewards[arm_hash] = self.arm_rewards.get(arm_hash, 0.0) - 2.0

    def record_latency(self, key, provider: str, model: str, latency_ms: float, **kwargs):
        with self._lock:
            arm_hash = self._hash_arm(provider, model, key)
            
            # Speed bonus: Maximize reward for low latency
            # e.g. 200ms yields +0.8 | 1000ms yields +0.0 | 2000ms yields -1.0
            latency_bonus = max(-1.0, 1.0 - (latency_ms / 1000.0))
            self.arm_rewards[arm_hash] = self.arm_rewards.get(arm_hash, 0.0) + latency_bonus
