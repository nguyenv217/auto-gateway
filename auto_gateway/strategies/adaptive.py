from __future__ import annotations

# Ported from callai.rotator.strategies.adaptive (sync candidate generation + health tracking)

import random
import threading
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator
import json

from ..providers.base import BaseProvider
from .base import BaseStrategy


_SMALL_MODELS = [
    "llama-3.1-8b-instant",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "liquid/lfm-2.5-1.2b-thinking:free",
    "llama3.1-8b",
]


class ErrorType(Enum):
    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    QUOTA = "quota"
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"
    NETWORK = "network"
    TOOLS_UNSUPPORTED = "tools_unsupported"
    MEDIA_UNSUPPORTED = "media_unsupported"
    UNKNOWN = "unknown"


@dataclass
class PerErrorBackoffState:
    retry_count: int = 0
    last_delay: float = 0
    session_stopped: bool = False
    next_retry_time: float = 0


@dataclass
class ErrorBackoffConfig:
    initial_delay: float
    max_delay: float
    multiplier: float
    max_retries: int
    circuit_threshold: int
    recovery_time: int
    stop_for_session: bool


@dataclass
class HealthMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    latency_samples: list = field(default_factory=list)
    error_counts: dict = field(default_factory=dict)
    last_success_time: float = 0
    last_failure_time: float = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    error_backoff_state: dict[str, PerErrorBackoffState] = field(default_factory=dict)
    rate_limit_messages: dict[str, int] = field(default_factory=dict)
    rate_limited_this_session: bool = False

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.5
        return self.successful_requests / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        if not self.latency_samples:
            return 1000
        return statistics.mean(self.latency_samples)

    @property
    def health_score(self) -> float:
        success_score = self.success_rate * 40
        latency_score = max(0, 30 - (self.avg_latency_ms / 10000) * 30)
        stability_score = 20 if self.consecutive_successes >= 3 else max(0, 20 - self.consecutive_failures * 5)
        recency_score = 10
        if self.last_failure_time > self.last_success_time:
            time_since_failure = time.time() - self.last_failure_time
            if time_since_failure < 60:
                recency_score = 5
            elif time_since_failure < 300:
                recency_score = 7
        return success_score + latency_score + stability_score + recency_score


@dataclass
class CircuitState:
    failure_count: int = 0
    last_failure_time: float = 0
    is_open: bool = False
    next_retry_time: float = 0
    half_open_successes: int = 0

    def record_failure(self, threshold: int = 5, recovery_time: int = 60):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= threshold:
            self.is_open = True
            self.next_retry_time = time.time() + recovery_time

    def record_success(self):
        if self.is_open:
            self.half_open_successes += 1
            if self.half_open_successes >= 3:
                self.reset()
        else:
            self.failure_count = 0

    def reset(self):
        self.failure_count = 0
        self.is_open = False
        self.next_retry_time = 0
        self.half_open_successes = 0

    def should_attempt(self) -> bool:
        if not self.is_open:
            return True
        if time.time() >= self.next_retry_time:
            self.is_open = False
            return True
        return False


class AdaptiveStrategy(BaseStrategy):
    CIRCUIT_FAILURE_THRESHOLD = 5
    CIRCUIT_RECOVERY_TIME = 60

    HEALTH_EXCELLENT = 80
    HEALTH_GOOD = 60
    HEALTH_FAIR = 40
    HEALTH_POOR = 20

    WEIGHT_EXCELLENT = 2.0
    WEIGHT_GOOD = 1.5
    WEIGHT_FAIR = 1.0
    WEIGHT_POOR = 0.5

    LATENCY_FAST = 500
    LATENCY_MEDIUM = 2000
    LATENCY_SLOW = 5000

    ERROR_BACKOFF_CONFIGS = {
        ErrorType.RATE_LIMIT: ErrorBackoffConfig(
            initial_delay=5,
            max_delay=60,
            multiplier=2.0,
            max_retries=5,
            circuit_threshold=5,
            recovery_time=30,
            stop_for_session=False,
        ),
        ErrorType.AUTH: ErrorBackoffConfig(
            initial_delay=10.0,
            max_delay=300,
            multiplier=1.5,
            max_retries=3,
            circuit_threshold=3,
            recovery_time=120,
            stop_for_session=False,
        ),
        ErrorType.QUOTA: ErrorBackoffConfig(
            initial_delay=60.0,
            max_delay=3600,
            multiplier=2.0,
            max_retries=2,
            circuit_threshold=2,
            recovery_time=300,
            stop_for_session=True,
        ),
        ErrorType.TIMEOUT: ErrorBackoffConfig(
            initial_delay=5.0,
            max_delay=120,
            multiplier=1.5,
            max_retries=5,
            circuit_threshold=5,
            recovery_time=60,
            stop_for_session=False,
        ),
        ErrorType.NETWORK: ErrorBackoffConfig(
            initial_delay=1.0,
            max_delay=30,
            multiplier=2.0,
            max_retries=10,
            circuit_threshold=8,
            recovery_time=15,
            stop_for_session=False,
        ),
        ErrorType.TOOLS_UNSUPPORTED: ErrorBackoffConfig(
            initial_delay=0.0,
            max_delay=0.0,
            multiplier=1.0,
            max_retries=0,
            circuit_threshold=0,
            recovery_time=0,
            stop_for_session=False,
        ),
        ErrorType.MEDIA_UNSUPPORTED: ErrorBackoffConfig(
            initial_delay=0.0,
            max_delay=0.0,
            multiplier=1.0,
            max_retries=0,
            circuit_threshold=0,
            recovery_time=0,
            stop_for_session=False,
        ),
    }

    def __init__(
        self,
        providers: dict[str, BaseProvider],
        all_models: dict[str, dict[str, list[str]]],
        persistence_path: str | None = None,
        enable_circuit_breaker: bool = True,
        enable_latency_aware: bool = True,
        enable_cost_aware: bool = True,
    ):
        self.providers = providers
        self.all_models = all_models
        self.persistence_path = persistence_path

        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_latency_aware = enable_latency_aware
        self.enable_cost_aware = enable_cost_aware

        self.health_registry: dict[str, HealthMetrics] = {}
        self.circuit_breakers: dict[str, CircuitState] = {}

        self._lock = threading.RLock()

        if persistence_path:
            self._load_state()

        for pname in providers.keys():
            self.circuit_breakers.setdefault(pname, CircuitState())

    def generate_targets(
        self,
        provider: str | None,
        models: list[str] | None,
        shuffle: bool,
        message_hash: str | None = None,
        is_new_session: bool = False,
    ) -> Iterator[tuple[str, str, str | None, list[str]]]:
        error_container: list[str] = []

        with self._lock:
            if is_new_session:
                self._on_new_session()
            candidates = self._build_candidates(provider, models, error_container, message_hash)

        if not candidates:
            return iter([])

        weighted_candidates = self._calculate_weights(candidates)
        if shuffle:
            random.shuffle(weighted_candidates)

        weighted_candidates.sort(key=lambda x: x["weight"], reverse=True)

        for c in weighted_candidates:
            yield c["pname"], c["model"], c["key"], c["features"]

    def _build_candidates(self, provider, models, error_container, message_hash: str | None = None):
        candidates = []

        has_cloudflare = [m in self.all_models.get("cloudflare", {}) for m in models] if models else []
        if models and any(has_cloudflare) and not all(has_cloudflare):
            error_container.append("Please specify only either text-based or image-generating models at once")
            return []

        for pname, prov in self.providers.items():
            if provider:
                if pname != provider:
                    continue
            # blocked providers/models from original; keep minimal for now
            elif pname in {"cloudflare", "debug"}:
                if not models or not any(m in self.all_models.get(pname, {}) for m in models):
                    continue

            if self.enable_circuit_breaker:
                circuit = self.circuit_breakers.get(pname, CircuitState())
                if not circuit.should_attempt():
                    continue

            provider_models = self.all_models.get(pname, {})
            for mname, features in provider_models.items():
                if models and mname not in models:
                    continue

                keys = prov.get_keys() or [None]
                for key in keys:
                    key_hash = self._hash_key(pname, mname, key)

                    if self._should_backoff_any(key_hash):
                        continue

                    if message_hash and self._is_message_too_large(pname, mname, message_hash):
                        continue

                    metrics = self.health_registry.setdefault(key_hash, HealthMetrics())
                    candidates.insert(self._get_index(models, mname), {
                        "provider": prov,
                        "pname": pname,
                        "model": mname,
                        "key": key,
                        "features": features,
                        "metrics": metrics,
                        "key_hash": key_hash,
                    })

        return candidates

    def _should_backoff_any(self, key_hash: str) -> bool:
        for et in ErrorType:
            if self.should_backoff(key_hash, et):
                return True
        return False

    def _get_index(self, L, entry):
        if not L:
            return -1
        for i, e in enumerate(L):
            if e == entry:
                return i
        return -1

    def _calculate_weights(self, candidates):
        weighted = []
        for c in candidates:
            metrics: HealthMetrics = c["metrics"]
            health_score = metrics.health_score

            if health_score >= self.HEALTH_EXCELLENT:
                base_weight = self.WEIGHT_EXCELLENT
            elif health_score >= self.HEALTH_GOOD:
                base_weight = self.WEIGHT_GOOD
            elif health_score >= self.HEALTH_FAIR:
                base_weight = self.WEIGHT_FAIR
            else:
                base_weight = self.WEIGHT_POOR

            if self.enable_latency_aware:
                base_weight *= self._get_latency_factor(metrics.avg_latency_ms)

            if self.enable_cost_aware:
                base_weight *= self._get_cost_factor(c["model"])

            if metrics.consecutive_successes >= 2:
                base_weight *= 1.2
            if metrics.consecutive_failures >= 1:
                base_weight *= 0.8
            if c["model"] in _SMALL_MODELS:
                base_weight *= 0.3

            weighted.append({**c, "weight": base_weight})
        return weighted

    def _get_latency_factor(self, latency_ms: float) -> float:
        if latency_ms < self.LATENCY_FAST:
            return 1.2
        if latency_ms < self.LATENCY_MEDIUM:
            return 1.0
        if latency_ms < self.LATENCY_SLOW:
            return 0.8
        return 0.6

    def _get_cost_factor(self, model: str) -> float:
        return 1.0

    def record_failure(self, key, models, provider, error_type: str = "unknown", message_hash: str | None = None):
        with self._lock:
            key_hash = self._find_key_hash(provider, key)
            if not key_hash:
                return

            metrics = self.health_registry.get(key_hash)
            if not metrics:
                return

            metrics.total_requests += 1
            metrics.failed_requests += 1
            metrics.consecutive_failures += 1
            metrics.consecutive_successes = 0
            metrics.last_failure_time = time.time()

            et = ErrorType.UNKNOWN
            try:
                # error_type already bucketed or plain; keep heuristic minimal
                et = ErrorType(error_type)
            except Exception:
                pass

            if et == ErrorType.RATE_LIMIT and message_hash:
                metrics.rate_limited_this_session = True
                metrics.rate_limit_messages[message_hash] = metrics.rate_limit_messages.get(message_hash, 0) + 1

            delay = self._get_backoff_delay(key_hash, et)
            if delay is not None:
                pass

            if self.enable_circuit_breaker:
                circuit = self.circuit_breakers.get(provider)
                if circuit:
                    circuit.record_failure(threshold=self.CIRCUIT_FAILURE_THRESHOLD, recovery_time=self.CIRCUIT_RECOVERY_TIME)

            self._save_state()

    def record_success(self, key, models, provider):
        with self._lock:
            key_hash = self._find_key_hash(provider, key)
            if not key_hash:
                return
            metrics = self.health_registry.get(key_hash)
            if metrics:
                metrics.total_requests += 1
                metrics.successful_requests += 1
                metrics.consecutive_successes += 1
                metrics.consecutive_failures = 0
                metrics.last_success_time = time.time()
                metrics.error_backoff_state.clear()

            if self.enable_circuit_breaker:
                circuit = self.circuit_breakers.get(provider)
                if circuit:
                    circuit.record_success()

            self._save_state()

    def record_latency(self, key, provider, model, latency_ms: float, **kwargs):
        with self._lock:
            key_hash = self._find_key_hash(provider, key)
            if not key_hash:
                return
            metrics = self.health_registry.get(key_hash)
            if metrics:
                metrics.total_latency_ms += latency_ms
                metrics.latency_samples.append(latency_ms)
                if len(metrics.latency_samples) > 100:
                    metrics.latency_samples = metrics.latency_samples[-100:]

    def should_backoff(self, key_hash: str, error_type: ErrorType) -> bool:
        metrics = self.health_registry.get(key_hash)
        if not metrics:
            return False
        state = metrics.error_backoff_state.get(error_type.value)
        if not state:
            return False
        if state.session_stopped:
            return True
        if time.time() < state.next_retry_time:
            return True
        return False

    def _get_backoff_delay(self, key_hash: str, error_type: ErrorType) -> float | None:
        metrics = self.health_registry.get(key_hash)
        if not metrics:
            return None

        config = self.ERROR_BACKOFF_CONFIGS.get(error_type)
        if not config:
            return None

        state = metrics.error_backoff_state.get(error_type.value)
        if not state:
            state = PerErrorBackoffState()
            metrics.error_backoff_state[error_type.value] = state

        if state.session_stopped:
            return None
        if state.retry_count >= config.max_retries:
            if config.stop_for_session:
                state.session_stopped = True
            return None

        delay = config.initial_delay * (config.multiplier ** state.retry_count)
        delay = min(delay, config.max_delay)

        state.retry_count += 1
        state.last_delay = delay
        state.next_retry_time = time.time() + delay

        return delay

    def _is_message_too_large(self, pname: str, mname: str, message_hash: str) -> bool:
        for k, metrics in self.health_registry.items():
            if k.startswith(f"{pname}:{mname}:"):
                if metrics.rate_limited_this_session:
                    return True
                if message_hash in metrics.rate_limit_messages and metrics.rate_limit_messages[message_hash] >= 2:
                    return True
        return False

    def _on_new_session(self):
        for metric in self.health_registry.values():
            metric.rate_limited_this_session = False

    def _hash_key(self, pname: str, mname: str, key) -> str:
        key_str = str(key) if key else "none"
        return f"{pname}:{mname}:{key_str[:20]}"

    def _find_key_hash(self, provider: str, key) -> str | None:
        for hash_key in self.health_registry.keys():
            if hash_key.startswith(f"{provider}:"):
                key_str = str(key) if key else "none"
                if hash_key.endswith(key_str[:20]):
                    return hash_key
        return None

    def _save_state(self):
        if not self.persistence_path:
            return
        try:
            state = {
                "health_registry": {
                    k: {
                        "total_requests": v.total_requests,
                        "successful_requests": v.successful_requests,
                        "failed_requests": v.failed_requests,
                        "total_latency_ms": v.total_latency_ms,
                        "latency_samples": v.latency_samples,
                        "error_backoff_state": {ek: {
                            "retry_count": s.retry_count,
                            "last_delay": s.last_delay,
                            "session_stopped": s.session_stopped,
                            "next_retry_time": s.next_retry_time,
                        } for ek, s in v.error_backoff_state.items()},
                        "rate_limit_messages": v.rate_limit_messages,
                        "rate_limited_this_session": v.rate_limited_this_session,
                        "last_success_time": v.last_success_time,
                        "last_failure_time": v.last_failure_time,
                        "consecutive_failures": v.consecutive_failures,
                        "consecutive_successes": v.consecutive_successes,
                        "error_counts": v.error_counts,
                    }
                    for k, v in self.health_registry.items()
                },
                "circuit_breakers": {
                    k: {
                        "failure_count": v.failure_count,
                        "last_failure_time": v.last_failure_time,
                        "is_open": v.is_open,
                        "next_retry_time": v.next_retry_time,
                        "half_open_successes": v.half_open_successes,
                    }
                    for k, v in self.circuit_breakers.items()
                },
                "timestamp": time.time(),
            }
            path = Path(self.persistence_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_state(self):
        if not self.persistence_path:
            return
        try:
            path = Path(self.persistence_path)
            if not path.exists():
                return
            state = json.loads(path.read_text(encoding="utf-8"))
            for k, v in state.get("health_registry", {}).items():
                hm = HealthMetrics(
                    total_requests=v.get("total_requests", 0),
                    successful_requests=v.get("successful_requests", 0),
                    failed_requests=v.get("failed_requests", 0),
                    total_latency_ms=v.get("total_latency_ms", 0.0),
                    latency_samples=v.get("latency_samples", []),
                    rate_limit_messages=v.get("rate_limit_messages", {}),
                    rate_limited_this_session=v.get("rate_limited_this_session", False),
                    last_success_time=v.get("last_success_time", 0),
                    last_failure_time=v.get("last_failure_time", 0),
                    consecutive_failures=v.get("consecutive_failures", 0),
                    consecutive_successes=v.get("consecutive_successes", 0),
                    error_counts=v.get("error_counts", {}),
                )
                for ek, s in v.get("error_backoff_state", {}).items():
                    hm.error_backoff_state[ek] = PerErrorBackoffState(
                        retry_count=s.get("retry_count", 0),
                        last_delay=s.get("last_delay", 0),
                        session_stopped=s.get("session_stopped", False),
                        next_retry_time=s.get("next_retry_time", 0),
                    )
                self.health_registry[k] = hm

            for k, v in state.get("circuit_breakers", {}).items():
                self.circuit_breakers[k] = CircuitState(
                    failure_count=v.get("failure_count", 0),
                    last_failure_time=v.get("last_failure_time", 0),
                    is_open=v.get("is_open", False),
                    next_retry_time=v.get("next_retry_time", 0),
                    half_open_successes=v.get("half_open_successes", 0),
                )
        except Exception:
            pass

