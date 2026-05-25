# auto-gateway

**OpenAI-compatible API gateway with intelligent provider routing, failover, and tunneling.**

`auto-gateway` exposes a single `POST /v1/chat/completions` endpoint that transparently routes requests to multiple AI providers (OpenAI-compatible, Google Gemini, etc.) using configurable strategies. It supports streaming (SSE), tool calls, vision/media filtering, automatic failover, and public URL tunneling via ngrok or cloudflared.

---

## Table of Contents

- [Why auto-gateway?](#why-auto-gateway)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Routing Strategies](#routing-strategies)
- [Provider Architecture](#provider-architecture)
- [Network & Tunneling](#network--tunneling)
- [CLI Reference](#cli-reference)
- [Development](#development)
- [Testing](#testing)
- [Extending](#extending)

---

## Why auto-gateway?

- **Single OpenAI-compatible endpoint** — Drop-in replacement for OpenAI clients. No SDK changes needed.
- **Provider failover** — If one provider fails, automatically try the next.
- **Adaptive routing** — Latency-aware routing with circuit breakers and health tracking (optional).
- **Tunneling built-in** — Expose your local gateway publicly via ngrok or cloudflared with zero config.
- **Async everything** — Fully async stack (FastAPI + httpx) for high concurrency.
- **Extensible** — Add custom providers or routing strategies in minutes.

---

## Quick Start

```bash
# Install
pip install auto-gateway

# Create a config file
cp config.json.example config.json
# Edit config.json with your API keys

# Start the gateway
auto-gateway start --config config.json --port 8000

# Test it
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}],"stream":false}'
```

### Development install

```bash
git clone <repo>
cd auto-gateway
pip install -e ".[dev]"
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client (curl, SDK)                   │
│             POST /v1/chat/completions                   │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                    FastAPI Server                       │
│              core/server.py + core/models.py            │
│  ┌──────────────────────────────────────────────────┐   │
│  │          ProviderRouter (core/router.py)         │   │
│  │  - routes to provider via Strategy               │   │
│  │  - message filtering (vision/media/video)        │   │
│  │  - tool call SSE chunking                        │   │
│  │  - failover on exception                         │   │
│  └─────────────────────────┬────────────────────────┘   │
│                            │                            │
│                 ┌──────────▼───────┐                    │
│                 │  Strategy:       │                    │
│                 │  * Sequential    │                    │
│                 │  * Adaptive      │                    │
│                 │  * Bandit/UCB1   │                    │
│                 └──────────┬───────┘                    │
│                            │                            │
└────────────────────────────┼────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────┐
│                         Providers                        │
│  ┌─────────────────┐  ┌─────────────────┐                │
│  │ OpenAICompatible│  │   Google        │                │
│  │ (httpx.Async)   │  │ (genai thread)  │                │
│  └─────────────────┘  └─────────────────┘                │
└──────────────────────────────────────────────────────────┘
```

### Request flow

1. **Client** sends OpenAI-compatible JSON to `POST /v1/chat/completions`
2. **FastAPI server** validates the payload via Pydantic models
3. **ProviderRouter** delegates to the configured **Strategy** to obtain an ordered list of `(provider, model, key, features)` tuples
4. Router tries each target in order:
   - Calls `provider.call()` (non-streaming) or `provider.call_stream()` (streaming)
   - On success: records metrics and returns response
   - On failure: records error, tries next target
5. **Response** is formatted as an OpenAI-compatible JSON or SSE stream with `[DONE]` terminator

---

## Configuration

### config.json schema

```jsonc
{
  "server": {
    "host": "127.0.0.1",          // Bind address
    "port": 8000,                  // Port number
    "api_key": "my-awesome-api-key", // Server auth key (via `Authorization: Bearer`)
    "socket_path": null,           // UNIX socket path (optional, overrides host:port)
    "tunnel": "none"               // "none" | "ngrok" | "cloudflared"
  },
  "router": {
    "strategy": "adaptive",        // "sequential" | "adaptive" | "bandit"
    "retries": 1                   // Retries per key-provider-model pair
  },
  "providers": [
    {
      "type": "openai_compatible",  // Provider type
      "name": "local_openai",       // Unique name for routing
      "base_url": "http://localhost:8001/v1",  // API base URL
      "api_key": null,              // API key (or env var reference)
      "models": {                   // Model name -> features
        "gpt-4o-mini": ["vision", "tool_calls"],
        "gpt-4o": []
      },
      "extra_body": {}              // Extra params sent with every request
    },
    {
      "type": "google",
      "name": "gemini",
      "api_key": ["GOOGLE_API_KEY_1", "GOOGLE_API_KEY_2", ...],
      "models": {
        "gemini-1.5-flash": ["vision"]
      }
    }
  ],
  "extra": {
    "tunnels": {                    // Tunnel-specific config (optional)
      "ngrok_authtoken": "YOUR_NGROK_AUTHTOKEN",
      "cloudflared_binary": "cloudflared"
    }
  }
}
```

### API Key Formats

The `api_key` field supports three formats:

| Format | Example | Behavior |
|--------|---------|----------|
| **String** | `"sk-abc123"` | Single key — used for every request |
| **List** | `["sk-abc", "sk-def"]` | Keys are rotated for load balancing / failover (strategy-dependent) |
| **Dict (aliases)** | `{"us-east": "sk-abc", "us-west": "sk-def"}` | Keys are mapped to named aliases for per-request key selection |

#### Dict / Alias Format

When `api_key` is a dict, each key-value pair maps an **alias** to an **actual API key**:

```json
{
  "type": "openai_compatible",
  "name": "my_provider",
  "base_url": "https://api.openai.com/v1",
  "api_key": {
    "primary": "sk-abc123-primary-key",
    "secondary": "sk-def456-fallback-key",
    "customer_a": "sk-ghi789-customer-a-key"
  },
  "models": { "gpt-4o-mini": [] }
}
```

This allows clients to specify which key should be used on a per-request basis (see [API Reference: provider & alias params](#alias-feature)).

### Provider types

| Type | Class | Description |
|------|-------|-------------|
| `openai_compatible` | `OpenAICompatibleProvider` | Any OpenAI-compatible API (OpenAI, Anthropic via proxy, local vLLM, etc.) |
| `google` | `GoogleProvider` | Google Gemini via `google-genai` SDK |

### Model features

Features are strings that enable message filtering in the router:

| Feature | Effect |
|---------|--------|
| `vision` | Image content (`image_url`) is forwarded to provider |
| `media` | Media content is forwarded |
| `video_vision` | Video content is forwarded |
| `tool_calls` | Specify that this model supports tool calling |
| *(none)* | Image/media/video content is stripped from messages. No tool calling. |

---

## API Reference

### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint.

#### Request

```json
{
  "model": "gpt-4o-mini",
  "messages": [{"role": "user", "content": "Hello!"}],
  "temperature": 0.0,
  "stream": false,
  "tools": null,
  "tool_choice": null,
  "extra_body": {}
}
```

If `api_key` is specified in config.json's `server` config, pass header `Authorization: Bearer $api_key`.

#### Response (non-streaming)

```json
{
  "id": "chatcmpl_abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "gpt-4o-mini",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15
  }
}
```

#### Response (streaming)

Server-Sent Events stream:

```
data: {"id":"chatcmpl_xyz","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl_xyz","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl_xyz","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Alias Feature

When providers are configured with `api_key` as a **dict** (see [Configuration: API Key Formats](#api-key-formats)), you can specify which alias/key to use on each request by adding optional `provider` and `alias` fields to the request body:

```json
{
  "model": "gpt-4o-mini",
  "messages": [{"role": "user", "content": "Hello!"}],
  "provider": "my_provider",
  "alias": "primary"
}
```

| Field | Effect |
|-------|--------|
| `provider: string` | If set, only the named provider is considered. Skips all other providers. |
| `alias: string` | If set, only the key mapped to this alias is used for the chosen provider(s). |
| `strict_alias: bool` | Default `true`. When `true`, only the aliased key is tried — no fallback to other keys in the same provider. When `false`, if the aliased key fails, the strategy falls back to other keys from the same provider before moving to the next provider. |

#### Behavior Details

1. **`provider` only** — routes exclusively to that provider (with all its keys available for rotation/failover)
2. **`alias` only** — only providers that have a key registered under that alias are considered; within each matching provider, only the specific aliased key is used (subject to `strict_alias`)
3. **`provider` + `alias`** — routes to the specific provider and pins to the exact key for that alias
4. **Neither** — default behavior: all providers and all keys are eligible (original behavior)

#### Failover Behavior with Alias

- **Within the same provider**: when an alias is specified with `strict_alias=true` (default), the strategy yields exactly **one key** per provider. If that key fails, the router does NOT try other keys within the same provider — it moves to the next candidate provider (if any).
- **With `strict_alias=false`**: if the aliased key fails, the strategy falls back to the other keys of the same provider before moving to the next provider.
- **Across providers**: if multiple providers define the **same alias name**, the router will try them in strategy order. If the first provider's aliased key fails, the next provider with the same alias is attempted.
- **If no provider has the alias**: the request results in `AllProvidersExhaustedError` (no targets matched).

Example: two providers with the same alias:

```json
// config.json
{
  "providers": [
    {
      "name": "fallback_provider",
      "type": "openai_compatible",
      "base_url": "https://api.openai.com/v1",
      "api_key": { "primary": "sk-abc" },
      "models": { "gpt-4o-mini": [] }
    },
    {
      "name": "backup_provider",
      "type": "openai_compatible",
      "base_url": "https://backup.example.com/v1",
      "api_key": { "primary": "sk-xyz" },
      "models": { "gpt-4o-mini": [] }
    }
  ]
}
```

```json
// Request – if fallback_provider fails, backup_provider is tried next
{ "model": "gpt-4o-mini", "messages": [...], "alias": "primary" }
```

### `GET /v1/models`

Returns list of available models aggregated from all providers.

#### Response
```json
{
  "object": "list",
  "data": [
    { "id": "gpt-4o-mini", "object": "model", "created": 1700000000, "owned_by": "openai" },
    { "id": "gemini-1.5-flash", "object": "model", "created": 1700000000, "owned_by": "google" }
  ]
}
```

### `GET /health`

#### Response
```json
{
  "status": "ok",
  "providers": 2
}
```

#### Error handling

| Scenario | Status | Behavior |
|----------|--------|----------|
| All providers fail | 200 | Returns empty content `""` with `finish_reason: "stop"` |
| Invalid payload | 422 | FastAPI validation error |
| Provider timeout | — | Falls through to next provider automatically |

---

## Routing Strategies

### Sequential Strategy

`auto_gateway/strategies/sequential.py`

Simple ordered rotation. Providers are tried in the order they appear in `all_models`. If a provider fails, the next one in sequence is attempted.

Configuration: `"strategy": "sequential"`

### Adaptive Strategy

`auto_gateway/strategies/adaptive.py`

Health-aware routing with:

- **Health scoring**: Combines success rate (40%), average latency (30%), and stability (20%) for a `health_score`
- **Circuit breakers**: After `circuit_threshold` consecutive failures, a provider is temporarily skipped
- **Per-error backoff**: Rate limits, auth errors, and quotas have independent backoff timers with configurable delays and multipliers
- **Latency tracking**: Rolling window of latency samples for scoring
- **Persistence**: Health state can be persisted to disk (optional, via `persistence_path`)
- **Small model preference**: Models in `_SMALL_MODELS` list get a routing bonus

Configuration: `"strategy": "adaptive"`

> **Note**: Adaptive strategy is ported from the `callai` project and may have additional configuration knobs exposed in the future.

### Bandit / UCB1 Strategy

`auto_gateway/strategies/bandit.py`

Multi-armed bandit routing using the UCB1 algorithm. Mathematically balances exploring under-tested keys/providers against exploiting those with historical high reliability and low latency.

Configuration: `"strategy": "bandit"`

---

## Provider Architecture

### Built-in providers

#### `OpenAICompatibleProvider` (`providers/openai_compatible.py`)

- Uses `httpx.AsyncClient` for async HTTP
- Supports both `call()` and `call_stream()`
- Passes headers, tools, tool_choice, and extra_body
- Subclass `OpenAIProvider` preconfigured for `https://api.openai.com/v1`

#### `GoogleProvider` (`providers/google.py`)

- Uses `google-genai` SDK via `asyncio.to_thread()` for synchronous execution
- Supports system instructions, multimodal content (images), function calling
- Returns normalized `ProviderCallResult` with text, reasoning, tool_calls, usage

### Provider interface

All providers extend `BaseProvider` (`providers/base.py`):

```python
class BaseProvider(ABC):
    def __init__(self, name: str, keys: list[str] | None, models: dict[str, list[str]], key_aliases: dict[str, str] | None = None):
        ...

    @abstractmethod
    async def call(self, *, key: str, model: str, messages: list[ChatMessage], timeout: float, tools: Optional[list[dict[str, Any]]] = None, tool_choice: str, extra_body: dict[str, Any] = None) -> ProviderCallResult:
        """Non-streaming call. Returns ProviderCallResult TypedDict."""

    async def call_stream(self, *, key, model, messages, timeout, tools, tool_choice, extra_body=None) -> AsyncIterator[BaseProviderDelta]:
        """Streaming call. Yields delta dicts with type/content/finish_reason/tool_calls fields."""
```

### Provider registry (`providers/registry.py`)

```python
from auto_gateway.providers.registry import register_provider, get_provider_factory

@register_provider("my_custom")
def create_my_provider(config) -> BaseProvider:
    ...
```

---

## Network & Tunneling

### Local server

Default: `http://127.0.0.1:8000`

The gateway supports binding to a **UNIX domain socket** instead of TCP:

```json
{
  "server": {
    "socket_path": "/tmp/gateway.sock",
    "host": "127.0.0.1",
    "port": 8000
  }
}
```

If `socket_path` is provided, the server binds to the socket instead of TCP.

### ngrok tunnel

```bash
auto-gateway start --config config.json --tunnel ngrok
```

Requires `NGROK_AUTHTOKEN` environment variable or configured in `config.json` under `extra.tunnels.ngrok_authtoken`.

### cloudflared tunnel

```bash
auto-gateway start --config config.json --tunnel cloudflared
```

Requires `cloudflared` binary on `PATH` (or configured in `config.json` under `extra.tunnels.cloudflared_binary`).

The public URL is extracted from the `*.trycloudflare.com` output and logged at startup.

### Tunnel info

```python
from auto_gateway.network.hosting import TunnelInfo

info = TunnelInfo(public_url="https://abc123.ngrok.io", backend="ngrok")
```

---

## CLI Reference

```bash
auto-gateway [OPTIONS] COMMAND [ARGS]
```

### `start`

Start the gateway server.

```bash
auto-gateway start --config config.json [--host 0.0.0.0] [--port 8000] [--tunnel none]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | (required) | Path to config.json |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8000` | Port number |
| `--tunnel` | `none` | Tunnel backend: `none`, `ngrok`, or `cloudflared` |

### `check`

Validate configuration and print provider summary.

```bash
auto-gateway check --config config.json
# Output:
# OK: providers=2 strategy=adaptive tunnel=none
# - local_openai: type=openai_compatible, models=['gpt-4o-mini']
# - gemini: type=google, models=['gemini-1.5-flash']
```

### `save-global`

Save your specified configuration to ~/.auto-gateway/config.json.

```bash
auto-gateway save-global --config config.json

```

Afterward, you can start without specifying `--config`, i.e. `auto-gateway start`.


### `version`

Print version.

```bash
auto-gateway version
# auto-gateway 0.1.0
```

---

## Development

### Project structure

```
auto-gateway/
├── auto_gateway/
│   ├── __init__.py
│   ├── cli/
│   │   └── main.py              # Typer CLI commands
│   ├── config/
│   │   ├── manager.py           # Config file loading
│   │   └── schema.py            # Pydantic config models
│   ├── core/
│   │   ├── models.py            # OpenAI API request/response models
│   │   ├── router.py            # ProviderRouter with route/route_stream
│   │   ├── router_tool_calls_helpers.py  # Tool call SSE chunking
│   │   ├── router_toolcalls_patch.py     # Re-exports
│   │   ├── sse_repair.py        # SSE stream repair utilities
│   │   └── server.py            # FastAPI application setup
│   ├── network/
│   │   ├── hosting.py           # start_ngrok, start_cloudflared, start_tunnel
│   │   ├── hosting_test_utils.py
│   │   ├── tunnels.py
│   │   └── uvicorn_runner.py    # UDS/TCP app runner
│   ├── providers/
│   │   ├── base.py              # BaseProvider ABC
│   │   ├── google.py            # Google provider
│   │   ├── openai_compatible.py # OpenAI-compatible provider
│   │   └── registry.py          # Provider factory registry
│   └── strategies/
│       ├── adaptive.py          # Health-aware routing
│       ├── bandit.py            # UCB1 bandit routing
│       ├── base.py              # BaseStrategy ABC
│       └── sequential.py        # Ordered rotation
├── tests/
│   ├── test_alias_feature.py              # 24 alias tests
│   ├── test_comprehensive_api.py          # 19 comprehensive tests
│   ├── test_openai_streaming_delta_shapes.py # SSE delta validation
│   ├── test_simple_smoke.py
│   ├── test_smoke_server.py               # End-to-end smoke test
│   ├── test_sse_repair.py                 # SSE repair tests
│   ├── test_streaming_and_failover.py      # Streaming + failover
│   └── test_tunnel_url_parsing.py          # Cloudflared URL parsing
├── config.json.example
├── pyproject.toml
└── README.md
```


### Adding a new provider

1. Create `auto_gateway/providers/my_provider.py`:

```python
from .base import BaseProvider, ProviderCallResult

class MyProvider(BaseProvider):
    def __init__(self, keys, models, **kwargs):
        super().__init__(name="my", keys=keys, models=models)
        # Custom init

    async def call(self, *, key, model, messages, timeout, tools, tool_choice, extra_body=None):
        # Implement async call
        return ProviderCallResult(text=..., reasoning=..., tool_calls=..., usage=...)

    async def call_stream(self, *, key, model, messages, timeout, tools, tool_choice, extra_body=None):
        # Yield BaseProviderDelta dicts
        yield {"type": "content", "content": "..."}
        yield {"type": "finish", "finish_reason": "stop"}
```

2. Register in the provider factory:

```python
from .registry import register_provider

@register_provider("my")
def create_my_provider(config):
    return MyProvider(
        keys=[config.api_key],
        models=config.models,
    )
```

3. Add to `config/schema.py` as a new `ProviderBaseConfig` variant if needed.

### Adding a new strategy

1. Create `auto_gateway/strategies/my_strategy.py` extending `BaseStrategy`:

```python
from .base import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, providers, all_models):
        self.providers = providers
        self.all_models = all_models

    def generate_targets(self, provider, models, shuffle, alias=None, message_hash=None, is_new_session=False):
        # Yield (provider_name, model_name, api_key, features)
        ...
```

2. Wire it in `cli/main.py` and `config/schema.py`.

### Streaming delta protocol

Providers communicate streaming events to the router via `BaseProviderDelta` dicts:

```python
# Text content delta
{"type": "content", "content": "Hello"}

# Tool call delta (OpenAI-compatible)
{"type": "tool_calls", "index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": "{}"}}

# Finish signal
{"type": "finish", "finish_reason": "stop"}
```

The router translates these into OpenAI SSE `data: {...}\n\n` chunks with `[DONE]` termination.

---

## Extending

### Custom tunnel backends

Implement in `auto_gateway/network/hosting.py`:

```python
@dataclass
class TunnelInfo:
    public_url: str
    backend: str

async def start_my_tunnel(port: int, config: dict) -> TunnelInfo:
    ...
```

Wire in `start_tunnel()` and the CLI `--tunnel` option.

### Custom config formats

The `config/manager.py` loads JSON. For YAML or TOML support, add a format detector and parser there.

### Middleware / hooks

FastAPI middleware can be added directly in `core/server.py`:

```python
app = FastAPI()
app.add_middleware(MyMiddleware, ...)
```

---

## License

MIT
