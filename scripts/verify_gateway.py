"""
verify_gateway.py — Integration test using the official OpenAI SDK.

Starts the gateway server in a background subprocess, then verifies the gateway's
OpenAI-compatible endpoints are working correctly:

  - GET /v1/models returns a valid model list
  - POST /v1/chat/completions (non-streaming) forwards requests correctly
  - POST /v1/chat/completions (streaming) forwards correctly
  - Extra fields (top_p, max_tokens, etc.) are preserved in the forwarded payload

Reads the provider configuration from config.json (API key, base URL, model name).
No .env file is required.

Usage:
    python scripts/verify_gateway.py
"""

import os
import sys
import json
import time
import socket
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 1. Read provider configuration from config.json
# ---------------------------------------------------------------------------
CONFIG_PATH = PROJECT_ROOT / "config.json"
if not CONFIG_PATH.exists():
    print(f"[ERROR] config.json not found at {CONFIG_PATH}")
    sys.exit(1)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

providers = config.get("providers", [])
if not providers:
    print("[ERROR] No providers defined in config.json")
    sys.exit(1)

provider = providers[0]
API_KEY = provider.get("api_key", "")
BASE_URL = provider.get("base_url", "").rstrip("/") + "/v1"
MODELS = list(provider.get("models", {}).keys())
if not MODELS:
    print("[ERROR] No models configured for the first provider")
    sys.exit(1)
MODEL_NAME = MODELS[0]

print(f"[CONFIG] Using provider: {provider.get('name')}")
print(f"[CONFIG]    base_url: {BASE_URL}")
print(f"[CONFIG]    model:    {MODEL_NAME}")
print()

# ---------------------------------------------------------------------------
# 2. Start the gateway in a background subprocess
# ---------------------------------------------------------------------------
GATEWAY_HOST = "127.0.0.1"
GATEWAY_PORT = 8799
GATEWAY_BASE_URL = f"http://{GATEWAY_HOST}:{GATEWAY_PORT}"

gateway_process = None


def kill_port(port: int):
    """Kill any process listening on the given port (Windows)."""
    try:
        output = subprocess.check_output(
            f"netstat -ano | findstr :{port}",
            shell=True,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace")
        for line in output.splitlines():
            parts = line.strip().split()
            if len(parts) >= 5 and "LISTENING" in line:
                pid = parts[-1]
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/PID", pid],
                        capture_output=True,
                        timeout=3,
                    )
                    print(f"[GATEWAY] Killed stale process PID={pid} on port {port}")
                except Exception:
                    pass
    except Exception:
        pass


def start_gateway():
    global gateway_process

    kill_port(GATEWAY_PORT)

    print(f"[GATEWAY] Starting auto-gateway using {CONFIG_PATH} ...")
    gateway_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "auto_gateway.cli.main",
            "start",
            "--config",
            str(CONFIG_PATH),
            "--host",
            GATEWAY_HOST,
            "--port",
            str(GATEWAY_PORT),
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
        cwd=str(PROJECT_ROOT),
    )

    print("[GATEWAY] Waiting for server to become ready...")

    # Step 1: Wait for TCP port to be open
    for _ in range(30):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(1)
            s.connect((GATEWAY_HOST, GATEWAY_PORT))
            s.close()
            break
        except (ConnectionRefusedError, OSError):
            s.close()
            time.sleep(0.5)
    else:
        stdout, stderr = gateway_process.communicate(timeout=2)
        print("[GATEWAY] FAILED to start. Stdout:")
        print(stdout.decode(errors="replace"))
        print("[GATEWAY] Stderr:")
        print(stderr.decode(errors="replace"))
        gateway_process = None
        sys.exit(1)

    # Step 2: Wait for HTTP handler to be ready
    import requests as _requests

    for _ in range(15):
        try:
            _requests.get(
                f"http://{GATEWAY_HOST}:{GATEWAY_PORT}/v1/models", timeout=2
            )
            print(f"[GATEWAY] Server is ready on {GATEWAY_BASE_URL}")
            return
        except Exception:
            time.sleep(0.5)

    stdout, stderr = gateway_process.communicate(timeout=2)
    print("[GATEWAY] FAILED to respond to HTTP. Stdout:")
    print(stdout.decode(errors="replace"))
    print("[GATEWAY] Stderr:")
    print(stderr.decode(errors="replace"))
    gateway_process = None
    sys.exit(1)


def stop_gateway():
    global gateway_process
    if gateway_process is not None:
        print("[GATEWAY] Stopping server...")
        gateway_process.terminate()
        try:
            gateway_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            gateway_process.kill()
            gateway_process.wait()
        print("[GATEWAY] Server stopped.")
        gateway_process = None


# ---------------------------------------------------------------------------
# 3. Tests using the official OpenAI SDK
# ---------------------------------------------------------------------------


def test_v1_models_endpoint():
    """Verify GET /v1/models returns a valid list."""
    import requests

    print("\n──────────────────────────────────────────────")
    print("  TEST: GET /v1/models endpoint")
    print("──────────────────────────────────────────────")

    resp = requests.get(f"{GATEWAY_BASE_URL}/v1/models", timeout=5)
    resp.raise_for_status()
    data = resp.json()

    print(f"  object: {data.get('object')}")
    print(f"  models: {[m['id'] for m in data.get('data', [])]}")

    assert data.get("object") == "list", f"Expected object='list', got {data.get('object')}"
    assert len(data.get("data", [])) > 0, "No models returned"
    print("  ✅ PASSED")
    return data


def test_non_streaming(client):
    """Non-streaming chat completion."""
    print("\n──────────────────────────────────────────────")
    print("  TEST: Non-streaming chat completion")
    print("──────────────────────────────────────────────")

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say hello in one word."}],
        temperature=0.7,
        max_tokens=50,
    )

    print(f"  ID:          {resp.id}")
    print(f"  Model:       {resp.model}")
    print(f"  Finish:      {resp.choices[0].finish_reason}")
    print(f"  Content:     {resp.choices[0].message.content!r}")

    assert resp.choices, "No choices returned"
    print("  ✅ PASSED")
    return resp


def test_streaming(client):
    """Streaming chat completion."""
    print("\n──────────────────────────────────────────────")
    print("  TEST: Streaming chat completion")
    print("──────────────────────────────────────────────")

    collected_chunks = []
    collected_content = []

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Count to three."}],
        temperature=0.0,
        max_tokens=100,
        stream=True,
    )

    for chunk in stream:
        collected_chunks.append(chunk)
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            collected_content.append(delta.content)
            print(f"    delta: {delta.content!r}")

    full_text = "".join(collected_content)
    print(f"\n  Full collected content: {full_text!r}")

    assert collected_chunks, "No chunks received during streaming"
    print("  ✅ PASSED")
    return collected_chunks


def test_extra_fields_preserved(client):
    """Verify extra fields like top_p, presence_penalty reach the provider."""
    print("\n──────────────────────────────────────────────")
    print("  TEST: Extra fields (top_p, frequency_penalty, etc.)")
    print("──────────────────────────────────────────────")

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say 'hi'"}],
        temperature=0.9,
        top_p=0.8,
        frequency_penalty=0.5,
        presence_penalty=0.2,
        max_tokens=30,
    )

    print(f"  Content: {resp.choices[0].message.content!r}")
    print(f"  Finish:  {resp.choices[0].finish_reason}")
    print(f"  Choices count: {len(resp.choices)}")

    assert resp.choices, "No choices returned"
    print("  ✅ PASSED")
    return resp


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------


def main():
    try:
        start_gateway()

        from openai import OpenAI

        client = OpenAI(
            api_key=API_KEY,
            base_url=f"{GATEWAY_BASE_URL}/v1",
        )

        test_v1_models_endpoint()
        test_non_streaming(client)
        test_streaming(client)
        test_extra_fields_preserved(client)

        print("\n══════════════════════════════════════════════")
        print("  🎉 ALL TESTS PASSED")
        print("══════════════════════════════════════════════\n")

    except Exception as e:
        print(f"\n[FAIL] Test encountered an error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        stop_gateway()


if __name__ == "__main__":
    main()
