from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Optional
import subprocess
import threading

@dataclass(frozen=True)
class TunnelInfo:
    public_url: str
    backend: str


async def start_ngrok(*, port: int, ngrok_authtoken: Optional[str] = None) -> TunnelInfo:
    try:
        from pyngrok import ngrok
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pyngrok is required for ngrok tunneling") from e

    if ngrok_authtoken:
        ngrok.set_auth_token(ngrok_authtoken)

    # Ensure we always start a fresh tunnel.
    public_url = ngrok.connect(port, "http").public_url
    return TunnelInfo(public_url=public_url, backend="ngrok")

def _drain_cloudflared_stdout(stdout, pattern, result_list):
    # Continuously read from stdout to prevent OS buffer deadlocks
    for line in iter(stdout.readline, b''):
        if not line:
            break
        text = line.decode("utf-8", errors="ignore")
        if not result_list:
            m = pattern.search(text)
            if m:
                result_list.append(f"https://{m.group(1)}")

async def start_cloudflared(
    *,
    port: int,
    binary: str = "cloudflared",
) -> TunnelInfo:
    # Cloudflared usually prints URLs to stdout; parse for the first trycloudflare domain.
    cmd = [binary, "tunnel", "--url", f"http://127.0.0.1:{port}"]

    # Use Popen instead of asyncio to detach from the temporary CLI event loop
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    pattern = re.compile(r"https?://([\w-]+\.trycloudflare\.com)")
    result_list: list[str] = []
    
    # Start a daemon thread to continuously drain stdout
    t = threading.Thread(
        target=_drain_cloudflared_stdout,
        args=(proc.stdout, pattern, result_list),
        daemon=True
    )
    t.start()

    # Wait briefly for URL.
    deadline = asyncio.get_running_loop().time() + 20
    public_url: Optional[str] = None
    
    while asyncio.get_running_loop().time() < deadline:
        if result_list:
            public_url = result_list[0]
            break
        await asyncio.sleep(0.1)

    if not public_url:
        raise RuntimeError("Failed to start cloudflared tunnel (public URL not found)")

    return TunnelInfo(public_url=public_url, backend="cloudflared")


async def start_tunnel(tunnel: str, *, port: int, config: dict | None = None) -> Optional[TunnelInfo]:
    config = config or {}

    if tunnel == "none":
        return None
    if tunnel == "ngrok":
        return await start_ngrok(
            port=port,
            ngrok_authtoken=config.get("ngrok_authtoken"),
        )
    if tunnel == "cloudflared":
        return await start_cloudflared(
            port=port,
            binary=config.get("cloudflared_binary", "cloudflared"),
        )

    raise ValueError(f"Unsupported tunnel backend: {tunnel}")

