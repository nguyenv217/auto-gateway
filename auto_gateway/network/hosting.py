from __future__ import annotations

import asyncio
import re
import sys
from dataclasses import dataclass
from typing import Optional


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


async def start_cloudflared(
    *,
    port: int,
    binary: str = "cloudflared",
) -> TunnelInfo:
    # Cloudflared usually prints URLs to stdout; parse for the first trycloudflare domain.
    # Example line: "trycloudflare.com" or "https://xxxxx.trycloudflare.com"

    cmd = [binary, "tunnel", "--url", f"http://127.0.0.1:{port}"]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    if proc.stdout is None:  # pragma: no cover
        raise RuntimeError("cloudflared stdout not available")

    pattern = re.compile(r"https?://([\w-]+\.trycloudflare\.com)")

    # Wait briefly for URL.
    deadline = asyncio.get_running_loop().time() + 20
    public_url: Optional[str] = None
    while asyncio.get_running_loop().time() < deadline:
        line = await proc.stdout.readline()
        if not line:
            await asyncio.sleep(0.05)
            continue
        text = line.decode("utf-8", errors="ignore")
        m = pattern.search(text)
        if m:
            public_url = f"https://{m.group(1)}"
            break

    if not public_url:
        # Surface some logs to help debugging.
        raise RuntimeError("Failed to start cloudflared tunnel (public URL not found)")

    # Keep process running (daemon mode not supported here).
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

