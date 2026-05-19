from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from auto_gateway.network.hosting import start_cloudflared


class FakeStdout:
    def __init__(self, lines: list[bytes]):
        self._lines = lines
        self._i = 0

    async def readline(self):
        if self._i >= len(self._lines):
            await asyncio.sleep(0)
            return b""
        line = self._lines[self._i]
        self._i += 1
        return line


class FakeProc:
    def __init__(self, lines: list[bytes]):
        self.stdout = FakeStdout(lines)


@pytest.mark.asyncio
async def test_cloudflared_parses_try_url(monkeypatch):
    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProc(
            [
                b"INFO Starting tunnel...\n",
                b"https://abc123.trycloudflare.com (http://127.0.0.1:8000)\n",

            ]
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    info = await start_cloudflared(port=8000, binary="cloudflared")
    assert info.public_url == "https://abc123.trycloudflare.com"
    assert info.backend == "cloudflared"

