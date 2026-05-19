from __future__ import annotations

import asyncio
from typing import Optional


async def read_lines(proc, *, max_lines: int = 200, timeout_s: float = 2.0):
    lines = []
    deadline = asyncio.get_running_loop().time() + timeout_s
    while len(lines) < max_lines and asyncio.get_running_loop().time() < deadline:
        if proc.stdout is None:
            break
        line = await proc.stdout.readline()
        if not line:
            await asyncio.sleep(0.01)
            continue
        lines.append(line.decode("utf-8", errors="ignore"))
    return lines

