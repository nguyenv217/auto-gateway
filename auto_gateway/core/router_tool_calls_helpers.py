from __future__ import annotations

import json
from typing import Any


def chunk_bytes_tool_calls(*, chatcmpl_id: str, created: int, model: str, tool_calls_delta: list[dict[str, Any]], finish_reason: str | None) -> bytes:
    payload = {
        "id": chatcmpl_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"tool_calls": tool_calls_delta},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")

