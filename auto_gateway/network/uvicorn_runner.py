from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import logging
logger = logging.getLogger("uvicorn_runner")

def run_uvicorn_app(*, app, host: str, port: int, socket_path: Optional[str] = None) -> None:
    import uvicorn

    # UDS support is tricky across OSes.
    # - Uvicorn supports `uds=` (expects filesystem path), but many Windows builds don't.
    # - If socket_path is provided on Windows, fall back to TCP.
    if socket_path and socket_path.strip() and not (socket_path.startswith("\\\\") or ":" in socket_path):
        # unix-style path
        try:
            uvicorn.run(app, uds=socket_path, host="0.0.0.0", port=port)
            return
        except TypeError:
            # uds kw unsupported
            pass
        except Exception:
            # fall back to tcp
            pass

    uvicorn.run(app, host=host, port=port)

