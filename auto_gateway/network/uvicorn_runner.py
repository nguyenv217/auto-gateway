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
            logger.info(f"Attempting to start with UDS at {socket_path}")
            uvicorn.run(app, uds=socket_path, host="0.0.0.0", port=port)
            return
        except TypeError as e:
            # uds kw unsupported
            logger.warning(f"UDS not supported, falling back to TCP: {e}")
            pass
        except Exception as e:
            # fall back to tcp
            logger.error(f"UDS startup failed: {e}", exc_info=True)
            pass

    logger.info(f"Starting server on {host}:{port}")
    try:
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        logger.error(f"TCP startup failed: {e}", exc_info=True)
        raise e

