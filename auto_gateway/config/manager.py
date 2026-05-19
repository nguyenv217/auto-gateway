from __future__ import annotations

import json
from pathlib import Path

from .schema import GatewayConfig


def load_config(path: str | Path) -> GatewayConfig:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return GatewayConfig.model_validate(data)

