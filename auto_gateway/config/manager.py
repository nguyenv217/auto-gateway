from __future__ import annotations

import json
import shutil

from pathlib import Path

GLOBAL_CONFIG_DIR = Path.home() / ".auto-gateway"
GLOBAL_CONFIG_PATH = GLOBAL_CONFIG_DIR / "config.json"

from .schema import GatewayConfig



def load_config(path: str | Path) -> GatewayConfig:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return GatewayConfig.model_validate(data)


def save_global_config(source_path: str | Path) -> None:
    """Copy the source config file to the global config location."""
    GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, GLOBAL_CONFIG_PATH)


def load_global_config() -> GatewayConfig:
    """Load GatewayConfig from the global config file."""
    if not GLOBAL_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Global config not found at {GLOBAL_CONFIG_PATH}. "
            f"Run 'auto-gateway save_global --config <path>' first."
        )
    data = json.loads(GLOBAL_CONFIG_PATH.read_text(encoding="utf-8"))
    return GatewayConfig.model_validate(data)
