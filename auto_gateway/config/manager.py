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


def save_global_config(source_path: str | Path, alias: str | None = None) -> None:
    """Copy the source config file to the global config location."""
    GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if not alias:
        shutil.copy2(source_path, GLOBAL_CONFIG_PATH)
    else:
        if len(alias.strip().split('.')) > 2: # abc.def.json
            raise ValueError("Alias can only have one period")
        shutil.copy2(source_path, GLOBAL_CONFIG_DIR / f"config.{alias}.json")


def load_global_config(alias: str | None = None) -> GatewayConfig:
    """Load GatewayConfig from the global config file."""
    config_path = GLOBAL_CONFIG_PATH if not alias else GLOBAL_CONFIG_DIR / f"config.{alias}.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Global config not found at {config_path}. "
            f"Run 'auto-gateway save_global --config <path>' first."
        )
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return GatewayConfig.model_validate(data)
