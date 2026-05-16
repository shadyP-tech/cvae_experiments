from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.config.schema import validate_config


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return validate_config(cfg)
