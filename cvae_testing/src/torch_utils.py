from __future__ import annotations

from typing import Any

import torch


def safe_torch_load(path: Any, *, map_location: Any = "cpu") -> Any:
    """Load torch artifacts with safer defaults and backward compatibility."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        return torch.load(path, map_location=map_location)
