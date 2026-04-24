from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from src.torch_utils import safe_torch_load


def training_state_path(model_checkpoint_path: Path) -> Path:
    stem = model_checkpoint_path.stem
    return model_checkpoint_path.with_name(f"{stem}.training.pt")


def save_resume_state(
    state_path: Path,
    *,
    model_payload: Dict[str, Any],
    optimizer_state: Dict[str, Any],
    history: Dict[str, list[float]],
    epoch: int,
    best_metric: float,
    bad_epochs: int,
    meta: Dict[str, Any] | None = None,
) -> None:
    payload = {
        "model_payload": model_payload,
        "optimizer_state": optimizer_state,
        "history": history,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "bad_epochs": int(bad_epochs),
        "meta": meta or {},
    }
    torch.save(payload, state_path)


def load_resume_state(state_path: Path) -> Dict[str, Any]:
    return safe_torch_load(state_path, map_location="cpu")
