from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from src.app.bootstrap import resolve_config_path
from src.data.datasets.breakhis import prepare_breakhis_records
from src.data.datasets.camelyon17 import prepare_camelyon17_records


def _resolve_split_domain_caps(cfg: Dict[str, Any]) -> Dict[str, int] | None:
    data_cfg = cfg.get("data", {})
    split_caps = data_cfg.get("split_domain_caps")
    fixed_split_caps = data_cfg.get("fixed_split_caps")

    if split_caps is not None and fixed_split_caps is not None:
        raise ValueError("Use only one of data.split_domain_caps or data.fixed_split_caps")

    resolved = split_caps if split_caps is not None else fixed_split_caps
    if resolved is not None:
        if not isinstance(resolved, dict):
            raise ValueError("data.split_domain_caps/data.fixed_split_caps must be a dictionary")
        return {
            "train": int(resolved.get("train", 0)),
            "val": int(resolved.get("val", 0)),
            "test": int(resolved.get("test", 0)),
        }

    profile = str(data_cfg.get("split_cap_profile", "legacy")).strip().lower()
    if profile == "development":
        return {"train": 250, "val": 100, "test": 200}
    if profile == "final":
        return {"train": 1000, "val": 250, "test": 1000}

    return None


def _prepare_breakhis(project_root: Path, cfg: Dict[str, Any]) -> Tuple[list[Any], Dict[str, Any]]:
    root = resolve_config_path(project_root, str(cfg["data"]["root"]))
    split_domain_caps = _resolve_split_domain_caps(cfg)
    cap_per_domain_raw = cfg["data"].get("max_samples_per_domain")
    return prepare_breakhis_records(
        root=root,
        extensions=cfg["data"]["image_extensions"],
        split=cfg["data"]["split"],
        cap_per_domain=int(cap_per_domain_raw) if cap_per_domain_raw is not None else None,
        seed=int(cfg["seed"]),
        require_patient_ids=bool(cfg["data"]["require_patient_ids"]),
        split_domain_caps=split_domain_caps,
    )


def _prepare_camelyon17(project_root: Path, cfg: Dict[str, Any]) -> Tuple[list[Any], Dict[str, Any]]:
    root = resolve_config_path(project_root, str(cfg["data"]["root"]))
    split_domain_caps = _resolve_split_domain_caps(cfg)
    cap_per_domain_raw = cfg["data"].get("max_samples_per_domain")
    return prepare_camelyon17_records(
        root=root,
        extensions=cfg["data"]["image_extensions"],
        split=cfg["data"]["split"],
        cap_per_domain=int(cap_per_domain_raw) if cap_per_domain_raw is not None else None,
        seed=int(cfg["seed"]),
        require_patient_ids=bool(cfg["data"]["require_patient_ids"]),
        domain_field=str(cfg["data"].get("domain_field", "center")),
        metadata_file=str(cfg["data"].get("metadata_file", "metadata.csv")),
        use_metadata_split=bool(cfg["data"].get("use_metadata_split", False)),
        split_domain_caps=split_domain_caps,
    )


DATASET_REGISTRY = {
    "breakhis": _prepare_breakhis,
    "camelyon17": _prepare_camelyon17,
}


def prepare_dataset_records(project_root: Path, cfg: Dict[str, Any]) -> Tuple[list[Any], Dict[str, Any]]:
    dataset_type = str(cfg.get("data", {}).get("dataset_type", "breakhis")).lower()
    adapter = DATASET_REGISTRY.get(dataset_type)
    if adapter is None:
        raise ValueError(f"Unsupported data.dataset_type: {dataset_type}. Available: {sorted(DATASET_REGISTRY)}")
    return adapter(project_root, cfg)
