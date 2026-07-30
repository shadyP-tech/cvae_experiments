"""Frozen dataset-owned configuration for physical multiscale extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from midogpp_thesis.common.midogpp import MIDOGPP_ELIGIBLE_CENTERS
from midogpp_thesis.workspace.runtime import MidogppWorkspace

from .bridge import (
    MAXIMUM_EQUAL_CENTER_BACC_DELTA,
    MAXIMUM_RELATIVE_L2,
    MINIMUM_COSINE,
    MINIMUM_PREDICTION_AGREEMENT,
)


EXPECTED_FOV_UM = (28.0, 56.0, 112.0)


@dataclass(frozen=True)
class PhysicalMultiscaleBuildConfig:
    repo_root: Path
    config_path: Path
    raw_root: Path
    raw_metadata_path: Path
    base_manifest_path: Path
    canonical_cache_path: Path
    canonical_reference_root: Path
    contract_root: Path
    b_cache_root: Path
    c_cache_root: Path
    eligible_centers: tuple[str, ...]
    fov_um: tuple[float, ...]
    mpp_min: float
    mpp_max: float
    anisotropy_relative_max: float
    dual_source_relative_max: float
    padding_fraction_max: float
    padding_rgb: tuple[int, int, int]
    resize_interpolation: str
    resize_antialias: bool
    model_ref: str
    model_revision: str
    expected_model_config_sha256: str
    expected_checkpoint_file_sha256: str
    expected_state_dict_sha256: str
    expected_preprocessing_config_hash: str
    device: str
    batch_size: int
    require_tiled_reader: bool
    experiment_seed: int
    bridge_minimum_cosine: float = MINIMUM_COSINE
    bridge_maximum_relative_l2: float = MAXIMUM_RELATIVE_L2
    bridge_minimum_prediction_agreement: float = MINIMUM_PREDICTION_AGREEMENT
    bridge_maximum_equal_center_bacc_delta: float = MAXIMUM_EQUAL_CENTER_BACC_DELTA


def load_build_config(
    path: str | Path,
    *,
    require_inputs: bool = True,
    workspace: MidogppWorkspace | None = None,
) -> PhysicalMultiscaleBuildConfig:
    """Load and resolve the production config through catalog IDs."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("Physical multiscale build config must be a mapping.")
    ws = workspace or MidogppWorkspace.load()
    ws.validate()
    used: set[str] = set()
    resolved = ws.resolve_value(
        payload,
        require_inputs=require_inputs,
        used_inputs=used,
    )
    if not isinstance(resolved, Mapping):
        raise ValueError("Resolved physical multiscale config must be a mapping.")
    expected_inputs = {
        "midogpp_raw_tiff_source_v1",
        "midogpp_dataset_contract_annotation_patch_v1",
        "midogpp_virchow2_xyxy_feature_cache_seed42",
        "midogpp_output_eligible_tuned_real_reference_v2",
    }
    if used != expected_inputs:
        raise ValueError(
            f"Physical multiscale build inputs drifted: expected={sorted(expected_inputs)}, "
            f"actual={sorted(used)}"
        )
    inputs = _mapping(resolved, "inputs")
    outputs = _mapping(resolved, "outputs")
    physical = _mapping(resolved, "physical_scale")
    geometry = _mapping(resolved, "geometry")
    bridge = _mapping(resolved, "bridge")
    model = _mapping(resolved, "model")
    run = _mapping(resolved, "run")
    eligible = tuple(str(value) for value in run.get("eligible_centers", ()))
    fovs = tuple(float(value) for value in physical.get("fov_um", ()))
    padding_rgb = tuple(int(value) for value in geometry.get("padding_rgb", ()))
    cfg = PhysicalMultiscaleBuildConfig(
        repo_root=ws.repo_root,
        config_path=config_path.resolve(),
        raw_root=Path(str(inputs["raw_root"])),
        raw_metadata_path=Path(str(inputs["raw_metadata_path"])),
        base_manifest_path=Path(str(inputs["base_manifest_path"])),
        canonical_cache_path=Path(str(inputs["canonical_cache_path"])),
        canonical_reference_root=Path(str(inputs["canonical_reference_root"])),
        contract_root=_repo_path(ws.repo_root, outputs["contract_root"]),
        b_cache_root=_repo_path(ws.repo_root, outputs["b_cache_root"]),
        c_cache_root=_repo_path(ws.repo_root, outputs["c_cache_root"]),
        eligible_centers=eligible,
        fov_um=fovs,
        mpp_min=float(physical["mpp_min"]),
        mpp_max=float(physical["mpp_max"]),
        anisotropy_relative_max=float(physical["anisotropy_relative_max"]),
        dual_source_relative_max=float(physical["dual_source_relative_max"]),
        padding_fraction_max=float(geometry["padding_fraction_max"]),
        padding_rgb=padding_rgb,  # type: ignore[arg-type]
        resize_interpolation=str(geometry["resize_interpolation"]),
        resize_antialias=bool(geometry["resize_antialias"]),
        model_ref=str(model["model_ref"]),
        model_revision=str(model["model_revision"]),
        expected_model_config_sha256=str(model["expected_model_config_sha256"]),
        expected_checkpoint_file_sha256=str(
            model["expected_checkpoint_file_sha256"]
        ),
        expected_state_dict_sha256=str(model["expected_state_dict_sha256"]),
        expected_preprocessing_config_hash=str(
            model["expected_preprocessing_config_hash"]
        ),
        device=str(run["device"]),
        batch_size=int(run["batch_size"]),
        require_tiled_reader=bool(run["require_tiled_reader"]),
        experiment_seed=int(run["experiment_seed"]),
        bridge_minimum_cosine=float(bridge["minimum_cosine"]),
        bridge_maximum_relative_l2=float(bridge["maximum_relative_l2"]),
        bridge_minimum_prediction_agreement=float(
            bridge["minimum_prediction_agreement"]
        ),
        bridge_maximum_equal_center_bacc_delta=float(
            bridge["maximum_absolute_equal_center_bacc_delta"]
        ),
    )
    validate_build_config(cfg)
    return cfg


def validate_build_config(config: PhysicalMultiscaleBuildConfig) -> None:
    if config.eligible_centers != MIDOGPP_ELIGIBLE_CENTERS:
        raise ValueError("Physical multiscale production requires exact eligible centers.")
    if config.fov_um != EXPECTED_FOV_UM:
        raise ValueError(f"Physical FOV grid must remain {EXPECTED_FOV_UM}.")
    if not 0.0 < config.mpp_min < config.mpp_max:
        raise ValueError("MPP bounds must be positive and ordered.")
    if config.anisotropy_relative_max != 0.01 or config.dual_source_relative_max != 0.01:
        raise ValueError("MPP anisotropy and dual-source tolerances must remain 1%.")
    if config.padding_fraction_max != 0.10:
        raise ValueError("Padding hard-stop must remain 10%.")
    if config.padding_rgb != (255, 255, 255):
        raise ValueError("Physical crop padding must remain white.")
    if config.resize_interpolation != "bicubic" or not config.resize_antialias:
        raise ValueError("Physical crops require bicubic antialiased resizing.")
    if config.batch_size <= 0:
        raise ValueError("Extraction batch size must be positive.")
    if config.experiment_seed != 42:
        raise ValueError("Physical multiscale experiment seed must remain 42.")
    revision = config.model_revision.lower()
    if (
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("Virchow2 model revision must be an exact commit.")
    if (
        config.bridge_minimum_cosine != MINIMUM_COSINE
        or config.bridge_maximum_relative_l2 != MAXIMUM_RELATIVE_L2
        or config.bridge_minimum_prediction_agreement
        != MINIMUM_PREDICTION_AGREEMENT
        or config.bridge_maximum_equal_center_bacc_delta
        != MAXIMUM_EQUAL_CENTER_BACC_DELTA
    ):
        raise ValueError("Physical multiscale JPEG bridge thresholds drifted.")


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Physical multiscale config section {key!r} must be a mapping.")
    return value


def _repo_path(repo_root: Path, raw: object) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else repo_root / path
