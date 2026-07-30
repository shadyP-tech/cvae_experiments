"""Frozen dataset-owned configuration for annotation-local physical multiscale v2."""

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
from .config import EXPECTED_FOV_UM


PROFILE_ID = "physical_multiscale_annotation_local_pooling_pilot_v2"
CONTRACT_SCHEMA_VERSION = "midogpp_physical_multiscale_annotation_local_contract_v2"
B_REPRESENTATION_ID = "annotation_jpeg_fixed_center_b_v2"
C_REPRESENTATION_ID = "physical_multiscale_annotation_local_c_v2"
B_FEATURE_DIM = 3840
C_FEATURE_DIM = 11520


@dataclass(frozen=True)
class PhysicalMultiscaleV2BuildConfig:
    repo_root: Path
    config_path: Path
    raw_root: Path
    raw_metadata_path: Path
    base_manifest_path: Path
    canonical_cache_path: Path
    canonical_reference_root: Path
    contract_root: Path
    cache_bundle_root: Path
    eligible_centers: tuple[str, ...]
    fov_um: tuple[float, ...]
    mpp_min: float
    mpp_max: float
    anisotropy_relative_max: float
    dual_source_relative_max: float
    resize_interpolation: str
    resize_antialias: bool
    output_size_px: int
    model_ref: str
    model_revision: str
    expected_model_config_sha256: str
    expected_checkpoint_file_sha256: str
    expected_state_dict_sha256: str
    expected_preprocessing_config_hash: str
    expected_timm_version: str
    expected_torch_version: str
    expected_pillow_version: str
    device: str
    batch_size: int
    require_tiled_reader: bool
    experiment_seed: int
    expected_row_count: int
    expected_slide_count: int
    bridge_minimum_cosine: float = MINIMUM_COSINE
    bridge_maximum_relative_l2: float = MAXIMUM_RELATIVE_L2
    bridge_minimum_prediction_agreement: float = MINIMUM_PREDICTION_AGREEMENT
    bridge_maximum_equal_center_bacc_delta: float = MAXIMUM_EQUAL_CENTER_BACC_DELTA

    @property
    def b_cache_root(self) -> Path:
        return self.cache_bundle_root / "b_3840"

    @property
    def c_cache_root(self) -> Path:
        return self.cache_bundle_root / "c_11520"


def load_build_config_v2(
    path: str | Path,
    *,
    require_inputs: bool = True,
    workspace: MidogppWorkspace | None = None,
) -> PhysicalMultiscaleV2BuildConfig:
    """Resolve the exact v2 dataset build profile through catalog ownership."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("Physical multiscale v2 build config must be a mapping.")
    ws = workspace or MidogppWorkspace.load()
    ws.validate()
    used: set[str] = set()
    resolved = ws.resolve_value(
        payload,
        require_inputs=require_inputs,
        used_inputs=used,
    )
    if not isinstance(resolved, Mapping):
        raise ValueError("Resolved physical multiscale v2 config must be a mapping.")
    expected_inputs = {
        "midogpp_raw_tiff_source_v1",
        "midogpp_dataset_contract_annotation_patch_v1",
        "midogpp_virchow2_xyxy_feature_cache_seed42",
        "midogpp_output_eligible_tuned_real_reference_v2",
    }
    if used != expected_inputs:
        raise ValueError(
            f"Physical multiscale v2 inputs drifted: expected={sorted(expected_inputs)}, "
            f"actual={sorted(used)}"
        )
    artifact = _mapping(resolved, "artifact")
    if artifact.get("name") != PROFILE_ID:
        raise ValueError(f"Physical multiscale v2 profile must be {PROFILE_ID}.")
    inputs = _mapping(resolved, "inputs")
    outputs = _mapping(resolved, "outputs")
    physical = _mapping(resolved, "physical_scale")
    geometry = _mapping(resolved, "geometry")
    bridge = _mapping(resolved, "bridge")
    model = _mapping(resolved, "model")
    runtime = _mapping(resolved, "runtime_identity")
    run = _mapping(resolved, "run")
    config = PhysicalMultiscaleV2BuildConfig(
        repo_root=ws.repo_root,
        config_path=config_path.resolve(),
        raw_root=Path(str(inputs["raw_root"])),
        raw_metadata_path=Path(str(inputs["raw_metadata_path"])),
        base_manifest_path=Path(str(inputs["base_manifest_path"])),
        canonical_cache_path=Path(str(inputs["canonical_cache_path"])),
        canonical_reference_root=Path(str(inputs["canonical_reference_root"])),
        contract_root=_repo_path(ws.repo_root, outputs["contract_root"]),
        cache_bundle_root=_repo_path(ws.repo_root, outputs["cache_bundle_root"]),
        eligible_centers=tuple(str(value) for value in run["eligible_centers"]),
        fov_um=tuple(float(value) for value in physical["fov_um"]),
        mpp_min=float(physical["mpp_min"]),
        mpp_max=float(physical["mpp_max"]),
        anisotropy_relative_max=float(physical["anisotropy_relative_max"]),
        dual_source_relative_max=float(physical["dual_source_relative_max"]),
        resize_interpolation=str(geometry["resize_interpolation"]),
        resize_antialias=bool(geometry["resize_antialias"]),
        output_size_px=int(geometry["output_size_px"]),
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
        expected_timm_version=str(runtime["timm"]),
        expected_torch_version=str(runtime["torch"]),
        expected_pillow_version=str(runtime["pillow"]),
        device=str(run["device"]),
        batch_size=int(run["batch_size"]),
        require_tiled_reader=bool(run["require_tiled_reader"]),
        experiment_seed=int(run["experiment_seed"]),
        expected_row_count=int(run["expected_row_count"]),
        expected_slide_count=int(run["expected_slide_count"]),
        bridge_minimum_cosine=float(bridge["minimum_cosine"]),
        bridge_maximum_relative_l2=float(bridge["maximum_relative_l2"]),
        bridge_minimum_prediction_agreement=float(
            bridge["minimum_prediction_agreement"]
        ),
        bridge_maximum_equal_center_bacc_delta=float(
            bridge["maximum_absolute_equal_center_bacc_delta"]
        ),
    )
    validate_build_config_v2(config)
    return config


def validate_build_config_v2(config: PhysicalMultiscaleV2BuildConfig) -> None:
    if config.eligible_centers != MIDOGPP_ELIGIBLE_CENTERS:
        raise ValueError("Physical multiscale v2 requires exact eligible centers.")
    if config.fov_um != EXPECTED_FOV_UM:
        raise ValueError(f"Physical multiscale v2 FOV grid must remain {EXPECTED_FOV_UM}.")
    if not 0.0 < config.mpp_min < config.mpp_max:
        raise ValueError("MPP bounds must be positive and ordered.")
    if config.anisotropy_relative_max != 0.01 or config.dual_source_relative_max != 0.01:
        raise ValueError("MPP anisotropy and dual-source tolerances must remain 1%.")
    if config.resize_interpolation != "bicubic" or not config.resize_antialias:
        raise ValueError("Physical multiscale v2 requires bicubic antialiased resizing.")
    if config.output_size_px != 224:
        raise ValueError("Physical multiscale v2 output size must remain 224.")
    if config.batch_size <= 0 or config.experiment_seed != 42:
        raise ValueError("Physical multiscale v2 batch size/seed drifted.")
    if config.expected_row_count != 9648 or config.expected_slide_count != 216:
        raise ValueError("Physical multiscale v2 cohort cardinalities must remain 9648/216.")
    revision = config.model_revision.lower()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Virchow2 model revision must be an exact commit.")
    identity_values = (
        config.expected_model_config_sha256,
        config.expected_checkpoint_file_sha256,
        config.expected_state_dict_sha256,
        config.expected_preprocessing_config_hash,
        config.expected_timm_version,
        config.expected_torch_version,
        config.expected_pillow_version,
    )
    if any(not value or value.startswith("TODO_") for value in identity_values):
        raise ValueError("Physical multiscale v2 runtime identity must be resolved.")
    if (
        config.bridge_minimum_cosine != MINIMUM_COSINE
        or config.bridge_maximum_relative_l2 != MAXIMUM_RELATIVE_L2
        or config.bridge_minimum_prediction_agreement
        != MINIMUM_PREDICTION_AGREEMENT
        or config.bridge_maximum_equal_center_bacc_delta
        != MAXIMUM_EQUAL_CENTER_BACC_DELTA
    ):
        raise ValueError("Physical multiscale v2 JPEG bridge thresholds drifted.")


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Physical multiscale v2 section {key!r} must be a mapping.")
    return value


def _repo_path(repo_root: Path, raw: object) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else repo_root / path
