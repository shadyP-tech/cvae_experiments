"""Frozen configuration for the bounded Uniform-B spatial diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ..protocol import ProtocolError
from ..schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS

CACHE_NAME = "uniform_b_spatial_token_cache_v1"
EXPERIMENT_NAME = "uniform_b_spatial_probe_v1"
EXPERIMENT_ID = "midogpp.oracle.uniform_b_spatial_probe.v1"
OUTPUT_ID = "midogpp_output_uniform_b_spatial_probe_v1"
DATASET_ID = "midogpp_dataset_contract_annotation_patch_v1"
CANONICAL_CACHE_ID = "midogpp_virchow2_uniform_b_canonical_train_cache_seed42"
SPATIAL_CACHE_ID = "midogpp_virchow2_uniform_b_spatial_token_cache_seed42"
CANONICAL_REFERENCE_ID = "midogpp_output_uniform_b_canonical_reference_v1"
NONLINEAR_REFERENCE_ID = "midogpp_output_uniform_b_nystroem_nonlinear_probe_v1"
MODEL_REF = "hf-hub:paige-ai/Virchow2"
MODEL_REVISION = "3158645804b69e3f3bc4439d4116edddf0840a72"
MODEL_CONFIG_SHA256 = "7db445b996bb165e88fe70e826c2ebb530539a2b1d136aa16eeb847df5f1e3db"
CHECKPOINT_FILE_SHA256 = "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b"
STATE_DICT_SHA256 = "91084959869cb53bf76e5038e5dc8a8ddc1ef8359a886fa22c19b4e8c62e112a"
PREPROCESSING_HASH = "4fb7d9ab76d1da72"
CANONICAL_DIM = 3840
GLOBAL_DIM = 2560
TOKEN_WIDTH = 1280
TOKEN_GRID_SIDE = 4
SPATIAL_DIM = 7680
EXPECTED_ROWS = 9648
PRIMARY_SEED = 42
STABILITY_SEEDS = (17, 101)
GAMMA_SAMPLE_SEED = 42017
GAMMA_SAMPLE_CAP = 512

@dataclass(frozen=True)
class SpatialCacheConfig:
    name: str
    root: Path
    repo_root: Path
    manifest_path: Path
    canonical_b_cache_path: Path
    hf_hub_cache_path: Path
    hf_hub_local_files_only: bool
    eligible_centers: tuple[str, ...]
    expected_rows: int
    model_ref: str
    model_revision: str
    expected_model_config_sha256: str
    expected_checkpoint_file_sha256: str
    expected_state_dict_sha256: str
    expected_preprocessing_config_hash: str
    experiment_seed: int
    devices: tuple[str, ...]
    workers_per_device: int
    batch_size_per_device: int
    extraction_precision: str
    token_storage_dtype: str

@dataclass(frozen=True)
class SpatialGateConfig:
    mean_bacc_delta_min: float
    strict_center_wins_min: int
    worst_center_bacc_delta_min: float
    worst_center_class_direction_delta_min: float
    hard_core_net_rescue_min_exclusive: int
    stability_mean_delta_min_exclusive: float
    stability_primary_deviation_max: float
    stability_worst_center_delta_min: float
    bootstrap_replicates: int
    bootstrap_seed: int

@dataclass(frozen=True)
class SpatialRuntimeConfig:
    outer_jobs: int
    threads_per_job: int
    use_gpu_for_classifiers: bool

@dataclass(frozen=True)
class SpatialProbeConfig:
    name: str
    artifact_root: Path
    manifest_path: Path
    canonical_b_cache_path: Path
    spatial_cache_root: Path
    spatial_cache_config_path: Path
    spatial_feature_cache_path: Path
    canonical_reference_root: Path
    nonlinear_reference_root: Path
    heldout_centers: tuple[str, ...]
    expected_rows: int
    expected_feature_dim: int
    linear_lock: str
    nystroem_lock: str
    primary_landmark_seed: int
    stability_landmark_seeds: tuple[int, ...]
    gamma_sample_seed: int
    gamma_sample_cap: int
    threshold_policy: str
    gate: SpatialGateConfig
    runtime: SpatialRuntimeConfig
    claim_boundary: Mapping[str, object]

def load_spatial_cache_config(path: str | Path) -> SpatialCacheConfig:
    payload = _payload(path)
    cache, inputs, model, run = (_mapping(payload, key) for key in ("cache", "inputs", "model", "run"))
    config = SpatialCacheConfig(
        name=str(cache["name"]), root=Path(str(cache["root"])),
        repo_root=Path(str(inputs["repo_root"])), manifest_path=Path(str(inputs["manifest_path"])),
        canonical_b_cache_path=Path(str(inputs["canonical_b_cache_path"])),
        hf_hub_cache_path=Path(str(model["hf_hub_cache_path"])),
        hf_hub_local_files_only=bool(model["hf_hub_local_files_only"]),
        eligible_centers=tuple(str(v) for v in run["eligible_centers"]), expected_rows=int(run["expected_rows"]),
        model_ref=str(model["model_ref"]), model_revision=str(model["model_revision"]),
        expected_model_config_sha256=str(model["expected_model_config_sha256"]),
        expected_checkpoint_file_sha256=str(model["expected_checkpoint_file_sha256"]),
        expected_state_dict_sha256=str(model["expected_state_dict_sha256"]),
        expected_preprocessing_config_hash=str(model["expected_preprocessing_config_hash"]),
        experiment_seed=int(run["experiment_seed"]), devices=tuple(str(v) for v in run["devices"]),
        workers_per_device=int(run["workers_per_device"]), batch_size_per_device=int(run["batch_size_per_device"]),
        extraction_precision=str(run["extraction_precision"]), token_storage_dtype=str(run["token_storage_dtype"]),
    )
    if (config.name != CACHE_NAME or config.eligible_centers != MIDOGPP_ELIGIBLE_CENTERS
        or config.expected_rows != EXPECTED_ROWS or config.model_ref != MODEL_REF
        or config.model_revision != MODEL_REVISION or config.expected_model_config_sha256 != MODEL_CONFIG_SHA256
        or config.expected_checkpoint_file_sha256 != CHECKPOINT_FILE_SHA256
        or config.expected_state_dict_sha256 != STATE_DICT_SHA256
        or config.expected_preprocessing_config_hash != PREPROCESSING_HASH or config.experiment_seed != 42
        or config.hf_hub_cache_path != Path("/home/stud/spark/.cache/huggingface/hub")
        or config.hf_hub_local_files_only is not True
        or config.devices != ("cuda:0", "cuda:1") or config.workers_per_device != 1
        or config.batch_size_per_device != 32 or config.extraction_precision != "float32"
        or config.token_storage_dtype != "float16"):
        raise ProtocolError("Uniform-B spatial cache protocol drifted.")
    return config

def load_spatial_probe_config(path: str | Path) -> SpatialProbeConfig:
    payload = _payload(path)
    experiment, inputs, run, classifiers, gate, runtime, claim = (
        _mapping(payload, key) for key in ("experiment", "inputs", "run", "classifiers", "progression_gate", "runtime", "claim_boundary")
    )
    heldouts = MIDOGPP_ELIGIBLE_CENTERS if str(run.get("heldout_centers", "")).lower() == "all" else tuple(str(v) for v in run["heldout_centers"])
    config = SpatialProbeConfig(
        name=str(experiment["name"]), artifact_root=Path(str(experiment["artifact_root"])),
        manifest_path=Path(str(inputs["manifest_path"])), canonical_b_cache_path=Path(str(inputs["canonical_b_cache_path"])),
        spatial_cache_root=Path(str(inputs["spatial_cache_root"])), spatial_cache_config_path=Path(str(inputs["spatial_cache_config_path"])),
        spatial_feature_cache_path=Path(str(inputs["spatial_feature_cache_path"])),
        canonical_reference_root=Path(str(inputs["canonical_reference_root"])), nonlinear_reference_root=Path(str(inputs["nonlinear_reference_root"])),
        heldout_centers=tuple(heldouts), expected_rows=int(run["expected_rows"]), expected_feature_dim=int(run["expected_feature_dim"]),
        linear_lock=str(classifiers["linear_lock"]), nystroem_lock=str(classifiers["nystroem_lock"]),
        primary_landmark_seed=int(classifiers["primary_landmark_seed"]),
        stability_landmark_seeds=tuple(int(v) for v in classifiers["stability_landmark_seeds"]),
        gamma_sample_seed=int(classifiers["gamma_sample_seed"]), gamma_sample_cap=int(classifiers["gamma_sample_cap"]),
        threshold_policy=str(classifiers["threshold_policy"]),
        gate=SpatialGateConfig(
            mean_bacc_delta_min=float(gate["mean_bacc_delta_min"]), strict_center_wins_min=int(gate["strict_center_wins_min"]),
            worst_center_bacc_delta_min=float(gate["worst_center_bacc_delta_min"]),
            worst_center_class_direction_delta_min=float(gate["worst_center_class_direction_delta_min"]),
            hard_core_net_rescue_min_exclusive=int(gate["hard_core_net_rescue_min_exclusive"]),
            stability_mean_delta_min_exclusive=float(gate["stability_mean_delta_min_exclusive"]),
            stability_primary_deviation_max=float(gate["stability_primary_deviation_max"]),
            stability_worst_center_delta_min=float(gate["stability_worst_center_delta_min"]),
            bootstrap_replicates=int(gate["bootstrap_replicates"]), bootstrap_seed=int(gate["bootstrap_seed"]),
        ),
        runtime=SpatialRuntimeConfig(int(runtime["outer_jobs"]), int(runtime["threads_per_job"]), bool(runtime["use_gpu_for_classifiers"])),
        claim_boundary=dict(claim),
    )
    expected_gate = SpatialGateConfig(0.005, 6, -0.01, -0.05, 0, 0.0, 0.01, -0.01, 2000, 42)
    if (config.name != EXPERIMENT_NAME or config.heldout_centers != MIDOGPP_ELIGIBLE_CENTERS
        or config.expected_rows != EXPECTED_ROWS or config.expected_feature_dim != SPATIAL_DIM
        or config.linear_lock != "canonical_b_per_outer_selected_spec"
        or config.nystroem_lock != "canonical_bplus_per_outer_selected_capacity"
        or config.primary_landmark_seed != PRIMARY_SEED or config.stability_landmark_seeds != STABILITY_SEEDS
        or config.gamma_sample_seed != GAMMA_SAMPLE_SEED or config.gamma_sample_cap != GAMMA_SAMPLE_CAP
        or config.threshold_policy != "predict" or config.gate != expected_gate
        or config.runtime != SpatialRuntimeConfig(4, 3, False)):
        raise ProtocolError("Uniform-B spatial diagnostic protocol drifted.")
    required_claim = {
        "claim_scope": "diagnostic_only", "diagnostic_surface_previously_inspected": True,
        "representation_change_isolated": True, "may_replace_canonical_reference": False,
        "may_feed_recipe_selection": False, "may_feed_deployable_selection": False,
        "validation_features_generated": False, "test_features_generated": False,
    }
    if any(config.claim_boundary.get(k) != v for k, v in required_claim.items()):
        raise ProtocolError("Uniform-B spatial claim boundary drifted.")
    return config

def _payload(path: str | Path) -> Mapping[str, object]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ProtocolError("Uniform-B spatial config must be a mapping.")
    return payload

def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Uniform-B spatial config section {key!r} must be a mapping.")
    return value
