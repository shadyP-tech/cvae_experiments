"""Frozen bounded protocol for robust Nyström versus bilinear B+."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ..protocol import ProtocolError
from ..schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS


EXPERIMENT_NAME = "uniform_b_robust_interaction_probe_v1"
EXPERIMENT_ID = "midogpp.oracle.uniform_b_robust_interaction_probe.v1"
OUTPUT_ID = "midogpp_output_uniform_b_robust_interaction_probe_v1"
DATASET_ID = "midogpp_dataset_contract_annotation_patch_v1"
CACHE_ID = "midogpp_virchow2_uniform_b_canonical_train_cache_seed42"
CANONICAL_ID = "midogpp_output_uniform_b_canonical_reference_v1"
NONLINEAR_ID = "midogpp_output_uniform_b_nystroem_nonlinear_probe_v1"
MULTISCALE_ID = (
    "midogpp_output_real_feature_physical_multiscale_clipped_bbox_"
    "annotation_local_pooling_pilot_v3"
)
ROBUST_OBJECTIVES = ("equal_group", "group_dro_eta_0.1", "group_dro_eta_0.5")
BILINEAR_RANKS = (4, 8, 16)


@dataclass(frozen=True)
class RobustInteractionConfig:
    name: str
    artifact_root: Path
    manifest_path: Path
    feature_cache_path: Path
    canonical_root: Path
    nonlinear_root: Path
    multiscale_root: Path
    heldout_centers: tuple[str, ...]
    robust_objectives: tuple[str, ...]
    dro_iterations: int
    bilinear_ranks: tuple[int, ...]
    global_dim: int
    local_dim: int
    bilinear_epochs: int
    bilinear_learning_rate: float
    bilinear_weight_decay: float
    bilinear_batch_size: int
    primary_seed: int
    stability_seeds: tuple[int, ...]
    cpu_pair_jobs: int
    cpu_threads_per_job: int
    gpu_devices: tuple[int, ...]
    claim_boundary: Mapping[str, object]


def load_robust_interaction_config(path: str | Path) -> RobustInteractionConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ProtocolError("Robust-interaction config must be a mapping.")
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    run = _mapping(payload, "run")
    robust = _mapping(payload, "robust_nystroem")
    bilinear = _mapping(payload, "bilinear")
    runtime = _mapping(payload, "runtime")
    claim = _mapping(payload, "claim_boundary")
    heldouts = (
        MIDOGPP_ELIGIBLE_CENTERS
        if str(run["heldout_centers"]).lower() == "all"
        else tuple(str(value) for value in run["heldout_centers"])
    )
    config = RobustInteractionConfig(
        name=str(experiment["name"]),
        artifact_root=Path(str(experiment["artifact_root"])),
        manifest_path=Path(str(inputs["manifest_path"])),
        feature_cache_path=Path(str(inputs["feature_cache_path"])),
        canonical_root=Path(str(inputs["canonical_reference_root"])),
        nonlinear_root=Path(str(inputs["nonlinear_probe_root"])),
        multiscale_root=Path(str(inputs["multiscale_probe_root"])),
        heldout_centers=tuple(heldouts),
        robust_objectives=tuple(str(value) for value in robust["objectives"]),
        dro_iterations=int(robust["dro_iterations"]),
        bilinear_ranks=tuple(int(value) for value in bilinear["ranks"]),
        global_dim=int(bilinear["global_dim"]),
        local_dim=int(bilinear["local_dim"]),
        bilinear_epochs=int(bilinear["epochs"]),
        bilinear_learning_rate=float(bilinear["learning_rate"]),
        bilinear_weight_decay=float(bilinear["weight_decay"]),
        bilinear_batch_size=int(bilinear["batch_size"]),
        primary_seed=int(run["primary_seed"]),
        stability_seeds=tuple(int(value) for value in run["stability_seeds"]),
        cpu_pair_jobs=int(runtime["cpu_pair_jobs"]),
        cpu_threads_per_job=int(runtime["cpu_threads_per_job"]),
        gpu_devices=tuple(int(value) for value in runtime["gpu_devices"]),
        claim_boundary=dict(claim),
    )
    _validate(config)
    return config


def _validate(config: RobustInteractionConfig) -> None:
    required_claim = {
        "claim_scope": "diagnostic_only",
        "already_inspected_train_surface": True,
        "validation_scored": False,
        "test_scored": False,
        "may_replace_canonical_reference": False,
        "may_feed_deployable_selection": False,
    }
    if (
        config.name != EXPERIMENT_NAME
        or config.heldout_centers != MIDOGPP_ELIGIBLE_CENTERS
        or config.robust_objectives != ROBUST_OBJECTIVES
        or config.dro_iterations != 8
        or config.bilinear_ranks != BILINEAR_RANKS
        or config.global_dim != 2560
        or config.local_dim != 1280
        or config.global_dim + config.local_dim != 3840
        or config.bilinear_epochs != 1
        or config.bilinear_learning_rate != 0.001
        or config.bilinear_weight_decay != 0.001
        or config.bilinear_batch_size != 1024
        or config.primary_seed != 42
        or config.stability_seeds != (17, 101)
        or config.cpu_pair_jobs != 4
        or config.cpu_threads_per_job != 3
        or config.gpu_devices != (0, 1)
        or any(config.claim_boundary.get(key) != value for key, value in required_claim.items())
    ):
        raise ProtocolError("Robust-interaction protocol drifted.")


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Robust-interaction section {key!r} must be a mapping.")
    return value
