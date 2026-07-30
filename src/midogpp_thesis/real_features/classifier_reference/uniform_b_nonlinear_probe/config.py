"""Frozen protocol configuration for the canonical-B nonlinear probe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from midogpp_thesis.common.hashing import stable_hash

from ..protocol import ProtocolError
from ..schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS


EXPERIMENT_NAME = "uniform_b_nystroem_nonlinear_probe_v1"
EXPERIMENT_ID = "midogpp.oracle.uniform_b_nystroem_nonlinear_probe.v1"
OUTPUT_ID = "midogpp_output_uniform_b_nystroem_nonlinear_probe_v1"
DATASET_ID = "midogpp_dataset_contract_annotation_patch_v1"
CACHE_ID = "midogpp_virchow2_uniform_b_canonical_train_cache_seed42"
CANONICAL_REFERENCE_ID = "midogpp_output_uniform_b_canonical_reference_v1"
REPRESENTATION_ID = "annotation_jpeg_fixed_center_b_v3"
EXPECTED_FEATURE_DIM = 3840
EXPECTED_TRAIN_ROWS = 9648
WIDTH_MULTIPLIERS = (0.5, 1.0, 2.0)
COMPONENTS = (256, 512, 1024)
LOGISTIC_CS = (0.01, 0.1, 1.0, 10.0)
PRIMARY_LANDMARK_SEED = 42
STABILITY_LANDMARK_SEEDS = (17, 101)
GAMMA_SAMPLE_SEED = 42017
GAMMA_SAMPLE_CAP = 512
EXPECTED_CANDIDATES = 36
EXPECTED_PAIR_FRAMES = 36
EXPECTED_NYSTROEM_TRANSFORMS = 324
EXPECTED_SELECTOR_CELLS = 2592


@dataclass(frozen=True)
class Candidate:
    width_multiplier: float
    n_components: int
    logistic_c: float

    @property
    def candidate_id(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "family": "standard_scaler_nystroem_rbf_l2_logistic",
            "width_multiplier": float(self.width_multiplier),
            "n_components": int(self.n_components),
            "logistic_c": float(self.logistic_c),
        }


@dataclass(frozen=True)
class GateConfig:
    mean_bacc_delta_min: float
    strict_center_wins_min: int
    worst_center_delta_min: float
    mean_positive_recall_delta_min_exclusive: float
    mean_specificity_delta_min: float
    supplemental_mean_delta_min_exclusive: float
    supplemental_primary_deviation_max: float
    supplemental_worst_center_delta_min: float
    bootstrap_replicates: int
    bootstrap_seed: int


@dataclass(frozen=True)
class RuntimeConfig:
    pair_jobs: int
    threads_per_job: int
    use_gpu: bool


@dataclass(frozen=True)
class NonlinearProbeConfig:
    name: str
    artifact_root: Path
    manifest_path: Path
    feature_cache_path: Path
    canonical_reference_root: Path
    heldout_centers: tuple[str, ...]
    expected_feature_dim: int
    expected_train_rows: int
    width_multipliers: tuple[float, ...]
    components: tuple[int, ...]
    logistic_cs: tuple[float, ...]
    primary_landmark_seed: int
    stability_landmark_seeds: tuple[int, ...]
    gamma_sample_seed: int
    gamma_sample_cap: int
    classifier_max_iter: int
    threshold_policy: str
    gate: GateConfig
    runtime: RuntimeConfig
    claim_boundary: Mapping[str, object]

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        return tuple(
            Candidate(width, components, c_value)
            for width in self.width_multipliers
            for components in self.components
            for c_value in self.logistic_cs
        )


def load_nonlinear_probe_config(path: str | Path) -> NonlinearProbeConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ProtocolError("Uniform-B nonlinear probe config must be a mapping.")
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    run = _mapping(payload, "run")
    grid = _mapping(payload, "nonlinear_grid")
    gate = _mapping(payload, "progression_gate")
    runtime = _mapping(payload, "runtime")
    claim = _mapping(payload, "claim_boundary")
    heldouts = (
        MIDOGPP_ELIGIBLE_CENTERS
        if str(run.get("heldout_centers", "")).lower() == "all"
        else tuple(str(value) for value in run["heldout_centers"])
    )
    config = NonlinearProbeConfig(
        name=str(experiment["name"]),
        artifact_root=Path(str(experiment["artifact_root"])),
        manifest_path=Path(str(inputs["manifest_path"])),
        feature_cache_path=Path(str(inputs["feature_cache_path"])),
        canonical_reference_root=Path(str(inputs["canonical_reference_root"])),
        heldout_centers=tuple(heldouts),
        expected_feature_dim=int(run["expected_feature_dim"]),
        expected_train_rows=int(run["expected_train_rows"]),
        width_multipliers=tuple(float(value) for value in grid["width_multipliers"]),
        components=tuple(int(value) for value in grid["nystroem_components"]),
        logistic_cs=tuple(float(value) for value in grid["logistic_c"]),
        primary_landmark_seed=int(grid["primary_landmark_seed"]),
        stability_landmark_seeds=tuple(
            int(value) for value in grid["stability_landmark_seeds"]
        ),
        gamma_sample_seed=int(grid["gamma_sample_seed"]),
        gamma_sample_cap=int(grid["gamma_sample_cap"]),
        classifier_max_iter=int(grid["classifier_max_iter"]),
        threshold_policy=str(grid["threshold_policy"]),
        gate=GateConfig(
            mean_bacc_delta_min=float(gate["mean_bacc_delta_min"]),
            strict_center_wins_min=int(gate["strict_center_wins_min"]),
            worst_center_delta_min=float(gate["worst_center_delta_min"]),
            mean_positive_recall_delta_min_exclusive=float(
                gate["mean_positive_recall_delta_min_exclusive"]
            ),
            mean_specificity_delta_min=float(gate["mean_specificity_delta_min"]),
            supplemental_mean_delta_min_exclusive=float(
                gate["supplemental_mean_delta_min_exclusive"]
            ),
            supplemental_primary_deviation_max=float(
                gate["supplemental_primary_deviation_max"]
            ),
            supplemental_worst_center_delta_min=float(
                gate["supplemental_worst_center_delta_min"]
            ),
            bootstrap_replicates=int(gate["bootstrap_replicates"]),
            bootstrap_seed=int(gate["bootstrap_seed"]),
        ),
        runtime=RuntimeConfig(
            pair_jobs=int(runtime["pair_jobs"]),
            threads_per_job=int(runtime["threads_per_job"]),
            use_gpu=bool(runtime["use_gpu"]),
        ),
        claim_boundary=dict(claim),
    )
    _validate(config, grid)
    return config


def _validate(config: NonlinearProbeConfig, grid: Mapping[str, object]) -> None:
    if (
        config.name != EXPERIMENT_NAME
        or config.heldout_centers != MIDOGPP_ELIGIBLE_CENTERS
        or config.expected_feature_dim != EXPECTED_FEATURE_DIM
        or config.expected_train_rows != EXPECTED_TRAIN_ROWS
        or config.width_multipliers != WIDTH_MULTIPLIERS
        or config.components != COMPONENTS
        or config.logistic_cs != LOGISTIC_CS
        or config.primary_landmark_seed != PRIMARY_LANDMARK_SEED
        or config.stability_landmark_seeds != STABILITY_LANDMARK_SEEDS
        or config.gamma_sample_seed != GAMMA_SAMPLE_SEED
        or config.gamma_sample_cap != GAMMA_SAMPLE_CAP
        or config.classifier_max_iter != 5000
        or config.threshold_policy != "predict"
        or len(config.candidates) != EXPECTED_CANDIDATES
        or int(grid.get("expected_candidate_count", -1)) != EXPECTED_CANDIDATES
    ):
        raise ProtocolError("Uniform-B nonlinear grid drifted from its frozen protocol.")
    expected_gate = GateConfig(
        mean_bacc_delta_min=0.01,
        strict_center_wins_min=6,
        worst_center_delta_min=-0.01,
        mean_positive_recall_delta_min_exclusive=0.0,
        mean_specificity_delta_min=-0.01,
        supplemental_mean_delta_min_exclusive=0.0,
        supplemental_primary_deviation_max=0.01,
        supplemental_worst_center_delta_min=-0.01,
        bootstrap_replicates=2000,
        bootstrap_seed=42,
    )
    if config.gate != expected_gate:
        raise ProtocolError("Uniform-B nonlinear progression gate drifted.")
    if (
        config.runtime.pair_jobs < 1
        or config.runtime.threads_per_job < 1
        or config.runtime.use_gpu
    ):
        raise ProtocolError("Uniform-B nonlinear runtime contract drifted.")
    required_claim = {
        "claim_scope": "diagnostic_only",
        "diagnostic_surface_previously_inspected": True,
        "may_replace_canonical_reference": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "new_center_generalization_claimed": False,
        "validation_features_generated": False,
        "validation_predictions_generated": False,
    }
    if any(config.claim_boundary.get(key) != value for key, value in required_claim.items()):
        raise ProtocolError("Uniform-B nonlinear claim boundary drifted.")


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Uniform-B nonlinear config section {key!r} must be a mapping.")
    return value
