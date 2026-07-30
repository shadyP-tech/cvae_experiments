"""Frozen configuration for the reviewed Uniform-B canonical reference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ..classifier_grid import build_classifier_specs
from ..classifiers import ClassifierSpec, classifier_grid_hash
from ..matched_reference import CANONICAL_GRID_HASH, CANONICAL_GRID_SIZE
from ..protocol import ProtocolError
from ..schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS


CACHE_NAME = "uniform_b_canonical_train_cache_v1"
EXPERIMENT_NAME = "uniform_b_canonical_real_feature_reference_v1"
REPRESENTATION_ID = "annotation_jpeg_fixed_center_b_v3"
EXPECTED_FEATURE_DIM = 3840
EXPECTED_TRAIN_ROWS = 9648
PROMOTION_REVIEW_ID = "uniform_b_phase_b_promotion_review_2026_07_25"
CONFIRMATION_SUMMARY_SHA256 = (
    "3fdb44acae1d6ed9220efb48310a6f36647fb0d49f8eddd47c0a070558d284a8"
)
CONFIRMATION_PROTOCOL_SHA256 = (
    "8382653658656ff14410785308fc30c82d77c7ade81c10e8dc14f907e43a12e2"
)


@dataclass(frozen=True)
class UniformBCanonicalCacheConfig:
    name: str
    manifest_path: Path
    source_b_cache_root: Path
    cache_root: Path
    eligible_centers: tuple[str, ...]
    expected_train_rows: int
    expected_feature_dim: int


@dataclass(frozen=True)
class UniformBCanonicalReferenceConfig:
    name: str
    artifact_root: Path
    manifest_path: Path
    feature_cache_path: Path
    confirmation_root: Path
    heldout_centers: tuple[str, ...]
    experiment_seed: int
    classifier_seed: int
    expected_feature_dim: int
    classifier_specs: tuple[ClassifierSpec, ...]
    review: Mapping[str, object]
    claim_boundary: Mapping[str, object]


def load_uniform_b_canonical_cache_config(
    path: str | Path,
) -> UniformBCanonicalCacheConfig:
    payload = _payload(path)
    cache = _mapping(payload, "cache")
    inputs = _mapping(payload, "inputs")
    run = _mapping(payload, "run")
    config = UniformBCanonicalCacheConfig(
        name=str(cache["name"]),
        manifest_path=Path(str(inputs["manifest_path"])),
        source_b_cache_root=Path(str(inputs["source_b_cache_root"])),
        cache_root=Path(str(cache["root"])),
        eligible_centers=tuple(str(value) for value in run["eligible_centers"]),
        expected_train_rows=int(run["expected_train_rows"]),
        expected_feature_dim=int(run["expected_feature_dim"]),
    )
    if (
        config.name != CACHE_NAME
        or config.eligible_centers != MIDOGPP_ELIGIBLE_CENTERS
        or config.expected_train_rows != EXPECTED_TRAIN_ROWS
        or config.expected_feature_dim != EXPECTED_FEATURE_DIM
    ):
        raise ProtocolError("Uniform-B canonical cache protocol drifted.")
    return config


def load_uniform_b_canonical_reference_config(
    path: str | Path,
) -> UniformBCanonicalReferenceConfig:
    payload = _payload(path)
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    run = _mapping(payload, "run")
    grid = _mapping(payload, "classifier_grid")
    review = _mapping(payload, "promotion_review")
    claim = _mapping(payload, "claim_boundary")
    heldouts_raw = run.get("heldout_centers", "all")
    heldouts = (
        MIDOGPP_ELIGIBLE_CENTERS
        if str(heldouts_raw).lower() == "all"
        else tuple(str(value) for value in heldouts_raw)
    )
    specs = build_classifier_specs(
        c_grid=_csv(grid.get("c_grid")),
        penalties=_csv(grid.get("penalty")),
        solvers=_csv(grid.get("solver")),
        class_weights=_csv(grid.get("class_weight")),
        max_iters=_csv(grid.get("max_iter")),
        classifier_seed=int(run["classifier_seed"]),
    )
    config = UniformBCanonicalReferenceConfig(
        name=str(experiment["name"]),
        artifact_root=Path(str(experiment["artifact_root"])),
        manifest_path=Path(str(inputs["manifest_path"])),
        feature_cache_path=Path(str(inputs["feature_cache_path"])),
        confirmation_root=Path(str(inputs["confirmation_root"])),
        heldout_centers=tuple(heldouts),
        experiment_seed=int(run["experiment_seed"]),
        classifier_seed=int(run["classifier_seed"]),
        expected_feature_dim=int(run["expected_feature_dim"]),
        classifier_specs=tuple(specs),
        review=dict(review),
        claim_boundary=dict(claim),
    )
    _validate_reference_config(config, grid)
    return config


def _validate_reference_config(
    config: UniformBCanonicalReferenceConfig,
    grid: Mapping[str, object],
) -> None:
    if (
        config.name != EXPERIMENT_NAME
        or config.heldout_centers != MIDOGPP_ELIGIBLE_CENTERS
        or config.experiment_seed != 42
        or config.classifier_seed != 23
        or config.expected_feature_dim != EXPECTED_FEATURE_DIM
        or len(config.classifier_specs) != CANONICAL_GRID_SIZE
        or classifier_grid_hash(config.classifier_specs) != CANONICAL_GRID_HASH
        or int(grid.get("expected_candidate_count", -1)) != CANONICAL_GRID_SIZE
        or str(grid.get("expected_grid_hash", "")) != CANONICAL_GRID_HASH
        or str(grid.get("threshold_policy", "")) != "predict"
    ):
        raise ProtocolError("Uniform-B canonical reference protocol drifted.")
    required_review = {
        "review_id": PROMOTION_REVIEW_ID,
        "status": "approved",
        "approved_representation": REPRESENTATION_ID,
        "confirmation_decision": "CONFIRMED_WITHIN_CENTER",
        "confirmation_summary_sha256": CONFIRMATION_SUMMARY_SHA256,
        "confirmation_protocol_sha256": CONFIRMATION_PROTOCOL_SHA256,
        "test_split_consumed_for_representation_adoption": True,
        "classifier_locks_imported_from_diagnostics": False,
        "canonical_a_retained": True,
        "automatic_downstream_migration": False,
    }
    if any(config.review.get(key) != value for key, value in required_review.items()):
        raise ProtocolError("Uniform-B canonical promotion review drifted.")
    required_claim = {
        "claim_scope": "real_feature_transfer_only",
        "claim_role": "canonical_real_feature_reference",
        "new_center_generalization_claimed": False,
        "external_dataset_generalization_claimed": False,
        "test_split_available_for_fresh_representation_selection": False,
        "uses_cvae": False,
        "uses_router": False,
    }
    if any(config.claim_boundary.get(key) != value for key, value in required_claim.items()):
        raise ProtocolError("Uniform-B canonical reference claim boundary drifted.")


def _payload(path: str | Path) -> Mapping[str, object]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ProtocolError("Uniform-B canonical config must be a mapping.")
    return payload


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Uniform-B canonical config section {key!r} must be a mapping.")
    return value


def _csv(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join("none" if item is None else str(item) for item in value)
    return str(value)
