"""Strict, path-injectable configuration for the fresh exact-tail surface."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...protocol import ProtocolError
from ..metadata_compatibility.contracts import (
    DOMAIN_MAPPING_MEMBER,
    DOMAIN_MAPPING_SHA256,
    INPUT_ARTIFACT_ID as METADATA_PROFILE_ARTIFACT_ID,
)
from .contracts import (
    CENTERS,
    CLAIM_SCOPE,
    DEVELOPMENT_CACHE_ARTIFACT_ID,
    DEVELOPMENT_MANIFEST_ARTIFACT_ID,
    DEVELOPMENT_RESERVATION_ARTIFACT_ID,
    EXPECTED_COARSE_TASK_COUNT,
    EXPECTED_PREDICTION_CELL_COUNT,
    EXPECTED_SOURCE_STREAM_COUNT,
    EXPECTED_UTILITY_ROW_COUNT,
    EXPERIMENT_ID,
    GENERATION_SEEDS,
    MINIMUM_SUPPORT_CASE_COUNT,
    OUTPUT_ARTIFACT_ID,
    SOURCE_PREFIX_ROWS_PER_CLASS,
    TRAINING_SEEDS,
)
from .runtime import WorkstationRuntimePlan


STAGE_ID = "60_routing_and_composition"
CONFIG_SCHEMA_VERSION = "midogpp_exact_tail_utility_surface_config_v1"
FRESH_ATTESTATION_SCHEMA = "midogpp_utility_aligned_fresh_reservation_v1"
# Freeze this experiment family's graph locally.  Source-inner orchestration is
# a design reference, never an artifact-identity dependency.
EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"

CLASSIFIER = ClassifierSpec(
    C=0.01,
    penalty="l2",
    solver="lbfgs",
    max_iter=3000,
    class_weight=None,
    random_state=23,
    l1_ratio=None,
    threshold_policy="predict",
    scaler_fit="synthetic_train_only",
)

INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    DEVELOPMENT_RESERVATION_ARTIFACT_ID,
    DEVELOPMENT_CACHE_ARTIFACT_ID,
    DEVELOPMENT_MANIFEST_ARTIFACT_ID,
    METADATA_PROFILE_ARTIFACT_ID,
)


@dataclass(frozen=True)
class ExactTailUtilitySurfaceConfig:
    artifact_root: Path
    expert_bank_root: Path
    generation_lock_root: Path
    development_reservation_root: Path
    development_cache_root: Path
    development_manifest_path: Path
    metadata_profile_root: Path
    reservation_attestation_path: Path
    experiment_id: str
    output_artifact_id: str
    input_artifact_ids: tuple[str, ...]
    protocol: Mapping[str, object]
    classifier: ClassifierSpec
    runtime: WorkstationRuntimePlan
    claim_boundary: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol", MappingProxyType(dict(self.protocol)))
        object.__setattr__(
            self, "claim_boundary", MappingProxyType(dict(self.claim_boundary))
        )

    @property
    def contract_hash(self) -> str:
        return stable_hash(
            {
                "schema_version": CONFIG_SCHEMA_VERSION,
                "experiment_id": self.experiment_id,
                "output_artifact_id": self.output_artifact_id,
                "input_artifact_ids": list(self.input_artifact_ids),
                "protocol": dict(self.protocol),
                "classifier": self.classifier.to_payload(),
                "claim_boundary": dict(self.claim_boundary),
            }
        )


@dataclass(frozen=True)
class FreshInputAttestation:
    reservation_index_hash: str
    development_cache_binding_hash: str
    development_manifest_sha256: str
    target_evaluation_binding_hash: str
    metadata_profile_sha256: str
    attestation_sha256: str


def load_exact_tail_utility_surface_config(
    path: str | Path,
) -> ExactTailUtilitySurfaceConfig:
    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError(f"Cannot read exact-tail config: {config_path}.") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("Exact-tail config must be a mapping.")
    _exact_keys(
        payload,
        {"schema_version", "experiment", "inputs", "protocol", "classifier", "runtime", "claim_boundary"},
        "top-level",
    )
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ProtocolError("Exact-tail config schema drifted.")
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    base = config_path.parent
    config = ExactTailUtilitySurfaceConfig(
        artifact_root=_path(base, experiment.get("artifact_root"), "artifact root"),
        expert_bank_root=_path(base, inputs.get("expert_bank_root"), "expert bank"),
        generation_lock_root=_path(base, inputs.get("generation_lock_root"), "generation lock"),
        development_reservation_root=_path(
            base, inputs.get("development_reservation_root"), "development reservation"
        ),
        development_cache_root=_path(
            base, inputs.get("development_cache_root"), "development cache"
        ),
        development_manifest_path=_path(
            base, inputs.get("development_manifest_path"), "development manifest"
        ),
        metadata_profile_root=_path(
            base, inputs.get("metadata_profile_root"), "metadata profile"
        ),
        reservation_attestation_path=_path(
            base, inputs.get("reservation_attestation_path"), "fresh attestation"
        ),
        experiment_id=str(experiment.get("id", "")),
        output_artifact_id=str(experiment.get("output_artifact_id", "")),
        input_artifact_ids=tuple(str(value) for value in inputs.get("artifact_ids", ())),
        protocol=dict(_mapping(payload, "protocol")),
        classifier=_classifier(_mapping(payload, "classifier")),
        runtime=_runtime(_mapping(payload, "runtime")),
        claim_boundary=dict(_mapping(payload, "claim_boundary")),
    )
    _validate_config(config, experiment=experiment, inputs=inputs)
    return config


def validate_fresh_inputs_ready(
    config: ExactTailUtilitySurfaceConfig,
) -> FreshInputAttestation:
    """Block planned placeholders before GPU initialization or label access."""

    if config.protocol.get("fresh_reservation_status") != "ready":
        raise ProtocolError(
            "Exact-tail experiment remains planned; a fresh case-disjoint "
            "development reservation must be promoted to ready."
        )
    for path, role in (
        (config.expert_bank_root, "expert bank"),
        (config.generation_lock_root, "GenerationLock"),
        (config.development_reservation_root, "development reservation"),
        (config.development_cache_root, "development cache"),
        (config.development_manifest_path, "development manifest"),
        (config.metadata_profile_root / DOMAIN_MAPPING_MEMBER, "metadata profile mapping"),
        (config.reservation_attestation_path, "fresh reservation attestation"),
    ):
        if not path.exists():
            raise ProtocolError(f"Exact-tail required {role} is absent: {path}.")
    try:
        raw = json.loads(config.reservation_attestation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Exact-tail fresh reservation attestation is unreadable.") from exc
    required = {
        "schema_version",
        "status",
        "dataset_family",
        "development_reservation_artifact_id",
        "development_cache_artifact_id",
        "development_manifest_artifact_id",
        "metadata_profile_artifact_id",
        "center_universe",
        "whole_case_support_evaluation_disjoint",
        "minimum_independent_support_cases_per_query",
        "development_target_evaluation_disjoint",
        "reservation_cache_and_index_contain_labels",
        "scoring_manifest_contains_only_development_evaluation_rows",
        "scoring_manifest_labels_opened_before_global_seal",
        "consumed_stage60_or_stage90_rows_reused",
        "reservation_index_hash",
        "development_cache_binding_hash",
        "development_manifest_sha256",
        "target_evaluation_binding_hash",
        "metadata_profile_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ProtocolError("Exact-tail fresh reservation attestation schema drifted.")
    fixed = {
        "schema_version": FRESH_ATTESTATION_SCHEMA,
        "status": "READY",
        "dataset_family": "MIDOG++",
        "development_reservation_artifact_id": DEVELOPMENT_RESERVATION_ARTIFACT_ID,
        "development_cache_artifact_id": DEVELOPMENT_CACHE_ARTIFACT_ID,
        "development_manifest_artifact_id": DEVELOPMENT_MANIFEST_ARTIFACT_ID,
        "metadata_profile_artifact_id": METADATA_PROFILE_ARTIFACT_ID,
        "center_universe": list(CENTERS),
        "whole_case_support_evaluation_disjoint": True,
        "minimum_independent_support_cases_per_query": MINIMUM_SUPPORT_CASE_COUNT,
        "development_target_evaluation_disjoint": True,
        "reservation_cache_and_index_contain_labels": False,
        "scoring_manifest_contains_only_development_evaluation_rows": True,
        "scoring_manifest_labels_opened_before_global_seal": False,
        "consumed_stage60_or_stage90_rows_reused": False,
    }
    if any(raw.get(key) != value for key, value in fixed.items()):
        raise ProtocolError("Exact-tail fresh reservation attestation failed closed.")
    manifest_sha = _sha256_file(config.development_manifest_path)
    if raw.get("development_manifest_sha256") != manifest_sha:
        raise ProtocolError("Exact-tail development manifest hash drifted.")
    metadata_sha = _sha256_file(config.metadata_profile_root / DOMAIN_MAPPING_MEMBER)
    if metadata_sha != DOMAIN_MAPPING_SHA256 or raw.get("metadata_profile_sha256") != metadata_sha:
        raise ProtocolError("Exact-tail metadata profile mapping hash drifted.")
    for key in ("reservation_index_hash", "target_evaluation_binding_hash"):
        _hash(str(raw.get(key, "")), key, {16, 64})
    _hash(
        str(raw.get("development_cache_binding_hash", "")),
        "development_cache_binding_hash",
        {64},
    )
    return FreshInputAttestation(
        reservation_index_hash=str(raw["reservation_index_hash"]),
        development_cache_binding_hash=str(raw["development_cache_binding_hash"]),
        development_manifest_sha256=manifest_sha,
        target_evaluation_binding_hash=str(raw["target_evaluation_binding_hash"]),
        metadata_profile_sha256=metadata_sha,
        attestation_sha256=_sha256_file(config.reservation_attestation_path),
    )


def _validate_config(
    config: ExactTailUtilitySurfaceConfig,
    *,
    experiment: Mapping[str, object],
    inputs: Mapping[str, object],
) -> None:
    _exact_keys(experiment, {"id", "artifact_root", "output_artifact_id"}, "experiment")
    _exact_keys(
        inputs,
        {
            "artifact_ids",
            "expert_bank_root",
            "generation_lock_root",
            "development_reservation_root",
            "development_cache_root",
            "development_manifest_path",
            "metadata_profile_root",
            "reservation_attestation_path",
        },
        "inputs",
    )
    if (
        config.experiment_id != EXPERIMENT_ID
        or config.output_artifact_id != OUTPUT_ARTIFACT_ID
        or config.input_artifact_ids != INPUT_ARTIFACT_IDS
    ):
        raise ProtocolError("Exact-tail experiment or artifact identity drifted.")
    expected_protocol = {
        "dataset_family": "MIDOG++",
        "stage": STAGE_ID,
        "center_universe": list(CENTERS),
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "inner_geometry": "seven_by_144_base_plus_126_single_source_tail",
        "source_prefix_rows_per_class": SOURCE_PREFIX_ROWS_PER_CLASS,
        "source_stream_count": EXPECTED_SOURCE_STREAM_COUNT,
        "coarse_task_count": EXPECTED_COARSE_TASK_COUNT,
        "prediction_cell_count": EXPECTED_PREDICTION_CELL_COUNT,
        "utility_row_count": EXPECTED_UTILITY_ROW_COUNT,
        "outer_target_excluded_from_query_and_source_roles": True,
        "whole_case_support_evaluation_disjoint": True,
        "minimum_independent_support_cases_per_query": MINIMUM_SUPPORT_CASE_COUNT,
        "development_target_evaluation_disjoint": True,
        "all_predictions_sealed_before_development_labels": True,
        "development_labels_used_for_scoring_only": True,
        "target_support_labels_used": False,
        "target_evaluation_labels_used": False,
        "source_experts_updated": False,
        "seed_selection_performed": False,
        "fresh_reservation_status": config.protocol.get("fresh_reservation_status"),
    }
    if config.protocol.get("fresh_reservation_status") not in {"planned", "ready"}:
        raise ProtocolError("Exact-tail fresh reservation status is invalid.")
    if dict(config.protocol) != expected_protocol:
        raise ProtocolError("Exact-tail protocol section drifted.")
    if config.classifier.to_payload() != CLASSIFIER.to_payload():
        raise ProtocolError("Exact-tail classifier contract drifted.")
    if dict(config.claim_boundary) != {
        "claim_scope": CLAIM_SCOPE,
        "utility_learnability_only": True,
        "routing_improvement_claimed": False,
        "target_downstream_utility_claimed": False,
        "stage90_artifacts_used": False,
        "may_feed_only_locked_utility_aligned_policy": True,
    }:
        raise ProtocolError("Exact-tail claim boundary drifted.")


def _classifier(raw: Mapping[str, object]) -> ClassifierSpec:
    expected = CLASSIFIER.to_payload()
    if dict(raw) != expected:
        raise ProtocolError("Exact-tail classifier YAML drifted.")
    return CLASSIFIER


def _runtime(raw: Mapping[str, object]) -> WorkstationRuntimePlan:
    expected = WorkstationRuntimePlan().to_payload()
    if dict(raw) != expected:
        raise ProtocolError("Exact-tail runtime YAML drifted.")
    return WorkstationRuntimePlan()


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Exact-tail config section {key!r} must be a mapping.")
    return value


def _exact_keys(raw: Mapping[str, object], expected: set[str], role: str) -> None:
    if {str(key) for key in raw} != expected:
        raise ProtocolError(f"Exact-tail {role} config keys drifted.")


def _path(base: Path, value: object, role: str) -> Path:
    rendered = str(value or "")
    if not rendered:
        raise ProtocolError(f"Exact-tail {role} path is empty.")
    if rendered.startswith(("artifact://", "output://")):
        return Path(rendered)
    path = Path(rendered)
    return path if path.is_absolute() else (base / path).resolve()


def _hash(value: str, role: str, lengths: set[int]) -> None:
    if len(value) not in lengths or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ProtocolError(f"Exact-tail attested {role} is malformed.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "CLASSIFIER",
    "CONFIG_SCHEMA_VERSION",
    "FRESH_ATTESTATION_SCHEMA",
    "INPUT_ARTIFACT_IDS",
    "STAGE_ID",
    "ExactTailUtilitySurfaceConfig",
    "FreshInputAttestation",
    "load_exact_tail_utility_surface_config",
    "validate_fresh_inputs_ready",
)
