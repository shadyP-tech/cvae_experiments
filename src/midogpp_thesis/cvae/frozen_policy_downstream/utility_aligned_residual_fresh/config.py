"""Strict configuration for the utility-aligned fresh Stage-70 evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...protocol import ProtocolError
from ..fresh_runtime_contract import (
    CLASSIFIER_THREADS_PER_WORKER,
    CLASSIFIER_WORKERS,
    DOWNSTREAM_CLASSIFIER,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    MINIMUM_ARTIFACT_DISK_FREE_BYTES,
    MINIMUM_GPU_FREE_MIB_PER_DEVICE,
    MINIMUM_LOGICAL_CPU_COUNT,
    MINIMUM_PHYSICAL_RAM_BYTES,
    OPTIONAL_LOCAL_SCRATCH_ROOT,
    SOURCE_BLOCK_PER_CLASS,
    WORKSTATION_PROFILE,
    canonical_runtime_payload,
)
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    GENERATION_SEEDS,
    GLOBAL_ACTION_ID,
    PERMUTATION_ACTION_ID,
    PRIMARY_ENDPOINT,
    ROUTED_ACTION_ID,
    TRAINING_SEEDS,
    UNIFORM_ACTION_ID,
)


EXPERIMENT_ID = (
    "midogpp.frozen_policy_downstream."
    "uniform_b_v2_utility_aligned_residual_fresh.v1"
)
EXPERIMENT_NAME = "uniform_b_v2_utility_aligned_residual_fresh_v1"
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_utility_aligned_residual_fresh_v1"
)
PUBLICATION_STATUS = "BLOCKED_PENDING_FRESH_SURFACE"

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
POLICY_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_utility_aligned_residual_policy_lock_v1"
)
RESERVATION_ARTIFACT_ID = "midogpp_utility_aligned_fresh_target_reservation_v1"
TARGET_CACHE_ARTIFACT_ID = "midogpp_utility_aligned_fresh_target_cache_v1"
SCORING_MANIFEST_ARTIFACT_ID = "midogpp_utility_aligned_fresh_target_manifest_v1"
INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    POLICY_ARTIFACT_ID,
    RESERVATION_ARTIFACT_ID,
    TARGET_CACHE_ARTIFACT_ID,
    SCORING_MANIFEST_ARTIFACT_ID,
)


def canonical_protocol_payload() -> dict[str, object]:
    return {
        "dataset_family": "MIDOG++",
        "feature_backbone": "Virchow2",
        "feature_frame": "annotation_jpeg_fixed_center_b_v3",
        "stage": "70_frozen_policy_downstream",
        "centers": list(CENTERS),
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "target_support_evaluation_case_disjoint": True,
        "minimum_independent_support_cases_per_target": 8,
        "target_expert_excluded": True,
        "policy_and_reservation_frozen_before_target_cache": True,
        "active_fresh_reservation_required_at_runtime": True,
        "every_logical_action_target_seed_prediction_sealed": True,
        "composition_aliases_deduplicated_before_classifier_fit": True,
        "target_labels_available_to_generation_or_prediction": False,
        "labels_opened_for_scoring_only_after_global_seal": True,
        "policy_update_after_scoring_forbidden": True,
    }


def canonical_evaluation_payload() -> dict[str, object]:
    return {
        "main_action_ids": [
            BASE_ACTION_ID,
            UNIFORM_ACTION_ID,
            GLOBAL_ACTION_ID,
            ROUTED_ACTION_ID,
        ],
        "permutation_control_action_id": PERMUTATION_ACTION_ID,
        "single_source_tail_action_namespace": "single_source_tail",
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_threshold": 0.5,
        "inference_unit": "target_center",
        "effective_sample_size": len(CENTERS),
        "primary_contrasts": ["R-B", "R-G_delta", "R-U"],
        "permutation_contrast": "R-P",
        "secondary_contrasts": ["U-B", "G_delta-B"],
        "success_requires_positive_one_sided_lcb": [
            "R-B",
            "R-G_delta",
            "R-U",
            "R-P",
        ],
        "oracle_diagnostics_terminal_only": True,
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "claim_scope": "synthetic_downstream_utility",
        "fresh_confirmatory_only": True,
        "consumed_test_used": False,
        "consumed_validation_used": False,
        "consumed_stage70_used": False,
        "consumed_stage90_used": False,
        "oracle_may_update_policy": False,
        "negative_result_is_valid": True,
    }


@dataclass(frozen=True)
class UtilityAlignedResidualFreshConfig:
    source_path: Path
    artifact_root: Path
    expert_bank_root: Path
    generation_lock_root: Path
    policy_root: Path
    fresh_reservation_path: Path
    fresh_target_cache_root: Path
    fresh_scoring_manifest_path: Path
    expected_bank_lock_hash: str
    expected_generation_lock_hash: str
    protocol: Mapping[str, object]
    evaluation: Mapping[str, object]
    classifier: ClassifierSpec
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]
    contract_hash: str

    @property
    def experiment_id(self) -> str:
        return EXPERIMENT_ID

    @property
    def output_artifact_id(self) -> str:
        return OUTPUT_ARTIFACT_ID

    @property
    def input_artifact_ids(self) -> tuple[str, ...]:
        return INPUT_ARTIFACT_IDS


def load_utility_aligned_residual_fresh_config(
    path: str | Path,
) -> UtilityAlignedResidualFreshConfig:
    source = Path(path).expanduser().resolve()
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read utility-aligned fresh config.") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "experiment",
        "inputs",
        "protocol",
        "evaluation",
        "classifier",
        "runtime",
        "claim_boundary",
    }:
        raise ProtocolError("Utility-aligned fresh config fields drifted.")
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    protocol = _mapping(payload, "protocol")
    evaluation = _mapping(payload, "evaluation")
    classifier_raw = _mapping(payload, "classifier")
    runtime = _mapping(payload, "runtime")
    claim = _mapping(payload, "claim_boundary")

    if (
        set(experiment) != {"id", "name", "artifact_root", "status"}
        or
        experiment.get("id") != EXPERIMENT_ID
        or experiment.get("name") != EXPERIMENT_NAME
        or experiment.get("status") != PUBLICATION_STATUS
    ):
        raise ProtocolError("Utility-aligned fresh experiment identity drifted.")
    identity = {
        "expert_bank_artifact_id": EXPERT_BANK_ARTIFACT_ID,
        "generation_lock_artifact_id": GENERATION_LOCK_ARTIFACT_ID,
        "policy_artifact_id": POLICY_ARTIFACT_ID,
        "reservation_artifact_id": RESERVATION_ARTIFACT_ID,
        "target_cache_artifact_id": TARGET_CACHE_ARTIFACT_ID,
        "scoring_manifest_artifact_id": SCORING_MANIFEST_ARTIFACT_ID,
        "expected_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
        "expected_generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
    }
    locations = {
        "expert_bank_root",
        "generation_lock_root",
        "policy_root",
        "fresh_reservation_path",
        "fresh_target_cache_root",
        "fresh_scoring_manifest_path",
    }
    if set(inputs) != set(identity).union(locations) or any(
        inputs.get(key) != value for key, value in identity.items()
    ):
        raise ProtocolError("Utility-aligned fresh input identities drifted.")
    if dict(protocol) != canonical_protocol_payload():
        raise ProtocolError("Utility-aligned fresh protocol drifted.")
    if dict(evaluation) != canonical_evaluation_payload():
        raise ProtocolError("Utility-aligned fresh evaluation contract drifted.")
    if dict(runtime) != canonical_runtime_payload():
        raise ProtocolError("Utility-aligned fresh workstation schedule drifted.")
    if dict(claim) != canonical_claim_boundary_payload():
        raise ProtocolError("Utility-aligned fresh claim boundary drifted.")
    classifier = _classifier(classifier_raw)
    if classifier != DOWNSTREAM_CLASSIFIER:
        raise ProtocolError("Utility-aligned fresh classifier drifted.")

    scientific = {
        "experiment_id": EXPERIMENT_ID,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "protocol": dict(protocol),
        "evaluation": dict(evaluation),
        "classifier": classifier.to_payload(),
        "claim_boundary": dict(claim),
    }
    return UtilityAlignedResidualFreshConfig(
        source_path=source,
        artifact_root=_path(source.parent, experiment.get("artifact_root")),
        expert_bank_root=_path(source.parent, inputs["expert_bank_root"]),
        generation_lock_root=_path(source.parent, inputs["generation_lock_root"]),
        policy_root=_path(source.parent, inputs["policy_root"]),
        fresh_reservation_path=_path(source.parent, inputs["fresh_reservation_path"]),
        fresh_target_cache_root=_path(source.parent, inputs["fresh_target_cache_root"]),
        fresh_scoring_manifest_path=_path(
            source.parent, inputs["fresh_scoring_manifest_path"]
        ),
        expected_bank_lock_hash=EXPECTED_BANK_LOCK_HASH,
        expected_generation_lock_hash=EXPECTED_GENERATION_LOCK_HASH,
        protocol=dict(protocol),
        evaluation=dict(evaluation),
        classifier=classifier,
        runtime=dict(runtime),
        claim_boundary=dict(claim),
        contract_hash=stable_hash(scientific),
    )


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Utility-aligned fresh config lacks {key!r}.")
    return value


def _path(base: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError("Utility-aligned fresh artifact path is invalid.")
    rendered = value.strip()
    if rendered.startswith(("artifact://", "output://")):
        return Path(rendered)
    path = Path(rendered).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _classifier(payload: Mapping[str, object]) -> ClassifierSpec:
    if set(payload) != set(DOWNSTREAM_CLASSIFIER.to_payload()):
        raise ProtocolError("Utility-aligned fresh classifier fields drifted.")
    try:
        return ClassifierSpec(
            family=str(payload["family"]),
            C=float(payload["C"]),
            penalty=str(payload["penalty"]),
            solver=str(payload["solver"]),
            max_iter=int(payload["max_iter"]),
            class_weight=(
                None if payload["class_weight"] is None else str(payload["class_weight"])
            ),
            random_state=int(payload["random_state"]),
            l1_ratio=(None if payload["l1_ratio"] is None else float(payload["l1_ratio"])),
            threshold_policy=str(payload["threshold_policy"]),
            scaler_fit=str(payload["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Utility-aligned fresh classifier is invalid.") from exc


__all__ = (
    "CLASSIFIER_THREADS_PER_WORKER",
    "CLASSIFIER_WORKERS",
    "DOWNSTREAM_CLASSIFIER",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "EXPERT_BANK_ARTIFACT_ID",
    "GENERATION_LOCK_ARTIFACT_ID",
    "INPUT_ARTIFACT_IDS",
    "OPTIONAL_LOCAL_SCRATCH_ROOT",
    "OUTPUT_ARTIFACT_ID",
    "POLICY_ARTIFACT_ID",
    "RESERVATION_ARTIFACT_ID",
    "SCORING_MANIFEST_ARTIFACT_ID",
    "SOURCE_BLOCK_PER_CLASS",
    "TARGET_CACHE_ARTIFACT_ID",
    "UtilityAlignedResidualFreshConfig",
    "WORKSTATION_PROFILE",
    "canonical_claim_boundary_payload",
    "canonical_evaluation_payload",
    "canonical_protocol_payload",
    "canonical_runtime_payload",
    "load_utility_aligned_residual_fresh_config",
)
