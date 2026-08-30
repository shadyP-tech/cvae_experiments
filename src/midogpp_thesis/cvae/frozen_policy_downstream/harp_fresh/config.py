"""Path-independent closed Stage-70 configuration for fresh HARP replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError
from ...runtime.harp_probability_menu.hashing import canonical_sha256
from ...runtime.frozen_source_streams import SOURCE_ROWS_PER_CLASS
from ..fresh_runtime_contract import (
    CLASSIFIER_THREADS_PER_WORKER,
    CLASSIFIER_WORKERS,
    DOWNSTREAM_CLASSIFIER,
    MINIMUM_ARTIFACT_DISK_FREE_BYTES,
    MINIMUM_GPU_FREE_MIB_PER_DEVICE,
    MINIMUM_LOGICAL_CPU_COUNT,
    MINIMUM_PHYSICAL_RAM_BYTES,
    OPTIONAL_LOCAL_SCRATCH_ROOT,
    WORKSTATION_PROFILE,
)


CONFIG_SCHEMA = "midogpp_harp_fresh_stage70_config_v1"
HARP_SOURCE_CACHE_FORMAT = (
    "frozen_source_streams_single_float32_npy_memmap_lock_index"
)
EXPERIMENT_ID = "midogpp.frozen_policy_downstream.uniform_b_v2_harp_fresh.v1"
OUTPUT_ARTIFACT_ID = "midogpp_output_uniform_b_v2_harp_fresh_v1"
EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
POLICY_ARTIFACT_ID = "midogpp_output_uniform_b_v2_harp_policy_lock_v1"
RESERVATION_ARTIFACT_ID = "midogpp_harp_fresh_target_reservation_v1"
TARGET_CACHE_ARTIFACT_ID = "midogpp_harp_fresh_target_cache_v1"
SCORING_MANIFEST_ARTIFACT_ID = "midogpp_harp_fresh_target_scoring_manifest_v1"
INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    POLICY_ARTIFACT_ID,
    RESERVATION_ARTIFACT_ID,
    TARGET_CACHE_ARTIFACT_ID,
    SCORING_MANIFEST_ARTIFACT_ID,
)
PATH_KEYS = (
    "expert_bank_root",
    "generation_lock_root",
    "policy_root",
    "reservation_root",
    "target_cache_root",
    "scoring_manifest_path",
)
_PATH_VALUES = (
    f"artifact://{EXPERT_BANK_ARTIFACT_ID}",
    f"artifact://{GENERATION_LOCK_ARTIFACT_ID}",
    f"artifact://{POLICY_ARTIFACT_ID}",
    f"artifact://{RESERVATION_ARTIFACT_ID}",
    f"artifact://{TARGET_CACHE_ARTIFACT_ID}",
    f"artifact://{SCORING_MANIFEST_ARTIFACT_ID}/manifest.csv",
)


def canonical_harp_runtime_payload() -> dict[str, object]:
    """Return HARP's complete physical runtime without inheriting peer semantics."""

    return {
        "workstation_profile": WORKSTATION_PROFILE,
        "generation_devices": ["cuda:0", "cuda:1"],
        "cuda_visible_devices": "0,1",
        "generation_workers_per_device": 1,
        "source_block_per_class": SOURCE_ROWS_PER_CLASS,
        "classifier_workers": CLASSIFIER_WORKERS,
        "classifier_threads_per_worker": CLASSIFIER_THREADS_PER_WORKER,
        "multiprocessing_start_method": "spawn",
        "tf32_disabled_in_gpu_workers": True,
        "amp_enabled": False,
        "parent_cuda_context_forbidden": True,
        "gpu_and_cpu_phases_disjoint": True,
        "source_cache_format": HARP_SOURCE_CACHE_FORMAT,
        "prediction_cache_format": "float32_npy_memmap",
        "persistent_cache_policy": "hash_validated_resume",
        "optional_local_scratch_root": OPTIONAL_LOCAL_SCRATCH_ROOT,
        "canonical_publication_requires_validated_atomic_copy": True,
        "minimum_logical_cpu_count": MINIMUM_LOGICAL_CPU_COUNT,
        "minimum_physical_ram_bytes": MINIMUM_PHYSICAL_RAM_BYTES,
        "minimum_artifact_disk_free_bytes": MINIMUM_ARTIFACT_DISK_FREE_BYTES,
        "minimum_gpu_free_mib_per_device": MINIMUM_GPU_FREE_MIB_PER_DEVICE,
    }


def canonical_protocol_payload() -> dict[str, object]:
    return {
        "dataset_family": "MIDOG++",
        "stage": "70_frozen_policy_downstream",
        "centers": list(CENTERS),
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "fresh_reservation_frozen_before_target_cache": True,
        "harp_policy_frozen_before_target_cache": True,
        "support_evaluation_cases_globally_disjoint": True,
        "outer_target_expert_excluded": True,
        "complete_B_U_and_Hxe_menu_before_routing": True,
        "all_routes_and_routed_vectors_sealed_before_labels": True,
        "evaluation_labels_used_for_scoring_only_after_global_seal": True,
        "evaluation_labels_available_to_generation_prediction_or_routing": False,
        "policy_update_after_scoring_forbidden": True,
    }


def canonical_evaluation_payload() -> dict[str, object]:
    return {
        "primary_endpoint": "exact_nine_probability_ensemble_balanced_accuracy",
        "baseline_action": "B",
        "matched_budget_reference_action": "U",
        "candidate_action_namespace": "Hxe",
        "lambda_grid": [0.25, 0.5, 0.75, 1.0],
        "lambda_semantics": (
            "post_classifier_predictive_probability_ensemble_not_generated_distribution"
        ),
        "physical_expert_routing_primary_lambda": 1.0,
        "report_budget_effect_U_minus_B": True,
        "report_physical_routing_effect_Hxe_lambda1_minus_U": True,
        "report_predictive_routing_effect_selected_lambda_minus_U": True,
        "report_full_sealed_action_oracle_diagnostics": True,
        "oracle_diagnostics_may_feed_policy": False,
        "threshold": 0.5,
        "inference_unit": "target_center",
        "inference_unit_count": len(CENTERS),
        "case_equal_weighting_within_center": True,
        "center_equal_weighting": True,
        "seed_cells_independent_inference_units": False,
        "report_two_sided_95_percent_t_interval": True,
        "report_one_sided_95_percent_lower_bound": True,
        "report_wins_ties_losses": True,
        "proper_losses": ["brier", "log_loss"],
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "claim_scope": "synthetic_downstream_utility",
        "fresh_reserved_surface_only": True,
        "consumed_test_used": False,
        "consumed_validation_used": False,
        "consumed_stage90_used": False,
        "policy_selection_after_target_labels_forbidden": True,
        "success_claim_requires_completed_validated_bundle": True,
        "negative_result_is_valid": True,
    }


@dataclass(frozen=True, kw_only=True)
class HarpFreshStage70Config:
    source_path: Path
    locations: Mapping[str, str]
    protocol: Mapping[str, object]
    classifier: ClassifierSpec
    runtime: Mapping[str, object]
    evaluation: Mapping[str, object]
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


def _mapping(raw: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Fresh HARP config lacks {name!r}.")
    return value


def _classifier(raw: Mapping[str, object]) -> ClassifierSpec:
    if set(raw) != set(DOWNSTREAM_CLASSIFIER.to_payload()):
        raise ProtocolError("Fresh HARP classifier keys drifted.")
    try:
        value = ClassifierSpec(
            family=str(raw["family"]),
            C=float(raw["C"]),
            penalty=str(raw["penalty"]),
            solver=str(raw["solver"]),
            max_iter=int(raw["max_iter"]),
            class_weight=(None if raw["class_weight"] is None else str(raw["class_weight"])),
            random_state=int(raw["random_state"]),
            l1_ratio=(None if raw["l1_ratio"] is None else float(raw["l1_ratio"])),
            threshold_policy=str(raw["threshold_policy"]),
            scaler_fit=str(raw["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Fresh HARP classifier is malformed.") from exc
    if value != DOWNSTREAM_CLASSIFIER:
        raise ProtocolError("Fresh HARP classifier escaped the frozen contract.")
    return value


def _require_exact(
    observed: Mapping[str, object], expected: Mapping[str, object], role: str
) -> None:
    if dict(observed) != dict(expected):
        raise ProtocolError(f"Fresh HARP {role} contract drifted.")


def _reject_placeholders(value: object) -> None:
    tokens = {"PENDING", "PLACEHOLDER", "TODO", "TBD", "REPLACE_ME", "CHANGEME"}
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_placeholders(key)
            _reject_placeholders(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _reject_placeholders(item)
    elif type(value) is str and value.upper() in tokens:
        raise ProtocolError("Fresh HARP config contains a placeholder.")


def load_harp_fresh_stage70_config(path: str | Path) -> HarpFreshStage70Config:
    """Validate identity URIs while keeping workstation paths out of the hash."""

    source = Path(path).resolve()
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read fresh HARP Stage-70 config.") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("Fresh HARP Stage-70 config must be a mapping.")
    expected_top = {
        "schema_version",
        "experiment",
        "inputs",
        "protocol",
        "classifier",
        "runtime",
        "evaluation",
        "claim_boundary",
    }
    if set(payload) != expected_top or payload.get("schema_version") != CONFIG_SCHEMA:
        raise ProtocolError("Fresh HARP Stage-70 top-level schema drifted.")
    _reject_placeholders(payload)
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    protocol = _mapping(payload, "protocol")
    classifier_raw = _mapping(payload, "classifier")
    runtime = _mapping(payload, "runtime")
    evaluation = _mapping(payload, "evaluation")
    boundary = _mapping(payload, "claim_boundary")
    if set(experiment) != {"id", "artifact_root", "output_artifact_id"} or dict(experiment) != {
        "id": EXPERIMENT_ID,
        "artifact_root": f"output://{OUTPUT_ARTIFACT_ID}",
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
    }:
        raise ProtocolError("Fresh HARP experiment identity drifted.")
    if set(inputs) != {"artifact_ids", "paths"}:
        raise ProtocolError("Fresh HARP input schema drifted.")
    if tuple(inputs.get("artifact_ids", ())) != INPUT_ARTIFACT_IDS:
        raise ProtocolError("Fresh HARP input artifact order drifted.")
    locations = inputs.get("paths")
    if (
        not isinstance(locations, Mapping)
        or tuple(locations) != PATH_KEYS
        or tuple(locations.values()) != _PATH_VALUES
    ):
        raise ProtocolError("Fresh HARP path identities drifted.")
    _require_exact(protocol, canonical_protocol_payload(), "protocol")
    _require_exact(runtime, canonical_harp_runtime_payload(), "runtime")
    _require_exact(evaluation, canonical_evaluation_payload(), "evaluation")
    _require_exact(boundary, canonical_claim_boundary_payload(), "claim boundary")
    classifier = _classifier(classifier_raw)
    scientific = {
        "schema_version": CONFIG_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "protocol": dict(protocol),
        "classifier": classifier.to_payload(),
        "runtime": dict(runtime),
        "evaluation": dict(evaluation),
        "claim_boundary": dict(boundary),
    }
    return HarpFreshStage70Config(
        source_path=source,
        locations=MappingProxyType({key: str(locations[key]) for key in PATH_KEYS}),
        protocol=MappingProxyType(dict(protocol)),
        classifier=classifier,
        runtime=MappingProxyType(dict(runtime)),
        evaluation=MappingProxyType(dict(evaluation)),
        claim_boundary=MappingProxyType(dict(boundary)),
        contract_hash=canonical_sha256(scientific),
    )


__all__ = (
    "CONFIG_SCHEMA",
    "EXPERIMENT_ID",
    "EXPERT_BANK_ARTIFACT_ID",
    "GENERATION_LOCK_ARTIFACT_ID",
    "HARP_SOURCE_CACHE_FORMAT",
    "HarpFreshStage70Config",
    "INPUT_ARTIFACT_IDS",
    "OUTPUT_ARTIFACT_ID",
    "PATH_KEYS",
    "POLICY_ARTIFACT_ID",
    "RESERVATION_ARTIFACT_ID",
    "SCORING_MANIFEST_ARTIFACT_ID",
    "TARGET_CACHE_ARTIFACT_ID",
    "canonical_claim_boundary_payload",
    "canonical_evaluation_payload",
    "canonical_harp_runtime_payload",
    "canonical_protocol_payload",
    "load_harp_fresh_stage70_config",
)
