"""Strict configuration for the residual top-up Stage-90 diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...protocol import ProtocolError
from .contracts import *  # noqa: F403 - one intentionally central frozen contract


_TOP_LEVEL = frozenset(
    {
        "experiment",
        "inputs",
        "protocol",
        "actions",
        "classifier",
        "selection",
        "runtime",
        "claim_boundary",
    }
)


def canonical_protocol_payload() -> dict[str, object]:
    return {
        "dataset_family": "MIDOG++",
        "stage": STAGE_ID,  # noqa: F405
        "validation_split": VALIDATION_SPLIT,  # noqa: F405
        "centers": list(CENTERS),  # noqa: F405
        "excluded_center": EXCLUDED_CENTER,  # noqa: F405
        "training_seeds": list(TRAINING_SEEDS),  # noqa: F405
        "generation_seeds": list(GENERATION_SEEDS),  # noqa: F405
        "seed_pairing": "cartesian_product_report_all_nine_no_seed_selection",
        "support_case_count_per_center": SUPPORT_CASE_COUNT,  # noqa: F405
        "support_split_seed": SUPPORT_SPLIT_SEED,  # noqa: F405
        "support_partition_namespace": SUPPORT_PARTITION_NAMESPACE,  # noqa: F405
        "whole_case_support_evaluation_disjoint": True,
        "support_labels_used": False,
        "evaluation_labels_available_before_global_prediction_seal": False,
        "evaluation_labels_available_to_action_prediction": False,
        "inner_query_evaluation_labels_used_after_global_seal_for_outer_H_calibration": True,
        "target_H_evaluation_labels_used_for_own_selection": False,
        "evaluation_embeddings_available_to_router": False,
        "target_expert_excluded": True,
        "inner_outer_target_excluded": True,
        "inner_query_expert_excluded": True,
        "all_actions_predictions_globally_sealed_before_any_label_access": True,
        "previous_stage90_router_or_utility_inputs_used": False,
    }


def canonical_actions_payload() -> dict[str, object]:
    return {
        "family": "immutable_equal_union_backbone_with_residual_topup_v1",
        "action_ids": list(TARGET_ACTION_IDS),  # noqa: F405
        "development_action_ids": list(DEVELOPMENT_ACTION_IDS),  # noqa: F405
        "primary_control_action_id": PRIMARY_CONTROL_ACTION_ID,  # noqa: F405
        "primary_routed_action_id": PRIMARY_ROUTED_ACTION_ID,  # noqa: F405
        "base_only_role": "separate_budget_reference_not_primary_control",
        "target_source_count": TARGET_SOURCE_COUNT,  # noqa: F405
        "target_base_per_source": TARGET_BASE_PER_SOURCE,  # noqa: F405
        "target_base_total_per_class": TARGET_BASE_TOTAL_PER_CLASS,  # noqa: F405
        "target_topup_total_per_class": TARGET_TOPUP_TOTAL_PER_CLASS,  # noqa: F405
        "target_matched_total_per_class": TARGET_MATCHED_TOTAL_PER_CLASS,  # noqa: F405
        "development_source_count": DEVELOPMENT_SOURCE_COUNT,  # noqa: F405
        "development_base_per_source": DEVELOPMENT_BASE_PER_SOURCE,  # noqa: F405
        "development_base_total_per_class": DEVELOPMENT_BASE_TOTAL_PER_CLASS,  # noqa: F405
        "development_topup_total_per_class": DEVELOPMENT_TOPUP_TOTAL_PER_CLASS,  # noqa: F405
        "development_matched_total_per_class": DEVELOPMENT_MATCHED_TOTAL_PER_CLASS,  # noqa: F405
        "topup_fraction_of_base": TOPUP_FRACTION_OF_BASE,  # noqa: F405
        "source_cache_prefix_per_class": MAX_SOURCE_PREFIX_PER_CLASS,  # noqa: F405
        "base_window_start": 0,
        "topup_window_starts_after_base": True,
        "class_agnostic_topup": True,
        "energy_direction": ENERGY_RANK_SEMANTICS,  # noqa: F405
        "energy_claim_role": "label_free_proxy_only_not_nelbo_or_utility",
        "maximum_final_source_weight": MAX_SOURCE_WEIGHT,  # noqa: F405
        "minimum_final_effective_sources": MIN_EFFECTIVE_SOURCES,  # noqa: F405
        "no_action_or_budget_search": True,
    }


def canonical_selection_payload() -> dict[str, object]:
    return {
        "rule": SELECTION_RULE,  # noqa: F405
        "cluster_unit": SELECTION_CLUSTER_UNIT,  # noqa: F405
        "confidence_level": SELECTION_CONFIDENCE_LEVEL,  # noqa: F405
        "threshold": SELECTION_THRESHOLD,  # noqa: F405
        "response": "paired_bacc_energy_topup_minus_uniform_topup",
        "query_center_means_average_all_nine_seed_cells": True,
        "outer_H_excluded_from_calibration": True,
        "model_family": "fixed_no_parameter_two_action_lcb_gate",
        "nested_hyperparameter_selection": False,
        "seed_risk": "report_only_no_seed_selection",
        "fallback_action_id": UNIFORM_TOPUP_ACTION_ID,  # noqa: F405
        "target_H_labels_used_for_selection": False,
    }


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "workstation_profile": WORKSTATION_PROFILE,  # noqa: F405
        "generation_devices": list(GENERATION_DEVICES),  # noqa: F405
        "cuda_visible_devices": "0,1",
        "generation_workers_per_device": 1,
        "classifier_workers": CLASSIFIER_WORKERS,  # noqa: F405
        "classifier_threads_per_worker": CLASSIFIER_THREADS_PER_WORKER,  # noqa: F405
        "multiprocessing_start_method": "spawn",
        "tf32_disabled_in_gpu_workers": True,
        "dependency_version_policy": "presence_gate_versions_report_only",
        "generated_cache_format": "float32_npy_memmap",
        "one_expert_per_gpu_at_a_time": True,
        "phase_disjoint_gpu_and_cpu_pools": True,
        "minimum_logical_cpu_count": MINIMUM_WORKSTATION_LOGICAL_CPU_COUNT,  # noqa: F405
        "minimum_physical_ram_bytes": MINIMUM_WORKSTATION_RAM_BYTES,  # noqa: F405
        "minimum_artifact_disk_free_bytes": MINIMUM_WORKSTATION_DISK_FREE_BYTES,  # noqa: F405
        "minimum_gpu_free_mib_per_device": MINIMUM_WORKSTATION_GPU_FREE_MIB,  # noqa: F405
        "maximum_unique_classifier_fit_count": MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT,  # noqa: F405
        "resume_policy": (
            "hash_validated_source_and_prediction_task_checkpoints_"
            "with_completed_product_reuse"
        ),
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "consumed_validation_data": True,
        "method_development_is_posthoc": True,
        "terminal_stage90_diagnostic": True,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "proxy_is_nelbo_compatibility": False,
        "proxy_is_downstream_utility": False,
        "promotion_eligible": False,
        "oracle_eligible": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }


@dataclass(frozen=True)
class ResidualTopupDiagnosticConfig:
    source_path: Path
    artifact_root: Path
    expert_bank_root: Path
    generation_lock_root: Path
    equal_union_policy_root: Path
    validation_cache_root: Path
    validation_manifest_path: Path
    protocol: Mapping[str, object]
    actions: Mapping[str, object]
    classifier: ClassifierSpec
    selection: Mapping[str, object]
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]
    contract_hash: str

    @property
    def experiment_id(self) -> str:
        return EXPERIMENT_ID  # noqa: F405

    @property
    def output_artifact_id(self) -> str:
        return OUTPUT_ARTIFACT_ID  # noqa: F405

    @property
    def input_artifact_ids(self) -> tuple[str, ...]:
        return INPUT_ARTIFACT_IDS  # noqa: F405


def load_residual_topup_config(path: str | Path) -> ResidualTopupDiagnosticConfig:
    source = Path(path).resolve()
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read residual top-up config.") from exc
    if not isinstance(payload, Mapping) or set(payload) != set(_TOP_LEVEL):
        raise ProtocolError("Residual top-up top-level config drifted.")
    _reject_pending(payload)
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    protocol = _mapping(payload, "protocol")
    actions = _mapping(payload, "actions")
    selection = _mapping(payload, "selection")
    runtime = _mapping(payload, "runtime")
    claim = _mapping(payload, "claim_boundary")
    expected_experiment = {
        "id": EXPERIMENT_ID,  # noqa: F405
        "name": EXPERIMENT_NAME,  # noqa: F405
        "artifact_root": experiment.get("artifact_root"),
        "claim_scope": CLAIM_SCOPE,  # noqa: F405
        "status": PUBLICATION_STATUS,  # noqa: F405
    }
    if set(experiment) != set(expected_experiment) or any(
        experiment[key] != value for key, value in expected_experiment.items()
    ):
        raise ProtocolError("Residual top-up experiment identity drifted.")
    expected_inputs = {
        "expert_bank_artifact_id": EXPERT_BANK_ARTIFACT_ID,  # noqa: F405
        "generation_lock_artifact_id": GENERATION_LOCK_ARTIFACT_ID,  # noqa: F405
        "equal_union_policy_artifact_id": EQUAL_UNION_POLICY_ARTIFACT_ID,  # noqa: F405
        "validation_cache_artifact_id": VALIDATION_CACHE_ARTIFACT_ID,  # noqa: F405
        "validation_manifest_artifact_id": VALIDATION_MANIFEST_ARTIFACT_ID,  # noqa: F405
        "expected_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,  # noqa: F405
        "expected_generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,  # noqa: F405
        "expected_equal_union_policy_lock_hash": EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,  # noqa: F405
        "expected_validation_cache_semantic_id": VALIDATION_CACHE_SEMANTIC_ID,  # noqa: F405
        "expected_validation_cache_representation_id": VALIDATION_CACHE_REPRESENTATION_ID,  # noqa: F405
        "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,  # noqa: F405
    }
    location_keys = {
        "expert_bank_root",
        "generation_lock_root",
        "equal_union_policy_root",
        "validation_cache_root",
        "validation_manifest_path",
    }
    if set(inputs) != set(expected_inputs).union(location_keys) or any(
        inputs.get(key) != value for key, value in expected_inputs.items()
    ):
        raise ProtocolError("Residual top-up input identity drifted.")
    _require_exact(protocol, canonical_protocol_payload(), "protocol")
    _require_exact(actions, canonical_actions_payload(), "actions")
    _require_exact(selection, canonical_selection_payload(), "selection")
    _require_exact(runtime, canonical_runtime_payload(), "runtime")
    _require_exact(claim, canonical_claim_boundary_payload(), "claim boundary")
    classifier = _classifier(_mapping(payload, "classifier"))
    if classifier != DOWNSTREAM_CLASSIFIER:  # noqa: F405
        raise ProtocolError("Residual top-up classifier drifted.")
    locations = {
        "expert_bank_root": (EXPERT_BANK_ARTIFACT_ID, ""),  # noqa: F405
        "generation_lock_root": (GENERATION_LOCK_ARTIFACT_ID, ""),  # noqa: F405
        "equal_union_policy_root": (EQUAL_UNION_POLICY_ARTIFACT_ID, ""),  # noqa: F405
        "validation_cache_root": (VALIDATION_CACHE_ARTIFACT_ID, ""),  # noqa: F405
        "validation_manifest_path": (VALIDATION_MANIFEST_ARTIFACT_ID, "manifest.csv"),  # noqa: F405
    }
    for key, (artifact_id, member) in locations.items():
        _validate_artifact_uri(inputs[key], artifact_id=artifact_id, member=member)
    artifact_root = _string(experiment["artifact_root"], "artifact root")
    if artifact_root.startswith("output://") and artifact_root != f"output://{OUTPUT_ARTIFACT_ID}":  # noqa: F405
        raise ProtocolError("Residual top-up output identity drifted.")
    scientific = {
        "experiment_id": EXPERIMENT_ID,  # noqa: F405
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),  # noqa: F405
        "protocol": dict(protocol),
        "actions": dict(actions),
        "classifier": classifier.to_payload(),
        "selection": dict(selection),
        "claim_boundary": dict(claim),
    }
    return ResidualTopupDiagnosticConfig(
        source_path=source,
        artifact_root=_path(source.parent, artifact_root),
        expert_bank_root=_path(source.parent, inputs["expert_bank_root"]),
        generation_lock_root=_path(source.parent, inputs["generation_lock_root"]),
        equal_union_policy_root=_path(source.parent, inputs["equal_union_policy_root"]),
        validation_cache_root=_path(source.parent, inputs["validation_cache_root"]),
        validation_manifest_path=_path(source.parent, inputs["validation_manifest_path"]),
        protocol=dict(protocol),
        actions=dict(actions),
        classifier=classifier,
        selection=dict(selection),
        runtime=dict(runtime),
        claim_boundary=dict(claim),
        contract_hash=stable_hash(scientific),
    )


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Residual top-up config lacks {key!r}.")
    return value


def _classifier(payload: Mapping[str, object]) -> ClassifierSpec:
    expected_keys = {
        "family", "C", "penalty", "solver", "max_iter", "class_weight",
        "random_state", "l1_ratio", "threshold_policy", "scaler_fit",
    }
    if set(payload) != expected_keys:
        raise ProtocolError("Residual top-up classifier schema drifted.")
    try:
        return ClassifierSpec(
            family=_string(payload["family"], "classifier family"),
            C=_number(payload["C"], "classifier C"),
            penalty=_string(payload["penalty"], "classifier penalty"),
            solver=_string(payload["solver"], "classifier solver"),
            max_iter=_integer(payload["max_iter"], "classifier max_iter"),
            class_weight=None if payload["class_weight"] is None else _string(payload["class_weight"], "class weight"),
            random_state=_integer(payload["random_state"], "classifier random state"),
            l1_ratio=None if payload["l1_ratio"] is None else _number(payload["l1_ratio"], "l1 ratio"),
            threshold_policy=_string(payload["threshold_policy"], "threshold policy"),
            scaler_fit=_string(payload["scaler_fit"], "scaler fit"),
        )
    except (KeyError, ValueError) as exc:
        raise ProtocolError("Residual top-up classifier is invalid.") from exc


def _validate_artifact_uri(value: object, *, artifact_id: str, member: str) -> None:
    rendered = _string(value, "artifact location")
    expected = f"artifact://{artifact_id}" + (f"/{member}" if member else "")
    if rendered.startswith("artifact://") and rendered != expected:
        raise ProtocolError(f"Artifact URI must be {expected!r}.")


def _path(base: Path, value: object) -> Path:
    rendered = _string(value, "path")
    if rendered.startswith(("artifact://", "output://")):
        return Path(rendered)
    raw = Path(rendered).expanduser()
    return raw.resolve() if raw.is_absolute() else (base / raw).resolve()


def _require_exact(observed: object, expected: object, role: str) -> None:
    if not _strict_equal(observed, expected):
        raise ProtocolError(f"Residual top-up {role} drifted.")


def _strict_equal(observed: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return observed is expected
    if isinstance(expected, Mapping):
        return isinstance(observed, Mapping) and set(observed) == set(expected) and all(
            _strict_equal(observed[key], value) for key, value in expected.items()
        )
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        return isinstance(observed, Sequence) and not isinstance(observed, (str, bytes)) and len(observed) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _string(value: object, role: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProtocolError(f"{role} must be a nonempty string.")
    return value


def _number(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{role} must be numeric.")
    return float(value)


def _integer(value: object, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{role} must be an integer.")
    return int(value)


def _reject_pending(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).upper().startswith(("TODO", "TBD", "PENDING")):
                raise ProtocolError("Residual top-up config contains a pending key.")
            _reject_pending(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _reject_pending(item)
    elif isinstance(value, str) and value.strip().upper() in {"TODO", "TBD", "PENDING"}:
        raise ProtocolError("Residual top-up config contains a pending value.")


__all__ = (
    "ResidualTopupDiagnosticConfig",
    "canonical_actions_payload",
    "canonical_claim_boundary_payload",
    "canonical_protocol_payload",
    "canonical_runtime_payload",
    "canonical_selection_payload",
    "load_residual_topup_config",
)
