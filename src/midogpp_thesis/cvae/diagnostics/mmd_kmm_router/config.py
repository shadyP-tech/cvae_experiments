"""Fail-closed configuration for the Stage-90 MMD/KMM router diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...protocol import ProtocolError
from ...routing.mmd_kmm_mixture import (
    KMMGateConfig,
    KMMOptimizationConfig,
    PriorControlConfig,
)
from .contracts import (
    CENTERS,
    CLAIM_SCOPE,
    CLASSIFIER_THREADS_PER_WORKER,
    CLASSIFIER_WORKERS,
    COMMON_FEATURE_DIM,
    COMMON_FRAME_HASH,
    DOWNSTREAM_CLASSIFIER,
    DUPLICATE_DIRECTION_COSINE,
    DUPLICATE_WEIGHT_L1,
    ENERGY_REFERENCE_RHO,
    ENERGY_REFERENCE_TEMPERATURE,
    EQUAL_UNION_POLICY_ARTIFACT_ID,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_MANIFEST_SHA256,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_DEVICES,
    GENERATION_LOCK_ARTIFACT_ID,
    GENERATION_SEEDS,
    INPUT_ARTIFACT_IDS,
    KERNEL_BATCH_ROWS,
    KERNEL_DEVICES,
    KMM_MAX_ITERATIONS,
    KMM_MINIMUM_PROXY_IMPROVEMENT,
    KMM_OPTIMALITY_TOLERANCE,
    KMM_REGULARIZATION,
    KMM_SOLVER_TOLERANCE,
    MAXIMUM_GENERATION_SEED_L1,
    MAXIMUM_PRIOR_SENSITIVITY_L1,
    MAXIMUM_SUPPORT_L1,
    MAXIMUM_TRAINING_SEED_L1,
    MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT,
    MAX_SOURCE_PREFIX_PER_CLASS,
    MAX_SOURCE_WEIGHT,
    MINIMUM_DIRECTION_COSINE,
    MIN_EFFECTIVE_SOURCES,
    NYSTROEM_COMPONENTS,
    NYSTROEM_GAMMA,
    NYSTROEM_RANDOM_STATE,
    OUTPUT_ARTIFACT_ID,
    PRIOR_CLASSIFIER,
    PRIOR_PROBABILITY_CLIP,
    PRIOR_SENSITIVITY_POSITIVE_PRIORS,
    PRIOR_TEMPERATURE,
    PUBLICATION_STATUS,
    ROUTER_PREFIX_PER_CLASS,
    STAGE_ID,
    SUPPORT_CASE_COUNT,
    SUPPORT_PARTITION_NAMESPACE,
    SUPPORT_SPLIT_SEED,
    TRAINING_SEEDS,
    VALIDATION_CACHE_ARTIFACT_ID,
    VALIDATION_CACHE_REPRESENTATION_ID,
    VALIDATION_CACHE_SEMANTIC_ID,
    VALIDATION_MANIFEST_ARTIFACT_ID,
    WORKSTATION_PROFILE,
)


_TOP_LEVEL_KEYS = frozenset(
    {"experiment", "inputs", "protocol", "proxy", "classifier", "runtime", "claim_boundary"}
)
_EXPERIMENT_KEYS = frozenset({"id", "name", "artifact_root", "claim_scope", "status"})
_INPUT_KEYS = frozenset(
    {
        "expert_bank_root",
        "generation_lock_root",
        "equal_union_policy_root",
        "validation_cache_root",
        "validation_manifest_path",
        "expert_bank_artifact_id",
        "generation_lock_artifact_id",
        "equal_union_policy_artifact_id",
        "validation_cache_artifact_id",
        "validation_manifest_artifact_id",
        "expected_bank_lock_hash",
        "expected_generation_lock_hash",
        "expected_equal_union_policy_lock_hash",
        "expected_validation_cache_semantic_id",
        "expected_validation_cache_representation_id",
        "expected_manifest_sha256",
    }
)


def canonical_protocol_payload() -> dict[str, object]:
    return {
        "dataset_family": "MIDOG++",
        "stage": STAGE_ID,
        "validation_split": "val",
        "centers": list(CENTERS),
        "excluded_center": "4",
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "seed_pairing": "cartesian_product_report_all_nine_no_seed_selection",
        "support_case_count_per_center": SUPPORT_CASE_COUNT,
        "support_split_seed": SUPPORT_SPLIT_SEED,
        "support_partition_namespace": SUPPORT_PARTITION_NAMESPACE,
        "support_partition_source": "label_free_case_identity_hash_rank",
        "support_evaluation_case_disjoint": True,
        "support_evaluation_sample_disjoint": True,
        "support_labels_used": False,
        "target_expert_excluded": True,
        "all_retained_seeds_aggregated_for_router": True,
        "evaluation_embeddings_available_to_router": False,
        "global_target_predictions_sealed_before_any_label_access": True,
        "previous_stage90_router_or_utility_inputs_used": False,
    }


def canonical_proxy_payload() -> dict[str, object]:
    return {
        "family": "class_prior_controlled_mmd_kmm",
        "common_frame_semantics": "common_inverse_virchow2",
        "common_feature_dim": COMMON_FEATURE_DIM,
        "common_frame_hash": COMMON_FRAME_HASH,
        "source_prefix_per_class": MAX_SOURCE_PREFIX_PER_CLASS,
        "router_fit_prefix_per_class": ROUTER_PREFIX_PER_CLASS,
        "source_pool_balance": "equal_source_seed_class_prefix",
        "preprocessing": "target_excluded_source_pool_standard_scaler",
        "kernel": "rbf_nystroem",
        "nystroem_components": NYSTROEM_COMPONENTS,
        "nystroem_gamma": NYSTROEM_GAMMA,
        "nystroem_random_state": NYSTROEM_RANDOM_STATE,
        "prior_classifier": PRIOR_CLASSIFIER.to_payload(),
        "prior_probability_clip": PRIOR_PROBABILITY_CLIP,
        "prior_temperature": PRIOR_TEMPERATURE,
        "prior_reference_positive_prior": 0.5,
        "prior_sensitivity_positive_priors": list(PRIOR_SENSITIVITY_POSITIVE_PRIORS),
        "kmm_regularization": KMM_REGULARIZATION,
        "minimum_proxy_improvement": KMM_MINIMUM_PROXY_IMPROVEMENT,
        "maximum_source_weight": MAX_SOURCE_WEIGHT,
        "minimum_effective_sources": MIN_EFFECTIVE_SOURCES,
        "solver_tolerance": KMM_SOLVER_TOLERANCE,
        "optimality_tolerance": KMM_OPTIMALITY_TOLERANCE,
        "max_iterations": KMM_MAX_ITERATIONS,
        "maximum_support_l1": MAXIMUM_SUPPORT_L1,
        "maximum_training_seed_l1": MAXIMUM_TRAINING_SEED_L1,
        "maximum_generation_seed_l1": MAXIMUM_GENERATION_SEED_L1,
        "maximum_prior_sensitivity_l1": MAXIMUM_PRIOR_SENSITIVITY_L1,
        "minimum_direction_cosine": MINIMUM_DIRECTION_COSINE,
        "duplicate_direction_cosine": DUPLICATE_DIRECTION_COSINE,
        "duplicate_weight_l1": DUPLICATE_WEIGHT_L1,
        "energy_reference_rho": ENERGY_REFERENCE_RHO,
        "energy_reference_temperature": ENERGY_REFERENCE_TEMPERATURE,
        "energy_reference_source": "recomputed_label_free_from_current_support",
        "integer_allocation": "positive_hamilton_largest_remainder",
        "total_generated_samples_per_class": 1024,
    }


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "workstation_profile": WORKSTATION_PROFILE,
        "generation_devices": list(GENERATION_DEVICES),
        "kernel_devices": list(KERNEL_DEVICES),
        "generation_workers_per_device": 1,
        "kernel_workers_per_device": 1,
        "classifier_workers": CLASSIFIER_WORKERS,
        "classifier_threads_per_worker": CLASSIFIER_THREADS_PER_WORKER,
        "multiprocessing_start_method": "spawn",
        "generated_cache_format": "float32_npy_memmap",
        "kernel_batch_rows": KERNEL_BATCH_ROWS,
        "one_expert_per_gpu_at_a_time": True,
        "maximum_unique_classifier_fit_count": MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT,
        "resume_policy": "hash_validated_phase_and_cell_checkpoints",
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "consumed_validation_data": True,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "proxy_is_downstream_utility": False,
        "promotion_eligible": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }


@dataclass(frozen=True)
class MMDKMMRouterDiagnosticConfig:
    source_path: Path
    artifact_root: Path
    expert_bank_root: Path
    generation_lock_root: Path
    equal_union_policy_root: Path
    validation_cache_root: Path
    validation_manifest_path: Path
    protocol: Mapping[str, object]
    proxy: Mapping[str, object]
    classifier: ClassifierSpec
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]
    contract_hash: str

    @property
    def input_artifact_ids(self) -> tuple[str, ...]:
        return INPUT_ARTIFACT_IDS

    @property
    def prior_control(self) -> PriorControlConfig:
        return PriorControlConfig(
            probability_clip=PRIOR_PROBABILITY_CLIP,
            temperature=PRIOR_TEMPERATURE,
            sensitivity_positive_priors=PRIOR_SENSITIVITY_POSITIVE_PRIORS,
            reference_positive_prior=0.5,
        )

    @property
    def optimization(self) -> KMMOptimizationConfig:
        return KMMOptimizationConfig(
            regularization=KMM_REGULARIZATION,
            minimum_proxy_improvement=KMM_MINIMUM_PROXY_IMPROVEMENT,
            max_source_weight=MAX_SOURCE_WEIGHT,
            minimum_effective_sources=MIN_EFFECTIVE_SOURCES,
            solver_tolerance=KMM_SOLVER_TOLERANCE,
            optimality_tolerance=KMM_OPTIMALITY_TOLERANCE,
            max_iterations=KMM_MAX_ITERATIONS,
        )

    @property
    def gates(self) -> KMMGateConfig:
        return KMMGateConfig(
            maximum_support_l1=MAXIMUM_SUPPORT_L1,
            maximum_training_seed_l1=MAXIMUM_TRAINING_SEED_L1,
            maximum_generation_seed_l1=MAXIMUM_GENERATION_SEED_L1,
            maximum_prior_sensitivity_l1=MAXIMUM_PRIOR_SENSITIVITY_L1,
            minimum_direction_cosine=MINIMUM_DIRECTION_COSINE,
            duplicate_direction_cosine=DUPLICATE_DIRECTION_COSINE,
            duplicate_weight_l1=DUPLICATE_WEIGHT_L1,
        )


def load_mmd_kmm_router_config(path: str | Path) -> MMDKMMRouterDiagnosticConfig:
    source = Path(path).resolve()
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError(f"Cannot read MMD/KMM diagnostic config: {source}.") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("MMD/KMM diagnostic config must be a mapping.")
    _exact_keys(payload, _TOP_LEVEL_KEYS, "top level")
    _reject_pending(payload)
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    protocol = _mapping(payload, "protocol")
    proxy = _mapping(payload, "proxy")
    classifier_raw = _mapping(payload, "classifier")
    runtime = _mapping(payload, "runtime")
    claim = _mapping(payload, "claim_boundary")
    _exact_keys(experiment, _EXPERIMENT_KEYS, "experiment")
    _exact_keys(inputs, _INPUT_KEYS, "inputs")
    _require_exact(protocol, canonical_protocol_payload(), "protocol")
    _require_exact(proxy, canonical_proxy_payload(), "proxy")
    _require_exact(runtime, canonical_runtime_payload(), "runtime")
    _require_exact(claim, canonical_claim_boundary_payload(), "claim boundary")
    classifier = _classifier(classifier_raw)
    if classifier != DOWNSTREAM_CLASSIFIER:
        raise ProtocolError("MMD/KMM downstream classifier differs from GenerationLock.")

    exact = {
        "experiment id": (experiment.get("id"), EXPERIMENT_ID),
        "experiment name": (experiment.get("name"), EXPERIMENT_NAME),
        "claim scope": (experiment.get("claim_scope"), CLAIM_SCOPE),
        "status": (experiment.get("status"), PUBLICATION_STATUS),
        "expert bank id": (inputs.get("expert_bank_artifact_id"), EXPERT_BANK_ARTIFACT_ID),
        "generation lock id": (inputs.get("generation_lock_artifact_id"), GENERATION_LOCK_ARTIFACT_ID),
        "control policy id": (inputs.get("equal_union_policy_artifact_id"), EQUAL_UNION_POLICY_ARTIFACT_ID),
        "cache alias id": (inputs.get("validation_cache_artifact_id"), VALIDATION_CACHE_ARTIFACT_ID),
        "manifest alias id": (inputs.get("validation_manifest_artifact_id"), VALIDATION_MANIFEST_ARTIFACT_ID),
        "bank lock": (inputs.get("expected_bank_lock_hash"), EXPECTED_BANK_LOCK_HASH),
        "generation lock": (inputs.get("expected_generation_lock_hash"), EXPECTED_GENERATION_LOCK_HASH),
        "control policy lock": (inputs.get("expected_equal_union_policy_lock_hash"), EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH),
        "cache semantic id": (inputs.get("expected_validation_cache_semantic_id"), VALIDATION_CACHE_SEMANTIC_ID),
        "cache representation": (inputs.get("expected_validation_cache_representation_id"), VALIDATION_CACHE_REPRESENTATION_ID),
        "manifest sha256": (inputs.get("expected_manifest_sha256"), EXPECTED_MANIFEST_SHA256),
    }
    drift = [role for role, values in exact.items() if values[0] != values[1]]
    if drift:
        raise ProtocolError(f"MMD/KMM config identities drifted: {drift!r}.")
    locations = (
        ("expert_bank_root", EXPERT_BANK_ARTIFACT_ID, ""),
        ("generation_lock_root", GENERATION_LOCK_ARTIFACT_ID, ""),
        ("equal_union_policy_root", EQUAL_UNION_POLICY_ARTIFACT_ID, ""),
        ("validation_cache_root", VALIDATION_CACHE_ARTIFACT_ID, ""),
        ("validation_manifest_path", VALIDATION_MANIFEST_ARTIFACT_ID, "manifest.csv"),
    )
    for key, artifact_id, member in locations:
        _validate_artifact_uri(inputs[key], artifact_id=artifact_id, member=member)
    artifact_root_raw = _string(experiment["artifact_root"], "artifact root")
    if artifact_root_raw.startswith("output://") and artifact_root_raw != f"output://{OUTPUT_ARTIFACT_ID}":
        raise ProtocolError("MMD/KMM output artifact identity drifted.")

    scientific = {
        "experiment_id": EXPERIMENT_ID,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "protocol": dict(protocol),
        "proxy": dict(proxy),
        "classifier": classifier.to_payload(),
        "claim_boundary": dict(claim),
    }
    return MMDKMMRouterDiagnosticConfig(
        source_path=source,
        artifact_root=_path(source.parent, artifact_root_raw),
        expert_bank_root=_path(source.parent, inputs["expert_bank_root"]),
        generation_lock_root=_path(source.parent, inputs["generation_lock_root"]),
        equal_union_policy_root=_path(source.parent, inputs["equal_union_policy_root"]),
        validation_cache_root=_path(source.parent, inputs["validation_cache_root"]),
        validation_manifest_path=_path(source.parent, inputs["validation_manifest_path"]),
        protocol=dict(protocol),
        proxy=dict(proxy),
        classifier=classifier,
        runtime=dict(runtime),
        claim_boundary=dict(claim),
        contract_hash=stable_hash(scientific),
    )


def _classifier(payload: Mapping[str, object]) -> ClassifierSpec:
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
        raise ProtocolError("MMD/KMM classifier configuration is invalid.") from exc


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"MMD/KMM config lacks mapping {key!r}.")
    return value


def _exact_keys(payload: Mapping[object, object], expected: frozenset[str], role: str) -> None:
    if set(payload) != set(expected) or any(not isinstance(key, str) for key in payload):
        raise ProtocolError(f"MMD/KMM {role} keys drifted.")


def _require_exact(observed: Mapping[str, object], expected: Mapping[str, object], role: str) -> None:
    if not _strict_equal(observed, expected):
        raise ProtocolError(f"MMD/KMM {role} values drifted.")


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


def _validate_artifact_uri(value: object, *, artifact_id: str, member: str) -> None:
    rendered = _string(value, "artifact location")
    expected = f"artifact://{artifact_id}" + (f"/{member}" if member else "")
    if rendered.startswith("artifact://") and rendered != expected:
        raise ProtocolError(f"MMD/KMM artifact URI must be {expected!r}.")


def _path(base: Path, value: object) -> Path:
    rendered = _string(value, "path")
    if rendered.startswith(("artifact://", "output://")):
        return Path(rendered)
    raw = Path(rendered).expanduser()
    return raw.resolve() if raw.is_absolute() else (base / raw).resolve()


def _string(value: object, role: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProtocolError(f"MMD/KMM {role} must be a nonempty string.")
    return value


def _integer(value: object, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"MMD/KMM {role} must be an integer.")
    return value


def _number(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ProtocolError(f"MMD/KMM {role} must be finite numeric.")
    return float(value)


def _reject_pending(value: object, location: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_pending(nested, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _reject_pending(nested, f"{location}[{index}]")
    elif isinstance(value, str) and value.upper().startswith("PENDING"):
        raise ProtocolError(f"MMD/KMM config contains PENDING at {location}.")


__all__ = (
    "MMDKMMRouterDiagnosticConfig",
    "canonical_claim_boundary_payload",
    "canonical_protocol_payload",
    "canonical_proxy_payload",
    "canonical_runtime_payload",
    "load_mmd_kmm_router_config",
)
