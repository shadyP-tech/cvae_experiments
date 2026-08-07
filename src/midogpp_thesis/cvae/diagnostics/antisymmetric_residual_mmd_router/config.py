"""Strict configuration for the cross-fitted antisymmetric MMD diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...protocol import ProtocolError
from ...routing.antisymmetric_residual_mmd import AntisymmetricResidualConfig
from ...routing.mmd_kmm_mixture import (
    ConditionalContrastConfig,
    PriorControlConfig,
)
from .contracts import *  # noqa: F403 - the frozen contract is intentionally central


_TOP_LEVEL = frozenset(
    {"experiment", "inputs", "protocol", "proxy", "classifier", "runtime", "claim_boundary"}
)


def canonical_protocol_payload() -> dict[str, object]:
    return {
        "dataset_family": "MIDOG++",
        "stage": STAGE_ID,  # noqa: F405
        "validation_split": "val",
        "centers": list(CENTERS),  # noqa: F405
        "excluded_center": "4",
        "training_seeds": list(TRAINING_SEEDS),  # noqa: F405
        "generation_seeds": list(GENERATION_SEEDS),  # noqa: F405
        "seed_pairing": "cartesian_product_report_all_nine_no_seed_selection",
        "fixed_calibration_case_count_per_center": SUPPORT_CASE_COUNT,  # noqa: F405
        "fixed_calibration_split_seed": SUPPORT_SPLIT_SEED,  # noqa: F405
        "fixed_calibration_partition_namespace": SUPPORT_PARTITION_NAMESPACE,  # noqa: F405
        "cross_fit_namespace": CROSS_FIT_NAMESPACE,  # noqa: F405
        "cross_fit_mode": CROSS_FIT_MODE,  # noqa: F405
        "heldout_case_excluded_from_own_route": True,
        "fixed_calibration_cases_never_scored": True,
        "whole_case_support_evaluation_disjoint": True,
        "support_labels_used": False,
        "evaluation_labels_available_to_router": False,
        "evaluation_embeddings_available_to_router": True,
        "cohort_evaluation_embeddings_available_for_other_case_routes": True,
        "heldout_evaluation_embeddings_available_to_own_route": False,
        "target_expert_excluded": True,
        "all_retained_seeds_aggregated_for_router": True,
        "cohort_unlabeled_embeddings_used_for_other_case_routes": True,
        "global_crossfit_predictions_sealed_before_any_label_access": True,
        "previous_stage90_router_or_utility_inputs_used": False,
    }


def canonical_proxy_payload() -> dict[str, object]:
    return {
        "family": ROUTER_MODE,  # noqa: F405
        "claim_role": "label_free_proxy_compatibility_only",
        "common_frame_semantics": "common_inverse_virchow2",
        "common_feature_dim": COMMON_FEATURE_DIM,  # noqa: F405
        "common_frame_hash": COMMON_FRAME_HASH,  # noqa: F405
        "source_prefix_per_class": MAX_SOURCE_PREFIX_PER_CLASS,  # noqa: F405
        "router_fit_prefix_per_class": ROUTER_PREFIX_PER_CLASS,  # noqa: F405
        "source_pool_balance": "equal_source_seed_class_prefix",
        "preprocessing": "target_excluded_source_pool_standard_scaler",
        "kernel": "rbf_nystroem",
        "nystroem_components": NYSTROEM_COMPONENTS,  # noqa: F405
        "nystroem_gamma": NYSTROEM_GAMMA,  # noqa: F405
        "nystroem_random_state": NYSTROEM_RANDOM_STATE,  # noqa: F405
        "prior_classifier": PRIOR_CLASSIFIER.to_payload(),  # noqa: F405
        "prior_probability_clip": PRIOR_PROBABILITY_CLIP,  # noqa: F405
        "prior_temperature": PRIOR_TEMPERATURE,  # noqa: F405
        "prior_reference_positive_prior": 0.5,
        "prior_sensitivity_positive_priors": list(PRIOR_SENSITIVITY_POSITIVE_PRIORS),  # noqa: F405
        "class_weights": list(CLASS_WEIGHTS),  # noqa: F405
        "contrast_weight": CONTRAST_WEIGHT,  # noqa: F405
        "weight_parameterization": "w_class0=uniform+delta;w_class1=uniform-delta",
        "source_total_exposure_preserved_across_classes": True,
        "residual_l1_radius_per_class": RESIDUAL_L1_RADIUS,  # noqa: F405
        "residual_l1_enforced_inside_solver": True,
        "residual_l2_regularization": RESIDUAL_L2_REGULARIZATION,  # noqa: F405
        "robust_worst_variant_penalty": ROBUST_WORST_VARIANT_PENALTY,  # noqa: F405
        "minimum_robust_proxy_improvement": MINIMUM_ROBUST_PROXY_IMPROVEMENT,  # noqa: F405
        "maximum_source_weight_per_class": MAX_SOURCE_WEIGHT,  # noqa: F405
        "minimum_effective_sources_per_class": MIN_EFFECTIVE_SOURCES,  # noqa: F405
        "minimum_soft_class_mass_per_case": MINIMUM_SOFT_CLASS_MASS_PER_CASE,  # noqa: F405
        "minimum_soft_class_effective_rows_per_case": MINIMUM_SOFT_CLASS_EFFECTIVE_ROWS_PER_CASE,  # noqa: F405
        "robust_variant_axes": [
            "support_case",
            "training_seed",
            "generation_seed",
            "class_prior_sensitivity",
        ],
        "solver": "scipy_slsqp_epigraph_with_exact_l1_halfspaces",
        "solver_tolerance": SOLVER_TOLERANCE,  # noqa: F405
        "max_iterations": MAX_SOLVER_ITERATIONS,  # noqa: F405
        "integer_allocation": "antisymmetric_floor_largest_fraction_remainder",
        "total_generated_samples_per_class": TOTAL_PER_CLASS,  # noqa: F405
    }


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "workstation_profile": WORKSTATION_PROFILE,  # noqa: F405
        "generation_devices": list(GENERATION_DEVICES),  # noqa: F405
        "kernel_devices": list(KERNEL_DEVICES),  # noqa: F405
        "cuda_visible_devices": "0,1",
        "generation_workers_per_device": 1,
        "kernel_workers_per_device": 1,
        "classifier_workers": CLASSIFIER_WORKERS,  # noqa: F405
        "classifier_threads_per_worker": CLASSIFIER_THREADS_PER_WORKER,  # noqa: F405
        "multiprocessing_start_method": "spawn",
        "tf32_disabled_in_gpu_workers": True,
        "dependency_version_policy": "presence_gate_versions_report_only",
        "generated_cache_format": "float32_npy_memmap",
        "kernel_batch_rows": KERNEL_BATCH_ROWS,  # noqa: F405
        "one_expert_per_gpu_at_a_time": True,
        "target_kernel_workspace_reused_across_case_folds": True,
        "minimum_logical_cpu_count": MINIMUM_WORKSTATION_LOGICAL_CPU_COUNT,  # noqa: F405
        "minimum_physical_ram_bytes": MINIMUM_WORKSTATION_RAM_BYTES,  # noqa: F405
        "minimum_artifact_disk_free_bytes": MINIMUM_WORKSTATION_DISK_FREE_BYTES,  # noqa: F405
        "minimum_gpu_free_mib_per_device": MINIMUM_WORKSTATION_GPU_FREE_MIB,  # noqa: F405
        "maximum_unique_classifier_fit_count": MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT,  # noqa: F405
        "resume_policy": "hash_validated_phase_fold_and_cell_checkpoints",
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "consumed_validation_data": True,
        "cross_fitted_transductive_diagnostic": True,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "routing_quality_claimed": False,
        "heldout_target_utility_claimed": False,
        "proxy_is_nelbo_compatibility": False,
        "proxy_is_downstream_utility": False,
        "promotion_eligible": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }


@dataclass(frozen=True)
class AntisymmetricResidualMMDDiagnosticConfig:
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
    def experiment_id(self) -> str:
        return EXPERIMENT_ID  # noqa: F405

    @property
    def output_artifact_id(self) -> str:
        return OUTPUT_ARTIFACT_ID  # noqa: F405

    @property
    def input_artifact_ids(self) -> tuple[str, ...]:
        return INPUT_ARTIFACT_IDS  # noqa: F405

    @property
    def prior_control(self) -> PriorControlConfig:
        """Return the frozen source-only soft-responsibility control."""

        return PriorControlConfig(
            probability_clip=PRIOR_PROBABILITY_CLIP,  # noqa: F405
            temperature=PRIOR_TEMPERATURE,  # noqa: F405
            sensitivity_positive_priors=PRIOR_SENSITIVITY_POSITIVE_PRIORS,  # noqa: F405
            reference_positive_prior=0.5,
        )

    @property
    def conditional_contrast(self) -> ConditionalContrastConfig:
        """Return the fixed class-conditional proxy construction."""

        return ConditionalContrastConfig(
            class_weights=CLASS_WEIGHTS,  # noqa: F405
            contrast_weight=CONTRAST_WEIGHT,  # noqa: F405
            maximum_uniform_l1=RESIDUAL_L1_RADIUS,  # noqa: F405
            minimum_soft_class_mass_per_case=MINIMUM_SOFT_CLASS_MASS_PER_CASE,  # noqa: F405
            minimum_soft_class_effective_rows_per_case=MINIMUM_SOFT_CLASS_EFFECTIVE_ROWS_PER_CASE,  # noqa: F405
            component_tolerance=1.0e-10,
        )

    @property
    def residual_optimization(self) -> AntisymmetricResidualConfig:
        """Return the solver-level trust-region and robust-loss contract."""

        return AntisymmetricResidualConfig(
            worst_variant_penalty=ROBUST_WORST_VARIANT_PENALTY,  # noqa: F405
            l2_shrinkage=RESIDUAL_L2_REGULARIZATION,  # noqa: F405
            max_source_weight=MAX_SOURCE_WEIGHT,  # noqa: F405
            minimum_effective_sources=MIN_EFFECTIVE_SOURCES,  # noqa: F405
            maximum_uniform_l1=RESIDUAL_L1_RADIUS,  # noqa: F405
            minimum_soft_class_mass_per_case=MINIMUM_SOFT_CLASS_MASS_PER_CASE,  # noqa: F405
            minimum_soft_class_effective_rows_per_case=MINIMUM_SOFT_CLASS_EFFECTIVE_ROWS_PER_CASE,  # noqa: F405
            minimum_robust_improvement=MINIMUM_ROBUST_PROXY_IMPROVEMENT,  # noqa: F405
            variant_worsening_tolerance=1.0e-10,
            solver_tolerance=SOLVER_TOLERANCE,  # noqa: F405
            max_iterations=MAX_SOLVER_ITERATIONS,  # noqa: F405
        )


def load_antisymmetric_residual_mmd_config(
    path: str | Path,
) -> AntisymmetricResidualMMDDiagnosticConfig:
    source = Path(path).resolve()
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read antisymmetric residual-MMD config.") from exc
    if not isinstance(payload, Mapping) or set(payload) != set(_TOP_LEVEL):
        raise ProtocolError("Antisymmetric residual-MMD top-level config drifted.")
    _reject_pending(payload)
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    protocol = _mapping(payload, "protocol")
    proxy = _mapping(payload, "proxy")
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
        raise ProtocolError("Antisymmetric residual-MMD experiment identity drifted.")
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
        raise ProtocolError("Antisymmetric residual-MMD input identity drifted.")
    _require_exact(protocol, canonical_protocol_payload(), "protocol")
    _require_exact(proxy, canonical_proxy_payload(), "proxy")
    _require_exact(runtime, canonical_runtime_payload(), "runtime")
    _require_exact(claim, canonical_claim_boundary_payload(), "claim boundary")
    classifier = _classifier(_mapping(payload, "classifier"))
    if classifier != DOWNSTREAM_CLASSIFIER:  # noqa: F405
        raise ProtocolError("Antisymmetric residual-MMD classifier drifted.")
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
        raise ProtocolError("Antisymmetric residual-MMD output identity drifted.")
    scientific = {
        "experiment_id": EXPERIMENT_ID,  # noqa: F405
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),  # noqa: F405
        "protocol": dict(protocol),
        "proxy": dict(proxy),
        "classifier": classifier.to_payload(),
        "claim_boundary": dict(claim),
    }
    return AntisymmetricResidualMMDDiagnosticConfig(
        source_path=source,
        artifact_root=_path(source.parent, artifact_root),
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


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Antisymmetric residual-MMD config lacks {key!r}.")
    return value


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
        raise ProtocolError("Antisymmetric residual-MMD classifier is invalid.") from exc


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
        raise ProtocolError(f"Antisymmetric residual-MMD {role} drifted.")


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


def _integer(value: object, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{role} must be an integer.")
    return value


def _number(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ProtocolError(f"{role} must be finite numeric.")
    return float(value)


def _reject_pending(value: object, location: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_pending(nested, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _reject_pending(nested, f"{location}[{index}]")
    elif isinstance(value, str) and value.upper().startswith("PENDING"):
        raise ProtocolError(f"Config contains PENDING at {location}.")


__all__ = (
    "AntisymmetricResidualMMDDiagnosticConfig",
    "canonical_claim_boundary_payload",
    "canonical_protocol_payload",
    "canonical_proxy_payload",
    "canonical_runtime_payload",
    "load_antisymmetric_residual_mmd_config",
)
