"""Fail-closed Stage-90 configuration for the dense residual diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...protocol import ProtocolError
from .contracts import (
    ACTION_IDS,
    ACTION_LIBRARY_HASH,
    CLAIM_SCOPE,
    CLASSIFIER,
    CLASS_PRIOR,
    COMPATIBILITY_SEMANTICS,
    CONTROL_ACTION_ID,
    CENTERS,
    DEVELOPMENT_TOTAL_PER_CLASS,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TARGET_UNIQUE_CLASSIFIER_FIT_COUNT,
    EXPECTED_TOTAL_CLASSIFIER_FIT_COUNT,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERT_BANK_ARTIFACT_ID,
    FALLBACK_ACTION_ID,
    FORBIDDEN_STAGE60_INPUT_ARTIFACT_IDS,
    GENERATION_LOCK_ARTIFACT_ID,
    GENERATION_SEEDS,
    INPUT_ARTIFACT_IDS,
    MAX_SOURCE_WEIGHT,
    MAXIMUM_RESIDENT_GENERATED_EMBEDDING_BYTES,
    MAXIMUM_RESIDENT_GENERATED_SOURCE_BLOCKS,
    MIN_EFFECTIVE_SOURCE_COUNT,
    MINIMUM_INTEGER_ALLOCATION_PER_SOURCE,
    NONUNIFORM_PASS_RULE,
    OUTPUT_ARTIFACT_ID,
    PRIMARY_METRIC,
    PUBLICATION_STATUS,
    RHO_VALUES,
    SECONDARY_METRIC,
    SELECTION_OBJECTIVE,
    STAGE_ID,
    SUPPORT_CASE_COUNT,
    SUPPORT_PARTITION_NAMESPACE,
    SUPPORT_SPLIT_SEED,
    TEMPERATURE,
    TOTAL_PER_CLASS,
    TRAINING_SEEDS,
    VALIDATION_CACHE_ARTIFACT_ID,
    VALIDATION_CACHE_REPRESENTATION_ID,
    VALIDATION_CACHE_SEMANTIC_ID,
    VALIDATION_MANIFEST_ARTIFACT_ID,
    VALIDATION_SPLIT,
    action_library,
    development_queries,
    legal_sources,
    target_sources,
)


_TOP_LEVEL_KEYS = frozenset(
    {
        "experiment",
        "inputs",
        "protocol",
        "compatibility",
        "router",
        "classifier",
        "selection",
        "runtime",
        "claim_boundary",
    }
)
_EXPERIMENT_KEYS = frozenset(
    {"id", "name", "artifact_root", "claim_scope", "status"}
)
_INPUT_KEYS = frozenset(
    {
        "expert_bank_root",
        "generation_lock_root",
        "validation_cache_root",
        "validation_manifest_path",
        "expert_bank_artifact_id",
        "generation_lock_artifact_id",
        "validation_cache_artifact_id",
        "validation_manifest_artifact_id",
        "expected_bank_lock_hash",
        "expected_generation_lock_hash",
        "expected_validation_cache_semantic_id",
        "expected_validation_cache_representation_id",
        "expected_manifest_sha256",
    }
)
_CLASSIFIER_KEYS = frozenset(CLASSIFIER.to_payload())
_RUNTIME_KEYS = frozenset(
    {
        "compatibility_device",
        "generation_device",
        "classifier_device",
        "threads_per_fit",
        "expected_development_classifier_fit_count",
        "expected_target_unique_classifier_fit_count",
        "maximum_total_classifier_fit_count",
        "maximum_resident_generated_source_blocks",
        "maximum_resident_generated_embedding_bytes",
        "control_alias_reuses_rho0_fit",
    }
)


def canonical_protocol_payload() -> dict[str, object]:
    return {
        "dataset_family": "MIDOG++",
        "stage": STAGE_ID,
        "validation_split": VALIDATION_SPLIT,
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
        "development_queries_by_outer_target": {
            center: list(development_queries(center)) for center in CENTERS
        },
        "development_sources_by_outer_target_and_query": {
            outer: {
                query: list(legal_sources(outer_target=outer, query_center=query))
                for query in development_queries(outer)
            }
            for outer in CENTERS
        },
        "target_sources_by_target": {
            center: list(target_sources(center)) for center in CENTERS
        },
        "outer_target_excluded_from_development_queries": True,
        "outer_target_excluded_from_development_experts": True,
        "pseudo_target_excluded_from_development_experts": True,
        "source_experts_frozen_source_only": True,
        "expert_checkpoint_frame_sampler_updates_allowed": False,
        "all_action_development_predictions_sealed_before_development_labels": True,
        "all_action_target_predictions_materialized_before_any_label_access": True,
        "selected_and_control_target_predictions_sealed_before_target_labels": True,
    }


def canonical_compatibility_payload() -> dict[str, object]:
    return {
        "semantics": COMPATIBILITY_SEMANTICS,
        "energy_definition": (
            "class_marginalized_common_space_posterior_mean_reconstruction_"
            "mse_plus_latent_dim_normalized_analytic_ps_kl_fixed_class_prior_half"
        ),
        "target_support_labels_used": False,
        "query_rows_consumed": "support_partition_rows_only",
        "query_evaluation_embeddings_consumed": False,
        "class_prior": list(CLASS_PRIOR),
        "class_prior_source": "fixed_protocol_prior",
        "posterior_latent_evaluation": "posterior_mean_deterministic",
        "reconstruction_term": "common_3840_inverse_frame_mse_mean",
        "kl_term": (
            "analytic_kl_q_diag_to_ps_class_full_gaussian_normalized_by_latent_dim"
        ),
        "class_marginalization": "negative_logsumexp_with_fixed_class_prior",
        "own_source_calibration_location": "case_equal_median",
        "own_source_calibration_scale": "1.4826_mad_with_sample_std_fallback",
        "own_source_calibration_scale_floor": "fixed_positive_numeric_floor",
        "own_source_calibration_scale_floor_value": 1.0e-6,
        "query_calibration": "robust_difference_z_score",
        "calibration_semantics": (
            "own_source_case_equal_median_mad_robust_z_then_fixed_three_seed_mean"
        ),
        "replica_aggregation": "arithmetic_mean_across_all_three_training_seed_scores",
        "score_direction": "lower_energy_z_is_more_compatible",
        "exact_nelbo_claimed": False,
        "compatibility_is_proxy": True,
    }


def canonical_router_payload() -> dict[str, object]:
    return {
        "family": "control_anchored_dense_residual_soft_router_v1",
        "weighting_semantics": (
            "uniform_anchored_residual_softmax_negative_calibrated_energy_"
            "automatic_max_weight_and_effective_source_constraints"
        ),
        "action_ids": list(ACTION_IDS),
        "rhos": list(RHO_VALUES),
        "temperature": TEMPERATURE,
        "compatibility_to_weight_map": "softmax_of_negative_calibrated_energy_over_tau",
        "residual_formula": "w_equals_one_minus_rho_times_uniform_plus_rho_times_softmax",
        "absolute_max_source_weight": MAX_SOURCE_WEIGHT,
        "minimum_effective_source_count": MIN_EFFECTIVE_SOURCE_COUNT,
        "effective_source_count_definition": "one_over_sum_squared_weights",
        "constraint_projection": "deterministic_bounded_simplex_projection",
        "integer_allocation": "deterministic_hamilton_largest_remainder",
        "integer_allocation_semantics": (
            "positive_lower_bound_hamilton_largest_remainder_canonical_source_ties"
        ),
        "integer_allocation_tie_break": "canonical_source_center_order",
        "minimum_integer_allocation_per_source": (
            MINIMUM_INTEGER_ALLOCATION_PER_SOURCE
        ),
        "development_total_generated_samples_per_class": (
            DEVELOPMENT_TOTAL_PER_CLASS
        ),
        "target_total_generated_samples_per_class": TOTAL_PER_CLASS,
        "class_labels": [0, 1],
        "source_stream_slice": "frozen_generation_lock_first_n_prefix_per_class",
        "composition_semantics": (
            "canonical_source_order_class_prefix_then_fixed_classwise_shuffle"
        ),
        "rho0_exact_equal_union": True,
        "rho0_action_id": CONTROL_ACTION_ID,
        "target_conditioned_labels_used_for_weights": False,
        "source_seed_or_action_posthoc_selection": False,
        "action_library_hash": ACTION_LIBRARY_HASH,
        "actions": [action.to_payload() for action in action_library()],
    }


def canonical_selection_payload() -> dict[str, object]:
    return {
        "primary_metric": PRIMARY_METRIC,
        "secondary_metric": SECONDARY_METRIC,
        "secondary_metric_may_select": False,
        "objective": SELECTION_OBJECTIVE,
        "objective_direction": "minimize",
        "regret_reference": "best_fixed_action_within_development_cell",
        "mean_regret_weight": 1.0,
        "upper_quartile_cvar_regret_weight": 0.5,
        "upper_quartile_cvar_definition": (
            "mean_of_largest_ceil_25_percent_regrets"
        ),
        "uniform_l2_penalty_weight": 0.01,
        "aggregation": "equal_weight_over_q_not_H_and_all_nine_seed_cells",
        "nonuniform_pass_rule": NONUNIFORM_PASS_RULE,
        "fallback_action_id": FALLBACK_ACTION_ID,
        "tie_break": "smallest_rho_then_lexicographic_action_id",
        "selection_scope": "per_outer_target_consumed_validation_diagnostic_only",
        "target_H_labels_may_select": False,
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "diagnostic_only": True,
        "publication_status": PUBLICATION_STATUS,
        "consumed_validation_data": True,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "routing_quality_claimed": False,
        "equal_union_superiority_claim_allowed": False,
        "promotion_allowed": False,
        "expert_bank_promotion_allowed": False,
        "routing_policy_promotion_allowed": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "deployment_claim_allowed": False,
        "test_labels_consumed": False,
        "external_generalization_claim_allowed": False,
    }


@dataclass(frozen=True)
class DenseResidualDiagnosticConfig:
    source_path: Path
    artifact_root: Path
    expert_bank_root: Path
    generation_lock_root: Path
    validation_cache_root: Path
    validation_manifest_path: Path
    expert_bank_artifact_id: str
    generation_lock_artifact_id: str
    validation_cache_artifact_id: str
    validation_manifest_artifact_id: str
    expected_bank_lock_hash: str
    expected_generation_lock_hash: str
    expected_validation_cache_semantic_id: str
    expected_validation_cache_representation_id: str
    expected_manifest_sha256: str
    protocol: Mapping[str, object]
    compatibility: Mapping[str, object]
    router: Mapping[str, object]
    classifier: ClassifierSpec
    selection: Mapping[str, object]
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]
    contract_hash: str

    @property
    def input_artifact_ids(self) -> tuple[str, ...]:
        return (
            self.expert_bank_artifact_id,
            self.generation_lock_artifact_id,
            self.validation_cache_artifact_id,
            self.validation_manifest_artifact_id,
        )

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.router["action_ids"])  # type: ignore[index]

    @property
    def compatibility_device(self) -> str:
        return str(self.runtime["compatibility_device"])

    @property
    def generation_device(self) -> str:
        return str(self.runtime["generation_device"])

    @property
    def classifier_device(self) -> str:
        return str(self.runtime["classifier_device"])

    @property
    def threads_per_fit(self) -> int:
        return int(self.runtime["threads_per_fit"])


def load_dense_residual_diagnostic_config(
    path: str | Path,
) -> DenseResidualDiagnosticConfig:
    source = Path(path).resolve()
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError(f"Cannot read dense residual diagnostic config: {source}.") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("Dense residual diagnostic config must be a mapping.")
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, "top-level config")
    _reject_pending_placeholders(payload)

    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    protocol = _mapping(payload, "protocol")
    compatibility = _mapping(payload, "compatibility")
    router = _mapping(payload, "router")
    classifier_raw = _mapping(payload, "classifier")
    selection = _mapping(payload, "selection")
    runtime = _mapping(payload, "runtime")
    claim_boundary = _mapping(payload, "claim_boundary")

    _require_exact_keys(experiment, _EXPERIMENT_KEYS, "experiment")
    _require_exact_keys(inputs, _INPUT_KEYS, "inputs")
    _require_exact_keys(
        protocol, frozenset(canonical_protocol_payload()), "protocol"
    )
    _require_exact_keys(
        compatibility,
        frozenset(canonical_compatibility_payload()),
        "compatibility",
    )
    _require_exact_keys(router, frozenset(canonical_router_payload()), "router")
    _require_exact_keys(classifier_raw, _CLASSIFIER_KEYS, "classifier")
    _require_exact_keys(selection, frozenset(canonical_selection_payload()), "selection")
    _require_exact_keys(runtime, _RUNTIME_KEYS, "runtime")
    _require_exact_keys(
        claim_boundary,
        frozenset(canonical_claim_boundary_payload()),
        "claim boundary",
    )

    observed_input_ids = {
        str(inputs["expert_bank_artifact_id"]),
        str(inputs["generation_lock_artifact_id"]),
        str(inputs["validation_cache_artifact_id"]),
        str(inputs["validation_manifest_artifact_id"]),
    }
    forbidden = observed_input_ids.intersection(FORBIDDEN_STAGE60_INPUT_ARTIFACT_IDS)
    if forbidden:
        raise ProtocolError(
            "Dense residual diagnostic rejects original Stage-60 validation aliases: "
            f"{sorted(forbidden)!r}."
        )

    _require_exact_values(protocol, canonical_protocol_payload(), "protocol")
    _require_exact_values(
        compatibility,
        canonical_compatibility_payload(),
        "compatibility",
    )
    _require_exact_values(router, canonical_router_payload(), "router")
    _require_exact_values(selection, canonical_selection_payload(), "selection")
    _require_exact_values(
        claim_boundary,
        canonical_claim_boundary_payload(),
        "claim boundary",
    )

    classifier = _classifier(classifier_raw)
    if classifier != CLASSIFIER:
        raise ProtocolError("Dense residual classifier differs from the frozen control.")

    exact = {
        "experiment id": (str(experiment["id"]), EXPERIMENT_ID),
        "experiment name": (str(experiment["name"]), EXPERIMENT_NAME),
        "claim scope": (str(experiment["claim_scope"]), CLAIM_SCOPE),
        "publication status": (str(experiment["status"]), PUBLICATION_STATUS),
        "expert-bank artifact": (
            str(inputs["expert_bank_artifact_id"]),
            EXPERT_BANK_ARTIFACT_ID,
        ),
        "GenerationLock artifact": (
            str(inputs["generation_lock_artifact_id"]),
            GENERATION_LOCK_ARTIFACT_ID,
        ),
        "validation-cache diagnostic alias": (
            str(inputs["validation_cache_artifact_id"]),
            VALIDATION_CACHE_ARTIFACT_ID,
        ),
        "validation-manifest diagnostic alias": (
            str(inputs["validation_manifest_artifact_id"]),
            VALIDATION_MANIFEST_ARTIFACT_ID,
        ),
        "bank lock": (
            str(inputs["expected_bank_lock_hash"]),
            EXPECTED_BANK_LOCK_HASH,
        ),
        "generation lock": (
            str(inputs["expected_generation_lock_hash"]),
            EXPECTED_GENERATION_LOCK_HASH,
        ),
        "cache semantic id": (
            str(inputs["expected_validation_cache_semantic_id"]),
            VALIDATION_CACHE_SEMANTIC_ID,
        ),
        "cache representation id": (
            str(inputs["expected_validation_cache_representation_id"]),
            VALIDATION_CACHE_REPRESENTATION_ID,
        ),
        "validation manifest SHA-256": (
            str(inputs["expected_manifest_sha256"]),
            EXPECTED_MANIFEST_SHA256,
        ),
    }
    drift = [
        f"{role}: observed={observed!r}, expected={expected!r}"
        for role, (observed, expected) in exact.items()
        if observed != expected
    ]
    if drift:
        raise ProtocolError("Dense residual diagnostic identity drifted: " + "; ".join(drift))

    _validate_artifact_location(
        inputs["expert_bank_root"],
        artifact_id=EXPERT_BANK_ARTIFACT_ID,
        role="expert-bank root",
    )
    _validate_artifact_location(
        inputs["generation_lock_root"],
        artifact_id=GENERATION_LOCK_ARTIFACT_ID,
        role="GenerationLock root",
    )
    _validate_artifact_location(
        inputs["validation_cache_root"],
        artifact_id=VALIDATION_CACHE_ARTIFACT_ID,
        role="validation-cache diagnostic alias",
    )
    _validate_artifact_location(
        inputs["validation_manifest_path"],
        artifact_id=VALIDATION_MANIFEST_ARTIFACT_ID,
        role="validation-manifest diagnostic alias",
        member="manifest.csv",
    )
    output_location = _required_string(experiment["artifact_root"], "artifact root")
    if output_location.startswith("output://") and output_location != (
        f"output://{OUTPUT_ARTIFACT_ID}"
    ):
        raise ProtocolError("Dense residual output alias identity drifted.")

    runtime_payload = dict(runtime)
    for key in ("compatibility_device", "generation_device"):
        value = _required_string(runtime_payload[key], key)
        if value != "cpu" and not value.startswith("cuda"):
            raise ProtocolError(f"Dense residual {key} must name cpu or a CUDA device.")
    if _required_string(runtime_payload["classifier_device"], "classifier device") != "cpu":
        raise ProtocolError("Dense residual classifier execution must remain on CPU.")
    threads = _integer(runtime_payload["threads_per_fit"], "threads_per_fit")
    if threads <= 0:
        raise ProtocolError("Dense residual threads_per_fit must be positive.")
    expected_runtime_values = {
        "expected_development_classifier_fit_count": (
            EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT
        ),
        "expected_target_unique_classifier_fit_count": (
            EXPECTED_TARGET_UNIQUE_CLASSIFIER_FIT_COUNT
        ),
        "maximum_total_classifier_fit_count": EXPECTED_TOTAL_CLASSIFIER_FIT_COUNT,
        "maximum_resident_generated_source_blocks": (
            MAXIMUM_RESIDENT_GENERATED_SOURCE_BLOCKS
        ),
        "maximum_resident_generated_embedding_bytes": (
            MAXIMUM_RESIDENT_GENERATED_EMBEDDING_BYTES
        ),
    }
    for key, expected_value in expected_runtime_values.items():
        if _integer(runtime_payload[key], key) != expected_value:
            raise ProtocolError(
                f"Dense residual runtime budget {key} drifted from {expected_value}."
            )
    if runtime_payload["control_alias_reuses_rho0_fit"] is not True:
        raise ProtocolError("Dense residual rho0 control fit alias must remain enabled.")

    scientific_payload = {
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "publication_status": PUBLICATION_STATUS,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "expected_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
        "expected_generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
        "expected_validation_cache_semantic_id": VALIDATION_CACHE_SEMANTIC_ID,
        "expected_validation_cache_representation_id": (
            VALIDATION_CACHE_REPRESENTATION_ID
        ),
        "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "protocol": dict(protocol),
        "compatibility": dict(compatibility),
        "router": dict(router),
        "classifier": classifier.to_payload(),
        "selection": dict(selection),
        "claim_boundary": dict(claim_boundary),
    }

    return DenseResidualDiagnosticConfig(
        source_path=source,
        artifact_root=_path(source.parent, experiment["artifact_root"], "artifact root"),
        expert_bank_root=_path(
            source.parent, inputs["expert_bank_root"], "expert-bank root"
        ),
        generation_lock_root=_path(
            source.parent, inputs["generation_lock_root"], "GenerationLock root"
        ),
        validation_cache_root=_path(
            source.parent, inputs["validation_cache_root"], "validation-cache root"
        ),
        validation_manifest_path=_path(
            source.parent,
            inputs["validation_manifest_path"],
            "validation manifest path",
        ),
        expert_bank_artifact_id=EXPERT_BANK_ARTIFACT_ID,
        generation_lock_artifact_id=GENERATION_LOCK_ARTIFACT_ID,
        validation_cache_artifact_id=VALIDATION_CACHE_ARTIFACT_ID,
        validation_manifest_artifact_id=VALIDATION_MANIFEST_ARTIFACT_ID,
        expected_bank_lock_hash=EXPECTED_BANK_LOCK_HASH,
        expected_generation_lock_hash=EXPECTED_GENERATION_LOCK_HASH,
        expected_validation_cache_semantic_id=VALIDATION_CACHE_SEMANTIC_ID,
        expected_validation_cache_representation_id=(
            VALIDATION_CACHE_REPRESENTATION_ID
        ),
        expected_manifest_sha256=EXPECTED_MANIFEST_SHA256,
        protocol=dict(protocol),
        compatibility=dict(compatibility),
        router=dict(router),
        classifier=classifier,
        selection=dict(selection),
        runtime=runtime_payload,
        claim_boundary=dict(claim_boundary),
        contract_hash=stable_hash(scientific_payload),
    )


def _classifier(payload: Mapping[str, object]) -> ClassifierSpec:
    try:
        return ClassifierSpec(
            family=_required_string(payload["family"], "classifier family"),
            C=_finite_float(payload["C"], "classifier C"),
            penalty=_required_string(payload["penalty"], "classifier penalty"),
            solver=_required_string(payload["solver"], "classifier solver"),
            max_iter=_integer(payload["max_iter"], "classifier max_iter"),
            class_weight=(
                None
                if payload["class_weight"] is None
                else _required_string(payload["class_weight"], "classifier class weight")
            ),
            random_state=_integer(
                payload["random_state"], "classifier random state"
            ),
            l1_ratio=(
                None
                if payload["l1_ratio"] is None
                else _finite_float(payload["l1_ratio"], "classifier l1 ratio")
            ),
            threshold_policy=_required_string(
                payload["threshold_policy"], "classifier threshold policy"
            ),
            scaler_fit=_required_string(payload["scaler_fit"], "classifier scaler fit"),
        )
    except (KeyError, ValueError) as exc:
        raise ProtocolError("Dense residual classifier configuration is invalid.") from exc


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Dense residual config lacks mapping {key!r}.")
    return value


def _require_exact_keys(
    payload: Mapping[object, object], required: frozenset[str], role: str
) -> None:
    if any(not isinstance(key, str) for key in payload):
        raise ProtocolError(f"Dense residual {role} keys must be strings.")
    observed = set(payload)
    if observed != set(required):
        raise ProtocolError(
            f"Dense residual {role} keys drifted: "
            f"missing={sorted(set(required) - observed)!r}, "
            f"extra={sorted(observed - set(required))!r}."
        )


def _require_exact_values(
    observed: Mapping[str, object], expected: Mapping[str, object], role: str
) -> None:
    mismatches = [
        key
        for key, expected_value in expected.items()
        if not _strict_equal(observed.get(key), expected_value)
    ]
    if mismatches:
        raise ProtocolError(
            f"Dense residual {role} values drifted: {sorted(mismatches)!r}."
        )


def _strict_equal(observed: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return observed is expected
    if isinstance(expected, Mapping):
        return isinstance(observed, Mapping) and set(observed) == set(expected) and all(
            _strict_equal(observed[key], expected_value)
            for key, expected_value in expected.items()
        )
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        return (
            isinstance(observed, Sequence)
            and not isinstance(observed, (str, bytes))
            and len(observed) == len(expected)
            and all(
                _strict_equal(observed_value, expected_value)
                for observed_value, expected_value in zip(
                    observed, expected, strict=True
                )
            )
        )
    return observed == expected


def _path(base: Path, value: object, role: str) -> Path:
    rendered = _required_string(value, role)
    if rendered.startswith(("artifact://", "output://")):
        return Path(rendered)
    raw = Path(rendered).expanduser()
    return raw.resolve() if raw.is_absolute() else (base / raw).resolve()


def _validate_artifact_location(
    value: object,
    *,
    artifact_id: str,
    role: str,
    member: str = "",
) -> None:
    rendered = _required_string(value, role)
    expected = f"artifact://{artifact_id}"
    if member:
        expected = f"{expected}/{member}"
    if rendered.startswith("artifact://") and rendered != expected:
        raise ProtocolError(
            f"Dense residual {role} must use exact diagnostic alias {expected!r}."
        )


def _required_string(value: object, role: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProtocolError(f"Dense residual {role} must be a trimmed non-empty string.")
    return value


def _integer(value: object, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"Dense residual {role} must be an integer.")
    return value


def _finite_float(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"Dense residual {role} must be numeric.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ProtocolError(f"Dense residual {role} must be finite.")
    return parsed


def _reject_pending_placeholders(value: object, *, location: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_pending_placeholders(nested, location=f"{location}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _reject_pending_placeholders(nested, location=f"{location}[{index}]")
        return
    if isinstance(value, str) and value.strip().upper().startswith("PENDING"):
        raise ProtocolError(
            f"Dense residual production config contains PENDING at {location}."
        )


__all__ = (
    "DenseResidualDiagnosticConfig",
    "canonical_claim_boundary_payload",
    "canonical_compatibility_payload",
    "canonical_protocol_payload",
    "canonical_router_payload",
    "canonical_selection_payload",
    "load_dense_residual_diagnostic_config",
)
