"""Fail-closed configuration for the Uniform-B v2 equal-union policy lock."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ...common.hashing import stable_hash
from ..expert_bank.uniform_b_v2_promotion.contracts import legal_routing_sources
from ..protocol import ProtocolError
from .contracts import (
    CENTERS,
    CLAIM_SCOPE,
    EXPECTED_ASSIGNMENT_COUNT,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_CONFIG_CONTRACT_HASH,
    EXPECTED_GENERATION_CONTENT_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_REPLICATE_PLAN_HASH,
    EXPECTED_REPLICATE_COUNT,
    EXPECTED_SOURCE_PLAN_HASH,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    GENERATION_SEEDS,
    OUTPUT_ARTIFACT_ID,
    POLICY_FAMILY,
    POLICY_NAMESPACE,
    REPLICATE_POLICY,
    SOURCE_BUDGET_PER_CLASS,
    SOURCES_PER_TARGET,
    TOTAL_PER_CLASS,
    TRAINING_SEEDS,
)


@dataclass(frozen=True)
class UniformBV2EqualUnionPolicyConfig:
    experiment_id: str
    name: str
    artifact_root: Path
    bank_root: Path
    generation_lock_root: Path
    bank_artifact_id: str
    generation_lock_artifact_id: str
    expected_bank_lock_hash: str
    expected_generation_lock_hash: str
    expected_generation_content_hash: str
    expected_source_plan_hash: str
    expected_replicate_plan_hash: str
    policy_contract: Mapping[str, object]
    composition_execution: Mapping[str, object]
    future_evaluation_contract: Mapping[str, object]
    execution: Mapping[str, object]
    claim_boundary: Mapping[str, object]

    @property
    def contract_hash(self) -> str:
        """Hash the path- and runtime-independent Stage-60 contract."""

        return stable_hash(
            {
                "experiment_id": self.experiment_id,
                "bank_artifact_id": self.bank_artifact_id,
                "generation_lock_artifact_id": self.generation_lock_artifact_id,
                "expected_bank_lock_hash": self.expected_bank_lock_hash,
                "expected_generation_lock_hash": self.expected_generation_lock_hash,
                "expected_generation_content_hash": self.expected_generation_content_hash,
                "expected_source_plan_hash": self.expected_source_plan_hash,
                "expected_replicate_plan_hash": self.expected_replicate_plan_hash,
                "policy_contract": dict(self.policy_contract),
                "composition_execution": dict(self.composition_execution),
                "future_evaluation_contract": dict(self.future_evaluation_contract),
                "execution": dict(self.execution),
                "claim_boundary": dict(self.claim_boundary),
            }
        )

    @property
    def centers(self) -> tuple[str, ...]:
        return _strings(self.policy_contract.get("centers"))

    @property
    def training_seeds(self) -> tuple[int, ...]:
        return _ints(self.policy_contract.get("training_seeds"))

    @property
    def generation_seeds(self) -> tuple[int, ...]:
        return _ints(self.policy_contract.get("generation_seeds"))


def load_equal_union_policy_config(
    path: str | Path,
) -> UniformBV2EqualUnionPolicyConfig:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ProtocolError("Uniform-B v2 equal-union policy config must be a mapping.")
    _require_exact_keys(
        payload,
        {
            "experiment",
            "inputs",
            "policy_contract",
            "composition_execution",
            "future_evaluation_contract",
            "execution",
            "claim_boundary",
        },
        "top-level config",
    )
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    base = config_path.parent
    config = UniformBV2EqualUnionPolicyConfig(
        experiment_id=str(experiment.get("id", "")),
        name=str(experiment.get("name", "")),
        artifact_root=_path(base, experiment.get("artifact_root")),
        bank_root=_path(base, inputs.get("bank_root")),
        generation_lock_root=_path(base, inputs.get("generation_lock_root")),
        bank_artifact_id=str(inputs.get("bank_artifact_id", "")),
        generation_lock_artifact_id=str(inputs.get("generation_lock_artifact_id", "")),
        expected_bank_lock_hash=str(inputs.get("expected_bank_lock_hash", "")),
        expected_generation_lock_hash=str(
            inputs.get("expected_generation_lock_hash", "")
        ),
        expected_generation_content_hash=str(
            inputs.get("expected_generation_content_hash", "")
        ),
        expected_source_plan_hash=str(inputs.get("expected_source_plan_hash", "")),
        expected_replicate_plan_hash=str(inputs.get("expected_replicate_plan_hash", "")),
        policy_contract=dict(_mapping(payload, "policy_contract")),
        composition_execution=dict(_mapping(payload, "composition_execution")),
        future_evaluation_contract=dict(_mapping(payload, "future_evaluation_contract")),
        execution=dict(_mapping(payload, "execution")),
        claim_boundary=dict(_mapping(payload, "claim_boundary")),
    )
    _validate(config, experiment=experiment, inputs=inputs)
    return config


def _validate(
    config: UniformBV2EqualUnionPolicyConfig,
    *,
    experiment: Mapping[str, object],
    inputs: Mapping[str, object],
) -> None:
    _require_exact_keys(experiment, {"id", "name", "artifact_root"}, "experiment")
    _require_exact_keys(
        inputs,
        {
            "bank_root",
            "generation_lock_root",
            "bank_artifact_id",
            "generation_lock_artifact_id",
            "expected_bank_lock_hash",
            "expected_generation_lock_hash",
            "expected_generation_content_hash",
            "expected_source_plan_hash",
            "expected_replicate_plan_hash",
        },
        "inputs",
    )
    exact = {
        "experiment_id": (config.experiment_id, EXPERIMENT_ID),
        "name": (config.name, EXPERIMENT_NAME),
        "bank_artifact_id": (config.bank_artifact_id, EXPERT_BANK_ARTIFACT_ID),
        "generation_lock_artifact_id": (
            config.generation_lock_artifact_id,
            GENERATION_LOCK_ARTIFACT_ID,
        ),
        "expected_bank_lock_hash": (
            config.expected_bank_lock_hash,
            EXPECTED_BANK_LOCK_HASH,
        ),
        "expected_generation_lock_hash": (
            config.expected_generation_lock_hash,
            EXPECTED_GENERATION_LOCK_HASH,
        ),
        "expected_generation_content_hash": (
            config.expected_generation_content_hash,
            EXPECTED_GENERATION_CONTENT_HASH,
        ),
        "expected_source_plan_hash": (
            config.expected_source_plan_hash,
            EXPECTED_SOURCE_PLAN_HASH,
        ),
        "expected_replicate_plan_hash": (
            config.expected_replicate_plan_hash,
            EXPECTED_REPLICATE_PLAN_HASH,
        ),
        "centers": (config.centers, CENTERS),
        "training_seeds": (config.training_seeds, TRAINING_SEEDS),
        "generation_seeds": (config.generation_seeds, GENERATION_SEEDS),
    }
    mismatch = [
        f"{key}: observed={observed!r}, expected={expected!r}"
        for key, (observed, expected) in exact.items()
        if observed != expected
    ]
    if mismatch:
        raise ProtocolError(
            "Uniform-B v2 equal-union policy identity drifted: " + "; ".join(mismatch)
        )

    expected_policy: dict[str, object] = {
        "family": POLICY_FAMILY,
        "namespace": POLICY_NAMESPACE,
        "centers": list(CENTERS),
        "class_labels": [0, 1],
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "seed_pairing": "cartesian_product",
        "replicate_policy": REPLICATE_POLICY,
        "candidate_sources_by_target": {
            target: list(legal_routing_sources(target)) for target in CENTERS
        },
        "target_identity_role": (
            "fold_identity_candidate_exclusion_and_label_blind_shuffle_seeding_only"
        ),
        "source_policy": "all_eligible_sources",
        "sources_per_target": SOURCES_PER_TARGET,
        "source_budget_per_class": SOURCE_BUDGET_PER_CLASS,
        "total_per_class": TOTAL_PER_CLASS,
        "prefix_allocation": True,
        "source_counts_equal": True,
        "target_expert_excluded": True,
        "all_eligible_sources_retained": True,
        "no_expert_selection": True,
        "no_seed_selection": True,
        "no_source_ranking": True,
        "no_learned_source_weighting": True,
        "expected_replicate_count": EXPECTED_REPLICATE_COUNT,
        "expected_assignments_per_replicate": SOURCES_PER_TARGET,
        "expected_assignment_count": EXPECTED_ASSIGNMENT_COUNT,
        "policy_frozen_before_target_evaluation": True,
    }
    _require_exact_values(config.policy_contract, expected_policy, "policy contract")
    _require_exact_values(
        config.composition_execution,
        {
            "source_slice": "first_n_from_generation_lock_stream",
            "source_prefix_per_class": SOURCE_BUDGET_PER_CLASS,
            "source_concatenation_order": "candidate_sources_by_target_order",
            "class_composition_order": [0, 1],
            "shuffle_scope": "independently_within_class_after_union",
            "shuffle_algorithm": "numpy_generator_pcg64_permutation",
            "shuffle_seed_source": "generation_lock_equal_union_replicate_plan",
            "shuffle_seed_field": "class_shuffle_seed_by_label",
            "shuffle_applied_once_per_class": True,
            "final_class_concatenation_order": [0, 1],
            "embedding_dtype": "float32",
            "embedding_dim": 3840,
            "labels_derived_from_class_stream": True,
        },
        "composition execution",
    )
    _require_exact_values(
        config.future_evaluation_contract,
        {
            "authorization": "separate_stage70_target_eval_artifact_required",
            "future_target_evaluation_rows": (
                "all_authorized_target_rows_paired_across_policies"
            ),
            "row_filtering": "none",
            "row_subsampling": "none",
            "row_order": "authorized_artifact_canonical_order",
            "labels_access": "metrics_only_after_predictions",
            "support_rows_used": False,
            "target_identity_role": (
                "fold_membership_candidate_exclusion_and_label_blind_shuffle_seeding_only"
            ),
            "target_identity_as_predictive_feature": False,
            "identity_overlap_audit_required": True,
            "evaluation_occurs_in_stage60": False,
            "evaluation_stage": "70_frozen_policy_downstream",
        },
        "future evaluation contract",
    )
    _require_exact_values(
        config.execution,
        {
            "lock_only": True,
            "generation_performed": False,
            "model_training_allowed": False,
            "sampler_refit_allowed": False,
            "frame_refit_allowed": False,
            "classifier_fit_allowed": False,
            "target_dataset_access_allowed": False,
            "support_set_access_allowed": False,
            "metric_computation_allowed": False,
        },
        "execution",
    )
    _require_exact_values(
        config.claim_boundary,
        {
            "strict_claim_firewall": True,
            "claim_scope": CLAIM_SCOPE,
            "lock_only": True,
            "may_feed_deployable_selection": True,
            "canonical_control": True,
            "policy_frozen_before_stage70": True,
            "source_only_frozen_state": True,
            "target_identity_used_for_fold_candidate_exclusion_and_label_blind_shuffle_only": True,
            "target_identity_used_as_predictive_feature": False,
            "target_data_used": False,
            "target_support_used": False,
            "target_labels_used": False,
            "target_evaluation_labels_used": False,
            "target_metadata_used": False,
            "compatibility_scores_computed": False,
            "nelbo_computed": False,
            "expert_selection_performed": False,
            "seed_selection_performed": False,
            "source_ranking_performed": False,
            "source_weighting_learned": False,
            "routing_policy_frozen": True,
            "routing_quality_claimed": False,
            "generation_performed": False,
            "classifier_fit_performed": False,
            "bacc_computed": False,
            "macro_f1_computed": False,
            "downstream_utility_computed": False,
            "stage20_scores_reused": False,
        },
        "claim boundary",
    )
    if (
        str(config.artifact_root).startswith("output:")
        and config.artifact_root.name != OUTPUT_ARTIFACT_ID
    ):
        raise ProtocolError("Unexpected Uniform-B v2 equal-union policy output identity.")
    if config.contract_hash != EXPECTED_CONFIG_CONTRACT_HASH:
        raise ProtocolError("Uniform-B v2 equal-union config contract identity drifted.")


def _require_exact_values(
    observed: Mapping[str, object], expected: Mapping[str, object], label: str
) -> None:
    _require_exact_keys(observed, set(expected), label)
    mismatch = [
        f"{key}: observed={observed.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if observed.get(key) != value
    ]
    if mismatch:
        raise ProtocolError(
            f"Uniform-B v2 equal-union {label} drifted: " + "; ".join(mismatch)
        )


def _require_exact_keys(
    observed: Mapping[str, object], expected: set[str], label: str
) -> None:
    actual = {str(key) for key in observed}
    if actual != expected:
        raise ProtocolError(
            f"Uniform-B v2 equal-union {label} keys drifted: "
            f"observed={sorted(actual)!r}, expected={sorted(expected)!r}."
        )


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Equal-union policy config section {key!r} must be a mapping.")
    return value


def _ints(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ProtocolError("Equal-union policy config expected an integer list.")
    return tuple(int(item) for item in value)


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ProtocolError("Equal-union policy config expected a string list.")
    return tuple(str(item) for item in value)


def _path(base: Path, value: object) -> Path:
    rendered = str(value or "")
    if not rendered:
        raise ProtocolError("Equal-union policy config path is empty.")
    if rendered.startswith(("artifact://", "output://")):
        return Path(rendered)
    path = Path(rendered)
    return path if path.is_absolute() else (base / path).resolve()


__all__ = (
    "UniformBV2EqualUnionPolicyConfig",
    "load_equal_union_policy_config",
)
