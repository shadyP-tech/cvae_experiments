"""Fail-closed configuration for the utility/regret policy lock."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ....common.hashing import stable_hash
from ...generation.contracts import (
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
)
from ...protocol import ProtocolError
from ..contracts import (
    EXPECTED_ASSIGNMENT_TABLE_HASH as EXPECTED_EQUAL_UNION_ASSIGNMENT_TABLE_HASH,
    EXPECTED_POLICY_LOCK_HASH as EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,
    EXPECTED_POLICY_PLAN_HASH as EXPECTED_EQUAL_UNION_POLICY_PLAN_HASH,
)
from .contracts import (
    BOOTSTRAP_MAX_ATTEMPTS,
    BOOTSTRAP_SEED,
    BOOTSTRAP_VALID_REPLICATES,
    CENTERS,
    CLAIM_SCOPE,
    CONSUMPTION_RULE_HASH,
    EQUAL_UNION_ARTIFACT_ID,
    EXPECTED_CONFIG_CONTRACT_HASH,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    GENERATION_SEEDS,
    MARGIN_LOWER_QUANTILE,
    OUTPUT_ARTIFACT_ID,
    POLICY_FAMILY,
    POLICY_NAMESPACE,
    PRIMARY_UTILITY,
    SECONDARY_METRIC,
    TRAINING_SEEDS,
    UTILITY_ARTIFACT_ID,
    WIN_PROBABILITY_THRESHOLD,
)


@dataclass(frozen=True)
class UtilityRegretPolicyConfig:
    experiment_id: str
    name: str
    artifact_root: Path
    bank_root: Path
    generation_lock_root: Path
    equal_union_root: Path
    utility_root: Path
    input_artifact_ids: tuple[str, ...]
    expected_bank_lock_hash: str
    expected_generation_lock_hash: str
    expected_equal_union_policy_lock_hash: str
    expected_equal_union_policy_plan_hash: str
    expected_equal_union_assignment_table_hash: str
    expected_consumption_rule_hash: str
    policy_contract: Mapping[str, object]
    execution: Mapping[str, object]
    claim_boundary: Mapping[str, object]

    @property
    def contract_hash(self) -> str:
        return stable_hash(
            {
                "experiment_id": self.experiment_id,
                "input_artifact_ids": list(self.input_artifact_ids),
                "expected_bank_lock_hash": self.expected_bank_lock_hash,
                "expected_generation_lock_hash": self.expected_generation_lock_hash,
                "expected_equal_union_policy_lock_hash": (
                    self.expected_equal_union_policy_lock_hash
                ),
                "expected_equal_union_policy_plan_hash": (
                    self.expected_equal_union_policy_plan_hash
                ),
                "expected_equal_union_assignment_table_hash": (
                    self.expected_equal_union_assignment_table_hash
                ),
                "expected_consumption_rule_hash": self.expected_consumption_rule_hash,
                "policy_contract": dict(self.policy_contract),
                "execution": dict(self.execution),
                "claim_boundary": dict(self.claim_boundary),
            }
        )


def load_utility_regret_policy_config(path: str | Path) -> UtilityRegretPolicyConfig:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ProtocolError("Utility/regret policy config must be a mapping.")
    _exact_keys(
        payload,
        {"experiment", "inputs", "policy_contract", "execution", "claim_boundary"},
        "top-level config",
    )
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    base = config_path.parent
    ids = (
        str(inputs.get("bank_artifact_id", "")),
        str(inputs.get("generation_lock_artifact_id", "")),
        str(inputs.get("equal_union_artifact_id", "")),
        str(inputs.get("utility_artifact_id", "")),
    )
    config = UtilityRegretPolicyConfig(
        experiment_id=str(experiment.get("id", "")),
        name=str(experiment.get("name", "")),
        artifact_root=_path(base, experiment.get("artifact_root")),
        bank_root=_path(base, inputs.get("bank_root")),
        generation_lock_root=_path(base, inputs.get("generation_lock_root")),
        equal_union_root=_path(base, inputs.get("equal_union_root")),
        utility_root=_path(base, inputs.get("utility_root")),
        input_artifact_ids=ids,
        expected_bank_lock_hash=str(inputs.get("expected_bank_lock_hash", "")),
        expected_generation_lock_hash=str(
            inputs.get("expected_generation_lock_hash", "")
        ),
        expected_equal_union_policy_lock_hash=str(
            inputs.get("expected_equal_union_policy_lock_hash", "")
        ),
        expected_equal_union_policy_plan_hash=str(
            inputs.get("expected_equal_union_policy_plan_hash", "")
        ),
        expected_equal_union_assignment_table_hash=str(
            inputs.get("expected_equal_union_assignment_table_hash", "")
        ),
        expected_consumption_rule_hash=str(
            inputs.get("expected_consumption_rule_hash", "")
        ),
        policy_contract=dict(_mapping(payload, "policy_contract")),
        execution=dict(_mapping(payload, "execution")),
        claim_boundary=dict(_mapping(payload, "claim_boundary")),
    )
    _validate(config)
    return config


def _validate(config: UtilityRegretPolicyConfig) -> None:
    exact = {
        "experiment_id": (config.experiment_id, EXPERIMENT_ID),
        "name": (config.name, EXPERIMENT_NAME),
        "input_artifact_ids": (
            config.input_artifact_ids,
            (
                EXPERT_BANK_ARTIFACT_ID,
                GENERATION_LOCK_ARTIFACT_ID,
                EQUAL_UNION_ARTIFACT_ID,
                UTILITY_ARTIFACT_ID,
            ),
        ),
        "bank_lock_hash": (config.expected_bank_lock_hash, EXPECTED_BANK_LOCK_HASH),
        "generation_lock_hash": (
            config.expected_generation_lock_hash,
            EXPECTED_GENERATION_LOCK_HASH,
        ),
        "equal_union_policy_lock_hash": (
            config.expected_equal_union_policy_lock_hash,
            EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,
        ),
        "equal_union_policy_plan_hash": (
            config.expected_equal_union_policy_plan_hash,
            EXPECTED_EQUAL_UNION_POLICY_PLAN_HASH,
        ),
        "equal_union_assignment_table_hash": (
            config.expected_equal_union_assignment_table_hash,
            EXPECTED_EQUAL_UNION_ASSIGNMENT_TABLE_HASH,
        ),
        "consumption_rule_hash": (
            config.expected_consumption_rule_hash,
            CONSUMPTION_RULE_HASH,
        ),
    }
    mismatch = [
        f"{key}: observed={observed!r}, expected={expected!r}"
        for key, (observed, expected) in exact.items()
        if observed != expected
    ]
    if mismatch:
        raise ProtocolError("Utility/regret policy identity drifted: " + "; ".join(mismatch))
    expected_policy = {
        "family": POLICY_FAMILY,
        "namespace": POLICY_NAMESPACE,
        "centers": list(CENTERS),
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "candidate_unit": "source_center",
        "primary_utility": PRIMARY_UTILITY,
        "secondary_metric": SECONDARY_METRIC,
        "policy_consumption_lock_hash": CONSUMPTION_RULE_HASH,
        "outer_fold_filter": "query_center_not_H_and_candidate_source_not_H",
        "expected_queries_per_outer": 8,
        "expected_candidates_per_outer": 8,
        "expected_legal_candidates_per_query": 7,
        "expected_seed_pairs": 9,
        "expected_utility_rows": 648,
        "expected_outer_regret_cells": 4536,
        "expected_candidate_summaries": 72,
        "expected_selections": 9,
        "target_expert_excluded": True,
        "target_query_rows_excluded_before_any_transform": True,
        "target_candidate_rows_excluded_before_any_transform": True,
        "regret_definition": (
            "paired_best_candidate_bacc_within_q_training_generation_cell_"
            "minus_candidate_bacc"
        ),
        "aggregation": (
            "equal_mean_over_7_retained_pseudo_targets_and_all_9_paired_"
            "training_generation_seed_cells"
        ),
        "bootstrap_levels": [
            "pseudo_target_centers",
            "cases_within_pseudo_target",
            "paired_training_generation_seed_cells",
        ],
        "pass_action": "unique_winner_single_source_full_budget",
        "fallback_action": "exact_frozen_equal_union",
        "macro_f1_may_select": False,
        "no_seed_selection": True,
        "policy_frozen_before_stage70": True,
    }
    _exact_values(config.policy_contract, expected_policy, "policy contract")
    _exact_values(
        config.execution,
        {
            "generation_allowed": False,
            "classifier_fit_allowed": False,
            "dataset_manifest_access_allowed": False,
            "feature_cache_access_allowed": False,
            "raw_label_access_allowed": False,
            "utility_table_access_allowed": True,
            "case_confusion_table_access_allowed": True,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_valid_replicates": BOOTSTRAP_VALID_REPLICATES,
            "bootstrap_max_attempts": BOOTSTRAP_MAX_ATTEMPTS,
            "unique_winner_probability_threshold": WIN_PROBABILITY_THRESHOLD,
            "margin_lower_quantile": MARGIN_LOWER_QUANTILE,
        },
        "execution",
    )
    _exact_values(
        config.claim_boundary,
        {
            "strict_claim_firewall": True,
            "claim_scope": CLAIM_SCOPE,
            "may_feed_deployable_selection": True,
            "source_inner_training_evidence_only": True,
            "target_data_used": False,
            "target_support_used": False,
            "target_labels_used": False,
            "target_evaluation_labels_used": False,
            "stage20_consumed_rows_used": False,
            "stage50_used": False,
            "stage90_used": False,
            "nelbo_used": False,
            "macro_f1_used_for_selection": False,
            "seed_selection_performed": False,
            "routing_policy_frozen": True,
            "routing_quality_claimed": False,
            "downstream_utility_claimed": False,
        },
        "claim boundary",
    )
    if str(config.artifact_root).startswith("output:") and config.artifact_root.name != OUTPUT_ARTIFACT_ID:
        raise ProtocolError("Unexpected utility/regret policy output identity.")
    if config.contract_hash != EXPECTED_CONFIG_CONTRACT_HASH:
        raise ProtocolError("Utility/regret policy config contract identity drifted.")


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Utility/regret config section {key!r} must be a mapping.")
    return value


def _exact_values(observed: Mapping[str, object], expected: Mapping[str, object], label: str) -> None:
    if dict(observed) != dict(expected):
        raise ProtocolError(f"Utility/regret {label} drifted.")


def _exact_keys(observed: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(observed) != expected:
        raise ProtocolError(f"Utility/regret {label} keys drifted.")


def _path(base: Path, value: object) -> Path:
    rendered = str(value or "")
    if not rendered:
        raise ProtocolError("Utility/regret config path is empty.")
    if rendered.startswith(("artifact://", "output://")):
        return Path(rendered)
    path = Path(rendered)
    return path if path.is_absolute() else (base / path).resolve()


__all__ = ("UtilityRegretPolicyConfig", "load_utility_regret_policy_config")
