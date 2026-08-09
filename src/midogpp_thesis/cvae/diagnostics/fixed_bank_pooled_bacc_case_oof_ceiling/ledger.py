"""Exact parent/amendment hash-chain validation for pooled-BACC v2."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_file
from .experiment_contracts import (
    EXPERIMENT_ID,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    LEDGER_AMENDMENT_ARTIFACT_ID,
)


class LedgerInputConfig(Protocol):
    experiment_id: str
    test_consumption_ledger_path: Path
    ledger_amendment_path: Path


@dataclass(frozen=True)
class ValidatedLedgerChain:
    parent: Mapping[str, object]
    amendment: Mapping[str, object]


def load_validated_ledger_chain(config: LedgerInputConfig) -> ValidatedLedgerChain:
    parent = _json(config.test_consumption_ledger_path)
    parent_sha = sha256_file(config.test_consumption_ledger_path)
    if (
        config.experiment_id != EXPERIMENT_ID
        or parent_sha != EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        or parent.get("schema_version") != "midogpp_uniform_b_test_consumption_ledger_v1"
        or parent.get("status") != "CONSUMED_FOR_REPRESENTATION_ADOPTION"
        or parent.get("split") != "test"
        or parent.get("may_be_reused_as_fresh_representation_selection_evidence") is not False
        or _descriptive_reuse_permission(parent) is not True
    ):
        raise ProtocolError("Pooled-BACC parent test-consumption ledger drifted.")
    amendment = _json(config.ledger_amendment_path)
    required_true = (
        "all_target_probabilities_globally_sealed_before_any_label_access",
        "support_labels_used",
        "single_class_cases_retained_as_sufficient_statistics",
        "evaluation_labels_opened_only_after_all_observed_and_null_actions_sealed",
        "G_H_and_pairwise_priors_sealed_before_H_support_access",
        "all_45_observed_and_450000_null_actions_sealed_before_evaluation_labels",
        "permutation_baseline_B_fixed",
        "permutation_eight_Hxe_multiset_preserved",
    )
    required_false = (
        "fresh_evidence",
        "target_expert_used",
        "shared_model_updated_with_target_labels",
        "promotion_eligible",
        "may_feed_stage50",
        "may_feed_stage60",
        "may_feed_stage70",
        "may_feed_another_stage90",
        "generic_consumer_authorized",
        "prior_stage90_outputs_used",
        "v1_output_used",
        "v1_scratch_or_checkpoint_used",
    )
    required_permutation_semantics = {
        "permutation_primary_statistic": "equal_center_R_minus_G_H",
        "permutation_upper_tail_output_field": "one_sided_p_value",
        "permutation_lower_tail_output_field": "lower_tail_p_value",
        "permutation_two_sided_output_field": "two_sided_p_value",
        "permutation_upper_tail_p_value_formula": "(1+count(null>=observed))/(K+1)",
        "permutation_lower_tail_p_value_formula": "(1+count(null<=observed))/(K+1)",
        "permutation_two_sided_p_value_formula": "min(1,2*min(upper,lower))",
        "permutation_derangement_family": (
            "case_sha256_candidate_order_counter_splitmix64_"
            "nonzero_cyclic_shift_1_to_7_v1"
        ),
        "permutation_candidate_order": (
            "case_specific_sha256_of_seed_fold_id_case_id_action_then_action"
        ),
        "permutation_shift_generator": (
            "independent_counter_splitmix64_per_fold_case_permutation_index"
        ),
        "permutation_shift_range_inclusive": [1, 7],
        "permutation_zero_shift_allowed": False,
        "uniform_over_all_derangements": False,
    }
    if (
        sha256_file(config.ledger_amendment_path) != EXPECTED_LEDGER_AMENDMENT_SHA256
        or amendment.get("schema_version") != "midogpp_test_consumption_ledger_amendment_v2"
        or amendment.get("amendment_id") != LEDGER_AMENDMENT_ARTIFACT_ID
        or amendment.get("parent_sha256") != parent_sha
        or amendment.get("authorized_consumer_experiment_ids") != [config.experiment_id]
        or amendment.get("authorization_scope")
        != "one_additional_terminal_label_aware_pooled_bacc_case_oof_ceiling_v2"
        or amendment.get("support_utility") != "pooled_exact_bacc"
        or amendment.get("uncertainty_unit") != "paired_whole_case_cluster"
        or amendment.get("zero_headroom_normalized_regret") != 0.0
        or amendment.get("zero_headroom_tolerance") != 1.0e-12
        or amendment.get("zero_headroom_interpretation")
        != "no_routing_opportunity"
        or amendment.get("permutation_unit")
        != "complete_candidate_sufficient_statistic_block_derangement_within_H_fold_and_support_case"
        or any(
            amendment.get(key) != expected
            for key, expected in required_permutation_semantics.items()
        )
        or any(amendment.get(key) is not True for key in required_true)
        or any(amendment.get(key) is not False for key in required_false)
    ):
        raise ProtocolError("Pooled-BACC ledger amendment chain or whitelist drifted.")
    return ValidatedLedgerChain(
        parent=MappingProxyType(dict(parent)),
        amendment=MappingProxyType(dict(amendment)),
    )


def _descriptive_reuse_permission(parent: Mapping[str, object]) -> object:
    canonical = "may_be_reused_for_descriptive_locked_model_scoring"
    published = "may_be_reused_for_descriptive_locked-model_scoring"
    if canonical in parent and published in parent and parent[canonical] != parent[published]:
        raise ProtocolError("Pooled-BACC parent ledger has conflicting descriptive-use aliases.")
    return parent[published] if published in parent else parent.get(canonical)


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read pooled-BACC ledger JSON: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("Pooled-BACC ledger must be a JSON object.")
    return value


__all__ = ("ValidatedLedgerChain", "load_validated_ledger_chain")
