"""Scientific contracts for the frozen source-inner utility/regret policy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    TRAINING_SEEDS,
)
from ...generation.contracts import GENERATION_SEEDS, TOTAL_PER_CLASS
from ...protocol import ProtocolError
from ..contracts import (
    EXPECTED_ASSIGNMENT_TABLE_HASH as EXPECTED_EQUAL_UNION_ASSIGNMENT_TABLE_HASH,
    EXPECTED_POLICY_LOCK_HASH as EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,
    EXPECTED_POLICY_PLAN_HASH as EXPECTED_EQUAL_UNION_POLICY_PLAN_HASH,
)
from ..source_inner_utility.contracts import (
    BOOTSTRAP_LOWER_QUANTILE,
    BOOTSTRAP_MAX_ATTEMPTS,
    BOOTSTRAP_SEED,
    BOOTSTRAP_VALID_REPLICATES,
    POLICY_CONSUMPTION_LOCK_HASH,
    PRIMARY_UTILITY_METRIC,
    SECONDARY_METRIC as SOURCE_INNER_SECONDARY_METRIC,
    UNIQUE_WINNER_PROBABILITY_MIN,
    UTILITY_POLICY_FAMILY,
    policy_consumption_lock_payload,
)


EXPERIMENT_ID = (
    "midogpp.routing_and_composition."
    "uniform_b_v2_utility_regret_policy_lock.v1"
)
EXPERIMENT_NAME = "uniform_b_v2_utility_regret_policy_lock_v1"
OUTPUT_ARTIFACT_ID = "midogpp_output_uniform_b_v2_utility_regret_policy_lock_v1"
UTILITY_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_source_inner_candidate_utility_v1"
)
EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
EQUAL_UNION_ARTIFACT_ID = "midogpp_output_uniform_b_v2_equal_union_policy_lock_v1"
CLAIM_SCOPE = "routing_and_composition"

POLICY_FAMILY = UTILITY_POLICY_FAMILY
POLICY_NAMESPACE = "uniform_b_v2_source_inner_utility_regret_policy.v1"
PRIMARY_UTILITY = PRIMARY_UTILITY_METRIC
SECONDARY_METRIC = SOURCE_INNER_SECONDARY_METRIC
WIN_PROBABILITY_THRESHOLD = UNIQUE_WINNER_PROBABILITY_MIN
MARGIN_LOWER_QUANTILE = BOOTSTRAP_LOWER_QUANTILE
SINGLE_SOURCE_BUDGET_PER_CLASS = TOTAL_PER_CLASS
EQUAL_UNION_SOURCE_BUDGET_PER_CLASS = 128
EXPECTED_UTILITY_ROWS = 648
EXPECTED_REGRET_CELLS = 4_536
EXPECTED_CANDIDATE_SUMMARIES = 72
EXPECTED_SELECTIONS = 9
SEED_PAIR_COUNT = 9
EXPECTED_CONFIG_CONTRACT_HASH = "6a7a9a58c25fa2a5"
CONSUMPTION_RULE_HASH = POLICY_CONSUMPTION_LOCK_HASH

OUTPUT_SEMANTIC_IDENTITIES = {
    "policy_lock_contract": "midogpp_uniform_b_v2_utility_regret_policy_lock_v1",
    "config_contract_hash": EXPECTED_CONFIG_CONTRACT_HASH,
    "policy_consumption_lock_hash": CONSUMPTION_RULE_HASH,
    "policy_family": POLICY_FAMILY,
    "expert_bank_lock_hash": "9972a41dcd4814cd",
    "generation_lock_hash": "34e551425710362e",
    "equal_union_policy_lock_hash": EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,
    "equal_union_policy_plan_hash": EXPECTED_EQUAL_UNION_POLICY_PLAN_HASH,
    "equal_union_assignment_table_hash": (
        EXPECTED_EQUAL_UNION_ASSIGNMENT_TABLE_HASH
    ),
}


def consumption_rule_payload() -> dict[str, object]:
    """Return the exact rule frozen before validation labels were opened."""

    return deepcopy(policy_consumption_lock_payload())


@dataclass(frozen=True)
class RegretCell:
    outer_target_center: str
    query_center: str
    candidate_source: str
    training_seed: int
    generation_seed: int
    bacc: float
    macro_f1: float
    oracle_bacc: float
    regret: float
    source_stream_id: str
    utility_row_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_uniform_b_v2_outer_regret_cell_v1",
            "outer_target_center": self.outer_target_center,
            "query_center": self.query_center,
            "candidate_source": self.candidate_source,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "bacc": self.bacc,
            "macro_f1": self.macro_f1,
            "oracle_bacc": self.oracle_bacc,
            "regret": self.regret,
            "source_stream_id": self.source_stream_id,
            "utility_row_hash": self.utility_row_hash,
            "outer_target_query_excluded": True,
            "outer_target_candidate_excluded": True,
        }


@dataclass(frozen=True)
class CandidateSummary:
    outer_target_center: str
    candidate_source: str
    mean_regret: float
    mean_bacc: float
    mean_macro_f1: float
    query_count: int
    seed_pair_count: int
    cell_count: int

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_uniform_b_v2_candidate_regret_summary_v1",
            "outer_target_center": self.outer_target_center,
            "candidate_source": self.candidate_source,
            "mean_regret": self.mean_regret,
            "mean_bacc": self.mean_bacc,
            "mean_macro_f1": self.mean_macro_f1,
            "query_count": self.query_count,
            "seed_pair_count": self.seed_pair_count,
            "cell_count": self.cell_count,
            "seed_selection_performed": False,
        }


@dataclass(frozen=True)
class BootstrapResult:
    outer_target_center: str
    observed_best_source: str
    observed_runner_up_source: str
    observed_best_mean_regret: float
    observed_runner_up_mean_regret: float
    observed_margin: float
    unique_observed_winner: bool
    unique_winner_probability: float
    margin_lower_2_5: float
    margin_upper_97_5: float
    valid_replicates: int
    attempted_replicates: int
    rejected_replicates: int
    gate_passed: bool
    gate_reason: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_uniform_b_v2_regret_bootstrap_result_v1",
            "outer_target_center": self.outer_target_center,
            "observed_best_source": self.observed_best_source,
            "observed_runner_up_source": self.observed_runner_up_source,
            "observed_best_mean_regret": self.observed_best_mean_regret,
            "observed_runner_up_mean_regret": self.observed_runner_up_mean_regret,
            "observed_margin": self.observed_margin,
            "unique_observed_winner": self.unique_observed_winner,
            "unique_winner_probability": self.unique_winner_probability,
            "margin_lower_2_5": self.margin_lower_2_5,
            "margin_upper_97_5": self.margin_upper_97_5,
            "valid_replicates": self.valid_replicates,
            "attempted_replicates": self.attempted_replicates,
            "rejected_replicates": self.rejected_replicates,
            "gate_passed": self.gate_passed,
            "gate_reason": self.gate_reason,
            "selection_probability_threshold": WIN_PROBABILITY_THRESHOLD,
            "margin_lower_quantile": MARGIN_LOWER_QUANTILE,
            "consumption_rule_hash": CONSUMPTION_RULE_HASH,
        }


@dataclass(frozen=True)
class PolicySelection:
    selection_id: str
    target_center: str
    action: str
    candidate_sources: tuple[str, ...]
    selected_source: str
    retained_sources: tuple[str, ...]
    source_budget_per_class: int
    total_per_class: int
    gate_reason: str
    bootstrap: BootstrapResult

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_uniform_b_v2_utility_regret_selection_v1",
            "selection_id": self.selection_id,
            "target_center": self.target_center,
            "action": self.action,
            "candidate_sources": "|".join(self.candidate_sources),
            "candidate_source_count": len(self.candidate_sources),
            "selected_source": self.selected_source,
            "retained_sources": "|".join(self.retained_sources),
            "retained_source_count": len(self.retained_sources),
            "source_budget_per_class": self.source_budget_per_class,
            "total_per_class": self.total_per_class,
            "gate_reason": self.gate_reason,
            "observed_best_source": self.bootstrap.observed_best_source,
            "observed_runner_up_source": self.bootstrap.observed_runner_up_source,
            "observed_margin": self.bootstrap.observed_margin,
            "unique_winner_probability": self.bootstrap.unique_winner_probability,
            "margin_lower_2_5": self.bootstrap.margin_lower_2_5,
            "target_labels_used": False,
            "target_support_used": False,
            "seed_selection_performed": False,
        }


@dataclass(frozen=True)
class PolicyAssignment:
    assignment_id: str
    selection_id: str
    target_center: str
    training_seed: int
    generation_seed: int
    replicate_id: str
    action: str
    source_center: str
    source_stream_id: str
    canonical_candidate_ordinal: int
    selected_source_ordinal: int
    source_budget_per_class: int
    total_per_class: int
    class_shuffle_seed_0: int
    class_shuffle_seed_1: int
    equal_union_assignment_id: str
    exact_equal_union_fallback: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_uniform_b_v2_utility_regret_assignment_v1",
            "assignment_id": self.assignment_id,
            "selection_id": self.selection_id,
            "target_center": self.target_center,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "replicate_id": self.replicate_id,
            "action": self.action,
            "source_center": self.source_center,
            "source_stream_id": self.source_stream_id,
            "canonical_candidate_ordinal": self.canonical_candidate_ordinal,
            "selected_source_ordinal": self.selected_source_ordinal,
            "source_budget_per_class": self.source_budget_per_class,
            "source_prefix_start_per_class": 0,
            "source_prefix_stop_per_class": self.source_budget_per_class,
            "total_per_class": self.total_per_class,
            "class_shuffle_seed_0": self.class_shuffle_seed_0,
            "class_shuffle_seed_1": self.class_shuffle_seed_1,
            "equal_union_assignment_id": self.equal_union_assignment_id,
            "exact_equal_union_fallback": self.exact_equal_union_fallback,
            "target_expert_excluded": True,
            "seed_selection_performed": False,
        }


@dataclass(frozen=True)
class UtilityRegretPolicyLock:
    _payload: Mapping[str, object]

    def __post_init__(self) -> None:
        payload = dict(self._payload)
        observed = payload.get("policy_lock_hash")
        unhashed = {key: value for key, value in payload.items() if key != "policy_lock_hash"}
        if observed != stable_hash(unhashed):
            raise ProtocolError("Utility/regret policy lock hash drifted.")

    @property
    def policy_lock_hash(self) -> str:
        return str(self._payload["policy_lock_hash"])

    def to_payload(self) -> dict[str, object]:
        return deepcopy(dict(self._payload))


__all__ = (
    "BOOTSTRAP_MAX_ATTEMPTS",
    "BOOTSTRAP_SEED",
    "BOOTSTRAP_VALID_REPLICATES",
    "BootstrapResult",
    "CENTERS",
    "CLAIM_SCOPE",
    "CONSUMPTION_RULE_HASH",
    "CandidateSummary",
    "EQUAL_UNION_ARTIFACT_ID",
    "EQUAL_UNION_SOURCE_BUDGET_PER_CLASS",
    "EXPECTED_CANDIDATE_SUMMARIES",
    "EXPECTED_CONFIG_CONTRACT_HASH",
    "EXPECTED_REGRET_CELLS",
    "EXPECTED_SELECTIONS",
    "EXPECTED_UTILITY_ROWS",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "EXPERT_BANK_ARTIFACT_ID",
    "GENERATION_LOCK_ARTIFACT_ID",
    "GENERATION_SEEDS",
    "MARGIN_LOWER_QUANTILE",
    "OUTPUT_ARTIFACT_ID",
    "OUTPUT_SEMANTIC_IDENTITIES",
    "POLICY_FAMILY",
    "POLICY_NAMESPACE",
    "PolicyAssignment",
    "PolicySelection",
    "RegretCell",
    "SEED_PAIR_COUNT",
    "SINGLE_SOURCE_BUDGET_PER_CLASS",
    "TRAINING_SEEDS",
    "UTILITY_ARTIFACT_ID",
    "UtilityRegretPolicyLock",
    "WIN_PROBABILITY_THRESHOLD",
    "consumption_rule_payload",
)
