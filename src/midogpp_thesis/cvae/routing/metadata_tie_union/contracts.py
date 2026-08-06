"""Scientific identities for the frozen metadata exact-match tie-union policy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    OUTPUT_ARTIFACT_ID as EXPERT_BANK_ARTIFACT_ID,
    TRAINING_SEEDS,
)
from ...generation.contracts import (
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    OUTPUT_ARTIFACT_ID as GENERATION_LOCK_ARTIFACT_ID,
    REPLICATE_POLICY,
    TOTAL_PER_CLASS,
)
from ...protocol import ProtocolError
from ..contracts import (
    EXPECTED_ASSIGNMENT_TABLE_HASH as EXPECTED_EQUAL_UNION_ASSIGNMENT_TABLE_HASH,
    EXPECTED_GENERATION_CONTENT_HASH,
    EXPECTED_POLICY_LOCK_HASH as EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,
    EXPECTED_POLICY_PLAN_HASH as EXPECTED_EQUAL_UNION_POLICY_PLAN_HASH,
    EXPECTED_REPLICATE_PLAN_HASH,
    EXPECTED_SOURCE_PLAN_HASH,
    OUTPUT_ARTIFACT_ID as EQUAL_UNION_POLICY_ARTIFACT_ID,
)


EXPERIMENT_ID = (
    "midogpp.routing_and_composition."
    "uniform_b_v2_metadata_tie_union_policy_lock.v1"
)
EXPERIMENT_NAME = "uniform_b_v2_metadata_tie_union_policy_lock_v1"
OUTPUT_ARTIFACT_ID = "midogpp_output_uniform_b_v2_metadata_tie_union_policy_lock_v1"
COMPATIBILITY_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_metadata_exact_match_compatibility_v1"
)
CLAIM_SCOPE = "routing_and_composition"

POLICY_FAMILY = "metadata_exact_match_all_max_ties_equal_union_fixed_count"
POLICY_NAMESPACE = "uniform_b_v2_metadata_tie_union_policy_lock_v1"
POLICY_DECISION = "FROZEN_AS_METADATA_EXACT_MATCH_TIE_UNION_COMPARISON_POLICY"
PUBLICATION_STATE = "POLICY_FROZEN_FOR_MATCHED_STAGE70_EVALUATION"

EXPECTED_SELECTION_COUNT = 9
EXPECTED_REPLICATE_COUNT = 81
EXPECTED_ASSIGNMENT_COUNT = 153
STAGE40_MAX_SOURCE_BLOCK_PER_CLASS = 1024

# Compatibility identities are frozen by the independent exact-match artifact.
EXPECTED_COMPATIBILITY_LOCK_HASH = "4b46b3d157b07781"
EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH = "aec9e0b5b09a1fe5"
EXPECTED_CONFIG_CONTRACT_HASH = "df69d7481b0fd62a"
EXPECTED_POLICY_LOCK_HASH = "27f16953b32c46cd"
EXPECTED_POLICY_PLAN_HASH = "ca10b4ed038ccdba"
EXPECTED_SELECTION_TABLE_HASH = "ba611c0180149d79"
EXPECTED_ASSIGNMENT_TABLE_HASH = "b3bc2187806f8788"

SELECTED_SOURCES_BY_TARGET: Mapping[str, tuple[str, ...]] = {
    "0": ("5",),
    "1": ("2",),
    "2": ("1",),
    "3": ("1", "2"),
    "5": ("6", "7"),
    "6": ("5", "7"),
    "7": ("5", "6", "8", "9"),
    "8": ("7", "9"),
    "9": ("7", "8"),
}

SOURCE_BUDGET_BY_TIE_COUNT: Mapping[int, int] = {
    1: 1024,
    2: 512,
    4: 256,
}

OUTPUT_SEMANTIC_IDENTITIES = {
    "policy_lock_contract": (
        "midogpp_uniform_b_v2_metadata_tie_union_policy_lock_v1"
    ),
    "config_contract_hash": EXPECTED_CONFIG_CONTRACT_HASH,
    "policy_lock_hash": EXPECTED_POLICY_LOCK_HASH,
    "policy_plan_hash": EXPECTED_POLICY_PLAN_HASH,
    "selection_table_hash": EXPECTED_SELECTION_TABLE_HASH,
    "assignment_table_hash": EXPECTED_ASSIGNMENT_TABLE_HASH,
    "compatibility_lock_hash": EXPECTED_COMPATIBILITY_LOCK_HASH,
    "compatibility_score_table_hash": EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH,
    "equal_union_policy_lock_hash": EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,
    "generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
    "expert_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
}


@dataclass(frozen=True)
class MetadataTieUnionPolicyLock:
    """Immutable, self-hashing Stage-60 comparison-policy lock."""

    _payload: Mapping[str, object]

    def __post_init__(self) -> None:
        try:
            payload = deepcopy(dict(self._payload))
        except Exception as exc:
            raise ProtocolError("Metadata tie-union policy lock is not copyable.") from exc
        observed = payload.get("policy_lock_hash")
        unhashed = {key: value for key, value in payload.items() if key != "policy_lock_hash"}
        if observed != stable_hash(unhashed):
            raise ProtocolError("Metadata tie-union policy-lock hash drifted.")
        _validate_frozen_policy_lock(payload)
        object.__setattr__(self, "_payload", payload)

    @property
    def policy_lock_hash(self) -> str:
        return str(self._payload["policy_lock_hash"])

    @property
    def compatibility_lock_hash(self) -> str:
        upstreams = self._payload.get("upstreams")
        if not isinstance(upstreams, Mapping):
            raise ProtocolError("Metadata tie-union policy lock lacks upstream identities.")
        return str(upstreams["compatibility_lock_hash"])

    def to_payload(self) -> dict[str, object]:
        return deepcopy(dict(self._payload))


@dataclass(frozen=True)
class PolicySelection:
    """One target-level all-maximum-ties metadata selection."""

    selection_id: str
    target_center: str
    candidate_source_centers: tuple[str, ...]
    candidate_exact_match_scores: tuple[int, ...]
    selected_source_centers: tuple[str, ...]
    maximum_exact_match_score: int
    tie_count: int
    source_budget_per_class: int
    total_per_class: int = TOTAL_PER_CLASS

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_uniform_b_v2_metadata_tie_selection_v1",
            "selection_id": self.selection_id,
            "target_center": self.target_center,
            "candidate_source_centers": list(self.candidate_source_centers),
            "candidate_exact_match_scores": list(self.candidate_exact_match_scores),
            "selected_source_centers": list(self.selected_source_centers),
            "maximum_exact_match_score": self.maximum_exact_match_score,
            "tie_count": self.tie_count,
            "source_budget_per_class": self.source_budget_per_class,
            "total_per_class": self.total_per_class,
            "selection_rule": "retain_all_sources_tied_at_maximum_exact_match_score",
            "canonical_candidate_order_role": "ordering_only_never_tie_break",
            "all_maximum_ties_retained": True,
            "tie_break_applied": False,
            "metadata_proxy_only": True,
        }


@dataclass(frozen=True)
class PolicyAssignment:
    """One selected source prefix in one frozen target/seed replicate."""

    assignment_id: str
    selection_id: str
    replicate_id: str
    target_center: str
    training_seed: int
    generation_seed: int
    source_center: str
    source_stream_id: str
    canonical_candidate_ordinal: int
    selected_source_ordinal: int
    maximum_exact_match_score: int
    tie_count: int
    source_budget_per_class: int

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_uniform_b_v2_metadata_tie_union_assignment_v1",
            "assignment_id": self.assignment_id,
            "selection_id": self.selection_id,
            "replicate_id": self.replicate_id,
            "target_center": self.target_center,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "source_center": self.source_center,
            "source_stream_id": self.source_stream_id,
            "canonical_candidate_ordinal": self.canonical_candidate_ordinal,
            "selected_source_ordinal": self.selected_source_ordinal,
            "maximum_exact_match_score": self.maximum_exact_match_score,
            "tie_count": self.tie_count,
            "source_budget_per_class": self.source_budget_per_class,
            "source_prefix_start_per_class": 0,
            "source_prefix_stop_per_class": self.source_budget_per_class,
            "selection_rank": None,
            "tie_break_applied": False,
            "learned_weight": None,
            "target_expert": False,
        }


@dataclass(frozen=True)
class PolicyReplicate:
    """One metadata-tie union paired to the frozen equal-union replicate."""

    replicate_id: str
    selection_id: str
    target_center: str
    training_seed: int
    generation_seed: int
    candidate_source_centers: tuple[str, ...]
    selected_source_centers: tuple[str, ...]
    selected_source_stream_ids: tuple[str, ...]
    assignment_ids: tuple[str, ...]
    class_shuffle_seed_by_label: Mapping[str, int]
    maximum_exact_match_score: int
    tie_count: int
    source_budget_per_class: int
    total_per_class: int = TOTAL_PER_CLASS

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_uniform_b_v2_metadata_tie_union_replicate_v1",
            "replicate_id": self.replicate_id,
            "equal_union_replicate_id": self.replicate_id,
            "selection_id": self.selection_id,
            "target_center": self.target_center,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "candidate_source_centers": list(self.candidate_source_centers),
            "selected_source_centers": list(self.selected_source_centers),
            "selected_source_stream_ids": list(self.selected_source_stream_ids),
            "assignment_ids": list(self.assignment_ids),
            "class_shuffle_seed_by_label": dict(self.class_shuffle_seed_by_label),
            "maximum_exact_match_score": self.maximum_exact_match_score,
            "tie_count": self.tie_count,
            "source_budget_per_class": self.source_budget_per_class,
            "total_per_class": self.total_per_class,
            "policy_family": POLICY_FAMILY,
            "selection_rule": "retain_all_sources_tied_at_maximum_exact_match_score",
            "canonical_candidate_order_role": "ordering_only_never_tie_break",
            "all_maximum_ties_retained": True,
            "target_expert_excluded": True,
            "source_counts_equal": True,
            "seed_selection_performed": False,
            "tie_break_applied": False,
            "source_weighting_learned": False,
        }


# Short alias used by consumers that already know the package identifies the policy.
TieUnionPolicyLock = MetadataTieUnionPolicyLock


def _validate_frozen_policy_lock(payload: Mapping[str, object]) -> None:
    expected_fields = {
        "schema_version",
        "experiment_id",
        "claim_scope",
        "config_contract_hash",
        "upstreams",
        "policy",
        "composition_execution",
        "future_evaluation_contract",
        "selection_table_hash",
        "policy_plan_hash",
        "assignment_table_hash",
        "firewalls",
        "policy_lock_hash",
    }
    if set(payload) != expected_fields:
        raise ProtocolError("Metadata tie-union policy-lock schema drifted.")
    expected_values = {
        "schema_version": "midogpp_uniform_b_v2_metadata_tie_union_policy_lock_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config_contract_hash": EXPECTED_CONFIG_CONTRACT_HASH,
        "selection_table_hash": EXPECTED_SELECTION_TABLE_HASH,
        "policy_plan_hash": EXPECTED_POLICY_PLAN_HASH,
        "assignment_table_hash": EXPECTED_ASSIGNMENT_TABLE_HASH,
        "policy_lock_hash": EXPECTED_POLICY_LOCK_HASH,
    }
    if any(payload.get(key) != value for key, value in expected_values.items()):
        raise ProtocolError("Metadata tie-union policy-lock semantic identity drifted.")
    upstreams = payload.get("upstreams")
    expected_upstreams = {
        "bank_artifact_id": EXPERT_BANK_ARTIFACT_ID,
        "bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
        "generation_lock_artifact_id": GENERATION_LOCK_ARTIFACT_ID,
        "generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
        "generation_content_hash": EXPECTED_GENERATION_CONTENT_HASH,
        "source_plan_hash": EXPECTED_SOURCE_PLAN_HASH,
        "replicate_plan_hash": EXPECTED_REPLICATE_PLAN_HASH,
        "equal_union_policy_artifact_id": EQUAL_UNION_POLICY_ARTIFACT_ID,
        "equal_union_policy_lock_hash": EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,
        "equal_union_policy_plan_hash": EXPECTED_EQUAL_UNION_POLICY_PLAN_HASH,
        "equal_union_assignment_table_hash": (
            EXPECTED_EQUAL_UNION_ASSIGNMENT_TABLE_HASH
        ),
        "metadata_compatibility_artifact_id": COMPATIBILITY_ARTIFACT_ID,
        "compatibility_lock_hash": EXPECTED_COMPATIBILITY_LOCK_HASH,
        "compatibility_score_table_hash": EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH,
    }
    if not isinstance(upstreams, Mapping) or dict(upstreams) != expected_upstreams:
        raise ProtocolError("Metadata tie-union upstream semantic identities drifted.")
    policy = payload.get("policy")
    composition = payload.get("composition_execution")
    future = payload.get("future_evaluation_contract")
    firewalls = payload.get("firewalls")
    if (
        not isinstance(policy, Mapping)
        or policy.get("family") != POLICY_FAMILY
        or policy.get("namespace") != POLICY_NAMESPACE
        or policy.get("centers") != list(CENTERS)
        or policy.get("selected_sources_by_target")
        != {
            target: list(SELECTED_SOURCES_BY_TARGET[target])
            for target in CENTERS
        }
        or policy.get("all_maximum_ties_retained") is not True
        or policy.get("tie_break_forbidden") is not True
        or policy.get("no_seed_selection") is not True
        or not isinstance(composition, Mapping)
        or composition.get("shuffle_seed_reused_exactly") is not True
        or not isinstance(future, Mapping)
        or future.get("evaluation_occurs_in_stage60") is not False
        or not isinstance(firewalls, Mapping)
        or firewalls.get("target_samples_used") is not False
        or firewalls.get("target_support_used") is not False
        or firewalls.get("target_labels_used") is not False
        or firewalls.get("routing_quality_claimed") is not False
        or firewalls.get("downstream_utility_computed") is not False
    ):
        raise ProtocolError("Metadata tie-union frozen policy semantics drifted.")


__all__ = (
    "CENTERS",
    "CLAIM_SCOPE",
    "COMPATIBILITY_ARTIFACT_ID",
    "EQUAL_UNION_POLICY_ARTIFACT_ID",
    "EXPECTED_ASSIGNMENT_COUNT",
    "EXPECTED_ASSIGNMENT_TABLE_HASH",
    "EXPECTED_BANK_LOCK_HASH",
    "EXPECTED_COMPATIBILITY_LOCK_HASH",
    "EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH",
    "EXPECTED_CONFIG_CONTRACT_HASH",
    "EXPECTED_EQUAL_UNION_ASSIGNMENT_TABLE_HASH",
    "EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH",
    "EXPECTED_EQUAL_UNION_POLICY_PLAN_HASH",
    "EXPECTED_GENERATION_CONTENT_HASH",
    "EXPECTED_GENERATION_LOCK_HASH",
    "EXPECTED_POLICY_LOCK_HASH",
    "EXPECTED_POLICY_PLAN_HASH",
    "EXPECTED_REPLICATE_COUNT",
    "EXPECTED_REPLICATE_PLAN_HASH",
    "EXPECTED_SELECTION_COUNT",
    "EXPECTED_SELECTION_TABLE_HASH",
    "EXPECTED_SOURCE_PLAN_HASH",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "EXPERT_BANK_ARTIFACT_ID",
    "GENERATION_LOCK_ARTIFACT_ID",
    "GENERATION_SEEDS",
    "MetadataTieUnionPolicyLock",
    "OUTPUT_ARTIFACT_ID",
    "OUTPUT_SEMANTIC_IDENTITIES",
    "POLICY_DECISION",
    "POLICY_FAMILY",
    "POLICY_NAMESPACE",
    "PUBLICATION_STATE",
    "PolicyAssignment",
    "PolicyReplicate",
    "PolicySelection",
    "REPLICATE_POLICY",
    "SELECTED_SOURCES_BY_TARGET",
    "SOURCE_BUDGET_BY_TIE_COUNT",
    "STAGE40_MAX_SOURCE_BLOCK_PER_CLASS",
    "TOTAL_PER_CLASS",
    "TRAINING_SEEDS",
    "TieUnionPolicyLock",
)
