"""Scientific identities for the frozen Uniform-B v2 equal-union policy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping

from ...common.hashing import stable_hash
from ..expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    OUTPUT_ARTIFACT_ID as EXPERT_BANK_ARTIFACT_ID,
    TRAINING_SEEDS,
)
from ..generation.contracts import (
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    OUTPUT_ARTIFACT_ID as GENERATION_LOCK_ARTIFACT_ID,
    REPLICATE_POLICY,
    SOURCE_BUDGET_PER_CLASS,
    TOTAL_PER_CLASS,
)
from ..protocol import ProtocolError


EXPERIMENT_ID = "midogpp.routing_and_composition.uniform_b_v2_equal_union_policy_lock.v1"
EXPERIMENT_NAME = "uniform_b_v2_equal_union_policy_lock_v1"
OUTPUT_ARTIFACT_ID = "midogpp_output_uniform_b_v2_equal_union_policy_lock_v1"
CLAIM_SCOPE = "routing_and_composition"

POLICY_FAMILY = "target_excluded_equal_union_fixed_count"
POLICY_NAMESPACE = "uniform_b_v2_equal_union_policy_lock_v1"
POLICY_DECISION = "FROZEN_AS_CANONICAL_EQUAL_UNION_ROUTING_CONTROL"
PUBLICATION_STATE = "POLICY_FROZEN_FOR_STAGE70_EVALUATION"
EXPECTED_REPLICATE_COUNT = 81
EXPECTED_ASSIGNMENT_COUNT = 648
SOURCES_PER_TARGET = 8
EXPECTED_CONFIG_CONTRACT_HASH = "e581a1bf98762031"
EXPECTED_POLICY_LOCK_HASH = "4b9ea514308b084f"
EXPECTED_POLICY_PLAN_HASH = "9ec24122d7d0cdf1"
EXPECTED_ASSIGNMENT_TABLE_HASH = "c85415c1b953c04e"
EXPECTED_GENERATION_CONTENT_HASH = "0d1ed12ee31c6427"
EXPECTED_SOURCE_PLAN_HASH = "4571805ace4777a4"
EXPECTED_REPLICATE_PLAN_HASH = "2040f9bb67a6f49d"


@dataclass(frozen=True)
class EqualUnionPolicyLock:
    """Immutable, self-hashing policy-lock payload."""

    _payload: Mapping[str, object]

    def __post_init__(self) -> None:
        payload = dict(self._payload)
        observed = payload.get("policy_lock_hash")
        unhashed = {key: value for key, value in payload.items() if key != "policy_lock_hash"}
        if observed != stable_hash(unhashed):
            raise ProtocolError("Uniform-B v2 equal-union policy-lock hash drifted.")

    @property
    def policy_lock_hash(self) -> str:
        return str(self._payload["policy_lock_hash"])

    @property
    def generation_lock_hash(self) -> str:
        upstream = self._payload.get("upstreams")
        if not isinstance(upstream, Mapping):
            raise ProtocolError("Equal-union policy lock lacks upstream identities.")
        return str(upstream["generation_lock_hash"])

    def to_payload(self) -> dict[str, object]:
        return deepcopy(dict(self._payload))


@dataclass(frozen=True)
class PolicyAssignment:
    """One fixed source contribution to one target-excluded replicate."""

    assignment_id: str
    replicate_id: str
    target_center: str
    training_seed: int
    generation_seed: int
    source_center: str
    source_stream_id: str
    source_ordinal: int
    source_budget_per_class: int = SOURCE_BUDGET_PER_CLASS

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_uniform_b_v2_equal_union_assignment_v1",
            "assignment_id": self.assignment_id,
            "replicate_id": self.replicate_id,
            "target_center": self.target_center,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "source_center": self.source_center,
            "source_stream_id": self.source_stream_id,
            "source_ordinal": self.source_ordinal,
            "source_budget_per_class": self.source_budget_per_class,
            "selection_rank": None,
            "selection_score": None,
            "learned_weight": None,
            "target_expert": False,
        }


@dataclass(frozen=True)
class PolicyReplicate:
    """One immutable equal-union composition decision."""

    replicate_id: str
    target_center: str
    training_seed: int
    generation_seed: int
    candidate_source_centers: tuple[str, ...]
    source_stream_ids: tuple[str, ...]
    assignment_ids: tuple[str, ...]
    class_shuffle_seed_by_label: Mapping[str, int]
    source_budget_per_class: int = SOURCE_BUDGET_PER_CLASS
    total_per_class: int = TOTAL_PER_CLASS

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_uniform_b_v2_equal_union_policy_replicate_v1",
            "replicate_id": self.replicate_id,
            "target_center": self.target_center,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "candidate_source_centers": list(self.candidate_source_centers),
            "source_stream_ids": list(self.source_stream_ids),
            "assignment_ids": list(self.assignment_ids),
            "class_shuffle_seed_by_label": dict(self.class_shuffle_seed_by_label),
            "source_budget_per_class": self.source_budget_per_class,
            "total_per_class": self.total_per_class,
            "policy_family": POLICY_FAMILY,
            "all_eligible_sources_retained": True,
            "target_expert_excluded": True,
            "source_counts_equal": True,
            "expert_selection_performed": False,
            "seed_selection_performed": False,
            "source_ranking_performed": False,
            "source_weighting_learned": False,
        }


__all__ = (
    "CENTERS",
    "CLAIM_SCOPE",
    "EXPECTED_ASSIGNMENT_COUNT",
    "EXPECTED_ASSIGNMENT_TABLE_HASH",
    "EXPECTED_BANK_LOCK_HASH",
    "EXPECTED_CONFIG_CONTRACT_HASH",
    "EXPECTED_GENERATION_CONTENT_HASH",
    "EXPECTED_GENERATION_LOCK_HASH",
    "EXPECTED_POLICY_LOCK_HASH",
    "EXPECTED_POLICY_PLAN_HASH",
    "EXPECTED_REPLICATE_PLAN_HASH",
    "EXPECTED_REPLICATE_COUNT",
    "EXPECTED_SOURCE_PLAN_HASH",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "EXPERT_BANK_ARTIFACT_ID",
    "EqualUnionPolicyLock",
    "GENERATION_LOCK_ARTIFACT_ID",
    "GENERATION_SEEDS",
    "OUTPUT_ARTIFACT_ID",
    "POLICY_DECISION",
    "POLICY_FAMILY",
    "POLICY_NAMESPACE",
    "PUBLICATION_STATE",
    "PolicyAssignment",
    "PolicyReplicate",
    "REPLICATE_POLICY",
    "SOURCE_BUDGET_PER_CLASS",
    "SOURCES_PER_TARGET",
    "TOTAL_PER_CLASS",
    "TRAINING_SEEDS",
)
