"""Frozen identities and structural contracts for the case-OOF diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

from ...protocol import ProtocolError


EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_validation_"
    "residual_topup_b_u_g_s_case_oof.v1"
)
EXPERIMENT_NAME = (
    "uniform_b_v2_consumed_validation_residual_topup_b_u_g_s_case_oof_v1"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_validation_"
    "residual_topup_b_u_g_s_case_oof_v1"
)
STAGE_ID = "90_oracles_and_diagnostics"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "EXPLORATORY_CONSUMED_DATA_ONLY"

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
EQUAL_UNION_POLICY_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_equal_union_policy_lock_v1"
)
VALIDATION_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_residual_topup_b_u_g_s_case_oof_validation_cache_v1"
)
VALIDATION_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_residual_topup_b_u_g_s_case_oof_validation_manifest_v1"
)
INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    EQUAL_UNION_POLICY_ARTIFACT_ID,
    VALIDATION_CACHE_ARTIFACT_ID,
    VALIDATION_MANIFEST_ARTIFACT_ID,
)

CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
TRAINING_SEEDS = (17, 42, 101)
GENERATION_SEEDS = (17, 42, 101)
FIXED_SUPPORT_CASE_COUNT_PER_CENTER = 2
EXPECTED_TOTAL_CASE_COUNT = 44
EXPECTED_CASE_OOF_FOLD_COUNT = 26
EXPECTED_GLOBAL_PROXY_SCORE_COUNT_PER_TARGET = 8 * 2 * 7 * 3
EXPECTED_SUPPORT_PROXY_SCORE_COUNT_PER_TARGET = 2 * 8 * 3
EXPECTED_PROXY_SCORE_COUNT_PER_TARGET = (
    EXPECTED_GLOBAL_PROXY_SCORE_COUNT_PER_TARGET
    + EXPECTED_SUPPORT_PROXY_SCORE_COUNT_PER_TARGET
)
EXPECTED_PROXY_SCORE_COUNT = len(CENTERS) * EXPECTED_PROXY_SCORE_COUNT_PER_TARGET

GLOBAL_QUERY_ROLE = "global_fixed_support"
SUPPORT_QUERY_ROLE = "target_fixed_support"
QUERY_ROLES = (GLOBAL_QUERY_ROLE, SUPPORT_QUERY_ROLE)
PROXY_ENERGY_SEMANTICS = (
    "class_marginalized_common_space_reconstruction_mse_plus_"
    "latent_dim_normalized_analytic_ps_kl"
)

BASE_ACTION_ID = "base_equal_union"
UNIFORM_ACTION_ID = "uniform_residual_topup"
GLOBAL_ACTION_ID = "global_rank_residual_topup"
SUPPORT_ACTION_ID = "support_rank_residual_topup"
PERMUTATION_ACTION_ID = "support_rank_permutation_control"
SINGLE_SOURCE_TAIL_PREFIX = "single_source_tail::"
PRIMARY_ACTION_IDS = (
    BASE_ACTION_ID,
    UNIFORM_ACTION_ID,
    GLOBAL_ACTION_ID,
    SUPPORT_ACTION_ID,
)
MAIN_AND_CONTROL_ACTION_IDS = (*PRIMARY_ACTION_IDS, PERMUTATION_ACTION_ID)
EXPECTED_ACTION_COUNT_PER_TARGET = len(MAIN_AND_CONTROL_ACTION_IDS) + 8
EXPECTED_FROZEN_ACTION_COUNT = len(CENTERS) * EXPECTED_ACTION_COUNT_PER_TARGET
EXPECTED_UNIQUE_CLASSIFIER_FIT_COUNT = (
    len(CENTERS)
    * len(TRAINING_SEEDS)
    * len(GENERATION_SEEDS)
    * EXPECTED_ACTION_COUNT_PER_TARGET
)
EXPECTED_SEALED_PREDICTION_CELL_COUNT = (
    EXPECTED_CASE_OOF_FOLD_COUNT
    * len(TRAINING_SEEDS)
    * len(GENERATION_SEEDS)
    * EXPECTED_ACTION_COUNT_PER_TARGET
)


class ValidationRowLike(Protocol):
    sample_id: str
    case_id: str
    center: str
    partition_role: str


class FixedSupportPartitionLike(Protocol):
    support_rows_by_center: Mapping[str, Sequence[ValidationRowLike]]
    evaluation_rows_by_center: Mapping[str, Sequence[ValidationRowLike]]
    lock_hash: str


@dataclass(frozen=True)
class ProxyScoreRow:
    """One label-free case/expert-replica proxy score."""

    outer_target: str
    query_role: str
    query_center: str
    case_id: str
    candidate_source: str
    training_seed: int
    row_count: int
    proxy_energy: float
    partition_role: str = "support"
    labels_used: bool = False
    evaluation_embeddings_used: bool = False
    source_experts_updated: bool = False
    exact_nelbo_claimed: bool = False
    proxy_energy_semantics: str = PROXY_ENERGY_SEMANTICS

    def __post_init__(self) -> None:
        target = str(self.outer_target)
        role = str(self.query_role)
        query = str(self.query_center)
        case_id = str(self.case_id)
        source = str(self.candidate_source)
        try:
            energy = float(self.proxy_energy)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError("Case-OOF proxy energy must be finite.") from exc
        if (
            target not in CENTERS
            or role not in QUERY_ROLES
            or query not in CENTERS
            or not case_id
            or case_id.strip() != case_id
            or source not in CENTERS
            or source == target
            or self.training_seed not in TRAINING_SEEDS
            or isinstance(self.row_count, bool)
            or not isinstance(self.row_count, int)
            or self.row_count <= 0
            or not math.isfinite(energy)
            or self.partition_role != "support"
            or type(self.labels_used) is not bool
            or self.labels_used
            or type(self.evaluation_embeddings_used) is not bool
            or self.evaluation_embeddings_used
            or type(self.source_experts_updated) is not bool
            or self.source_experts_updated
            or type(self.exact_nelbo_claimed) is not bool
            or self.exact_nelbo_claimed
            or self.proxy_energy_semantics != PROXY_ENERGY_SEMANTICS
        ):
            raise ProtocolError("Case-OOF proxy score identity drifted.")
        if role == GLOBAL_QUERY_ROLE and (query == target or source == query):
            raise ProtocolError("Case-OOF G score failed H/q exclusion.")
        if role == SUPPORT_QUERY_ROLE and query != target:
            raise ProtocolError("Case-OOF S score must use fixed support from H.")
        object.__setattr__(self, "outer_target", target)
        object.__setattr__(self, "query_role", role)
        object.__setattr__(self, "query_center", query)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "candidate_source", source)
        object.__setattr__(self, "proxy_energy", energy)


@dataclass(frozen=True)
class CaseProxyBallot:
    outer_target: str
    query_role: str
    query_center: str
    case_id: str
    candidate_sources: tuple[str, ...]
    mean_proxy_energy_by_source: Mapping[str, float]
    normalized_midrank_by_source: Mapping[str, float]
    ballot_hash: str

    def __post_init__(self) -> None:
        sources = tuple(self.candidate_sources)
        energies = {str(key): float(value) for key, value in self.mean_proxy_energy_by_source.items()}
        ranks = {str(key): float(value) for key, value in self.normalized_midrank_by_source.items()}
        if (
            not sources
            or sources != tuple(sorted(sources))
            or set(energies) != set(sources)
            or set(ranks) != set(sources)
            or not all(math.isfinite(value) for value in energies.values())
            or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in ranks.values())
            or not _is_hash(self.ballot_hash)
        ):
            raise ProtocolError("Case-OOF proxy ballot is malformed.")
        object.__setattr__(self, "candidate_sources", sources)
        object.__setattr__(self, "mean_proxy_energy_by_source", MappingProxyType(energies))
        object.__setattr__(self, "normalized_midrank_by_source", MappingProxyType(ranks))


@dataclass(frozen=True)
class ProxyRankSummary:
    outer_target: str
    query_role: str
    candidate_sources: tuple[str, ...]
    mean_normalized_midrank_by_source: Mapping[str, float]
    priority_by_source: Mapping[str, float]
    ballot_count_by_source: Mapping[str, int]
    ballots: tuple[CaseProxyBallot, ...]
    rank_hash: str

    def __post_init__(self) -> None:
        sources = tuple(self.candidate_sources)
        midranks = {str(key): float(value) for key, value in self.mean_normalized_midrank_by_source.items()}
        priorities = {str(key): float(value) for key, value in self.priority_by_source.items()}
        counts = {str(key): int(value) for key, value in self.ballot_count_by_source.items()}
        if (
            self.outer_target not in CENTERS
            or self.query_role not in QUERY_ROLES
            or sources != candidate_sources(self.outer_target)
            or set(midranks) != set(sources)
            or set(priorities) != set(sources)
            or set(counts) != set(sources)
            or not all(0.0 <= value <= 1.0 for value in midranks.values())
            or any(abs(priorities[source] - (1.0 - midranks[source])) > 1e-15 for source in sources)
            or len(set(counts.values())) != 1
            or min(counts.values(), default=0) <= 0
            or not self.ballots
            or not _is_hash(self.rank_hash)
        ):
            raise ProtocolError("Case-OOF proxy-rank summary is malformed.")
        object.__setattr__(self, "candidate_sources", sources)
        object.__setattr__(self, "mean_normalized_midrank_by_source", MappingProxyType(midranks))
        object.__setattr__(self, "priority_by_source", MappingProxyType(priorities))
        object.__setattr__(self, "ballot_count_by_source", MappingProxyType(counts))
        object.__setattr__(self, "ballots", tuple(self.ballots))


@dataclass(frozen=True)
class TargetRankSurface:
    outer_target: str
    global_summary: ProxyRankSummary
    support_summary: ProxyRankSummary

    def __post_init__(self) -> None:
        if (
            self.outer_target not in CENTERS
            or self.global_summary.outer_target != self.outer_target
            or self.support_summary.outer_target != self.outer_target
            or self.global_summary.query_role != GLOBAL_QUERY_ROLE
            or self.support_summary.query_role != SUPPORT_QUERY_ROLE
        ):
            raise ProtocolError("Case-OOF target rank surface drifted.")


def candidate_sources(target_center: object) -> tuple[str, ...]:
    target = str(target_center)
    if target not in CENTERS:
        raise ProtocolError("Case-OOF target center is unknown.")
    return tuple(center for center in CENTERS if center != target)


def global_candidate_sources(
    outer_target: object,
    query_center: object,
) -> tuple[str, ...]:
    target = str(outer_target)
    query = str(query_center)
    if target not in CENTERS or query not in CENTERS or query == target:
        raise ProtocolError("Case-OOF global H/q geometry is invalid.")
    return tuple(center for center in CENTERS if center not in {target, query})


def tail_action_id(source_center: object) -> str:
    source = str(source_center)
    if source not in CENTERS:
        raise ProtocolError("Case-OOF tail source is unknown.")
    return f"{SINGLE_SOURCE_TAIL_PREFIX}{source}"


def tail_source(action_id: object) -> str | None:
    value = str(action_id)
    if not value.startswith(SINGLE_SOURCE_TAIL_PREFIX):
        return None
    source = value.removeprefix(SINGLE_SOURCE_TAIL_PREFIX)
    if source not in CENTERS:
        raise ProtocolError("Case-OOF tail action source is unknown.")
    return source


def expected_action_ids(target_center: object) -> tuple[str, ...]:
    return (
        *MAIN_AND_CONTROL_ACTION_IDS,
        *(tail_action_id(source) for source in candidate_sources(target_center)),
    )


def _is_hash(value: object) -> bool:
    text = str(value)
    return len(text) in {16, 64} and all(character in "0123456789abcdef" for character in text)


__all__ = tuple(name for name in globals() if name.isupper()) + (
    "CaseProxyBallot",
    "FixedSupportPartitionLike",
    "ProxyRankSummary",
    "ProxyScoreRow",
    "TargetRankSurface",
    "ValidationRowLike",
    "candidate_sources",
    "expected_action_ids",
    "global_candidate_sources",
    "tail_action_id",
    "tail_source",
)
