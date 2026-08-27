"""SCEPTRE: source-family evidence ranking with exact-B tie fallback."""

from .candidate_menu import (
    bind_target_excluded_source_families,
    build_candidate_menu,
    build_candidate_menu_from_keys,
    build_source_family,
    build_target_excluded_source_families,
)
from .contracts import (
    CandidateMenu,
    ExactBFallback,
    FamilyProxyScore,
    RankedWinnerSet,
    RawRoute,
    RouteDecision,
    SourceFamily,
    INVALID_EVIDENCE_REASON,
    NONFINITE_EVIDENCE_REASON,
    UNSUPPORTED_EVIDENCE_REASON,
)
from .control import (
    ControlValidationReceipt,
    validate_candidate_and_b_control,
    validate_control_plan,
)
from .proxy_score import (
    aggregate_family_proxy_score,
    aggregate_menu_proxy_scores,
    aggregate_training_replica_scores,
    average_training_replicas_before_source_ranking,
)
from .ranking import (
    normalized_true_midranks,
    rank_family_proxy_scores,
    route_raw_proxy_evidence_or_exact_b,
    route_unique_winner_or_exact_b,
    select_raw_route,
    select_raw_evidence_or_exact_b,
)
from .validation import (
    SemanticReplayReceipt,
    assert_import_source_fence,
    replay_semantic_contract,
)


__all__ = (
    "CandidateMenu",
    "ControlValidationReceipt",
    "ExactBFallback",
    "FamilyProxyScore",
    "RankedWinnerSet",
    "RawRoute",
    "RouteDecision",
    "SemanticReplayReceipt",
    "SourceFamily",
    "INVALID_EVIDENCE_REASON",
    "NONFINITE_EVIDENCE_REASON",
    "UNSUPPORTED_EVIDENCE_REASON",
    "aggregate_family_proxy_score",
    "aggregate_menu_proxy_scores",
    "aggregate_training_replica_scores",
    "assert_import_source_fence",
    "average_training_replicas_before_source_ranking",
    "bind_target_excluded_source_families",
    "build_candidate_menu",
    "build_candidate_menu_from_keys",
    "build_source_family",
    "build_target_excluded_source_families",
    "normalized_true_midranks",
    "rank_family_proxy_scores",
    "replay_semantic_contract",
    "route_unique_winner_or_exact_b",
    "route_raw_proxy_evidence_or_exact_b",
    "select_raw_evidence_or_exact_b",
    "select_raw_route",
    "validate_candidate_and_b_control",
    "validate_control_plan",
)
