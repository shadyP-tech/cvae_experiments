"""Stable facade for freezing the case-OOF action plan.

Action contracts, target-library construction, and reconstructive plan
validation are intentionally kept in cohesive sibling modules.  This facade
preserves the public planning API used by execution and artifact writers.
"""

from __future__ import annotations

from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .action_contracts import (
    BASE_ACTION_KIND,
    BASE_PER_SOURCE_PER_CLASS,
    BASE_TOTAL_PER_CLASS,
    GLOBAL_POLICY_ACTION_KIND,
    MATCHED_TOTAL_PER_CLASS,
    PERMUTATION_POLICY_ACTION_KIND,
    SINGLE_SOURCE_POLICY_ACTION_KIND,
    SUPPORT_POLICY_ACTION_KIND,
    TOPUP_TOTAL_PER_CLASS,
    UNIFORM_POLICY_ACTION_KIND,
    FrozenCaseOOFAction,
    canonical_source_identity_permutation,
)
from .action_library import build_case_oof_action_library
from .contracts import (
    EXPECTED_ACTION_COUNT_PER_TARGET,
    EXPECTED_CASE_OOF_FOLD_COUNT,
    EXPECTED_FROZEN_ACTION_COUNT,
    EXPERIMENT_ID,
    TargetRankSurface,
)
from .partitions import CaseOOFSurface
from .plan_contracts import CaseOOFPlanSurface


def build_case_oof_plan(
    rank_surface: Mapping[str, TargetRankSurface],
    crossfit: CaseOOFSurface,
    *,
    config_contract_hash: str,
) -> CaseOOFPlanSurface:
    """Freeze all 117 actions and assign them unchanged to 26 folds."""

    if not isinstance(crossfit, CaseOOFSurface) or not _is_hash(
        config_contract_hash
    ):
        raise ProtocolError("Case-OOF plan inputs are invalid.")
    library = build_case_oof_action_library(rank_surface)
    actions_by_fold = {
        fold.fold_id: library.actions_by_target[fold.target_center]
        for fold in crossfit.folds
    }
    fold_target_by_id = {
        fold.fold_id: fold.target_center for fold in crossfit.folds
    }
    lock_unhashed: dict[str, object] = {
        "schema_version": "midogpp_residual_topup_case_oof_router_plan_lock_v1",
        "experiment_id": EXPERIMENT_ID,
        "config_contract_hash": config_contract_hash,
        "case_oof_surface_lock_hash": crossfit.lock_hash,
        "action_library_hash": library.action_library_hash,
        "frozen_action_count": EXPECTED_FROZEN_ACTION_COUNT,
        "action_count_per_target": EXPECTED_ACTION_COUNT_PER_TARGET,
        "case_oof_fold_count": EXPECTED_CASE_OOF_FOLD_COUNT,
        "rank_hashes_by_target": {
            target: dict(library.rank_hashes_by_target[target])
            for target in library.actions_by_target
        },
        "fold_target_by_id": fold_target_by_id,
        "action_hashes_by_target": {
            target: [
                action.action_hash
                for action in library.actions_by_target[target]
            ]
            for target in library.actions_by_target
        },
        "action_hashes_by_fold": {
            fold_id: [action.action_hash for action in actions]
            for fold_id, actions in actions_by_fold.items()
        },
        "support_rank_fixed_across_target_folds": True,
        "all_actions_frozen_before_evaluation_label_access": True,
        "other_evaluation_embeddings_used_for_route": False,
        "evaluation_labels_used_for_route": False,
        "oracle_or_downstream_outcomes_used": False,
    }
    lock_payload = {
        **lock_unhashed,
        "router_plan_lock_hash": stable_hash(lock_unhashed),
    }
    return CaseOOFPlanSurface(
        action_library=library,
        actions_by_fold=actions_by_fold,
        fold_target_by_id=fold_target_by_id,
        lock_payload=lock_payload,
    )


def _is_hash(value: object) -> bool:
    text = str(value)
    return len(text) in {16, 64} and all(
        character in "0123456789abcdef" for character in text
    )


__all__ = (
    "BASE_ACTION_KIND",
    "BASE_PER_SOURCE_PER_CLASS",
    "BASE_TOTAL_PER_CLASS",
    "CaseOOFPlanSurface",
    "FrozenCaseOOFAction",
    "GLOBAL_POLICY_ACTION_KIND",
    "MATCHED_TOTAL_PER_CLASS",
    "PERMUTATION_POLICY_ACTION_KIND",
    "SINGLE_SOURCE_POLICY_ACTION_KIND",
    "SUPPORT_POLICY_ACTION_KIND",
    "TOPUP_TOTAL_PER_CLASS",
    "UNIFORM_POLICY_ACTION_KIND",
    "build_case_oof_plan",
    "canonical_source_identity_permutation",
)
