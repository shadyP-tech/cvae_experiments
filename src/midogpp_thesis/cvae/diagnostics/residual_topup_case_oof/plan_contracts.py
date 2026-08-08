"""Reconstructive validation and lookup surface for a case-OOF plan."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .action_contracts import FrozenCaseOOFAction
from .action_library import CaseOOFActionLibrary
from .contracts import (
    CENTERS,
    EXPECTED_ACTION_COUNT_PER_TARGET,
    EXPECTED_CASE_OOF_FOLD_COUNT,
    EXPECTED_FROZEN_ACTION_COUNT,
    EXPERIMENT_ID,
)


@dataclass(frozen=True)
class CaseOOFPlanSurface:
    """Closed target action library plus fold-to-target assignments."""

    action_library: CaseOOFActionLibrary
    actions_by_fold: Mapping[str, tuple[FrozenCaseOOFAction, ...]]
    fold_target_by_id: Mapping[str, str]
    lock_payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.action_library, CaseOOFActionLibrary):
            raise ProtocolError("Case-OOF action library is invalid.")
        by_fold = {
            str(fold_id): tuple(actions)
            for fold_id, actions in self.actions_by_fold.items()
        }
        fold_targets = {
            str(fold_id): str(target)
            for fold_id, target in self.fold_target_by_id.items()
        }
        lock = dict(self.lock_payload)
        if (
            len(by_fold) != EXPECTED_CASE_OOF_FOLD_COUNT
            or tuple(by_fold) != tuple(fold_targets)
        ):
            raise ProtocolError("Case-OOF plan fold cardinality drifted.")
        for fold_id, actions in by_fold.items():
            target = fold_targets[fold_id]
            if (
                target not in CENTERS
                or actions != self.action_library.actions_by_target[target]
            ):
                raise ProtocolError("Case-OOF fold action assignment drifted.")
        action_hashes_by_target = {
            target: [
                action.action_hash
                for action in self.action_library.actions_by_target[target]
            ]
            for target in CENTERS
        }
        action_hashes_by_fold = {
            fold_id: [action.action_hash for action in actions]
            for fold_id, actions in by_fold.items()
        }
        if (
            lock.get("schema_version")
            != "midogpp_residual_topup_case_oof_router_plan_lock_v1"
            or lock.get("experiment_id") != EXPERIMENT_ID
            or not _is_hash(lock.get("config_contract_hash"))
            or not _is_hash(lock.get("case_oof_surface_lock_hash"))
            or lock.get("action_library_hash")
            != self.action_library.action_library_hash
            or lock.get("frozen_action_count")
            != EXPECTED_FROZEN_ACTION_COUNT
            or lock.get("action_count_per_target")
            != EXPECTED_ACTION_COUNT_PER_TARGET
            or lock.get("case_oof_fold_count")
            != EXPECTED_CASE_OOF_FOLD_COUNT
            or lock.get("rank_hashes_by_target")
            != {
                target: dict(
                    self.action_library.rank_hashes_by_target[target]
                )
                for target in CENTERS
            }
            or lock.get("fold_target_by_id") != fold_targets
            or lock.get("action_hashes_by_target")
            != action_hashes_by_target
            or lock.get("action_hashes_by_fold") != action_hashes_by_fold
            or lock.get("support_rank_fixed_across_target_folds") is not True
            or lock.get("all_actions_frozen_before_evaluation_label_access")
            is not True
            or lock.get("other_evaluation_embeddings_used_for_route")
            is not False
            or lock.get("evaluation_labels_used_for_route") is not False
            or lock.get("oracle_or_downstream_outcomes_used") is not False
            or lock.get("router_plan_lock_hash")
            != stable_hash(
                {
                    key: value
                    for key, value in lock.items()
                    if key != "router_plan_lock_hash"
                }
            )
        ):
            raise ProtocolError("Case-OOF plan lock is invalid.")
        object.__setattr__(self, "actions_by_fold", MappingProxyType(by_fold))
        object.__setattr__(
            self,
            "fold_target_by_id",
            MappingProxyType(fold_targets),
        )
        object.__setattr__(self, "lock_payload", MappingProxyType(lock))

    @property
    def actions_by_target(
        self,
    ) -> Mapping[str, tuple[FrozenCaseOOFAction, ...]]:
        return self.action_library.actions_by_target

    @property
    def action_library_payload(self) -> Mapping[str, object]:
        return self.action_library.payload

    @property
    def action_count(self) -> int:
        return self.action_library.action_count

    @property
    def action_library_hash(self) -> str:
        return self.action_library.action_library_hash

    @property
    def lock_hash(self) -> str:
        return str(self.lock_payload["router_plan_lock_hash"])

    def action(
        self,
        target_center: object,
        action_id: object,
    ) -> FrozenCaseOOFAction:
        return self.action_library.action(target_center, action_id)

    def action_for(
        self,
        target_center: object,
        action_id: object,
    ) -> FrozenCaseOOFAction:
        """Compatibility spelling for execution-layer consumers."""

        return self.action(target_center, action_id)

    def actions_for_target(
        self,
        target_center: object,
    ) -> tuple[FrozenCaseOOFAction, ...]:
        target = str(target_center)
        if target not in self.actions_by_target:
            raise ProtocolError("Case-OOF action target is unknown.")
        return self.actions_by_target[target]

    def to_action_library_payload(self) -> dict[str, object]:
        return self.action_library.to_payload()

    def to_lock_payload(self) -> dict[str, object]:
        return dict(self.lock_payload)


def _is_hash(value: object) -> bool:
    text = str(value)
    return len(text) in {16, 64} and all(
        character in "0123456789abcdef" for character in text
    )


__all__ = ("CaseOOFPlanSurface",)
