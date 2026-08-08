"""Construction and validation of the frozen target-level action library."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from ...routing.residual_topup import (
    build_borda_directed_topup_action,
    build_single_source_tail_action,
    build_uniform_topup_action,
    target_topup_geometry,
)
from ...routing.residual_topup.hashing import canonical_sha256
from .action_contracts import (
    BASE_ACTION_KIND,
    BASE_ACTION_SEMANTICS,
    BASE_PER_SOURCE_PER_CLASS,
    BASE_TOTAL_PER_CLASS,
    GLOBAL_ACTION_SEMANTICS,
    GLOBAL_POLICY_ACTION_KIND,
    PERMUTATION_ACTION_SEMANTICS,
    PERMUTATION_POLICY_ACTION_KIND,
    SINGLE_SOURCE_ACTION_SEMANTICS,
    SINGLE_SOURCE_POLICY_ACTION_KIND,
    SUPPORT_ACTION_SEMANTICS,
    SUPPORT_POLICY_ACTION_KIND,
    UNIFORM_ACTION_SEMANTICS,
    UNIFORM_POLICY_ACTION_KIND,
    FrozenCaseOOFAction,
    canonical_source_identity_permutation,
    make_frozen_case_oof_action,
)
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    EXPECTED_ACTION_COUNT_PER_TARGET,
    EXPECTED_FROZEN_ACTION_COUNT,
    EXPERIMENT_ID,
    GLOBAL_ACTION_ID,
    PERMUTATION_ACTION_ID,
    SUPPORT_ACTION_ID,
    UNIFORM_ACTION_ID,
    TargetRankSurface,
    candidate_sources,
    expected_action_ids,
    tail_action_id,
)


@dataclass(frozen=True)
class CaseOOFActionLibrary:
    """All 117 predeclared actions and their source-rank identities."""

    actions_by_target: Mapping[str, tuple[FrozenCaseOOFAction, ...]]
    rank_hashes_by_target: Mapping[str, Mapping[str, str]]
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        actions_by_target = {
            str(target): tuple(actions)
            for target, actions in self.actions_by_target.items()
        }
        rank_hashes = {
            str(target): MappingProxyType(
                {str(key): str(value) for key, value in hashes.items()}
            )
            for target, hashes in self.rank_hashes_by_target.items()
        }
        payload = dict(self.payload)
        if tuple(actions_by_target) != CENTERS or tuple(rank_hashes) != CENTERS:
            raise ProtocolError("Case-OOF action-library target order drifted.")
        observed_hashes: set[str] = set()
        for target in CENTERS:
            actions = actions_by_target[target]
            if (
                len(actions) != EXPECTED_ACTION_COUNT_PER_TARGET
                or tuple(action.action_id for action in actions)
                != expected_action_ids(target)
                or any(action.outer_target != target for action in actions)
                or set(rank_hashes[target])
                != {"global_rank_hash", "support_rank_hash"}
                or not all(_is_hash(value) for value in rank_hashes[target].values())
            ):
                raise ProtocolError("Case-OOF target action menu drifted.")
            for action in actions:
                if action.action_hash in observed_hashes:
                    raise ProtocolError(
                        "Case-OOF frozen action hashes must be globally unique."
                    )
                observed_hashes.add(action.action_hash)
        expected_hash = canonical_sha256(
            {
                key: value
                for key, value in payload.items()
                if key != "action_library_hash"
            }
        )
        if (
            len(observed_hashes) != EXPECTED_FROZEN_ACTION_COUNT
            or payload.get("schema_version")
            != "midogpp_residual_topup_case_oof_action_library_v1"
            or payload.get("action_count") != EXPECTED_FROZEN_ACTION_COUNT
            or payload.get("action_library_hash") != expected_hash
        ):
            raise ProtocolError("Case-OOF action library is malformed.")
        object.__setattr__(
            self,
            "actions_by_target",
            MappingProxyType(actions_by_target),
        )
        object.__setattr__(
            self,
            "rank_hashes_by_target",
            MappingProxyType(rank_hashes),
        )
        object.__setattr__(self, "payload", MappingProxyType(payload))

    @property
    def action_count(self) -> int:
        return sum(len(actions) for actions in self.actions_by_target.values())

    @property
    def action_library_hash(self) -> str:
        return str(self.payload["action_library_hash"])

    def action(
        self,
        target_center: object,
        action_id: object,
    ) -> FrozenCaseOOFAction:
        target = str(target_center)
        identifier = str(action_id)
        if target not in self.actions_by_target:
            raise ProtocolError("Case-OOF action lookup target is unknown.")
        for action in self.actions_by_target[target]:
            if action.action_id == identifier:
                return action
        raise ProtocolError("Case-OOF action lookup identifier is unknown.")

    def to_payload(self) -> dict[str, object]:
        return dict(self.payload)


def build_case_oof_action_library(
    rank_surface: Mapping[str, TargetRankSurface],
) -> CaseOOFActionLibrary:
    """Build B/U/G/S/P and every Hxe action from fixed rank summaries."""

    if (
        not isinstance(rank_surface, Mapping)
        or tuple(str(target) for target in rank_surface) != CENTERS
    ):
        raise ProtocolError("Case-OOF rank-surface target order drifted.")
    actions_by_target: dict[str, tuple[FrozenCaseOOFAction, ...]] = {}
    rank_hashes_by_target: dict[str, dict[str, str]] = {}
    for target in CENTERS:
        surface = rank_surface[target]
        if (
            not isinstance(surface, TargetRankSurface)
            or surface.outer_target != target
        ):
            raise ProtocolError("Case-OOF target rank surface drifted.")
        sources = candidate_sources(target)
        geometry = target_topup_geometry(sources)
        permutation = canonical_source_identity_permutation(sources)
        global_ranks = dict(
            surface.global_summary.mean_normalized_midrank_by_source
        )
        support_ranks = dict(
            surface.support_summary.mean_normalized_midrank_by_source
        )
        permuted_ranks = {
            source: support_ranks[permutation[source]] for source in sources
        }
        zero_topup = {source: 0 for source in sources}
        base_final = {
            label: {
                source: BASE_PER_SOURCE_PER_CLASS for source in sources
            }
            for label in (0, 1)
        }
        actions: list[FrozenCaseOOFAction] = [
            make_frozen_case_oof_action(
                target=target,
                action_id=BASE_ACTION_ID,
                policy_id="B",
                action_kind=BASE_ACTION_KIND,
                action_semantics=BASE_ACTION_SEMANTICS,
                sources=sources,
                ranks={},
                permutation={},
                selected_source=None,
                direction={},
                topup=zero_topup,
                final=base_final,
                core=None,
                diagnostic_control=False,
            )
        ]
        actions.extend(
            (
                _wrapped_action(
                    target=target,
                    action_id=UNIFORM_ACTION_ID,
                    policy_id="U",
                    action_kind=UNIFORM_POLICY_ACTION_KIND,
                    action_semantics=UNIFORM_ACTION_SEMANTICS,
                    core=build_uniform_topup_action(geometry),
                ),
                _wrapped_action(
                    target=target,
                    action_id=GLOBAL_ACTION_ID,
                    policy_id="G",
                    action_kind=GLOBAL_POLICY_ACTION_KIND,
                    action_semantics=GLOBAL_ACTION_SEMANTICS,
                    core=build_borda_directed_topup_action(
                        global_ranks,
                        geometry=geometry,
                    ),
                    ranks=global_ranks,
                ),
                _wrapped_action(
                    target=target,
                    action_id=SUPPORT_ACTION_ID,
                    policy_id="S",
                    action_kind=SUPPORT_POLICY_ACTION_KIND,
                    action_semantics=SUPPORT_ACTION_SEMANTICS,
                    core=build_borda_directed_topup_action(
                        support_ranks,
                        geometry=geometry,
                    ),
                    ranks=support_ranks,
                ),
                _wrapped_action(
                    target=target,
                    action_id=PERMUTATION_ACTION_ID,
                    policy_id="P",
                    action_kind=PERMUTATION_POLICY_ACTION_KIND,
                    action_semantics=PERMUTATION_ACTION_SEMANTICS,
                    core=build_borda_directed_topup_action(
                        permuted_ranks,
                        geometry=geometry,
                    ),
                    ranks=permuted_ranks,
                    permutation=permutation,
                    diagnostic_control=True,
                ),
            )
        )
        actions.extend(
            _wrapped_action(
                target=target,
                action_id=tail_action_id(source),
                policy_id=f"Hxe::{source}",
                action_kind=SINGLE_SOURCE_POLICY_ACTION_KIND,
                action_semantics=SINGLE_SOURCE_ACTION_SEMANTICS,
                core=build_single_source_tail_action(
                    source,
                    geometry=geometry,
                ),
                selected_source=source,
                diagnostic_control=True,
            )
            for source in sources
        )
        actions_by_target[target] = tuple(actions)
        rank_hashes_by_target[target] = {
            "global_rank_hash": surface.global_summary.rank_hash,
            "support_rank_hash": surface.support_summary.rank_hash,
        }

    unhashed: dict[str, object] = {
        "schema_version": "midogpp_residual_topup_case_oof_action_library_v1",
        "experiment_id": EXPERIMENT_ID,
        "centers": list(CENTERS),
        "action_count": EXPECTED_FROZEN_ACTION_COUNT,
        "actions_by_target": {
            target: [action.to_payload() for action in actions_by_target[target]]
            for target in CENTERS
        },
        "actions_frozen_before_evaluation_label_access": True,
        "fixed_support_rank_reused_across_target_folds": True,
        "oracle_or_downstream_outcomes_used": False,
    }
    payload = {**unhashed, "action_library_hash": canonical_sha256(unhashed)}
    return CaseOOFActionLibrary(
        actions_by_target=actions_by_target,
        rank_hashes_by_target=rank_hashes_by_target,
        payload=payload,
    )


def _wrapped_action(
    *,
    target: str,
    action_id: str,
    policy_id: str,
    action_kind: str,
    action_semantics: str,
    core,
    ranks: Mapping[str, float] | None = None,
    permutation: Mapping[str, str] | None = None,
    selected_source: str | None = None,
    diagnostic_control: bool = False,
) -> FrozenCaseOOFAction:
    return make_frozen_case_oof_action(
        target=target,
        action_id=action_id,
        policy_id=policy_id,
        action_kind=action_kind,
        action_semantics=action_semantics,
        sources=core.geometry.source_order,
        ranks=dict(ranks or {}),
        permutation=dict(permutation or {}),
        selected_source=selected_source,
        direction=dict(core.direction_weights),
        topup=dict(core.topup_counts),
        final={
            label: dict(core.final_counts_by_class[label])
            for label in (0, 1)
        },
        core=core,
        diagnostic_control=diagnostic_control,
    )


def _is_hash(value: object) -> bool:
    text = str(value)
    return len(text) in {16, 64} and all(
        character in "0123456789abcdef" for character in text
    )


__all__ = (
    "CaseOOFActionLibrary",
    "build_case_oof_action_library",
    "canonical_source_identity_permutation",
)
