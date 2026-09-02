"""Source-only development-surface construction for HARP v9.

This module owns the only development-label join in the production pipeline.
It consumes labels only after the runner has sealed the label-free physical and
compatibility artifacts, and it rejects any label outside that sealed universe.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ...protocol import ProtocolError
from ...routing.policy_calibrated_residual_router_v9 import (
    EffectiveMenu,
    SourceActionOutcome,
)
from ...routing.harp_protocol import canonical_hash
from .compatibility_adapter import (
    CompatibilityAdapterState,
    compatibility_state_from_artifact,
)
from .contracts import ActionKind, ArtifactValue, LabelFreeOuterMenu
from .directional_surfaces import attach_source_outcomes


CompatibilityLoader = Callable[[ArtifactValue], CompatibilityAdapterState]


@dataclass(frozen=True, slots=True)
class SourceDevelopmentState:
    """All source case menus plus development-only endpoint outcomes."""

    effective_menus: tuple[EffectiveMenu, ...]
    outcomes: tuple[SourceActionOutcome, ...]

    def __post_init__(self) -> None:
        menus = tuple(
            sorted(
                self.effective_menus,
                key=lambda row: (row.outer_target_id, row.query_center_id, row.case_id),
            )
        )
        outcomes = tuple(
            sorted(
                self.outcomes,
                key=lambda row: (
                    row.action.outer_target_id,
                    row.action.query_center_id,
                    row.action.case_id,
                    row.action.action_id,
                ),
            )
        )
        menu_membership = {
            (
                menu.outer_target_id,
                menu.query_center_id,
                menu.case_id,
                action.action_id,
            ): action.action_hash
            for menu in menus
            for action in menu.actions
        }
        outcome_membership = {
            (
                row.action.outer_target_id,
                row.action.query_center_id,
                row.action.case_id,
                row.action.action_id,
            ): row.action.action_hash
            for row in outcomes
        }
        if (
            not menus
            or any(menu.query_center_id == menu.outer_target_id for menu in menus)
            or len(menu_membership) != sum(len(menu.actions) for menu in menus)
            or len(outcome_membership) != len(outcomes)
            or outcome_membership != menu_membership
            or len({row.outcome_hash for row in outcomes}) != len(outcomes)
        ):
            raise ProtocolError("HARP v9 source development state crossed sealed menus.")
        object.__setattr__(self, "effective_menus", menus)
        object.__setattr__(self, "outcomes", outcomes)


def build_source_development_artifact(
    menus: Sequence[LabelFreeOuterMenu],
    compatibility: ArtifactValue,
    development_labels: object,
    *,
    config: object,
    compatibility_loader: CompatibilityLoader = compatibility_state_from_artifact,
) -> ArtifactValue:
    """Join sealed development labels and serialize the source-only surface."""

    menu_rows = tuple(menus)
    state = compatibility_loader(compatibility)
    label_rows = tuple(development_labels)  # type: ignore[arg-type]
    try:
        label_index = {
            (str(row.center), str(row.case_id), str(row.sample_id)): int(row.label)
            for row in label_rows
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProtocolError("HARP v9 source-development labels are malformed.") from exc
    if (
        len(label_index) != len(label_rows)
        or any(value not in (0, 1) for value in label_index.values())
    ):
        raise ProtocolError("HARP v9 source-development label identities drifted.")

    observations: list[SourceActionOutcome] = []
    source_menus: list[EffectiveMenu] = []
    expected_label_keys: set[tuple[str, str, str]] = set()
    for menu in menu_rows:
        queries = tuple(
            sorted(
                {
                    block.query_center_id
                    for block in menu.blocks
                    if block.surface_role == "development"
                }
            )
        )
        for query in queries:
            baseline = tuple(
                block
                for block in menu.blocks
                if block.surface_role == "development"
                and block.query_center_id == query
                and block.action_kind is ActionKind.B
            )
            if len(baseline) != 1:
                raise ProtocolError("HARP v9 development context lacks exact B.")
            scoped_keys = {
                (case, sample)
                for case, sample in zip(
                    baseline[0].case_ids, baseline[0].sample_ids, strict=True
                )
            }
            expected_label_keys.update(
                (query, case, sample) for case, sample in scoped_keys
            )
            try:
                scoped_labels = {
                    (case, sample): label_index[(query, case, sample)]
                    for case, sample in scoped_keys
                }
            except KeyError as exc:
                raise ProtocolError(
                    "HARP v9 development labels do not cover the sealed source menu."
                ) from exc
            effective = state.menus(menu.outer_target_id, query)
            source_menus.extend(effective)
            observations.extend(
                attach_source_outcomes(
                    effective,
                    baseline[0],
                    source_labels=scoped_labels,
                )
            )
    if set(label_index) != expected_label_keys:
        raise ProtocolError(
            "HARP v9 source-development labels exceed or omit the sealed menu universe."
        )
    ordered = tuple(
        sorted(
            observations,
            key=lambda row: (
                row.action.outer_target_id,
                row.action.query_center_id,
                row.action.case_id,
                row.action.action_id,
            ),
        )
    )
    if not ordered or len({row.outcome_hash for row in ordered}) != len(ordered):
        raise ProtocolError("HARP v9 source-development response inventory drifted.")
    names = ordered[0].action.feature_names
    if any(row.action.feature_names != names for row in ordered):
        raise ProtocolError("HARP v9 source feature schema differs across actions.")
    development_state = SourceDevelopmentState(tuple(source_menus), ordered)
    body = {
        "schema_version": "midogpp_harp_v9_pairwise_residual_action_development_surface_v1",
        "config_hash": getattr(config, "config_hash"),
        "outer_targets": list(getattr(config, "protocol")["centers"]),
        "observation_count": len(ordered),
        "feature_names": list(names),
        "source_response_hashes": [row.outcome_hash for row in ordered],
        "effective_menu_count": len(development_state.effective_menus),
        "active_case_count": sum(bool(menu.actions) for menu in development_state.effective_menus),
        "effective_menu_hashes": [menu.menu_hash for menu in development_state.effective_menus],
        "rows": [
            {
                "outer_target_id": row.action.outer_target_id,
                "query_center_id": row.action.query_center_id,
                "case_id": row.action.case_id,
                "action_id": row.action.action_id,
                "action_kind": row.action.action_kind,
                "direction": row.action.direction.value,
                "candidate_source_id": row.action.candidate_source_id,
                "action_hash": row.action.action_hash,
                "source_response_hash": row.outcome_hash,
            }
            for row in ordered
        ],
        "strict_outer_H_query_candidate_exclusion": True,
        "effective_menu_sealed_before_development_labels": True,
        "all_margins_and_structural_noops_excluded": True,
        "exact_B_is_explicit_zero_effect_control": True,
        "budget_U_minus_B_and_allocation_Hxe_minus_U_contrasts_available": True,
        "within_case_pairwise_preferences_available": True,
        "selected_action_gain_harm_and_proper_loss_targets_available": True,
        "response_scope": "SOURCE_DEVELOPMENT_ONLY",
        "evaluation_labels_used": False,
    }
    return ArtifactValue(
        state=development_state,
        manifest={**body, "surface_hash": canonical_hash(body)},
        arrays={
            "feature_values": np.asarray(
                [row.action.feature_values for row in ordered], dtype=np.float64
            ),
            "endpoint_effects": np.asarray(
                [
                    (row.bacc_gain, row.brier_delta, row.log_delta)
                    for row in ordered
                ],
                dtype=np.float64,
            ),
        },
    )


__all__ = ("SourceDevelopmentState", "build_source_development_artifact")
