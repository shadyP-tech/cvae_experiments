"""Label-free opportunity filtering and target-action persistence for HARP v5."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ...routing.compatibility_conditioned_directional_router import (
    TargetAction,
    build_label_free_opportunity,
)
from ...routing.harp_protocol import canonical_hash
from .compatibility_adapter import (
    CompatibilityAdapterState,
    compatibility_state_from_artifact,
)
from .contracts import ActionKind, ArtifactValue, LabelFreeOuterMenu
from .directional_surfaces import build_target_directional_actions
from .model_adapter import (
    RouterAdmissionState,
    RouterFitState,
    TargetEvidenceState,
    predict_target_evidence,
    target_evidence_manifest,
)
from .production_validation import (
    case_ids,
    decode_cells,
    float32_cells,
    receipts_for_pool,
    require_sha256,
    require_state,
    target_case_blocks,
)


CompatibilityLoader = Callable[[ArtifactValue], CompatibilityAdapterState]


def _ordered_target_actions(
    menus: tuple[LabelFreeOuterMenu, ...],
    compatibility_state: CompatibilityAdapterState,
) -> tuple[TargetAction, ...]:
    actions: list[TargetAction] = []
    for menu in menus:
        pool = compatibility_state.pool(menu.outer_target_id, menu.outer_target_id)
        actions.extend(
            build_target_directional_actions(
                menu,
                candidate_pool=pool,
                compatibility_receipts=receipts_for_pool(
                    compatibility_state, menu.outer_target_id, menu.outer_target_id
                ),
            )
        )
    return tuple(
        sorted(
            actions,
            key=lambda row: (
                row.feature.outer_target_id,
                row.feature.case_id,
                row.feature.action_id,
            ),
        )
    )


def _label_free_opportunity_inventory(
    menus: tuple[LabelFreeOuterMenu, ...],
    actions: tuple[TargetAction, ...],
) -> tuple[list[dict[str, object]], set[tuple[str, str, str]]]:
    rows: list[dict[str, object]] = []
    active_keys: set[tuple[str, str, str]] = set()
    for menu in menus:
        baseline = menu.target_block(ActionKind.B)
        for case_id in case_ids(baseline):
            scoped = tuple(
                row
                for row in actions
                if row.feature.outer_target_id == menu.outer_target_id
                and row.feature.case_id == case_id
            )
            _, baseline_values, _ = target_case_blocks(menu, case_id)
            opportunity = build_label_free_opportunity(
                baseline_probability_bytes=float32_cells(baseline_values),
                actions=scoped,
            )
            active_keys.update(
                (menu.outer_target_id, case_id, action_id)
                for action_id in opportunity.active_representative_ids
            )
            rows.append(
                {
                    "outer_target_id": menu.outer_target_id,
                    "case_id": case_id,
                    "opportunity_hash": opportunity.opportunity_hash,
                    "candidate_action_count": len(opportunity.candidate_action_ids),
                    "active_representative_ids": list(
                        opportunity.active_representative_ids
                    ),
                    "structural_noop_action_ids": [
                        row.action_id
                        for row in opportunity.members
                        if row.structural_noop
                    ],
                }
            )
    return rows, active_keys


def _target_artifact_arrays(
    state: TargetEvidenceState,
) -> tuple[dict[str, np.ndarray], list[int]]:
    probability_rows = [decode_cells(row.probability_bytes) for row in state.actions]
    offsets = [0]
    for values in probability_rows:
        offsets.append(offsets[-1] + len(values))
    arrays = {
        "feature_values": np.asarray(
            [row.feature.feature_values for row in state.actions], dtype=np.float64
        ),
        "probabilities": np.concatenate(probability_rows).astype(
            np.float32, copy=False
        ),
        "probability_offsets": np.asarray(offsets, dtype=np.int64),
        "bounded_evidence": np.asarray(
            [
                (
                    row.prediction.opportunity_probability,
                    row.prediction.ranking_score,
                    *row.prediction.predicted_effects.as_tuple(),
                    row.bounds.bacc_lcb,
                    row.bounds.brier_ucb,
                    row.bounds.log_ucb,
                )
                for row in state.evidence
            ],
            dtype=np.float64,
        ).reshape((-1, 8)),
    }
    return arrays, offsets


def build_complete_target_action_artifact(
    menus: Sequence[LabelFreeOuterMenu],
    compatibility: ArtifactValue,
    fit: ArtifactValue,
    admission: ArtifactValue,
    *,
    config: object,
    compatibility_loader: CompatibilityLoader = compatibility_state_from_artifact,
    predict_fn: Callable[..., TargetEvidenceState] = predict_target_evidence,
) -> ArtifactValue:
    """Build, filter, predict, and serialize the complete label-free action set."""

    del config
    menu_rows = tuple(menus)
    compatibility_state = compatibility_loader(compatibility)
    fit_state = require_state(fit, RouterFitState, role="fitted router")
    require_state(admission, RouterAdmissionState, role="learnability admission")
    ordered_actions = _ordered_target_actions(menu_rows, compatibility_state)
    opportunity_rows, active_keys = _label_free_opportunity_inventory(
        menu_rows, ordered_actions
    )
    predicted = predict_fn(ordered_actions, fit_state)
    bounded = tuple(
        row
        for row in predicted.evidence
        if (
            row.prediction.feature.outer_target_id,
            row.prediction.feature.case_id,
            row.prediction.feature.action_id,
        )
        in active_keys
    )
    target_state = TargetEvidenceState(
        predicted.actions,
        bounded,
        predicted.failed_uncertainty_action_ids,
    )
    arrays, offsets = _target_artifact_arrays(target_state)
    inner = target_evidence_manifest(target_state)
    body = {
        **inner,
        "outer_menu_hashes": {
            menu.outer_target_id: menu.menu_hash for menu in menu_rows
        },
        "model_hash": require_sha256(fit.manifest.get("model_hash"), role="model hash"),
        "compatibility_hash": require_sha256(
            compatibility.manifest.get("compatibility_hash"),
            role="compatibility hash",
        ),
        "admission_hash": require_sha256(
            admission.manifest.get("admission_hash"), role="admission hash"
        ),
        "opportunity_cases": opportunity_rows,
        "opportunity_case_count": len(opportunity_rows),
        "active_label_free_action_count": len(active_keys),
        "complete_actions_retained_for_audit": True,
        "rows": [
            {
                "outer_target_id": row.feature.outer_target_id,
                "case_id": row.feature.case_id,
                "action_id": row.feature.action_id,
                "action_kind": row.feature.action_kind.value,
                "direction": row.feature.direction.value,
                "candidate_source_id": row.feature.candidate_source_id,
                "feature_hash": row.feature.feature_hash,
                "probability_hash": row.feature.probability_hash,
                "target_action_hash": row.target_action_hash,
                "sample_ids": list(row.sample_ids),
                "probability_offset_start": offsets[ordinal],
                "probability_offset_stop": offsets[ordinal + 1],
            }
            for ordinal, row in enumerate(target_state.actions)
        ],
        "evidence_rows": [
            {
                "outer_target_id": row.prediction.feature.outer_target_id,
                "case_id": row.prediction.feature.case_id,
                "action_id": row.prediction.feature.action_id,
                "prediction_hash": row.prediction.prediction_hash,
                "evidence_hash": row.evidence_hash,
                "safe_vs_baseline": row.safe_vs_baseline,
            }
            for row in target_state.evidence
        ],
        "evaluation_labels_used": False,
    }
    return ArtifactValue(
        state=target_state,
        manifest={**body, "target_action_hash": canonical_hash(body)},
        arrays=arrays,
    )


__all__ = ("build_complete_target_action_artifact",)
