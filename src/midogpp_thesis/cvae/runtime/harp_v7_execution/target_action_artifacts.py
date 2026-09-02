"""Sealed target effective menus and source-only case predictions for HARP v7."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from .compatibility_adapter import (
    CompatibilityAdapterState,
    compatibility_state_from_artifact,
)
from .contracts import (
    ActionKind,
    ArtifactValue,
    LabelFreeOuterMenu,
    array_bytes_sha256,
)
from .model_adapter import (
    RouterAdmissionState,
    RouterFitState,
    TargetEvidenceState,
    predict_target_evidence,
    target_evidence_manifest,
)
from .production_validation import case_indices, require_sha256, require_state


CompatibilityLoader = Callable[[ArtifactValue], CompatibilityAdapterState]


def _decode_hex(values: Sequence[str]) -> np.ndarray:
    try:
        raw = b"".join(bytes.fromhex(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("HARP v7 target effective probability hex is malformed.") from exc
    output = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
    if not len(output) or not np.isfinite(output).all():
        raise ProtocolError("HARP v7 target effective probability vector is nonfinite.")
    return output


def _target_rows(
    menus: tuple[LabelFreeOuterMenu, ...], state: TargetEvidenceState
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    physical = {menu.outer_target_id: menu for menu in menus}
    probability_rows: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for effective in state.menus:
        baseline = physical[effective.outer_target_id].target_block(ActionKind.B)
        indices = case_indices(baseline, effective.case_id)
        sample_ids = tuple(baseline.sample_ids[int(index)] for index in indices)
        for action in effective.actions:
            values = _decode_hex(action.action_probability_hex)
            if len(values) != len(sample_ids):
                raise ProtocolError("HARP v7 target action/sample geometry drifted.")
            ordinal = len(probability_rows)
            probability_rows.append(values)
            rows.append(
                {
                    "outer_target_id": effective.outer_target_id,
                    "case_id": effective.case_id,
                    "action_id": action.action_id,
                    "action_kind": action.action_kind,
                    "direction": action.direction.value,
                    "candidate_source_id": action.candidate_source_id,
                    "action_hash": action.action_hash,
                    "menu_hash": effective.menu_hash,
                    "sample_ids": list(sample_ids),
                    "probability_hash": array_bytes_sha256(values),
                    "probability_row_ordinal": ordinal,
                }
            )
    offsets = [0]
    for values in probability_rows:
        offsets.append(offsets[-1] + len(values))
    predictions = state.predictions
    score_values = [score.score for row in predictions for score in row.action_scores]
    score_offsets = [0]
    for row in predictions:
        score_offsets.append(score_offsets[-1] + len(row.action_scores))
    arrays = {
        "probabilities": (
            np.concatenate(probability_rows).astype(np.float32, copy=False)
            if probability_rows
            else np.asarray([], dtype=np.float32)
        ),
        "probability_offsets": np.asarray(offsets, dtype=np.int64),
        "case_prediction_values": np.asarray(
            [
                (row.opportunity_probability, row.rank_margin)
                for row in predictions
            ],
            dtype=np.float64,
        ).reshape((-1, 2)),
        "action_score_values": np.asarray(score_values, dtype=np.float64),
        "action_score_offsets": np.asarray(score_offsets, dtype=np.int64),
    }
    for ordinal, row in enumerate(rows):
        row["probability_offset_start"] = offsets[ordinal]
        row["probability_offset_stop"] = offsets[ordinal + 1]
    return rows, arrays


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
    """Score the already-sealed target effective menus without target outcomes."""

    del config
    physical_menus = tuple(menus)
    compatibility_state = compatibility_loader(compatibility)
    fit_state = require_state(fit, RouterFitState, role="fitted router")
    require_state(admission, RouterAdmissionState, role="per-outer policy admission")
    target_menus = tuple(
        menu
        for menu in compatibility_state.effective_menus
        if menu.query_center_id == menu.outer_target_id
    )
    target_state = predict_fn(target_menus, fit_state)
    rows, arrays = _target_rows(physical_menus, target_state)
    body = {
        **target_evidence_manifest(target_state),
        "outer_menu_hashes": {
            menu.outer_target_id: menu.menu_hash for menu in physical_menus
        },
        "model_hash": require_sha256(fit.manifest.get("model_hash"), role="model hash"),
        "compatibility_hash": require_sha256(
            compatibility.manifest.get("compatibility_hash"), role="compatibility hash"
        ),
        "admission_hash": require_sha256(
            admission.manifest.get("admission_hash"), role="admission hash"
        ),
        "rows": rows,
        "active_label_free_action_count": len(rows),
        "target_effective_menu_count": len(target_state.menus),
        "all_margins_excluded_before_prediction": True,
        "exact_top1_physical_action_only": True,
        "unevaluated_mixtures_used": False,
        "evaluation_labels_used": False,
    }
    return ArtifactValue(
        state=target_state,
        manifest={**body, "target_action_hash": canonical_hash(body)},
        arrays=arrays,
    )


__all__ = ("build_complete_target_action_artifact",)
