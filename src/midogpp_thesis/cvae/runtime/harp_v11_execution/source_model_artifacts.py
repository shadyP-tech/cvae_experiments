"""Durable pairwise scores and nested selected-policy OOF replay artifacts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from .compatibility_adapter import (
    CompatibilityAdapterState,
    compatibility_state_from_artifact,
)
from .contracts import ArtifactValue
from .model_adapter import (
    RouterAdmissionState,
    RouterFitState,
    admission_manifest,
    build_source_only_admission,
    fit_outer_routers,
    model_manifest,
)
from .production_validation import require_sha256, require_state
from .source_development import SourceDevelopmentState


CompatibilityLoader = Callable[[ArtifactValue], CompatibilityAdapterState]


def _selected_action_id(
    prediction: object,
    *,
    acceptance_threshold: float,
    rank_margin_threshold: float = 0.0,
    policy_enabled: bool = True,
) -> str:
    action_id = getattr(prediction, "top_action_id", None)
    acceptance = float(getattr(prediction, "acceptance_probability", 0.0))
    if (
        not policy_enabled
        or action_id in (None, "B")
        or acceptance < float(acceptance_threshold)
        or float(getattr(prediction, "rank_margin", 0.0))
        < float(rank_margin_threshold)
    ):
        return "B"
    return str(action_id)


def _numeric_oof_arrays(state: RouterFitState) -> dict[str, np.ndarray]:
    case_rows: list[tuple[float, ...]] = []
    score_rows: list[tuple[float, ...]] = []
    score_offsets = [0]
    for bundle in state.bundles:
        for prediction in bundle.lodo.oof_predictions:
            case_rows.append(
                (
                    float(prediction.acceptance_probability),
                    float(prediction.rank_margin),
                    float(len(prediction.action_scores)),
                    float(prediction.top_action_id not in (None, "B")),
                )
            )
            score_rows.extend(
                (
                    float(score.pairwise_score),
                    float(score.predicted_budget_gain),
                    float(score.predicted_allocation_gain),
                    float(score.predicted_total_gain),
                    float(score.predicted_harm_probability),
                    float(score.predicted_brier_delta),
                    float(score.predicted_log_delta),
                    float(score.acceptance_probability),
                    float(score.model_available),
                )
                for score in prediction.action_scores
            )
            score_offsets.append(len(score_rows))
    return {
        "oof_case_values": np.asarray(case_rows, dtype=np.float64).reshape((-1, 4)),
        "oof_action_scores": np.asarray(score_rows, dtype=np.float64).reshape((-1, 9)),
        "oof_action_score_offsets": np.asarray(score_offsets, dtype=np.int64),
    }


def build_source_router_artifact(
    development: ArtifactValue,
    compatibility: ArtifactValue,
    *,
    config: object,
    compatibility_loader: CompatibilityLoader = compatibility_state_from_artifact,
    fit_fn: Callable[..., RouterFitState] = fit_outer_routers,
) -> ArtifactValue:
    """Fit all outer routers and persist every numeric nested-LODO score."""

    development_state = require_state(
        development, SourceDevelopmentState, role="source-development surface"
    )
    compatibility_state = compatibility_loader(compatibility)
    known_menu_hashes = {menu.menu_hash for menu in compatibility_state.effective_menus}
    if any(
        menu.menu_hash not in known_menu_hashes
        for menu in development_state.effective_menus
    ):
        raise ProtocolError("HARP v11 model rows escaped the sealed effective menu.")
    fitted = fit_fn(
        development_state,
        model_config=getattr(config, "model"),
        runtime_config=getattr(config, "runtime"),
    )
    body = {
        **model_manifest(fitted),
        "development_surface_hash": require_sha256(
            development.manifest.get("surface_hash"), role="development surface hash"
        ),
        "compatibility_hash": require_sha256(
            compatibility.manifest.get("compatibility_hash"), role="compatibility hash"
        ),
        "effective_menu_hash": canonical_hash(
            [menu.menu_hash for menu in development_state.effective_menus]
        ),
        "all_preprocessing_fit_inside_source_lodo": True,
        "regularization_hyperparameters_predeclared_fixed_before_source_lodo": True,
        "regularization_hyperparameter_selection_performed": False,
        "evaluation_labels_used": False,
    }
    return ArtifactValue(
        state=fitted,
        manifest={**body, "model_hash": canonical_hash(body)},
        arrays=_numeric_oof_arrays(fitted),
    )


def _policy_oof_replay(
    fitted: RouterFitState,
    admitted: RouterAdmissionState,
    development: SourceDevelopmentState,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    rows: list[dict[str, object]] = []
    numeric: list[tuple[float, ...]] = []
    nested_numeric: list[tuple[float, ...]] = []
    outcomes_by_case: dict[tuple[str, str, str], dict[str, object]] = {}
    menus_by_case = {
        (menu.outer_target_id, menu.query_center_id, menu.case_id): menu
        for menu in development.effective_menus
    }
    for outcome in development.outcomes:
        key = (
            outcome.action.outer_target_id,
            outcome.action.query_center_id,
            outcome.action.case_id,
        )
        outcomes_by_case.setdefault(key, {})[outcome.action.action_id] = outcome
    for bundle in fitted.bundles:
        policy = admitted.for_outer(bundle.outer_target_id)
        heldout_thresholds = {
            center: (threshold, margin)
            for center, threshold, margin in policy.calibration.heldout_thresholds
        }
        for prediction in bundle.lodo.oof_predictions:
            key = (
                bundle.outer_target_id,
                prediction.query_center_id,
                prediction.case_id,
            )
            menu = menus_by_case[key]
            outcomes = outcomes_by_case.get(key, {})
            best_gain = max(
                (0.0, *(float(value.bacc_gain) for value in outcomes.values()))
            )
            selected_id = _selected_action_id(
                prediction,
                acceptance_threshold=policy.calibration.acceptance_threshold,
                rank_margin_threshold=policy.calibration.rank_margin_threshold,
                policy_enabled=policy.policy_enabled,
            )
            reason = (
                "SOURCE_OOF_ROUTED_POLICY_ACCEPTED_TOP1"
                if selected_id != "B"
                else "EXACT_B_SOURCE_POLICY_ABSTENTION"
            )
            selected = outcomes.get(selected_id)
            bacc = 0.0 if selected is None else float(selected.bacc_gain)
            brier = 0.0 if selected is None else float(selected.brier_delta)
            log_delta = 0.0 if selected is None else float(selected.log_delta)
            regret = best_gain - bacc
            try:
                nested_threshold, nested_margin = heldout_thresholds[
                    prediction.query_center_id
                ]
            except KeyError as exc:
                raise ProtocolError(
                    "HARP v11 nested policy replay lacks a held-source threshold."
                ) from exc
            nested_fold = next(
                (
                    fold
                    for fold in bundle.lodo.nested_policy_folds
                    if fold.heldout_center_id == prediction.query_center_id
                ),
                None,
            )
            if nested_fold is None:
                raise ProtocolError("HARP v11 nested policy replay lacks its fold hash.")
            nested_prediction = next(
                (
                    row
                    for row in nested_fold.heldout_predictions
                    if row.case_id == prediction.case_id
                ),
                None,
            )
            if nested_prediction is None:
                raise ProtocolError(
                    "HARP v11 nested policy replay lacks a held-source prediction."
                )
            nested_selected_id = _selected_action_id(
                nested_prediction,
                acceptance_threshold=nested_threshold,
                rank_margin_threshold=nested_margin,
            )
            nested_selected = outcomes.get(nested_selected_id)
            nested_bacc = (
                0.0 if nested_selected is None else float(nested_selected.bacc_gain)
            )
            nested_brier = (
                0.0 if nested_selected is None else float(nested_selected.brier_delta)
            )
            nested_log_delta = (
                0.0 if nested_selected is None else float(nested_selected.log_delta)
            )
            nested_regret = best_gain - nested_bacc
            rows.append(
                {
                    "outer_target_id": bundle.outer_target_id,
                    "query_center_id": prediction.query_center_id,
                    "case_id": prediction.case_id,
                    "prediction": dict(prediction.public_payload()),
                    "active_action_ids": [action.action_id for action in menu.actions],
                    "active_action_hashes": [action.action_hash for action in menu.actions],
                    "selected_action_id": selected_id,
                    "reason": reason,
                    "observed_bacc_gain": bacc,
                    "observed_brier_delta": brier,
                    "observed_log_delta": log_delta,
                    "best_observed_bacc_gain": best_gain,
                    "regret": regret,
                    "acceptance_threshold": policy.calibration.acceptance_threshold,
                    "rank_margin_threshold": policy.calibration.rank_margin_threshold,
                    "nested_selected_action_id": nested_selected_id,
                    "nested_acceptance_probability": (
                        nested_prediction.acceptance_probability
                    ),
                    "nested_rank_margin": nested_prediction.rank_margin,
                    "nested_observed_bacc_gain": nested_bacc,
                    "nested_observed_brier_delta": nested_brier,
                    "nested_observed_log_delta": nested_log_delta,
                    "nested_regret": nested_regret,
                    "nested_acceptance_threshold": nested_threshold,
                    "nested_rank_margin_threshold": nested_margin,
                    "nested_threshold_training_center_ids": list(
                        nested_fold.training_center_ids
                    ),
                    "nested_policy_fold_hash": nested_fold.fold_hash,
                    "nested_policy_replay_hash": (
                        policy.calibration.nested_replay.replay_hash
                    ),
                    "heldout_model_hash": prediction.model_hash,
                    "nested_heldout_model_hash": nested_prediction.model_hash,
                    "nested_prediction_hash": nested_prediction.prediction_hash,
                    "menu_hash": prediction.menu_hash,
                    "prediction_hash": prediction.prediction_hash,
                }
            )
            numeric.append(
                (
                    float(prediction.acceptance_probability),
                    float(prediction.rank_margin),
                    float(selected_id != "B"),
                    bacc,
                    brier,
                    log_delta,
                    best_gain,
                    regret,
                )
            )
            nested_numeric.append(
                (
                    float(nested_prediction.acceptance_probability),
                    float(nested_prediction.rank_margin),
                    float(nested_selected_id != "B"),
                    nested_bacc,
                    nested_brier,
                    nested_log_delta,
                    best_gain,
                    nested_regret,
                )
            )
    return (
        rows,
        np.asarray(numeric, dtype=np.float64).reshape((-1, 8)),
        np.asarray(nested_numeric, dtype=np.float64).reshape((-1, 8)),
    )


def build_source_admission_artifact(
    fitted: ArtifactValue,
    development: ArtifactValue,
    *,
    config: object,
    admission_fn: Callable[..., RouterAdmissionState] = build_source_only_admission,
) -> ArtifactValue:
    """Persist local admission plus the deployed whole-policy OOF replay."""

    fit_state = require_state(fitted, RouterFitState, role="fitted router")
    development_state = require_state(
        development, SourceDevelopmentState, role="source-development surface"
    )
    admitted = admission_fn(
        fit_state,
        development_state,
        model_config=getattr(config, "model"),
    )
    replay_rows, replay_values, nested_replay_values = _policy_oof_replay(
        fit_state, admitted, development_state
    )
    body = {
        **admission_manifest(admitted),
        "model_hash": require_sha256(fitted.manifest.get("model_hash"), role="model hash"),
        "development_surface_hash": require_sha256(
            development.manifest.get("surface_hash"), role="development surface hash"
        ),
        "source_policy_oof_rows": replay_rows,
        "source_policy_oof_case_count": len(replay_rows),
        "complete_rank_accept_route_or_exact_b_replayed": True,
        "nested_held_source_threshold_policy_replayed": True,
        "per_action_worst_center_certificate_used": False,
        "per_outer_local_admission": True,
        "global_kill_switch_used": False,
        "source_only": True,
        "evaluation_labels_used": False,
    }
    return ArtifactValue(
        state=admitted,
        manifest={**body, "admission_hash": canonical_hash(body)},
        arrays={
            "source_policy_oof_values": replay_values,
            "nested_source_policy_oof_values": nested_replay_values,
        },
    )


__all__ = ("build_source_admission_artifact", "build_source_router_artifact")
