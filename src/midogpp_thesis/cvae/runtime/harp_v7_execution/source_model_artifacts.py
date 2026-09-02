"""Durable nested-source model and whole-policy OOF replay artifacts."""

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


def _numeric_oof_arrays(state: RouterFitState) -> dict[str, np.ndarray]:
    case_rows = []
    score_rows = []
    score_offsets = [0]
    for bundle in state.bundles:
        for prediction in bundle.lodo.oof_predictions:
            case_rows.append(
                (
                    prediction.opportunity_probability,
                    prediction.rank_margin,
                    float(len(prediction.action_scores)),
                )
            )
            score_rows.extend((score.score,) for score in prediction.action_scores)
            score_offsets.append(len(score_rows))
    return {
        "oof_case_values": np.asarray(case_rows, dtype=np.float64).reshape((-1, 3)),
        "oof_action_scores": np.asarray(score_rows, dtype=np.float64).reshape((-1, 1)),
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
    """Fit all outer routers and persist every numeric nested-LODO replay."""

    development_state = require_state(
        development, SourceDevelopmentState, role="source-development surface"
    )
    compatibility_state = compatibility_loader(compatibility)
    known_menu_hashes = {menu.menu_hash for menu in compatibility_state.effective_menus}
    if any(
        menu.menu_hash not in known_menu_hashes
        for menu in development_state.effective_menus
    ):
        raise ProtocolError("HARP v7 model rows escaped the sealed effective menu.")
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
        "all_preprocessing_and_hyperparameters_nested_inside_source_lodo": True,
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
            center: (opportunity, margin)
            for center, opportunity, margin in policy.calibration.heldout_thresholds
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
            selected_id = "B"
            reason = "EXACT_B_SOURCE_POLICY_ABSTENTION"
            if (
                policy.policy_enabled
                and prediction.top_action_id is not None
                and prediction.opportunity_probability
                >= policy.calibration.opportunity_threshold
                and prediction.passes_rank_margin(
                    policy.calibration.rank_margin_threshold
                )
            ):
                selected_id = prediction.top_action_id
                reason = "SOURCE_OOF_ROUTED_EXACT_TOP1"
            selected = outcomes.get(selected_id)
            bacc = 0.0 if selected is None else float(selected.bacc_gain)
            brier = 0.0 if selected is None else float(selected.brier_delta)
            log_delta = 0.0 if selected is None else float(selected.log_delta)
            regret = best_gain - bacc
            try:
                nested_opportunity_threshold, nested_rank_margin_threshold = (
                    heldout_thresholds[prediction.query_center_id]
                )
            except KeyError as exc:
                raise ProtocolError(
                    "HARP v7 nested policy replay lacks a held-source threshold."
                ) from exc
            nested_selected_id = "B"
            if (
                prediction.top_action_id is not None
                and prediction.opportunity_probability >= nested_opportunity_threshold
                and prediction.passes_rank_margin(nested_rank_margin_threshold)
            ):
                nested_selected_id = prediction.top_action_id
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
            nested_fold = next(
                (
                    fold
                    for fold in bundle.lodo.nested_policy_folds
                    if fold.heldout_center_id == prediction.query_center_id
                ),
                None,
            )
            if nested_fold is None:
                raise ProtocolError("HARP v7 nested policy replay lacks its fold hash.")
            rows.append(
                {
                    "outer_target_id": bundle.outer_target_id,
                    "query_center_id": prediction.query_center_id,
                    "case_id": prediction.case_id,
                    "opportunity_probability": prediction.opportunity_probability,
                    "rank_margin": prediction.rank_margin,
                    "active_action_ids": [action.action_id for action in menu.actions],
                    "active_action_hashes": [action.action_hash for action in menu.actions],
                    "action_scores": [
                        {
                            "action_id": score.action_id,
                            "direction": score.direction.value,
                            "score": score.score,
                            "action_hash": score.action_hash,
                        }
                        for score in prediction.action_scores
                    ],
                    "selected_action_id": selected_id,
                    "reason": reason,
                    "observed_bacc_gain": bacc,
                    "observed_brier_delta": brier,
                    "observed_log_delta": log_delta,
                    "best_observed_bacc_gain": best_gain,
                    "regret": regret,
                    "nested_selected_action_id": nested_selected_id,
                    "nested_observed_bacc_gain": nested_bacc,
                    "nested_observed_brier_delta": nested_brier,
                    "nested_observed_log_delta": nested_log_delta,
                    "nested_regret": nested_regret,
                    "nested_opportunity_threshold": nested_opportunity_threshold,
                    "nested_rank_margin_threshold": nested_rank_margin_threshold,
                    "nested_threshold_training_center_ids": list(
                        nested_fold.training_center_ids
                    ),
                    "nested_policy_fold_hash": nested_fold.fold_hash,
                    "nested_policy_replay_hash": (
                        policy.calibration.nested_replay.replay_hash
                    ),
                    "heldout_model_hash": prediction.model_hash,
                    "menu_hash": prediction.menu_hash,
                    "prediction_hash": prediction.prediction_hash,
                }
            )
            numeric.append(
                (
                    prediction.opportunity_probability,
                    prediction.rank_margin,
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
                    prediction.opportunity_probability,
                    prediction.rank_margin,
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
    """Persist local admission plus the actually deployed whole-policy OOF replay."""

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
        "actual_route_or_exact_b_replayed": True,
        "nested_held_source_threshold_policy_replayed": True,
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
