"""Durable action certificates and nested whole-policy OOF replay artifacts."""

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


def _certificate_confidence(prediction: object, action_id: str | None) -> float:
    if action_id is None:
        return 0.0
    certificate = next(
        (
            row
            for row in getattr(prediction, "action_certificates")
            if row.action_id == action_id
        ),
        None,
    )
    if certificate is None or not certificate.safe:
        return 0.0
    return 1.0 - float(certificate.harm_probability_ucb)


def _selected_action_id(
    prediction: object,
    *,
    certificate_confidence_threshold: float,
    rank_margin_threshold: float,
) -> str:
    action_id = getattr(prediction, "top_action_id")
    if (
        action_id is None
        or _certificate_confidence(prediction, action_id)
        < certificate_confidence_threshold
        or not prediction.passes_rank_margin(rank_margin_threshold)
    ):
        return "B"
    return str(action_id)


def _numeric_oof_arrays(state: RouterFitState) -> dict[str, np.ndarray]:
    case_rows = []
    certificate_rows = []
    certificate_offsets = [0]
    for bundle in state.bundles:
        for prediction in bundle.lodo.oof_predictions:
            case_rows.append(
                (
                    prediction.certificate_confidence_diagnostic,
                    prediction.rank_margin,
                    float(len(prediction.safe_action_ids)),
                    float(len(prediction.action_certificates)),
                )
            )
            certificate_rows.extend(
                (
                    certificate.estimate.predicted_bacc_gain,
                    certificate.estimate.predicted_harm_probability,
                    certificate.estimate.predicted_brier_delta,
                    certificate.estimate.predicted_log_delta,
                    certificate.gain_lcb,
                    certificate.harm_probability_ucb,
                    certificate.brier_delta_ucb,
                    certificate.log_delta_ucb,
                    certificate.harm_brier_risk,
                    certificate.harm_log_loss_risk,
                    float(certificate.estimate.model_available),
                    float(certificate.safe),
                )
                for certificate in prediction.action_certificates
            )
            certificate_offsets.append(len(certificate_rows))
    return {
        "oof_case_values": np.asarray(case_rows, dtype=np.float64).reshape((-1, 4)),
        "oof_action_certificates": np.asarray(
            certificate_rows, dtype=np.float64
        ).reshape((-1, 12)),
        "oof_action_certificate_offsets": np.asarray(
            certificate_offsets, dtype=np.int64
        ),
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
        raise ProtocolError("HARP v8 model rows escaped the sealed effective menu.")
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
            if policy.policy_enabled:
                selected_id = _selected_action_id(
                    prediction,
                    certificate_confidence_threshold=(
                        policy.calibration.certificate_confidence_threshold
                    ),
                    rank_margin_threshold=policy.calibration.rank_margin_threshold,
                )
                if selected_id != "B":
                    reason = "SOURCE_OOF_ROUTED_CERTIFIED_EXACT_TOP1"
            selected = outcomes.get(selected_id)
            bacc = 0.0 if selected is None else float(selected.bacc_gain)
            brier = 0.0 if selected is None else float(selected.brier_delta)
            log_delta = 0.0 if selected is None else float(selected.log_delta)
            regret = best_gain - bacc
            try:
                nested_certificate_confidence_threshold, nested_rank_margin_threshold = (
                    heldout_thresholds[prediction.query_center_id]
                )
            except KeyError as exc:
                raise ProtocolError(
                    "HARP v8 nested policy replay lacks a held-source threshold."
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
                raise ProtocolError("HARP v8 nested policy replay lacks its fold hash.")
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
                    "HARP v8 nested policy replay lacks a held-source prediction."
                )
            nested_selected_id = _selected_action_id(
                nested_prediction,
                certificate_confidence_threshold=(
                    nested_certificate_confidence_threshold
                ),
                rank_margin_threshold=nested_rank_margin_threshold,
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
                    "certificate_confidence_diagnostic": (
                        prediction.certificate_confidence_diagnostic
                    ),
                    "selected_certificate_confidence": _certificate_confidence(
                        prediction, prediction.top_action_id
                    ),
                    "rank_margin": prediction.rank_margin,
                    "safe_action_ids": list(prediction.safe_action_ids),
                    "active_action_ids": [action.action_id for action in menu.actions],
                    "active_action_hashes": [action.action_hash for action in menu.actions],
                    "action_certificates": [
                        {
                            "action_id": certificate.action_id,
                            "direction": certificate.direction.value,
                            "action_hash": certificate.action_hash,
                            "action_group": certificate.estimate.action_group,
                            "model_available": (
                                certificate.estimate.model_available
                            ),
                            "predicted_bacc_gain": (
                                certificate.estimate.predicted_bacc_gain
                            ),
                            "predicted_harm_probability": (
                                certificate.estimate.predicted_harm_probability
                            ),
                            "predicted_brier_delta": (
                                certificate.estimate.predicted_brier_delta
                            ),
                            "predicted_log_delta": (
                                certificate.estimate.predicted_log_delta
                            ),
                            "gain_lcb": certificate.gain_lcb,
                            "harm_probability_ucb": certificate.harm_probability_ucb,
                            "brier_delta_ucb": certificate.brier_delta_ucb,
                            "log_delta_ucb": certificate.log_delta_ucb,
                            "harm_brier_risk": certificate.harm_brier_risk,
                            "harm_log_loss_risk": certificate.harm_log_loss_risk,
                            "calibration_cell_hash": (
                                certificate.calibration_cell_hash
                            ),
                            "safe": certificate.safe,
                            "failed_gates": list(certificate.failed_gates),
                            "certificate_hash": certificate.certificate_hash,
                        }
                        for certificate in prediction.action_certificates
                    ],
                    "selected_action_id": selected_id,
                    "reason": reason,
                    "observed_bacc_gain": bacc,
                    "observed_brier_delta": brier,
                    "observed_log_delta": log_delta,
                    "best_observed_bacc_gain": best_gain,
                    "regret": regret,
                    "nested_selected_action_id": nested_selected_id,
                    "nested_certificate_confidence": _certificate_confidence(
                        nested_prediction, nested_prediction.top_action_id
                    ),
                    "nested_rank_margin": nested_prediction.rank_margin,
                    "nested_safe_action_ids": list(nested_prediction.safe_action_ids),
                    "nested_observed_bacc_gain": nested_bacc,
                    "nested_observed_brier_delta": nested_brier,
                    "nested_observed_log_delta": nested_log_delta,
                    "nested_regret": nested_regret,
                    "nested_certificate_confidence_threshold": (
                        nested_certificate_confidence_threshold
                    ),
                    "nested_rank_margin_threshold": nested_rank_margin_threshold,
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
                    _certificate_confidence(prediction, prediction.top_action_id),
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
                    _certificate_confidence(
                        nested_prediction, nested_prediction.top_action_id
                    ),
                    nested_prediction.rank_margin,
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
