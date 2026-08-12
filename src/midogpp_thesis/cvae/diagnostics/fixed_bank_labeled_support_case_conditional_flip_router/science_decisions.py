"""Three-role selection, calibration, and pre-evaluation decision phases."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.threshold_flip_case_router import (
    DirectionSharedCalibration,
    StaticSelection,
    TwoHeadRidgeModel,
    build_calibration_row,
    fit_direction_shared_calibration,
    predict_two_head,
    select_case_action,
)
from .constants import (
    B_ACTION_ID,
    CENTERS,
    FEATURE_NAMES,
    U_ACTION_ID,
    a1_action_id,
    candidate_sources,
)
from .hashing import canonical_hash
from .products import DecisionBundle, MethodDecision
from .science_common import (
    _assert_science_config,
    _case_contribution,
    _core_feature,
    _feature_index,
    _label_index,
    _label_surface_hash,
    _probability_index,
)
from .science_contracts import DecisionPhaseResult, DonorPhaseResult
from .science_donor import _safe_static_selection


def build_fold_decision_phase(
    *,
    probability_surface: object,
    prelabel: object,
    partition: object,
    manager: object,
    donor_phase: DonorPhaseResult,
    config: object,
) -> DecisionPhaseResult:
    """Select, calibrate, and seal seven methods for all 45 held folds."""

    _assert_science_config(config)
    probability = _probability_index(probability_surface)
    features = _feature_index(prelabel)
    static_rows: list[Mapping[str, object]] = []
    calibration_rows: list[Mapping[str, object]] = []
    decisions: list[MethodDecision] = []
    fold_seals: dict[tuple[str, int], str] = {}
    static_by_fold: dict[tuple[str, int], Mapping[str, StaticSelection]] = {}
    calibration_by_fold: dict[
        tuple[str, int], Mapping[str, DirectionSharedCalibration]
    ] = {}

    for target in CENTERS:
        ordinary_model = donor_phase.model_by_target[target]
        permutation_model = donor_phase.permutation_model_by_target[target]
        global_selection = donor_phase.global_selection_by_target[target]
        for fold_ordinal in range(5):
            fold = partition.fold(target, fold_ordinal)
            selection_labels = tuple(manager.open_selection_labels(target, fold_ordinal))
            selection_index = _label_index(selection_labels)
            selection_targets = {
                a1_action_id(source): tuple(
                    _case_contribution(
                        probability,
                        selection_index,
                        target_center=target,
                        case_id=case_id,
                        action_id=a1_action_id(source),
                    )
                    for case_id in fold.selection_case_ids
                )
                for source in candidate_sources(target)
            }
            support_selection = _safe_static_selection(selection_targets)
            static = MappingProxyType(
                {"G": global_selection, "S": support_selection}
            )
            static_by_fold[(target, fold_ordinal)] = static
            static_payload = {
                "target_center": target,
                "fold_ordinal": fold_ordinal,
                "selection_case_ids": list(fold.selection_case_ids),
                "selection_label_identity_hash": _label_surface_hash(selection_labels),
                "ordinary_model_hash": ordinary_model.model_hash,
                "G_static": global_selection.to_payload(),
                "S_static": support_selection.to_payload(),
            }
            static_rows.append(
                {**static_payload, "row_hash": canonical_hash(static_payload)}
            )

            calibration_labels = tuple(
                manager.open_calibration_labels(target, fold_ordinal)
            )
            calibration_index = _label_index(calibration_labels)
            n_positive = sum(label.value == 1 for label in calibration_labels)
            n_negative = sum(label.value == 0 for label in calibration_labels)
            # Calibration is a fold-level nuisance fit, not a property of the
            # selected challenger.  Fit one ordinary two-slope map from all
            # eight A1 actions and reuse that exact object for both static
            # selection regimes.  The blocked permutation control receives a
            # separate same-capacity all-action fit.
            calibration_actions = tuple(
                a1_action_id(source) for source in candidate_sources(target)
            )
            ordinary_calibration = _fit_fold_calibration(
                ordinary_model,
                calibration_actions,
                fold.calibration_case_ids,
                target,
                probability,
                features,
                calibration_index,
                n_positive,
                n_negative,
            )
            calibrations = {
                "F_G": ordinary_calibration,
                "F_S": ordinary_calibration,
                "F_P": _fit_fold_calibration(
                    permutation_model,
                    calibration_actions,
                    fold.calibration_case_ids,
                    target,
                    probability,
                    features,
                    calibration_index,
                    n_positive,
                    n_negative,
                ),
            }
            calibration_by_fold[(target, fold_ordinal)] = MappingProxyType(
                calibrations
            )
            calibration_payload = {
                "target_center": target,
                "fold_ordinal": fold_ordinal,
                "calibration_case_ids": list(fold.calibration_case_ids),
                "calibration_label_identity_hash": _label_surface_hash(
                    calibration_labels
                ),
                "calibration_n_positive": n_positive,
                "calibration_n_negative": n_negative,
                "calibration_action_ids": list(calibration_actions),
                "ordinary_calibration_shared_by_F_G_and_F_S": True,
                "permutation_calibration_same_capacity_all_actions": True,
                "F_G": calibrations["F_G"].to_payload(),
                "F_S": calibrations["F_S"].to_payload(),
                "F_P": calibrations["F_P"].to_payload(),
            }
            calibration_rows.append(
                {
                    **calibration_payload,
                    "row_hash": canonical_hash(calibration_payload),
                }
            )

            fold_decisions = _evaluation_fold_decisions(
                target=target,
                fold_ordinal=fold_ordinal,
                evaluation_case_ids=fold.evaluation_case_ids,
                ordinary_model=ordinary_model,
                permutation_model=permutation_model,
                global_selection=global_selection,
                support_selection=support_selection,
                calibrations=calibrations,
                features=features,
            )
            fold_payload = {
                "schema_version": "fixed_bank_flip_router_fold_decision_seal_v1",
                "target_center": target,
                "fold_ordinal": fold_ordinal,
                "evaluation_case_ids": list(fold.evaluation_case_ids),
                "ordinary_model_hash": ordinary_model.model_hash,
                "permutation_model_hash": permutation_model.model_hash,
                "static_row_hash": static_rows[-1]["row_hash"],
                "calibration_row_hash": calibration_rows[-1]["row_hash"],
                "decisions": [row.to_payload() for row in fold_decisions],
                "held_evaluation_labels_used": False,
            }
            fold_seal = canonical_hash(fold_payload)
            manager.record_fold_decision_seal(target, fold_ordinal, fold_seal)
            fold_seals[(target, fold_ordinal)] = fold_seal
            decisions.extend(fold_decisions)

    decision_payload = {
        "schema_version": "fixed_bank_flip_router_decision_bundle_v1",
        "decisions": [row.to_payload() for row in decisions],
        "fold_seals": {
            f"{key[0]}::{key[1]}": value
            for key, value in sorted(fold_seals.items())
        },
        "evaluation_labels_used": False,
    }
    bundle = DecisionBundle(
        tuple(decisions),
        fold_seals,
        canonical_hash(decision_payload),
    )
    static_unhashed = {
        "schema_version": "fixed_bank_flip_router_static_selection_seals_v1",
        "selection_count": len(static_rows),
        "rows": list(static_rows),
        "evaluation_labels_used": False,
    }
    calibration_unhashed = {
        "schema_version": "fixed_bank_flip_router_calibration_seals_v1",
        "calibration_count": len(calibration_rows),
        "rows": list(calibration_rows),
        "calibration_only_label_scope": True,
        "evaluation_labels_used": False,
    }
    return DecisionPhaseResult(
        static_rows=tuple(static_rows),
        calibration_rows=tuple(calibration_rows),
        static_seal_payload=MappingProxyType(
            {
                **static_unhashed,
                "static_selection_surface_hash": canonical_hash(static_unhashed),
            }
        ),
        calibration_seal_payload=MappingProxyType(
            {
                **calibration_unhashed,
                "calibration_surface_hash": canonical_hash(calibration_unhashed),
            }
        ),
        bundle=bundle,
        static_by_fold=MappingProxyType(static_by_fold),
        calibration_by_fold=MappingProxyType(calibration_by_fold),
    )

def _fit_fold_calibration(
    model: TwoHeadRidgeModel,
    action_ids: Sequence[str],
    case_ids: Sequence[str],
    target_center: str,
    probability: Mapping[tuple[str, str, str, str], object],
    features: Mapping[tuple[str, str, str], object],
    labels: Mapping[tuple[str, str, str], int],
    n_positive: int,
    n_negative: int,
) -> DirectionSharedCalibration:
    if not action_ids or n_positive <= 0 or n_negative <= 0:
        return fit_direction_shared_calibration(
            (),
            calibration_n_positive=n_positive,
            calibration_n_negative=n_negative,
        )
    rows = []
    for action_id in action_ids:
        for case_id in case_ids:
            feature = _core_feature(features[(target_center, case_id, action_id)])
            prediction = predict_two_head(model, feature)
            target = _case_contribution(
                probability,
                labels,
                target_center=target_center,
                case_id=case_id,
                action_id=action_id,
            )
            rows.append(
                build_calibration_row(
                    prediction=prediction,
                    features=feature,
                    target=target,
                    calibration_n_positive=n_positive,
                    calibration_n_negative=n_negative,
                )
            )
    return fit_direction_shared_calibration(
        rows,
        calibration_n_positive=n_positive,
        calibration_n_negative=n_negative,
    )

def _evaluation_fold_decisions(
    *,
    target: str,
    fold_ordinal: int,
    evaluation_case_ids: Sequence[str],
    ordinary_model: TwoHeadRidgeModel,
    permutation_model: TwoHeadRidgeModel,
    global_selection: StaticSelection,
    support_selection: StaticSelection,
    calibrations: Mapping[str, DirectionSharedCalibration],
    features: Mapping[tuple[str, str, str], object],
) -> tuple[MethodDecision, ...]:
    result: list[MethodDecision] = []
    for case_id in evaluation_case_ids:
        result.extend(
            (
                _static_method(target, fold_ordinal, case_id, "B", B_ACTION_ID),
                _static_method(target, fold_ordinal, case_id, "U", U_ACTION_ID),
                _static_selection_method(
                    target, fold_ordinal, case_id, "G_static", global_selection
                ),
                _static_selection_method(
                    target, fold_ordinal, case_id, "S_static", support_selection
                ),
                _flip_method(
                    target,
                    fold_ordinal,
                    case_id,
                    "F_G",
                    global_selection,
                    ordinary_model,
                    calibrations["F_G"],
                    features,
                ),
                _flip_method(
                    target,
                    fold_ordinal,
                    case_id,
                    "F_S",
                    support_selection,
                    ordinary_model,
                    calibrations["F_S"],
                    features,
                ),
                _flip_method(
                    target,
                    fold_ordinal,
                    case_id,
                    "F_P",
                    support_selection,
                    permutation_model,
                    calibrations["F_P"],
                    features,
                ),
            )
        )
    return tuple(result)


def _static_method(
    target: str, fold: int, case_id: str, method: str, action: str
) -> MethodDecision:
    return MethodDecision(
        target,
        fold,
        case_id,
        method,
        action,
        action,
        0.0,
        0.0,
        0.0,
        "fixed_control",
    )


def _static_selection_method(
    target: str,
    fold: int,
    case_id: str,
    method: str,
    selection: StaticSelection,
) -> MethodDecision:
    return MethodDecision(
        target,
        fold,
        case_id,
        method,
        selection.action_id,
        selection.action_id,
        selection.exact_gain,
        0.0,
        selection.exact_gain,
        "selection_support_static" if method == "S_static" else "loco_global_static",
    )


def _flip_method(
    target: str,
    fold: int,
    case_id: str,
    method: str,
    selection: StaticSelection,
    model: TwoHeadRidgeModel,
    calibration: DirectionSharedCalibration,
    features: Mapping[tuple[str, str, str], object],
) -> MethodDecision:
    if selection.action_id == B_ACTION_ID:
        return MethodDecision(
            target,
            fold,
            case_id,
            method,
            B_ACTION_ID,
            B_ACTION_ID,
            0.0,
            0.0,
            0.0,
            "static_fallback_B",
        )
    feature = _core_feature(features[(target, case_id, selection.action_id)])
    core_decision = select_case_action(
        method_id=method,
        challenger=selection,
        features=feature,
        prediction=predict_two_head(model, feature),
        calibration=calibration,
    )
    return MethodDecision(
        target,
        fold,
        case_id,
        method,
        core_decision.selected_action_id,
        core_decision.challenger_action_id,
        core_decision.predicted_gain,
        core_decision.standard_error,
        core_decision.lower_confidence_bound,
        core_decision.reason,
    )
