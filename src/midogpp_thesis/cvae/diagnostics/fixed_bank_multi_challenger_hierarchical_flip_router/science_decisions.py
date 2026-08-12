"""Three-role menu, calibration, and pre-evaluation decision sealing."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.hierarchical_multi_challenger import (
    CalibrationObservation,
    DirectionalCalibration,
    baseline_action_score,
    build_calibration_observation,
    build_candidate_menu,
    fit_direction_calibration,
    predict_direction,
    score_action_against_baseline,
    select_action_with_margin,
)
from ...routing.threshold_flip_case_router import (
    DirectionSharedCalibration,
    StaticSelection,
    build_calibration_row,
    fit_direction_shared_calibration,
    predict_two_head,
    select_case_action,
)
from .constants import (
    B_ACTION_ID,
    CENTERS,
    U_ACTION_ID,
    a1_action_id,
    candidate_sources,
)
from .hashing import canonical_hash
from .products import (
    DonorPhaseResult,
    FoldPhaseResult,
    MethodDecision,
    semantic_decision_payload,
)
from .science_common import (
    case_contribution,
    core_feature,
    direction_counts,
    feature_index,
    label_index,
    label_surface_hash,
    probability_index,
)
from .semantic_payloads import (
    calibration_observation_semantic_payload,
    calibration_semantic_hash,
    directional_calibration_semantic_payload,
    score_semantic_payload,
)


def build_fold_decision_phase(
    *,
    probability_surface: object,
    prelabel: object,
    partition: object,
    manager: object,
    donor_phase: DonorPhaseResult,
    config: object,
) -> FoldPhaseResult:
    """Seal B/U/S/F-single/G/R/P decisions for all 45 held folds."""

    _assert_decision_config(config)
    probability = probability_index(probability_surface)
    features = feature_index(prelabel)
    menu_rows: list[Mapping[str, object]] = []
    calibration_rows: list[Mapping[str, object]] = []
    score_rows: list[Mapping[str, object]] = []
    decisions: list[MethodDecision] = []
    fold_seals: dict[tuple[str, int], str] = {}
    menus = {}
    calibrations_by_fold = {}
    partition_seed = int(getattr(config, "protocol")["partition_seed"])

    for target in CENTERS:
        family_models = donor_phase.models_by_target_family[target]
        single_model = donor_phase.single_models_by_target[target]
        for fold_ordinal in range(5):
            fold = partition.fold(target, fold_ordinal)
            selection_labels = tuple(
                manager.open_selection_labels(target, fold_ordinal)
            )
            selection_index = label_index(selection_labels)
            selection_targets = {
                a1_action_id(source): tuple(
                    case_contribution(
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
            menu = build_candidate_menu(selection_targets)
            menus[(target, fold_ordinal)] = menu
            menu_unhashed = {
                "target_center": target,
                "fold_ordinal": fold_ordinal,
                "selection_case_ids": list(fold.selection_case_ids),
                "selection_label_identity_hash": label_surface_hash(selection_labels),
                "all_eight_fixed_B_referenced_scores": [
                    row.to_payload() for row in menu.ranked_support_actions
                ],
                "menu": menu.to_payload(),
                "held_evaluation_labels_used": False,
            }
            menu_row = {**menu_unhashed, "row_hash": canonical_hash(menu_unhashed)}
            menu_rows.append(menu_row)

            calibration_labels = tuple(
                manager.open_calibration_labels(target, fold_ordinal)
            )
            calibration_index = label_index(calibration_labels)
            n_positive = sum(int(row.value) == 1 for row in calibration_labels)
            n_negative = sum(int(row.value) == 0 for row in calibration_labels)
            family_calibrations: dict[
                str, Mapping[str, DirectionalCalibration]
            ] = {}
            family_observation_hashes: dict[str, str] = {}
            for family in ("G", "R", "P"):
                feature_case_map = _case_derangement(
                    fold.calibration_case_ids,
                    seed=partition_seed
                    + 1000 * int(target)
                    + 100 * fold_ordinal
                    + (1 if family == "P" else 0),
                    active=family == "P",
                )
                observations = _calibration_observations(
                    target=target,
                    case_ids=fold.calibration_case_ids,
                    feature_case_map=feature_case_map,
                    action_ids=tuple(
                        action for action in menu.action_ids if action != B_ACTION_ID
                    ),
                    models=family_models[family],
                    probability=probability,
                    features=features,
                    labels=calibration_index,
                )
                family_observation_hashes[family] = canonical_hash(
                    [calibration_observation_semantic_payload(row) for row in observations]
                )
                calibrated = {
                    direction: fit_direction_calibration(
                        tuple(
                            row for row in observations if row.direction == direction
                        ),
                        direction=direction,
                        menu_hash=menu.menu_hash,
                    )
                    for direction in ("0to1", "1to0")
                }
                family_calibrations[family] = MappingProxyType(calibrated)
            calibrations_by_fold[(target, fold_ordinal)] = MappingProxyType(
                family_calibrations
            )
            single_calibration = _fit_single_calibration(
                model=single_model,
                target=target,
                case_ids=fold.calibration_case_ids,
                action_ids=tuple(
                    a1_action_id(source) for source in candidate_sources(target)
                ),
                probability=probability,
                features=features,
                labels=calibration_index,
                n_positive=n_positive,
                n_negative=n_negative,
            )
            calibration_unhashed = {
                "target_center": target,
                "fold_ordinal": fold_ordinal,
                "calibration_case_ids": list(fold.calibration_case_ids),
                "calibration_label_identity_hash": label_surface_hash(
                    calibration_labels
                ),
                "calibration_n_positive": n_positive,
                "calibration_n_negative": n_negative,
                "menu_hash": menu.menu_hash,
                "menu_action_ids": list(menu.action_ids),
                "donor_composite_model_hash": donor_phase.model_seals[target][
                    "composite_model_hash"
                ],
                "observation_hashes_by_family": family_observation_hashes,
                "family_calibrations": {
                    family: {
                        direction: row.to_payload()
                        for direction, row in sorted(by_direction.items())
                    }
                    for family, by_direction in sorted(
                        family_calibrations.items()
                    )
                },
                "single_challenger_calibration": single_calibration.to_payload(),
                "shared_donor_models_updated_with_target_labels": False,
                "held_evaluation_labels_used": False,
            }
            calibration_row = {
                **calibration_unhashed,
                "row_hash": calibration_semantic_hash(calibration_unhashed),
            }
            calibration_rows.append(calibration_row)

            fold_scores: list[Mapping[str, object]] = []
            fold_decisions: list[MethodDecision] = []
            evaluation_permutation = _case_derangement(
                fold.evaluation_case_ids,
                seed=partition_seed + 10_000 * int(target) + fold_ordinal,
                active=True,
            )
            for case_id in fold.evaluation_case_ids:
                fold_decisions.extend(
                    _static_decisions(
                        target=target,
                        fold_ordinal=fold_ordinal,
                        case_id=case_id,
                        menu=menu,
                    )
                )
                fold_decisions.append(
                    _single_decision(
                        target=target,
                        fold_ordinal=fold_ordinal,
                        case_id=case_id,
                        menu=menu,
                        model=single_model,
                        calibration=single_calibration,
                        features=features,
                    )
                )
                for family, method_id in (
                    ("G", "G_multi"),
                    ("R", "R_multi"),
                    ("P", "P_multi"),
                ):
                    feature_case_id = (
                        evaluation_permutation[case_id]
                        if family == "P"
                        else case_id
                    )
                    method_decision, case_score_rows = _multi_decision(
                        target=target,
                        fold_ordinal=fold_ordinal,
                        case_id=case_id,
                        feature_case_id=feature_case_id,
                        method_id=method_id,
                        family=family,
                        menu=menu,
                        models=family_models[family],
                        calibrations=family_calibrations[family],
                        features=features,
                        n_positive=n_positive,
                        n_negative=n_negative,
                    )
                    fold_decisions.append(method_decision)
                    fold_scores.extend(case_score_rows)
            fold_scores = sorted(
                fold_scores,
                key=lambda row: (
                    str(row["case_id"]),
                    str(row["method_id"]),
                    str(row["action_id"]),
                ),
            )
            score_rows.extend(fold_scores)
            fold_decisions = sorted(
                fold_decisions,
                key=lambda row: (row.case_id, row.method_id),
            )
            fold_unhashed = {
                "schema_version": "fixed_bank_multi_challenger_fold_decision_seal_v1",
                "target_center": target,
                "fold_ordinal": fold_ordinal,
                "evaluation_case_ids": list(fold.evaluation_case_ids),
                "menu_row_hash": menu_row["row_hash"],
                "calibration_row_hash": calibration_row["row_hash"],
                "score_surface_hash": canonical_hash(
                    [score_semantic_payload(row) for row in fold_scores]
                ),
                "decisions": [
                    semantic_decision_payload(row) for row in fold_decisions
                ],
                "held_evaluation_labels_used": False,
            }
            fold_seal = canonical_hash(fold_unhashed)
            manager.record_fold_decision_seal(target, fold_ordinal, fold_seal)
            fold_seals[(target, fold_ordinal)] = fold_seal
            decisions.extend(fold_decisions)

    decisions = sorted(
        decisions,
        key=lambda row: (
            row.target_center,
            row.fold_ordinal,
            row.case_id,
            row.method_id,
        ),
    )
    decision_payload = {
        "schema_version": "fixed_bank_multi_challenger_decision_bundle_v1",
        "decisions": [semantic_decision_payload(row) for row in decisions],
        "fold_seals": {
            f"{key[0]}::{key[1]}": value
            for key, value in sorted(fold_seals.items())
        },
        "evaluation_labels_used": False,
    }
    return FoldPhaseResult(
        menu_rows=tuple(menu_rows),
        calibration_rows=tuple(calibration_rows),
        score_rows=tuple(score_rows),
        decisions=tuple(decisions),
        fold_seal_hashes=fold_seals,
        menu_by_fold=MappingProxyType(menus),
        calibrations_by_fold_family=MappingProxyType(calibrations_by_fold),
        decision_bundle_hash=canonical_hash(decision_payload),
    )


def _calibration_observations(
    *,
    target: str,
    case_ids: Sequence[str],
    feature_case_map: Mapping[str, str],
    action_ids: Sequence[str],
    models: Mapping[str, object],
    probability: Mapping[tuple[str, str, str, str], object],
    features: Mapping[tuple[str, str, str], object],
    labels: Mapping[tuple[str, str, str], int],
) -> tuple[CalibrationObservation, ...]:
    rows: list[CalibrationObservation] = []
    for case_id in case_ids:
        for action_id in action_ids:
            feature = core_feature(
                features[(target, feature_case_map[case_id], action_id)]
            )
            counts = direction_counts(
                probability,
                labels,
                target_center=target,
                case_id=case_id,
                action_id=action_id,
            )
            for direction in ("0to1", "1to0"):
                success_count, trial_count = counts[direction]
                prediction = predict_direction(
                    models[direction],
                    candidate_source=_source_from_action(action_id),
                    feature_names=feature.feature_names,
                    values=feature.values,
                )
                rows.append(
                    build_calibration_observation(
                        case_id=case_id,
                        action_id=action_id,
                        direction=direction,
                        success_count=success_count,
                        trial_count=trial_count,
                        prediction=prediction,
                    )
                )
    return tuple(rows)


def _fit_single_calibration(
    *,
    model: object,
    target: str,
    case_ids: Sequence[str],
    action_ids: Sequence[str],
    probability: Mapping[tuple[str, str, str, str], object],
    features: Mapping[tuple[str, str, str], object],
    labels: Mapping[tuple[str, str, str], int],
    n_positive: int,
    n_negative: int,
) -> DirectionSharedCalibration:
    if n_positive <= 0 or n_negative <= 0:
        return fit_direction_shared_calibration(
            (),
            calibration_n_positive=n_positive,
            calibration_n_negative=n_negative,
        )
    rows = []
    for action_id in action_ids:
        for case_id in case_ids:
            feature = core_feature(features[(target, case_id, action_id)])
            rows.append(
                build_calibration_row(
                    prediction=predict_two_head(model, feature),
                    features=feature,
                    target=case_contribution(
                        probability,
                        labels,
                        target_center=target,
                        case_id=case_id,
                        action_id=action_id,
                    ),
                    calibration_n_positive=n_positive,
                    calibration_n_negative=n_negative,
                )
            )
    return fit_direction_shared_calibration(
        rows,
        calibration_n_positive=n_positive,
        calibration_n_negative=n_negative,
    )


def _static_decisions(
    *, target: str, fold_ordinal: int, case_id: str, menu: object
) -> tuple[MethodDecision, ...]:
    anchor_score = next(
        (
            row.exact_gain
            for row in menu.ranked_support_actions
            if row.action_id == menu.anchor_action_id
        ),
        0.0,
    )
    return (
        _fixed_decision(target, fold_ordinal, case_id, "B", "B", menu.menu_hash),
        _fixed_decision(target, fold_ordinal, case_id, "U", "U", menu.menu_hash),
        MethodDecision(
            target,
            fold_ordinal,
            case_id,
            "S_static",
            menu.anchor_action_id,
            menu.anchor_action_id,
            menu.anchor_action_id,
            "B",
            float(anchor_score),
            0.0,
            0.0,
            0.0,
            0.0,
            float(anchor_score),
            "selection_support_static",
            menu.menu_hash,
        ),
    )


def _fixed_decision(
    target: str,
    fold_ordinal: int,
    case_id: str,
    method_id: str,
    action_id: str,
    menu_hash: str,
) -> MethodDecision:
    return MethodDecision(
        target,
        fold_ordinal,
        case_id,
        method_id,
        action_id,
        action_id,
        action_id,
        action_id,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        "fixed_control",
        menu_hash,
    )


def _single_decision(
    *,
    target: str,
    fold_ordinal: int,
    case_id: str,
    menu: object,
    model: object,
    calibration: DirectionSharedCalibration,
    features: Mapping[tuple[str, str, str], object],
) -> MethodDecision:
    anchor = str(menu.anchor_action_id)
    if anchor == B_ACTION_ID:
        return _fixed_decision(
            target, fold_ordinal, case_id, "F_single", B_ACTION_ID, menu.menu_hash
        )
    feature = core_feature(features[(target, case_id, anchor)])
    core = select_case_action(
        method_id="F_single",
        challenger=StaticSelection(anchor, 0.0, 0.0, False),
        features=feature,
        prediction=predict_two_head(model, feature),
        calibration=calibration,
    )
    return MethodDecision(
        target,
        fold_ordinal,
        case_id,
        "F_single",
        core.selected_action_id,
        anchor,
        anchor if core.predicted_gain >= 0.0 else B_ACTION_ID,
        B_ACTION_ID if core.predicted_gain >= 0.0 else anchor,
        core.predicted_gain,
        core.predicted_gain,
        core.standard_error,
        0.0,
        core.standard_error,
        core.lower_confidence_bound,
        core.reason,
        menu.menu_hash,
    )


def _multi_decision(
    *,
    target: str,
    fold_ordinal: int,
    case_id: str,
    feature_case_id: str,
    method_id: str,
    family: str,
    menu: object,
    models: Mapping[str, object],
    calibrations: Mapping[str, DirectionalCalibration],
    features: Mapping[tuple[str, str, str], object],
    n_positive: int,
    n_negative: int,
) -> tuple[MethodDecision, tuple[Mapping[str, object], ...]]:
    scores = [baseline_action_score(models=models)]
    for action_id in menu.action_ids:
        if action_id == B_ACTION_ID:
            continue
        feature = core_feature(features[(target, feature_case_id, action_id)])
        predictions = {
            direction: predict_direction(
                models[direction],
                candidate_source=_source_from_action(action_id),
                feature_names=feature.feature_names,
                values=feature.values,
            )
            for direction in ("0to1", "1to0")
        }
        if (
            n_positive <= 0
            or n_negative <= 0
            or not all(row.valid for row in calibrations.values())
        ):
            score = baseline_action_score(models=models)
            score = score.__class__(
                action_id=action_id,
                expected_gain=0.0,
                epistemic_variance=0.0,
                calibration_variance=0.0,
                model_gradients=score.model_gradients,
                calibration_gradients=score.calibration_gradients,
            )
        else:
            score = score_action_against_baseline(
                action_id=action_id,
                predictions=predictions,
                models=models,
                calibrations=calibrations,
                flip_counts={
                    "0to1": feature.flip_0to1_count,
                    "1to0": feature.flip_1to0_count,
                },
                n_positive=n_positive,
                n_negative=n_negative,
            )
        scores.append(score)
    core = select_action_with_margin(
        case_id=case_id,
        method_id=method_id,
        menu=menu,
        scores=scores,
        models=models,
        calibrations=calibrations,
    )
    decision = MethodDecision(
        target,
        fold_ordinal,
        case_id,
        method_id,
        core.selected_action_id,
        core.anchor_action_id,
        core.best_action_id,
        core.runner_up_action_id,
        core.predicted_gain,
        core.action_margin,
        core.epistemic_standard_error,
        core.calibration_standard_error,
        core.margin_standard_error,
        core.margin_lcb,
        core.reason,
        menu.menu_hash,
    )
    rows = []
    for rank, score in enumerate(
        sorted(scores, key=lambda row: (-row.expected_gain, row.action_id)),
        start=1,
    ):
        payload = {
            "target_center": target,
            "fold_ordinal": fold_ordinal,
            "case_id": case_id,
            "feature_case_id": feature_case_id,
            "method_id": method_id,
            "model_family": family,
            "action_id": score.action_id,
            "rank": rank,
            **score.to_payload(),
            "menu_hash": menu.menu_hash,
            "evaluation_labels_used": False,
        }
        rows.append(
            {
                **payload,
                "row_hash": canonical_hash(score_semantic_payload(payload)),
            }
        )
    return decision, tuple(rows)


def _case_derangement(
    case_ids: Sequence[str], *, seed: int, active: bool
) -> Mapping[str, str]:
    import random

    cases = tuple(sorted(str(value) for value in case_ids))
    if not active:
        return MappingProxyType({case: case for case in cases})
    if len(cases) < 2:
        raise ProtocolError("Permutation inference requires two cases.")
    shuffled = list(cases)
    random.Random(int(seed)).shuffle(shuffled)
    rotated = shuffled[1:] + shuffled[:1]
    mapping = dict(zip(shuffled, rotated, strict=True))
    if any(key == value for key, value in mapping.items()):
        raise ProtocolError("Permutation inference failed to derange cases.")
    return MappingProxyType(mapping)


def _source_from_action(action_id: str) -> str:
    prefix = "A1::source="
    value = str(action_id)
    if not value.startswith(prefix) or not value[len(prefix) :]:
        raise ProtocolError("Multi-challenger source action identity drifted.")
    return value[len(prefix) :]


# Backward-compatible private aliases for existing focused tests.  Producers
# and validators import the public versioned contract module directly.
_semantic_score_payload = score_semantic_payload
_calibration_semantic_hash = calibration_semantic_hash
_semantic_directional_calibration = directional_calibration_semantic_payload
_semantic_calibration_observation = calibration_observation_semantic_payload


def _assert_decision_config(config: object) -> None:
    routing = getattr(config, "routing")
    if (
        int(routing.get("candidate_menu_top_k", -1)) != 3
        or float(routing.get("support_prior_cases", -1.0)) != 8.0
        or float(routing.get("calibration_alpha", -1.0)) != 4.0
        or float(routing.get("action_margin_z", -1.0)) != 1.96
        or routing.get("support_ranking_reference") != "B"
    ):
        raise ProtocolError("Multi-challenger decision config drifted.")


__all__ = ("build_fold_decision_phase",)
