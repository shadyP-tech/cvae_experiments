"""Genuine held-L residual and source-ordering assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Sequence

from ....protocol import ProtocolError
from ....routing.pairwise_primitive_utility import (
    OOFResidualObservation,
    P_ACTION_ID,
    UncertaintyCalibration,
    calibrate_clustered_uncertainty,
    canonical_sha256,
)
from ..folds import OuterFoldPlanV4
from .admission import HeldLSourceOrderingCase
from .pool_indexed_pairwise_fit import HeldLActionPrediction
from .source_products import DerivedSourceScienceProducts


@dataclass(frozen=True, slots=True)
class HeldLCalibrationProducts:
    residual_observations: tuple[OOFResidualObservation, ...]
    unsupported_component_keys: tuple[tuple[str, str, str], ...]
    ordering_cases: tuple[HeldLSourceOrderingCase, ...]
    uncertainty_calibration: UncertaintyCalibration
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        unsupported = tuple(sorted(tuple(value) for value in self.unsupported_component_keys))
        if not self.residual_observations or not self.ordering_cases or not isinstance(self.uncertainty_calibration, UncertaintyCalibration) or len(set(unsupported)) != len(unsupported):
            raise ProtocolError("OE-PPUR v4 held-L calibration products are incomplete.")
        object.__setattr__(self, "unsupported_component_keys", unsupported)
        object.__setattr__(self, "receipt_hash", canonical_sha256({
            "schema": "oe_ppur_v4_genuine_held_L_calibration_products_v1",
            "residual_inventory": tuple((row.center_id, row.case_id, row.action_id, row.comparator_id, row.metric, row.source_scope_receipt_hash) for row in self.residual_observations),
            "ordering_case_inventory": tuple((row.center_id, row.case_id, row.active_representative_ids, row.held_l_scope_receipt_hash, row.held_l_model_hash) for row in self.ordering_cases),
            "uncertainty_calibration_hash": self.uncertainty_calibration.calibration_hash,
            "unsupported_component_keys": unsupported,
            "unsupported_component_policy": "PER_CASE_EXACT_P_NO_POOLING_FALLBACK",
            "genuine_held_L": True,
            "target_labels_used": False,
        }))


def build_genuine_held_l_calibration(
    products: DerivedSourceScienceProducts,
    predictions: Sequence[HeldLActionPrediction],
    *,
    plan: OuterFoldPlanV4,
) -> HeldLCalibrationProducts:
    """Build action/contrast residuals only for active unique representatives."""

    if not isinstance(products, DerivedSourceScienceProducts) or not isinstance(plan, OuterFoldPlanV4):
        raise ProtocolError("OE-PPUR v4 held-L calibration requires derived source products.")
    prediction_rows = tuple(sorted(tuple(predictions), key=lambda row: (row.center_id, row.case_id, row.action_id)))
    prediction_by_key = {(row.center_id, row.case_id, row.action_id): row for row in prediction_rows}
    expected_keys = {
        (case.center_id, case.case_id, action)
        for case in products.cases
        for action in case.opportunity_receipt.active_representative_ids
    }
    if set(prediction_by_key) != expected_keys or len(prediction_by_key) != len(prediction_rows):
        raise ProtocolError("OE-PPUR v4 held-L predictions do not exactly cover active opportunities.")
    scope_by_l = {scope.L: scope for scope in plan.scopes}
    if len(scope_by_l) != len(plan.scopes):
        raise ProtocolError("OE-PPUR v4 calibration scopes do not rotate L exactly once.")
    model_scope_by_center: dict[str, tuple[str, str]] = {}
    for row in prediction_rows:
        current = (row.source_scope_receipt_hash, row.model_hash)
        previous = model_scope_by_center.setdefault(row.center_id, current)
        if previous != current or row.source_scope_receipt_hash != scope_by_l[row.center_id].receipt_hash:
            raise ProtocolError("OE-PPUR v4 held-L predictions mixed source models/scopes.")
    residuals: list[OOFResidualObservation] = []
    ordering_cases: list[HeldLSourceOrderingCase] = []
    for case in products.cases:
        active = case.opportunity_receipt.active_representative_ids
        scope_model = model_scope_by_center.get(case.center_id)
        if scope_model is None:
            raise ProtocolError("OE-PPUR v4 source center has no calibratable active case.")
        scope_hash, model_hash = scope_model
        predicted = tuple((action, prediction_by_key[(case.center_id, case.case_id, action)].predicted_score) for action in active)
        realized = tuple((action, case.realized_utility(action).bacc_gain) for action in active)
        ordering_cases.append(HeldLSourceOrderingCase(
            center_id=case.center_id,
            case_id=case.case_id,
            predicted_scores=predicted,
            realized_bacc_gains=realized,
            active_representative_ids=active,
            candidate_pool_receipt_hash=case.candidate_pool_receipt_hash,
            held_l_scope_receipt_hash=scope_hash,
            held_l_model_hash=model_hash,
        ))
        for action in active:
            utility = case.utility(action)
            observed = case.realized_utility(action)
            score = prediction_by_key[(case.center_id, case.case_id, action)].predicted_score
            for metric, predicted_value, observed_value in (
                ("bacc", utility.bacc_gain, observed.bacc_gain),
                ("brier", utility.brier_loss_delta, observed.brier_loss_delta),
                ("log", utility.log_loss_delta, observed.log_loss_delta),
            ):
                residuals.append(OOFResidualObservation(
                    center_id=case.center_id,
                    case_id=case.case_id,
                    oof_held_center=case.center_id,
                    action_id=action,
                    comparator_id=P_ACTION_ID,
                    metric=metric,
                    predicted=predicted_value,
                    observed=observed_value,
                    source_scope_receipt_hash=scope_hash,
                ))
            comparators = (P_ACTION_ID, *(candidate for candidate in active if candidate != action))
            for comparator in comparators:
                comparator_score = 0.0 if comparator == P_ACTION_ID else prediction_by_key[(case.center_id, case.case_id, comparator)].predicted_score
                comparator_realized = 0.0 if comparator == P_ACTION_ID else case.realized_utility(comparator).bacc_gain
                residuals.append(OOFResidualObservation(
                    center_id=case.center_id,
                    case_id=case.case_id,
                    oof_held_center=case.center_id,
                    action_id=action,
                    comparator_id=comparator,
                    metric="pairwise",
                    predicted=score - comparator_score,
                    observed=observed.bacc_gain - comparator_realized,
                    source_scope_receipt_hash=scope_hash,
                ))
    grouped_residuals: dict[tuple[str, str, str], list[OOFResidualObservation]] = defaultdict(list)
    for row in residuals:
        grouped_residuals[(row.action_id, row.comparator_id, row.metric)].append(row)
    expected_l = {scope.L for scope in plan.scopes}
    supported_keys = {
        key
        for key, values in grouped_residuals.items()
        if {row.center_id for row in values} == expected_l
    }
    unsupported_keys = tuple(sorted(set(grouped_residuals) - supported_keys))
    supported_residuals = tuple(
        sorted(
            (row for key in supported_keys for row in grouped_residuals[key]),
            key=lambda row: (row.action_id, row.comparator_id, row.metric, row.center_id, row.case_id),
        )
    )
    if not supported_residuals:
        raise ProtocolError("OE-PPUR v4 uncertainty has no all-L supported component.")
    calibration = calibrate_clustered_uncertainty(supported_residuals, calibration_scopes=plan.neutral_scopes)
    ordered_cases = tuple(sorted(ordering_cases, key=lambda row: (row.center_id, row.case_id)))
    if {(row.center_id, row.case_id) for row in ordered_cases} != set(plan.source_case_inventory):
        raise ProtocolError("OE-PPUR v4 held-L admission omitted or invented source cases.")
    return HeldLCalibrationProducts(
        supported_residuals,
        unsupported_keys,
        ordered_cases,
        calibration,
    )


__all__ = ("HeldLCalibrationProducts", "build_genuine_held_l_calibration")
