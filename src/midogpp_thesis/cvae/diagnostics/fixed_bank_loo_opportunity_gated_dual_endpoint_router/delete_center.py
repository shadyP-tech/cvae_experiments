"""Full terminal delete-one-center recomputation of G, I, R, and portfolio."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ...protocol import ProtocolError
from .composition import compose_portfolio_predictions
from .constants import CENTERS
from .identification_products import CaseIdentificationDecision
from .identification_reselection import reselect_identification_for_priors
from .prediction_products import MethodPrediction
from .predictions import (
    compose_identification_predictions,
    compose_physical_action_predictions,
    compose_robust_predictions,
)
from .response_products import BinaryLabel, CaseActionConfusion, DirectionalGain
from .robust import select_robust_arm_decisions
from .sensitivity_prior import compute_delete_center_donor_priors
from .split_plans import WholeCaseLooPlan


def _equal_center_bacc(
    predictions: Sequence[MethodPrediction],
    labels: Sequence[BinaryLabel],
    *,
    centers: Sequence[str],
) -> float:
    truth = {row.key: row.value for row in labels if row.target_center in centers}
    predicted = {row.key: row.hard_prediction for row in predictions if row.target_center in centers}
    if set(truth) != set(predicted):
        raise ProtocolError("OGDE delete-center predictions and labels are unaligned.")
    values: list[float] = []
    for center in centers:
        keys = tuple(key for key in truth if key[0] == center)
        y = np.asarray([truth[key] for key in keys], dtype=np.int8)
        p = np.asarray([predicted[key] for key in keys], dtype=np.int8)
        positive, negative = y == 1, y == 0
        if not bool(np.any(positive)) or not bool(np.any(negative)):
            raise ProtocolError("OGDE delete-center utility requires both classes per center.")
        values.append(
            0.5
            * (
                float(np.mean(p[positive] == 1, dtype=np.float64))
                + float(np.mean(p[negative] == 0, dtype=np.float64))
            )
        )
    return float(np.mean(values, dtype=np.float64))


def full_pipeline_delete_one_center(
    *,
    surface: object,
    plans: Sequence[WholeCaseLooPlan],
    directional_support_gains: Sequence[DirectionalGain],
    identification_decisions: Sequence[CaseIdentificationDecision],
    terminal_confusions: Sequence[CaseActionConfusion],
    terminal_labels: Sequence[BinaryLabel],
) -> tuple[dict[str, object], ...]:
    primary = {
        (row.target_center, row.case_id): row
        for row in identification_decisions
        if row.method_id == "I_OPPORTUNITY_GATED"
    }
    gains: dict[tuple[str, str], list[DirectionalGain]] = {}
    for row in directional_support_gains:
        if row.excluded_case_id is None:
            raise ProtocolError("OGDE delete-center support gain lacks route case identity.")
        gains.setdefault((row.query_center, row.excluded_case_id), []).append(row)
    if len(primary) != len(plans) or set(primary) != {plan.key for plan in plans} or set(gains) != set(primary):
        raise ProtocolError("OGDE delete-center route inputs lack all 218 cases.")
    baseline = compose_physical_action_predictions(surface, action_id="B")
    output: list[dict[str, object]] = []
    for deleted in CENTERS:
        retained_centers = tuple(center for center in CENTERS if center != deleted)
        priors_by_target = {
            target: compute_delete_center_donor_priors(
                terminal_confusions,
                heldout_center=target,
                deleted_query_center=deleted,
            )
            for target in retained_centers
        }
        identification: list[CaseIdentificationDecision] = []
        robust = []
        for plan in plans:
            if plan.target_center == deleted:
                continue
            priors = priors_by_target[plan.target_center]
            identification.append(
                reselect_identification_for_priors(primary[plan.key], priors)
            )
            robust.extend(
                select_robust_arm_decisions(
                    plan,
                    gains[plan.key],
                    priors,  # type: ignore[arg-type]
                )
            )
        i_predictions = compose_identification_predictions(surface, identification)
        r_predictions = compose_robust_predictions(surface, robust)
        portfolio = compose_portfolio_predictions(i_predictions, r_predictions)
        bacc_b = _equal_center_bacc(baseline, terminal_labels, centers=retained_centers)
        bacc_portfolio = _equal_center_bacc(portfolio, terminal_labels, centers=retained_centers)
        output.append(
            {
                "deleted_center": deleted,
                "retained_center_count": len(retained_centers),
                "B_equal_center_bacc": bacc_b,
                "OGDE_PORTFOLIO_equal_center_bacc": bacc_portfolio,
                "gain_over_B": bacc_portfolio - bacc_b,
                "deleted_center_removed_from_evaluation": True,
                "deleted_center_removed_from_all_G_query_contributions": True,
                "fixed_expert_bank_preserved": True,
                "G_recomputed": True,
                "normalization_recomputed": True,
                "identification_reselected": True,
                "robust_nine_arms_reselected": True,
                "portfolio_recomposed": True,
                "terminal_sensitivity_only": True,
            }
        )
    return tuple(output)


__all__ = ("full_pipeline_delete_one_center",)
