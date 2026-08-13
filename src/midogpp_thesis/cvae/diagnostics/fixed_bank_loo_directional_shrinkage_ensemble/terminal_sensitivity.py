"""Whole-pipeline center deletion, arm ablation, and descriptive inference."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np
from scipy.stats import t as student_t

from ...protocol import ProtocolError
from .constants import (
    ARM_IDS,
    B_ACTION_ID,
    CENTERS,
    U_ACTION_ID,
    a1_action_id,
    physical_action_ids,
)
from .decisions import select_arm_decisions
from .loo_plans import WholeCaseLooPlan
from .probability_surfaces import ProbabilityIndex, hard_prediction
from .products import (
    CaseActionConfusion,
    CaseArmDecision,
    CaseMethodConfusion,
    DirectionalGain,
    DonorPrior,
    EqualCenterContrast,
)
from .scoring import pooled_bacc, score_loo_directional_gains
from .terminal_scoring import terminal_truth


@dataclass(frozen=True)
class ReducedDonorPrior:
    """A delete-center G value without weakening the canonical 7-q product."""

    heldout_center: str
    source: str
    direction: str
    exact: Fraction


def canonical_case_action_counts(
    rows: Sequence[CaseActionConfusion],
) -> dict[tuple[str, str, str], CaseActionConfusion]:
    result: dict[tuple[str, str, str], CaseActionConfusion] = {}
    for row in rows:
        key = (row.target_center, row.case_id, row.action_id)
        previous = result.setdefault(key, row)
        if previous != row:
            raise ProtocolError("DCSE repeated route support counts disagree.")
    return result


def support_gains_by_route(
    plans: Sequence[WholeCaseLooPlan], rows: Sequence[CaseActionConfusion]
) -> dict[tuple[str, str], tuple[DirectionalGain, ...]]:
    canonical = canonical_case_action_counts(rows)
    output: dict[tuple[str, str], tuple[DirectionalGain, ...]] = {}
    for plan in plans:
        scoped = tuple(
            canonical[(plan.target_center, case_id, action)]
            for case_id in plan.support_case_ids
            for action in physical_action_ids(plan.target_center)
        )
        output[plan.key] = score_loo_directional_gains(scoped, plan)
    return output


def _reduced_priors(
    rows: Sequence[DonorPrior], *, deleted_center: str
) -> dict[str, tuple[ReducedDonorPrior, ...]]:
    output: dict[str, tuple[ReducedDonorPrior, ...]] = {}
    for target in CENTERS:
        if target == deleted_center:
            continue
        values = []
        for row in rows:
            if row.heldout_center != target:
                continue
            legal = tuple(
                gain
                for gain in row.query_gains
                if gain.query_center != deleted_center
            )
            if not legal:
                raise ProtocolError(
                    "DCSE delete-center recomputation removed every donor query."
                )
            exact = sum((gain.exact for gain in legal), Fraction(0)) / len(legal)
            values.append(
                ReducedDonorPrior(target, row.source, row.direction, exact)
            )
        if len(values) != 16:
            raise ProtocolError("DCSE reduced donor prior surface is incomplete.")
        output[target] = tuple(values)
    return output


def _score_decision_subset(
    probability_surface: object,
    decisions: Sequence[CaseArmDecision],
    terminal_labels: Sequence[object],
    *,
    method_id: str,
) -> tuple[CaseMethodConfusion, ...]:
    """Score an arm subset directly, avoiding transient hashed predictions."""

    truth = terminal_truth(terminal_labels)
    index = ProbabilityIndex(
        tuple(getattr(probability_surface, "rows", probability_surface))
    )
    grouped: dict[tuple[str, str], list[CaseArmDecision]] = defaultdict(list)
    for row in decisions:
        grouped[(row.target_center, row.case_id)].append(row)
    output: list[CaseMethodConfusion] = []
    for (target, case_id), case_rows in sorted(
        grouped.items(),
        key=lambda item: (CENTERS.index(item[0][0]), item[0][1]),
    ):
        ordered = tuple(
            sorted(case_rows, key=lambda row: ARM_IDS.index(row.arm_id))
        )
        baseline_rows = index.rows_for_case_action(
            target, case_id, B_ACTION_ID
        )
        labels: list[int] = []
        predictions: list[int] = []
        for baseline_row in baseline_rows:
            key = (target, case_id, baseline_row.sample_id)
            if key not in truth:
                raise ProtocolError(
                    "DCSE arm-subset sensitivity lacks a terminal label."
                )
            baseline = float(baseline_row.probability_mean)
            branch = hard_prediction(baseline)
            selected = tuple(
                row.decision_for_baseline_class(branch).selected_source
                for row in ordered
            )
            values = np.asarray(
                [
                    baseline
                    if source is None
                    else index[
                        (
                            target,
                            case_id,
                            baseline_row.sample_id,
                            a1_action_id(source),
                        )
                    ].probability_mean
                    for source in selected
                ],
                dtype=np.float64,
            )
            labels.append(truth[key])
            predictions.append(int(np.mean(values, dtype=np.float64) >= 0.5))
        label_array = np.asarray(labels, dtype=np.int8)
        prediction_array = np.asarray(predictions, dtype=np.int8)
        positive = label_array == 1
        negative = ~positive
        output.append(
            CaseMethodConfusion(
                target,
                case_id,
                method_id,
                int(np.sum(positive & (prediction_array == 1), dtype=np.int64)),
                int(np.sum(negative & (prediction_array == 0), dtype=np.int64)),
                int(np.sum(negative & (prediction_array == 1), dtype=np.int64)),
                int(np.sum(positive & (prediction_array == 0), dtype=np.int64)),
            )
        )
    return tuple(output)


def _restricted_contrast(
    confusions: Sequence[CaseMethodConfusion],
    *,
    method_id: str,
    reference_id: str,
    centers: Sequence[str],
) -> Fraction:
    differences = []
    for center in centers:
        method_rows = tuple(
            row
            for row in confusions
            if row.method_id == method_id and row.target_center == center
        )
        reference_rows = tuple(
            row
            for row in confusions
            if row.method_id == reference_id and row.target_center == center
        )
        method_metric = pooled_bacc(
            method_rows, scope_id=f"center={center}", method_id=method_id
        )
        reference_metric = pooled_bacc(
            reference_rows,
            scope_id=f"center={center}",
            method_id=reference_id,
        )
        differences.append(method_metric.exact - reference_metric.exact)
    return sum(differences, Fraction(0)) / len(differences)


def delete_one_center_rows(
    *,
    probability_surface: object,
    plans: Sequence[WholeCaseLooPlan],
    support_gains: Mapping[tuple[str, str], Sequence[DirectionalGain]],
    donor_priors: Sequence[DonorPrior],
    terminal_labels: Sequence[object],
    preterminal_confusions: Sequence[CaseMethodConfusion],
) -> tuple[dict[str, object], ...]:
    truth = tuple(terminal_labels)
    output: list[dict[str, object]] = []
    for deleted in CENTERS:
        reduced = _reduced_priors(donor_priors, deleted_center=deleted)
        decisions = tuple(
            decision
            for plan in plans
            if plan.target_center != deleted
            for decision in select_arm_decisions(
                method_id="DCSE_LOO",
                target_center=plan.target_center,
                case_id=plan.case_id,
                support_gains=tuple(support_gains[plan.key]),
                donor_priors=reduced[plan.target_center],  # type: ignore[arg-type]
            )
        )
        allowed = tuple(center for center in CENTERS if center != deleted)
        filtered_labels = tuple(
            row for row in truth if str(row.target_center) in allowed
        )
        dcse_confusions = _score_decision_subset(
            probability_surface,
            decisions,
            filtered_labels,
            method_id=f"DCSE_LOO_delete_center_{deleted}",
        )
        fixed_confusions = tuple(
            row
            for row in preterminal_confusions
            if row.method_id in {B_ACTION_ID, U_ACTION_ID}
            and row.target_center in allowed
        )
        confusions = (*dcse_confusions, *fixed_confusions)
        dcse_id = f"DCSE_LOO_delete_center_{deleted}"
        for reference in (B_ACTION_ID, U_ACTION_ID):
            exact = _restricted_contrast(
                confusions,
                method_id=dcse_id,
                reference_id=reference,
                centers=allowed,
            )
            output.append(
                {
                    "schema_version": "fixed_bank_dcse_whole_pipeline_delete_one_center_v1",
                    "deleted_center": deleted,
                    "method_id": "DCSE_LOO",
                    "reference_id": reference,
                    "contrast_id": f"DCSE_LOO-{reference}",
                    "remaining_centers": list(allowed),
                    "remaining_center_count": len(allowed),
                    "exact_fraction": [exact.numerator, exact.denominator],
                    "estimate": float(exact),
                    "deleted_center_removed_from_evaluation": True,
                    "deleted_center_removed_from_all_G_query_contributions": True,
                    "support_S_recomputed_or_preserved_by_unaffected_route": True,
                    "all_endpoint_decisions_recomputed": True,
                    "descriptive_only": True,
                    "is_gate_input": True,
                }
            )
    return tuple(output)


def leave_one_arm_rows(
    *,
    probability_surface: object,
    arm_decisions: Sequence[CaseArmDecision],
    terminal_labels: Sequence[object],
    baseline_confusions: Sequence[CaseMethodConfusion],
) -> tuple[dict[str, object], ...]:
    canonical = tuple(
        row for row in arm_decisions if row.method_id == "DCSE_LOO"
    )
    output = []
    for deleted_arm in ARM_IDS:
        method = f"DCSE_LOO_without_{deleted_arm}"
        ablation_confusions = _score_decision_subset(
            probability_surface,
            tuple(row for row in canonical if row.arm_id != deleted_arm),
            terminal_labels,
            method_id=method,
        )
        confusions = (*ablation_confusions, *baseline_confusions)
        exact = _restricted_contrast(
            confusions,
            method_id=method,
            reference_id=B_ACTION_ID,
            centers=CENTERS,
        )
        output.append(
            {
                "schema_version": "fixed_bank_dcse_leave_one_arm_ablation_v1",
                "deleted_arm_id": deleted_arm,
                "retained_arm_ids": [
                    arm for arm in ARM_IDS if arm != deleted_arm
                ],
                "retained_arm_count": len(ARM_IDS) - 1,
                "method_id": "DCSE_LOO",
                "reference_id": B_ACTION_ID,
                "contrast_id": "DCSE_LOO-B",
                "exact_fraction": [exact.numerator, exact.denominator],
                "estimate": float(exact),
                "probability_average_recomputed_over_eight_preserved_arm_identities": True,
                "endpoint_decisions_reselected": False,
                "descriptive_only": True,
                "is_gate_input": True,
            }
        )
    return tuple(output)


def descriptive_inference(
    *,
    primary: EqualCenterContrast,
    delete_one_center: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    center_values = np.asarray(
        [
            float(Fraction(numerator, denominator))
            for _center, numerator, denominator in primary.center_differences
        ],
        dtype=np.float64,
    )
    critical = float(student_t.ppf(0.975, df=len(CENTERS) - 1))
    mean = float(np.mean(center_values, dtype=np.float64))
    standard_error = float(
        np.std(center_values, ddof=1) / math.sqrt(len(center_values))
    )
    delete_values = np.asarray(
        [
            float(row["estimate"])
            for row in delete_one_center
            if row["reference_id"] == B_ACTION_ID
        ],
        dtype=np.float64,
    )
    if delete_values.shape != (len(CENTERS),):
        raise ProtocolError(
            "DCSE descriptive jackknife lacks nine whole-pipeline deletions."
        )
    jackknife_mean = float(np.mean(delete_values, dtype=np.float64))
    jackknife_se = float(
        math.sqrt(
            (len(CENTERS) - 1)
            / len(CENTERS)
            * float(
                np.sum(
                    (delete_values - jackknife_mean) ** 2,
                    dtype=np.float64,
                )
            )
        )
    )
    bias_corrected = len(CENTERS) * float(primary.estimate) - (
        len(CENTERS) - 1
    ) * jackknife_mean
    return {
        "schema_version": "fixed_bank_dcse_descriptive_inference_v1",
        "nominal_t_interval": {
            "estimate": mean,
            "standard_error": standard_error,
            "degrees_of_freedom": len(CENTERS) - 1,
            "critical_value": critical,
            "lower": mean - critical * standard_error,
            "upper": mean + critical * standard_error,
            "outer_unit": "target_center",
            "descriptive_only": True,
            "is_gate": False,
        },
        "whole_pipeline_delete_one_center_jackknife": {
            "full_estimate": float(primary.estimate),
            "mean_delete_one_estimate": jackknife_mean,
            "bias_corrected_estimate": bias_corrected,
            "standard_error": jackknife_se,
            "critical_value": critical,
            "lower": bias_corrected - critical * jackknife_se,
            "upper": bias_corrected + critical * jackknife_se,
            "outer_unit": "target_center",
            "all_G_query_contributions_recomputed": True,
            "descriptive_only": True,
            "is_gate": False,
        },
        "nominal_t_and_jackknife_are_descriptive": True,
        "matched_G_is_not_a_gate": True,
    }


__all__ = (
    "canonical_case_action_counts",
    "delete_one_center_rows",
    "descriptive_inference",
    "leave_one_arm_rows",
    "support_gains_by_route",
)
