"""Label-free probability construction for B/U, I, controls, and robust R."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import ARM_IDS, B_ACTION_ID, U_ACTION_ID, a1_action_id
from .identification_products import CaseIdentificationDecision
from .prediction_products import MethodPrediction
from .probability_surfaces import ExactNineProbabilityRow, ExactNineProbabilitySurface, ProbabilityIndex
from .robust_products import RobustArmDecision


def _index(
    surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow] | ProbabilityIndex,
) -> ProbabilityIndex:
    return surface_or_rows if isinstance(surface_or_rows, ProbabilityIndex) else ProbabilityIndex(surface_or_rows)


def _cell(index: ProbabilityIndex, target: str, case: str, sample: str, action: str) -> float:
    try:
        return float(index[(target, case, sample, action)].probability_mean)
    except KeyError as exc:
        raise ProtocolError("OGDE composition lacks a physical probability cell.") from exc


def compose_physical_action_predictions(
    surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow] | ProbabilityIndex,
    *,
    action_id: str,
    method_id: str | None = None,
) -> tuple[MethodPrediction, ...]:
    if action_id not in {B_ACTION_ID, U_ACTION_ID}:
        raise ProtocolError("OGDE physical baseline composition supports only B or U.")
    method = action_id if method_id is None else str(method_id)
    index = _index(surface_or_rows)
    rows = tuple(row for row in index.values() if row.action_id == action_id)
    output: list[MethodPrediction] = []
    for row in sorted(rows, key=lambda value: value.sample_key):
        baseline = _cell(index, row.target_center, row.case_id, row.sample_id, B_ACTION_ID)
        output.append(
            MethodPrediction(
                row.target_center,
                row.case_id,
                row.sample_id,
                method,
                row.probability_mean,
                int(row.probability_mean >= 0.5),
                int(baseline >= 0.5),
                f"physical_action::{action_id}",
                None,
                (),
                "frozen_exact_nine_physical_probability",
            )
        )
    return tuple(output)


def compose_identification_case_predictions(
    surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow] | ProbabilityIndex,
    decision: CaseIdentificationDecision,
    *,
    control: str = "primary",
) -> tuple[MethodPrediction, ...]:
    methods = {
        "primary": decision.method_id,
        "gate_only": "I_GATE_ONLY",
        "source_only": "I_SOURCE_ONLY",
    }
    if control not in methods:
        raise ProtocolError("OGDE identification composition control drifted.")
    method_id = methods[control]
    index = _index(surface_or_rows)
    baseline_rows = index.rows_for_case_action(decision.target_center, decision.case_id, B_ACTION_ID)
    if not baseline_rows:
        raise ProtocolError("OGDE identification composition lacks held-case B rows.")
    output: list[MethodPrediction] = []
    for row in baseline_rows:
        baseline = row.probability_mean
        branch = decision.decision_for_baseline_class(int(baseline >= 0.5))
        if control == "primary":
            selected = branch.selected_source
            sources = () if selected is None else (selected,)
            probability = baseline if selected is None else _cell(
                index, decision.target_center, decision.case_id, row.sample_id, a1_action_id(selected)
            )
            reason = branch.decision_reason
        elif control == "gate_only":
            selected = branch.selected_source
            sources = branch.eligible_sources if selected is not None else ()
            probability = baseline if not sources else float(
                np.mean(
                    np.asarray(
                        [
                            _cell(index, decision.target_center, decision.case_id, row.sample_id, a1_action_id(source))
                            for source in sources
                        ],
                        dtype=np.float64,
                    ),
                    dtype=np.float64,
                )
            )
            reason = "canonical_I_OFF_mask_then_mean_all_positive_eligible_A1"
        else:
            selected = branch.source_only_selected_source
            sources = () if selected is None else (selected,)
            probability = baseline if selected is None else _cell(
                index, decision.target_center, decision.case_id, row.sample_id, a1_action_id(selected)
            )
            reason = "canonical_normalized_rank_without_strict_positive_OFF_gate"
        output.append(
            MethodPrediction(
                decision.target_center,
                decision.case_id,
                row.sample_id,
                method_id,
                probability,
                int(probability >= 0.5),
                int(baseline >= 0.5),
                f"identification::{control}",
                selected,
                tuple(sources),
                reason,
            )
        )
    return tuple(output)


def compose_robust_case_predictions(
    surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow] | ProbabilityIndex,
    decisions: Sequence[RobustArmDecision],
) -> tuple[MethodPrediction, ...]:
    arms = tuple(decisions)
    if tuple(row.arm_id for row in arms) != ARM_IDS or len({row.arm_id for row in arms}) != len(ARM_IDS):
        raise ProtocolError("OGDE robust composition requires all nine preserved arms.")
    identities = {(row.method_id, row.target_center, row.case_id) for row in arms}
    if len(identities) != 1:
        raise ProtocolError("OGDE robust composition mixes methods or cases.")
    method_id, target, case = next(iter(identities))
    index = _index(surface_or_rows)
    baseline_rows = index.rows_for_case_action(target, case, B_ACTION_ID)
    output: list[MethodPrediction] = []
    for row in baseline_rows:
        baseline = row.probability_mean
        branch = int(baseline >= 0.5)
        selected = tuple(arm.decision_for_baseline_class(branch).selected_source for arm in arms)
        arm_probabilities = np.asarray(
            [
                baseline
                if source is None
                else _cell(index, target, case, row.sample_id, a1_action_id(source))
                for source in selected
            ],
            dtype=np.float64,
        )
        probability = float(np.mean(arm_probabilities, dtype=np.float64))
        output.append(
            MethodPrediction(
                target,
                case,
                row.sample_id,
                method_id,
                probability,
                int(probability >= 0.5),
                branch,
                "robust::nine_preserved_arms",
                None,
                selected,
                "float64_mean_all_nine_arm_probabilities_duplicates_preserved",
            )
        )
    return tuple(output)


def compose_identification_predictions(
    surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow] | ProbabilityIndex,
    decisions: Sequence[CaseIdentificationDecision],
    *,
    control: str = "primary",
) -> tuple[MethodPrediction, ...]:
    return tuple(
        prediction
        for decision in decisions
        for prediction in compose_identification_case_predictions(surface_or_rows, decision, control=control)
    )


def compose_robust_predictions(
    surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow] | ProbabilityIndex,
    decisions: Sequence[RobustArmDecision],
) -> tuple[MethodPrediction, ...]:
    grouped: dict[tuple[str, str, str], list[RobustArmDecision]] = {}
    for row in decisions:
        grouped.setdefault((row.method_id, row.target_center, row.case_id), []).append(row)
    return tuple(
        prediction
        for key in sorted(grouped)
        for prediction in compose_robust_case_predictions(
            surface_or_rows,
            tuple(sorted(grouped[key], key=lambda row: ARM_IDS.index(row.arm_id))),
        )
    )


__all__ = (
    "MethodPrediction",
    "compose_identification_case_predictions",
    "compose_identification_predictions",
    "compose_physical_action_predictions",
    "compose_robust_case_predictions",
    "compose_robust_predictions",
)
