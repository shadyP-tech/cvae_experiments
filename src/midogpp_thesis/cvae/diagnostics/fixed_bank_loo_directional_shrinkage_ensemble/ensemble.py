"""Probability-level composition of the nine preserved directional arms."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    ARM_IDS,
    B_ACTION_ID,
    HARD_THRESHOLD,
    U_ACTION_ID,
    a1_action_id,
    candidate_sources,
)
from .probability_surfaces import ExactNineProbabilityRow, ProbabilityIndex, hard_prediction
from .products import CaseArmDecision, CaseControlDecision, MethodPrediction


DESCRIPTIVE_METHOD_IDS = (
    "DCSE_hard_vote_descriptive",
    "DCSE_unique_mean_descriptive",
    "uniform_A1_mean_descriptive",
    "DCSE_zero_to_one_only_descriptive",
    "DCSE_one_to_zero_only_descriptive",
)


def _cell_probability(
    probability: Mapping[tuple[str, str, str, str], object],
    key: tuple[str, str, str, str],
) -> float:
    try:
        row = probability[key]
    except KeyError as exc:
        raise ProtocolError(f"DCSE composed prediction lacks physical cell {key}.") from exc
    value = float(getattr(row, "probability_mean", getattr(row, "probability", np.nan)))
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ProtocolError("DCSE composed physical probability lies outside [0,1].")
    return value


def compose_arm_probability(
    *,
    baseline_probability: float,
    selected_source: str | None,
    selected_source_probability: float | None,
) -> float:
    """OFF contributes B; an active source contributes its exact-nine mean."""

    baseline = float(baseline_probability)
    if not np.isfinite(baseline) or not 0.0 <= baseline <= 1.0:
        raise ProtocolError("DCSE baseline probability lies outside [0,1].")
    if selected_source is None:
        if selected_source_probability is not None:
            raise ProtocolError("DCSE OFF arm cannot receive a source probability.")
        return baseline
    if selected_source_probability is None:
        raise ProtocolError("DCSE active arm lacks its source probability.")
    value = float(selected_source_probability)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ProtocolError("DCSE source probability lies outside [0,1].")
    return value


def compose_case_predictions(
    probabilities: ProbabilityIndex | Sequence[ExactNineProbabilityRow] | object,
    decisions: Sequence[CaseArmDecision],
    *,
    method_id: str,
) -> tuple[MethodPrediction, ...]:
    """Branch on B hard class, then average all nine arm probabilities."""

    rows = tuple(decisions)
    if tuple(row.arm_id for row in rows) != ARM_IDS or len({row.arm_id for row in rows}) != len(ARM_IDS):
        raise ProtocolError("DCSE composition requires the nine preserved arm identities.")
    identities = {(row.target_center, row.case_id, row.method_id) for row in rows}
    if len(identities) != 1 or next(iter(identities))[2] != str(method_id):
        raise ProtocolError("DCSE composition decisions mix cases or methods.")
    target, case_id, _ = next(iter(identities))
    index = probabilities if isinstance(probabilities, ProbabilityIndex) else ProbabilityIndex(
        tuple(getattr(probabilities, "rows", probabilities))
    )
    baseline_rows = index.rows_for_case_action(target, case_id, B_ACTION_ID)
    if not baseline_rows:
        raise ProtocolError("DCSE composition lacks held-case baseline rows.")
    output: list[MethodPrediction] = []
    for baseline_row in baseline_rows:
        baseline = float(baseline_row.probability_mean)
        branch = hard_prediction(baseline)
        selected = tuple(
            row.decision_for_baseline_class(branch).selected_source for row in rows
        )
        arm_probabilities = tuple(
            compose_arm_probability(
                baseline_probability=baseline,
                selected_source=source,
                selected_source_probability=(
                    None
                    if source is None
                    else _cell_probability(
                        index,
                        (target, case_id, baseline_row.sample_id, a1_action_id(source)),
                    )
                ),
            )
            for source in selected
        )
        output.append(
            MethodPrediction(
                target_center=target,
                case_id=case_id,
                sample_id=baseline_row.sample_id,
                method_id=method_id,
                probability=float(np.mean(np.asarray(arm_probabilities, dtype=np.float64), dtype=np.float64)),
                baseline_hard_class=branch,
                selected_sources_by_arm=selected,
            )
        )
    return tuple(output)


def compose_method_predictions(
    probabilities: ProbabilityIndex | Sequence[ExactNineProbabilityRow] | object,
    decisions: Sequence[CaseArmDecision],
    *,
    method_id: str,
) -> tuple[MethodPrediction, ...]:
    grouped: dict[tuple[str, str], list[CaseArmDecision]] = {}
    for row in decisions:
        if row.method_id != method_id:
            raise ProtocolError("DCSE method composition received another method's decision.")
        grouped.setdefault((row.target_center, row.case_id), []).append(row)
    if not grouped:
        raise ProtocolError("DCSE method composition cannot be empty.")
    return tuple(
        prediction
        for key in sorted(grouped)
        for prediction in compose_case_predictions(
            probabilities,
            tuple(sorted(grouped[key], key=lambda row: ARM_IDS.index(row.arm_id))),
            method_id=method_id,
        )
    )


def compose_control_predictions(
    probabilities: ProbabilityIndex | Sequence[ExactNineProbabilityRow] | object,
    decisions: Sequence[CaseControlDecision],
    *,
    method_id: str,
) -> tuple[MethodPrediction, ...]:
    index = probabilities if isinstance(probabilities, ProbabilityIndex) else ProbabilityIndex(
        tuple(getattr(probabilities, "rows", probabilities))
    )
    values = tuple(decisions)
    if not values or any(row.method_id != method_id for row in values):
        raise ProtocolError("DCSE control composition is empty or mixes methods.")
    output: list[MethodPrediction] = []
    for decision in sorted(values, key=lambda row: (row.target_center, row.case_id)):
        baseline_rows = index.rows_for_case_action(
            decision.target_center, decision.case_id, B_ACTION_ID
        )
        if not baseline_rows:
            raise ProtocolError("DCSE control composition lacks baseline rows.")
        for baseline_row in baseline_rows:
            baseline = float(baseline_row.probability_mean)
            branch = hard_prediction(baseline)
            directional = decision.decision_for_baseline_class(branch)
            selected = directional.selected_source
            if method_id == "DLOO_raw":
                probability = (
                    baseline
                    if selected is None
                    else _cell_probability(
                        index,
                        (
                            decision.target_center,
                            decision.case_id,
                            baseline_row.sample_id,
                            a1_action_id(selected),
                        ),
                    )
                )
                selections = (selected,)
            elif method_id == "LOO_frequency_committee":
                # The selected source is only the modal stability descriptor.
                # The declared committee prediction averages *all* nested
                # delete-one votes, with OFF votes contributing B, and applies
                # the sole threshold only after that probability mean.
                values: list[float] = []
                counts: list[int] = []
                selections_list: list[str | None] = []
                for source, numerator, denominator in directional.candidate_values:
                    frequency = Fraction(numerator, denominator)
                    value = (
                        baseline
                        if source is None
                        else _cell_probability(
                            index,
                            (
                                decision.target_center,
                                decision.case_id,
                                baseline_row.sample_id,
                                a1_action_id(source),
                            ),
                        )
                    )
                    vote_count = frequency * directional.nested_support_case_count
                    if vote_count.denominator != 1:
                        raise ProtocolError(
                            "DCSE committee frequency is not an exact nested-vote count."
                        )
                    values.append(value)
                    counts.append(vote_count.numerator)
                    selections_list.extend([source] * vote_count.numerator)
                if len(selections_list) != directional.nested_support_case_count:
                    raise ProtocolError("DCSE committee frequencies do not sum to one.")
                probability = compose_frequency_weighted_probability(
                    values, counts, nested_count=directional.nested_support_case_count
                )
                selections = tuple(selections_list)
            else:
                raise ProtocolError("DCSE control composition method drifted.")
            output.append(
                MethodPrediction(
                    decision.target_center,
                    decision.case_id,
                    baseline_row.sample_id,
                    method_id,
                    probability,
                    branch,
                    selections,
                )
            )
    return tuple(output)


def compose_frequency_weighted_probability(
    endpoint_probabilities: Sequence[float],
    vote_counts: Sequence[int],
    *,
    nested_count: int,
) -> float:
    """Float64 mean of all nested endpoint votes in canonical candidate order."""

    values = np.asarray(tuple(endpoint_probabilities), dtype=np.float64)
    counts = np.asarray(tuple(vote_counts), dtype=np.int64)
    if (
        values.ndim != 1
        or values.size == 0
        or values.shape != counts.shape
        or not np.isfinite(values).all()
        or np.any((values < 0.0) | (values > 1.0))
        or np.any(counts < 0)
        or isinstance(nested_count, bool)
        or int(nested_count) <= 0
        or int(np.sum(counts, dtype=np.int64)) != int(nested_count)
    ):
        raise ProtocolError("DCSE frequency-weighted endpoint inputs are malformed.")
    return float(
        np.sum(values * counts, dtype=np.float64) / np.float64(nested_count)
    )


def compose_direction_decomposition_predictions(
    probabilities: ProbabilityIndex | Sequence[ExactNineProbabilityRow] | object,
    decisions: Sequence[CaseArmDecision],
    *,
    direction: str,
) -> tuple[MethodPrediction, ...]:
    """Apply DCSE endpoints only on one B-defined direction; use B otherwise."""

    if direction not in {"zero_to_one", "one_to_zero"}:
        raise ProtocolError("DCSE direction-decomposition control direction drifted.")
    rows = tuple(decisions)
    if tuple(row.arm_id for row in rows) != ARM_IDS or any(
        row.method_id != "DCSE_LOO" for row in rows
    ):
        raise ProtocolError("DCSE direction decomposition requires one canonical DCSE route.")
    target, case_id = rows[0].target_center, rows[0].case_id
    if any((row.target_center, row.case_id) != (target, case_id) for row in rows):
        raise ProtocolError("DCSE direction decomposition mixes held routes.")
    method_id = f"DCSE_{direction}_only_descriptive"
    index = probabilities if isinstance(probabilities, ProbabilityIndex) else ProbabilityIndex(
        tuple(getattr(probabilities, "rows", probabilities))
    )
    active_branch = 0 if direction == "zero_to_one" else 1
    output: list[MethodPrediction] = []
    for baseline_row in index.rows_for_case_action(target, case_id, B_ACTION_ID):
        baseline = float(baseline_row.probability_mean)
        branch = hard_prediction(baseline)
        selected = (
            tuple(row.decision_for_baseline_class(branch).selected_source for row in rows)
            if branch == active_branch
            else (None,) * len(ARM_IDS)
        )
        values = tuple(
            baseline
            if source is None
            else _cell_probability(
                index,
                (target, case_id, baseline_row.sample_id, a1_action_id(source)),
            )
            for source in selected
        )
        output.append(
            MethodPrediction(
                target,
                case_id,
                baseline_row.sample_id,
                method_id,
                float(np.mean(np.asarray(values, dtype=np.float64), dtype=np.float64)),
                branch,
                selected,
            )
        )
    return tuple(output)


def compose_descriptive_control_predictions(
    probabilities: ProbabilityIndex | Sequence[ExactNineProbabilityRow] | object,
    dcse_decisions: Sequence[CaseArmDecision],
) -> tuple[MethodPrediction, ...]:
    """Compose all five declared, sealed, non-gating descriptive controls."""

    grouped: dict[tuple[str, str], list[CaseArmDecision]] = {}
    for row in dcse_decisions:
        if row.method_id != "DCSE_LOO":
            raise ProtocolError("DCSE descriptive controls received non-DCSE decisions.")
        grouped.setdefault((row.target_center, row.case_id), []).append(row)
    if not grouped:
        raise ProtocolError("DCSE descriptive control decision surface is empty.")
    by_method: dict[str, list[MethodPrediction]] = {
        method: [] for method in DESCRIPTIVE_METHOD_IDS
    }
    for key in sorted(grouped, key=lambda value: (value[0], value[1])):
        rows = tuple(sorted(grouped[key], key=lambda row: ARM_IDS.index(row.arm_id)))
        by_method[DESCRIPTIVE_METHOD_IDS[0]].extend(
            compose_hard_vote_comparator(probabilities, rows)
        )
        by_method[DESCRIPTIVE_METHOD_IDS[1]].extend(
            compose_unique_source_mean_comparator(probabilities, rows)
        )
        by_method[DESCRIPTIVE_METHOD_IDS[3]].extend(
            compose_direction_decomposition_predictions(
                probabilities, rows, direction="zero_to_one"
            )
        )
        by_method[DESCRIPTIVE_METHOD_IDS[4]].extend(
            compose_direction_decomposition_predictions(
                probabilities, rows, direction="one_to_zero"
            )
        )
    by_method[DESCRIPTIVE_METHOD_IDS[2]].extend(
        compose_uniform_a1_comparator(probabilities)
    )
    return tuple(row for method in DESCRIPTIVE_METHOD_IDS for row in by_method[method])


def fixed_physical_method_predictions(
    probabilities: ProbabilityIndex | Sequence[ExactNineProbabilityRow] | object,
    *,
    method_id: str,
) -> tuple[MethodPrediction, ...]:
    if method_id not in {B_ACTION_ID, U_ACTION_ID}:
        raise ProtocolError("DCSE fixed physical prediction supports only B or U.")
    index = probabilities if isinstance(probabilities, ProbabilityIndex) else ProbabilityIndex(
        tuple(getattr(probabilities, "rows", probabilities))
    )
    rows = tuple(
        row for key, row in index.items() if key[3] == method_id
    )
    if not rows:
        raise ProtocolError("DCSE fixed physical prediction surface is empty.")
    output = []
    for row in sorted(rows, key=lambda value: value.sample_key):
        baseline = _cell_probability(
            index,
            (row.target_center, row.case_id, row.sample_id, B_ACTION_ID),
        )
        output.append(
            MethodPrediction(
                row.target_center,
                row.case_id,
                row.sample_id,
                method_id,
                float(row.probability_mean),
                hard_prediction(baseline),
                (),
            )
        )
    return tuple(output)


def compose_uniform_a1_comparator(
    probabilities: ProbabilityIndex | Sequence[ExactNineProbabilityRow] | object,
    *,
    method_id: str = "uniform_A1_mean_descriptive",
) -> tuple[MethodPrediction, ...]:
    index = probabilities if isinstance(probabilities, ProbabilityIndex) else ProbabilityIndex(
        tuple(getattr(probabilities, "rows", probabilities))
    )
    baseline_rows = tuple(row for key, row in index.items() if key[3] == B_ACTION_ID)
    output = []
    for baseline_row in sorted(baseline_rows, key=lambda value: value.sample_key):
        target, case_id, sample_id = baseline_row.sample_key
        values = tuple(
            _cell_probability(index, (target, case_id, sample_id, a1_action_id(source)))
            for source in candidate_sources(target)
        )
        output.append(
            MethodPrediction(
                target,
                case_id,
                sample_id,
                method_id,
                float(np.mean(np.asarray(values, dtype=np.float64), dtype=np.float64)),
                hard_prediction(baseline_row.probability_mean),
                tuple(candidate_sources(target)),
            )
        )
    return tuple(output)


def compose_hard_vote_comparator(
    probabilities: ProbabilityIndex | Sequence[ExactNineProbabilityRow] | object,
    decisions: Sequence[CaseArmDecision],
    *,
    method_id: str = "DCSE_hard_vote_descriptive",
) -> tuple[MethodPrediction, ...]:
    """Descriptive comparator: threshold each arm, majority vote, 4.5 => class 1."""

    rows = tuple(decisions)
    if tuple(row.arm_id for row in rows) != ARM_IDS:
        raise ProtocolError("DCSE hard-vote comparator needs the canonical nine arms.")
    target, case_id = rows[0].target_center, rows[0].case_id
    index = probabilities if isinstance(probabilities, ProbabilityIndex) else ProbabilityIndex(
        tuple(getattr(probabilities, "rows", probabilities))
    )
    output: list[MethodPrediction] = []
    for baseline_row in index.rows_for_case_action(target, case_id, B_ACTION_ID):
        baseline = float(baseline_row.probability_mean)
        branch = hard_prediction(baseline)
        selected = tuple(row.decision_for_baseline_class(branch).selected_source for row in rows)
        arm_probabilities = tuple(
            baseline
            if source is None
            else _cell_probability(index, (target, case_id, baseline_row.sample_id, a1_action_id(source)))
            for source in selected
        )
        vote_probability = sum(hard_prediction(value) for value in arm_probabilities) / len(arm_probabilities)
        output.append(
            MethodPrediction(target, case_id, baseline_row.sample_id, method_id, vote_probability, branch, selected)
        )
    return tuple(output)


def compose_unique_source_mean_comparator(
    probabilities: ProbabilityIndex | Sequence[ExactNineProbabilityRow] | object,
    decisions: Sequence[CaseArmDecision],
    *,
    method_id: str = "DCSE_unique_mean_descriptive",
) -> tuple[MethodPrediction, ...]:
    """Descriptive comparator that collapses duplicate selected source identities."""

    rows = tuple(decisions)
    if tuple(row.arm_id for row in rows) != ARM_IDS:
        raise ProtocolError("DCSE unique-mean comparator needs the canonical nine arms.")
    target, case_id = rows[0].target_center, rows[0].case_id
    index = probabilities if isinstance(probabilities, ProbabilityIndex) else ProbabilityIndex(
        tuple(getattr(probabilities, "rows", probabilities))
    )
    output: list[MethodPrediction] = []
    for baseline_row in index.rows_for_case_action(target, case_id, B_ACTION_ID):
        baseline = float(baseline_row.probability_mean)
        branch = hard_prediction(baseline)
        selected = tuple(row.decision_for_baseline_class(branch).selected_source for row in rows)
        unique = tuple(dict.fromkeys(selected))
        values = tuple(
            baseline
            if source is None
            else _cell_probability(index, (target, case_id, baseline_row.sample_id, a1_action_id(source)))
            for source in unique
        )
        output.append(
            MethodPrediction(
                target,
                case_id,
                baseline_row.sample_id,
                method_id,
                float(np.mean(np.asarray(values, dtype=np.float64), dtype=np.float64)),
                branch,
                selected,
            )
        )
    return tuple(output)


def arm_selection_frequencies(
    decisions: Sequence[CaseArmDecision], *, baseline_class: int
) -> Mapping[str | None, Fraction]:
    rows = tuple(decisions)
    if tuple(row.arm_id for row in rows) != ARM_IDS:
        raise ProtocolError("DCSE arm frequencies require all nine arms.")
    selected = tuple(row.decision_for_baseline_class(baseline_class).selected_source for row in rows)
    counts = Counter(selected)
    return {source: Fraction(count, len(rows)) for source, count in sorted(counts.items(), key=lambda item: (-1 if item[0] is None else int(item[0])))}


__all__ = (
    "DESCRIPTIVE_METHOD_IDS",
    "arm_selection_frequencies",
    "compose_arm_probability",
    "compose_case_predictions",
    "compose_control_predictions",
    "compose_descriptive_control_predictions",
    "compose_direction_decomposition_predictions",
    "compose_frequency_weighted_probability",
    "compose_hard_vote_comparator",
    "compose_method_predictions",
    "compose_uniform_a1_comparator",
    "compose_unique_source_mean_comparator",
    "fixed_physical_method_predictions",
)
