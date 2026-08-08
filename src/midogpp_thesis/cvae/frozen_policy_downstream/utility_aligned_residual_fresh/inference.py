"""Center-cluster inference and terminal H x e oracle diagnostics."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from ...protocol import ProtocolError
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    CONFIDENCE_LEVEL,
    EXPECTED_SEED_CELL_COUNT,
    CenterContrast,
    ContrastInference,
    FreshEvaluationReport,
    OracleDiagnostic,
    PERMUTATION_CONTRAST,
    PRIMARY_CONTRASTS,
    ROUTED_ACTION_ID,
    SECONDARY_CONTRASTS,
    ScoredEvaluation,
    TIE_ATOL,
    TRAINING_SEEDS,
    GENERATION_SEEDS,
    legal_sources,
    tail_action_id,
)
from .prediction_seal import (
    PredictionSealCapability,
    read_sealed_prediction_snapshot,
)
from .scoring import score_sealed_predictions


def build_center_contrasts(scored: ScoredEvaluation) -> tuple[CenterContrast, ...]:
    ensemble = {row.key: row for row in scored.ensemble_metrics}
    seed = {row.key: row for row in scored.seed_cell_metrics}
    output: list[CenterContrast] = []
    specifications = PRIMARY_CONTRASTS + (PERMUTATION_CONTRAST,) + SECONDARY_CONTRASTS
    for contrast_id, left_action, right_action in specifications:
        role = (
            "predeclared_primary_center_contrast"
            if contrast_id in {row[0] for row in PRIMARY_CONTRASTS}
            else "predeclared_permutation_control"
            if contrast_id == PERMUTATION_CONTRAST[0]
            else "predeclared_secondary_center_contrast"
        )
        for target in CENTERS:
            try:
                left = ensemble[(target, left_action)]
                right = ensemble[(target, right_action)]
            except KeyError as exc:
                raise ProtocolError("Utility-aligned contrast coverage is incomplete.") from exc
            seed_deltas = []
            for training_seed in TRAINING_SEEDS:
                for generation_seed in GENERATION_SEEDS:
                    seed_deltas.append(
                        seed[(target, training_seed, generation_seed, left_action)].bacc
                        - seed[(target, training_seed, generation_seed, right_action)].bacc
                    )
            if len(seed_deltas) != EXPECTED_SEED_CELL_COUNT:
                raise ProtocolError("Utility-aligned paired seed coverage drifted.")
            output.append(
                CenterContrast(
                    contrast_id=contrast_id,
                    target_center=target,
                    left_action_id=left_action,
                    right_action_id=right_action,
                    probability_ensemble_bacc_delta=left.bacc - right.bacc,
                    descriptive_seed_cell_mean_bacc_delta=float(np.mean(seed_deltas)),
                    contrast_role=role,
                )
            )
    return tuple(output)


def infer_center_contrasts(
    rows: Sequence[CenterContrast],
) -> tuple[ContrastInference, ...]:
    rows = tuple(rows)
    ids = tuple(
        item[0]
        for item in PRIMARY_CONTRASTS + (PERMUTATION_CONTRAST,) + SECONDARY_CONTRASTS
    )
    output: list[ContrastInference] = []
    for contrast_id in ids:
        group = [row for row in rows if row.contrast_id == contrast_id]
        if len(group) != len(CENTERS) or {row.target_center for row in group} != set(CENTERS):
            raise ProtocolError("Utility-aligned inference requires nine centers.")
        values = np.asarray(
            [row.probability_ensemble_bacc_delta for row in group], dtype=np.float64
        )
        estimate = float(np.mean(values))
        standard_error = float(np.std(values, ddof=1)) / math.sqrt(len(values))
        two_sided = float(
            student_t.ppf(0.5 + CONFIDENCE_LEVEL / 2.0, df=len(values) - 1)
        )
        one_sided = float(student_t.ppf(CONFIDENCE_LEVEL, df=len(values) - 1))
        output.append(
            ContrastInference(
                contrast_id=contrast_id,
                mean_probability_ensemble_bacc_delta=estimate,
                two_sided_95_ci_low=estimate - two_sided * standard_error,
                two_sided_95_ci_high=estimate + two_sided * standard_error,
                one_sided_95_lcb=estimate - one_sided * standard_error,
                wins=int(np.sum(values > TIE_ATOL)),
                ties=int(np.sum(np.abs(values) <= TIE_ATOL)),
                losses=int(np.sum(values < -TIE_ATOL)),
                center_count=len(values),
                contrast_role=group[0].contrast_role,
            )
        )
    return tuple(output)


def compute_oracle_diagnostics(
    scored: ScoredEvaluation,
    capability: PredictionSealCapability,
) -> tuple[OracleDiagnostic, ...]:
    """Read the sealed H x e matrix without emitting a replacement policy."""

    state = read_sealed_prediction_snapshot(capability)
    if scored.prediction_seal_hash != state.seal_hash:
        raise ProtocolError("Utility-aligned oracle diagnostics escaped their seal.")
    ensemble = {row.key: row for row in scored.ensemble_metrics}
    output: list[OracleDiagnostic] = []
    for target in CENTERS:
        routed = state.plan.action_for(target, ROUTED_ACTION_ID)
        utilities = {
            source: float(ensemble[(target, tail_action_id(source))].bacc)
            for source in legal_sources(target)
        }
        oracle_source = min(utilities, key=lambda source: (-utilities[source], source))
        base = float(ensemble[(target, BASE_ACTION_ID)].bacc)
        routed_utility = float(ensemble[(target, ROUTED_ACTION_ID)].bacc)
        oracle = utilities[oracle_source]
        utility_range = max(utilities.values()) - min(utilities.values())
        gap = max(0.0, oracle - routed_utility)
        output.append(
            OracleDiagnostic(
                target_center=target,
                routed_source=routed.selected_source,
                oracle_source=oracle_source,
                routed_top1_agreement=routed.selected_source == oracle_source,
                base_bacc=base,
                routed_bacc=routed_utility,
                oracle_bacc=oracle,
                oracle_headroom_over_base_bacc=oracle - base,
                routed_oracle_gap_bacc=gap,
                normalized_routed_oracle_gap=(
                    gap / utility_range if utility_range > 0.0 else 0.0
                ),
                prediction_seal_hash=state.seal_hash,
                diagnostic_only=True,
                may_update_frozen_policy=False,
            )
        )
    return tuple(output)


def evaluate_sealed_predictions(
    capability: PredictionSealCapability,
    labels_by_row_id: Mapping[object, object],
) -> FreshEvaluationReport:
    scored = score_sealed_predictions(capability, labels_by_row_id)
    center_rows = build_center_contrasts(scored)
    return FreshEvaluationReport(
        scored=scored,
        center_contrasts=center_rows,
        contrast_inference=infer_center_contrasts(center_rows),
        oracle_diagnostics=compute_oracle_diagnostics(scored, capability),
        prediction_seal_hash=scored.prediction_seal_hash,
        policy_update_emitted=False,
    )


run_fresh_evaluation = evaluate_sealed_predictions


__all__ = (
    "build_center_contrasts",
    "compute_oracle_diagnostics",
    "evaluate_sealed_predictions",
    "infer_center_contrasts",
    "run_fresh_evaluation",
)
