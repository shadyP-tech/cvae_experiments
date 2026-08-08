"""Center-cluster inference and sealed H x e oracle diagnostics."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    CONFIDENCE_LEVEL,
    EXPECTED_SEED_CELL_COUNT,
    CenterContrast,
    ContrastInference,
    FreshEvaluationReport,
    GENERATION_SEEDS,
    PERMUTATION_CONTRAST,
    PRIMARY_CONTRASTS,
    PRIMARY_ENDPOINT,
    SECONDARY_CONTRASTS,
    ScoredEvaluation,
    TIE_ATOL,
    TRAINING_SEEDS,
)
from .oracle_diagnostics import compute_oracle_diagnostics
from .prediction_seal import PredictionSealCapability
from .scoring import score_sealed_predictions
from .scored_validation import validate_scored_evaluation


def build_center_contrasts(
    scored: ScoredEvaluation,
) -> tuple[CenterContrast, ...]:
    """Compute predeclared primary and S-P deltas within each target center."""

    validate_scored_evaluation(scored)
    ensemble = {row.key: row for row in scored.ensemble_metrics}
    seed = {row.key: row for row in scored.seed_cell_metrics}
    output: list[CenterContrast] = []
    specifications = (
        PRIMARY_CONTRASTS + SECONDARY_CONTRASTS + (PERMUTATION_CONTRAST,)
    )
    for contrast_id, left_action, right_action in specifications:
        if contrast_id == "S-P":
            role = "predeclared_source_identity_permutation_diagnostic"
        elif contrast_id in {row[0] for row in PRIMARY_CONTRASTS}:
            role = "predeclared_primary_center_contrast"
        else:
            role = "predeclared_secondary_center_contrast"
        for target in CENTERS:
            try:
                left_ensemble = ensemble[(target, left_action)]
                right_ensemble = ensemble[(target, right_action)]
            except KeyError as exc:
                raise ProtocolError(
                    "Fresh Stage-70 ensemble contrast coverage is incomplete."
                ) from exc
            seed_differences: list[float] = []
            for training_seed in TRAINING_SEEDS:
                for generation_seed in GENERATION_SEEDS:
                    try:
                        left_seed = seed[
                            (target, training_seed, generation_seed, left_action)
                        ]
                        right_seed = seed[
                            (target, training_seed, generation_seed, right_action)
                        ]
                    except KeyError as exc:
                        raise ProtocolError(
                            "Fresh Stage-70 paired seed-cell contrast is incomplete."
                        ) from exc
                    seed_differences.append(left_seed.bacc - right_seed.bacc)
            if len(seed_differences) != EXPECTED_SEED_CELL_COUNT:
                raise ProtocolError(
                    "Fresh Stage-70 paired seed-cell contrast coverage drifted."
                )
            output.append(
                CenterContrast(
                    contrast_id=contrast_id,
                    target_center=target,
                    left_action_id=left_action,
                    right_action_id=right_action,
                    probability_ensemble_bacc_delta=(
                        left_ensemble.bacc - right_ensemble.bacc
                    ),
                    descriptive_seed_cell_mean_bacc_delta=float(
                        np.mean(np.asarray(seed_differences, dtype=np.float64))
                    ),
                    contrast_role=role,
                    primary_endpoint=PRIMARY_ENDPOINT,
                )
            )
    return tuple(output)


def infer_center_contrasts(
    rows: Sequence[CenterContrast],
) -> tuple[ContrastInference, ...]:
    """Use the nine targets—not 81 seed cells—as independent units."""

    rows = tuple(rows)
    expected_ids = tuple(
        specification[0]
        for specification in PRIMARY_CONTRASTS + SECONDARY_CONTRASTS
    ) + (PERMUTATION_CONTRAST[0],)
    output: list[ContrastInference] = []
    for contrast_id in expected_ids:
        group = [row for row in rows if row.contrast_id == contrast_id]
        if (
            len(group) != len(CENTERS)
            or {row.target_center for row in group} != set(CENTERS)
            or len({row.target_center for row in group}) != len(CENTERS)
            or any(row.primary_endpoint != PRIMARY_ENDPOINT for row in group)
        ):
            raise ProtocolError(
                "Fresh Stage-70 inference requires one row per target center."
            )
        values = np.asarray(
            [row.probability_ensemble_bacc_delta for row in group],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ProtocolError("Fresh Stage-70 center contrasts must be finite.")
        n = len(values)
        estimate = float(np.mean(values))
        standard_deviation = float(np.std(values, ddof=1))
        standard_error = standard_deviation / math.sqrt(float(n))
        two_sided_critical = float(
            student_t.ppf(0.5 + CONFIDENCE_LEVEL / 2.0, df=n - 1)
        )
        one_sided_critical = float(student_t.ppf(CONFIDENCE_LEVEL, df=n - 1))
        ties = int(np.sum(np.abs(values) <= TIE_ATOL))
        wins = int(np.sum(values > TIE_ATOL))
        losses = int(np.sum(values < -TIE_ATOL))
        output.append(
            ContrastInference(
                contrast_id=contrast_id,
                mean_probability_ensemble_bacc_delta=estimate,
                two_sided_95_ci_low=estimate
                - two_sided_critical * standard_error,
                two_sided_95_ci_high=estimate
                + two_sided_critical * standard_error,
                one_sided_95_lcb=estimate
                - one_sided_critical * standard_error,
                wins=wins,
                ties=ties,
                losses=losses,
                center_count=n,
                contrast_role=group[0].contrast_role,
                primary_endpoint=PRIMARY_ENDPOINT,
            )
        )
    return tuple(output)


def evaluate_sealed_predictions(
    capability: PredictionSealCapability,
    labels_by_row_id: Mapping[object, object],
) -> FreshEvaluationReport:
    """Run the complete fixed fresh evaluation without any policy update path."""

    scored = score_sealed_predictions(capability, labels_by_row_id)
    center_rows = build_center_contrasts(scored)
    inference = infer_center_contrasts(center_rows)
    oracle = compute_oracle_diagnostics(scored, capability)
    return FreshEvaluationReport(
        scored=scored,
        center_contrasts=center_rows,
        contrast_inference=inference,
        oracle_diagnostics=oracle,
        prediction_seal_hash=scored.prediction_seal_hash,
        primary_endpoint=PRIMARY_ENDPOINT,
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
