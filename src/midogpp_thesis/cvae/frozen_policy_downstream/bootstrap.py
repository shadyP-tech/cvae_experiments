"""Paired descriptive center/case bootstrap for the frozen 3x3 seed design."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..expert_bank.uniform_b_v2_promotion.contracts import GENERATION_SEEDS, TRAINING_SEEDS
from ..protocol import ProtocolError
from .contracts import CONTROL_ARM, METADATA_ARM, UTILITY_ARM
from .scoring import CaseConfusionRow


@dataclass(frozen=True)
class BootstrapSummary:
    comparison_id: str
    observed_mean_bacc_delta: float
    bootstrap_mean_bacc_delta: float
    percentile_2_5: float
    percentile_97_5: float
    seed: int
    valid_replicates: int
    attempted_replicates: int
    rejected_replicates: int
    centers_resampled: bool = True
    cases_resampled_within_center: bool = True
    full_crossed_seed_grid_retained: bool = True

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage70_descriptive_bootstrap_v1",
            "comparison_id": self.comparison_id,
            "observed_mean_bacc_delta": self.observed_mean_bacc_delta,
            "bootstrap_mean_bacc_delta": self.bootstrap_mean_bacc_delta,
            "percentile_2_5": self.percentile_2_5,
            "percentile_97_5": self.percentile_97_5,
            "seed": self.seed,
            "valid_replicates": self.valid_replicates,
            "attempted_replicates": self.attempted_replicates,
            "rejected_replicates": self.rejected_replicates,
            "centers_resampled": self.centers_resampled,
            "cases_resampled_within_center": self.cases_resampled_within_center,
            "full_crossed_seed_grid_retained": self.full_crossed_seed_grid_retained,
            "flattened_seed_pairs_resampled_as_iid": False,
            "invalid_class_denominator_draws_rejected": True,
            "interval_role": "descriptive_resampling_uncertainty_only",
            "fresh_confirmatory_inference": False,
        }


def paired_descriptive_bootstrap(
    rows: Sequence[CaseConfusionRow],
    *,
    seed: int = 70,
    valid_replicates: int = 2000,
    max_attempts: int = 20000,
) -> tuple[BootstrapSummary, BootstrapSummary]:
    if valid_replicates <= 0 or max_attempts < valid_replicates:
        raise ProtocolError("Stage-70 bootstrap replicate limits are invalid.")
    lookup = {
        (
            row.policy_id,
            row.target_center,
            row.training_seed,
            row.generation_seed,
            row.case_id,
        ): (row.tn, row.fp, row.fn, row.tp)
        for row in rows
    }
    if len(lookup) != len(rows):
        raise ProtocolError("Stage-70 bootstrap case rows are duplicated.")
    centers = sorted({row.target_center for row in rows})
    if len(centers) != 9:
        raise ProtocolError("Stage-70 bootstrap requires all nine eligible centers.")
    cases_by_center = {
        center: sorted({row.case_id for row in rows if row.target_center == center})
        for center in centers
    }
    for center, cases in cases_by_center.items():
        if not cases:
            raise ProtocolError(f"Stage-70 bootstrap center {center} has no cases.")
    _validate_complete_lookup(lookup, cases_by_center)

    observed = _evaluate_draw(
        lookup,
        tuple((center, tuple(cases_by_center[center])) for center in centers),
    )
    if observed is None:
        raise ProtocolError("Observed Stage-70 case surface lacks a class denominator.")
    rng = np.random.default_rng(seed)
    metadata_deltas: list[float] = []
    utility_deltas: list[float] = []
    attempts = 0
    rejected = 0
    while len(metadata_deltas) < valid_replicates and attempts < max_attempts:
        attempts += 1
        drawn_centers = rng.choice(centers, size=len(centers), replace=True)
        draw: list[tuple[str, tuple[str, ...]]] = []
        for center_raw in drawn_centers.tolist():
            center = str(center_raw)
            cases = cases_by_center[center]
            sampled = tuple(
                str(value)
                for value in rng.choice(cases, size=len(cases), replace=True).tolist()
            )
            draw.append((center, sampled))
        values = _evaluate_draw(lookup, tuple(draw))
        if values is None:
            rejected += 1
            continue
        metadata_deltas.append(values[METADATA_ARM] - values[CONTROL_ARM])
        utility_deltas.append(values[UTILITY_ARM] - values[CONTROL_ARM])
    if len(metadata_deltas) != valid_replicates:
        raise ProtocolError(
            "Stage-70 bootstrap could not obtain the predeclared valid replicate count."
        )
    observed_metadata = observed[METADATA_ARM] - observed[CONTROL_ARM]
    observed_utility = observed[UTILITY_ARM] - observed[CONTROL_ARM]
    if observed_utility != 0.0 or any(value != 0.0 for value in utility_deltas):
        raise ProtocolError("Utility/control bootstrap must be deterministic equivalence.")
    return (
        _summary(
            "metadata_max_tie_union_minus_equal_union",
            observed_metadata,
            metadata_deltas,
            seed=seed,
            attempts=attempts,
            rejected=rejected,
        ),
        _summary(
            "utility_regret_minus_equal_union_equivalence",
            observed_utility,
            utility_deltas,
            seed=seed,
            attempts=attempts,
            rejected=rejected,
        ),
    )


def _evaluate_draw(
    lookup: dict[tuple[str, str, int, int, str], tuple[int, int, int, int]],
    draw: tuple[tuple[str, tuple[str, ...]], ...],
) -> dict[str, float] | None:
    arms = (CONTROL_ARM, METADATA_ARM, UTILITY_ARM)
    values: dict[str, list[float]] = {arm: [] for arm in arms}
    for center, sampled_cases in draw:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                for arm in arms:
                    tn = fp = fn = tp = 0
                    for case_id in sampled_cases:
                        counts = lookup[(arm, center, training_seed, generation_seed, case_id)]
                        tn += counts[0]
                        fp += counts[1]
                        fn += counts[2]
                        tp += counts[3]
                    negative = tn + fp
                    positive = tp + fn
                    if negative <= 0 or positive <= 0:
                        return None
                    values[arm].append(0.5 * ((tn / negative) + (tp / positive)))
    expected = len(draw) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
    if any(len(observed) != expected for observed in values.values()):
        raise ProtocolError("Stage-70 bootstrap full crossed seed grid drifted.")
    return {arm: sum(observed) / len(observed) for arm, observed in values.items()}


def _validate_complete_lookup(
    lookup: dict[tuple[str, str, int, int, str], tuple[int, int, int, int]],
    cases_by_center: dict[str, list[str]],
) -> None:
    expected = {
        (arm, center, training_seed, generation_seed, case_id)
        for arm in (CONTROL_ARM, METADATA_ARM, UTILITY_ARM)
        for center, cases in cases_by_center.items()
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
        for case_id in cases
    }
    if set(lookup) != expected:
        raise ProtocolError("Stage-70 bootstrap case/arm/seed surface is incomplete.")


def _summary(
    comparison_id: str,
    observed: float,
    values: list[float],
    *,
    seed: int,
    attempts: int,
    rejected: int,
) -> BootstrapSummary:
    array = np.asarray(values, dtype=np.float64)
    return BootstrapSummary(
        comparison_id=comparison_id,
        observed_mean_bacc_delta=float(observed),
        bootstrap_mean_bacc_delta=float(np.mean(array)),
        percentile_2_5=float(np.quantile(array, 0.025)),
        percentile_97_5=float(np.quantile(array, 0.975)),
        seed=seed,
        valid_replicates=len(values),
        attempted_replicates=attempts,
        rejected_replicates=rejected,
    )


__all__ = ("BootstrapSummary", "paired_descriptive_bootstrap")
