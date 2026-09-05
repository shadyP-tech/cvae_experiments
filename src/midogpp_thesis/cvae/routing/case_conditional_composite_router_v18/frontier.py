"""Complete source-only candidate frontiers after label-free prediction seals."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from .composition import build_baseline_composite
from .hashing import canonical_hash
from .records import SealedOOFSelection, SelectedOOFRecord
from .stacked_fitting import HeldCandidatePrediction, POLICY_ARM_ID, choose_candidate
from .truth import SupportTruthCapability


def policy_moments(records: Sequence[SelectedOOFRecord | _FrontierObservation]) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in records:
        route = float(row.route_selected)
        grouped[row.center_id].append((row.bacc_gain, route, route * float(row.harm), route * row.brier_delta, route * row.log_loss_delta))
    means = np.mean([np.mean(values, axis=0) for _, values in sorted(grouped.items())], axis=0)
    coverage = float(means[1])
    risks = means[2:] / coverage if coverage else np.zeros(3)
    return {
        "gain": float(means[0]), "equal_center_route_coverage": coverage,
        "harm_moment_all_cases": float(means[2] - .25 * coverage),
        "brier_moment_all_cases": float(means[3] - .002 * coverage),
        "logloss_moment_all_cases": float(means[4] - .005 * coverage),
        "routed_harm": float(risks[0]), "routed_brier_delta": float(risks[1]), "routed_logloss_delta": float(risks[2]),
    }


def failed_constraints(records: Sequence[SelectedOOFRecord | _FrontierObservation], *, minimum_centers: int = 1) -> tuple[str, ...]:
    moments = policy_moments(records)
    routed_centers = {row.center_id for row in records if row.route_selected}
    failed = []
    if not routed_centers:
        failed.append("ZERO_ROUTED_CASES")
    if len(routed_centers) < minimum_centers:
        failed.append("INSUFFICIENT_CENTER_COVERAGE")
    if moments["gain"] <= 0.0:
        failed.append("NONPOSITIVE_SIGNED_GAIN")
    if moments["routed_harm"] > .25:
        failed.append("HARM_CONSTRAINT_FAILED")
    if moments["routed_brier_delta"] > .002:
        failed.append("BRIER_CONSTRAINT_FAILED")
    if moments["routed_logloss_delta"] > .005:
        failed.append("LOGLOSS_CONSTRAINT_FAILED")
    return tuple(failed)


def seal_selections(held: Sequence[HeldCandidatePrediction], threshold: float, *, enabled: bool = True) -> tuple[SealedOOFSelection, ...]:
    result = []
    for row in held:
        composite, score, _ = choose_candidate(row.menu, row.candidates, threshold, enabled=enabled)
        result.append(SealedOOFSelection(
            outer_fold=row.fold, composite=composite, requested_arm_id=POLICY_ARM_ID,
            route_score=score, route_threshold=threshold,
            training_case_keys=row.training_case_keys, model_hash=row.model_hash,
        ))
    return tuple(result)


def _summary(records: Sequence[SelectedOOFRecord | _FrontierObservation]) -> dict[str, object]:
    counts = Counter(row.center_id for row in records if row.route_selected)
    return {
        "case_count": len(records), "route_count": sum(counts.values()),
        "baseline_fallback_count": sum(not row.route_selected for row in records),
        "probability_changed_count": sum(row.probability_changed for row in records),
        "prediction_changed_count": sum(row.prediction_changed for row in records),
        "center_coverage": len(counts), "routed_cases_by_center": dict(sorted(counts.items())),
        "utility_risk_moments": policy_moments(records),
        "failed_constraints": list(failed_constraints(records)),
    }


@dataclass(frozen=True, slots=True)
class _FrontierObservation:
    """Already-authenticated candidate outcome; no redundant selection hash."""

    center_id: str
    case_id: str
    bacc_gain: float
    brier_delta: float
    log_loss_delta: float
    route_selected: bool
    probability_changed: bool
    prediction_changed: bool

    @property
    def harm(self) -> bool:
        return self.bacc_gain < 0.0


def build_candidate_frontier(
    held: Sequence[HeldCandidatePrediction], capability: SupportTruthCapability,
    *, thresholds: Sequence[float], stage: str, normalization_menus: Sequence[object] | None = None,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Project sealed candidate observations without refitting or resealing rows.

    The candidate prediction seals bind every candidate probability and estimate.
    This one frontier seal additionally binds the threshold grid and scoring case
    inventory before truth opens. The actual selected OOF seals are unchanged.
    """
    held = tuple(held)
    normalization_menus = tuple(row.menu for row in held) if normalization_menus is None else tuple(normalization_menus)
    arm_ids = tuple(dict.fromkeys(candidate.arm_id for row in held for candidate in row.candidates))
    pretruth_hash = canonical_hash({
        "prediction_seals": tuple(row.prediction_seal_hash for row in held),
        "thresholds": tuple(float(value) for value in thresholds),
        "normalization_case_keys": tuple((menu.center_id, menu.case_id) for menu in normalization_menus),
        "frontier_projection": "sealed_actual_composite_observation_replay_v18",
    })
    # The winner is invariant across thresholds until it abstains. Build it once
    # per case; threshold sweeps only choose cached winner or exact B records.
    baselines = tuple(build_baseline_composite(row.menu) for row in held)
    winners = tuple(choose_candidate(row.menu, row.candidates, 0.0) for row in held)
    unique = {value.candidate.composite.composite_hash: value.candidate.composite
              for row in held for value in row.candidates if value.candidate.composite is not None}
    unique.update({row.composite_hash: row for row in baselines})
    composites = tuple(unique.values())
    outcomes = capability.scoped(normalization_menus).score_composites(composites, normalized=True)
    observations = {
        outcome.composite.composite_hash: _FrontierObservation(
            outcome.composite.center_id, outcome.composite.case_id,
            outcome.bacc_gain, outcome.brier_delta, outcome.log_loss_delta,
            outcome.composite.route_selected, outcome.composite.probability_changed,
            outcome.composite.prediction_changed,
        ) for outcome in outcomes
    }
    base_observations = tuple(observations[row.composite_hash] for row in baselines)
    by_arm = tuple({value.arm_id: value for value in row.candidates} for row in held)
    output = []
    for arm_id in (*arm_ids, POLICY_ARM_ID):
        eligibility = Counter()
        raw_records, candidates, scores = [], [], []
        estimate_values = defaultdict(list)
        for ordinal, row in enumerate(held):
            base = base_observations[ordinal]
            if arm_id == POLICY_ARM_ID:
                winner, score, _ = winners[ordinal]
                eligible = any(value.candidate.eligible and value.arm_id != "B" for value in row.candidates)
                eligibility["eligible" if eligible else "ineligible"] += 1
                observation = observations[winner.composite_hash]
                raw_records.append(observation)
                candidates.append(observation)
                scores.append(score)
                estimate = next((value.prediction for value in row.candidates
                                 if value.candidate.composite is not None
                                 and value.candidate.composite.composite_hash == winner.composite_hash), None)
            else:
                value = by_arm[ordinal][arm_id]
                eligibility["eligible" if value.candidate.eligible else "ineligible"] += 1
                eligibility["duplicate"] += int(value.candidate.duplicate_of is not None)
                if value.candidate.ineligible_reason:
                    for reason in value.candidate.ineligible_reason.split(";"):
                        eligibility[reason] += 1
                if not value.screened:
                    eligibility["predictive_screen_rejected"] += 1
                if value.candidate.eligible and value.arm_id != "B" and not value.hard_prediction_changed:
                    eligibility["NO_HARD_PREDICTION_CHANGE"] += 1
                if value.prediction is not None:
                    prediction = value.prediction
                    for reason, failed in (
                        ("PREDICTED_NONPOSITIVE_GAIN", prediction.predicted_gain <= 0),
                        ("PREDICTED_HARM_EXCEEDS_LIMIT", prediction.predicted_harm > .25),
                        ("PREDICTED_BRIER_EXCEEDS_LIMIT", prediction.predicted_brier_delta > .002),
                        ("PREDICTED_LOGLOSS_EXCEEDS_LIMIT", prediction.predicted_logloss_delta > .005),
                    ):
                        eligibility[reason] += int(failed)
                raw_records.append(observations[value.candidate.composite.composite_hash] if value.candidate.eligible else base)
                candidates.append(observations[value.candidate.composite.composite_hash] if value.screened else base)
                scores.append(value.route_score)
                estimate = value.prediction if value.candidate.eligible else None
            if estimate is not None:
                estimate_values[row.menu.center_id].append((
                    estimate.predicted_gain, estimate.predicted_harm,
                    estimate.predicted_brier_delta, estimate.predicted_logloss_delta,
                    estimate.safe_positive_probability, estimate.approximate_gain_lower_score,
                ))
        estimate_names = ("predicted_gain", "predicted_harm", "predicted_brier_delta",
                          "predicted_logloss_delta", "safe_positive_probability",
                          "approximate_gain_lower_score")
        estimate_means = (dict(zip(estimate_names, map(float, np.mean([
            np.mean(values, axis=0) for _, values in sorted(estimate_values.items())
        ], axis=0)), strict=True)) if estimate_values else {name: None for name in estimate_names})
        raw_moments = policy_moments(raw_records)
        raw_safe_count = sum(row.bacc_gain > 0 and row.brier_delta <= 0 and row.log_loss_delta <= 0 for row in raw_records)
        for threshold in thresholds:
            threshold = float(threshold)
            counts = eligibility.copy()
            if arm_id != POLICY_ARM_ID:
                below_count = sum(score <= threshold for score in scores)
                if below_count:
                    counts["not_above_threshold"] += below_count
            records = tuple(candidate if score > threshold else base for candidate, score, base in zip(candidates, scores, base_observations, strict=True))
            payload = {
                "stage": stage, "folds": sorted({row.fold for row in held}),
                "arm_id": arm_id, "threshold": threshold,
                "eligible_count": counts["eligible"], "ineligible_count": counts["ineligible"],
                "duplicate_count": counts["duplicate"], "eligibility_and_screen_counts": dict(sorted(counts.items())),
                **_summary(records),
                "prediction_means_among_available_candidates": estimate_means,
                "prediction_estimate_case_count": sum(len(values) for values in estimate_values.values()),
                "prediction_estimate_center_count": len(estimate_values),
                "prediction_means_are_equal_center_case_means": True,
                "lower_score_is_heuristic_not_confidence_bound": True,
                "raw_eligible_candidate_moments_before_predictive_screen": raw_moments,
                "raw_eligible_safe_positive_count": raw_safe_count,
                "pretruth_candidate_prediction_seal_hash": pretruth_hash,
                "normalization_case_keys": tuple((menu.center_id, menu.case_id) for menu in normalization_menus),
                "summary_case_keys": tuple((row.menu.center_id, row.menu.case_id) for row in held),
                "fold_moments_use_complete_oof_scope_class_weights": True,
                "candidate_outcomes_diagnostic_only": True, "raw_labels_persisted": False,
            }
            output.append({**payload, "frontier_row_hash": canonical_hash(payload)})
    oracle_by_center = defaultdict(list)
    safe_positive = 0
    for row in held:
        available = [observations[value.candidate.composite.composite_hash] for value in row.candidates
                     if value.candidate.composite is not None and value.candidate.duplicate_of is None]
        safe = [value.bacc_gain for value in available if value.brier_delta <= 0.0 and value.log_loss_delta <= 0.0]
        best = max((0.0, *safe))
        oracle_by_center[row.menu.center_id].append(best)
        safe_positive += int(best > 0.0)
    oracle = {
        "stage": stage, "case_count": len(held), "safe_positive_case_count": safe_positive,
        "actual_proposed_menu_safe_oracle_gain": float(np.mean([np.mean(values) for values in oracle_by_center.values()])),
        "pretruth_candidate_prediction_seal_hash": pretruth_hash,
        "oracle_is_retrospective_diagnostic_only": True, "oracle_used_for_selection": False,
    }
    return tuple(output), {**oracle, "oracle_hash": canonical_hash(oracle)}
