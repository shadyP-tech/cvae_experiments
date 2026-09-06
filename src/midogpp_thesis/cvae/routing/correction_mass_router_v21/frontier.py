"""Complete source-only candidate frontiers after label-free prediction seals."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from .composition import build_baseline_composite
from .hashing import canonical_hash
from .decision_evidence import decision_evidence
from .records import SealedOOFSelection, SelectedOOFRecord
from .stacked_fitting import HeldCandidatePrediction, POLICY_ARM_ID, choose_candidate, unthresholded_winner
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


def failed_constraints(records: Sequence[SelectedOOFRecord | _FrontierObservation], *, minimum_centers: int = 1, minimum_cases: int = 1, minimum_cases_per_center: int = 1) -> tuple[str, ...]:
    moments = policy_moments(records)
    routed_centers = {row.center_id for row in records if row.route_selected}
    failed = []
    if not routed_centers:
        failed.append("ZERO_ROUTED_CASES")
    counts = Counter(row.center_id for row in records if row.route_selected)
    if sum(counts.values()) < minimum_cases:
        failed.append("INSUFFICIENT_ROUTED_CASES")
    if sum(n >= minimum_cases_per_center for n in counts.values()) < minimum_centers:
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
        composite, score, reason = choose_candidate(row.menu, row.candidates, threshold, enabled=enabled, winner_prediction=row.winner_prediction)
        result.append(SealedOOFSelection(
            outer_fold=row.fold, composite=composite, requested_arm_id=POLICY_ARM_ID,
            route_score=score, route_threshold=threshold,
            policy_enabled=enabled, fallback_reason=reason,
            training_case_keys=row.training_case_keys, model_hash=row.model_hash,
            **decision_evidence(unthresholded_winner(row.candidates), row.winner_prediction),
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


def _estimate_means(held, estimates, mask):
    """Condition the all-case equal-center weights on the exact selected mask."""
    sizes = Counter(row.menu.center_id for row in held)
    totals = np.zeros(6)
    mass = np.zeros(6)
    for row, estimate, selected in zip(held, estimates, mask, strict=True):
        if estimate is not None and selected:
            weight = 1.0 / len(sizes) / sizes[row.menu.center_id]
            for index, value in enumerate(estimate):
                if value is not None:
                    totals[index] += weight*value
                    mass[index] += weight
    names = ("predicted_gain", "predicted_harm", "predicted_brier_delta",
             "predicted_logloss_delta", "safe_positive_probability", "risk_adjusted_score")
    return {name: float(totals[i]/mass[i]) if mass[i] else None for i, name in enumerate(names)}


def build_candidate_frontier(held, capability, *, thresholds, stage,
                             normalization_menus=None, include_detailed_joins=True):
    """Replay sealed predictions; threshold sweeps never refit or change winners."""
    held = tuple(held)
    normalization_menus = tuple(row.menu for row in held) if normalization_menus is None else tuple(normalization_menus)
    arm_ids = tuple(dict.fromkeys(value.arm_id for row in held for value in row.candidates))
    normalization_keys = tuple((menu.center_id, menu.case_id) for menu in normalization_menus)
    pretruth_hash = canonical_hash({"prediction_seals": tuple(row.prediction_seal_hash for row in held),
        "thresholds": tuple(float(value) for value in thresholds),
        "normalization_case_keys": normalization_keys,
        "frontier_projection": "sealed_complete_winner_observation_replay_v21"})
    baselines = tuple(build_baseline_composite(row.menu) for row in held)
    winners = tuple(unthresholded_winner(row.candidates) for row in held)
    unique = {value.candidate.composite.composite_hash: value.candidate.composite
        for row in held for value in row.candidates if value.candidate.composite is not None}
    unique.update({row.composite_hash: row for row in baselines})
    outcomes = capability.scoped(normalization_menus).score_composites(tuple(unique.values()), normalized=True)
    outcome_by_hash = {row.composite.composite_hash: row for row in outcomes}
    observations = {row.composite.composite_hash: _FrontierObservation(
        row.composite.center_id, row.composite.case_id, row.bacc_gain,
        row.brier_delta, row.log_loss_delta, row.composite.route_selected,
        row.composite.probability_changed, row.composite.prediction_changed) for row in outcomes}
    bases = tuple(observations[row.composite_hash] for row in baselines)
    by_arm = tuple({value.arm_id: value for value in row.candidates} for row in held)
    output = []
    for arm_id in (*arm_ids, POLICY_ARM_ID):
        counts, raw, estimates, values = Counter(), [], [], []
        for ordinal, row in enumerate(held):
            value = winners[ordinal] if arm_id == POLICY_ARM_ID else by_arm[ordinal].get(arm_id)
            values.append(value)
            eligible = value is not None and value.candidate.eligible
            counts["eligible" if eligible else "ineligible"] += 1
            counts["duplicate"] += int(value is not None and value.candidate.duplicate_of is not None)
            if value is not None:
                for reason in (value.candidate.ineligible_reason or "").split(";"):
                    if reason:
                        counts[reason] += 1
                counts["NO_HARD_PREDICTION_CHANGE"] += int(eligible and value.arm_id != "B" and not value.hard_prediction_changed)
                counts["PREDICTED_NONPOSITIVE_GAIN"] += int(eligible and value.risk_adjusted_score <= 0)
                counts["PREDICTED_BRIER_SCREEN_FAILED"] += int(value.prediction is not None and value.prediction.predicted_brier_delta > .002)
                counts["PREDICTED_LOGLOSS_SCREEN_FAILED"] += int(value.prediction is not None and value.prediction.predicted_logloss_delta > .005)
            raw.append(observations[value.candidate.composite.composite_hash] if eligible else bases[ordinal])
            p = None if value is None else value.prediction
            gate = row.winner_prediction if arm_id == POLICY_ARM_ID else None
            estimates.append(None if p is None else (
                p.predicted_gain, None if gate is None else gate.harm_probability,
                p.predicted_brier_delta, p.predicted_logloss_delta,
                None, p.predicted_gain))
        for threshold in thresholds:
            records, mask = [], []
            for ordinal, row in enumerate(held):
                value = values[ordinal]
                if arm_id == POLICY_ARM_ID:
                    composite, _, _ = choose_candidate(row.menu, row.candidates, threshold,
                                                       winner_prediction=row.winner_prediction)
                    observation = observations[composite.composite_hash]
                else:
                    # Candidate rows describe a raw-candidate score frontier,
                    # not a counterfactual gate that was never trained on it.
                    take = value is not None and value.screened
                    observation = raw[ordinal] if take else bases[ordinal]
                records.append(observation)
                mask.append(observation.route_selected)
            payload = {"stage": stage, "folds": sorted({row.fold for row in held}),
                "evidence_variants": tuple(sorted({row.patch_control.evidence_variant for row in held if row.patch_control is not None})),
                "arm_id": arm_id, "threshold": float(threshold),
                "threshold_score_scope": "COMPLETE_WINNER_GATE" if arm_id == POLICY_ARM_ID else "CANDIDATE_MEAN_EFFECT_SCREEN_ONLY_THRESHOLD_NOT_APPLIED",
                "eligible_count": counts["eligible"], "ineligible_count": counts["ineligible"],
                "duplicate_count": counts["duplicate"], "eligibility_and_screen_counts": dict(sorted(counts.items())),
                **_summary(records),
                "prediction_means_among_available_candidates": _estimate_means(held, estimates, [True] * len(held)),
                "prediction_means_among_routed_candidates": _estimate_means(held, estimates, mask),
                "prediction_means_are_threshold_matched": True,
                "prediction_means_use_all_case_equal_center_weights_conditioned_on_mask": True,
                "prediction_means_are_equal_center_case_means": True,
                "prediction_estimate_case_count": sum(value is not None for value in estimates),
                "raw_eligible_candidate_moments_before_predictive_screen": policy_moments(raw),
                "raw_eligible_safe_positive_count": sum(r.bacc_gain > 0 and r.brier_delta <= 0 and r.log_loss_delta <= 0 for r in raw),
                "pretruth_candidate_prediction_seal_hash": pretruth_hash,
                "normalization_case_keys": normalization_keys,
                "summary_case_keys": tuple((row.menu.center_id, row.menu.case_id) for row in held),
                "fold_moments_use_complete_oof_scope_class_weights": True,
                "individual_predicted_proper_loss_screen_used": True,
                "candidate_harm_probability_estimated": False,
                "threshold_rows_are_enabled_policy_diagnostics": True,
                "actual_nested_policy_admission_applied": False,
                "candidate_outcomes_diagnostic_only": True, "raw_labels_persisted": False}
            output.append({**payload, "frontier_row_hash": canonical_hash(payload)})
    from .frontier_joins import detailed_prediction_joins
    joins, winner_diagnostics = detailed_prediction_joins(held, outcome_by_hash, thresholds, stage, pretruth_hash) if include_detailed_joins else ((), ())
    oracle_centers = defaultdict(list)
    safe_count = 0
    for row in held:
        available = [outcome_by_hash[value.candidate.composite.composite_hash] for value in row.candidates
            if value.candidate.composite is not None and value.candidate.duplicate_of is None]
        safe = [value.bacc_gain for value in available if value.safe_positive]
        safe_count += bool(safe)
        oracle_centers[row.menu.center_id].append(max((0.0, *safe)))
    controls = tuple(row.patch_control for row in held if row.patch_control is not None)
    control_scores = capability.scoped(normalization_menus).score_patch_controls(controls, composites=tuple(unique.values())) if controls else ()
    diagnostic = {"stage": stage,
        "evidence_variants": tuple(sorted({row.patch_control.evidence_variant for row in held if row.patch_control is not None})),
        "patch_evidence_direct_control": control_scores, "case_count": len(held),
        "safe_positive_case_count": safe_count,
        "equal_center_proper_loss_safe_oracle_gain": float(np.mean([np.mean(v) for v in oracle_centers.values()])),
        "oracle_used_for_selection": False,
        "pretruth_candidate_prediction_seal_hash": pretruth_hash,
        "candidate_prediction_outcome_joins": joins, "winner_gate_diagnostics": winner_diagnostics,
        "raw_labels_persisted": False}
    return tuple(output), {**diagnostic, "diagnostic_hash": canonical_hash(diagnostic)}
