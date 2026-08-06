"""Paired three-level uncertainty gate for source-inner regret selection."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    BOOTSTRAP_MAX_ATTEMPTS,
    BOOTSTRAP_SEED,
    BOOTSTRAP_VALID_REPLICATES,
    CENTERS,
    GENERATION_SEEDS,
    MARGIN_LOWER_QUANTILE,
    TRAINING_SEEDS,
    WIN_PROBABILITY_THRESHOLD,
    BootstrapResult,
    CandidateSummary,
)


ConfusionKey = tuple[str, str, int, int, str]


def validate_case_confusions(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[ConfusionKey, tuple[int, int, int, int]], dict[str, tuple[str, ...]]]:
    """Validate paired per-case confusion counts for every legal utility cell."""

    indexed: dict[ConfusionKey, tuple[int, int, int, int]] = {}
    cases_by_query: dict[str, set[str]] = {query: set() for query in CENTERS}
    observed_sets: dict[tuple[str, str, int, int], set[str]] = {}
    for row in rows:
        query = str(row.get("pseudo_target_center", ""))
        candidate = str(row.get("candidate_source_center", ""))
        training_seed = _integer(row.get("training_seed"), "training_seed")
        generation_seed = _integer(row.get("generation_seed"), "generation_seed")
        case_id = str(row.get("case_id", ""))
        if (
            query not in CENTERS
            or candidate not in CENTERS
            or query == candidate
            or training_seed not in TRAINING_SEEDS
            or generation_seed not in GENERATION_SEEDS
            or not case_id
        ):
            raise ProtocolError("Illegal source-inner case-confusion key.")
        key = (query, candidate, training_seed, generation_seed, case_id)
        if key in indexed:
            raise ProtocolError(f"Duplicate source-inner case-confusion key: {key}.")
        counts = tuple(
            _nonnegative_integer(row.get(field), field)
            for field in ("tn", "fp", "fn", "tp")
        )
        n_rows = _nonnegative_integer(row.get("n"), "n")
        if sum(counts) != n_rows or n_rows <= 0:
            raise ProtocolError("Source-inner case-confusion row count drifted.")
        if row.get("eval_labels_used_for_scoring_only") is not True:
            raise ProtocolError("Case-confusion labels are not marked scoring-only.")
        indexed[key] = counts  # type: ignore[assignment]
        cases_by_query[query].add(case_id)
        observed_sets.setdefault((query, candidate, training_seed, generation_seed), set()).add(case_id)
    if not indexed:
        raise ProtocolError("Source-inner case-confusion table is empty.")
    expected_cells = {
        (query, candidate, training_seed, generation_seed)
        for query in CENTERS
        for candidate in CENTERS
        if candidate != query
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    }
    if set(observed_sets) != expected_cells:
        raise ProtocolError("Source-inner case-confusion cell coverage drifted.")
    for cell, cases in observed_sets.items():
        if cases != cases_by_query[cell[0]]:
            raise ProtocolError("Source-inner case pools are not paired across candidates/seeds.")
    if any(not cases for cases in cases_by_query.values()):
        raise ProtocolError("A pseudo-target has no validation cases.")
    frozen_cases = {
        query: tuple(sorted(cases_by_query[query]))
        for query in CENTERS
    }
    return indexed, frozen_cases


def reconstruct_bacc(
    index: Mapping[ConfusionKey, tuple[int, int, int, int]],
    *,
    query: str,
    candidate: str,
    training_seed: int,
    generation_seed: int,
    cases: Sequence[str],
) -> float | None:
    tn = fp = fn = tp = 0
    for case_id in cases:
        try:
            a, b, c, d = index[(query, candidate, training_seed, generation_seed, str(case_id))]
        except KeyError as exc:
            raise ProtocolError("Missing paired case-confusion row.") from exc
        tn += a
        fp += b
        fn += c
        tp += d
    negative = tn + fp
    positive = tp + fn
    if negative == 0 or positive == 0:
        return None
    return 0.5 * ((tn / negative) + (tp / positive))


def bootstrap_outer_policy(
    *,
    outer_target_center: str,
    summaries: Sequence[CandidateSummary],
    case_rows: Sequence[Mapping[str, object]],
    valid_replicates: int = BOOTSTRAP_VALID_REPLICATES,
    max_attempts: int = BOOTSTRAP_MAX_ATTEMPTS,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapResult:
    """Run the fixed paired q/case/seed bootstrap for one outer fold."""

    outer = str(outer_target_center)
    if outer not in CENTERS:
        raise ProtocolError("Unknown utility/regret outer center.")
    if valid_replicates <= 0 or max_attempts < valid_replicates:
        raise ProtocolError("Invalid utility/regret bootstrap budget.")
    index, cases_by_query = validate_case_confusions(case_rows)
    observed = [item for item in summaries if item.outer_target_center == outer]
    candidates = tuple(candidate for candidate in CENTERS if candidate != outer)
    if {item.candidate_source for item in observed} != set(candidates) or len(observed) != 8:
        raise ProtocolError("Outer candidate summaries are incomplete.")
    by_candidate = {item.candidate_source: item for item in observed}
    ordered = sorted(observed, key=lambda item: (item.mean_regret, CENTERS.index(item.candidate_source)))
    best = ordered[0]
    runner = ordered[1]
    best_ties = [
        item for item in ordered if math.isclose(item.mean_regret, best.mean_regret, abs_tol=1.0e-12)
    ]
    unique_observed = len(best_ties) == 1
    observed_margin = runner.mean_regret - best.mean_regret
    queries = tuple(query for query in CENTERS if query != outer)
    seed_pairs = tuple(
        (training_seed, generation_seed)
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )
    rng = np.random.default_rng(int(seed))
    winner_count = 0
    margins: list[float] = []
    attempts = rejected = 0
    while len(margins) < valid_replicates and attempts < max_attempts:
        attempts += 1
        query_draws = tuple(str(value) for value in rng.choice(queries, size=len(queries), replace=True))
        regrets: dict[str, list[float]] = defaultdict(list)
        valid = True
        for query in query_draws:
            available_cases = cases_by_query[query]
            case_draws = tuple(
                str(value)
                for value in rng.choice(
                    available_cases, size=len(available_cases), replace=True
                )
            )
            seed_indices = rng.integers(0, len(seed_pairs), size=len(seed_pairs))
            sampled_pairs = tuple(
                seed_pairs[int(index_value)] for index_value in seed_indices
            )
            legal_candidates = tuple(
                candidate for candidate in candidates if candidate != query
            )
            for training_seed, generation_seed in sampled_pairs:
                utility: dict[str, float] = {}
                for candidate in legal_candidates:
                    value = reconstruct_bacc(
                        index,
                        query=query,
                        candidate=candidate,
                        training_seed=training_seed,
                        generation_seed=generation_seed,
                        cases=case_draws,
                    )
                    if value is None:
                        valid = False
                        break
                    utility[candidate] = value
                if not valid:
                    break
                oracle = max(utility.values())
                for candidate, value in utility.items():
                    regrets[candidate].append(oracle - value)
            if not valid:
                break
        if not valid or any(not regrets.get(candidate) for candidate in candidates):
            rejected += 1
            continue
        means = {
            candidate: sum(regrets[candidate]) / len(regrets[candidate])
            for candidate in candidates
        }
        minimum = min(means.values())
        winners = [
            candidate
            for candidate, value in means.items()
            if math.isclose(value, minimum, abs_tol=1.0e-12)
        ]
        if len(winners) == 1 and winners[0] == best.candidate_source:
            winner_count += 1
        margins.append(means[runner.candidate_source] - means[best.candidate_source])
    if len(margins) != valid_replicates:
        raise ProtocolError(
            "Utility/regret bootstrap could not obtain the required valid replicates."
        )
    lower, upper = np.quantile(
        np.asarray(margins, dtype=np.float64),
        [MARGIN_LOWER_QUANTILE, 1.0 - MARGIN_LOWER_QUANTILE],
    ).tolist()
    win_probability = winner_count / valid_replicates
    gate_passed = (
        unique_observed
        and win_probability >= WIN_PROBABILITY_THRESHOLD
        and float(lower) > 0.0
    )
    if not unique_observed:
        reason = "observed_minimum_regret_tie"
    elif win_probability < WIN_PROBABILITY_THRESHOLD:
        reason = "unique_winner_probability_below_threshold"
    elif float(lower) <= 0.0:
        reason = "paired_regret_margin_lower_bound_not_positive"
    else:
        reason = "all_uncertainty_gates_passed"
    return BootstrapResult(
        outer_target_center=outer,
        observed_best_source=best.candidate_source,
        observed_runner_up_source=runner.candidate_source,
        observed_best_mean_regret=best.mean_regret,
        observed_runner_up_mean_regret=runner.mean_regret,
        observed_margin=observed_margin,
        unique_observed_winner=unique_observed,
        unique_winner_probability=win_probability,
        margin_lower_2_5=float(lower),
        margin_upper_97_5=float(upper),
        valid_replicates=valid_replicates,
        attempted_replicates=attempts,
        rejected_replicates=rejected,
        gate_passed=gate_passed,
        gate_reason=reason,
    )


def _integer(value: object, label: str) -> int:
    try:
        return int(str(value))
    except ValueError as exc:
        raise ProtocolError(f"Case-confusion {label} is invalid.") from exc


def _nonnegative_integer(value: object, label: str) -> int:
    observed = _integer(value, label)
    if observed < 0:
        raise ProtocolError(f"Case-confusion {label} is negative.")
    return observed


__all__ = (
    "bootstrap_outer_policy",
    "reconstruct_bacc",
    "validate_case_confusions",
)
