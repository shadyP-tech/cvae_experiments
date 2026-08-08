"""Domain-bootstrap ranking metrics for nested utility routing evaluation."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256
from .result_contracts import RankingMetrics
from .row_contracts import (
    INNER_CANDIDATE_COUNT,
    SEED_PAIR_COUNT,
    TARGET_CANDIDATE_COUNT,
)
from .surface_contracts import FeatureSurface


_GATE_TOLERANCE = 1.0e-12
_DOMAIN_BOOTSTRAP_REPLICATES = 10_000
_DOMAIN_BOOTSTRAP_SEED = 90_701
_ONE_SIDED_ALPHA = 0.05


def _ranking_metrics(
    surface: FeatureSurface,
    utility: np.ndarray,
    predictions: np.ndarray,
    *,
    model_role: str,
) -> tuple[RankingMetrics, dict[str, np.ndarray]]:
    if utility.shape != predictions.shape or utility.shape != (len(surface.rows),):
        raise ProtocolError("Ranking metric vectors do not align with the feature surface.")
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(surface.rows):
        grouped[(row.query_id, row.candidate_source)].append(index)
    aggregate: dict[tuple[str, str], tuple[float, float]] = {}
    for key, indices in grouped.items():
        if len(indices) != SEED_PAIR_COUNT:
            raise ProtocolError("Ranking metrics require all nine seed pairs per candidate.")
        aggregate[key] = (
            float(np.mean(predictions[indices], dtype=np.float64)),
            float(np.mean(utility[indices], dtype=np.float64)),
        )
    query_ids = tuple(sorted({query for query, _source in aggregate}))
    top1: list[float] = []
    correlations: list[float] = []
    gaps: list[float] = []
    pairwise: list[float] = []
    pairwise_by_query: list[float] = []
    selected_utility: list[float] = []
    oracle_utility: list[float] = []
    positive_gain: list[float] = []
    for query in query_ids:
        sources = tuple(sorted(source for q, source in aggregate if q == query))
        if len(sources) != INNER_CANDIDATE_COUNT:
            raise ProtocolError("Ranking query is not a seven-source list.")
        predicted = np.asarray([aggregate[(query, source)][0] for source in sources])
        truth = np.asarray([aggregate[(query, source)][1] for source in sources])
        selected_index = min(
            range(len(sources)), key=lambda index: (-float(predicted[index]), sources[index])
        )
        best = float(np.max(truth))
        worst = float(np.min(truth))
        selected = float(truth[selected_index])
        top1.append(float(abs(selected - best) <= _GATE_TOLERANCE))
        selected_utility.append(selected)
        oracle_utility.append(best)
        positive_gain.append(float(selected > _GATE_TOLERANCE))
        denominator = best - worst
        gaps.append(0.0 if denominator <= _GATE_TOLERANCE else (best - selected) / denominator)
        correlation = _spearman(predicted, truth)
        correlations.append(0.0 if correlation is None else correlation)
        query_pairwise: list[float] = []
        for left, right in combinations(range(len(sources)), 2):
            true_sign = _sign(float(truth[left] - truth[right]))
            predicted_sign = _sign(float(predicted[left] - predicted[right]))
            if true_sign == predicted_sign:
                score = 1.0
            elif true_sign == 0 or predicted_sign == 0:
                score = 0.5
            else:
                score = 0.0
            pairwise.append(score)
            query_pairwise.append(score)
        pairwise_by_query.append(float(np.mean(query_pairwise, dtype=np.float64)))
    query_values = {
        "top1": np.asarray(top1, dtype=np.float64),
        "spearman": np.asarray(correlations, dtype=np.float64),
        "normalized_gap": np.asarray(gaps, dtype=np.float64),
        "pairwise_accuracy": np.asarray(pairwise_by_query, dtype=np.float64),
        "selected_utility": np.asarray(selected_utility, dtype=np.float64),
        "oracle_utility": np.asarray(oracle_utility, dtype=np.float64),
        "positive_gain": np.asarray(positive_gain, dtype=np.float64),
    }
    values = {
        "query_count": len(query_ids),
        "seed_pair_count": SEED_PAIR_COUNT,
        "top1_oracle_agreement": float(np.mean(top1, dtype=np.float64)),
        "top1_lower_bound": _bootstrap_bound(query_values["top1"], upper=False),
        "mean_spearman": float(np.mean(correlations, dtype=np.float64)),
        "spearman_lower_bound": _bootstrap_bound(
            query_values["spearman"], upper=False
        ),
        "defined_spearman_queries": sum(
            _spearman_defined(
                np.asarray(
                    [aggregate[(query, source)][0] for source in sorted(source for q, source in aggregate if q == query)]
                ),
                np.asarray(
                    [aggregate[(query, source)][1] for source in sorted(source for q, source in aggregate if q == query)]
                ),
            )
            for query in query_ids
        ),
        "mean_normalized_oracle_gap": float(np.mean(gaps, dtype=np.float64)),
        "normalized_oracle_gap_upper_bound": _bootstrap_bound(
            query_values["normalized_gap"], upper=True
        ),
        "pairwise_accuracy": float(np.mean(pairwise, dtype=np.float64)),
        "mean_selected_utility_delta": float(np.mean(selected_utility, dtype=np.float64)),
        "selected_utility_lower_bound": _bootstrap_bound(
            query_values["selected_utility"], upper=False
        ),
        "mean_oracle_utility_delta": float(np.mean(oracle_utility, dtype=np.float64)),
        "positive_selected_gain_rate": float(np.mean(positive_gain, dtype=np.float64)),
    }
    payload = {
        "schema_version": "midogpp_utility_aligned_ranking_metrics_v1",
        "model_role": model_role,
        **values,
        "query_domains_are_independent_units": True,
        "seed_selection_performed": False,
    }
    return RankingMetrics(**values, metrics_hash=canonical_sha256(payload)), query_values


def _bootstrap_bound(values: np.ndarray, *, upper: bool) -> float:
    if values.ndim != 1 or len(values) != TARGET_CANDIDATE_COUNT:
        raise ProtocolError("Domain bootstrap requires exactly eight query-domain values.")
    rng = np.random.default_rng(_DOMAIN_BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        len(values),
        size=(_DOMAIN_BOOTSTRAP_REPLICATES, len(values)),
    )
    means = np.mean(values[indices], axis=1, dtype=np.float64)
    quantile = 1.0 - _ONE_SIDED_ALPHA if upper else _ONE_SIDED_ALPHA
    return float(np.quantile(means, quantile, method="linear"))


def _paired_bootstrap_lower_bound(values: np.ndarray) -> float:
    return _bootstrap_bound(values, upper=False)


def _spearman_defined(left: np.ndarray, right: np.ndarray) -> int:
    return int(_spearman(left, right) is not None)


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_centered = left_rank - float(left_rank.mean())
    right_centered = right_rank - float(right_rank.mean())
    denominator = float(
        np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered))
    )
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * float(cursor + end - 1)
        cursor = end
    return ranks


def _sign(value: float) -> int:
    if value > _GATE_TOLERANCE:
        return 1
    if value < -_GATE_TOLERANCE:
        return -1
    return 0


__all__: tuple[str, ...] = ()
