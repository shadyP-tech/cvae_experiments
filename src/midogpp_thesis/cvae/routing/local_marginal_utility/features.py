"""Label-free feature construction for local utility prediction.

Only compatibility energies and source identities enter this module.  Utility,
class labels, evaluation predictions, and outer-target outcomes are absent from
the API by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError


ENERGY_FEATURE_NAMES = (
    "calibrated_energy_z",
    "query_centered_energy_z",
    "query_scaled_energy_z",
    "query_rank_fraction",
    "query_gap_from_best_z",
)
SOURCE_INDICATOR_PREFIX = "source_indicator::"


@dataclass(frozen=True)
class LabelFreeFeatureMatrix:
    """Deterministically keyed label-free design matrix."""

    row_keys: tuple[tuple[str, str], ...]
    feature_names: tuple[str, ...]
    values: np.ndarray

    @property
    def query_clusters(self) -> tuple[str, ...]:
        return tuple(query for query, _source in self.row_keys)

    @property
    def source_centers(self) -> tuple[str, ...]:
        return tuple(source for _query, source in self.row_keys)


def build_energy_feature_matrix(
    calibrated_energy_by_query: Mapping[str, Mapping[str, float]],
    *,
    candidate_sources_by_query: Mapping[str, Sequence[str]] | None = None,
    include_source_indicators: bool = True,
) -> LabelFreeFeatureMatrix:
    """Construct within-query energy geometry without reading any utility.

    Rows are sorted by ``(query_cluster, source_center)``.  Source indicators
    use the lexicographically first source as the dropped reference category.
    Within-query ranks use average ranks for exact ties and are scaled to
    ``[0, 1]`` with zero denoting the lowest compatibility energy.
    """

    if not calibrated_energy_by_query:
        raise ProtocolError("Label-free energy features require at least one query.")
    queries = tuple(sorted(str(query) for query in calibrated_energy_by_query))
    if len(set(queries)) != len(queries) or any(not query for query in queries):
        raise ProtocolError("Query cluster IDs must be unique and nonempty.")
    if candidate_sources_by_query is not None and set(candidate_sources_by_query) != set(
        calibrated_energy_by_query
    ):
        raise ProtocolError("Candidate-source geometry must cover every query exactly.")

    normalized: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    all_sources: set[str] = set()
    for query in queries:
        raw_scores = calibrated_energy_by_query[query]
        if not raw_scores:
            raise ProtocolError("Every query must have at least one compatibility score.")
        if candidate_sources_by_query is None:
            sources = _canonical_sources(tuple(raw_scores))
        else:
            sources = _canonical_sources(candidate_sources_by_query[query])
            if set(raw_scores) != set(sources):
                raise ProtocolError(
                    "Compatibility scores must exactly cover the declared legal candidates."
                )
        scores = np.asarray([float(raw_scores[source]) for source in sources], dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ProtocolError("Compatibility energies must be finite.")
        normalized[query] = (sources, scores)
        all_sources.update(sources)

    source_universe = tuple(sorted(all_sources))
    indicator_sources = source_universe[1:] if include_source_indicators else ()
    feature_names = ENERGY_FEATURE_NAMES + tuple(
        f"{SOURCE_INDICATOR_PREFIX}{source}" for source in indicator_sources
    )
    row_keys: list[tuple[str, str]] = []
    rows: list[list[float]] = []
    for query in queries:
        sources, scores = normalized[query]
        centered = scores - float(np.mean(scores, dtype=np.float64))
        rms = float(np.sqrt(np.mean(centered * centered, dtype=np.float64)))
        scaled = centered / rms if rms > np.finfo(np.float64).eps else np.zeros_like(centered)
        ranks = _average_rank_fraction(scores)
        gaps = scores - float(scores.min())
        for index, source in enumerate(sources):
            row_keys.append((query, source))
            rows.append(
                [
                    float(scores[index]),
                    float(centered[index]),
                    float(scaled[index]),
                    float(ranks[index]),
                    float(gaps[index]),
                    *(1.0 if source == indicator else 0.0 for indicator in indicator_sources),
                ]
            )
    values = np.asarray(rows, dtype=np.float64)
    if values.shape != (len(row_keys), len(feature_names)) or not np.isfinite(values).all():
        raise ProtocolError("Label-free feature construction produced invalid geometry.")
    values.setflags(write=False)
    return LabelFreeFeatureMatrix(
        row_keys=tuple(row_keys),
        feature_names=feature_names,
        values=values,
    )


def _average_rank_fraction(values: np.ndarray) -> np.ndarray:
    count = len(values)
    if count == 1:
        return np.zeros(1, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(count, dtype=np.float64)
    cursor = 0
    while cursor < count:
        end = cursor + 1
        while end < count and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = 0.5 * float(cursor + end - 1)
        ranks[order[cursor:end]] = average_rank
        cursor = end
    return ranks / float(count - 1)


def _canonical_sources(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if (
        not normalized
        or any(not source or source.strip() != source for source in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise ProtocolError("Candidate source IDs must be unique, nonempty, and canonical.")
    return tuple(sorted(normalized))


__all__ = (
    "ENERGY_FEATURE_NAMES",
    "SOURCE_INDICATOR_PREFIX",
    "LabelFreeFeatureMatrix",
    "build_energy_feature_matrix",
)
