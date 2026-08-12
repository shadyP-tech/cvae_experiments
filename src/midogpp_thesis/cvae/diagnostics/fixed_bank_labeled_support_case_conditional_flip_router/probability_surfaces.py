"""Exact-nine aggregation and label-free threshold-flip features."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .actions import actions_for_target
from .constants import B_ACTION_ID, CENTERS, HARD_THRESHOLD, SEED_PAIR_COUNT
from .hashing import canonical_hash
from .products import (
    AggregatedProbabilityRow,
    CaseActionFeature,
    ExactNineProbabilitySurface,
    PrelabelSurface,
    SeedProbabilityRow,
)


def seed_probability_rows(prediction: object) -> tuple[SeedProbabilityRow, ...]:
    store = getattr(prediction, "store")
    rows: list[SeedProbabilityRow] = []
    seed_pairs = tuple((17, generation) for generation in (17, 42, 101)) + tuple(
        (training, generation)
        for training in (42, 101)
        for generation in (17, 42, 101)
    )
    for target in CENTERS:
        row_ids = store.rows_by_center[target]
        case_ids = store.case_ids_by_center[target]
        for action in actions_for_target(target):
            for ordinal, (training, generation) in enumerate(seed_pairs):
                values = store.probabilities(target, action.action_id, training, generation)
                for sample_id, case_id, probability in zip(row_ids, case_ids, values, strict=True):
                    rows.append(SeedProbabilityRow(
                        target, case_id, sample_id, action.action_id, ordinal,
                        float(probability), store.store_hash,
                    ))
    return tuple(rows)


def aggregate_exact_nine(
    rows: Sequence[SeedProbabilityRow],
) -> ExactNineProbabilitySurface:
    grouped: dict[tuple[str, str, str, str], list[SeedProbabilityRow]] = defaultdict(list)
    store_hashes = set()
    for row in rows:
        grouped[(row.target_center, row.case_id, row.sample_id, row.action_id)].append(row)
        store_hashes.add(row.probability_store_hash)
    if len(store_hashes) != 1:
        raise ProtocolError("Flip-router seed rows span multiple prediction stores.")
    result: list[AggregatedProbabilityRow] = []
    for key in sorted(grouped):
        values_by_seed = sorted(grouped[key], key=lambda row: row.seed_pair_ordinal)
        if tuple(row.seed_pair_ordinal for row in values_by_seed) != tuple(range(SEED_PAIR_COUNT)):
            raise ProtocolError("Flip-router exact-nine seed coverage drifted.")
        values = np.asarray([row.probability for row in values_by_seed], dtype=np.float64)
        result.append(AggregatedProbabilityRow(
            *key,
            probability_mean=float(np.mean(values, dtype=np.float64)),
            probability_sd=float(np.std(values, ddof=0, dtype=np.float64)),
            seed_pair_count=SEED_PAIR_COUNT,
            seed_probabilities=tuple(float(value) for value in values),
        ))
    store_hash = next(iter(store_hashes))
    surface_payload = {
        "schema_version": "fixed_bank_flip_router_exact_nine_surface_v1",
        "probability_store_hash": store_hash,
        "rows": [row.to_payload() for row in result],
        "predictions_sealed_before_labels": True,
    }
    return ExactNineProbabilitySurface(tuple(result), store_hash, canonical_hash(surface_payload))


def build_prelabel_surface(
    surface: ExactNineProbabilitySurface,
    *,
    prediction_seal_hash: str,
) -> PrelabelSurface:
    by_key = {row.key: row for row in surface.rows}
    cases: dict[tuple[str, str], set[str]] = defaultdict(set)
    samples: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in surface.rows:
        key = (row.target_center, row.case_id)
        cases[key].add(row.action_id)
        if row.action_id == B_ACTION_ID:
            samples[key].append(row.sample_id)
    features: list[CaseActionFeature] = []
    for target, case_id in sorted(cases):
        sample_ids = tuple(samples[(target, case_id)])
        if not sample_ids:
            raise ProtocolError("Flip-router case lacks baseline probabilities.")
        for action in actions_for_target(target)[2:]:
            baseline = tuple(by_key[(target, case_id, sample_id, B_ACTION_ID)] for sample_id in sample_ids)
            candidate = tuple(by_key[(target, case_id, sample_id, action.action_id)] for sample_id in sample_ids)
            values = _feature_values(baseline, candidate)
            features.append(CaseActionFeature(target, case_id, action.action_id, str(action.selected_source), values))
    unhashed = {
        "schema_version": "fixed_bank_flip_router_prelabel_surface_v1",
        "probability_surface_hash": surface.surface_hash,
        "prediction_seal_hash": prediction_seal_hash,
        "features": [row.to_payload() for row in features],
        "labels_used": False,
    }
    return PrelabelSurface(tuple(features), surface.surface_hash, prediction_seal_hash, canonical_hash(unhashed))


def _feature_values(
    baseline: Sequence[AggregatedProbabilityRow],
    candidate: Sequence[AggregatedProbabilityRow],
) -> tuple[float, ...]:
    b = np.asarray([row.probability_mean for row in baseline], dtype=np.float64)
    c = np.asarray([row.probability_mean for row in candidate], dtype=np.float64)
    b_hard = b >= HARD_THRESHOLD; c_hard = c >= HARD_THRESHOLD
    flip01 = (~b_hard) & c_hard; flip10 = b_hard & (~c_hard); flip = flip01 | flip10
    size = len(b)
    if size != len(c) or size <= 0:
        raise ProtocolError("Flip-router case probability geometry drifted.")
    if np.any(flip):
        b_margin = float(np.mean(np.abs(b[flip] - HARD_THRESHOLD), dtype=np.float64))
        c_margin = float(np.mean(np.abs(c[flip] - HARD_THRESHOLD), dtype=np.float64))
        signed = float(np.mean(c[flip] - b[flip], dtype=np.float64))
        robust = []
        disagreement = []
        flip_indices = np.flatnonzero(flip)
        for index in flip_indices:
            bs = np.asarray(baseline[int(index)].seed_probabilities) >= HARD_THRESHOLD
            cs = np.asarray(candidate[int(index)].seed_probabilities) >= HARD_THRESHOLD
            robust.append(float(np.mean((bs == b_hard[index]) & (cs == c_hard[index]), dtype=np.float64)))
            rate = float(np.mean(cs, dtype=np.float64))
            disagreement.append(2.0 * rate * (1.0 - rate))
        seed_robustness = float(np.mean(robust, dtype=np.float64))
        seed_disagreement = float(np.mean(disagreement, dtype=np.float64))
    else:
        b_margin = c_margin = signed = seed_robustness = seed_disagreement = 0.0
    return (
        float(np.sum(flip01)), float(np.mean(flip01, dtype=np.float64)),
        float(np.sum(flip10)), float(np.mean(flip10, dtype=np.float64)),
        1.0 if not np.any(flip) else 0.0,
        b_margin, c_margin, signed, seed_robustness, seed_disagreement, float(size),
    )


def probability_lookup(surface: ExactNineProbabilitySurface) -> Mapping[tuple[str, str, str, str], AggregatedProbabilityRow]:
    return {row.key: row for row in surface.rows}


__all__ = (
    "aggregate_exact_nine", "build_prelabel_surface", "probability_lookup", "seed_probability_rows",
)
