"""Exact-nine aggregation and sealed B-referenced label-free flip features."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.hierarchical_multi_challenger.hashing import canonical_hash
from .actions import actions_for_target
from .constants import (
    B_ACTION_ID,
    CENTERS,
    FEATURE_NAMES,
    GENERATION_SEEDS,
    HARD_THRESHOLD,
    SEED_PAIR_COUNT,
    TRAINING_SEEDS,
    a1_action_id,
    candidate_sources,
)


EXACT_NINE_SCHEMA = (
    "fixed_bank_multi_challenger_hierarchical_flip_router_exact_nine_surface_v1"
)
PRELABEL_SCHEMA = (
    "fixed_bank_multi_challenger_hierarchical_flip_router_prelabel_surface_v1"
)


def _finite(value: object, role: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ProtocolError(f"{role} must be finite.")
    return result


def _require_stable_hash(value: object, role: str) -> str:
    result = str(value)
    if not result or any(ch.isspace() for ch in result):
        raise ProtocolError(f"{role} must be a non-empty stable hash.")
    return result


def _require_sha256(value: object, role: str) -> str:
    result = str(value)
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise ProtocolError(f"{role} must be a lowercase SHA-256 digest.")
    return result


@dataclass(frozen=True, order=True)
class SeedProbabilityRow:
    target_center: str
    case_id: str
    sample_id: str
    action_id: str
    seed_pair_ordinal: int
    probability: float
    probability_store_hash: str

    def __post_init__(self) -> None:
        if (
            self.target_center not in CENTERS
            or not self.case_id
            or not self.sample_id
            or self.seed_pair_ordinal not in range(SEED_PAIR_COUNT)
            or isinstance(self.seed_pair_ordinal, bool)
        ):
            raise ProtocolError("Multi-challenger seed probability identity drifted.")
        legal = {
            B_ACTION_ID,
            "U",
            *(a1_action_id(source) for source in candidate_sources(self.target_center)),
        }
        if self.action_id not in legal:
            raise ProtocolError("Multi-challenger probability action drifted.")
        probability = _finite(self.probability, "probability")
        if not 0.0 <= probability <= 1.0:
            raise ProtocolError("Multi-challenger probability lies outside [0, 1].")
        _require_stable_hash(self.probability_store_hash, "probability_store_hash")
        object.__setattr__(self, "probability", probability)

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True, order=True)
class AggregatedProbabilityRow:
    target_center: str
    case_id: str
    sample_id: str
    action_id: str
    probability_mean: float
    probability_sd: float
    seed_pair_count: int
    seed_probabilities: tuple[float, ...]
    row_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        probabilities = tuple(
            _finite(value, "seed_probability") for value in self.seed_probabilities
        )
        mean = _finite(self.probability_mean, "probability_mean")
        sd = _finite(self.probability_sd, "probability_sd")
        if (
            self.target_center not in CENTERS
            or self.seed_pair_count != SEED_PAIR_COUNT
            or len(probabilities) != SEED_PAIR_COUNT
            or not 0.0 <= mean <= 1.0
            or sd < 0.0
            or not self.case_id
            or not self.sample_id
        ):
            raise ProtocolError("Multi-challenger exact-nine aggregate drifted.")
        object.__setattr__(self, "probability_mean", mean)
        object.__setattr__(self, "probability_sd", sd)
        object.__setattr__(self, "seed_probabilities", probabilities)
        object.__setattr__(self, "row_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.target_center, self.case_id, self.sample_id, self.action_id

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": (
                "fixed_bank_multi_challenger_hierarchical_flip_router_"
                "aggregated_probability_row_v1"
            ),
            "target_center": self.target_center,
            "case_id": self.case_id,
            "sample_id": self.sample_id,
            "action_id": self.action_id,
            "probability_mean": self.probability_mean,
            "probability_sd": self.probability_sd,
            "seed_pair_count": self.seed_pair_count,
            "seed_probabilities": list(self.seed_probabilities),
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "row_hash": self.row_hash}


@dataclass(frozen=True)
class ExactNineProbabilitySurface:
    rows: tuple[AggregatedProbabilityRow, ...]
    probability_store_hash: str
    surface_hash: str

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        _require_stable_hash(self.probability_store_hash, "probability_store_hash")
        _require_sha256(self.surface_hash, "surface_hash")
        if len({row.key for row in rows}) != len(rows):
            raise ProtocolError("Multi-challenger probability surface has duplicates.")
        expected = canonical_hash(
            {
                "schema_version": EXACT_NINE_SCHEMA,
                "probability_store_hash": self.probability_store_hash,
                "rows": [row.to_payload() for row in rows],
                "predictions_sealed_before_labels": True,
                "physical_action_count_per_target": 10,
            }
        )
        if self.surface_hash != expected:
            raise ProtocolError("Multi-challenger probability surface hash drifted.")
        object.__setattr__(self, "rows", rows)


@dataclass(frozen=True, order=True)
class CaseActionFeature:
    target_center: str
    case_id: str
    action_id: str
    selected_source: str
    values: tuple[float, ...]
    feature_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        values = tuple(_finite(value, "flip_feature") for value in self.values)
        if (
            self.target_center not in CENTERS
            or self.selected_source not in candidate_sources(self.target_center)
            or self.action_id != a1_action_id(self.selected_source)
            or not self.case_id
            or len(values) != len(FEATURE_NAMES)
        ):
            raise ProtocolError("Multi-challenger case-action feature drifted.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "feature_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str]:
        return self.target_center, self.case_id, self.action_id

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": (
                "fixed_bank_multi_challenger_hierarchical_flip_router_"
                "case_action_feature_v1"
            ),
            "target_center": self.target_center,
            "case_id": self.case_id,
            "action_id": self.action_id,
            "selected_source": self.selected_source,
            "reference_action_id": B_ACTION_ID,
            "feature_names": list(FEATURE_NAMES),
            "values": list(self.values),
            "labels_used": False,
            "pairwise_candidate_feature_tensor_present": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "feature_hash": self.feature_hash}


@dataclass(frozen=True)
class PrelabelSurface:
    features: tuple[CaseActionFeature, ...]
    probability_surface_hash: str
    prediction_seal_hash: str
    feature_surface_hash: str

    def __post_init__(self) -> None:
        features = tuple(self.features)
        _require_sha256(self.probability_surface_hash, "probability_surface_hash")
        _require_stable_hash(self.prediction_seal_hash, "prediction_seal_hash")
        _require_sha256(self.feature_surface_hash, "feature_surface_hash")
        if len({row.key for row in features}) != len(features):
            raise ProtocolError("Multi-challenger feature keys are duplicated.")
        expected = canonical_hash(
            {
                "schema_version": PRELABEL_SCHEMA,
                "probability_surface_hash": self.probability_surface_hash,
                "prediction_seal_hash": self.prediction_seal_hash,
                "features": [row.to_payload() for row in features],
                "labels_used": False,
                "reference_action_id": B_ACTION_ID,
                "pairwise_candidate_feature_tensor_present": False,
            }
        )
        if self.feature_surface_hash != expected:
            raise ProtocolError("Multi-challenger feature surface hash drifted.")
        object.__setattr__(self, "features", features)


def seed_probability_rows(prediction: object) -> tuple[SeedProbabilityRow, ...]:
    store = getattr(prediction, "store")
    rows: list[SeedProbabilityRow] = []
    seed_pairs = tuple(
        (training, generation)
        for training in TRAINING_SEEDS
        for generation in GENERATION_SEEDS
    )
    if len(seed_pairs) != SEED_PAIR_COUNT:
        raise ProtocolError("Multi-challenger exact-nine seed topology drifted.")
    for target in CENTERS:
        row_ids = store.rows_by_center[target]
        case_ids = store.case_ids_by_center[target]
        for action in actions_for_target(target):
            for ordinal, (training, generation) in enumerate(seed_pairs):
                values = store.probabilities(
                    target, action.action_id, training, generation
                )
                for sample_id, case_id, probability in zip(
                    row_ids, case_ids, values, strict=True
                ):
                    rows.append(
                        SeedProbabilityRow(
                            target,
                            case_id,
                            sample_id,
                            action.action_id,
                            ordinal,
                            float(probability),
                            store.store_hash,
                        )
                    )
    return tuple(rows)


def aggregate_exact_nine(
    rows: Sequence[SeedProbabilityRow],
) -> ExactNineProbabilitySurface:
    grouped: dict[
        tuple[str, str, str, str], list[SeedProbabilityRow]
    ] = defaultdict(list)
    store_hashes = set()
    for row in rows:
        grouped[(row.target_center, row.case_id, row.sample_id, row.action_id)].append(row)
        store_hashes.add(row.probability_store_hash)
    if not grouped or len(store_hashes) != 1:
        raise ProtocolError("Multi-challenger seed rows span invalid prediction stores.")
    result: list[AggregatedProbabilityRow] = []
    for key in sorted(grouped):
        values_by_seed = sorted(
            grouped[key], key=lambda row: row.seed_pair_ordinal
        )
        if tuple(row.seed_pair_ordinal for row in values_by_seed) != tuple(
            range(SEED_PAIR_COUNT)
        ):
            raise ProtocolError("Multi-challenger exact-nine coverage drifted.")
        values = np.asarray(
            [row.probability for row in values_by_seed], dtype=np.float64
        )
        result.append(
            AggregatedProbabilityRow(
                *key,
                probability_mean=float(np.mean(values, dtype=np.float64)),
                probability_sd=float(np.std(values, ddof=0, dtype=np.float64)),
                seed_pair_count=SEED_PAIR_COUNT,
                seed_probabilities=tuple(float(value) for value in values),
            )
        )
    store_hash = next(iter(store_hashes))
    payload = {
        "schema_version": EXACT_NINE_SCHEMA,
        "probability_store_hash": store_hash,
        "rows": [row.to_payload() for row in result],
        "predictions_sealed_before_labels": True,
        "physical_action_count_per_target": 10,
    }
    return ExactNineProbabilitySurface(
        tuple(result), store_hash, canonical_hash(payload)
    )


def build_prelabel_surface(
    surface: ExactNineProbabilitySurface,
    *,
    prediction_seal_hash: str,
) -> PrelabelSurface:
    by_key = {row.key: row for row in surface.rows}
    actions_by_case: dict[tuple[str, str], set[str]] = defaultdict(set)
    samples: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in surface.rows:
        key = (row.target_center, row.case_id)
        actions_by_case[key].add(row.action_id)
        if row.action_id == B_ACTION_ID:
            samples[key].append(row.sample_id)
    features: list[CaseActionFeature] = []
    for target, case_id in sorted(actions_by_case):
        expected_actions = {action.action_id for action in actions_for_target(target)}
        if actions_by_case[(target, case_id)] != expected_actions:
            raise ProtocolError("Multi-challenger case action coverage drifted.")
        sample_ids = tuple(samples[(target, case_id)])
        if not sample_ids:
            raise ProtocolError("Multi-challenger case lacks baseline probabilities.")
        for action in actions_for_target(target)[2:]:
            baseline = tuple(
                by_key[(target, case_id, sample_id, B_ACTION_ID)]
                for sample_id in sample_ids
            )
            candidate = tuple(
                by_key[(target, case_id, sample_id, action.action_id)]
                for sample_id in sample_ids
            )
            features.append(
                CaseActionFeature(
                    target,
                    case_id,
                    action.action_id,
                    str(action.selected_source),
                    _feature_values(baseline, candidate),
                )
            )
    payload = {
        "schema_version": PRELABEL_SCHEMA,
        "probability_surface_hash": surface.surface_hash,
        "prediction_seal_hash": prediction_seal_hash,
        "features": [row.to_payload() for row in features],
        "labels_used": False,
        "reference_action_id": B_ACTION_ID,
        "pairwise_candidate_feature_tensor_present": False,
    }
    return PrelabelSurface(
        tuple(features),
        surface.surface_hash,
        prediction_seal_hash,
        canonical_hash(payload),
    )


def _feature_values(
    baseline: Sequence[AggregatedProbabilityRow],
    candidate: Sequence[AggregatedProbabilityRow],
) -> tuple[float, ...]:
    if tuple(row.sample_id for row in baseline) != tuple(
        row.sample_id for row in candidate
    ):
        raise ProtocolError("Multi-challenger candidate row order drifted.")
    b = np.asarray(
        [row.probability_mean for row in baseline], dtype=np.float64
    )
    c = np.asarray(
        [row.probability_mean for row in candidate], dtype=np.float64
    )
    size = len(b)
    if size != len(c) or size <= 0:
        raise ProtocolError("Multi-challenger probability geometry drifted.")
    b_hard = b >= HARD_THRESHOLD
    c_hard = c >= HARD_THRESHOLD
    flip01 = (~b_hard) & c_hard
    flip10 = b_hard & (~c_hard)
    flip = flip01 | flip10
    if np.any(flip):
        b_margin = float(
            np.mean(np.abs(b[flip] - HARD_THRESHOLD), dtype=np.float64)
        )
        c_margin = float(
            np.mean(np.abs(c[flip] - HARD_THRESHOLD), dtype=np.float64)
        )
        signed = float(np.mean(c[flip] - b[flip], dtype=np.float64))
        robust: list[float] = []
        disagreement: list[float] = []
        for index in np.flatnonzero(flip):
            bs = np.asarray(
                baseline[int(index)].seed_probabilities,
                dtype=np.float64,
            ) >= HARD_THRESHOLD
            cs = np.asarray(
                candidate[int(index)].seed_probabilities,
                dtype=np.float64,
            ) >= HARD_THRESHOLD
            robust.append(
                float(
                    np.mean(
                        (bs == b_hard[index]) & (cs == c_hard[index]),
                        dtype=np.float64,
                    )
                )
            )
            rate = float(np.mean(cs, dtype=np.float64))
            disagreement.append(2.0 * rate * (1.0 - rate))
        seed_robustness = float(np.mean(robust, dtype=np.float64))
        seed_disagreement = float(np.mean(disagreement, dtype=np.float64))
    else:
        b_margin = 0.0
        c_margin = 0.0
        signed = 0.0
        seed_robustness = 0.0
        seed_disagreement = 0.0
    return (
        float(np.sum(flip01, dtype=np.int64)),
        float(np.mean(flip01, dtype=np.float64)),
        float(np.sum(flip10, dtype=np.int64)),
        float(np.mean(flip10, dtype=np.float64)),
        1.0 if not np.any(flip) else 0.0,
        b_margin,
        c_margin,
        signed,
        seed_robustness,
        seed_disagreement,
        float(size),
    )


def probability_lookup(
    surface: ExactNineProbabilitySurface,
) -> Mapping[tuple[str, str, str, str], AggregatedProbabilityRow]:
    return {row.key: row for row in surface.rows}


__all__ = (
    "EXACT_NINE_SCHEMA",
    "PRELABEL_SCHEMA",
    "AggregatedProbabilityRow",
    "CaseActionFeature",
    "ExactNineProbabilitySurface",
    "PrelabelSurface",
    "SeedProbabilityRow",
    "aggregate_exact_nine",
    "build_prelabel_surface",
    "probability_lookup",
    "seed_probability_rows",
)
