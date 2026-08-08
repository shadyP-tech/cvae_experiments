"""Post-seal scoring for the exact additive-tail response surface."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...metrics import balanced_accuracy
from ...protocol import ProtocolError
from ..utility_aligned import ExactTailUtilityRow, validate_exact_tail_utility_rows
from .contracts import (
    BASE_ACTION_ID,
    EXPECTED_UTILITY_ROW_COUNT,
    PRIMARY_METRIC,
    RESPONSE_SEMANTICS,
    DevelopmentPartition,
    expected_prediction_keys,
    expected_utility_keys,
    tail_action_id,
)
from .label_access import OpenedDevelopmentLabels
from .seals import GlobalPredictionSeal


PredictionKey = tuple[str, str, str, int, int]


@dataclass(frozen=True)
class SealedPredictionSurface:
    """In-memory view whose bytes are bound by a complete global seal."""

    predictions_by_key: Mapping[PredictionKey, np.ndarray]
    seal: GlobalPredictionSeal

    def __post_init__(self) -> None:
        self.seal.verify_complete()
        expected = expected_prediction_keys()
        observed = dict(self.predictions_by_key)
        if set(observed) != set(expected) or len(observed) != len(expected):
            raise ProtocolError("Exact-tail prediction surface coverage drifted.")
        cell_by_key = {cell.key: cell for cell in self.seal.cells}
        normalized: dict[PredictionKey, np.ndarray] = {}
        for key in expected:
            values = np.asarray(observed[key])
            if values.ndim != 1 or values.dtype != np.uint8 or not np.isin(
                values, [0, 1]
            ).all():
                raise ProtocolError("Exact-tail predictions must be binary uint8 vectors.")
            if array_sha256(values) != cell_by_key[key].prediction_sha256:
                raise ProtocolError("Exact-tail prediction bytes drifted from their cell seal.")
            values.setflags(write=False)
            normalized[key] = values
        object.__setattr__(self, "predictions_by_key", MappingProxyType(normalized))


@dataclass(frozen=True)
class ScoredExactTailUtilityRow:
    """One paired base-versus-tail utility response; no sample labels persist."""

    outer_target: str
    pseudo_query: str
    candidate_source: str
    training_seed: int
    generation_seed: int
    base_bacc: float
    tail_bacc: float
    delta_bacc: float
    evaluation_row_count: int
    evaluation_case_count: int
    evaluation_row_hash: str
    support_partition_hash: str
    prediction_seal_hash: str
    base_prediction_sha256: str
    tail_prediction_sha256: str
    utility_row_hash: str
    primary_metric: str = PRIMARY_METRIC
    response_semantics: str = RESPONSE_SEMANTICS
    target_support_labels_used: bool = False
    target_evaluation_labels_used: bool = False
    seed_selection_performed: bool = False

    def __post_init__(self) -> None:
        values = (float(self.base_bacc), float(self.tail_bacc), float(self.delta_bacc))
        if not all(math.isfinite(value) for value in values):
            raise ProtocolError("Exact-tail utility metrics must be finite.")
        if not math.isclose(
            self.delta_bacc,
            self.tail_bacc - self.base_bacc,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ProtocolError("Exact-tail utility response is not tail minus base.")
        if self.evaluation_row_count <= 0 or self.evaluation_case_count <= 0:
            raise ProtocolError("Exact-tail utility row has empty evaluation coverage.")
        if (
            self.primary_metric != PRIMARY_METRIC
            or self.response_semantics != RESPONSE_SEMANTICS
            or self.target_support_labels_used is not False
            or self.target_evaluation_labels_used is not False
            or self.seed_selection_performed is not False
        ):
            raise ProtocolError("Exact-tail utility row violates the claim firewall.")
        if self.utility_row_hash != stable_hash(self._unhashed_payload()):
            raise ProtocolError("Exact-tail utility row hash drifted.")

    @property
    def replicate_id(self) -> str:
        return f"train{self.training_seed}:gen{self.generation_seed}"

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_exact_additive_tail_utility_row_v1",
            "outer_target": self.outer_target,
            "pseudo_query": self.pseudo_query,
            "candidate_source": self.candidate_source,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "replicate_id": self.replicate_id,
            "base_bacc": self.base_bacc,
            "tail_bacc": self.tail_bacc,
            "delta_bacc": self.delta_bacc,
            "evaluation_row_count": self.evaluation_row_count,
            "evaluation_case_count": self.evaluation_case_count,
            "evaluation_row_hash": self.evaluation_row_hash,
            "support_partition_hash": self.support_partition_hash,
            "prediction_seal_hash": self.prediction_seal_hash,
            "base_prediction_sha256": self.base_prediction_sha256,
            "tail_prediction_sha256": self.tail_prediction_sha256,
            "primary_metric": self.primary_metric,
            "response_semantics": self.response_semantics,
            "development_labels_used_for_scoring_only": True,
            "target_support_labels_used": self.target_support_labels_used,
            "target_evaluation_labels_used": self.target_evaluation_labels_used,
            "seed_selection_performed": self.seed_selection_performed,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "utility_row_hash": self.utility_row_hash}


def score_exact_tail_utility_surface(
    predictions: SealedPredictionSurface,
    labels: OpenedDevelopmentLabels,
    partitions: Mapping[str, DevelopmentPartition],
) -> tuple[ScoredExactTailUtilityRow, ...]:
    """Score every exact tail only after the global prediction capability opens."""

    seal = predictions.seal
    if (
        labels.prediction_seal_hash != seal.seal_hash
        or labels.manifest_sha256 != seal.development_manifest_sha256
    ):
        raise ProtocolError("Exact-tail scoring labels bind another prediction surface.")
    normalized_partitions = {
        str(center): partition for center, partition in partitions.items()
    }
    cell_by_key = {cell.key: cell for cell in seal.cells}
    rows: list[ScoredExactTailUtilityRow] = []
    for outer, query, source, training_seed, generation_seed in expected_utility_keys():
        partition = normalized_partitions.get(query)
        if (
            partition is None
            or partition.reservation_hash != seal.partition_hash_by_center[query]
        ):
            raise ProtocolError("Exact-tail scoring partition escaped the seal.")
        truth = np.asarray(labels.labels_by_center[query], dtype=np.uint8)
        base_key = (outer, query, BASE_ACTION_ID, training_seed, generation_seed)
        tail_key = (
            outer,
            query,
            tail_action_id(source),
            training_seed,
            generation_seed,
        )
        base = predictions.predictions_by_key[base_key]
        tail = predictions.predictions_by_key[tail_key]
        if base.shape != truth.shape or tail.shape != truth.shape:
            raise ProtocolError("Exact-tail prediction/label row geometry drifted.")
        base_bacc = float(balanced_accuracy(truth.tolist(), base.tolist()))
        tail_bacc = float(balanced_accuracy(truth.tolist(), tail.tolist()))
        values: dict[str, object] = {
            "outer_target": outer,
            "pseudo_query": query,
            "candidate_source": source,
            "training_seed": training_seed,
            "generation_seed": generation_seed,
            "base_bacc": base_bacc,
            "tail_bacc": tail_bacc,
            "delta_bacc": tail_bacc - base_bacc,
            "evaluation_row_count": len(truth),
            "evaluation_case_count": len(
                {row.case_id for row in partition.evaluation_rows}
            ),
            "evaluation_row_hash": seal.evaluation_row_hash_by_center[query],
            "support_partition_hash": partition.reservation_hash,
            "prediction_seal_hash": seal.seal_hash,
            "base_prediction_sha256": cell_by_key[base_key].prediction_sha256,
            "tail_prediction_sha256": cell_by_key[tail_key].prediction_sha256,
            "utility_row_hash": "",
        }
        provisional = ScoredExactTailUtilityRow.__new__(ScoredExactTailUtilityRow)
        for key, value in values.items():
            object.__setattr__(provisional, key, value)
        for key, value in {
            "primary_metric": PRIMARY_METRIC,
            "response_semantics": RESPONSE_SEMANTICS,
            "target_support_labels_used": False,
            "target_evaluation_labels_used": False,
            "seed_selection_performed": False,
        }.items():
            object.__setattr__(provisional, key, value)
        values["utility_row_hash"] = stable_hash(provisional._unhashed_payload())
        rows.append(ScoredExactTailUtilityRow(**values))  # type: ignore[arg-type]
    if len(rows) != EXPECTED_UTILITY_ROW_COUNT:
        raise ProtocolError("Exact-tail scored utility coverage drifted.")
    return tuple(rows)


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def to_core_exact_tail_utility_rows(
    rows: Sequence[ScoredExactTailUtilityRow],
) -> tuple[ExactTailUtilityRow, ...]:
    """Convert the persisted producer schema to the shared mathematical core."""

    converted = tuple(
        ExactTailUtilityRow(
            outer_target_id=row.outer_target,
            query_id=row.pseudo_query,
            candidate_source=row.candidate_source,
            training_seed=row.training_seed,
            generation_seed=row.generation_seed,
            candidate_source_count=7,
            support_partition_hash=row.support_partition_hash,
            evaluation_partition_hash=row.evaluation_row_hash,
            prediction_seal_hash=row.prediction_seal_hash,
            base_prediction_hash=row.base_prediction_sha256,
            tail_prediction_hash=row.tail_prediction_sha256,
            base_bacc=row.base_bacc,
            tail_bacc=row.tail_bacc,
            support_eval_disjoint=True,
            predictions_sealed_before_labels=True,
            source_expert_frozen=True,
            target_labels_used_for_routing=False,
        )
        for row in rows
    )
    validate_exact_tail_utility_rows(converted)
    return converted


__all__ = (
    "PredictionKey",
    "ScoredExactTailUtilityRow",
    "SealedPredictionSurface",
    "array_sha256",
    "score_exact_tail_utility_surface",
    "to_core_exact_tail_utility_rows",
)
