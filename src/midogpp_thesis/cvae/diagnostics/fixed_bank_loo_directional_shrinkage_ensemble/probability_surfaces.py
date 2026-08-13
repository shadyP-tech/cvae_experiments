"""Exact-nine physical probability rows and deterministic indexing."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    B_ACTION_ID,
    HARD_THRESHOLD,
    SEED_PAIR_COUNT,
    physical_action_ids,
)
from .hashing import canonical_hash, require_sha256, require_stable_hash


EXACT_NINE_ROW_SCHEMA = "fixed_bank_dcse_exact_nine_probability_row_v1"
EXACT_NINE_SURFACE_SCHEMA = "fixed_bank_dcse_exact_nine_probability_surface_v1"


def _probability(value: object, role: str = "probability") -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ProtocolError(f"DCSE {role} must be finite and lie in [0,1].")
    return result


def exact_nine_mean(values: Sequence[float]) -> float:
    probabilities = np.asarray(tuple(values), dtype=np.float64)
    if probabilities.shape != (SEED_PAIR_COUNT,):
        raise ProtocolError("DCSE probability rows require the exact nine seed pairs.")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ProtocolError("DCSE exact-nine probabilities must lie in [0,1].")
    return float(np.mean(probabilities, dtype=np.float64))


def exact_nine_means(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Vectorized exact-nine means with scalar-row numerical parity."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] <= 0 or matrix.shape[1] != SEED_PAIR_COUNT:
        raise ProtocolError("DCSE exact-nine matrix must be non-empty sample-by-nine.")
    if not np.isfinite(matrix).all() or np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ProtocolError("DCSE exact-nine matrix contains illegal probabilities.")
    result = np.mean(matrix, axis=1, dtype=np.float64)
    result.setflags(write=False)
    return result


def hard_prediction(probability: object) -> int:
    """The sole DCSE threshold; an exact 0.5 is class 1."""

    return int(_probability(probability) >= HARD_THRESHOLD)


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
        if not self.case_id or not self.sample_id:
            raise ProtocolError("DCSE seed probability identity is empty.")
        if self.action_id not in physical_action_ids(self.target_center):
            raise ProtocolError("DCSE seed probability action is not target-legal.")
        if isinstance(self.seed_pair_ordinal, bool) or self.seed_pair_ordinal not in range(SEED_PAIR_COUNT):
            raise ProtocolError("DCSE seed-pair ordinal lies outside exact-nine.")
        object.__setattr__(self, "probability", _probability(self.probability))
        require_stable_hash(self.probability_store_hash, "probability_store_hash")

    @property
    def key(self) -> tuple[str, str, str, str, int]:
        return (
            self.target_center,
            self.case_id,
            self.sample_id,
            self.action_id,
            self.seed_pair_ordinal,
        )

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True, order=True)
class ExactNineProbabilityRow:
    target_center: str
    case_id: str
    sample_id: str
    action_id: str
    seed_probabilities: tuple[float, ...]
    probability_mean: float = field(init=False, compare=True)
    probability_sd: float = field(init=False, compare=True)
    row_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if not self.case_id or not self.sample_id:
            raise ProtocolError("DCSE exact-nine row identity is empty.")
        if self.action_id not in physical_action_ids(self.target_center):
            raise ProtocolError("DCSE exact-nine action is not target-legal.")
        values = tuple(_probability(value, "seed probability") for value in self.seed_probabilities)
        if len(values) != SEED_PAIR_COUNT:
            raise ProtocolError("DCSE exact-nine row does not contain nine values.")
        array = np.asarray(values, dtype=np.float64)
        object.__setattr__(self, "seed_probabilities", values)
        object.__setattr__(self, "probability_mean", float(np.mean(array, dtype=np.float64)))
        object.__setattr__(self, "probability_sd", float(np.std(array, ddof=0, dtype=np.float64)))
        object.__setattr__(self, "row_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.target_center, self.case_id, self.sample_id, self.action_id

    @property
    def sample_key(self) -> tuple[str, str, str]:
        return self.target_center, self.case_id, self.sample_id

    @property
    def hard_prediction(self) -> int:
        return hard_prediction(self.probability_mean)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": EXACT_NINE_ROW_SCHEMA,
            "target_center": self.target_center,
            "case_id": self.case_id,
            "sample_id": self.sample_id,
            "action_id": self.action_id,
            "seed_pair_count": SEED_PAIR_COUNT,
            "seed_probabilities": list(self.seed_probabilities),
            "probability_mean": self.probability_mean,
            "probability_sd": self.probability_sd,
            "mean_before_threshold": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "row_hash": self.row_hash}


@dataclass(frozen=True)
class ExactNineProbabilitySurface:
    rows: tuple[ExactNineProbabilityRow, ...]
    probability_store_hash: str
    surface_hash: str = ""

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        require_stable_hash(self.probability_store_hash, "probability_store_hash")
        if not rows or len({row.key for row in rows}) != len(rows):
            raise ProtocolError("DCSE exact-nine surface is empty or duplicated.")
        _validate_closed_sample_actions(rows)
        canonical = tuple(sorted(rows, key=lambda row: row.key))
        payload = {
            "schema_version": EXACT_NINE_SURFACE_SCHEMA,
            "probability_store_hash": self.probability_store_hash,
            "rows": [row.to_payload() for row in canonical],
            "physical_B_U_eight_A1": True,
            "predictions_sealed_before_labels": True,
        }
        expected = canonical_hash(payload)
        if self.surface_hash:
            if require_sha256(self.surface_hash, "surface_hash") != expected:
                raise ProtocolError("DCSE exact-nine surface hash drifted.")
        else:
            object.__setattr__(self, "surface_hash", expected)
        object.__setattr__(self, "rows", canonical)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": EXACT_NINE_SURFACE_SCHEMA,
            "probability_store_hash": self.probability_store_hash,
            "rows": [row.to_payload() for row in self.rows],
            "physical_B_U_eight_A1": True,
            "predictions_sealed_before_labels": True,
            "surface_hash": self.surface_hash,
        }


class ProbabilityIndex(Mapping[tuple[str, str, str, str], ExactNineProbabilityRow]):
    def __init__(self, rows: Sequence[ExactNineProbabilityRow] | ExactNineProbabilitySurface) -> None:
        source = tuple(rows.rows if isinstance(rows, ExactNineProbabilitySurface) else rows)
        cells = {row.key: row for row in source}
        if not source or len(cells) != len(source):
            raise ProtocolError("DCSE probability index is empty or duplicated.")
        grouped: dict[tuple[str, str, str], list[ExactNineProbabilityRow]] = defaultdict(list)
        for row in source:
            grouped[(row.target_center, row.case_id, row.action_id)].append(row)
        self._cells = MappingProxyType(cells)
        self._case_actions = MappingProxyType(
            {key: tuple(sorted(value, key=lambda row: row.sample_id)) for key, value in grouped.items()}
        )

    def __getitem__(self, key: tuple[str, str, str, str]) -> ExactNineProbabilityRow:
        return self._cells[key]

    def __iter__(self) -> Iterator[tuple[str, str, str, str]]:
        return iter(self._cells)

    def __len__(self) -> int:
        return len(self._cells)

    def rows_for_case_action(self, target: str, case_id: str, action_id: str) -> tuple[ExactNineProbabilityRow, ...]:
        return self._case_actions.get((str(target), str(case_id), str(action_id)), ())


def aggregate_exact_nine(rows: Sequence[SeedProbabilityRow]) -> ExactNineProbabilitySurface:
    values = tuple(rows)
    grouped: dict[tuple[str, str, str, str], list[SeedProbabilityRow]] = defaultdict(list)
    store_hashes = {row.probability_store_hash for row in values}
    for row in values:
        grouped[row.key[:4]].append(row)
    if not grouped or len(store_hashes) != 1:
        raise ProtocolError("DCSE seed rows span zero or multiple prediction stores.")
    aggregate: list[ExactNineProbabilityRow] = []
    for key in sorted(grouped):
        seed_rows = tuple(sorted(grouped[key], key=lambda row: row.seed_pair_ordinal))
        if tuple(row.seed_pair_ordinal for row in seed_rows) != tuple(range(SEED_PAIR_COUNT)):
            raise ProtocolError("DCSE seed rows lack exact-nine ordinal coverage.")
        aggregate.append(ExactNineProbabilityRow(*key, tuple(row.probability for row in seed_rows)))
    return ExactNineProbabilitySurface(tuple(aggregate), next(iter(store_hashes)))


def _validate_closed_sample_actions(rows: Sequence[ExactNineProbabilityRow]) -> None:
    actions: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        actions[row.sample_key].add(row.action_id)
    for (target, _case, _sample), observed in actions.items():
        if observed != set(physical_action_ids(target)):
            raise ProtocolError("Every DCSE sample requires B, U, and eight A1 rows.")
        if B_ACTION_ID not in observed:  # defensive clarity for the scientific contract
            raise ProtocolError("Every DCSE sample requires baseline B.")


# Compatibility names for runtime adapters and validators.
AggregatedProbabilityRow = ExactNineProbabilityRow
aggregate_exact_nine_probabilities = aggregate_exact_nine


__all__ = (
    "AggregatedProbabilityRow",
    "ExactNineProbabilityRow",
    "ExactNineProbabilitySurface",
    "ProbabilityIndex",
    "SeedProbabilityRow",
    "aggregate_exact_nine",
    "aggregate_exact_nine_probabilities",
    "exact_nine_mean",
    "exact_nine_means",
    "hard_prediction",
)
