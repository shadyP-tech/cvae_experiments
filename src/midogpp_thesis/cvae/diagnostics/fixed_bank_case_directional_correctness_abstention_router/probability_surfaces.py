"""Exact-nine probability products; labels are structurally unavailable here."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType

import numpy as np

from ...protocol import ProtocolError
from .constants import HARD_THRESHOLD, SEED_PAIR_COUNT, physical_action_ids
from .hashing import canonical_hash, require_sha256, require_stable_hash


def _probability(value: object) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ProtocolError("Abstention-router probability must lie in [0,1].")
    return result


def hard_prediction(value: object) -> int:
    return int(_probability(value) >= HARD_THRESHOLD)


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
            not self.case_id
            or not self.sample_id
            or self.action_id not in physical_action_ids(self.target_center)
            or isinstance(self.seed_pair_ordinal, bool)
            or self.seed_pair_ordinal not in range(SEED_PAIR_COUNT)
        ):
            raise ProtocolError("Abstention-router seed probability identity drifted.")
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
        values = tuple(_probability(value) for value in self.seed_probabilities)
        if (
            not self.case_id
            or not self.sample_id
            or self.action_id not in physical_action_ids(self.target_center)
            or len(values) != SEED_PAIR_COUNT
        ):
            raise ProtocolError("Abstention-router exact-nine row drifted.")
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
            "schema_version": "fixed_bank_cdca_exact_nine_probability_row_v1",
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
        rows = tuple(sorted(self.rows, key=lambda row: row.key))
        require_stable_hash(self.probability_store_hash, "probability_store_hash")
        if not rows or len({row.key for row in rows}) != len(rows):
            raise ProtocolError("Abstention-router probability surface is empty or duplicated.")
        grouped: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for row in rows:
            grouped[row.sample_key].add(row.action_id)
        if any(actions != set(physical_action_ids(key[0])) for key, actions in grouped.items()):
            raise ProtocolError("Abstention-router sample action coverage drifted.")
        payload = {
            "schema_version": "fixed_bank_cdca_exact_nine_probability_surface_v1",
            "probability_store_hash": self.probability_store_hash,
            "rows": [row.to_payload() for row in rows],
            "physical_B_U_eight_A1": True,
            "labels_used": False,
        }
        expected = canonical_hash(payload)
        if self.surface_hash and require_sha256(self.surface_hash, "surface_hash") != expected:
            raise ProtocolError("Abstention-router probability surface hash drifted.")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "surface_hash", expected)


class ProbabilityIndex(Mapping[tuple[str, str, str, str], ExactNineProbabilityRow]):
    def __init__(self, surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow]) -> None:
        rows = tuple(
            surface_or_rows.rows
            if isinstance(surface_or_rows, ExactNineProbabilitySurface)
            else surface_or_rows
        )
        cells = {row.key: row for row in rows}
        if not rows or len(cells) != len(rows):
            raise ProtocolError("Abstention-router probability index is empty or duplicated.")
        grouped: dict[tuple[str, str, str], list[ExactNineProbabilityRow]] = defaultdict(list)
        for row in rows:
            grouped[(row.target_center, row.case_id, row.action_id)].append(row)
        self._cells = MappingProxyType(cells)
        self._grouped = MappingProxyType(
            {key: tuple(sorted(values, key=lambda row: row.sample_id)) for key, values in grouped.items()}
        )

    def __getitem__(self, key: tuple[str, str, str, str]) -> ExactNineProbabilityRow:
        return self._cells[key]

    def __iter__(self) -> Iterator[tuple[str, str, str, str]]:
        return iter(self._cells)

    def __len__(self) -> int:
        return len(self._cells)

    def rows_for_case_action(
        self, target_center: str, case_id: str, action_id: str
    ) -> tuple[ExactNineProbabilityRow, ...]:
        return self._grouped.get((str(target_center), str(case_id), str(action_id)), ())


def aggregate_exact_nine(rows: Sequence[SeedProbabilityRow]) -> ExactNineProbabilitySurface:
    grouped: dict[tuple[str, str, str, str], list[SeedProbabilityRow]] = defaultdict(list)
    store_hashes = {row.probability_store_hash for row in rows}
    for row in rows:
        grouped[row.key[:4]].append(row)
    if not grouped or len(store_hashes) != 1:
        raise ProtocolError("Abstention-router seed rows span zero or multiple stores.")
    output = []
    for key in sorted(grouped):
        cells = tuple(sorted(grouped[key], key=lambda row: row.seed_pair_ordinal))
        if tuple(row.seed_pair_ordinal for row in cells) != tuple(range(SEED_PAIR_COUNT)):
            raise ProtocolError("Abstention-router seed coverage is not exact-nine.")
        output.append(ExactNineProbabilityRow(*key, tuple(row.probability for row in cells)))
    return ExactNineProbabilitySurface(tuple(output), next(iter(store_hashes)))


__all__ = (
    "ExactNineProbabilityRow",
    "ExactNineProbabilitySurface",
    "ProbabilityIndex",
    "SeedProbabilityRow",
    "aggregate_exact_nine",
    "hard_prediction",
)
