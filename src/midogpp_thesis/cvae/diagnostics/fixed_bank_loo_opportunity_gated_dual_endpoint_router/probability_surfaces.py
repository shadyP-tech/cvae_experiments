"""Exact-nine physical probability surface; labels are unavailable by design."""

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
        raise ProtocolError("OGDE probability must be finite and lie in [0,1].")
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
            raise ProtocolError("OGDE seed probability identity drifted.")
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
            raise ProtocolError("OGDE exact-nine probability identity drifted.")
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
            "schema_version": "fixed_bank_ogde_exact_nine_probability_row_v1",
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

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ExactNineProbabilityRow":
        row = cls(
            str(payload["target_center"]),
            str(payload["case_id"]),
            str(payload["sample_id"]),
            str(payload["action_id"]),
            tuple(float(value) for value in payload["seed_probabilities"]),  # type: ignore[arg-type]
        )
        if require_sha256(payload["row_hash"], "row_hash") != row.row_hash:
            raise ProtocolError("OGDE exact-nine row hash drifted after reload.")
        return row


@dataclass(frozen=True)
class ExactNineProbabilitySurface:
    rows: tuple[ExactNineProbabilityRow, ...]
    probability_store_hash: str
    surface_hash: str = ""

    def __post_init__(self) -> None:
        rows = tuple(sorted(self.rows, key=lambda row: row.key))
        require_stable_hash(self.probability_store_hash, "probability_store_hash")
        if not rows or len({row.key for row in rows}) != len(rows):
            raise ProtocolError("OGDE probability surface is empty or duplicated.")
        grouped: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for row in rows:
            grouped[row.sample_key].add(row.action_id)
        if any(actions != set(physical_action_ids(key[0])) for key, actions in grouped.items()):
            raise ProtocolError("OGDE sample action coverage is not B/U/eight-A1 closed.")
        payload = self._unhashed(rows)
        expected = canonical_hash(payload)
        if self.surface_hash and require_sha256(self.surface_hash, "surface_hash") != expected:
            raise ProtocolError("OGDE probability surface hash drifted.")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "surface_hash", expected)

    def _unhashed(self, rows: tuple[ExactNineProbabilityRow, ...] | None = None) -> dict[str, object]:
        values = self.rows if rows is None else rows
        return {
            "schema_version": "fixed_bank_ogde_exact_nine_probability_surface_v1",
            "probability_store_hash": self.probability_store_hash,
            "rows": [row.to_payload() for row in values],
            "physical_B_U_eight_A1": True,
            "labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "surface_hash": self.surface_hash}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ExactNineProbabilitySurface":
        rows = tuple(
            ExactNineProbabilityRow.from_payload(row)
            for row in payload["rows"]  # type: ignore[union-attr]
        )
        return cls(
            rows,
            str(payload["probability_store_hash"]),
            str(payload["surface_hash"]),
        )


class ProbabilityIndex(Mapping[tuple[str, str, str, str], ExactNineProbabilityRow]):
    def __init__(self, surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow]) -> None:
        # Runtime adapters and read-only regression fixtures expose the same
        # label-free row contract without constructing the persisted surface DTO.
        # Accept that structural boundary while continuing to validate every
        # indexed key below; labels never enter this object.
        row_source = getattr(surface_or_rows, "rows", surface_or_rows)
        rows = tuple(row_source)
        cells = {row.key: row for row in rows}
        if not rows or len(cells) != len(rows):
            raise ProtocolError("OGDE probability index is empty or duplicated.")
        grouped: dict[tuple[str, str, str], list[ExactNineProbabilityRow]] = defaultdict(list)
        for row in rows:
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

    def rows_for_case_action(self, target_center: object, case_id: object, action_id: object) -> tuple[ExactNineProbabilityRow, ...]:
        return self._case_actions.get((str(target_center), str(case_id), str(action_id)), ())


def aggregate_exact_nine(rows: Sequence[SeedProbabilityRow]) -> ExactNineProbabilitySurface:
    grouped: dict[tuple[str, str, str, str], list[SeedProbabilityRow]] = defaultdict(list)
    stores = {row.probability_store_hash for row in rows}
    for row in rows:
        grouped[row.key[:4]].append(row)
    if not grouped or len(stores) != 1:
        raise ProtocolError("OGDE seed rows span zero or multiple probability stores.")
    output: list[ExactNineProbabilityRow] = []
    for key in sorted(grouped):
        cells = tuple(sorted(grouped[key], key=lambda row: row.seed_pair_ordinal))
        if tuple(row.seed_pair_ordinal for row in cells) != tuple(range(SEED_PAIR_COUNT)):
            raise ProtocolError("OGDE seed rows lack exact-nine ordinal coverage.")
        output.append(ExactNineProbabilityRow(*key, tuple(row.probability for row in cells)))
    return ExactNineProbabilitySurface(tuple(output), next(iter(stores)))


aggregate_exact_nine_probabilities = aggregate_exact_nine


__all__ = (
    "ExactNineProbabilityRow",
    "ExactNineProbabilitySurface",
    "ProbabilityIndex",
    "SeedProbabilityRow",
    "aggregate_exact_nine",
    "aggregate_exact_nine_probabilities",
    "hard_prediction",
)
