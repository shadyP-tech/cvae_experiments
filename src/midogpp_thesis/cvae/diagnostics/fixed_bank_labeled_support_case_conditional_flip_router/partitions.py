"""Deterministic three-role whole-case partitions for all 45 decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Sequence

from ...protocol import ProtocolError
from .constants import CENTERS, OOF_FOLD_COUNT, OOF_FOLD_SEED, OOF_PARTITION_NAMESPACE
from .hashing import canonical_hash, nonempty_text, require_sha256


@dataclass(frozen=True, order=True)
class CaseIdentityRow:
    target_center: str
    case_id: str
    sample_id: str

    def __post_init__(self) -> None:
        if str(self.target_center) not in CENTERS:
            raise ProtocolError("Three-role partition center drifted.")
        nonempty_text(self.case_id, "case_id")
        nonempty_text(self.sample_id, "sample_id")

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True, order=True)
class ThreeRoleFold:
    target_center: str
    fold_ordinal: int
    selection_case_ids: tuple[str, ...]
    calibration_case_ids: tuple[str, ...]
    evaluation_case_ids: tuple[str, ...]
    fold_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        selection = tuple(sorted(str(value) for value in self.selection_case_ids))
        calibration = tuple(sorted(str(value) for value in self.calibration_case_ids))
        evaluation = tuple(sorted(str(value) for value in self.evaluation_case_ids))
        role_sets = tuple(map(set, (selection, calibration, evaluation)))
        if (
            self.target_center not in CENTERS
            or self.fold_ordinal not in range(OOF_FOLD_COUNT)
            or isinstance(self.fold_ordinal, bool)
            or any(not values for values in role_sets)
            or role_sets[0] & role_sets[1]
            or role_sets[0] & role_sets[2]
            or role_sets[1] & role_sets[2]
        ):
            raise ProtocolError("Three-role fold violates whole-case disjointness.")
        object.__setattr__(self, "selection_case_ids", selection)
        object.__setattr__(self, "calibration_case_ids", calibration)
        object.__setattr__(self, "evaluation_case_ids", evaluation)
        object.__setattr__(self, "fold_hash", canonical_hash(self._unhashed()))

    @property
    def fold_id(self) -> str:
        return f"H{self.target_center}::fold{self.fold_ordinal}"

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_flip_router_three_role_fold_v1",
            "partition_namespace": OOF_PARTITION_NAMESPACE,
            "target_center": self.target_center,
            "fold_ordinal": self.fold_ordinal,
            "selection_case_ids": list(self.selection_case_ids),
            "calibration_case_ids": list(self.calibration_case_ids),
            "evaluation_case_ids": list(self.evaluation_case_ids),
            "whole_case_role_disjoint": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "fold_id": self.fold_id, "fold_hash": self.fold_hash}


@dataclass(frozen=True)
class ThreeRolePartition:
    identities: tuple[CaseIdentityRow, ...]
    folds: tuple[ThreeRoleFold, ...]
    partition_seed: int
    partition_hash: str

    def __post_init__(self) -> None:
        identities = tuple(sorted(self.identities))
        folds = tuple(self.folds)
        require_sha256(self.partition_hash, "partition_hash")
        expected = tuple((center, fold) for center in CENTERS for fold in range(OOF_FOLD_COUNT))
        if tuple((row.target_center, row.fold_ordinal) for row in folds) != expected:
            raise ProtocolError("Three-role partition must contain exactly 45 folds.")
        cases = {
            center: {row.case_id for row in identities if row.target_center == center}
            for center in CENTERS
        }
        for center in CENTERS:
            center_folds = tuple(row for row in folds if row.target_center == center)
            evaluated = [case for row in center_folds for case in row.evaluation_case_ids]
            if set(evaluated) != cases[center] or len(evaluated) != len(set(evaluated)):
                raise ProtocolError("Each case must be evaluated exactly once.")
            for row in center_folds:
                calibration_expected = set(center_folds[(row.fold_ordinal + 1) % OOF_FOLD_COUNT].evaluation_case_ids)
                selection_expected = cases[center] - set(row.evaluation_case_ids) - calibration_expected
                if set(row.calibration_case_ids) != calibration_expected or set(row.selection_case_ids) != selection_expected:
                    raise ProtocolError("Three-role rotation drifted.")
        unhashed = _partition_payload(identities, folds, self.partition_seed)
        if self.partition_seed != OOF_FOLD_SEED or self.partition_hash != canonical_hash(unhashed):
            raise ProtocolError("Three-role partition hash drifted.")
        object.__setattr__(self, "identities", identities)

    def to_payload(self) -> dict[str, object]:
        return {**_partition_payload(self.identities, self.folds, self.partition_seed), "partition_hash": self.partition_hash}

    def fold(self, target_center: object, fold_ordinal: int) -> ThreeRoleFold:
        matches = tuple(
            row for row in self.folds
            if row.target_center == str(target_center) and row.fold_ordinal == int(fold_ordinal)
        )
        if len(matches) != 1:
            raise ProtocolError("Three-role fold is absent or duplicated.")
        return matches[0]


def build_three_role_partition(
    identities: Sequence[CaseIdentityRow],
    *,
    partition_seed: int = OOF_FOLD_SEED,
    expected_total_case_count: int | None = 218,
) -> ThreeRolePartition:
    rows = tuple(sorted(identities))
    case_keys = {(row.target_center, row.case_id) for row in rows}
    if not rows or (expected_total_case_count is not None and len(case_keys) != expected_total_case_count):
        raise ProtocolError("Three-role partition case count drifted.")
    folds: list[ThreeRoleFold] = []
    for center in CENTERS:
        cases = sorted(case for item_center, case in case_keys if item_center == center)
        ordered = sorted(
            cases,
            key=lambda case: (
                hashlib.sha256(f"{OOF_PARTITION_NAMESPACE}::{partition_seed}::{center}::{case}".encode()).hexdigest(),
                case,
            ),
        )
        buckets = tuple(tuple(ordered[index::OOF_FOLD_COUNT]) for index in range(OOF_FOLD_COUNT))
        universe = set(cases)
        for fold in range(OOF_FOLD_COUNT):
            evaluation = buckets[fold]
            calibration = buckets[(fold + 1) % OOF_FOLD_COUNT]
            selection = tuple(sorted(universe - set(evaluation) - set(calibration)))
            folds.append(ThreeRoleFold(center, fold, selection, calibration, evaluation))
    frozen = tuple(folds)
    payload = _partition_payload(rows, frozen, partition_seed)
    return ThreeRolePartition(rows, frozen, partition_seed, canonical_hash(payload))


def _partition_payload(
    identities: Sequence[CaseIdentityRow], folds: Sequence[ThreeRoleFold], seed: int
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_flip_router_three_role_partition_v1",
        "partition_namespace": OOF_PARTITION_NAMESPACE,
        "partition_seed": seed,
        "fold_count": OOF_FOLD_COUNT,
        "identities": [row.to_payload() for row in identities],
        "folds": [row.to_payload() for row in folds],
        "evaluation_case_coverage_exactly_once": True,
        "selection_calibration_evaluation_whole_case_disjoint": True,
        "label_free_partition": True,
        "prior_stage90_partition_consumed": False,
    }


__all__ = (
    "CaseIdentityRow",
    "ThreeRoleFold",
    "ThreeRolePartition",
    "build_three_role_partition",
)
