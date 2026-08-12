"""Deterministic three-role whole-case partitions for all 45 decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Sequence

from ...protocol import ProtocolError
from ...routing.hierarchical_multi_challenger.hashing import canonical_hash
from .constants import (
    CENTERS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    OOF_FOLD_COUNT,
    OOF_FOLD_SEED,
    OOF_PARTITION_NAMESPACE,
)


FOLD_SCHEMA = (
    "fixed_bank_multi_challenger_hierarchical_flip_router_three_role_fold_v1"
)
PARTITION_SCHEMA = (
    "fixed_bank_multi_challenger_hierarchical_flip_router_three_role_partition_v1"
)


def _nonempty(value: object, role: str) -> str:
    result = str(value)
    if not result:
        raise ProtocolError(f"{role} must be non-empty.")
    return result


def _require_sha256(value: object, role: str) -> str:
    result = str(value)
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise ProtocolError(f"{role} must be a lowercase SHA-256 digest.")
    return result


@dataclass(frozen=True, order=True)
class CaseIdentityRow:
    target_center: str
    case_id: str
    sample_id: str

    def __post_init__(self) -> None:
        center = str(self.target_center)
        if center not in CENTERS:
            raise ProtocolError("Multi-challenger partition center drifted.")
        object.__setattr__(self, "target_center", center)
        object.__setattr__(self, "case_id", _nonempty(self.case_id, "case_id"))
        object.__setattr__(self, "sample_id", _nonempty(self.sample_id, "sample_id"))

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
        target = str(self.target_center)
        selection = tuple(sorted(_nonempty(value, "case_id") for value in self.selection_case_ids))
        calibration = tuple(sorted(_nonempty(value, "case_id") for value in self.calibration_case_ids))
        evaluation = tuple(sorted(_nonempty(value, "case_id") for value in self.evaluation_case_ids))
        role_sets = tuple(map(set, (selection, calibration, evaluation)))
        if (
            target not in CENTERS
            or isinstance(self.fold_ordinal, bool)
            or int(self.fold_ordinal) not in range(OOF_FOLD_COUNT)
            or any(not values for values in role_sets)
            or any(len(values) != len(role_sets[index]) for index, values in enumerate((selection, calibration, evaluation)))
            or role_sets[0] & role_sets[1]
            or role_sets[0] & role_sets[2]
            or role_sets[1] & role_sets[2]
        ):
            raise ProtocolError("Three-role fold violates whole-case disjointness.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "fold_ordinal", int(self.fold_ordinal))
        object.__setattr__(self, "selection_case_ids", selection)
        object.__setattr__(self, "calibration_case_ids", calibration)
        object.__setattr__(self, "evaluation_case_ids", evaluation)
        object.__setattr__(self, "fold_hash", canonical_hash(self._unhashed()))

    @property
    def fold_id(self) -> str:
        return f"H{self.target_center}::fold{self.fold_ordinal}"

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": FOLD_SCHEMA,
            "partition_namespace": OOF_PARTITION_NAMESPACE,
            "target_center": self.target_center,
            "fold_ordinal": self.fold_ordinal,
            "selection_case_ids": list(self.selection_case_ids),
            "calibration_case_ids": list(self.calibration_case_ids),
            "evaluation_case_ids": list(self.evaluation_case_ids),
            "whole_case_role_disjoint": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._unhashed(),
            "fold_id": self.fold_id,
            "fold_hash": self.fold_hash,
        }


@dataclass(frozen=True)
class ThreeRolePartition:
    identities: tuple[CaseIdentityRow, ...]
    folds: tuple[ThreeRoleFold, ...]
    partition_seed: int
    partition_hash: str

    def __post_init__(self) -> None:
        identities = tuple(sorted(self.identities))
        folds = tuple(self.folds)
        _require_sha256(self.partition_hash, "partition_hash")
        if len({(row.target_center, row.sample_id) for row in identities}) != len(identities):
            raise ProtocolError("Partition sample identities are duplicated.")
        expected = tuple(
            (center, fold)
            for center in CENTERS
            for fold in range(OOF_FOLD_COUNT)
        )
        if tuple((row.target_center, row.fold_ordinal) for row in folds) != expected:
            raise ProtocolError("Three-role partition must contain exactly 45 folds.")
        cases = {
            center: {
                row.case_id for row in identities if row.target_center == center
            }
            for center in CENTERS
        }
        for center in CENTERS:
            center_folds = tuple(
                row for row in folds if row.target_center == center
            )
            evaluated = [
                case_id
                for row in center_folds
                for case_id in row.evaluation_case_ids
            ]
            if set(evaluated) != cases[center] or len(evaluated) != len(set(evaluated)):
                raise ProtocolError("Each case must be evaluated exactly once.")
            for row in center_folds:
                calibration_expected = set(
                    center_folds[
                        (row.fold_ordinal + 1) % OOF_FOLD_COUNT
                    ].evaluation_case_ids
                )
                selection_expected = (
                    cases[center]
                    - set(row.evaluation_case_ids)
                    - calibration_expected
                )
                if (
                    set(row.calibration_case_ids) != calibration_expected
                    or set(row.selection_case_ids) != selection_expected
                ):
                    raise ProtocolError("Three-role rotation drifted.")
        payload = _partition_payload(identities, folds, self.partition_seed)
        if (
            isinstance(self.partition_seed, bool)
            or int(self.partition_seed) != OOF_FOLD_SEED
            or self.partition_hash != canonical_hash(payload)
        ):
            raise ProtocolError("Three-role partition hash drifted.")
        object.__setattr__(self, "identities", identities)
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "partition_seed", int(self.partition_seed))

    def to_payload(self) -> dict[str, object]:
        return {
            **_partition_payload(self.identities, self.folds, self.partition_seed),
            "partition_hash": self.partition_hash,
        }

    def fold(self, target_center: object, fold_ordinal: int) -> ThreeRoleFold:
        key = (str(target_center), int(fold_ordinal))
        matches = tuple(
            row
            for row in self.folds
            if (row.target_center, row.fold_ordinal) == key
        )
        if len(matches) != 1:
            raise ProtocolError("Three-role fold is absent or duplicated.")
        return matches[0]


def build_three_role_partition(
    identities: Sequence[CaseIdentityRow],
    *,
    partition_seed: int = OOF_FOLD_SEED,
    expected_total_case_count: int | None = 218,
    enforce_canonical_center_counts: bool = True,
) -> ThreeRolePartition:
    rows = tuple(sorted(identities))
    if len({(row.target_center, row.sample_id) for row in rows}) != len(rows):
        raise ProtocolError("Partition sample identities are duplicated.")
    case_keys = {(row.target_center, row.case_id) for row in rows}
    if not rows or (
        expected_total_case_count is not None
        and len(case_keys) != expected_total_case_count
    ):
        raise ProtocolError("Three-role partition case count drifted.")
    if enforce_canonical_center_counts and expected_total_case_count is not None:
        observed = {
            center: sum(item_center == center for item_center, _ in case_keys)
            for center in CENTERS
        }
        if observed != EXPECTED_CASE_COUNTS_BY_CENTER:
            raise ProtocolError("Three-role partition center case counts drifted.")
    if isinstance(partition_seed, bool) or int(partition_seed) != OOF_FOLD_SEED:
        raise ProtocolError("Three-role partition seed drifted.")
    folds: list[ThreeRoleFold] = []
    for center in CENTERS:
        cases = sorted(
            case_id for item_center, case_id in case_keys if item_center == center
        )
        ordered = sorted(
            cases,
            key=lambda case_id: (
                hashlib.sha256(
                    (
                        f"{OOF_PARTITION_NAMESPACE}::{partition_seed}::"
                        f"{center}::{case_id}"
                    ).encode("utf-8")
                ).hexdigest(),
                case_id,
            ),
        )
        buckets = tuple(
            tuple(ordered[index::OOF_FOLD_COUNT])
            for index in range(OOF_FOLD_COUNT)
        )
        if any(not bucket for bucket in buckets):
            raise ProtocolError("Three-role partition produced an empty fold.")
        universe = set(cases)
        for fold in range(OOF_FOLD_COUNT):
            evaluation = buckets[fold]
            calibration = buckets[(fold + 1) % OOF_FOLD_COUNT]
            selection = tuple(
                sorted(universe - set(evaluation) - set(calibration))
            )
            folds.append(
                ThreeRoleFold(
                    center,
                    fold,
                    selection,
                    calibration,
                    evaluation,
                )
            )
    frozen = tuple(folds)
    payload = _partition_payload(rows, frozen, int(partition_seed))
    return ThreeRolePartition(
        rows,
        frozen,
        int(partition_seed),
        canonical_hash(payload),
    )


def _partition_payload(
    identities: Sequence[CaseIdentityRow],
    folds: Sequence[ThreeRoleFold],
    seed: int,
) -> dict[str, object]:
    return {
        "schema_version": PARTITION_SCHEMA,
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
    "FOLD_SCHEMA",
    "PARTITION_SCHEMA",
    "CaseIdentityRow",
    "ThreeRoleFold",
    "ThreeRolePartition",
    "build_three_role_partition",
)
