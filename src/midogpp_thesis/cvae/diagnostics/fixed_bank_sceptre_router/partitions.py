"""Deterministic whole-case selection/calibration/evaluation rotation."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Sequence

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError

from .hashing import canonical_hash, require_sha256


FOLD_COUNT = 5
PARTITION_SEED = 27082026
PARTITION_NAMESPACE = "midogpp_sceptre_whole_case_three_role_v1"


@dataclass(frozen=True, order=True, slots=True)
class CaseIdentity:
    target_center: str
    case_id: str
    sample_id: str

    def __post_init__(self) -> None:
        if self.target_center not in CENTERS or not self.case_id or not self.sample_id:
            raise ProtocolError("SCEPTRE case identity is invalid.")


@dataclass(frozen=True, order=True, slots=True)
class ThreeRoleFold:
    target_center: str
    fold_ordinal: int
    selection_case_ids: tuple[str, ...]
    calibration_case_ids: tuple[str, ...]
    evaluation_case_ids: tuple[str, ...]
    fold_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        selection = tuple(sorted(map(str, self.selection_case_ids)))
        calibration = tuple(sorted(map(str, self.calibration_case_ids)))
        evaluation = tuple(sorted(map(str, self.evaluation_case_ids)))
        role_sets = tuple(map(set, (selection, calibration, evaluation)))
        if (
            self.target_center not in CENTERS
            or isinstance(self.fold_ordinal, bool)
            or self.fold_ordinal not in range(FOLD_COUNT)
            or any(not values for values in role_sets)
            or role_sets[0] & role_sets[1]
            or role_sets[0] & role_sets[2]
            or role_sets[1] & role_sets[2]
        ):
            raise ProtocolError("SCEPTRE fold violates whole-case role disjointness.")
        object.__setattr__(self, "selection_case_ids", selection)
        object.__setattr__(self, "calibration_case_ids", calibration)
        object.__setattr__(self, "evaluation_case_ids", evaluation)
        object.__setattr__(self, "fold_hash", canonical_hash(self._payload()))

    @property
    def fold_id(self) -> str:
        return f"H{self.target_center}::fold{self.fold_ordinal}"

    def case_set_hash(self, role: str) -> str:
        role_name = str(role).upper()
        values = {
            "SELECTION": self.selection_case_ids,
            "CALIBRATION": self.calibration_case_ids,
            "EVALUATION": self.evaluation_case_ids,
        }.get(role_name)
        if values is None:
            raise ProtocolError("SCEPTRE fold case-set role is unknown.")
        return canonical_hash(
            {
                "schema_version": "sceptre_fold_case_set_v1",
                "fold_hash": self.fold_hash,
                "role": role_name,
                "case_ids": list(values),
                "whole_cases": True,
            }
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "sceptre_three_role_fold_v1",
            "partition_namespace": PARTITION_NAMESPACE,
            "target_center": self.target_center,
            "fold_ordinal": self.fold_ordinal,
            "selection_case_ids": list(self.selection_case_ids),
            "calibration_case_ids": list(self.calibration_case_ids),
            "evaluation_case_ids": list(self.evaluation_case_ids),
            "whole_case_role_disjoint": True,
        }


@dataclass(frozen=True, slots=True)
class ThreeRolePartition:
    identities: tuple[CaseIdentity, ...]
    folds: tuple[ThreeRoleFold, ...]
    partition_seed: int
    partition_hash: str

    def __post_init__(self) -> None:
        identities = tuple(sorted(self.identities))
        folds = tuple(self.folds)
        expected_keys = tuple(
            (center, fold) for center in CENTERS for fold in range(FOLD_COUNT)
        )
        if tuple((fold.target_center, fold.fold_ordinal) for fold in folds) != expected_keys:
            raise ProtocolError("SCEPTRE partition must contain exactly 45 folds.")
        cases = {
            center: {row.case_id for row in identities if row.target_center == center}
            for center in CENTERS
        }
        for center in CENTERS:
            center_folds = tuple(fold for fold in folds if fold.target_center == center)
            evaluated = [case for fold in center_folds for case in fold.evaluation_case_ids]
            if set(evaluated) != cases[center] or len(evaluated) != len(set(evaluated)):
                raise ProtocolError("SCEPTRE cases are not evaluated exactly once.")
            for fold in center_folds:
                calibration = set(
                    center_folds[(fold.fold_ordinal + 1) % FOLD_COUNT].evaluation_case_ids
                )
                selection = cases[center] - set(fold.evaluation_case_ids) - calibration
                if (
                    set(fold.calibration_case_ids) != calibration
                    or set(fold.selection_case_ids) != selection
                ):
                    raise ProtocolError("SCEPTRE three-role rotation drifted.")
        expected_hash = canonical_hash(_partition_payload(identities, folds, self.partition_seed))
        if self.partition_seed != PARTITION_SEED or require_sha256(
            self.partition_hash, "partition"
        ) != expected_hash:
            raise ProtocolError("SCEPTRE partition hash drifted.")
        object.__setattr__(self, "identities", identities)

    def fold(self, target_center: str, fold_ordinal: int) -> ThreeRoleFold:
        matches = tuple(
            fold
            for fold in self.folds
            if fold.target_center == str(target_center)
            and fold.fold_ordinal == int(fold_ordinal)
        )
        if len(matches) != 1:
            raise ProtocolError("SCEPTRE fold is absent or duplicated.")
        return matches[0]


def build_three_role_partition(
    identities: Sequence[CaseIdentity],
    *,
    expected_total_case_count: int | None = 218,
) -> ThreeRolePartition:
    rows = tuple(sorted(identities))
    if len(rows) != len(set(rows)):
        raise ProtocolError("SCEPTRE partition identities are duplicated.")
    sample_ids = tuple(row.sample_id for row in rows)
    if len(sample_ids) != len(set(sample_ids)):
        raise ProtocolError("SCEPTRE sample identity occurs in multiple case rows.")
    case_keys = {(row.target_center, row.case_id) for row in rows}
    if not rows or (
        expected_total_case_count is not None
        and len(case_keys) != expected_total_case_count
    ):
        raise ProtocolError("SCEPTRE partition case count drifted.")
    folds: list[ThreeRoleFold] = []
    for center in CENTERS:
        cases = sorted(case for row_center, case in case_keys if row_center == center)
        ordered = sorted(
            cases,
            key=lambda case: (
                hashlib.sha256(
                    f"{PARTITION_NAMESPACE}::{PARTITION_SEED}::{center}::{case}".encode()
                ).hexdigest(),
                case,
            ),
        )
        buckets = tuple(tuple(ordered[index::FOLD_COUNT]) for index in range(FOLD_COUNT))
        if any(not bucket for bucket in buckets):
            raise ProtocolError("SCEPTRE center lacks five nonempty case folds.")
        universe = set(cases)
        for fold in range(FOLD_COUNT):
            evaluation = buckets[fold]
            calibration = buckets[(fold + 1) % FOLD_COUNT]
            selection = tuple(sorted(universe - set(evaluation) - set(calibration)))
            folds.append(
                ThreeRoleFold(center, fold, selection, calibration, evaluation)
            )
    frozen = tuple(folds)
    payload = _partition_payload(rows, frozen, PARTITION_SEED)
    return ThreeRolePartition(rows, frozen, PARTITION_SEED, canonical_hash(payload))


def _partition_payload(
    identities: Sequence[CaseIdentity],
    folds: Sequence[ThreeRoleFold],
    seed: int,
) -> dict[str, object]:
    return {
        "schema_version": "sceptre_three_role_partition_v1",
        "partition_namespace": PARTITION_NAMESPACE,
        "partition_seed": seed,
        "fold_count": FOLD_COUNT,
        "identities": [
            {
                "target_center": row.target_center,
                "case_id": row.case_id,
                "sample_id": row.sample_id,
            }
            for row in identities
        ],
        "folds": [
            {**fold._payload(), "fold_id": fold.fold_id, "fold_hash": fold.fold_hash}
            for fold in folds
        ],
        "evaluation_case_coverage_exactly_once": True,
        "selection_calibration_evaluation_whole_case_disjoint": True,
        "label_free_partition": True,
    }


__all__ = (
    "CaseIdentity",
    "FOLD_COUNT",
    "PARTITION_NAMESPACE",
    "PARTITION_SEED",
    "ThreeRoleFold",
    "ThreeRolePartition",
    "build_three_role_partition",
)
