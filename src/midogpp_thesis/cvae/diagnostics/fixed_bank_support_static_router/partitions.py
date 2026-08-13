"""Deterministic five-fold whole-case support/evaluation partition."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Sequence

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_TOTAL_CASE_COUNT,
    OOF_FOLD_COUNT,
    PARTITION_NAMESPACE,
    PARTITION_SEED,
)
from .hashing import canonical_hash, require_sha256
from .products import CaseIdentityRow


@dataclass(frozen=True, order=True)
class CaseFold:
    target_center: str
    fold_ordinal: int
    support_case_ids: tuple[str, ...]
    evaluation_case_ids: tuple[str, ...]
    fold_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        support = tuple(sorted(str(value) for value in self.support_case_ids))
        evaluation = tuple(sorted(str(value) for value in self.evaluation_case_ids))
        if (
            self.target_center not in CENTERS
            or isinstance(self.fold_ordinal, bool)
            or self.fold_ordinal not in range(OOF_FOLD_COUNT)
            or not support
            or not evaluation
            or len(support) != len(set(support))
            or len(evaluation) != len(set(evaluation))
            or set(support) & set(evaluation)
        ):
            raise ProtocolError("Fold violates the five-fold whole-case contract.")
        object.__setattr__(self, "support_case_ids", support)
        object.__setattr__(self, "evaluation_case_ids", evaluation)
        object.__setattr__(self, "fold_hash", canonical_hash(self._unhashed()))

    @property
    def fold_id(self) -> str:
        return f"H{self.target_center}::fold{self.fold_ordinal}"

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_support_static_router_case_fold_v1",
            "partition_namespace": PARTITION_NAMESPACE,
            "target_center": self.target_center,
            "fold_ordinal": self.fold_ordinal,
            "support_case_ids": list(self.support_case_ids),
            "evaluation_case_ids": list(self.evaluation_case_ids),
            "support_is_other_four_folds": True,
            "whole_case_support_evaluation_disjoint": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "fold_id": self.fold_id, "fold_hash": self.fold_hash}


@dataclass(frozen=True)
class CaseOOFPartition:
    identities: tuple[CaseIdentityRow, ...]
    folds: tuple[CaseFold, ...]
    partition_seed: int
    partition_hash: str

    def __post_init__(self) -> None:
        identities = tuple(sorted(self.identities))
        folds = tuple(self.folds)
        require_sha256(self.partition_hash, "partition_hash")
        if self.partition_seed != PARTITION_SEED:
            raise ProtocolError("Partition seed drifted from the predeclared value.")
        if len({row.sample_key for row in identities}) != len(identities):
            raise ProtocolError("Partition sample identities are duplicated.")
        expected_keys = tuple((center, fold) for center in CENTERS for fold in range(OOF_FOLD_COUNT))
        if tuple((row.target_center, row.fold_ordinal) for row in folds) != expected_keys:
            raise ProtocolError("Partition must contain the canonical 45 routes.")
        cases_by_center = {
            center: {row.case_id for row in identities if row.target_center == center}
            for center in CENTERS
        }
        if (
            sum(len(value) for value in cases_by_center.values()) != EXPECTED_TOTAL_CASE_COUNT
            or {center: len(cases_by_center[center]) for center in CENTERS}
            != EXPECTED_CASE_COUNTS_BY_CENTER
        ):
            raise ProtocolError("Partition is not the exact 218-case MIDOG++ surface.")
        for center in CENTERS:
            center_folds = tuple(row for row in folds if row.target_center == center)
            evaluated = [case for row in center_folds for case in row.evaluation_case_ids]
            if len(evaluated) != len(set(evaluated)) or set(evaluated) != cases_by_center[center]:
                raise ProtocolError("Each case must be evaluated exactly once.")
            for row in center_folds:
                expected_support = cases_by_center[center] - set(row.evaluation_case_ids)
                if set(row.support_case_ids) != expected_support:
                    raise ProtocolError("Fold support must be exactly the other four folds.")
        if self.partition_hash != canonical_hash(_partition_payload(identities, folds, self.partition_seed)):
            raise ProtocolError("Partition hash drifted.")
        object.__setattr__(self, "identities", identities)
        object.__setattr__(self, "folds", folds)

    def fold(self, target_center: object, fold_ordinal: int) -> CaseFold:
        key = (str(target_center), int(fold_ordinal))
        for row in self.folds:
            if (row.target_center, row.fold_ordinal) == key:
                return row
        raise ProtocolError("Requested route fold is absent.")

    def evaluation_fold_for_case(self, target_center: object, case_id: object) -> CaseFold:
        matches = tuple(
            row
            for row in self.folds
            if row.target_center == str(target_center) and str(case_id) in row.evaluation_case_ids
        )
        if len(matches) != 1:
            raise ProtocolError("Every case must have exactly one evaluation fold.")
        return matches[0]

    def to_payload(self) -> dict[str, object]:
        return {
            **_partition_payload(self.identities, self.folds, self.partition_seed),
            "partition_hash": self.partition_hash,
        }


FiveFoldPartition = CaseOOFPartition


def build_five_fold_partition(
    identities: Sequence[CaseIdentityRow],
    *,
    partition_seed: int = PARTITION_SEED,
    expected_total_case_count: int | None = EXPECTED_TOTAL_CASE_COUNT,
) -> CaseOOFPartition:
    rows = tuple(sorted(identities))
    if not rows or len({row.sample_key for row in rows}) != len(rows):
        raise ProtocolError("Partition identities must be non-empty and unique.")
    if partition_seed != PARTITION_SEED:
        raise ProtocolError("Only the predeclared partition seed is allowed.")
    case_keys = {(row.target_center, row.case_id) for row in rows}
    if expected_total_case_count is not None and len(case_keys) != expected_total_case_count:
        raise ProtocolError(
            f"Expected {expected_total_case_count} whole cases, observed {len(case_keys)}."
        )
    folds: list[CaseFold] = []
    for center in CENTERS:
        cases = sorted(case for row_center, case in case_keys if row_center == center)
        if len(cases) != EXPECTED_CASE_COUNTS_BY_CENTER[center]:
            raise ProtocolError("Per-center MIDOG++ case counts drifted.")
        ordered = sorted(
            cases,
            key=lambda case: (
                hashlib.sha256(
                    f"{PARTITION_NAMESPACE}::{partition_seed}::{center}::{case}".encode()
                ).hexdigest(),
                case,
            ),
        )
        buckets = tuple(tuple(ordered[index::OOF_FOLD_COUNT]) for index in range(OOF_FOLD_COUNT))
        universe = set(cases)
        for ordinal, evaluation in enumerate(buckets):
            folds.append(
                CaseFold(
                    target_center=center,
                    fold_ordinal=ordinal,
                    support_case_ids=tuple(sorted(universe - set(evaluation))),
                    evaluation_case_ids=tuple(sorted(evaluation)),
                )
            )
    frozen = tuple(folds)
    payload = _partition_payload(rows, frozen, partition_seed)
    return CaseOOFPartition(rows, frozen, partition_seed, canonical_hash(payload))


build_case_oof_partition = build_five_fold_partition


def _partition_payload(
    identities: Sequence[CaseIdentityRow], folds: Sequence[CaseFold], seed: int
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_support_static_router_case_oof_partition_v1",
        "partition_namespace": PARTITION_NAMESPACE,
        "partition_seed": seed,
        "fold_count": OOF_FOLD_COUNT,
        "identities": [row.to_payload() for row in identities],
        "folds": [row.to_payload() for row in folds],
        "support_is_other_four_folds": True,
        "evaluation_case_coverage_exactly_once": True,
        "whole_case_support_evaluation_disjoint": True,
        "label_free_partition": True,
    }


__all__ = (
    "CaseFold",
    "CaseOOFPartition",
    "FiveFoldPartition",
    "build_case_oof_partition",
    "build_five_fold_partition",
)
