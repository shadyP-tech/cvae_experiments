"""Deterministic five-fold whole-case partitions for the pooled-BACC v2 audit."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Sequence

from ...protocol import ProtocolError
from .core_contracts import CaseIdentityRow
from .core_hashing import canonical_hash, require_sha256
from .scientific_constants import EXPECTED_FOLD_COUNT, EXPECTED_TOTAL_CASE_COUNT, MIDOGPP_CENTERS


@dataclass(frozen=True, order=True)
class CaseFold:
    target_center: str
    fold_ordinal: int
    support_case_ids: tuple[str, ...]
    evaluation_case_ids: tuple[str, ...]
    fold_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if self.target_center not in MIDOGPP_CENTERS:
            raise ProtocolError("Case fold contains an unknown target center.")
        if (
            isinstance(self.fold_ordinal, bool)
            or not isinstance(self.fold_ordinal, int)
            or self.fold_ordinal < 0
            or self.fold_ordinal >= EXPECTED_FOLD_COUNT
        ):
            raise ProtocolError("Case fold ordinal violates the five-fold lock.")
        support = tuple(sorted(str(value) for value in self.support_case_ids))
        evaluation = tuple(sorted(str(value) for value in self.evaluation_case_ids))
        if (
            not support
            or not evaluation
            or len(support) != len(set(support))
            or len(evaluation) != len(set(evaluation))
            or set(support).intersection(evaluation)
            or any(not value for value in (*support, *evaluation))
        ):
            raise ProtocolError("Case fold violates whole-case support/evaluation isolation.")
        object.__setattr__(self, "support_case_ids", support)
        object.__setattr__(self, "evaluation_case_ids", evaluation)
        object.__setattr__(self, "fold_hash", canonical_hash(self._unhashed()))

    @property
    def fold_id(self) -> str:
        return f"H{self.target_center}::fold{self.fold_ordinal}"

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pooled_bacc_case_fold_v2",
            "target_center": self.target_center,
            "fold_ordinal": self.fold_ordinal,
            "support_case_ids": list(self.support_case_ids),
            "evaluation_case_ids": list(self.evaluation_case_ids),
            "support_evaluation_disjoint": True,
            "whole_case_partition": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "fold_id": self.fold_id, "fold_hash": self.fold_hash}


@dataclass(frozen=True)
class CaseOOFPartition:
    identities: tuple[CaseIdentityRow, ...]
    folds: tuple[CaseFold, ...]
    partition_seed: int
    partition_hash: str
    fold_count: int = EXPECTED_FOLD_COUNT

    def __post_init__(self) -> None:
        identities = tuple(sorted(tuple(self.identities)))
        folds = tuple(self.folds)
        if self.fold_count != EXPECTED_FOLD_COUNT:
            raise ProtocolError("Only the predeclared five-fold partition is allowed.")
        if isinstance(self.partition_seed, bool) or not isinstance(self.partition_seed, int):
            raise ProtocolError("partition_seed must be an integer.")
        require_sha256(self.partition_hash, "partition_hash")
        identity_keys = {
            (row.target_center, row.case_id, row.sample_id) for row in identities
        }
        if not identities or len(identity_keys) != len(identities):
            raise ProtocolError("Partition identities must be globally unique.")
        expected_fold_keys = tuple(
            (center, fold)
            for center in MIDOGPP_CENTERS
            for fold in range(EXPECTED_FOLD_COUNT)
        )
        if tuple((fold.target_center, fold.fold_ordinal) for fold in folds) != expected_fold_keys:
            raise ProtocolError("Partition must contain all 45 canonical center-fold cells.")
        cases_by_center = {
            center: {row.case_id for row in identities if row.target_center == center}
            for center in MIDOGPP_CENTERS
        }
        for center in MIDOGPP_CENTERS:
            center_folds = tuple(fold for fold in folds if fold.target_center == center)
            evaluations = [case for fold in center_folds for case in fold.evaluation_case_ids]
            if (
                not cases_by_center[center]
                or len(evaluations) != len(set(evaluations))
                or set(evaluations) != cases_by_center[center]
                or any(
                    set(fold.support_case_ids)
                    != cases_by_center[center].difference(fold.evaluation_case_ids)
                    for fold in center_folds
                )
            ):
                raise ProtocolError("Partition does not evaluate every target case exactly once.")
        if canonical_hash(self._unhashed(identities, folds)) != self.partition_hash:
            raise ProtocolError("Case-OOF partition hash drifted.")
        object.__setattr__(self, "identities", identities)
        object.__setattr__(self, "folds", folds)

    def _unhashed(
        self,
        identities: Sequence[CaseIdentityRow] | None = None,
        folds: Sequence[CaseFold] | None = None,
    ) -> dict[str, object]:
        identity_rows = self.identities if identities is None else identities
        fold_rows = self.folds if folds is None else folds
        return _partition_payload(
            identity_rows,
            fold_rows,
            partition_seed=self.partition_seed,
            fold_count=self.fold_count,
        )

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "partition_hash": self.partition_hash}

    def fold(self, target_center: str, fold_ordinal: int) -> CaseFold:
        for value in self.folds:
            if value.target_center == str(target_center) and value.fold_ordinal == int(fold_ordinal):
                return value
        raise KeyError((target_center, fold_ordinal))

    def evaluation_fold_for_case(self, target_center: str, case_id: str) -> CaseFold:
        matches = tuple(
            fold
            for fold in self.folds
            if fold.target_center == str(target_center) and str(case_id) in fold.evaluation_case_ids
        )
        if len(matches) != 1:
            raise ProtocolError("A case must belong to exactly one evaluation fold.")
        return matches[0]


def build_case_oof_partition(
    identities: Sequence[CaseIdentityRow],
    *,
    partition_seed: int,
    expected_total_case_count: int | None = EXPECTED_TOTAL_CASE_COUNT,
) -> CaseOOFPartition:
    rows = tuple(sorted(tuple(identities)))
    if not rows:
        raise ProtocolError("Cannot partition an empty case surface.")
    if isinstance(partition_seed, bool) or not isinstance(partition_seed, int):
        raise ProtocolError("partition_seed must be an integer.")
    identity_keys = {(row.target_center, row.case_id, row.sample_id) for row in rows}
    if len(identity_keys) != len(rows):
        raise ProtocolError("Probability identities must be unique before partitioning.")
    case_keys = {(row.target_center, row.case_id) for row in rows}
    if expected_total_case_count is not None and len(case_keys) != int(expected_total_case_count):
        raise ProtocolError(
            f"Expected {expected_total_case_count} whole cases, observed {len(case_keys)}."
        )
    folds: list[CaseFold] = []
    for center in MIDOGPP_CENTERS:
        cases = sorted(case for row_center, case in case_keys if row_center == center)
        if len(cases) < EXPECTED_FOLD_COUNT:
            raise ProtocolError("Each target center needs at least five whole cases.")
        ordered = sorted(
            cases,
            key=lambda case: (
                hashlib.sha256(f"{partition_seed}::{center}::{case}".encode()).hexdigest(),
                case,
            ),
        )
        buckets = tuple(
            tuple(ordered[index::EXPECTED_FOLD_COUNT])
            for index in range(EXPECTED_FOLD_COUNT)
        )
        all_cases = set(cases)
        for fold_ordinal, evaluation in enumerate(buckets):
            folds.append(
                CaseFold(
                    target_center=center,
                    fold_ordinal=fold_ordinal,
                    support_case_ids=tuple(sorted(all_cases.difference(evaluation))),
                    evaluation_case_ids=tuple(sorted(evaluation)),
                )
            )
    canonical_folds = tuple(folds)
    return CaseOOFPartition(
        identities=rows,
        folds=canonical_folds,
        partition_seed=partition_seed,
        partition_hash=canonical_hash(
            _partition_payload(
                rows,
                canonical_folds,
                partition_seed=partition_seed,
                fold_count=EXPECTED_FOLD_COUNT,
            )
        ),
    )


def _partition_payload(
    identities: Sequence[CaseIdentityRow],
    folds: Sequence[CaseFold],
    *,
    partition_seed: int,
    fold_count: int,
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_pooled_bacc_case_oof_partition_v2",
        "partition_seed": partition_seed,
        "fold_count": fold_count,
        "identities": [row.to_payload() for row in identities],
        "folds": [fold.to_payload() for fold in folds],
        "evaluation_case_coverage_exactly_once": True,
        "support_evaluation_disjoint": True,
        "target_expert_excluded": True,
        "label_free_partition": True,
    }


__all__ = ("CaseFold", "CaseOOFPartition", "build_case_oof_partition")
