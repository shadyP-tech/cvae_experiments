"""Deterministic whole-case five-fold partitions for the residual stacker."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Sequence

from ...protocol import ProtocolError
from .core_hashing import canonical_hash
from .experiment_contracts import CENTERS, EXPECTED_TOTAL_CASE_COUNT, OOF_FOLD_COUNT
from .input_contracts import TestRowIdentity


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
            or type(self.fold_ordinal) is not int
            or not 0 <= self.fold_ordinal < OOF_FOLD_COUNT
            or not support
            or not evaluation
            or len(set(support)) != len(support)
            or len(set(evaluation)) != len(evaluation)
            or set(support).intersection(evaluation)
        ):
            raise ProtocolError("Residual-stacker fold violates whole-case isolation.")
        object.__setattr__(self, "support_case_ids", support)
        object.__setattr__(self, "evaluation_case_ids", evaluation)
        object.__setattr__(self, "fold_hash", canonical_hash(self._unhashed()))

    @property
    def fold_id(self) -> str:
        return f"H{self.target_center}::fold{self.fold_ordinal}"

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_hierarchical_residual_stacker_case_fold_v1",
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
    identities: tuple[TestRowIdentity, ...]
    folds: tuple[CaseFold, ...]
    partition_seed: int
    partition_hash: str
    fold_count: int = OOF_FOLD_COUNT

    def __post_init__(self) -> None:
        identities = tuple(sorted(self.identities))
        folds = tuple(self.folds)
        if self.fold_count != OOF_FOLD_COUNT or type(self.partition_seed) is not int:
            raise ProtocolError("Residual-stacker partition lock drifted.")
        keys = {(r.center, r.case_id, r.evaluation_row_id) for r in identities}
        expected_folds = tuple(
            (center, fold) for center in CENTERS for fold in range(OOF_FOLD_COUNT)
        )
        if not identities or len(keys) != len(identities):
            raise ProtocolError("Residual-stacker partition identities are not unique.")
        if tuple((f.target_center, f.fold_ordinal) for f in folds) != expected_folds:
            raise ProtocolError("Residual-stacker partition must contain all 45 folds.")
        cases_by_center = {
            center: {r.case_id for r in identities if r.center == center}
            for center in CENTERS
        }
        for center in CENTERS:
            center_folds = tuple(f for f in folds if f.target_center == center)
            evaluated = [case for fold in center_folds for case in fold.evaluation_case_ids]
            if (
                set(evaluated) != cases_by_center[center]
                or len(evaluated) != len(set(evaluated))
                or any(
                    set(fold.support_case_ids)
                    != cases_by_center[center].difference(fold.evaluation_case_ids)
                    for fold in center_folds
                )
            ):
                raise ProtocolError("Every residual-stacker case must be evaluated once.")
        if canonical_hash(self._unhashed(identities, folds)) != self.partition_hash:
            raise ProtocolError("Residual-stacker partition hash drifted.")
        object.__setattr__(self, "identities", identities)
        object.__setattr__(self, "folds", folds)

    def _unhashed(
        self,
        identities: Sequence[TestRowIdentity] | None = None,
        folds: Sequence[CaseFold] | None = None,
    ) -> dict[str, object]:
        return _partition_payload(
            self.identities if identities is None else identities,
            self.folds if folds is None else folds,
            self.partition_seed,
        )

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "partition_hash": self.partition_hash}

    def fold(self, target_center: str, fold_ordinal: int) -> CaseFold:
        for fold in self.folds:
            if fold.target_center == str(target_center) and fold.fold_ordinal == int(fold_ordinal):
                return fold
        raise KeyError((target_center, fold_ordinal))


def build_case_oof_partition(
    identities: Sequence[TestRowIdentity],
    *,
    partition_seed: int,
    expected_total_case_count: int | None = EXPECTED_TOTAL_CASE_COUNT,
) -> CaseOOFPartition:
    rows = tuple(sorted(identities))
    if not rows or type(partition_seed) is not int:
        raise ProtocolError("Cannot build an empty or unseeded residual-stacker partition.")
    keys = {(row.center, row.case_id, row.evaluation_row_id) for row in rows}
    case_keys = {(row.center, row.case_id) for row in rows}
    if len(keys) != len(rows):
        raise ProtocolError("Residual-stacker probability identities are duplicated.")
    if expected_total_case_count is not None and len(case_keys) != expected_total_case_count:
        raise ProtocolError(
            f"Expected {expected_total_case_count} whole cases, observed {len(case_keys)}."
        )
    folds: list[CaseFold] = []
    for center in CENTERS:
        cases = sorted(case for row_center, case in case_keys if row_center == center)
        if len(cases) < OOF_FOLD_COUNT:
            raise ProtocolError("Each target center needs at least five whole cases.")
        ordered = sorted(
            cases,
            key=lambda case: (
                hashlib.sha256(f"{partition_seed}::{center}::{case}".encode()).hexdigest(),
                case,
            ),
        )
        all_cases = set(cases)
        for ordinal in range(OOF_FOLD_COUNT):
            evaluation = tuple(ordered[ordinal::OOF_FOLD_COUNT])
            folds.append(
                CaseFold(
                    center,
                    ordinal,
                    tuple(sorted(all_cases.difference(evaluation))),
                    evaluation,
                )
            )
    canonical = tuple(folds)
    return CaseOOFPartition(
        rows,
        canonical,
        partition_seed,
        canonical_hash(_partition_payload(rows, canonical, partition_seed)),
    )


def _partition_payload(
    identities: Sequence[TestRowIdentity], folds: Sequence[CaseFold], partition_seed: int
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_hierarchical_residual_stacker_case_oof_partition_v1",
        "partition_seed": partition_seed,
        "fold_count": OOF_FOLD_COUNT,
        "identities": [row.to_payload() for row in identities],
        "folds": [fold.to_payload() for fold in folds],
        "evaluation_case_coverage_exactly_once": True,
        "support_evaluation_disjoint": True,
        "target_expert_excluded": True,
        "label_free_partition": True,
    }


__all__ = ("CaseFold", "CaseOOFPartition", "build_case_oof_partition")
