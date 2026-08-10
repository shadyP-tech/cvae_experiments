"""Experiment-owned deterministic whole-case OOF partitions.

The algorithm is intentionally simple, but its schema and hash namespace are
owned by this diagnostic.  No prior Stage-90 partition artifact or authority is
imported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Sequence

from ...protocol import ProtocolError
from .constants import MIDOGPP_CENTERS
from .experiment_contracts import (
    EXPECTED_TOTAL_CASE_COUNT,
    OOF_FOLD_COUNT,
    OOF_PARTITION_NAMESPACE,
)
from .hashing import canonical_hash, nonempty_text, require_sha256


@dataclass(frozen=True, order=True)
class CaseIdentityRow:
    target_center: str
    case_id: str
    sample_id: str

    def __post_init__(self) -> None:
        if self.target_center not in MIDOGPP_CENTERS:
            raise ProtocolError("Partition identity uses an unknown center.")
        nonempty_text(self.case_id, "case_id")
        nonempty_text(self.sample_id, "sample_id")

    def to_payload(self) -> dict[str, object]:
        return {
            "target_center": self.target_center,
            "case_id": self.case_id,
            "sample_id": self.sample_id,
        }


@dataclass(frozen=True, order=True)
class CaseFold:
    target_center: str
    fold_ordinal: int
    support_case_ids: tuple[str, ...]
    evaluation_case_ids: tuple[str, ...]
    fold_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        support = tuple(sorted(str(value) for value in self.support_case_ids))
        evaluation = tuple(
            sorted(str(value) for value in self.evaluation_case_ids)
        )
        if (
            self.target_center not in MIDOGPP_CENTERS
            or isinstance(self.fold_ordinal, bool)
            or self.fold_ordinal not in range(OOF_FOLD_COUNT)
            or not support
            or not evaluation
            or len(set(support)) != len(support)
            or len(set(evaluation)) != len(evaluation)
            or set(support).intersection(evaluation)
            or any(not value for value in (*support, *evaluation))
        ):
            raise ProtocolError("Case fold violates the whole-case split lock.")
        object.__setattr__(self, "support_case_ids", support)
        object.__setattr__(self, "evaluation_case_ids", evaluation)
        object.__setattr__(self, "fold_hash", canonical_hash(self._unhashed()))

    @property
    def fold_id(self) -> str:
        return f"H{self.target_center}::fold{self.fold_ordinal}"

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_actionability_case_fold_v1",
            "partition_namespace": OOF_PARTITION_NAMESPACE,
            "target_center": self.target_center,
            "fold_ordinal": self.fold_ordinal,
            "support_case_ids": list(self.support_case_ids),
            "evaluation_case_ids": list(self.evaluation_case_ids),
            "support_evaluation_disjoint": True,
            "whole_case_partition": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._unhashed(),
            "fold_id": self.fold_id,
            "fold_hash": self.fold_hash,
        }


@dataclass(frozen=True)
class CaseOOFPartition:
    identities: tuple[CaseIdentityRow, ...]
    folds: tuple[CaseFold, ...]
    partition_seed: int
    partition_hash: str
    partition_namespace: str = OOF_PARTITION_NAMESPACE
    fold_count: int = OOF_FOLD_COUNT

    def __post_init__(self) -> None:
        identities = tuple(sorted(self.identities))
        folds = tuple(self.folds)
        require_sha256(self.partition_hash, "partition_hash")
        if (
            self.partition_namespace != OOF_PARTITION_NAMESPACE
            or self.fold_count != OOF_FOLD_COUNT
            or isinstance(self.partition_seed, bool)
            or not isinstance(self.partition_seed, int)
            or not identities
            or len({row.to_payload()["sample_id"] for row in identities})
            != len(identities)
        ):
            raise ProtocolError("Case partition header or identity surface drifted.")
        expected_folds = tuple(
            (center, ordinal)
            for center in MIDOGPP_CENTERS
            for ordinal in range(OOF_FOLD_COUNT)
        )
        if tuple((row.target_center, row.fold_ordinal) for row in folds) != expected_folds:
            raise ProtocolError("Case partition must contain the exact 45 folds.")
        cases_by_center = {
            center: {
                row.case_id for row in identities if row.target_center == center
            }
            for center in MIDOGPP_CENTERS
        }
        for center in MIDOGPP_CENTERS:
            center_folds = tuple(
                row for row in folds if row.target_center == center
            )
            evaluated = tuple(
                case_id
                for row in center_folds
                for case_id in row.evaluation_case_ids
            )
            if (
                not cases_by_center[center]
                or len(evaluated) != len(set(evaluated))
                or set(evaluated) != cases_by_center[center]
                or any(
                    set(row.support_case_ids)
                    != cases_by_center[center].difference(
                        row.evaluation_case_ids
                    )
                    for row in center_folds
                )
            ):
                raise ProtocolError("A center is not evaluated exactly once by case.")
        if canonical_hash(self._unhashed(identities, folds)) != self.partition_hash:
            raise ProtocolError("Case partition hash drifted.")
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
        )

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "partition_hash": self.partition_hash}

    def fold(self, target_center: str, fold_ordinal: int) -> CaseFold:
        matches = tuple(
            row
            for row in self.folds
            if row.target_center == str(target_center)
            and row.fold_ordinal == int(fold_ordinal)
        )
        if len(matches) != 1:
            raise ProtocolError("Requested case fold is absent or duplicated.")
        return matches[0]

    def evaluation_fold_for_case(
        self, target_center: str, case_id: str
    ) -> CaseFold:
        matches = tuple(
            row
            for row in self.folds
            if row.target_center == str(target_center)
            and str(case_id) in row.evaluation_case_ids
        )
        if len(matches) != 1:
            raise ProtocolError("A case must belong to one evaluation fold.")
        return matches[0]


def build_case_oof_partition(
    identities: Sequence[CaseIdentityRow],
    *,
    partition_seed: int,
    expected_total_case_count: int = EXPECTED_TOTAL_CASE_COUNT,
) -> CaseOOFPartition:
    rows = tuple(sorted(identities))
    if (
        not rows
        or isinstance(partition_seed, bool)
        or not isinstance(partition_seed, int)
        or len({(row.target_center, row.case_id, row.sample_id) for row in rows})
        != len(rows)
    ):
        raise ProtocolError("Cannot partition malformed case identities.")
    case_keys = {(row.target_center, row.case_id) for row in rows}
    if len(case_keys) != expected_total_case_count:
        raise ProtocolError(
            f"Expected {expected_total_case_count} whole cases, observed {len(case_keys)}."
        )
    folds: list[CaseFold] = []
    for center in MIDOGPP_CENTERS:
        cases = sorted(case for row_center, case in case_keys if row_center == center)
        if len(cases) < OOF_FOLD_COUNT:
            raise ProtocolError("Every center requires at least five whole cases.")
        ordered = sorted(
            cases,
            key=lambda case: (
                hashlib.sha256(
                    f"{OOF_PARTITION_NAMESPACE}::{partition_seed}::{center}::{case}".encode()
                ).hexdigest(),
                case,
            ),
        )
        buckets = tuple(
            tuple(ordered[index::OOF_FOLD_COUNT])
            for index in range(OOF_FOLD_COUNT)
        )
        all_cases = set(cases)
        for ordinal, evaluation in enumerate(buckets):
            folds.append(
                CaseFold(
                    target_center=center,
                    fold_ordinal=ordinal,
                    support_case_ids=tuple(
                        sorted(all_cases.difference(evaluation))
                    ),
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
                rows, canonical_folds, partition_seed=partition_seed
            )
        ),
    )


def _partition_payload(
    identities: Sequence[CaseIdentityRow],
    folds: Sequence[CaseFold],
    *,
    partition_seed: int,
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_actionability_case_oof_partition_v1",
        "partition_namespace": OOF_PARTITION_NAMESPACE,
        "partition_seed": partition_seed,
        "fold_count": OOF_FOLD_COUNT,
        "identities": [row.to_payload() for row in identities],
        "folds": [row.to_payload() for row in folds],
        "evaluation_case_coverage_exactly_once": True,
        "support_evaluation_disjoint": True,
        "target_expert_excluded": True,
        "label_free_partition": True,
        "prior_stage90_partition_consumed": False,
    }


__all__ = (
    "CaseFold",
    "CaseIdentityRow",
    "CaseOOFPartition",
    "build_case_oof_partition",
)
