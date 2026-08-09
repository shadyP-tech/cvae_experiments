"""Fail-closed label capabilities for LOCO, local support, and final evaluation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_file
from .core_contracts import BinaryLabelRow
from .core_hashing import canonical_hash, require_sha256
from .experiment_contracts import (
    CENTERS,
    EXPECTED_CENTER_FOLD_COUNT,
    EXPECTED_MANIFEST_SHA256,
    OOF_FOLD_COUNT,
)
from .input_contracts import LabelFreeTestFrame, TestRowIdentity
from .partitions import CaseOOFPartition


@dataclass(frozen=True)
class LabelAccessEvent:
    role: str
    target_center: str | None
    fold_ordinal: int | None
    row_count: int
    case_count: int
    row_identity_hash: str
    label_identity_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "role": self.role,
            "target_center": self.target_center,
            "fold_ordinal": self.fold_ordinal,
            "row_count": self.row_count,
            "case_count": self.case_count,
            "row_identity_hash": self.row_identity_hash,
            "label_identity_hash": self.label_identity_hash,
            "raw_labels_persisted": False,
        }


class LabelCapabilityManager:
    """Mutable orchestration guard; scientific return values remain immutable."""

    def __init__(
        self,
        manifest_path: Path,
        frame: LabelFreeTestFrame,
        partition: CaseOOFPartition,
        *,
        global_prediction_seal_hash: str,
    ) -> None:
        require_sha256(global_prediction_seal_hash, "global_prediction_seal_hash")
        if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
            raise ProtocolError("Label capability manifest hash drifted.")
        identity_keys = {(row.center, row.case_id, row.evaluation_row_id) for row in frame.rows}
        if identity_keys != {
            (row.target_center, row.case_id, row.sample_id) for row in partition.identities
        }:
            raise ProtocolError("Label capability partition differs from the sealed frame.")
        self._manifest_path = Path(manifest_path)
        self._frame = frame
        self._partition = partition
        self._prediction_seal_hash = global_prediction_seal_hash
        self._loco_opened: set[str] = set()
        self._loco_prior_seals: dict[str, str] = {}
        self._support_opened: set[tuple[str, int]] = set()
        self._decision_hashes: dict[tuple[str, int], str] = {}
        self._all_decisions_seal_hash: str | None = None
        self._permutation_decision_seal_hash: str | None = None
        self._evaluation_opened = False
        self._events: list[LabelAccessEvent] = []

    def open_loco_prior_labels(self, heldout_target: str) -> tuple[BinaryLabelRow, ...]:
        target = str(heldout_target)
        if (
            target not in CENTERS
            or target in self._loco_opened
            or self._support_opened
            or self._decision_hashes
            or self._evaluation_opened
        ):
            raise ProtocolError("LOCO prior label capability opened out of order.")
        requested = tuple(row for row in self._frame.rows if row.center != target)
        labels = self._open_rows(requested, role="loco_other_centers", target=target, fold=None)
        if any(row.target_center == target for row in labels):
            raise ProtocolError("Held-out H labels entered G_H.")
        self._loco_opened.add(target)
        return labels

    def record_loco_prior_seal(self, heldout_target: str, prior_seal_hash: str) -> None:
        target = str(heldout_target)
        require_sha256(prior_seal_hash, "prior_seal_hash")
        if target not in self._loco_opened or target in self._loco_prior_seals or self._support_opened:
            raise ProtocolError("LOCO prior seal recorded out of order.")
        self._loco_prior_seals[target] = prior_seal_hash

    def open_fold_support_labels(self, target_center: str, fold_ordinal: int) -> tuple[BinaryLabelRow, ...]:
        target, ordinal = str(target_center), int(fold_ordinal)
        key = (target, ordinal)
        if (
            set(self._loco_prior_seals) != set(CENTERS)
            or key in self._support_opened
            or key in self._decision_hashes
            or self._all_decisions_seal_hash is not None
            or self._evaluation_opened
        ):
            raise ProtocolError("Fold support label capability opened out of order.")
        fold = self._partition.fold(target, ordinal)
        requested = tuple(
            row for row in self._frame.rows_by_center[target]
            if row.case_id in set(fold.support_case_ids)
        )
        if set(row.case_id for row in requested) != set(fold.support_case_ids):
            raise ProtocolError("Fold support label request has incomplete case coverage.")
        labels = self._open_rows(requested, role="same_H_fold_support", target=target, fold=ordinal)
        self._support_opened.add(key)
        return labels

    def record_fold_decision(self, target_center: str, fold_ordinal: int, decision_hash: str) -> None:
        target, ordinal = str(target_center), int(fold_ordinal)
        key = (target, ordinal)
        require_sha256(decision_hash, "decision_hash")
        if key not in self._support_opened or key in self._decision_hashes or self._evaluation_opened:
            raise ProtocolError("Fold decision recorded without its scoped support capability.")
        self._decision_hashes[key] = decision_hash

    def record_preevaluation_seals(
        self,
        decision_seal_hash: str,
        permutation_decision_seal_hash: str,
        *,
        decision_count: int,
    ) -> None:
        require_sha256(decision_seal_hash, "decision_seal_hash")
        require_sha256(permutation_decision_seal_hash, "permutation_decision_seal_hash")
        expected = {(center, fold) for center in CENTERS for fold in range(OOF_FOLD_COUNT)}
        if (
            set(self._decision_hashes) != expected
            or int(decision_count) != EXPECTED_CENTER_FOLD_COUNT
            or self._all_decisions_seal_hash is not None
            or self._evaluation_opened
        ):
            raise ProtocolError("All-decision seal cannot open before all 45 fold decisions.")
        self._all_decisions_seal_hash = decision_seal_hash
        self._permutation_decision_seal_hash = permutation_decision_seal_hash

    def open_oof_evaluation_labels(self) -> tuple[BinaryLabelRow, ...]:
        if (
            self._all_decisions_seal_hash is None
            or self._permutation_decision_seal_hash is None
            or self._evaluation_opened
        ):
            raise ProtocolError("Evaluation labels require the durable all-decision seal.")
        labels = self._open_rows(
            self._frame.rows,
            role="oof_evaluation_after_all_decisions",
            target=None,
            fold=None,
        )
        expected_case_keys = {
            (fold.target_center, case_id)
            for fold in self._partition.folds
            for case_id in fold.evaluation_case_ids
        }
        if {(row.target_center, row.case_id) for row in labels} != expected_case_keys:
            raise ProtocolError("OOF evaluation labels lack exact-once case coverage.")
        self._evaluation_opened = True
        return labels

    def access_report(self) -> Mapping[str, object]:
        payload = {
            "schema_version": "midogpp_label_aware_label_capability_report_v1",
            "status": "PASS" if self._evaluation_opened else "INCOMPLETE",
            "global_prediction_seal_hash": self._prediction_seal_hash,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "loco_prior_centers_opened": sorted(self._loco_opened),
            "loco_prior_seals": dict(sorted(self._loco_prior_seals.items())),
            "fold_support_capability_count": len(self._support_opened),
            "fold_decision_count": len(self._decision_hashes),
            "all_decisions_seal_hash": self._all_decisions_seal_hash,
            "permutation_decision_seal_hash": self._permutation_decision_seal_hash,
            "evaluation_labels_opened": self._evaluation_opened,
            "events": [event.to_payload() for event in self._events],
            "raw_labels_persisted": False,
            "whole_label_column_materialized": False,
            "source_expert_updated": False,
            "shared_model_updated": False,
            "evaluation_labels_used_for_decisions": False,
        }
        return MappingProxyType({**payload, "report_hash": canonical_hash(payload)})

    def _open_rows(
        self,
        rows: Sequence[TestRowIdentity],
        *,
        role: str,
        target: str | None,
        fold: int | None,
    ) -> tuple[BinaryLabelRow, ...]:
        requested = {row.manifest_row_index: row for row in rows}
        labels: list[BinaryLabelRow] = []
        try:
            handle = self._manifest_path.open(newline="", encoding="utf-8")
        except OSError as exc:
            raise ProtocolError("Cannot open scoped label manifest.") from exc
        with handle:
            reader = csv.DictReader(handle)
            required = {"sample_id", "case_id", "center", "split", "label"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ProtocolError("Scoped label manifest fields drifted.")
            for index, raw in enumerate(reader):
                wanted = requested.get(index)
                if wanted is None:
                    continue
                if (
                    str(raw["case_id"]) != wanted.case_id
                    or str(raw["center"]) != wanted.center
                    or str(raw["split"]) != wanted.split
                ):
                    raise ProtocolError("Scoped label manifest identity drifted.")
                labels.append(
                    BinaryLabelRow(
                        target_center=wanted.center,
                        case_id=wanted.case_id,
                        sample_id=wanted.evaluation_row_id,
                        label=_binary(raw["label"]),
                    )
                )
        if len(labels) != len(requested) or {row.sample_id for row in labels} != {row.evaluation_row_id for row in rows}:
            raise ProtocolError("Scoped label coverage drifted.")
        ordered = tuple(sorted(labels))
        self._events.append(
            LabelAccessEvent(
                role=role,
                target_center=target,
                fold_ordinal=fold,
                row_count=len(ordered),
                case_count=len({row.case_key for row in ordered}),
                row_identity_hash=canonical_hash(
                    {"sample_ids": [row.sample_id for row in ordered]}
                ),
                label_identity_hash=canonical_hash(
                    {
                        "labels": [
                            [row.target_center, row.case_id, row.sample_id, row.label]
                            for row in ordered
                        ]
                    }
                ),
            )
        )
        return ordered


def _binary(value: object) -> int:
    try:
        number = float(str(value))
    except ValueError as exc:
        raise ProtocolError("Scoped label is not numeric.") from exc
    if number not in (0.0, 1.0):
        raise ProtocolError("Scoped label lies outside {0,1}.")
    return int(number)


__all__ = ("LabelAccessEvent", "LabelCapabilityManager")
