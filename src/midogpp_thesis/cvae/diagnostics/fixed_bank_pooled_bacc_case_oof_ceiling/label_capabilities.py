"""Fail-closed LOCO/support/evaluation label capabilities for pooled-BACC v2."""

from __future__ import annotations

import csv
from dataclasses import dataclass
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
    EXPECTED_MIXED_CLASS_CASE_COUNT,
    EXPECTED_NEGATIVE_ONLY_CASE_COUNT,
    EXPECTED_NULL_ACTION_COUNT,
    EXPECTED_POSITIVE_ONLY_CASE_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    OOF_FOLD_COUNT,
    PERMUTATION_COUNT,
)
from .input_contracts import LabelFreeTestFrame, TestRowIdentity
from .case_partitions import CaseOOFPartition


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
    """The only manifest reader; enforces the complete label-access state machine."""

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
            raise ProtocolError("Pooled-BACC label-capability manifest hash drifted.")
        frame_keys = {(row.center, row.case_id, row.evaluation_row_id) for row in frame.rows}
        partition_keys = {
            (row.target_center, row.case_id, row.sample_id)
            for row in partition.identities
        }
        if frame_keys != partition_keys:
            raise ProtocolError("Pooled-BACC partition differs from the sealed frame.")
        self._manifest_path = Path(manifest_path)
        self._frame = frame
        self._partition = partition
        self._prediction_seal_hash = global_prediction_seal_hash
        self._loco_opened: set[str] = set()
        self._loco_prior_seals: dict[str, str] = {}
        self._support_opened: set[tuple[str, int]] = set()
        self._decision_hashes: dict[tuple[str, int], str] = {}
        self._all_decisions_seal_hash: str | None = None
        self._null_decision_seal_hash: str | None = None
        self._null_action_count = 0
        self._evaluation_opened = False
        self._topology_report: Mapping[str, object] | None = None
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
            raise ProtocolError("Pooled-BACC LOCO label capability opened out of order.")
        requested = tuple(row for row in self._frame.rows if row.center != target)
        labels = self._open_rows(
            requested, role="loco_other_centers", target=target, fold=None
        )
        if any(row.target_center == target for row in labels):
            raise ProtocolError("Held-out H labels entered pooled G_H.")
        for donor in CENTERS:
            if donor != target:
                _require_both_classes(
                    tuple(row for row in labels if row.target_center == donor),
                    f"LOCO donor {donor}",
                )
        self._loco_opened.add(target)
        return labels

    def record_loco_prior_seal(self, heldout_target: str, prior_seal_hash: str) -> None:
        target = str(heldout_target)
        require_sha256(prior_seal_hash, "prior_seal_hash")
        if (
            target not in self._loco_opened
            or target in self._loco_prior_seals
            or self._support_opened
        ):
            raise ProtocolError("Pooled-BACC LOCO prior seal recorded out of order.")
        self._loco_prior_seals[target] = prior_seal_hash

    def open_fold_support_labels(
        self, target_center: str, fold_ordinal: int
    ) -> tuple[BinaryLabelRow, ...]:
        target, ordinal = str(target_center), int(fold_ordinal)
        key = (target, ordinal)
        if (
            set(self._loco_prior_seals) != set(CENTERS)
            or key in self._support_opened
            or key in self._decision_hashes
            or self._all_decisions_seal_hash is not None
            or self._evaluation_opened
        ):
            raise ProtocolError("Pooled-BACC support capability opened out of order.")
        fold = self._partition.fold(target, ordinal)
        support_cases = set(fold.support_case_ids)
        requested = tuple(
            row
            for row in self._frame.rows_by_center[target]
            if row.case_id in support_cases
        )
        if {row.case_id for row in requested} != support_cases:
            raise ProtocolError("Pooled-BACC support request has incomplete case coverage.")
        labels = self._open_rows(
            requested, role="same_H_fold_support", target=target, fold=ordinal
        )
        _require_both_classes(labels, f"support H{target} fold {ordinal}")
        self._support_opened.add(key)
        return labels

    def record_fold_decision(
        self, target_center: str, fold_ordinal: int, decision_hash: str
    ) -> None:
        key = (str(target_center), int(fold_ordinal))
        require_sha256(decision_hash, "decision_hash")
        if (
            key not in self._support_opened
            or key in self._decision_hashes
            or self._evaluation_opened
        ):
            raise ProtocolError("Pooled-BACC decision lacks its scoped support capability.")
        self._decision_hashes[key] = decision_hash

    def record_preevaluation_seals(
        self,
        decision_seal_hash: str,
        null_decision_seal_hash: str,
        *,
        decision_count: int,
        permutation_count: int,
        null_action_count: int,
    ) -> None:
        require_sha256(decision_seal_hash, "decision_seal_hash")
        require_sha256(null_decision_seal_hash, "null_decision_seal_hash")
        expected = {
            (center, fold)
            for center in CENTERS
            for fold in range(OOF_FOLD_COUNT)
        }
        if (
            set(self._decision_hashes) != expected
            or int(decision_count) != EXPECTED_CENTER_FOLD_COUNT
            or int(permutation_count) != PERMUTATION_COUNT
            or int(null_action_count) != EXPECTED_NULL_ACTION_COUNT
            or self._all_decisions_seal_hash is not None
            or self._evaluation_opened
        ):
            raise ProtocolError(
                "Evaluation cannot open before all 45 observed and 450000 null actions are sealed."
            )
        self._all_decisions_seal_hash = decision_seal_hash
        self._null_decision_seal_hash = null_decision_seal_hash
        self._null_action_count = int(null_action_count)

    def open_oof_evaluation_labels(self) -> tuple[BinaryLabelRow, ...]:
        if (
            self._all_decisions_seal_hash is None
            or self._null_decision_seal_hash is None
            or self._null_action_count != EXPECTED_NULL_ACTION_COUNT
            or self._evaluation_opened
        ):
            raise ProtocolError("Evaluation labels require both durable observed/null seals.")
        labels = self._open_rows(
            self._frame.rows,
            role="oof_evaluation_after_all_observed_and_null_actions",
            target=None,
            fold=None,
        )
        expected_case_keys = {
            (fold.target_center, case_id)
            for fold in self._partition.folds
            for case_id in fold.evaluation_case_ids
        }
        if {(row.target_center, row.case_id) for row in labels} != expected_case_keys:
            raise ProtocolError("Pooled-BACC OOF evaluation lacks exact-once case coverage.")
        topology = audit_manifest_case_class_topology(labels, partition=self._partition)
        if topology["status"] != "PASS":
            raise ProtocolError("Pooled-BACC manifest class topology drifted.")
        self._topology_report = topology
        self._evaluation_opened = True
        return labels

    def access_report(self) -> Mapping[str, object]:
        payload = {
            "schema_version": "midogpp_pooled_bacc_label_capability_report_v2",
            "status": "PASS" if self._evaluation_opened else "INCOMPLETE",
            "global_prediction_seal_hash": self._prediction_seal_hash,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "loco_prior_centers_opened": sorted(self._loco_opened),
            "loco_and_pairwise_prior_seals": dict(sorted(self._loco_prior_seals.items())),
            "fold_support_capability_count": len(self._support_opened),
            "fold_decision_count": len(self._decision_hashes),
            "all_decisions_seal_hash": self._all_decisions_seal_hash,
            "null_decision_seal_hash": self._null_decision_seal_hash,
            "sealed_null_action_count": self._null_action_count,
            "evaluation_labels_opened": self._evaluation_opened,
            "manifest_case_class_topology": (
                dict(self._topology_report) if self._topology_report is not None else None
            ),
            "events": [event.to_payload() for event in self._events],
            "raw_labels_persisted": False,
            "per_case_bacc_persisted": False,
            "whole_label_column_materialized_only_after_preevaluation_seals": (
                self._evaluation_opened
            ),
            "whole_label_column_persisted": False,
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
            raise ProtocolError("Cannot open scoped pooled-BACC label manifest.") from exc
        with handle:
            reader = csv.DictReader(handle)
            required = {"sample_id", "case_id", "center", "split", "label"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ProtocolError("Scoped pooled-BACC manifest fields drifted.")
            for index, raw in enumerate(reader):
                wanted = requested.get(index)
                if wanted is None:
                    continue
                if (
                    str(raw["sample_id"]) != wanted.evaluation_row_id
                    or str(raw["case_id"]) != wanted.case_id
                    or str(raw["center"]) != wanted.center
                    or str(raw["split"]) != wanted.split
                ):
                    raise ProtocolError("Scoped pooled-BACC manifest identity drifted.")
                labels.append(
                    BinaryLabelRow(
                        target_center=wanted.center,
                        case_id=wanted.case_id,
                        sample_id=wanted.evaluation_row_id,
                        label=_binary(raw["label"]),
                    )
                )
        if len(labels) != len(requested) or {row.sample_id for row in labels} != {
            row.evaluation_row_id for row in rows
        }:
            raise ProtocolError("Scoped pooled-BACC label coverage drifted.")
        ordered = tuple(sorted(labels))
        self._events.append(
            LabelAccessEvent(
                role=role,
                target_center=target,
                fold_ordinal=fold,
                row_count=len(ordered),
                case_count=len({(row.target_center, row.case_id) for row in ordered}),
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


def audit_manifest_case_class_topology(
    labels: Sequence[BinaryLabelRow], *, partition: CaseOOFPartition
) -> Mapping[str, object]:
    """Validate 213 mixed + 4 negative-only + 1 positive-only whole cases."""

    by_case: dict[tuple[str, str], set[int]] = {}
    for row in labels:
        by_case.setdefault((row.target_center, row.case_id), set()).add(int(row.label))
    counts = {
        "mixed": sum(value == {0, 1} for value in by_case.values()),
        "negative_only": sum(value == {0} for value in by_case.values()),
        "positive_only": sum(value == {1} for value in by_case.values()),
    }
    if (
        len(by_case) != EXPECTED_TOTAL_CASE_COUNT
        or counts["mixed"] != EXPECTED_MIXED_CLASS_CASE_COUNT
        or counts["negative_only"] != EXPECTED_NEGATIVE_ONLY_CASE_COUNT
        or counts["positive_only"] != EXPECTED_POSITIVE_ONLY_CASE_COUNT
    ):
        raise ProtocolError("Manifest whole-case class topology drifted from 213+4+1.")
    for center in CENTERS:
        _require_both_classes(
            tuple(row for row in labels if row.target_center == center),
            f"full center {center}",
        )
    for fold in partition.folds:
        evaluation_cases = set(fold.evaluation_case_ids)
        support_cases = set(fold.support_case_ids)
        center_rows = tuple(row for row in labels if row.target_center == fold.target_center)
        _require_both_classes(
            tuple(row for row in center_rows if row.case_id in evaluation_cases),
            f"evaluation H{fold.target_center} fold {fold.fold_ordinal}",
        )
        _require_both_classes(
            tuple(row for row in center_rows if row.case_id in support_cases),
            f"support H{fold.target_center} fold {fold.fold_ordinal}",
        )
    return MappingProxyType(
        {
            "status": "PASS",
            "total_case_count": len(by_case),
            "mixed_class_case_count": counts["mixed"],
            "negative_only_case_count": counts["negative_only"],
            "positive_only_case_count": counts["positive_only"],
            "every_full_center_has_both_classes": True,
            "every_support_scope_has_both_classes": True,
            "every_evaluation_scope_has_both_classes": True,
        }
    )


def _require_both_classes(labels: Sequence[BinaryLabelRow], scope: str) -> None:
    if {int(row.label) for row in labels} != {0, 1}:
        raise ProtocolError(f"Pooled exact BACC scope lacks both binary classes: {scope}.")


def _binary(value: object) -> int:
    try:
        number = float(str(value))
    except ValueError as exc:
        raise ProtocolError("Scoped pooled-BACC label is not numeric.") from exc
    if number not in (0.0, 1.0):
        raise ProtocolError("Scoped pooled-BACC label lies outside {0,1}.")
    return int(number)


__all__ = (
    "LabelAccessEvent",
    "LabelCapabilityManager",
    "audit_manifest_case_class_topology",
)
