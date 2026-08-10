"""Scoped label capabilities and the residual-stacker phase state machine."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from ....data.contract.stage70_target_evaluation.contracts import evaluation_row_id
from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_file
from .case_partitions import CaseOOFPartition
from .contracts import BinaryLabel
from .core_hashing import canonical_hash
from .experiment_contracts import (
    CENTERS,
    EXPECTED_CENTER_FOLD_COUNT,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MIXED_CLASS_CASE_COUNT,
    EXPECTED_NEGATIVE_ONLY_CASE_COUNT,
    EXPECTED_POSITIVE_ONLY_CASE_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    OOF_FOLD_COUNT,
)
from .input_contracts import LabelFreeTestFrame, TestRowIdentity
from .scientific_constants import METHOD_IDS


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
            **self.__dict__,
            "raw_labels_persisted": False,
        }


class LabelCapabilityManager:
    """The sole manifest reader; labels cannot cross an unsealed boundary."""

    def __init__(
        self,
        manifest_path: Path,
        frame: LabelFreeTestFrame,
        partition: CaseOOFPartition,
        *,
        global_prediction_seal_hash: str,
        label_free_feature_seal_hash: str,
    ) -> None:
        _require_sha256(global_prediction_seal_hash, "global_prediction_seal_hash")
        _require_sha256(label_free_feature_seal_hash, "label_free_feature_seal_hash")
        manifest_sha = sha256_file(manifest_path)
        if manifest_sha != EXPECTED_MANIFEST_SHA256:
            raise ProtocolError("Residual-stacker label manifest hash drifted.")
        frame_keys = {(r.center, r.case_id, r.evaluation_row_id) for r in frame.rows}
        partition_keys = {
            (r.center, r.case_id, r.evaluation_row_id) for r in partition.identities
        }
        if frame_keys != partition_keys:
            raise ProtocolError("Residual-stacker partition differs from the sealed frame.")
        self._manifest_path = Path(manifest_path)
        self._manifest_sha256 = manifest_sha
        self._frame = frame
        self._partition = partition
        self._prediction_seal_hash = global_prediction_seal_hash
        self._feature_seal_hash = label_free_feature_seal_hash
        self._loco_opened: set[str] = set()
        self._model_seals: dict[str, tuple[str, str, str]] = {}
        self._support_opened: set[tuple[str, int]] = set()
        self._method_decisions: dict[tuple[str, int, str], str] = {}
        self._all_decisions_seal_hash: str | None = None
        self._permutation_provenance_hash: str | None = None
        self._evaluation_opened = False
        self._topology_report: Mapping[str, object] | None = None
        self._events: list[LabelAccessEvent] = []

    def open_loco_donor_labels(self, heldout_target: str) -> tuple[BinaryLabel, ...]:
        target = str(heldout_target)
        if (
            target not in CENTERS
            or target in self._loco_opened
            or self._support_opened
            or self._method_decisions
            or self._evaluation_opened
        ):
            raise ProtocolError("Residual-stacker LOCO capability opened out of order.")
        requested = tuple(row for row in self._frame.rows if row.center != target)
        labels = self._open_rows(
            requested, role="loco_donor", target=target, fold=None
        )
        if any(row.target_center == target for row in labels):
            raise ProtocolError("Held-out target labels entered LOCO residual model.")
        self._loco_opened.add(target)
        return labels

    # Explicit alias makes the state machine easy to dependency-inject in tests.
    open_loco_model_labels = open_loco_donor_labels

    def record_loco_model_seals(
        self,
        heldout_target: str,
        global_model_hash: str,
        residual_model_hash: str,
        permuted_model_hash: str,
    ) -> None:
        target = str(heldout_target)
        _require_sha256(global_model_hash, "global_model_hash")
        _require_sha256(residual_model_hash, "residual_model_hash")
        _require_sha256(permuted_model_hash, "permuted_model_hash")
        if (
            target not in self._loco_opened
            or target in self._model_seals
            or self._support_opened
        ):
            raise ProtocolError("Residual-stacker LOCO model seal recorded out of order.")
        self._model_seals[target] = (
            global_model_hash,
            residual_model_hash,
            permuted_model_hash,
        )

    def open_fold_support_labels(
        self, target_center: str, fold_ordinal: int
    ) -> tuple[BinaryLabel, ...]:
        target, ordinal = str(target_center), int(fold_ordinal)
        key = (target, ordinal)
        if (
            set(self._model_seals) != set(CENTERS)
            or key in self._support_opened
            or any(value[:2] == key for value in self._method_decisions)
            or self._all_decisions_seal_hash is not None
            or self._evaluation_opened
        ):
            raise ProtocolError("Residual-stacker support capability opened out of order.")
        fold = self._partition.fold(target, ordinal)
        support_cases = set(fold.support_case_ids)
        requested = tuple(
            row for row in self._frame.rows_by_center[target]
            if row.case_id in support_cases
        )
        if {row.case_id for row in requested} != support_cases:
            raise ProtocolError("Residual-stacker support request lacks whole cases.")
        labels = self._open_rows(
            requested, role="target_support", target=target, fold=ordinal
        )
        _require_both_classes(labels, f"support H{target} fold {ordinal}")
        self._support_opened.add(key)
        return labels

    def record_fold_method_decision(
        self,
        target_center: str,
        fold_ordinal: int,
        method_id: str,
        decision_hash: str,
    ) -> None:
        key = (str(target_center), int(fold_ordinal))
        method = str(method_id)
        full = (*key, method)
        _require_sha256(decision_hash, "decision_hash")
        if (
            key not in self._support_opened
            or method not in METHOD_IDS
            or full in self._method_decisions
            or self._all_decisions_seal_hash is not None
            or self._evaluation_opened
        ):
            raise ProtocolError("Residual-stacker decision lacks scoped support capability.")
        self._method_decisions[full] = decision_hash

    def record_preevaluation_seals(
        self,
        decision_seal_hash: str,
        permutation_provenance_hash: str,
        *,
        decision_count: int,
    ) -> None:
        _require_sha256(decision_seal_hash, "decision_seal_hash")
        _require_sha256(permutation_provenance_hash, "permutation_provenance_hash")
        expected = {
            (center, fold, method)
            for center in CENTERS
            for fold in range(OOF_FOLD_COUNT)
            for method in METHOD_IDS
        }
        if (
            set(self._method_decisions) != expected
            or decision_count != EXPECTED_CENTER_FOLD_COUNT * len(METHOD_IDS)
            or self._all_decisions_seal_hash is not None
            or self._evaluation_opened
        ):
            raise ProtocolError(
                "Evaluation requires all 225 method decisions and permutation provenance."
            )
        self._all_decisions_seal_hash = decision_seal_hash
        self._permutation_provenance_hash = permutation_provenance_hash

    def open_oof_evaluation_labels(self) -> tuple[BinaryLabel, ...]:
        if (
            self._all_decisions_seal_hash is None
            or self._permutation_provenance_hash is None
            or self._evaluation_opened
        ):
            raise ProtocolError("Evaluation labels require durable pre-evaluation seals.")
        labels = self._open_rows(
            self._frame.rows,
            role="terminal_evaluation",
            target=None,
            fold=None,
        )
        topology = audit_manifest_case_class_topology(labels, partition=self._partition)
        if topology["status"] != "PASS":
            raise ProtocolError("Residual-stacker manifest class topology drifted.")
        self._topology_report = topology
        self._evaluation_opened = True
        return labels

    def access_report(self) -> Mapping[str, object]:
        payload = {
            "schema_version": "midogpp_residual_stacker_label_capability_report_v1",
            "status": "PASS" if self._evaluation_opened else "INCOMPLETE",
            "global_prediction_seal_hash": self._prediction_seal_hash,
            "label_free_feature_seal_hash": self._feature_seal_hash,
            "manifest_sha256": self._manifest_sha256,
            "loco_centers_opened": sorted(self._loco_opened),
            "loco_model_seals": {
                key: {"G": value[0], "R": value[1], "P": value[2]}
                for key, value in sorted(self._model_seals.items())
            },
            "fold_support_capability_count": len(self._support_opened),
            "fold_method_decision_count": len(self._method_decisions),
            "all_decisions_seal_hash": self._all_decisions_seal_hash,
            "permutation_provenance_hash": self._permutation_provenance_hash,
            "evaluation_labels_opened": self._evaluation_opened,
            "manifest_case_class_topology": (
                dict(self._topology_report) if self._topology_report is not None else None
            ),
            "events": [event.to_payload() for event in self._events],
            "raw_labels_persisted": False,
            "per_case_bacc_persisted": False,
            "target_expert_used": False,
            "shared_model_updated_with_target_labels": False,
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
    ) -> tuple[BinaryLabel, ...]:
        requested = {row.manifest_row_index: row for row in rows}
        labels: list[BinaryLabel] = []
        try:
            handle = self._manifest_path.open(newline="", encoding="utf-8")
        except OSError as exc:
            raise ProtocolError("Cannot open scoped residual-stacker label manifest.") from exc
        with handle:
            reader = csv.DictReader(handle)
            required = {"case_id", "center", "split", "label"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ProtocolError("Scoped residual-stacker manifest fields drifted.")
            for index, raw in enumerate(reader):
                wanted = requested.get(index)
                if wanted is None:
                    continue
                if (
                    evaluation_row_id(EXPECTED_MANIFEST_SHA256, index)
                    != wanted.evaluation_row_id
                    or str(raw["case_id"]) != wanted.case_id
                    or str(raw["center"]) != wanted.center
                    or str(raw["split"]) != wanted.split
                ):
                    raise ProtocolError("Scoped residual-stacker manifest identity drifted.")
                labels.append(
                    BinaryLabel(
                        target_center=wanted.center,
                        case_id=wanted.case_id,
                        sample_id=wanted.evaluation_row_id,
                        label=_binary(raw["label"]),
                        label_scope=role,  # type: ignore[arg-type]
                    )
                )
        if sha256_file(self._manifest_path) != self._manifest_sha256:
            raise ProtocolError("Scoped residual-stacker manifest changed during access.")
        ordered = tuple(sorted(labels))
        if len(ordered) != len(requested) or {row.sample_id for row in ordered} != {
            row.evaluation_row_id for row in rows
        }:
            raise ProtocolError("Scoped residual-stacker label coverage drifted.")
        self._events.append(
            LabelAccessEvent(
                role=role,
                target_center=target,
                fold_ordinal=fold,
                row_count=len(ordered),
                case_count=len({(row.target_center, row.case_id) for row in ordered}),
                row_identity_hash=canonical_hash([row.sample_id for row in ordered]),
                label_identity_hash=canonical_hash(
                    [[row.target_center, row.case_id, row.sample_id, row.label] for row in ordered]
                ),
            )
        )
        return ordered


def audit_manifest_case_class_topology(
    labels: Sequence[BinaryLabel], *, partition: CaseOOFPartition
) -> Mapping[str, object]:
    by_case: dict[tuple[str, str], set[int]] = {}
    for row in labels:
        by_case.setdefault((row.target_center, row.case_id), set()).add(row.label)
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
        raise ProtocolError("Manifest whole-case topology drifted from 213+4+1.")
    for center in CENTERS:
        _require_both_classes(
            tuple(row for row in labels if row.target_center == center),
            f"full center {center}",
        )
    for fold in partition.folds:
        rows = tuple(row for row in labels if row.target_center == fold.target_center)
        _require_both_classes(
            tuple(row for row in rows if row.case_id in fold.support_case_ids),
            f"support {fold.fold_id}",
        )
        _require_both_classes(
            tuple(row for row in rows if row.case_id in fold.evaluation_case_ids),
            f"evaluation {fold.fold_id}",
        )
    return MappingProxyType(
        {
            "status": "PASS",
            "total_case_count": len(by_case),
            "mixed_class_case_count": counts["mixed"],
            "negative_only_case_count": counts["negative_only"],
            "positive_only_case_count": counts["positive_only"],
            "every_pooled_legal_scope_has_both_classes": True,
        }
    )


def _require_both_classes(labels: Sequence[BinaryLabel], scope: str) -> None:
    if {row.label for row in labels} != {0, 1}:
        raise ProtocolError(f"Pooled exact BACC scope lacks both binary classes: {scope}.")


def _binary(value: object) -> int:
    try:
        number = float(str(value))
    except ValueError as exc:
        raise ProtocolError("Scoped residual-stacker label is not numeric.") from exc
    if number not in (0.0, 1.0):
        raise ProtocolError("Scoped residual-stacker label lies outside {0,1}.")
    return int(number)


def _require_sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ProtocolError(f"{name} must be a lowercase SHA-256.")
    return text


__all__ = (
    "LabelAccessEvent",
    "LabelCapabilityManager",
    "audit_manifest_case_class_topology",
)
