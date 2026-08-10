"""Fail-closed label capabilities for the two-stage recoverability diagnostic."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from ....data.contract.stage70_target_evaluation.contracts import evaluation_row_id
from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_file
from .label_contracts import BinaryLabel
from .input_contracts import (
    LabelFreeTestFrame,
    TestRowIdentity,
)
from .case_partitions import CaseOOFPartition
from .constants import GEOMETRY_IDS, MIDOGPP_CENTERS
from .experiment_contracts import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MIXED_CLASS_CASE_COUNT,
    EXPECTED_NEGATIVE_ONLY_CASE_COUNT,
    EXPECTED_POSITIVE_ONLY_CASE_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    OOF_FOLD_COUNT,
)
from .hashing import canonical_hash, require_sha256


PRE_SUPPORT_METHOD_IDS = ("B", "U", "G", "R", "P")
PER_GEOMETRY_PRE_SUPPORT_METHOD_IDS = ("U", "G", "R", "P")
SUPPORT_METHOD_ID = "S_y"
EXPECTED_PRE_SUPPORT_DECISION_COUNT = len(MIDOGPP_CENTERS) * OOF_FOLD_COUNT * (
    1 + len(GEOMETRY_IDS) * len(PER_GEOMETRY_PRE_SUPPORT_METHOD_IDS)
)
EXPECTED_SUPPORT_DECISION_COUNT = (
    len(MIDOGPP_CENTERS) * OOF_FOLD_COUNT * len(GEOMETRY_IDS)
)
EXPECTED_ALL_DECISION_COUNT = (
    EXPECTED_PRE_SUPPORT_DECISION_COUNT + EXPECTED_SUPPORT_DECISION_COUNT
)


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
        return {**self.__dict__, "raw_labels_persisted": False}


class ActionabilityLabelCapabilityManager:
    """Sole manifest reader; each phase requires durable predecessor seals."""

    def __init__(
        self,
        manifest_path: Path,
        frame: LabelFreeTestFrame,
        partition: CaseOOFPartition,
        *,
        global_prediction_seal_hash: str,
        label_free_feature_seal_hash: str,
        action_library_hash: str,
    ) -> None:
        require_sha256(global_prediction_seal_hash, "global_prediction_seal_hash")
        require_sha256(label_free_feature_seal_hash, "label_free_feature_seal_hash")
        require_sha256(action_library_hash, "action_library_hash")
        manifest_sha = sha256_file(manifest_path)
        if manifest_sha != EXPECTED_MANIFEST_SHA256:
            raise ProtocolError("Actionability label manifest hash drifted.")
        frame_keys = {
            (row.center, row.case_id, row.evaluation_row_id) for row in frame.rows
        }
        partition_keys = {
            (row.target_center, row.case_id, row.sample_id)
            for row in partition.identities
        }
        if frame_keys != partition_keys:
            raise ProtocolError("Actionability partition differs from the sealed frame.")
        self._manifest_path = Path(manifest_path)
        self._manifest_sha256 = manifest_sha
        self._frame = frame
        self._partition = partition
        self._prediction_seal_hash = global_prediction_seal_hash
        self._feature_seal_hash = label_free_feature_seal_hash
        self._action_library_hash = action_library_hash
        self._loco_opened: set[str] = set()
        self._model_seals: dict[str, dict[str, str]] = {}
        self._pre_support_decisions: dict[tuple[str, int, str, str | None], str] = {}
        self._pre_support_seal_hash: str | None = None
        self._support_opened: set[tuple[str, int]] = set()
        self._support_decisions: dict[tuple[str, int, str], str] = {}
        self._all_decisions_seal_hash: str | None = None
        self._permutation_provenance_hash: str | None = None
        self._evaluation_opened = False
        self._topology_report: Mapping[str, object] | None = None
        self._events: list[LabelAccessEvent] = []

    def open_loco_donor_labels(self, heldout_target: str) -> tuple[BinaryLabel, ...]:
        target = str(heldout_target)
        if (
            target not in MIDOGPP_CENTERS
            or target in self._loco_opened
            or self._pre_support_decisions
            or self._support_opened
            or self._evaluation_opened
        ):
            raise ProtocolError("Actionability LOCO labels opened out of order.")
        requested = tuple(row for row in self._frame.rows if row.center != target)
        labels = self._open_rows(
            requested, role="loco_donor", target=target, fold=None
        )
        if any(row.target_center == target for row in labels):
            raise ProtocolError("Held-out H labels entered actionability G/R/P fitting.")
        self._loco_opened.add(target)
        return labels

    def record_loco_model_seals(
        self, heldout_target: str, model_hashes: Mapping[str, str]
    ) -> None:
        target = str(heldout_target)
        expected = {
            f"{geometry}:{family}"
            for geometry in GEOMETRY_IDS
            for family in ("G", "R", "P")
        }
        canonical = {str(key): str(value) for key, value in model_hashes.items()}
        if (
            target not in self._loco_opened
            or target in self._model_seals
            or set(canonical) != expected
            or self._pre_support_decisions
        ):
            raise ProtocolError("Actionability model seals recorded out of order.")
        for key, value in canonical.items():
            require_sha256(value, f"model seal {key}")
        self._model_seals[target] = dict(sorted(canonical.items()))

    def record_pre_support_decision(
        self,
        target_center: str,
        fold_ordinal: int,
        method_id: str,
        decision_hash: str,
        *,
        geometry_id: str | None,
    ) -> None:
        target, fold, method = (
            str(target_center),
            int(fold_ordinal),
            str(method_id),
        )
        require_sha256(decision_hash, "pre-support decision hash")
        if method == "B":
            valid_geometry = geometry_id is None
        else:
            valid_geometry = (
                method in PER_GEOMETRY_PRE_SUPPORT_METHOD_IDS
                and geometry_id in GEOMETRY_IDS
            )
        key = (target, fold, method, geometry_id)
        if (
            set(self._model_seals) != set(MIDOGPP_CENTERS)
            or target not in MIDOGPP_CENTERS
            or fold not in range(OOF_FOLD_COUNT)
            or not valid_geometry
            or key in self._pre_support_decisions
            or self._pre_support_seal_hash is not None
            or self._support_opened
        ):
            raise ProtocolError("Actionability pre-support decision is invalid.")
        self._pre_support_decisions[key] = decision_hash

    def record_pre_support_seal(self, seal_hash: str, *, decision_count: int) -> None:
        require_sha256(seal_hash, "pre-support decision seal")
        expected: set[tuple[str, int, str, str | None]] = set()
        for target in MIDOGPP_CENTERS:
            for fold in range(OOF_FOLD_COUNT):
                expected.add((target, fold, "B", None))
                expected.update(
                    (target, fold, method, geometry)
                    for geometry in GEOMETRY_IDS
                    for method in PER_GEOMETRY_PRE_SUPPORT_METHOD_IDS
                )
        if (
            set(self._pre_support_decisions) != expected
            or decision_count != EXPECTED_PRE_SUPPORT_DECISION_COUNT
            or self._pre_support_seal_hash is not None
            or self._support_opened
        ):
            raise ProtocolError("Target support requires every B/U/G/R/P decision.")
        self._pre_support_seal_hash = seal_hash

    def open_fold_support_labels(
        self, target_center: str, fold_ordinal: int
    ) -> tuple[BinaryLabel, ...]:
        target, fold = str(target_center), int(fold_ordinal)
        key = (target, fold)
        if (
            self._pre_support_seal_hash is None
            or key in self._support_opened
            or any(value[:2] == key for value in self._support_decisions)
            or self._all_decisions_seal_hash is not None
            or self._evaluation_opened
        ):
            raise ProtocolError("Actionability support labels opened out of order.")
        partition_fold = self._partition.fold(target, fold)
        support_cases = set(partition_fold.support_case_ids)
        requested = tuple(
            row
            for row in self._frame.rows_by_center[target]
            if row.case_id in support_cases
        )
        if {row.case_id for row in requested} != support_cases:
            raise ProtocolError("Actionability support request lacks whole cases.")
        labels = self._open_rows(
            requested, role="target_support", target=target, fold=fold
        )
        _require_both_classes(labels, f"support H{target} fold {fold}")
        self._support_opened.add(key)
        return labels

    def record_support_decision(
        self,
        target_center: str,
        fold_ordinal: int,
        geometry_id: str,
        decision_hash: str,
    ) -> None:
        key = (str(target_center), int(fold_ordinal))
        geometry = str(geometry_id)
        full = (*key, geometry)
        require_sha256(decision_hash, "support decision hash")
        if (
            key not in self._support_opened
            or geometry not in GEOMETRY_IDS
            or full in self._support_decisions
            or self._all_decisions_seal_hash is not None
            or self._evaluation_opened
        ):
            raise ProtocolError("S_y decision lacks its scoped support capability.")
        self._support_decisions[full] = decision_hash

    def record_preevaluation_seals(
        self,
        decision_seal_hash: str,
        permutation_provenance_hash: str,
        *,
        decision_count: int,
    ) -> None:
        require_sha256(decision_seal_hash, "all decision seal")
        require_sha256(permutation_provenance_hash, "permutation provenance")
        expected_support = {
            (target, fold, geometry)
            for target in MIDOGPP_CENTERS
            for fold in range(OOF_FOLD_COUNT)
            for geometry in GEOMETRY_IDS
        }
        if (
            set(self._support_decisions) != expected_support
            or decision_count != EXPECTED_ALL_DECISION_COUNT
            or self._all_decisions_seal_hash is not None
            or self._evaluation_opened
        ):
            raise ProtocolError("Evaluation requires every durable diagnostic decision.")
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
            raise ProtocolError("Actionability terminal label topology drifted.")
        self._topology_report = topology
        self._evaluation_opened = True
        return labels

    def access_report(self) -> Mapping[str, object]:
        payload = {
            "schema_version": "midogpp_actionability_label_capability_report_v1",
            "status": "PASS" if self._evaluation_opened else "INCOMPLETE",
            "global_prediction_seal_hash": self._prediction_seal_hash,
            "label_free_feature_seal_hash": self._feature_seal_hash,
            "action_library_hash": self._action_library_hash,
            "manifest_sha256": self._manifest_sha256,
            "loco_centers_opened": sorted(self._loco_opened),
            "loco_model_seals": {
                key: value for key, value in sorted(self._model_seals.items())
            },
            "pre_support_decision_count": len(self._pre_support_decisions),
            "pre_support_decision_seal_hash": self._pre_support_seal_hash,
            "fold_support_capability_count": len(self._support_opened),
            "support_decision_count": len(self._support_decisions),
            "all_decisions_seal_hash": self._all_decisions_seal_hash,
            "permutation_provenance_hash": self._permutation_provenance_hash,
            "evaluation_labels_opened": self._evaluation_opened,
            "manifest_case_class_topology": (
                dict(self._topology_report)
                if self._topology_report is not None
                else None
            ),
            "events": [event.to_payload() for event in self._events],
            "raw_labels_persisted": False,
            "per_case_bacc_persisted": False,
            "target_expert_used": False,
            "shared_model_updated_with_target_labels": False,
            "geometry_selected": False,
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
            raise ProtocolError("Cannot open scoped actionability manifest.") from exc
        with handle:
            reader = csv.DictReader(handle)
            required = {"case_id", "center", "split", "label"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ProtocolError("Actionability manifest fields drifted.")
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
                    raise ProtocolError("Actionability manifest identity drifted.")
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
            raise ProtocolError("Actionability manifest changed during label access.")
        ordered = tuple(sorted(labels))
        if len(ordered) != len(requested) or {row.sample_id for row in ordered} != {
            row.evaluation_row_id for row in rows
        }:
            raise ProtocolError("Actionability scoped label coverage drifted.")
        self._events.append(
            LabelAccessEvent(
                role=role,
                target_center=target,
                fold_ordinal=fold,
                row_count=len(ordered),
                case_count=len(
                    {(row.target_center, row.case_id) for row in ordered}
                ),
                row_identity_hash=canonical_hash(
                    [row.sample_id for row in ordered]
                ),
                label_identity_hash=canonical_hash(
                    [
                        [
                            row.target_center,
                            row.case_id,
                            row.sample_id,
                            row.label,
                        ]
                        for row in ordered
                    ]
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
    expected_cases = {
        (fold.target_center, case_id)
        for fold in partition.folds
        for case_id in fold.evaluation_case_ids
    }
    status = (
        "PASS"
        if len(by_case) == EXPECTED_TOTAL_CASE_COUNT
        and set(by_case) == expected_cases
        and counts["mixed"] == EXPECTED_MIXED_CLASS_CASE_COUNT
        and counts["negative_only"] == EXPECTED_NEGATIVE_ONLY_CASE_COUNT
        and counts["positive_only"] == EXPECTED_POSITIVE_ONLY_CASE_COUNT
        else "FAIL"
    )
    payload = {
        "status": status,
        "case_count": len(by_case),
        **counts,
        "single_class_cases_retained": True,
        "per_case_bacc_used": False,
    }
    return MappingProxyType({**payload, "topology_hash": canonical_hash(payload)})


def _require_both_classes(labels: Sequence[BinaryLabel], role: str) -> None:
    if {row.label for row in labels} != {0, 1}:
        raise ProtocolError(f"{role} must contain both classes.")


def _binary(value: object) -> int:
    text = str(value).strip()
    if text not in {"0", "1"}:
        raise ProtocolError("Actionability manifest label must be binary.")
    return int(text)


__all__ = (
    "ActionabilityLabelCapabilityManager",
    "EXPECTED_ALL_DECISION_COUNT",
    "EXPECTED_PRE_SUPPORT_DECISION_COUNT",
    "EXPECTED_SUPPORT_DECISION_COUNT",
    "LabelAccessEvent",
    "audit_manifest_case_class_topology",
)
