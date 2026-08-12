"""Per-decision label capabilities with held-evaluation noninterference."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from ....data.contract.stage70_target_evaluation.contracts import evaluation_row_id
from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_file
from .constants import CENTERS, OOF_FOLD_COUNT
from .experiment_contracts import EXPECTED_MANIFEST_SHA256
from .hashing import canonical_hash, require_sha256, require_stable_hash
from .input_contracts import LabelFreeTestFrame, TestRowIdentity
from .partitions import ThreeRoleFold, ThreeRolePartition


@dataclass(frozen=True, order=True)
class BinaryLabel:
    target_center: str
    case_id: str
    sample_id: str
    value: int

    def __post_init__(self) -> None:
        if (
            self.target_center not in CENTERS
            or self.value not in (0, 1)
            or isinstance(self.value, bool)
            or not self.case_id
            or not self.sample_id
        ):
            raise ProtocolError("Flip-router binary label is malformed.")


@dataclass(frozen=True)
class FoldPlan:
    target_center: str
    fold_ordinal: int
    selection_case_ids: tuple[str, ...]
    calibration_case_ids: tuple[str, ...]
    evaluation_case_ids: tuple[str, ...]
    prediction_seal_hash: str
    feature_seal_hash: str
    plan_hash: str

    @classmethod
    def from_fold(
        cls,
        fold: ThreeRoleFold,
        *,
        prediction_seal_hash: str,
        feature_seal_hash: str,
    ) -> "FoldPlan":
        require_stable_hash(prediction_seal_hash, "prediction_seal_hash")
        require_sha256(feature_seal_hash, "feature_seal_hash")
        unhashed = {
            "schema_version": "fixed_bank_flip_router_fold_plan_v1",
            "target_center": fold.target_center,
            "fold_ordinal": fold.fold_ordinal,
            "selection_case_ids": list(fold.selection_case_ids),
            "calibration_case_ids": list(fold.calibration_case_ids),
            "evaluation_case_ids": list(fold.evaluation_case_ids),
            "prediction_seal_hash": prediction_seal_hash,
            "feature_seal_hash": feature_seal_hash,
            "held_evaluation_labels_in_plan": False,
            "plan_hash_invariant_to_held_evaluation_label_values": True,
        }
        return cls(
            fold.target_center,
            fold.fold_ordinal,
            fold.selection_case_ids,
            fold.calibration_case_ids,
            fold.evaluation_case_ids,
            prediction_seal_hash,
            feature_seal_hash,
            canonical_hash(unhashed),
        )

    @property
    def key(self) -> tuple[str, int]:
        return self.target_center, self.fold_ordinal

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_flip_router_fold_plan_v1",
            "target_center": self.target_center,
            "fold_ordinal": self.fold_ordinal,
            "selection_case_ids": list(self.selection_case_ids),
            "calibration_case_ids": list(self.calibration_case_ids),
            "evaluation_case_ids": list(self.evaluation_case_ids),
            "prediction_seal_hash": self.prediction_seal_hash,
            "feature_seal_hash": self.feature_seal_hash,
            "held_evaluation_labels_in_plan": False,
            "plan_hash_invariant_to_held_evaluation_label_values": True,
            "plan_hash": self.plan_hash,
        }


@dataclass(frozen=True)
class LabelAccessEvent:
    role: str
    target_center: str | None
    fold_ordinal: int | None
    row_count: int
    case_count: int
    row_identity_hash: str
    intersects_own_evaluation: bool

    def to_payload(self) -> dict[str, object]:
        return {**self.__dict__, "raw_labels_persisted": False}


class FlipRouterLabelCapabilityManager:
    """Only manifest reader; each returned view is role and fold scoped."""

    def __init__(
        self,
        manifest_path: Path,
        frame: LabelFreeTestFrame,
        partition: ThreeRolePartition,
        *,
        prediction_seal_hash: str,
        feature_seal_hash: str,
    ) -> None:
        if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
            raise ProtocolError("Flip-router label manifest hash drifted.")
        require_stable_hash(prediction_seal_hash, "prediction_seal_hash")
        require_sha256(feature_seal_hash, "feature_seal_hash")
        frame_keys = {
            (row.center, row.case_id, row.evaluation_row_id) for row in frame.rows
        }
        partition_keys = {
            (row.target_center, row.case_id, row.sample_id)
            for row in partition.identities
        }
        if frame_keys != partition_keys:
            raise ProtocolError("Flip-router partition differs from sealed frame.")
        self._manifest_path = Path(manifest_path)
        self._frame = frame
        self._partition = partition
        self._prediction_seal_hash = prediction_seal_hash
        self._feature_seal_hash = feature_seal_hash
        self._plans: Mapping[tuple[str, int], FoldPlan] | None = None
        self._loco_opened: set[str] = set()
        self._donor_model_seals: dict[str, dict[str, str]] = {}
        self._selection_opened: set[tuple[str, int]] = set()
        self._calibration_opened: set[tuple[str, int]] = set()
        self._decision_seals: dict[tuple[str, int], str] = {}
        self._terminal_opened = False
        self._events: list[LabelAccessEvent] = []

    def seal_all_fold_plans(self) -> tuple[FoldPlan, ...]:
        if self._plans is not None or self._events:
            raise ProtocolError("Flip-router fold plans were sealed out of order.")
        plans = tuple(
            FoldPlan.from_fold(
                fold,
                prediction_seal_hash=self._prediction_seal_hash,
                feature_seal_hash=self._feature_seal_hash,
            )
            for fold in self._partition.folds
        )
        if len(plans) != len(CENTERS) * OOF_FOLD_COUNT:
            raise ProtocolError("Flip-router fold plan coverage drifted.")
        self._plans = MappingProxyType({plan.key: plan for plan in plans})
        return plans

    def open_loco_donor_labels(self, heldout_target: str) -> tuple[BinaryLabel, ...]:
        target = str(heldout_target)
        self._require_plans()
        if target not in CENTERS or target in self._loco_opened or self._terminal_opened:
            raise ProtocolError("Flip-router LOCO donor labels opened out of order.")
        rows = tuple(row for row in self._frame.rows if row.center != target)
        labels = self._open_rows(rows, role="loco_donor", target=target, fold=None)
        if any(label.target_center == target for label in labels):
            raise ProtocolError("Held-out H labels entered flip-router donor fitting.")
        self._loco_opened.add(target)
        return labels

    def record_H_specific_donor_model_seal(
        self,
        heldout_target: str,
        *,
        model_heldout_target: str,
        model_hash: str,
        provenance_hash: str,
    ) -> None:
        target = str(heldout_target)
        require_sha256(model_hash, "donor_model_hash")
        require_sha256(provenance_hash, "donor_model_provenance_hash")
        if (
            target not in self._loco_opened
            or target in self._donor_model_seals
            or str(model_heldout_target) != target
            or any(
                existing["model_hash"] == model_hash
                or existing["provenance_hash"] == provenance_hash
                for existing in self._donor_model_seals.values()
            )
            or self._selection_opened
            or self._terminal_opened
        ):
            raise ProtocolError("H-specific donor model seal was reused or misbound.")
        self._donor_model_seals[target] = {
            "heldout_target": target,
            "model_hash": model_hash,
            "provenance_hash": provenance_hash,
        }

    def open_selection_labels(self, target: str, fold_ordinal: int) -> tuple[BinaryLabel, ...]:
        plan = self._plan(target, fold_ordinal)
        key = plan.key
        if (
            plan.target_center not in self._donor_model_seals
            or key in self._selection_opened
            or self._terminal_opened
        ):
            raise ProtocolError("Flip-router selection labels opened out of order.")
        rows = tuple(
            row for row in self._frame.rows
            if row.center == plan.target_center and row.case_id in set(plan.selection_case_ids)
        )
        labels = self._open_rows(rows, role="source_selection", target=plan.target_center, fold=plan.fold_ordinal)
        self._selection_opened.add(key)
        return labels

    def open_calibration_labels(self, target: str, fold_ordinal: int) -> tuple[BinaryLabel, ...]:
        plan = self._plan(target, fold_ordinal)
        key = plan.key
        if key not in self._selection_opened or key in self._calibration_opened or self._terminal_opened:
            raise ProtocolError("Flip-router calibration labels opened out of order.")
        rows = tuple(
            row for row in self._frame.rows
            if row.center == plan.target_center and row.case_id in set(plan.calibration_case_ids)
        )
        labels = self._open_rows(rows, role="directional_calibration", target=plan.target_center, fold=plan.fold_ordinal)
        self._calibration_opened.add(key)
        return labels

    def record_fold_decision_seal(self, target: str, fold_ordinal: int, seal_hash: str) -> None:
        plan = self._plan(target, fold_ordinal)
        require_sha256(seal_hash, "fold_decision_seal_hash")
        if (
            plan.key not in self._calibration_opened
            or plan.key in self._decision_seals
            or self._terminal_opened
        ):
            raise ProtocolError("Flip-router fold decision sealed out of order.")
        self._decision_seals[plan.key] = seal_hash

    def open_terminal_evaluation_labels(self) -> tuple[BinaryLabel, ...]:
        self._require_plans()
        expected = {(center, fold) for center in CENTERS for fold in range(OOF_FOLD_COUNT)}
        if set(self._decision_seals) != expected or self._terminal_opened:
            raise ProtocolError("Terminal labels require all 45 fold decision seals.")
        labels = self._open_rows(
            self._frame.rows, role="terminal_evaluation", target=None, fold=None
        )
        self._terminal_opened = True
        return labels

    def report_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_flip_router_label_capability_report_v1",
            "status": "PASS" if self._terminal_opened else "INCOMPLETE",
            "prediction_seal_hash": self._prediction_seal_hash,
            "feature_seal_hash": self._feature_seal_hash,
            "fold_plan_count": 0 if self._plans is None else len(self._plans),
            "loco_target_count": len(self._loco_opened),
            "H_specific_donor_model_seal_count": len(self._donor_model_seals),
            "H_specific_donor_model_seals": {
                key: value for key, value in sorted(self._donor_model_seals.items())
            },
            "selection_capability_count": len(self._selection_opened),
            "calibration_capability_count": len(self._calibration_opened),
            "fold_decision_seal_count": len(self._decision_seals),
            "terminal_scoring_opened": self._terminal_opened,
            "every_nonterminal_access_excludes_its_own_evaluation_cases": all(
                not event.intersects_own_evaluation
                for event in self._events
                if event.role != "terminal_evaluation"
            ),
            "events": [event.to_payload() for event in self._events],
            "raw_labels_persisted": False,
        }

    def _require_plans(self) -> None:
        if self._plans is None:
            raise ProtocolError("Flip-router labels require durable prelabel fold plans.")

    def _plan(self, target: str, fold_ordinal: int) -> FoldPlan:
        self._require_plans()
        try:
            return self._plans[(str(target), int(fold_ordinal))]  # type: ignore[index]
        except KeyError as exc:
            raise ProtocolError("Flip-router fold plan is absent.") from exc

    def _open_rows(
        self,
        rows: Sequence[TestRowIdentity],
        *,
        role: str,
        target: str | None,
        fold: int | None,
    ) -> tuple[BinaryLabel, ...]:
        requested = {(row.center, row.case_id, row.evaluation_row_id): row for row in rows}
        labels = _read_labels(self._manifest_path, requested)
        intersects = False
        if target is not None and fold is not None:
            evaluation = set(self._plan(target, fold).evaluation_case_ids)
            intersects = any(
                label.target_center == target and label.case_id in evaluation
                for label in labels
            )
        elif role == "loco_donor" and target is not None:
            intersects = any(label.target_center == target for label in labels)
        if intersects:
            raise ProtocolError("Flip-router capability included held evaluation labels.")
        identities = [(label.target_center, label.case_id, label.sample_id) for label in labels]
        self._events.append(
            LabelAccessEvent(
                role,
                target,
                fold,
                len(labels),
                len({(label.target_center, label.case_id) for label in labels}),
                canonical_hash(identities),
                intersects,
            )
        )
        return labels


def _read_labels(
    path: Path,
    requested: Mapping[tuple[str, str, str], TestRowIdentity],
) -> tuple[BinaryLabel, ...]:
    found: dict[tuple[str, str, str], BinaryLabel] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for ordinal, raw in enumerate(reader):
                center = str(raw.get("center", ""))
                case_id = str(raw.get("case_id", ""))
                sample_id = evaluation_row_id(EXPECTED_MANIFEST_SHA256, ordinal)
                key = (center, case_id, sample_id)
                if key not in requested:
                    continue
                wanted = requested[key]
                if wanted.manifest_row_index != ordinal:
                    raise ProtocolError("Flip-router manifest order differs from sealed frame.")
                if key in found:
                    raise ProtocolError("Flip-router manifest contains a duplicate requested key.")
                try:
                    value = int(raw["label"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ProtocolError("Flip-router manifest label is malformed.") from exc
                found[key] = BinaryLabel(center, case_id, sample_id, value)
    except OSError as exc:
        raise ProtocolError("Cannot read flip-router label manifest.") from exc
    if set(found) != set(requested):
        raise ProtocolError("Flip-router label capability row coverage drifted.")
    return tuple(found[key] for key in requested)


__all__ = (
    "BinaryLabel",
    "FlipRouterLabelCapabilityManager",
    "FoldPlan",
    "LabelAccessEvent",
)
