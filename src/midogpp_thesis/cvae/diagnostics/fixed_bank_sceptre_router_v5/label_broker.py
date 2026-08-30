"""Manager-owned, role-scoped, one-shot target-label broker for v5."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from midogpp_thesis.data.contract.stage70_target_evaluation.contracts import (
    evaluation_row_id,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import sha256_file

from ..fixed_bank_sceptre_router.hashing import canonical_hash, require_sha256
from ..fixed_bank_sceptre_router.partitions import FOLD_COUNT, ThreeRolePartition
from ..fixed_bank_sceptre_router.phase_contracts import (
    PhaseCapability,
    TerminalEvaluationCapability,
)
from .phase_manager import CandidateSetPhaseManager


@dataclass(frozen=True, slots=True)
class ScopedRoleLabels:
    target_center: str
    fold_ordinal: int
    role: str
    case_set_hash: str
    row_ordinals: tuple[int, ...]
    observation_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    labels: tuple[int, ...]
    event_hash: str

    def __post_init__(self) -> None:
        count = len(self.labels)
        if (
            not count
            or len(self.row_ordinals) != count
            or len(self.observation_ids) != count
            or len(self.case_ids) != count
            or len(set(self.observation_ids)) != count
            or any(
                isinstance(value, bool) or value not in (0, 1)
                for value in self.labels
            )
            or set(self.labels) != {0, 1}
        ):
            raise ProtocolError("SCEPTRE v5 scoped-label geometry drifted.")
        require_sha256(self.case_set_hash, "scoped-label case set")
        require_sha256(self.event_hash, "scoped-label event")

    def __reduce__(self):  # pragma: no cover - explicit safety seam
        raise TypeError("SCEPTRE v5 raw scoped labels cannot be serialized.")


class RoleLabelBroker:
    """Bind label openings to process-local manager capabilities."""

    def __init__(
        self,
        *,
        manager: CandidateSetPhaseManager,
        partition: ThreeRolePartition,
        frame: object,
        manifest_path: str | Path,
        expected_manifest_sha256: str,
        prediction_store_hash: str,
        authorization_lease_hash: str,
    ) -> None:
        if not isinstance(manager, CandidateSetPhaseManager):
            raise ProtocolError("SCEPTRE v5 broker requires its phase manager.")
        if not isinstance(partition, ThreeRolePartition):
            raise ProtocolError("SCEPTRE v5 broker requires its partition.")
        path = Path(manifest_path)
        expected_manifest = require_sha256(
            expected_manifest_sha256, "SCEPTRE v5 test manifest"
        )
        prediction = require_sha256(
            prediction_store_hash, "SCEPTRE v5 prediction store"
        )
        lease = require_sha256(
            authorization_lease_hash, "SCEPTRE v5 authorization lease"
        )
        rows = tuple(getattr(frame, "rows", ()))
        required = (
            "row_ordinal",
            "manifest_row_index",
            "sample_id",
            "case_id",
            "center",
        )
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != expected_manifest
            or not rows
            or any(not hasattr(row, field) for row in rows for field in required)
        ):
            raise ProtocolError("SCEPTRE v5 broker inputs drifted.")
        row_ordinals = tuple(int(row.row_ordinal) for row in rows)
        manifest_ordinals = tuple(int(row.manifest_row_index) for row in rows)
        sample_ids = tuple(str(row.sample_id) for row in rows)
        case_keys = {(str(row.center), str(row.case_id)) for row in rows}
        partition_case_keys = {
            (identity.target_center, identity.case_id)
            for identity in partition.identities
        }
        if (
            any(value < 0 for value in (*row_ordinals, *manifest_ordinals))
            or len(set(row_ordinals)) != len(rows)
            or len(set(manifest_ordinals)) != len(rows)
            or len(set(sample_ids)) != len(rows)
            or any(not value or value.strip() != value for value in sample_ids)
            or case_keys != partition_case_keys
            or any(str(row.center) not in CENTERS for row in rows)
        ):
            raise ProtocolError("SCEPTRE v5 broker row identities drifted.")
        self._manager = manager
        self._partition = partition
        self._rows = rows
        self._manifest_path = path
        self._manifest_sha256 = expected_manifest
        self._prediction_store_hash = prediction
        self._lease_hash = lease
        self._role_rows = self._build_role_row_inventory()
        self._issued: dict[tuple[str, str, int], object] = {}
        self._consumed: set[tuple[str, str, int]] = set()
        self._events: list[Mapping[str, object]] = []
        self._terminal: TerminalEvaluationCapability | None = None

    @property
    def prediction_store_hash(self) -> str:
        return self._prediction_store_hash

    @property
    def partition_hash(self) -> str:
        return self._partition.partition_hash

    def issue_selection(self, target: str, fold: int) -> PhaseCapability:
        capability = self._manager.issue_selection_capability(target, fold)
        self._remember("SELECTION", capability)
        return capability

    def issue_calibration(self, target: str, fold: int) -> PhaseCapability:
        capability = self._manager.issue_calibration_capability(target, fold)
        self._remember("CALIBRATION", capability)
        return capability

    def open_selection(self, capability: PhaseCapability) -> ScopedRoleLabels:
        return self._open_phase("SELECTION", capability)

    def open_calibration(self, capability: PhaseCapability) -> ScopedRoleLabels:
        return self._open_phase("CALIBRATION", capability)

    def skip_calibration_without_labels(
        self, capability: PhaseCapability, *, support_decision_hash: str
    ) -> str:
        key = self._phase_key("CALIBRATION", capability)
        if self._issued.get(key) is not capability or key in self._consumed:
            raise ProtocolError("SCEPTRE v5 calibration skip is forged or reused.")
        fold = self._partition.fold(key[1], key[2])
        event = self._append_event(
            {
                "event": "CALIBRATION_SKIPPED_SUPPORT_FALLBACK",
                "target_center": key[1],
                "fold_ordinal": key[2],
                "case_set_hash": fold.case_set_hash("CALIBRATION"),
                "row_count": 0,
                "support_decision_hash": require_sha256(
                    support_decision_hash, "support fallback decision"
                ),
                "manifest_rows_decoded": 0,
                "raw_labels_persisted": False,
            }
        )
        self._consumed.add(key)
        return str(event["event_hash"])

    def activate_terminal(
        self, capability: TerminalEvaluationCapability
    ) -> None:
        if not isinstance(capability, TerminalEvaluationCapability) or self._terminal:
            raise ProtocolError("SCEPTRE v5 terminal capability is invalid or reused.")
        self._manager.activate_terminal_capability(capability)
        self._terminal = capability
        self._append_event(
            {
                "event": "TERMINAL_CAPABILITY_ACTIVATED",
                "target_center": None,
                "fold_ordinal": None,
                "case_set_hash": None,
                "row_count": 0,
                "terminal_capability_hash": require_sha256(
                    capability.capability_hash, "terminal capability"
                ),
                "manifest_rows_decoded": 0,
                "raw_labels_persisted": False,
            }
        )

    def open_evaluation(
        self,
        target: str,
        fold_ordinal: int,
        capability: TerminalEvaluationCapability,
    ) -> ScopedRoleLabels:
        if self._terminal is not capability:
            raise ProtocolError("SCEPTRE v5 evaluation lacks activated capability.")
        key = ("EVALUATION", str(target), int(fold_ordinal))
        if key in self._consumed:
            raise ProtocolError("SCEPTRE v5 evaluation scope opened twice.")
        fold = self._partition.fold(key[1], key[2])
        if capability.partition_hash != self._partition.partition_hash:
            raise ProtocolError("SCEPTRE v5 terminal partition drifted.")
        labels = self._decode(key, fold.case_set_hash("EVALUATION"))
        self._consumed.add(key)
        return labels

    def journal_payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "schema_version": "sceptre_v5_label_capability_journal_v1",
                "partition_hash": self._partition.partition_hash,
                "prediction_store_hash": self._prediction_store_hash,
                "authorization_lease_hash": self._lease_hash,
                "manifest_sha256": self._manifest_sha256,
                "events": [dict(row) for row in self._events],
                "raw_labels_persisted": False,
                "sample_paths_persisted": False,
                "journal_hash": self._journal_hash(),
            }
        )

    def _remember(self, role: str, capability: PhaseCapability) -> None:
        key = self._phase_key(role, capability)
        if key in self._issued:
            raise ProtocolError("SCEPTRE v5 broker capability issued twice.")
        self._issued[key] = capability

    def _open_phase(
        self, role: str, capability: PhaseCapability
    ) -> ScopedRoleLabels:
        key = self._phase_key(role, capability)
        if self._issued.get(key) is not capability or key in self._consumed:
            raise ProtocolError("SCEPTRE v5 label capability is forged or reused.")
        fold = self._partition.fold(key[1], key[2])
        labels = self._decode(key, fold.case_set_hash(role))
        self._consumed.add(key)
        return labels

    @staticmethod
    def _phase_key(
        role: str, capability: PhaseCapability
    ) -> tuple[str, str, int]:
        if not isinstance(capability, PhaseCapability) or capability.role != f"{role}_LABELS":
            raise ProtocolError("SCEPTRE v5 capability role drifted.")
        return role, capability.target_center, capability.fold_ordinal

    def _decode(
        self, key: tuple[str, str, int], case_set_hash: str
    ) -> ScopedRoleLabels:
        role, target, fold_ordinal = key
        rows = self._role_rows[key]
        identity_hash = canonical_hash(
            [
                {
                    "row_ordinal": int(row.row_ordinal),
                    "manifest_row_index": int(row.manifest_row_index),
                    "sample_id": str(row.sample_id),
                    "case_id": str(row.case_id),
                    "center": str(row.center),
                }
                for row in rows
            ]
        )
        labels = self._read_exact_rows(rows)
        event = self._append_event(
            {
                "event": f"{role}_LABELS_DECODED",
                "target_center": target,
                "fold_ordinal": fold_ordinal,
                "case_set_hash": case_set_hash,
                "row_count": len(rows),
                "manifest_rows_decoded": len(rows),
                "row_identity_set_hash": identity_hash,
                "raw_labels_persisted": False,
            }
        )
        return ScopedRoleLabels(
            target_center=target,
            fold_ordinal=fold_ordinal,
            role=role,
            case_set_hash=case_set_hash,
            row_ordinals=tuple(int(row.row_ordinal) for row in rows),
            observation_ids=tuple(str(row.sample_id) for row in rows),
            case_ids=tuple(str(row.case_id) for row in rows),
            labels=labels,
            event_hash=str(event["event_hash"]),
        )

    def _build_role_row_inventory(
        self,
    ) -> dict[tuple[str, str, int], tuple[object, ...]]:
        inventory = {}
        targets = tuple(dict.fromkeys(str(row.center) for row in self._rows))
        if targets != tuple(CENTERS):
            raise ProtocolError("SCEPTRE v5 target inventory drifted.")
        for target in targets:
            for fold_ordinal in range(FOLD_COUNT):
                fold = self._partition.fold(target, fold_ordinal)
                roles = {
                    "SELECTION": fold.selection_case_ids,
                    "CALIBRATION": fold.calibration_case_ids,
                    "EVALUATION": fold.evaluation_case_ids,
                }
                for role, case_ids in roles.items():
                    allowed = set(case_ids)
                    rows = tuple(
                        row
                        for row in self._rows
                        if str(row.center) == target
                        and str(row.case_id) in allowed
                    )
                    if not rows or {str(row.case_id) for row in rows} != allowed:
                        raise ProtocolError("SCEPTRE v5 role inventory drifted.")
                    inventory[(role, target, fold_ordinal)] = rows
        if len(inventory) != len(CENTERS) * FOLD_COUNT * 3:
            raise ProtocolError("SCEPTRE v5 role inventory is incomplete.")
        return inventory

    def _read_exact_rows(self, rows: tuple[object, ...]) -> tuple[int, ...]:
        requested = {int(row.manifest_row_index): row for row in rows}
        found: dict[int, int] = {}
        before = sha256_file(self._manifest_path)
        try:
            with self._manifest_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = tuple(next(reader))
                if any(name not in header for name in ("case_id", "center", "label")):
                    raise ProtocolError("SCEPTRE v5 manifest header drifted.")
                position = {name: header.index(name) for name in ("case_id", "center", "label")}
                for ordinal, raw in enumerate(reader):
                    expected = requested.get(ordinal)
                    if expected is None:
                        continue
                    sample_id = evaluation_row_id(self._manifest_sha256, ordinal)
                    if (
                        len(raw) != len(header)
                        or raw[position["center"]] != str(expected.center)
                        or raw[position["case_id"]] != str(expected.case_id)
                        or sample_id != str(expected.sample_id)
                    ):
                        raise ProtocolError("SCEPTRE v5 manifest row identity drifted.")
                    value = int(raw[position["label"]])
                    if value not in (0, 1) or ordinal in found:
                        raise ProtocolError("SCEPTRE v5 granted label drifted.")
                    found[ordinal] = value
        except ProtocolError:
            raise
        except (OSError, StopIteration, TypeError, ValueError) as exc:
            raise ProtocolError("Cannot decode SCEPTRE v5 scoped labels.") from exc
        if (
            before != self._manifest_sha256
            or sha256_file(self._manifest_path) != self._manifest_sha256
            or set(found) != set(requested)
        ):
            raise ProtocolError("SCEPTRE v5 manifest changed or coverage drifted.")
        return tuple(found[int(row.manifest_row_index)] for row in rows)

    def _append_event(self, body: Mapping[str, object]) -> Mapping[str, object]:
        predecessor = self._events[-1]["event_hash"] if self._events else None
        unhashed = {
            "schema_version": "sceptre_v5_label_event_v1",
            "event_ordinal": len(self._events),
            "predecessor_event_hash": predecessor,
            "prediction_store_hash": self._prediction_store_hash,
            "authorization_lease_hash": self._lease_hash,
            **dict(body),
        }
        event = MappingProxyType(
            {**unhashed, "event_hash": canonical_hash(unhashed)}
        )
        self._events.append(event)
        return event

    def _journal_hash(self) -> str:
        return canonical_hash(
            {
                "schema_version": "sceptre_v5_label_journal_chain_v1",
                "partition_hash": self._partition.partition_hash,
                "prediction_store_hash": self._prediction_store_hash,
                "authorization_lease_hash": self._lease_hash,
                "manifest_sha256": self._manifest_sha256,
                "event_hashes": [row["event_hash"] for row in self._events],
                "raw_labels_persisted": False,
            }
        )


__all__ = ("RoleLabelBroker", "ScopedRoleLabels")
