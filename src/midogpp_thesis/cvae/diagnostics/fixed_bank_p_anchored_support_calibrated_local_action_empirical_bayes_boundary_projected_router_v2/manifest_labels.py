"""Capability-gated decoding of the consumed MIDOG++ test labels.

The immutable cache remains the authority for row identity and order.  This
module opens the label-bearing manifest only while an exact donor, H-minus-c
support, or terminal capability is active.  Returned label views are
deliberately non-pickleable and expose no persistence method.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .hashing import canonical_hash
from .identity import (
    CENTERS,
    EXPECTED_CASE_COUNT,
    EXPECTED_TEST_MANIFEST_SHA256,
    EXPECTED_TEST_ROW_COUNT,
    GovernanceError,
)
from .input_contracts import LabelFreeTestFrame, TestRowIdentity
from .label_identity import LabelIdentityFrame
from .label_capabilities import (
    DONOR,
    SUPPORT,
    TERMINAL,
    LabelCapability,
    LabelCapabilityJournal,
)


_REQUIRED_COLUMNS = frozenset({"sample_id", "case_id", "label", "center", "split"})


@dataclass(frozen=True, slots=True, eq=False)
class ScopedCaseLabels:
    """In-memory labels for one preterminal capability scope."""

    kind: str
    scope_hash: str
    labels_by_center_case: Mapping[str, Mapping[str, np.ndarray]]
    row_count: int
    case_count: int
    identity_hash: str

    def __post_init__(self) -> None:
        outer: dict[str, Mapping[str, np.ndarray]] = {}
        observed_rows = 0
        observed_cases = 0
        for center, case_rows in self.labels_by_center_case.items():
            normalized: dict[str, np.ndarray] = {}
            for case_id, labels in case_rows.items():
                values = np.ascontiguousarray(labels, dtype=np.int8)
                if values.ndim != 1 or len(values) == 0 or not np.isin(values, (0, 1)).all():
                    raise GovernanceError("SCALE-BP v2 scoped labels drifted.")
                values.setflags(write=False)
                normalized[str(case_id)] = values
                observed_rows += len(values)
                observed_cases += 1
            if not normalized:
                raise GovernanceError("SCALE-BP v2 scoped label center is empty.")
            outer[str(center)] = MappingProxyType(normalized)
        if (
            self.kind not in {DONOR, SUPPORT}
            or not self.scope_hash
            or not self.identity_hash
            or observed_rows != int(self.row_count)
            or observed_cases != int(self.case_count)
        ):
            raise GovernanceError("SCALE-BP v2 scoped label inventory drifted.")
        object.__setattr__(self, "labels_by_center_case", MappingProxyType(outer))

    def labels_for_case(self, center: object, case_id: object) -> np.ndarray:
        try:
            return self.labels_by_center_case[str(center)][str(case_id)]
        except KeyError as exc:
            raise GovernanceError("SCALE-BP v2 requested labels are outside the scope.") from exc

    def case_ids(self, center: object) -> tuple[str, ...]:
        try:
            return tuple(self.labels_by_center_case[str(center)])
        except KeyError as exc:
            raise GovernanceError("SCALE-BP v2 requested label center is outside the scope.") from exc

    def __reduce__(self) -> object:  # pragma: no cover - exercised by pickle callers
        raise TypeError("SCALE-BP v2 raw label views cannot be serialized.")


@dataclass(frozen=True, slots=True, eq=False)
class TerminalLabelVector:
    """Exact frame-ordered terminal labels, available only after decision seal."""

    labels: np.ndarray
    centers: np.ndarray
    identity_hash: str
    scope_hash: str

    def __post_init__(self) -> None:
        labels = np.ascontiguousarray(self.labels, dtype=np.int8)
        centers = np.ascontiguousarray([str(value) for value in self.centers])
        if (
            labels.shape != (EXPECTED_TEST_ROW_COUNT,)
            or centers.shape != labels.shape
            or not np.isin(labels, (0, 1)).all()
            or set(str(value) for value in centers) != set(CENTERS)
            or not self.identity_hash
            or not self.scope_hash
        ):
            raise GovernanceError("SCALE-BP v2 terminal label vector drifted.")
        labels.setflags(write=False)
        centers.setflags(write=False)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "centers", centers)

    def __reduce__(self) -> object:  # pragma: no cover - exercised by pickle callers
        raise TypeError("SCALE-BP v2 terminal labels cannot be serialized.")


class ManifestLabelDecoder:
    """Bind a label-free cache frame to the byte-exact canonical manifest."""

    def __init__(
        self,
        frame: LabelFreeTestFrame | LabelIdentityFrame,
        manifest_path: str | Path,
    ):
        if not isinstance(frame, (LabelFreeTestFrame, LabelIdentityFrame)):
            raise GovernanceError("SCALE-BP v2 label decoder requires its cache frame.")
        path = Path(manifest_path)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise GovernanceError("SCALE-BP v2 label manifest path is unsafe.")
        if _sha256_file(path) != EXPECTED_TEST_MANIFEST_SHA256:
            raise GovernanceError("SCALE-BP v2 label manifest bytes drifted.")
        self._frame = frame
        self._manifest_path = path.resolve(strict=True)

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    def donor_identity_hash(self, outer_center: object) -> str:
        outer = str(outer_center)
        donors = tuple(center for center in CENTERS if center != outer)
        if outer not in CENTERS:
            raise GovernanceError("SCALE-BP v2 donor identity target is unknown.")
        rows = tuple(
            _identity_payload(row)
            for center in donors
            for row in self._frame.rows_by_center[center]
        )
        return canonical_hash(
            {
                "schema_version": "scale_bp_v2_donor_label_identity_v1",
                "outer_center": outer,
                "donor_centers": donors,
                "rows": rows,
            }
        )

    def support_identity_hashes(
        self, target_center: object, held_case_id: object
    ) -> tuple[str, str]:
        target, held = str(target_center), str(held_case_id)
        if target not in CENTERS:
            raise GovernanceError("SCALE-BP v2 support identity target is unknown.")
        target_rows = self._frame.rows_by_center[target]
        evaluation = tuple(row for row in target_rows if row.case_id == held)
        support = tuple(
            row
            for row in target_rows
            if row.patient_slide_group_id
            not in {item.patient_slide_group_id for item in evaluation}
        )
        if (
            not evaluation
            or not support
            or held in {row.case_id for row in support}
            or {row.patient_slide_group_id for row in evaluation}
            & {row.patient_slide_group_id for row in support}
        ):
            raise GovernanceError("SCALE-BP v2 H-minus-c identity boundary drifted.")
        support_hash = canonical_hash(
            {
                "schema_version": "scale_bp_v2_support_label_identity_v1",
                "target_center": target,
                "held_case_id": held,
                "rows": tuple(_identity_payload(row) for row in support),
            }
        )
        evaluation_hash = canonical_hash(
            {
                "schema_version": "scale_bp_v2_evaluation_label_identity_v1",
                "target_center": target,
                "held_case_id": held,
                "rows": tuple(_identity_payload(row) for row in evaluation),
            }
        )
        return support_hash, evaluation_hash

    def terminal_identity_hash(self) -> str:
        return canonical_hash(
            {
                "schema_version": "scale_bp_v2_terminal_label_identity_v1",
                "manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
                "frame_hash": self._frame.frame_hash,
                "rows": tuple(_identity_payload(row) for row in self._frame.rows),
                "row_count": EXPECTED_TEST_ROW_COUNT,
                "case_count": EXPECTED_CASE_COUNT,
            }
        )

    def decode_donor(
        self,
        journal: LabelCapabilityJournal,
        capability: LabelCapability,
        *,
        outer_center: object,
    ) -> ScopedCaseLabels:
        outer = str(outer_center)
        journal.assert_active(capability, kind=DONOR)
        identity_hash = self.donor_identity_hash(outer)
        donors = tuple(center for center in CENTERS if center != outer)
        expected_scope = canonical_hash(
            {
                "outer_center": outer,
                "donor_centers": list(donors),
                "row_identity_hash": identity_hash,
                "outer_center_labels_available": False,
                "target_expert_available": False,
            }
        )
        if capability.scope_hash not in {expected_scope, identity_hash}:
            raise GovernanceError("SCALE-BP v2 donor capability binding drifted.")
        selected_rows = tuple(
            row
            for center in donors
            for row in self._frame.rows_by_center[center]
        )
        labels = self._decode_selected(selected_rows)
        selected = _group_labels(
            self._frame, labels, centers=donors, selected_rows=selected_rows
        )
        return ScopedCaseLabels(
            DONOR,
            capability.scope_hash,
            selected,
            sum(len(values) for cases in selected.values() for values in cases.values()),
            sum(len(cases) for cases in selected.values()),
            identity_hash,
        )

    def decode_support(
        self,
        journal: LabelCapabilityJournal,
        capability: LabelCapability,
        *,
        target_center: object,
        held_case_id: object,
    ) -> ScopedCaseLabels:
        target, held = str(target_center), str(held_case_id)
        journal.assert_active(capability, kind=SUPPORT)
        support_hash, evaluation_hash = self.support_identity_hashes(target, held)
        expected_scope = canonical_hash(
            {
                "target_center": target,
                "held_case_id": held,
                "support_identity_hash": support_hash,
                "evaluation_identity_hash": evaluation_hash,
                "held_case_labels_available": False,
                "support_updates_global_state": False,
                "support_updates_source_experts": False,
                "support_tunes_hyperparameters": False,
            }
        )
        delegated_scope = canonical_hash(
            {
                "schema_version": "scale_bp_v2_worker_support_scope_v1",
                "held_case_id": held,
                "support_identity_hash": support_hash,
                "evaluation_identity_hash": evaluation_hash,
                "support_evaluation_disjoint": True,
            }
        )
        if capability.scope_hash not in {expected_scope, delegated_scope}:
            raise GovernanceError("SCALE-BP v2 support capability binding drifted.")
        evaluation_groups = {
            row.patient_slide_group_id
            for row in self._frame.rows_by_center[target]
            if row.case_id == held
        }
        selected_rows = tuple(
            row
            for row in self._frame.rows_by_center[target]
            if row.patient_slide_group_id not in evaluation_groups
        )
        labels = self._decode_selected(selected_rows)
        grouped = _group_labels(
            self._frame,
            labels,
            centers=(target,),
            selected_rows=selected_rows,
        )
        support_cases = dict(grouped[target])
        if held in support_cases:
            raise GovernanceError("SCALE-BP v2 held case entered support labels.")
        selected = {target: MappingProxyType(support_cases)}
        return ScopedCaseLabels(
            SUPPORT,
            capability.scope_hash,
            selected,
            sum(len(values) for values in support_cases.values()),
            len(support_cases),
            support_hash,
        )

    def decode_terminal(
        self,
        journal: LabelCapabilityJournal,
        capability: LabelCapability,
    ) -> TerminalLabelVector:
        journal.assert_active(capability, kind=TERMINAL)
        identity_hash = self.terminal_identity_hash()
        expected_scope = canonical_hash(
            {
                "terminal_identity_hash": identity_hash,
                "decision_seal_hash": capability.decision_seal_hash,
                "may_update_preterminal_state": False,
                "may_reselect_actions": False,
            }
        )
        if capability.scope_hash != expected_scope or capability.decision_seal_hash is None:
            raise GovernanceError("SCALE-BP v2 terminal capability binding drifted.")
        decoded = self._decode_selected(self._frame.rows)
        labels = np.ascontiguousarray(
            [decoded[row.row_ordinal] for row in self._frame.rows], dtype=np.int8
        )
        centers = np.asarray([row.center for row in self._frame.rows])
        return TerminalLabelVector(labels, centers, identity_hash, capability.scope_hash)

    def _decode_selected(
        self, selected_rows: tuple[TestRowIdentity, ...]
    ) -> Mapping[int, int]:
        """Parse only capability-approved CSV rows.

        Unselected lines are advanced as opaque text and are never passed to a
        CSV parser, so their label cells are not materialized under a donor or
        support capability.
        """

        selected = {row.manifest_row_index: row for row in selected_rows}
        if (
            not selected
            or len(selected) != len(selected_rows)
            or any(index < 0 for index in selected)
        ):
            raise GovernanceError("SCALE-BP v2 selected label identity drifted.")
        before = _sha256_file(self._manifest_path)
        decoded: dict[int, int] = {}
        try:
            with self._manifest_path.open("r", encoding="utf-8", newline="") as handle:
                header_line = handle.readline()
                header_rows = tuple(csv.reader((header_line,)))
                if len(header_rows) != 1:
                    raise GovernanceError("SCALE-BP v2 label manifest header drifted.")
                header = tuple(header_rows[0])
                if not _REQUIRED_COLUMNS <= set(header) or len(header) != len(set(header)):
                    raise GovernanceError("SCALE-BP v2 label manifest schema drifted.")
                positions = {name: header.index(name) for name in _REQUIRED_COLUMNS}
                for manifest_index, raw_line in enumerate(handle):
                    identity = selected.get(manifest_index)
                    if identity is None:
                        continue
                    parsed = tuple(csv.reader((raw_line,)))
                    if len(parsed) != 1 or len(parsed[0]) != len(header):
                        raise GovernanceError("SCALE-BP v2 selected manifest row drifted.")
                    cells = parsed[0]
                    expected_row_id = f"eval_{canonical_hash({'manifest_sha256': before, 'contract_row_index': identity.manifest_row_index})}"
                    raw_label = cells[positions["label"]]
                    if (
                        identity.sample_id != expected_row_id
                        or cells[positions["split"]] != "test"
                        or cells[positions["center"]] != identity.center
                        or cells[positions["case_id"]] != identity.case_id
                        or raw_label not in {"0", "1"}
                    ):
                        raise GovernanceError(
                            "SCALE-BP v2 cache/manifest label alignment drifted."
                        )
                    decoded[identity.row_ordinal] = int(raw_label)
        except (OSError, csv.Error) as exc:
            raise GovernanceError("SCALE-BP v2 label manifest is unreadable.") from exc
        if before != EXPECTED_TEST_MANIFEST_SHA256 or _sha256_file(self._manifest_path) != before:
            raise GovernanceError("SCALE-BP v2 label manifest changed during decoding.")
        if set(decoded) != {row.row_ordinal for row in selected_rows}:
            raise GovernanceError("SCALE-BP v2 selected label coverage drifted.")
        return MappingProxyType(decoded)


def _identity_payload(row: TestRowIdentity) -> dict[str, object]:
    return {
        "row_ordinal": row.row_ordinal,
        "manifest_row_index": row.manifest_row_index,
        "sample_id": row.sample_id,
        "case_id": row.case_id,
        "center": row.center,
        "patient_slide_group_id": row.patient_slide_group_id,
    }


def _group_labels(
    frame: LabelFreeTestFrame,
    labels: Mapping[int, int],
    *,
    centers: tuple[str, ...],
    selected_rows: tuple[TestRowIdentity, ...],
) -> Mapping[str, Mapping[str, np.ndarray]]:
    selected_ordinals = {row.row_ordinal for row in selected_rows}
    if set(labels) != selected_ordinals:
        raise GovernanceError("SCALE-BP v2 grouped label scope drifted.")
    result: dict[str, Mapping[str, np.ndarray]] = {}
    for center in centers:
        cases: dict[str, list[int]] = {}
        for row in frame.rows_by_center[center]:
            if row.row_ordinal in selected_ordinals:
                cases.setdefault(row.case_id, []).append(labels[row.row_ordinal])
        result[center] = MappingProxyType(
            {
                case: np.ascontiguousarray(values, dtype=np.int8)
                for case, values in cases.items()
            }
        )
    return MappingProxyType(result)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "ManifestLabelDecoder",
    "ScopedCaseLabels",
    "TerminalLabelVector",
)
