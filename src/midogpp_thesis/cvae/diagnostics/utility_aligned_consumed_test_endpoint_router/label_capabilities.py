"""Role-scoped label capabilities for the consumed-test state machine."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .artifact_io import sha256_file
from .contracts import CENTERS
from .input_contracts import LabelFreeTestFrame, row_identity_hash
from .partitions import ConsumedTestPartitionSurface, LabelFreeCaseRow
from .seals import DevelopmentPredictionCapability, TargetPredictionCapability


@dataclass(frozen=True, order=True)
class BinaryEvaluationLabel:
    center: str
    case_id: str
    sample_id: str
    manifest_row_index: int
    label: int
    label_scope: str

    def __post_init__(self) -> None:
        if self.center not in CENTERS or self.label not in {0, 1}:
            raise ProtocolError("Endpoint-router binary evaluation label drifted.")


@dataclass(frozen=True)
class ScopedDevelopmentLabels:
    outer_target: str
    labels_by_query: Mapping[str, tuple[BinaryEvaluationLabel, ...]]
    development_prediction_seal_hash: str
    row_identity_hashes_by_query: Mapping[str, str]
    capability_hash: str

    def __post_init__(self) -> None:
        normalized = {
            str(query): tuple(rows) for query, rows in self.labels_by_query.items()
        }
        row_hashes = {
            str(query): str(value)
            for query, value in self.row_identity_hashes_by_query.items()
        }
        expected_queries = tuple(
            center for center in CENTERS if center != self.outer_target
        )
        unhashed = {
            "schema_version": "midogpp_endpoint_router_scoped_development_labels_v1",
            "outer_target": self.outer_target,
            "query_centers": list(expected_queries),
            "row_identity_hashes_by_query": row_hashes,
            "development_prediction_seal_hash": self.development_prediction_seal_hash,
            "same_outer_H_labels_opened": False,
            "support_labels_opened": False,
        }
        if (
            self.outer_target not in CENTERS
            or tuple(normalized) != expected_queries
            or tuple(row_hashes) != expected_queries
            or any(row.center != query for query, rows in normalized.items() for row in rows)
            or any(row.label_scope != "cross_center_development_q" for rows in normalized.values() for row in rows)
            or any(len(rows) == 0 for rows in normalized.values())
            or any(len({row.sample_id for row in rows}) != len(rows) for rows in normalized.values())
            or any(
                row_hashes[query]
                != canonical_sha256([row.sample_id for row in rows])
                for query, rows in normalized.items()
            )
            or not _hash_like(self.development_prediction_seal_hash)
            or self.capability_hash != canonical_sha256(unhashed)
        ):
            raise ProtocolError("Endpoint-router development label scope escaped H/q.")
        object.__setattr__(self, "labels_by_query", MappingProxyType(normalized))
        object.__setattr__(self, "row_identity_hashes_by_query", MappingProxyType(row_hashes))

    def labels_for(self, query_center: object) -> np.ndarray:
        query = str(query_center)
        if query == self.outer_target or query not in self.labels_by_query:
            raise ProtocolError("Same-H labels are unavailable to plan H.")
        values = np.asarray(
            [row.label for row in self.labels_by_query[query]], dtype=np.int64
        )
        values.setflags(write=False)
        return values


@dataclass(frozen=True)
class TerminalEvaluationLabels:
    labels_by_center: Mapping[str, tuple[BinaryEvaluationLabel, ...]]
    target_prediction_seal_hash: str
    global_prelabel_seal_hash: str
    row_identity_hashes_by_center: Mapping[str, str]
    capability_hash: str

    def __post_init__(self) -> None:
        normalized = {
            str(center): tuple(rows) for center, rows in self.labels_by_center.items()
        }
        row_hashes = {
            str(center): str(value)
            for center, value in self.row_identity_hashes_by_center.items()
        }
        unhashed = {
            "schema_version": "midogpp_endpoint_router_terminal_label_capability_v1",
            "target_prediction_seal_hash": self.target_prediction_seal_hash,
            "global_prelabel_seal_hash": self.global_prelabel_seal_hash,
            "evaluation_row_counts_by_center": {
                center: len(rows) for center, rows in normalized.items()
            },
            "row_identity_hashes_by_center": row_hashes,
            "support_labels_opened": False,
            "raw_labels_persisted": False,
        }
        if (
            tuple(normalized) != CENTERS
            or tuple(row_hashes) != CENTERS
            or any(row.center != center for center, rows in normalized.items() for row in rows)
            or any(row.label_scope != "terminal_same_outer_H_evaluation" for rows in normalized.values() for row in rows)
            or any(len(rows) == 0 for rows in normalized.values())
            or any(len({row.sample_id for row in rows}) != len(rows) for rows in normalized.values())
            or any(
                row_hashes[center]
                != canonical_sha256([row.sample_id for row in rows])
                for center, rows in normalized.items()
            )
            or not _hash_like(self.target_prediction_seal_hash)
            or not _hash_like(self.global_prelabel_seal_hash)
            or self.capability_hash != canonical_sha256(unhashed)
        ):
            raise ProtocolError("Endpoint-router terminal label scope drifted.")
        object.__setattr__(self, "labels_by_center", MappingProxyType(normalized))

    def labels_for(self, center: object) -> np.ndarray:
        rows = self.labels_by_center.get(str(center))
        if rows is None:
            raise ProtocolError("Endpoint-router terminal center is invalid.")
        values = np.asarray([row.label for row in rows], dtype=np.int64)
        values.setflags(write=False)
        return values


def admit_manifest_without_labels(
    manifest_path: Path, *, expected_sha256: str
) -> Mapping[str, object]:
    observed = sha256_file(manifest_path)
    if observed != expected_sha256:
        raise ProtocolError("Endpoint-router manifest admission hash drifted.")
    unhashed = {
        "schema_version": "midogpp_endpoint_router_manifest_admission_v1",
        "status": "PASS",
        "manifest_sha256": observed,
        "manifest_parsed": False,
        "labels_opened": False,
        "domain_mapping_may_now_be_parsed": True,
    }
    return MappingProxyType(
        {**unhashed, "manifest_admission_hash": canonical_sha256(unhashed)}
    )


class EndpointRouterLabelCapabilityManager:
    """The only manifest CSV reader; every opening follows a durable seal."""

    def __init__(
        self,
        manifest_path: Path,
        frame: LabelFreeTestFrame,
        partitions: ConsumedTestPartitionSurface,
        *,
        expected_manifest_sha256: str,
        development_capability: DevelopmentPredictionCapability,
    ) -> None:
        manifest_sha = sha256_file(manifest_path)
        if (
            manifest_sha != expected_manifest_sha256
            or development_capability.store.partition_lock_hash != partitions.lock_hash
            or development_capability.store.cache_binding_hash != frame.cache_binding_hash
        ):
            raise ProtocolError("Endpoint-router label capability lineage drifted.")
        self._manifest_path = Path(manifest_path)
        self._manifest_sha256 = manifest_sha
        self._frame = frame
        self._partitions = partitions
        self._development = development_capability
        self._development_opened: set[str] = set()
        self._plan_hashes: dict[str, str] = {}
        self._target_capability: TargetPredictionCapability | None = None
        self._global_prelabel_seal_hash: str | None = None
        self._terminal_opened = False
        self._events: list[dict[str, object]] = []

    def open_development_labels(self, outer_target: object) -> ScopedDevelopmentLabels:
        outer = str(outer_target)
        if (
            outer not in CENTERS
            or outer in self._development_opened
            or self._plan_hashes
            or self._target_capability is not None
            or self._terminal_opened
        ):
            raise ProtocolError("Endpoint-router development labels opened out of order.")
        by_query: dict[str, tuple[BinaryEvaluationLabel, ...]] = {}
        for query in CENTERS:
            if query == outer:
                continue
            by_query[query] = self._open_rows(
                self._partitions.evaluation_rows_by_center[query],
                label_scope="cross_center_development_q",
                event_role=f"development_H{outer}_q{query}",
            )
        if outer in by_query or any(row.center == outer for rows in by_query.values() for row in rows):
            raise ProtocolError("Same-H labels entered the development capability.")
        unhashed = {
            "schema_version": "midogpp_endpoint_router_scoped_development_labels_v1",
            "outer_target": outer,
            "query_centers": list(by_query),
            "row_identity_hashes_by_query": {
                query: canonical_sha256([row.sample_id for row in rows])
                for query, rows in by_query.items()
            },
            "development_prediction_seal_hash": self._development.seal_hash,
            "same_outer_H_labels_opened": False,
            "support_labels_opened": False,
        }
        self._development_opened.add(outer)
        return ScopedDevelopmentLabels(
            outer_target=outer,
            labels_by_query=by_query,
            development_prediction_seal_hash=self._development.seal_hash,
            row_identity_hashes_by_query=unhashed["row_identity_hashes_by_query"],
            capability_hash=canonical_sha256(unhashed),
        )

    def record_target_policy_plan(self, target: object, plan_hash: str) -> None:
        center = str(target)
        if (
            set(self._development_opened) != set(CENTERS)
            or center not in CENTERS
            or center in self._plan_hashes
            or self._target_capability is not None
            or not _hash_like(plan_hash)
        ):
            raise ProtocolError("Endpoint-router target plan recorded out of order.")
        self._plan_hashes[center] = str(plan_hash)

    def record_global_target_seal(
        self,
        capability: TargetPredictionCapability,
        *,
        global_prelabel_seal_hash: str,
    ) -> None:
        if (
            tuple(self._plan_hashes) != CENTERS
            or self._target_capability is not None
            or not _hash_like(global_prelabel_seal_hash)
            or capability.seal_payload.get("target_policy_plan_hashes_by_center")
            != self._plan_hashes
            or capability.seal_payload.get("global_prelabel_seal_hash")
            != global_prelabel_seal_hash
        ):
            raise ProtocolError("Endpoint-router global target seal is incomplete.")
        self._target_capability = capability
        self._global_prelabel_seal_hash = str(global_prelabel_seal_hash)

    def open_terminal_evaluation_labels(self) -> TerminalEvaluationLabels:
        if (
            self._target_capability is None
            or self._global_prelabel_seal_hash is None
            or self._terminal_opened
            or tuple(self._plan_hashes) != CENTERS
        ):
            raise ProtocolError("Terminal labels require all nine sealed target plans.")
        by_center = {
            center: self._open_rows(
                self._partitions.evaluation_rows_by_center[center],
                label_scope="terminal_same_outer_H_evaluation",
                event_role=f"terminal_H{center}",
            )
            for center in CENTERS
        }
        for center, rows in by_center.items():
            if {row.label for row in rows} != {0, 1}:
                raise ProtocolError(f"Terminal center {center} lacks both classes.")
        self._terminal_opened = True
        unhashed = {
            "schema_version": "midogpp_endpoint_router_terminal_label_capability_v1",
            "target_prediction_seal_hash": self._target_capability.seal_hash,
            "global_prelabel_seal_hash": self._global_prelabel_seal_hash,
            "evaluation_row_counts_by_center": {
                center: len(rows) for center, rows in by_center.items()
            },
            "row_identity_hashes_by_center": {
                center: canonical_sha256([row.sample_id for row in rows])
                for center, rows in by_center.items()
            },
            "support_labels_opened": False,
            "raw_labels_persisted": False,
        }
        return TerminalEvaluationLabels(
            labels_by_center=by_center,
            target_prediction_seal_hash=self._target_capability.seal_hash,
            global_prelabel_seal_hash=self._global_prelabel_seal_hash,
            row_identity_hashes_by_center=unhashed["row_identity_hashes_by_center"],
            capability_hash=canonical_sha256(unhashed),
        )

    def access_report(self) -> Mapping[str, object]:
        payload = {
            "schema_version": "midogpp_endpoint_router_label_capability_report_v1",
            "status": "PASS" if self._terminal_opened else "INCOMPLETE",
            "manifest_sha256": self._manifest_sha256,
            "development_prediction_seal_hash": self._development.seal_hash,
            "development_outer_H_capabilities_opened": sorted(self._development_opened),
            "target_policy_plan_hashes_by_center": dict(self._plan_hashes),
            "target_prediction_seal_hash": (
                None if self._target_capability is None else self._target_capability.seal_hash
            ),
            "global_prelabel_seal_hash": self._global_prelabel_seal_hash,
            "terminal_evaluation_labels_opened": self._terminal_opened,
            "same_outer_H_evaluation_labels_used_for_plan_H": False,
            "cross_center_evaluation_labels_used_as_development_q_labels": True,
            "support_labels_opened": False,
            "raw_labels_persisted": False,
            "events": list(self._events),
        }
        return MappingProxyType({**payload, "report_hash": canonical_sha256(payload)})

    def _open_rows(
        self,
        rows: Sequence[LabelFreeCaseRow],
        *,
        label_scope: str,
        event_role: str,
    ) -> tuple[BinaryEvaluationLabel, ...]:
        requested = {row.manifest_row_index: row for row in rows}
        labels: list[BinaryEvaluationLabel] = []
        with self._manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {"case_id", "center", "split", "label"}.issubset(reader.fieldnames):
                raise ProtocolError("Endpoint-router manifest fields drifted.")
            for index, raw in enumerate(reader):
                wanted = requested.get(index)
                if wanted is None:
                    continue
                if (
                    _evaluation_row_id(self._manifest_sha256, index)
                    != wanted.evaluation_row_id
                    or str(raw["case_id"]) != wanted.case_id
                    or str(raw["center"]) != wanted.center
                    or str(raw["split"]) != "test"
                ):
                    raise ProtocolError("Endpoint-router manifest identity drifted.")
                labels.append(
                    BinaryEvaluationLabel(
                        center=wanted.center,
                        case_id=wanted.case_id,
                        sample_id=wanted.evaluation_row_id,
                        manifest_row_index=wanted.manifest_row_index,
                        label=_binary(raw["label"]),
                        label_scope=label_scope,
                    )
                )
        ordered = tuple(sorted(labels, key=lambda row: row.manifest_row_index))
        if (
            sha256_file(self._manifest_path) != self._manifest_sha256
            or len(ordered) != len(requested)
            or tuple(row.sample_id for row in ordered)
            != tuple(row.evaluation_row_id for row in rows)
        ):
            raise ProtocolError("Endpoint-router scoped label coverage drifted.")
        event_unhashed = {
            "role": event_role,
            "row_count": len(ordered),
            "case_count": len({row.case_id for row in ordered}),
            "row_identity_hash": row_identity_hash(rows),
            "label_identity_hash": canonical_sha256(
                [[row.sample_id, row.label] for row in ordered]
            ),
            "raw_labels_persisted": False,
        }
        self._events.append({**event_unhashed, "event_hash": canonical_sha256(event_unhashed)})
        return ordered


def _evaluation_row_id(manifest_sha256: str, index: int) -> str:
    return f"eval_{canonical_sha256({'manifest_sha256': manifest_sha256, 'contract_row_index': index})}"


def _binary(value: object) -> int:
    text = str(value).strip()
    if text not in {"0", "1"}:
        raise ProtocolError("Endpoint-router manifest label must be binary.")
    return int(text)


def _hash_like(value: object) -> bool:
    text = str(value)
    return len(text) in {16, 64} and all(char in "0123456789abcdef" for char in text)


__all__ = (
    "BinaryEvaluationLabel",
    "EndpointRouterLabelCapabilityManager",
    "ScopedDevelopmentLabels",
    "TerminalEvaluationLabels",
    "admit_manifest_without_labels",
)
