"""Typed post-admission phase contracts for the OE-PPUR v2 runner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math

from ...protocol import ProtocolError
from .execution.probability_matrix_receipts import ProbabilityMatrixShardSpec
from .hashing import canonical_hash, require_sha256
from .identity import (
    CENTERS,
    EXPECTED_CASE_COUNT,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)


@dataclass(frozen=True, slots=True)
class ServicePreflightReceipt:
    service_module: str
    service_source_sha256: str
    callback_inventory_hash: str
    spawn_probe_hash: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        module = str(self.service_module)
        required_prefix = (
            "midogpp_thesis.cvae.diagnostics."
            "fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_"
            "utility_router_v2"
        )
        if module != required_prefix and not module.startswith(required_prefix + "."):
            raise ProtocolError("OE-PPUR v2 execution service is outside its source seal.")
        for role in (
            "service_source_sha256",
            "callback_inventory_hash",
            "spawn_probe_hash",
        ):
            object.__setattr__(self, role, require_sha256(getattr(self, role), role))
        object.__setattr__(self, "service_module", module)
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v2_service_preflight_receipt_v1",
            "service_module": self.service_module,
            "service_source_sha256": self.service_source_sha256,
            "callback_inventory_hash": self.callback_inventory_hash,
            "spawn_probe_hash": self.spawn_probe_hash,
            "mutation_performed": False,
            "labels_opened": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


@dataclass(frozen=True, slots=True)
class ProbabilityMaterializationReceipt:
    shards: tuple[ProbabilityMatrixShardSpec, ...]
    row_binding_hash: str
    row_index_sha256: str
    row_alignment_receipt_hash: str
    gpu_prediction_batch_hash: str
    gpu_result_surface_sha256: str
    ordered_gpu_worker_result_hashes: tuple[str, ...]
    ordered_gpu_result_file_hashes: tuple[str, ...]
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        shards = tuple(self.shards)
        workers = tuple(
            require_sha256(value, "GPU worker-result hash")
            for value in self.ordered_gpu_worker_result_hashes
        )
        files = tuple(
            require_sha256(value, "GPU result-file hash")
            for value in self.ordered_gpu_result_file_hashes
        )
        if (
            not shards
            or any(not isinstance(row, ProbabilityMatrixShardSpec) for row in shards)
            or tuple(row.content_sha256 for row in shards) != files
            or any(
                row.row_binding_hash != self.row_binding_hash for row in shards
            )
            or len(files) != len(shards)
            or not workers
            or len(set(workers)) != len(workers)
        ):
            raise ProtocolError("OE-PPUR v2 materialization receipt drifted.")
        for role in (
            "row_binding_hash",
            "row_index_sha256",
            "row_alignment_receipt_hash",
            "gpu_prediction_batch_hash",
            "gpu_result_surface_sha256",
        ):
            object.__setattr__(self, role, require_sha256(getattr(self, role), role))
        object.__setattr__(self, "shards", shards)
        object.__setattr__(self, "ordered_gpu_worker_result_hashes", workers)
        object.__setattr__(self, "ordered_gpu_result_file_hashes", files)
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v2_probability_materialization_receipt_v1",
            "ordered_shards": [
                {
                    "path": row.path,
                    "content_sha256": row.content_sha256,
                    "row_interval": [row.row_start, row.row_stop],
                    "gpu_worker_result_sha256": row.gpu_worker_result_sha256,
                }
                for row in self.shards
            ],
            "row_binding_hash": self.row_binding_hash,
            "row_index_sha256": self.row_index_sha256,
            "row_alignment_receipt_hash": self.row_alignment_receipt_hash,
            "gpu_prediction_batch_hash": self.gpu_prediction_batch_hash,
            "gpu_result_surface_sha256": self.gpu_result_surface_sha256,
            "ordered_gpu_worker_result_hashes": list(
                self.ordered_gpu_worker_result_hashes
            ),
            "ordered_gpu_result_file_hashes": list(
                self.ordered_gpu_result_file_hashes
            ),
            "labels_opened": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


@dataclass(frozen=True, slots=True)
class OuterFoldExecutionReceipt:
    parsed_probability_matrix_receipt_hash: str
    outer_center_ids: tuple[str, ...]
    ordered_outer_result_hashes: tuple[str, ...]
    decision_source_hash: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        centers = tuple(str(value) for value in self.outer_center_ids)
        results = tuple(
            require_sha256(value, "outer result hash")
            for value in self.ordered_outer_result_hashes
        )
        if centers != CENTERS or len(results) != len(CENTERS):
            raise ProtocolError("OE-PPUR v2 outer-fold inventory drifted.")
        object.__setattr__(self, "outer_center_ids", centers)
        object.__setattr__(self, "ordered_outer_result_hashes", results)
        for role in (
            "parsed_probability_matrix_receipt_hash",
            "decision_source_hash",
        ):
            object.__setattr__(self, role, require_sha256(getattr(self, role), role))
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v2_outer_fold_execution_receipt_v1",
            "parsed_probability_matrix_receipt_hash": (
                self.parsed_probability_matrix_receipt_hash
            ),
            "outer_center_ids": list(self.outer_center_ids),
            "ordered_outer_result_hashes": list(self.ordered_outer_result_hashes),
            "decision_source_hash": self.decision_source_hash,
            "target_H_excluded_from_every_fit": True,
            "target_support_labels_used": False,
            "labels_opened": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


@dataclass(frozen=True, slots=True)
class AggregateOnlyTerminalReceipt:
    preterminal_attestation_hash: str
    preterminal_ledger_receipt_hash: str
    metric_names: tuple[str, ...]
    protected_metrics: tuple[float, ...]
    routed_metrics: tuple[float, ...]
    evaluated_case_count: int
    routed_case_count: int
    center_aggregate_hash: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.metric_names)
        protected = _finite_values(self.protected_metrics)
        routed = _finite_values(self.routed_metrics)
        if (
            names != ("bacc", "brier", "log")
            or len(protected) != len(names)
            or len(routed) != len(names)
            or type(self.evaluated_case_count) is not int
            or self.evaluated_case_count != EXPECTED_CASE_COUNT
            or type(self.routed_case_count) is not int
            or not 0 <= self.routed_case_count <= self.evaluated_case_count
        ):
            raise ProtocolError("OE-PPUR v2 terminal aggregate receipt drifted.")
        object.__setattr__(self, "metric_names", names)
        object.__setattr__(self, "protected_metrics", protected)
        object.__setattr__(self, "routed_metrics", routed)
        for role in (
            "preterminal_attestation_hash",
            "preterminal_ledger_receipt_hash",
            "center_aggregate_hash",
        ):
            object.__setattr__(self, role, require_sha256(getattr(self, role), role))
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v2_aggregate_only_terminal_receipt_v2",
            "preterminal_attestation_hash": self.preterminal_attestation_hash,
            "preterminal_ledger_receipt_hash": (
                self.preterminal_ledger_receipt_hash
            ),
            "metric_names": list(self.metric_names),
            "protected_metrics": list(self.protected_metrics),
            "routed_metrics": list(self.routed_metrics),
            "metric_deltas": [
                routed - protected
                for protected, routed in zip(
                    self.protected_metrics, self.routed_metrics, strict=True
                )
            ],
            "evaluated_case_count": self.evaluated_case_count,
            "routed_case_count": self.routed_case_count,
            "case_count": self.evaluated_case_count,
            "center_aggregate_hash": self.center_aggregate_hash,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "raw_labels_persisted": False,
            "per_row_labels_persisted": False,
            "per_case_labels_persisted": False,
            "fresh_evidence": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def assert_aggregate_only_payload(value: Mapping[str, object]) -> None:
    """Reject label-bearing or row/case-level terminal persistence by key."""

    forbidden_fragments = (
        "raw_label",
        "labels_by_",
        "row_label",
        "case_label",
        "target_label",
        "y_true",
        "ground_truth",
    )

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                folded = str(key).casefold()
                if any(fragment in folded for fragment in forbidden_fragments):
                    if nested not in (False, None, 0, (), [], {}):
                        raise ProtocolError(
                            "OE-PPUR v2 terminal payload contains label-bearing state."
                        )
                visit(nested)
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for nested in item:
                visit(nested)

    visit(value)


def _finite_values(values: Sequence[float]) -> tuple[float, ...]:
    rows = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in rows):
        raise ProtocolError("OE-PPUR v2 terminal metric is nonfinite.")
    return rows


__all__ = (
    "AggregateOnlyTerminalReceipt",
    "OuterFoldExecutionReceipt",
    "ProbabilityMaterializationReceipt",
    "ServicePreflightReceipt",
    "assert_aggregate_only_payload",
)
