"""Typed GPU-to-CPU probability-surface lineage for planned OE-PPUR.

The receipt in this module is deliberately label-free.  It proves that the
surface identifier handed to CPU outer-fold jobs is derived from the exact
coordinator-admitted GPU result bytes and, on the production path, from the
canonical 9,928-row alignment.  It grants no process, filesystem, or terminal
label capability.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import os

from ..hashing import canonical_hash, require_sha256
from ..identity import EXPECTED_TEST_ROW_COUNT
from ..protocol import ProtocolError
from .memmap import (
    CanonicalRowAlignmentReceipt,
    validate_canonical_row_alignment_receipt,
)
from .batch_receipts import ExecutionBatchResult


_SURFACE_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class CandidateProbabilitySurfaceReceipt:
    """Immutable identity for exact GPU-produced candidate probabilities."""

    gpu_prediction_batch_hash: str
    gpu_result_surface_sha256: str
    row_index_sha256: str
    row_alignment_receipt_hash: str
    output_file_hashes: tuple[str, ...]
    worker_result_hashes: tuple[str, ...]
    canonical_alignment_bound: bool
    _factory_token: InitVar[object] = None
    candidate_probability_surface_sha256: str = field(init=False)
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _SURFACE_FACTORY_TOKEN:
            raise ProtocolError(
                "OE-PPUR candidate probability surface bypassed its guarded factory."
            )
        batch = require_sha256(
            self.gpu_prediction_batch_hash,
            "candidate-surface GPU batch hash",
        )
        result_surface = require_sha256(
            self.gpu_result_surface_sha256,
            "candidate-surface GPU result surface",
        )
        row_index = require_sha256(
            self.row_index_sha256,
            "candidate-surface row index",
        )
        alignment = require_sha256(
            self.row_alignment_receipt_hash,
            "candidate-surface row-alignment receipt",
        )
        outputs = tuple(
            require_sha256(value, "candidate-surface output file hash")
            for value in self.output_file_hashes
        )
        worker_results = tuple(
            require_sha256(value, "candidate-surface worker result hash")
            for value in self.worker_result_hashes
        )
        if (
            not outputs
            or not worker_results
            or len(worker_results) < 2
            or type(self.canonical_alignment_bound) is not bool
        ):
            raise ProtocolError("OE-PPUR candidate probability surface drifted.")
        surface_body = {
            "schema_version": "oe_ppur_v1_candidate_probability_surface_v1",
            "gpu_prediction_batch_hash": batch,
            "gpu_result_surface_sha256": result_surface,
            "row_index_sha256": row_index,
            "ordered_output_file_hashes": outputs,
            "ordered_worker_result_hashes": worker_results,
            "labels_present": False,
        }
        surface = canonical_hash(surface_body)
        receipt_body = {
            "schema_version": "oe_ppur_v1_candidate_probability_surface_receipt_v1",
            "candidate_probability_surface_sha256": surface,
            "row_alignment_receipt_hash": alignment,
            "canonical_alignment_bound": self.canonical_alignment_bound,
            **surface_body,
        }
        object.__setattr__(self, "gpu_prediction_batch_hash", batch)
        object.__setattr__(self, "gpu_result_surface_sha256", result_surface)
        object.__setattr__(self, "row_index_sha256", row_index)
        object.__setattr__(self, "row_alignment_receipt_hash", alignment)
        object.__setattr__(self, "output_file_hashes", outputs)
        object.__setattr__(self, "worker_result_hashes", worker_results)
        object.__setattr__(
            self,
            "candidate_probability_surface_sha256",
            surface,
        )
        object.__setattr__(self, "receipt_hash", canonical_hash(receipt_body))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v1_candidate_probability_surface_receipt_v1",
            "gpu_prediction_batch_hash": self.gpu_prediction_batch_hash,
            "gpu_result_surface_sha256": self.gpu_result_surface_sha256,
            "row_index_sha256": self.row_index_sha256,
            "row_alignment_receipt_hash": self.row_alignment_receipt_hash,
            "ordered_output_file_hashes": self.output_file_hashes,
            "ordered_worker_result_hashes": self.worker_result_hashes,
            "candidate_probability_surface_sha256": (
                self.candidate_probability_surface_sha256
            ),
            "canonical_alignment_bound": self.canonical_alignment_bound,
            "labels_present": False,
            "receipt_hash": self.receipt_hash,
        }


def build_candidate_probability_surface_receipt(
    gpu_prediction_batch: ExecutionBatchResult,
    *,
    row_alignment_receipt: CanonicalRowAlignmentReceipt,
) -> CandidateProbabilitySurfaceReceipt:
    """Bind admitted GPU result bytes to the canonical physical row order."""

    alignment = validate_canonical_row_alignment_receipt(
        row_alignment_receipt
    )
    _validate_gpu_batch(gpu_prediction_batch)
    if (
        gpu_prediction_batch.row_index_sha256 != alignment.row_index_sha256
        or any(
            row.row_count != EXPECTED_TEST_ROW_COUNT
            for row in gpu_prediction_batch.receipts
        )
    ):
        raise ProtocolError(
            "OE-PPUR GPU probability surface row alignment drifted."
        )
    return _issue_surface_receipt(
        gpu_prediction_batch,
        row_alignment_receipt_hash=alignment.receipt_hash,
        canonical_alignment_bound=True,
    )


def validate_candidate_probability_surface_receipt(
    receipt: object,
    *,
    gpu_prediction_batch: ExecutionBatchResult | None = None,
    row_alignment_receipt: CanonicalRowAlignmentReceipt | None = None,
) -> CandidateProbabilitySurfaceReceipt:
    """Recompute structural hashes and optionally exact-match both parents."""

    if not isinstance(receipt, CandidateProbabilitySurfaceReceipt):
        raise ProtocolError("OE-PPUR candidate probability surface is untyped.")
    rebuilt = CandidateProbabilitySurfaceReceipt(
        gpu_prediction_batch_hash=receipt.gpu_prediction_batch_hash,
        gpu_result_surface_sha256=receipt.gpu_result_surface_sha256,
        row_index_sha256=receipt.row_index_sha256,
        row_alignment_receipt_hash=receipt.row_alignment_receipt_hash,
        output_file_hashes=receipt.output_file_hashes,
        worker_result_hashes=receipt.worker_result_hashes,
        canonical_alignment_bound=receipt.canonical_alignment_bound,
        _factory_token=_SURFACE_FACTORY_TOKEN,
    )
    if rebuilt != receipt:
        raise ProtocolError("OE-PPUR candidate probability surface hash drifted.")
    if gpu_prediction_batch is not None:
        _validate_gpu_batch(gpu_prediction_batch)
        expected = _issue_surface_receipt(
            gpu_prediction_batch,
            row_alignment_receipt_hash=receipt.row_alignment_receipt_hash,
            canonical_alignment_bound=receipt.canonical_alignment_bound,
        )
        if expected != receipt:
            raise ProtocolError(
                "OE-PPUR candidate surface differs from its GPU batch."
            )
    if row_alignment_receipt is not None:
        alignment = validate_canonical_row_alignment_receipt(
            row_alignment_receipt
        )
        if (
            not receipt.canonical_alignment_bound
            or receipt.row_alignment_receipt_hash != alignment.receipt_hash
            or receipt.row_index_sha256 != alignment.row_index_sha256
        ):
            raise ProtocolError(
                "OE-PPUR candidate surface differs from canonical rows."
            )
    return receipt


def _build_strict_test_candidate_probability_surface_receipt(
    gpu_prediction_batch: ExecutionBatchResult,
) -> CandidateProbabilitySurfaceReceipt:
    """Issue a noncanonical surface solely for isolated spawn transport tests."""

    _validate_gpu_batch(gpu_prediction_batch)
    if (
        "PYTEST_CURRENT_TEST" not in os.environ
        or any(
            row.result_evidence_mode != "strict_test_fixture"
            for row in gpu_prediction_batch.receipts
        )
    ):
        raise ProtocolError(
            "OE-PPUR synthetic candidate surfaces are pytest-only."
        )
    return _issue_surface_receipt(
        gpu_prediction_batch,
        row_alignment_receipt_hash=canonical_hash(
            {
                "schema_version": "oe_ppur_v1_strict_test_row_alignment_v1",
                "row_index_sha256": gpu_prediction_batch.row_index_sha256,
            }
        ),
        canonical_alignment_bound=False,
    )


def _issue_surface_receipt(
    batch: ExecutionBatchResult,
    *,
    row_alignment_receipt_hash: str,
    canonical_alignment_bound: bool,
) -> CandidateProbabilitySurfaceReceipt:
    return CandidateProbabilitySurfaceReceipt(
        gpu_prediction_batch_hash=batch.batch_hash,
        gpu_result_surface_sha256=batch.result_surface_sha256,
        row_index_sha256=batch.row_index_sha256,
        row_alignment_receipt_hash=row_alignment_receipt_hash,
        output_file_hashes=batch.result_file_hashes,
        worker_result_hashes=tuple(
            row.worker_result_hash for row in batch.receipts
        ),
        canonical_alignment_bound=canonical_alignment_bound,
        _factory_token=_SURFACE_FACTORY_TOKEN,
    )


def _validate_gpu_batch(batch: object) -> None:
    if (
        not isinstance(batch, ExecutionBatchResult)
        or batch.role != "gpu_prediction"
        or not batch.receipts
        or not batch.result_file_hashes
        or batch.labels_opened
        or batch.filesystem_mutation_count != 0
    ):
        raise ProtocolError("OE-PPUR GPU probability batch is invalid.")


__all__ = (
    "CandidateProbabilitySurfaceReceipt",
    "build_candidate_probability_surface_receipt",
    "validate_candidate_probability_surface_receipt",
)
