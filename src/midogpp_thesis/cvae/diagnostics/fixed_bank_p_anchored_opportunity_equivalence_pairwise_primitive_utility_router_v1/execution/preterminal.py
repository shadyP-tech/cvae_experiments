"""Typed label-free sealing of an already-computed OE-PPUR decision ledger.

This module is deliberately not a scientific callback runner. The neutral
pairwise-primitive-utility core must first produce the typed opportunity,
model, calibration, and selection contracts. This adapter accepts only the
complete ``SelectionDecisionLedger``, binds it to canonical cache rows and
worker-batch evidence, and emits the v1 non-authorized preterminal receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..contracts import (
    PreterminalPhaseReceipt,
    SelectionDecisionLedger,
    _issue_preterminal_phase_receipt,
)
from ..hashing import canonical_hash, require_sha256
from ..identity import ACTION_IDS, EXPECTED_CASE_COUNT, EXPECTED_TEST_ROW_COUNT
from ..protocol import ProtocolError
from .input_lineage import PreterminalInputLineage
from .batch_receipts import ExecutionBatchResult
from .surfaces import (
    validate_candidate_probability_surface_receipt,
)


@dataclass(frozen=True, slots=True)
class LabelFreePreterminalInputs:
    """Exact immutable identities visible before terminal label access."""

    lineage: PreterminalInputLineage
    action_ids: tuple[str, ...] = ACTION_IDS
    input_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.lineage, PreterminalInputLineage):
            raise ProtocolError("OE-PPUR preterminal input lineage is untyped.")
        manifest = self.lineage.manifest
        alignment = self.lineage.rows
        candidate = self.lineage.candidate_surface
        actions = tuple(str(value) for value in self.action_ids)
        if (
            alignment.manifest_receipt != manifest
            or alignment.manifest_receipt.receipt_hash != manifest.receipt_hash
            or manifest.row_count != EXPECTED_TEST_ROW_COUNT
            or manifest.case_count != EXPECTED_CASE_COUNT
            or alignment.row_count != EXPECTED_TEST_ROW_COUNT
            or alignment.case_count != EXPECTED_CASE_COUNT
            or candidate.row_index_sha256 != alignment.row_index_sha256
            or actions != ACTION_IDS
        ):
            raise ProtocolError("OE-PPUR preterminal input surface drifted.")
        object.__setattr__(self, "action_ids", actions)
        object.__setattr__(self, "input_hash", canonical_hash(self._payload()))

    @property
    def config_contract_hash(self) -> str:
        return self.lineage.config_protocol.config_contract_hash

    @property
    def protocol_contract_hash(self) -> str:
        return self.lineage.config_protocol.protocol_contract_hash

    @property
    def source_fence_receipt_hash(self) -> str:
        return self.lineage.config_protocol.source_fence.receipt_hash

    @property
    def fixed_bank_lock_hash(self) -> str:
        return self.lineage.expert_bank.bank_lock_hash

    @property
    def generation_lock_hash(self) -> str:
        return self.lineage.generation_lock.generation_lock_hash

    @property
    def candidate_probability_surface(self):
        return self.lineage.candidate_surface

    @property
    def manifest_receipt(self):
        return self.lineage.manifest

    @property
    def row_alignment_receipt(self):
        return self.lineage.rows

    @property
    def candidate_probability_surface_hash(self) -> str:
        return self.candidate_probability_surface.candidate_probability_surface_sha256

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v1_label_free_preterminal_inputs_v3",
            "preterminal_input_lineage_hash": self.lineage.lineage_hash,
            "config_protocol_receipt_hash": (
                self.lineage.config_protocol.receipt_hash
            ),
            "expert_bank_receipt_hash": self.lineage.expert_bank.receipt_hash,
            "generation_lock_receipt_hash": (
                self.lineage.generation_lock.receipt_hash
            ),
            "config_contract_hash": self.config_contract_hash,
            "protocol_contract_hash": self.protocol_contract_hash,
            "source_fence_receipt_hash": self.source_fence_receipt_hash,
            "fixed_bank_lock_hash": self.fixed_bank_lock_hash,
            "generation_lock_hash": self.generation_lock_hash,
            "candidate_probability_surface_hash": (
                self.candidate_probability_surface_hash
            ),
            "candidate_probability_surface_receipt_hash": (
                self.candidate_probability_surface.receipt_hash
            ),
            "manifest_receipt_hash": self.manifest_receipt.receipt_hash,
            "manifest_content_sha256": (
                self.manifest_receipt.manifest_content_sha256
            ),
            "case_inventory_hash": self.manifest_receipt.case_inventory_hash,
            "row_alignment_receipt_hash": (
                self.row_alignment_receipt.receipt_hash
            ),
            "physical_row_identity_sha256": (
                self.row_alignment_receipt.physical_row_identity_sha256
            ),
            "manifest_row_identity_sha256": (
                self.row_alignment_receipt.manifest_row_identity_sha256
            ),
            "cache_content_sha256": (
                self.row_alignment_receipt.cache_content_sha256
            ),
            "cache_row_order_sha256": (
                self.row_alignment_receipt.cache_row_order_sha256
            ),
            "row_count": EXPECTED_TEST_ROW_COUNT,
            "case_count": EXPECTED_CASE_COUNT,
            "action_ids": self.action_ids,
            "labels_available": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "input_hash": self.input_hash}


@dataclass(frozen=True, slots=True)
class PreterminalExecutionTelemetry:
    """Deterministic batch identities without timing-dependent evidence."""

    gpu_prediction_batch_hash: str
    cpu_outer_batch_hash: str
    gpu_worker_receipt_hashes: tuple[str, ...]
    cpu_worker_receipt_hashes: tuple[str, ...]
    row_index_sha256: str
    candidate_probability_surface_hash: str
    cpu_result_surface_sha256: str
    labels_opened: bool = False
    terminal_phase_executed: bool = False
    filesystem_mutation_count: int = 0
    telemetry_hash: str = field(init=False)

    def __post_init__(self) -> None:
        gpu = require_sha256(self.gpu_prediction_batch_hash, "GPU batch hash")
        cpu = require_sha256(self.cpu_outer_batch_hash, "CPU batch hash")
        gpu_rows = tuple(
            require_sha256(value, "GPU worker receipt hash")
            for value in self.gpu_worker_receipt_hashes
        )
        cpu_rows = tuple(
            require_sha256(value, "CPU worker receipt hash")
            for value in self.cpu_worker_receipt_hashes
        )
        row_index = require_sha256(self.row_index_sha256, "telemetry row-index hash")
        candidate = require_sha256(
            self.candidate_probability_surface_hash,
            "telemetry candidate probability surface hash",
        )
        cpu_results = require_sha256(
            self.cpu_result_surface_sha256,
            "telemetry CPU result surface hash",
        )
        if (
            not gpu_rows
            or not cpu_rows
            or len(set(gpu_rows)) != len(gpu_rows)
            or len(set(cpu_rows)) != len(cpu_rows)
            or bool(self.labels_opened)
            or bool(self.terminal_phase_executed)
            or int(self.filesystem_mutation_count) != 0
        ):
            raise ProtocolError("OE-PPUR preterminal telemetry drifted.")
        object.__setattr__(self, "gpu_prediction_batch_hash", gpu)
        object.__setattr__(self, "cpu_outer_batch_hash", cpu)
        object.__setattr__(self, "gpu_worker_receipt_hashes", gpu_rows)
        object.__setattr__(self, "cpu_worker_receipt_hashes", cpu_rows)
        object.__setattr__(self, "row_index_sha256", row_index)
        object.__setattr__(self, "candidate_probability_surface_hash", candidate)
        object.__setattr__(self, "cpu_result_surface_sha256", cpu_results)
        object.__setattr__(self, "telemetry_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v1_preterminal_execution_telemetry_v2",
            "gpu_prediction_batch_hash": self.gpu_prediction_batch_hash,
            "cpu_outer_batch_hash": self.cpu_outer_batch_hash,
            "gpu_worker_receipt_hashes": self.gpu_worker_receipt_hashes,
            "cpu_worker_receipt_hashes": self.cpu_worker_receipt_hashes,
            "row_index_sha256": self.row_index_sha256,
            "candidate_probability_surface_hash": (
                self.candidate_probability_surface_hash
            ),
            "cpu_result_surface_sha256": self.cpu_result_surface_sha256,
            "labels_opened": self.labels_opened,
            "terminal_phase_executed": self.terminal_phase_executed,
            "filesystem_mutation_count": self.filesystem_mutation_count,
        }


@dataclass(frozen=True, slots=True)
class SealedLabelFreePreterminalResult:
    """Actual typed decision-ledger seal; it grants no label capability."""

    inputs: LabelFreePreterminalInputs
    decision_ledger: SelectionDecisionLedger
    phase_receipt: PreterminalPhaseReceipt
    telemetry: PreterminalExecutionTelemetry
    labels_opened: bool = False
    terminal_capability_opened: bool = False
    filesystem_mutation_count: int = 0
    decision_execution_binding_hash: str = field(init=False)
    sealed_result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.inputs, LabelFreePreterminalInputs)
            or not isinstance(self.decision_ledger, SelectionDecisionLedger)
            or not isinstance(self.phase_receipt, PreterminalPhaseReceipt)
            or self.phase_receipt.decision_ledger != self.decision_ledger
            or self.decision_ledger.manifest_receipt != self.inputs.manifest_receipt
            or self.decision_ledger.annotation_manifest_receipt_hash
            != self.inputs.manifest_receipt.receipt_hash
            or bool(self.labels_opened)
            or bool(self.terminal_capability_opened)
            or int(self.filesystem_mutation_count) != 0
        ):
            raise ProtocolError("OE-PPUR sealed preterminal result drifted.")
        binding = canonical_hash(
            {
                "schema_version": "oe_ppur_v1_decision_execution_binding_v1",
                "selection_decision_ledger_hash": self.decision_ledger.ledger_hash,
                "row_alignment_receipt_hash": (
                    self.inputs.row_alignment_receipt.receipt_hash
                ),
                "row_index_sha256": self.telemetry.row_index_sha256,
                "candidate_probability_surface_hash": (
                    self.telemetry.candidate_probability_surface_hash
                ),
                "candidate_probability_surface_receipt_hash": (
                    self.inputs.candidate_probability_surface.receipt_hash
                ),
                "gpu_prediction_batch_hash": (
                    self.telemetry.gpu_prediction_batch_hash
                ),
                "cpu_outer_batch_hash": self.telemetry.cpu_outer_batch_hash,
                "cpu_result_surface_sha256": (
                    self.telemetry.cpu_result_surface_sha256
                ),
            }
        )
        object.__setattr__(self, "decision_execution_binding_hash", binding)
        object.__setattr__(
            self,
            "sealed_result_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v1_sealed_label_free_preterminal_v2",
                    "label_free_input_hash": self.inputs.input_hash,
                    "selection_decision_ledger_hash": self.decision_ledger.ledger_hash,
                    "preterminal_phase_hash": self.phase_receipt.phase_hash,
                    "telemetry_hash": self.telemetry.telemetry_hash,
                    "decision_execution_binding_hash": binding,
                    "manifest_receipt_hash": self.inputs.manifest_receipt.receipt_hash,
                    "row_alignment_receipt_hash": (
                        self.inputs.row_alignment_receipt.receipt_hash
                    ),
                    "labels_opened": False,
                    "terminal_capability_opened": False,
                    "filesystem_mutation_count": 0,
                }
            ),
        )


def seal_label_free_preterminal_result(
    inputs: LabelFreePreterminalInputs,
    *,
    decision_ledger: SelectionDecisionLedger,
    gpu_prediction_batch: ExecutionBatchResult,
    cpu_outer_batch: ExecutionBatchResult,
) -> SealedLabelFreePreterminalResult:
    """Bind genuine typed science products and worker evidence before labels."""

    if not isinstance(inputs, LabelFreePreterminalInputs):
        raise ProtocolError("OE-PPUR preterminal inputs are untyped.")
    if not isinstance(decision_ledger, SelectionDecisionLedger):
        raise ProtocolError("OE-PPUR preterminal decision ledger is untyped.")
    _validate_batch(gpu_prediction_batch, role="gpu_prediction")
    _validate_batch(cpu_outer_batch, role="cpu_outer")
    alignment = inputs.row_alignment_receipt
    candidate = validate_candidate_probability_surface_receipt(
        inputs.candidate_probability_surface,
        gpu_prediction_batch=gpu_prediction_batch,
        row_alignment_receipt=alignment,
    )
    cpu_input_hashes = cpu_outer_batch.verified_input_file_hashes
    if (
        decision_ledger.manifest_receipt != inputs.manifest_receipt
        or gpu_prediction_batch.row_index_sha256 != alignment.row_index_sha256
        or cpu_outer_batch.row_index_sha256 != alignment.row_index_sha256
        or gpu_prediction_batch.result_surface_sha256
        != candidate.gpu_result_surface_sha256
        or cpu_outer_batch.source_surface_sha256
        != inputs.candidate_probability_surface_hash
        or not cpu_input_hashes
        or any(
            digest not in candidate.output_file_hashes
            for digest in cpu_input_hashes
        )
    ):
        raise ProtocolError(
            "OE-PPUR decision ledger execution/row surface lineage drifted."
        )

    phase = _issue_preterminal_phase_receipt(
        config_contract_hash=inputs.config_contract_hash,
        protocol_contract_hash=inputs.protocol_contract_hash,
        source_fence_receipt_hash=inputs.source_fence_receipt_hash,
        decision_ledger=decision_ledger,
    )
    telemetry = PreterminalExecutionTelemetry(
        gpu_prediction_batch_hash=gpu_prediction_batch.batch_hash,
        cpu_outer_batch_hash=cpu_outer_batch.batch_hash,
        gpu_worker_receipt_hashes=tuple(
            row.receipt_hash for row in gpu_prediction_batch.receipts
        ),
        cpu_worker_receipt_hashes=tuple(
            row.receipt_hash for row in cpu_outer_batch.receipts
        ),
        row_index_sha256=alignment.row_index_sha256,
        candidate_probability_surface_hash=(
            inputs.candidate_probability_surface_hash
        ),
        cpu_result_surface_sha256=cpu_outer_batch.result_surface_sha256,
    )
    return SealedLabelFreePreterminalResult(
        inputs=inputs,
        decision_ledger=decision_ledger,
        phase_receipt=phase,
        telemetry=telemetry,
    )


def _validate_batch(batch: object, *, role: str) -> None:
    if (
        not isinstance(batch, ExecutionBatchResult)
        or batch.role != role
        or not batch.receipts
        or batch.labels_opened
        or batch.filesystem_mutation_count != 0
    ):
        raise ProtocolError(f"OE-PPUR {role} execution batch is invalid.")


__all__ = (
    "LabelFreePreterminalInputs",
    "PreterminalExecutionTelemetry",
    "SealedLabelFreePreterminalResult",
    "seal_label_free_preterminal_result",
)
