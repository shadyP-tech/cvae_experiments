"""Monotone COMPLETE-phase evidence validation for OE-PPUR v4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256
from .run_state import PHASE_ORDER


_TOKEN = object()


@dataclass(frozen=True, slots=True)
class CompleteLifecycleEvidenceReceipt:
    phase_evidence: tuple[tuple[str, str], ...]
    _factory_token: InitVar[object | None] = None
    evidence_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _TOKEN:
            raise ProtocolError("OE-PPUR v4 lifecycle evidence bypassed validation.")
        rows = tuple(
            (str(phase), require_sha256(digest, f"{phase} evidence hash"))
            for phase, digest in self.phase_evidence
        )
        if tuple(phase for phase, _digest in rows) != PHASE_ORDER[1:]:
            raise ProtocolError("OE-PPUR v4 complete phase inventory drifted.")
        object.__setattr__(self, "phase_evidence", rows)
        object.__setattr__(self, "evidence_hash", canonical_hash(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_complete_lifecycle_evidence_v1",
            "phase_evidence": [
                {"phase": phase, "evidence_hash": digest}
                for phase, digest in self.phase_evidence
            ],
        }

    def by_phase(self) -> dict[str, str]:
        return dict(self.phase_evidence)


def validate_complete_lifecycle_evidence(
    transitions: object,
    *,
    inputs_sealed_hash: object,
    prediction_seal_hash: object,
    preterminal_result_hash: object,
    preterminal_boundary_hash: object,
    terminal_receipt_hash: object,
    final_attestation_hash: object,
    final_bundle_receipt_hash: object,
) -> CompleteLifecycleEvidenceReceipt:
    observed = parse_complete_phase_evidence(transitions)
    final_bundle = require_sha256(final_bundle_receipt_hash, "final bundle receipt")
    expected = (
        ("INPUTS_SEALED", require_sha256(inputs_sealed_hash, "inputs sealed evidence")),
        (
            "PHYSICAL_PROBABILITIES_MATERIALIZED",
            require_sha256(prediction_seal_hash, "prediction seal evidence"),
        ),
        (
            "PRETERMINAL_DECISIONS_SEALED",
            require_sha256(preterminal_result_hash, "preterminal result evidence"),
        ),
        (
            "PRETERMINAL_ATTESTED",
            require_sha256(preterminal_boundary_hash, "preterminal boundary evidence"),
        ),
        (
            "TERMINAL_AGGREGATES_SCORED",
            require_sha256(terminal_receipt_hash, "terminal receipt evidence"),
        ),
        (
            "FINAL_ATTESTED",
            require_sha256(final_attestation_hash, "final attestation evidence"),
        ),
        ("COMPLETION_PENDING", final_bundle),
        ("COMPLETE", final_bundle),
    )
    if observed != expected or tuple(phase for phase, _digest in observed) != PHASE_ORDER[1:]:
        raise ProtocolError("OE-PPUR v4 complete lifecycle evidence drifted.")
    return CompleteLifecycleEvidenceReceipt(expected, _factory_token=_TOKEN)


def parse_complete_phase_evidence(
    transitions: object,
) -> tuple[tuple[str, str], ...]:
    if (
        not isinstance(transitions, Sequence)
        or isinstance(transitions, (str, bytes, bytearray))
        or not all(isinstance(row, Mapping) for row in transitions)
    ):
        raise ProtocolError("OE-PPUR v4 complete transitions are malformed.")
    observed = tuple(
        (
            str(row.get("to_phase")),
            require_sha256(row.get("evidence_hash"), "phase evidence hash"),
        )
        for row in transitions
    )
    if tuple(phase for phase, _digest in observed) != PHASE_ORDER[1:]:
        raise ProtocolError("OE-PPUR v4 complete phase inventory drifted.")
    return observed


__all__ = (
    "CompleteLifecycleEvidenceReceipt",
    "parse_complete_phase_evidence",
    "validate_complete_lifecycle_evidence",
)
