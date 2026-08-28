from __future__ import annotations

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.lifecycle_lineage import (
    CompleteLifecycleEvidenceReceipt,
    parse_complete_phase_evidence,
    validate_complete_lifecycle_evidence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.run_state import (
    PHASE_ORDER,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _hash(index: int) -> str:
    return f"{index:x}" * 64


def _transitions() -> list[dict[str, object]]:
    evidence = (
        _hash(1), _hash(2), _hash(3), _hash(4),
        _hash(5), _hash(6), _hash(7), _hash(7),
    )
    return [
        {"to_phase": phase, "evidence_hash": digest}
        for phase, digest in zip(PHASE_ORDER[1:], evidence, strict=True)
    ]


def test_complete_lifecycle_evidence_is_exact_and_shared() -> None:
    rows = _transitions()
    parsed = parse_complete_phase_evidence(rows)
    receipt = validate_complete_lifecycle_evidence(
        rows,
        inputs_sealed_hash=_hash(1),
        prediction_seal_hash=_hash(2),
        preterminal_result_hash=_hash(3),
        preterminal_boundary_hash=_hash(4),
        terminal_receipt_hash=_hash(5),
        final_attestation_hash=_hash(6),
        final_bundle_receipt_hash=_hash(7),
    )

    assert receipt.phase_evidence == parsed
    assert tuple(receipt.by_phase()) == PHASE_ORDER[1:]
    assert receipt.by_phase()["COMPLETION_PENDING"] == receipt.by_phase()["COMPLETE"]


def test_complete_lifecycle_evidence_rejects_order_and_hash_drift() -> None:
    rows = _transitions()
    rows[2], rows[3] = rows[3], rows[2]
    with pytest.raises(ProtocolError, match="phase inventory"):
        parse_complete_phase_evidence(rows)

    rows = _transitions()
    with pytest.raises(ProtocolError, match="lifecycle evidence drifted"):
        validate_complete_lifecycle_evidence(
            rows,
            inputs_sealed_hash=_hash(1),
            prediction_seal_hash=_hash(2),
            preterminal_result_hash=_hash(8),
            preterminal_boundary_hash=_hash(4),
            terminal_receipt_hash=_hash(5),
            final_attestation_hash=_hash(6),
            final_bundle_receipt_hash=_hash(7),
        )


def test_complete_lifecycle_receipt_cannot_be_caller_minted() -> None:
    with pytest.raises(ProtocolError, match="bypassed complete validation"):
        CompleteLifecycleEvidenceReceipt(
            phase_evidence=tuple(
                (str(row["to_phase"]), str(row["evidence_hash"]))
                for row in _transitions()
            )
        )
