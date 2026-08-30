"""Immutable aggregate-attestation DTOs for OE-PPUR v4."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Mapping

from ....protocol import ProtocolError
from ..hashing import canonical_hash, require_sha256


_FINAL_ATTESTATION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class FinalAggregateAttestationReceipt:
    terminal_receipt_hash: str
    terminal_file_sha256: str
    terminal_file_identity_sha256: str
    validator_runtime_sha256: str
    validator_process_pids: tuple[int, int]
    worker_attestation_hashes: tuple[str, str]
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _FINAL_ATTESTATION_TOKEN:
            raise ProtocolError(
                "OE-PPUR v4 final attestation bypassed fresh-process validation."
            )
        for role in (
            "terminal_receipt_hash",
            "terminal_file_sha256",
            "terminal_file_identity_sha256",
            "validator_runtime_sha256",
        ):
            object.__setattr__(
                self, role, require_sha256(getattr(self, role), role.replace("_", " "))
            )
        pids = tuple(int(value) for value in self.validator_process_pids)
        hashes = tuple(
            require_sha256(value, "final worker attestation hash")
            for value in self.worker_attestation_hashes
        )
        if (
            len(pids) != 2
            or len(set(pids)) != 2
            or any(value <= 0 for value in pids)
            or len(hashes) != 2
            or len(set(hashes)) != 2
        ):
            raise ProtocolError("OE-PPUR v4 final aggregate attestation drifted.")
        object.__setattr__(self, "validator_process_pids", pids)
        object.__setattr__(self, "worker_attestation_hashes", hashes)
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                "oe_ppur_v4_final_aggregate_fresh_process_attestation_v1"
            ),
            "terminal_receipt_hash": self.terminal_receipt_hash,
            "terminal_file_sha256": self.terminal_file_sha256,
            "terminal_file_identity_sha256": self.terminal_file_identity_sha256,
            "validator_runtime_sha256": self.validator_runtime_sha256,
            "validator_process_pids": list(self.validator_process_pids),
            "worker_attestation_hashes": list(self.worker_attestation_hashes),
            "fresh_process_count": 2,
            "aggregate_only": True,
            "raw_labels_present": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def _issue_final_aggregate_attestation(
    *,
    terminal_receipt_hash: str,
    terminal_file_sha256: str,
    terminal_file_identity_sha256: str,
    validator_runtime_sha256: str,
    validator_process_pids: tuple[int, int],
    worker_attestation_hashes: tuple[str, str],
    _validator_token: object,
) -> FinalAggregateAttestationReceipt:
    if _validator_token is not _FINAL_ATTESTATION_TOKEN:
        raise ProtocolError("OE-PPUR v4 final attestation issuance bypassed validation.")
    return FinalAggregateAttestationReceipt(
        terminal_receipt_hash=terminal_receipt_hash,
        terminal_file_sha256=terminal_file_sha256,
        terminal_file_identity_sha256=terminal_file_identity_sha256,
        validator_runtime_sha256=validator_runtime_sha256,
        validator_process_pids=validator_process_pids,
        worker_attestation_hashes=worker_attestation_hashes,
        _factory_token=_FINAL_ATTESTATION_TOKEN,
    )


def _reconstruct_final_aggregate_attestation(
    payload: Mapping[str, object],
) -> FinalAggregateAttestationReceipt:
    """Strictly reconstruct one persisted two-process attestation."""

    expected_keys = {
        "schema_version",
        "terminal_receipt_hash",
        "terminal_file_sha256",
        "terminal_file_identity_sha256",
        "validator_runtime_sha256",
        "validator_process_pids",
        "worker_attestation_hashes",
        "fresh_process_count",
        "aggregate_only",
        "raw_labels_present",
        "receipt_hash",
    }
    pids = payload.get("validator_process_pids")
    worker_hashes = payload.get("worker_attestation_hashes")
    if (
        set(payload) != expected_keys
        or payload.get("schema_version")
        != "oe_ppur_v4_final_aggregate_fresh_process_attestation_v1"
        or payload.get("fresh_process_count") != 2
        or payload.get("aggregate_only") is not True
        or payload.get("raw_labels_present") is not False
        or not isinstance(pids, list)
        or not isinstance(worker_hashes, list)
    ):
        raise ProtocolError("OE-PPUR v4 persisted final attestation schema drifted.")
    try:
        receipt = _issue_final_aggregate_attestation(
            terminal_receipt_hash=str(payload["terminal_receipt_hash"]),
            terminal_file_sha256=str(payload["terminal_file_sha256"]),
            terminal_file_identity_sha256=str(
                payload["terminal_file_identity_sha256"]
            ),
            validator_runtime_sha256=str(payload["validator_runtime_sha256"]),
            validator_process_pids=tuple(pids),  # type: ignore[arg-type]
            worker_attestation_hashes=tuple(worker_hashes),  # type: ignore[arg-type]
            _validator_token=_FINAL_ATTESTATION_TOKEN,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError("OE-PPUR v4 persisted final attestation drifted.") from exc
    if receipt.to_payload() != dict(payload):
        raise ProtocolError("OE-PPUR v4 persisted final attestation hash drifted.")
    return receipt


__all__ = (
    "FinalAggregateAttestationReceipt",
    "_FINAL_ATTESTATION_TOKEN",
    "_issue_final_aggregate_attestation",
    "_reconstruct_final_aggregate_attestation",
)
