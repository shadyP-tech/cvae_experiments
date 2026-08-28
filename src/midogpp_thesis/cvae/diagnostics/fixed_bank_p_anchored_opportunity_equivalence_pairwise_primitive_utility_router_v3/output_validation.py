"""Semantic and inventory validation for persisted OE-PPUR v3 outputs.

Only this module can issue a :class:`FinalAggregateBundleReceipt`.  Public
revalidation requires an already-issued receipt, while the output composer
uses the private post-persistence entry point exactly once.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
import hashlib
import os
from pathlib import Path
import stat

from ...protocol import ProtocolError
from .execution.preterminal_artifact import (
    FinalAggregateAttestationReceipt,
    _reconstruct_final_aggregate_attestation,
)
from .hashing import canonical_hash, require_sha256
from .identity import OUTPUT_ARTIFACT_ID, PUBLICATION_STATUS, TERMINAL_DECISION
from .output_persistence import (
    COMPLETE_CATALOG_MEMBERS,
    COMPLETE_INTERNAL_MEMBERS,
    CONTENT_INDEX_MEMBER,
    FINAL_ATTESTATION_MEMBER,
    FINAL_BINDING_MEMBER,
    FINAL_INDEXED_MEMBERS,
    TERMINAL_METRICS_MEMBER,
    TERMINAL_RESULT_MEMBER,
    VALIDATION_INDEX_MEMBER,
    VALIDATION_REPORT_MEMBER,
    _assert_aggregate_only,
    _read_json_object,
    _read_regular_bytes_nofollow,
    _reject_symlink_chain,
    _sha256_file,
    _stat_payload,
)
from .terminal.contracts import (
    AggregateOnlyTerminalReceipt,
    _reconstruct_persisted_aggregate_only_terminal_receipt,
)


_FINAL_BUNDLE_TOKEN = object()
_VALIDATION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class FinalAggregateBundleReceipt:
    """Hash-only receipt returned to the runner before ``COMPLETE``."""

    artifact_root: str
    content_index_hash: str
    validation_report_hash: str
    validation_index_hash: str
    final_attestation_hash: str
    terminal_receipt_hash: str
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _FINAL_BUNDLE_TOKEN:
            raise ProtocolError(
                "OE-PPUR v3 final bundle receipt bypassed durable validation."
            )
        root = Path(self.artifact_root)
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise ProtocolError("OE-PPUR v3 final bundle root is unsafe.")
        for role in (
            "content_index_hash",
            "validation_report_hash",
            "validation_index_hash",
            "final_attestation_hash",
            "terminal_receipt_hash",
        ):
            object.__setattr__(self, role, require_sha256(getattr(self, role), role))
        object.__setattr__(self, "artifact_root", str(root))
        object.__setattr__(self, "receipt_hash", canonical_hash(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_final_aggregate_bundle_receipt_v1",
            "artifact_root": self.artifact_root,
            "content_index_hash": self.content_index_hash,
            "validation_report_hash": self.validation_report_hash,
            "validation_index_hash": self.validation_index_hash,
            "final_attestation_hash": self.final_attestation_hash,
            "terminal_receipt_hash": self.terminal_receipt_hash,
        }


def _issue_final_aggregate_bundle(
    root: str | Path,
) -> FinalAggregateBundleReceipt:
    """Issue authority only after a complete persisted-bundle validation."""

    return _validate_final_aggregate_bundle(
        Path(root),
        expected_receipt=None,
        _validator_token=_VALIDATION_TOKEN,
    )


def validate_final_aggregate_bundle(
    root: str | Path,
    *,
    expected_receipt: FinalAggregateBundleReceipt,
) -> FinalAggregateBundleReceipt:
    """Revalidate an already-issued bundle without minting new authority."""

    if type(expected_receipt) is not FinalAggregateBundleReceipt:
        raise ProtocolError("OE-PPUR v3 final validation requires its typed receipt.")
    return _validate_final_aggregate_bundle(
        Path(root),
        expected_receipt=expected_receipt,
        _validator_token=_VALIDATION_TOKEN,
    )


def _validate_final_aggregate_bundle(
    destination: Path,
    *,
    expected_receipt: FinalAggregateBundleReceipt | None,
    _validator_token: object,
) -> FinalAggregateBundleReceipt:
    if _validator_token is not _VALIDATION_TOKEN:
        raise ProtocolError("OE-PPUR v3 final validation bypassed lifecycle authority.")
    destination = Path(os.path.abspath(destination))
    _reject_symlink_chain(destination)
    if (
        not destination.is_absolute()
        or not destination.is_dir()
        or destination.is_symlink()
        or destination == Path(destination.anchor)
    ):
        raise ProtocolError("OE-PPUR v3 final validation root is absent or unsafe.")
    payloads = {
        member: _read_json_object(destination / member)
        for member in FINAL_INDEXED_MEMBERS
    }
    for payload in payloads.values():
        _assert_aggregate_only(payload)

    terminal = _reconstruct_persisted_aggregate_only_terminal_receipt(
        payloads[TERMINAL_RESULT_MEMBER]
    )
    terminal_metrics_payload = _read_json_object(destination / TERMINAL_METRICS_MEMBER)
    terminal_metrics = _reconstruct_persisted_aggregate_only_terminal_receipt(
        terminal_metrics_payload
    )
    if terminal_metrics.to_payload() != terminal.to_payload():
        raise ProtocolError("OE-PPUR v3 terminal result/persisted metrics drifted.")

    final_attestation = _reconstruct_final_aggregate_attestation(
        payloads[FINAL_ATTESTATION_MEMBER]
    )
    terminal_raw, terminal_stat = _read_regular_bytes_nofollow(
        destination / TERMINAL_METRICS_MEMBER
    )
    terminal_identity_hash = canonical_hash(
        {
            "schema_version": "oe_ppur_v3_terminal_file_identity_v1",
            "stat": _stat_payload(terminal_stat),
        }
    )
    if (
        final_attestation.terminal_receipt_hash != terminal.receipt_hash
        or final_attestation.terminal_file_sha256
        != hashlib.sha256(terminal_raw).hexdigest()
        or final_attestation.terminal_file_identity_sha256
        != terminal_identity_hash
    ):
        raise ProtocolError("OE-PPUR v3 final attestation/terminal bytes drifted.")

    binding = payloads[FINAL_BINDING_MEMBER]
    _validate_final_binding(binding, terminal, final_attestation)
    index = _read_json_object(destination / CONTENT_INDEX_MEMBER)
    index_body = {
        key: value for key, value in index.items() if key != "content_index_hash"
    }
    expected_member_hashes = {
        member: _sha256_file(destination / member) for member in FINAL_INDEXED_MEMBERS
    }
    if (
        set(index)
        != {
            "schema_version",
            "binding_hash",
            "terminal_receipt_hash",
            "final_attestation_hash",
            "members",
            "publication_status",
            "terminal_decision",
            "fresh_evidence",
            "aggregate_only",
            "raw_labels_persisted",
            "v2_input_or_state_used",
            "cross_run_recovery_allowed",
            "content_index_hash",
        }
        or index.get("schema_version")
        != "oe_ppur_v3_final_aggregate_content_index_v1"
        or index.get("content_index_hash") != canonical_hash(index_body)
        or index.get("binding_hash") != canonical_hash(binding)
        or index.get("terminal_receipt_hash") != terminal.receipt_hash
        or index.get("final_attestation_hash") != final_attestation.receipt_hash
        or index.get("members") != expected_member_hashes
        or index.get("publication_status") != PUBLICATION_STATUS
        or index.get("terminal_decision") != TERMINAL_DECISION
        or index.get("fresh_evidence") is not False
        or index.get("aggregate_only") is not True
        or index.get("raw_labels_persisted") is not False
        or index.get("v2_input_or_state_used") is not False
        or index.get("cross_run_recovery_allowed") is not False
    ):
        raise ProtocolError("OE-PPUR v3 final content index drifted.")

    validation = _read_json_object(destination / VALIDATION_REPORT_MEMBER)
    validation_body = {
        key: value
        for key, value in validation.items()
        if key != "validation_report_hash"
    }
    if (
        validation.get("schema_version")
        != "oe_ppur_v3_final_aggregate_validation_report_v1"
        or validation.get("status") != "PASS"
        or validation.get("validation_report_hash") != canonical_hash(validation_body)
        or validation.get("content_index_hash") != index["content_index_hash"]
        or validation.get("final_attestation_hash")
        != final_attestation.receipt_hash
        or validation.get("terminal_receipt_hash") != terminal.receipt_hash
        or validation.get("preterminal_ledger_receipt_hash")
        != terminal.decision_ledger_receipt_hash
        or validation.get("aggregate_metrics") != dict(terminal.aggregate_metrics)
        or validation.get("exact_p_fallback_count")
        != terminal.exact_p_fallback_count
        or validation.get("raw_labels_persisted") is not False
        or validation.get("per_case_diagnostics_persisted") is not False
        or validation.get("v2_input_or_state_used") is not False
        or validation.get("cross_run_recovery_allowed") is not False
    ):
        raise ProtocolError("OE-PPUR v3 final validation report drifted.")

    validation_index = _read_json_object(destination / VALIDATION_INDEX_MEMBER)
    validation_index_body = {
        key: value
        for key, value in validation_index.items()
        if key != "validation_index_hash"
    }
    if (
        validation_index.get("schema_version")
        != "oe_ppur_v3_final_aggregate_validation_index_v1"
        or validation_index.get("status") != "PASS"
        or validation_index.get("validation_index_hash")
        != canonical_hash(validation_index_body)
        or validation_index.get("content_index_hash") != index["content_index_hash"]
        or validation_index.get("content_index_file_sha256")
        != _sha256_file(destination / CONTENT_INDEX_MEMBER)
        or validation_index.get("validation_report_hash")
        != validation["validation_report_hash"]
        or validation_index.get("validation_report_file_sha256")
        != _sha256_file(destination / VALIDATION_REPORT_MEMBER)
        or validation_index.get("final_attestation_hash")
        != final_attestation.receipt_hash
        or validation_index.get("final_attestation_file_sha256")
        != _sha256_file(destination / FINAL_ATTESTATION_MEMBER)
        or validation_index.get("fresh_process_count") != 2
        or validation_index.get("aggregate_only") is not True
        or validation_index.get("raw_labels_persisted") is not False
        or validation_index.get("v2_input_or_state_used") is not False
        or validation_index.get("cross_run_recovery_allowed") is not False
    ):
        raise ProtocolError("OE-PPUR v3 final validation index drifted.")

    receipt = FinalAggregateBundleReceipt(
        artifact_root=str(destination),
        content_index_hash=str(index["content_index_hash"]),
        validation_report_hash=str(validation["validation_report_hash"]),
        validation_index_hash=str(validation_index["validation_index_hash"]),
        final_attestation_hash=final_attestation.receipt_hash,
        terminal_receipt_hash=terminal.receipt_hash,
        _factory_token=_FINAL_BUNDLE_TOKEN,
    )
    if expected_receipt is not None and receipt != expected_receipt:
        raise ProtocolError("OE-PPUR v3 final bundle no longer matches its issued receipt.")
    return receipt


def _validate_final_binding(
    binding: Mapping[str, object],
    terminal: AggregateOnlyTerminalReceipt,
    final_attestation: FinalAggregateAttestationReceipt,
) -> None:
    expected_keys = {
        "schema_version",
        "output_artifact_id",
        "config_contract_hash",
        "protocol_hash",
        "seven_input_contract_hash",
        "source_seal_hash",
        "source_seal_receipt_hash",
        "source_supervision_contract_hash",
        "source_training_surface_receipt_hash",
        "source_training_surface_hash",
        "preterminal_boundary_receipt_hash",
        "preterminal_ledger_receipt_hash",
        "preterminal_attestation_receipt_hashes",
        "terminal_receipt_hash",
        "final_attestation_hash",
        "evaluated_case_count",
        "exact_p_fallback_count",
        "aggregate_only",
        "raw_labels_persisted",
        "v2_input_or_state_used",
        "cross_run_recovery_allowed",
    }
    hash_roles = expected_keys - {
        "schema_version",
        "output_artifact_id",
        "preterminal_attestation_receipt_hashes",
        "evaluated_case_count",
        "exact_p_fallback_count",
        "aggregate_only",
        "raw_labels_persisted",
        "v2_input_or_state_used",
        "cross_run_recovery_allowed",
    }
    attestations = binding.get("preterminal_attestation_receipt_hashes")
    if (
        set(binding) != expected_keys
        or binding.get("schema_version") != "oe_ppur_v3_final_aggregate_binding_v1"
        or binding.get("output_artifact_id") != OUTPUT_ARTIFACT_ID
        or any(
            require_sha256(binding.get(role), role) != binding.get(role)
            for role in hash_roles
        )
        or not isinstance(attestations, list)
        or len(attestations) != 2
        or len(set(attestations)) != 2
        or any(
            require_sha256(value, "preterminal attestation hash") != value
            for value in attestations
        )
        or binding.get("preterminal_boundary_receipt_hash")
        != terminal.boundary_receipt_hash
        or binding.get("preterminal_ledger_receipt_hash")
        != terminal.decision_ledger_receipt_hash
        or binding.get("terminal_receipt_hash") != terminal.receipt_hash
        or binding.get("final_attestation_hash") != final_attestation.receipt_hash
        or binding.get("evaluated_case_count") != terminal.evaluated_case_count
        or binding.get("exact_p_fallback_count") != terminal.exact_p_fallback_count
        or binding.get("aggregate_only") is not True
        or binding.get("raw_labels_persisted") is not False
        or binding.get("v2_input_or_state_used") is not False
        or binding.get("cross_run_recovery_allowed") is not False
    ):
        raise ProtocolError("OE-PPUR v3 final aggregate binding drifted.")


def validate_complete_artifact_inventory(root: str | Path) -> str:
    """Require exactly the catalog members plus the internal immutable lock."""

    destination = Path(os.path.abspath(root))
    _reject_symlink_chain(destination)
    if not destination.is_dir() or destination == Path(destination.anchor):
        raise ProtocolError("OE-PPUR v3 complete artifact root is unsafe.")
    observed: list[str] = []
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise ProtocolError("OE-PPUR v3 complete artifact contains a symlink.")
        if path.is_dir():
            continue
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ProtocolError("OE-PPUR v3 complete artifact member is unsafe.")
        observed.append(path.relative_to(destination).as_posix())
    expected = tuple(sorted((*COMPLETE_CATALOG_MEMBERS, *COMPLETE_INTERNAL_MEMBERS)))
    if tuple(sorted(observed)) != expected:
        raise ProtocolError("OE-PPUR v3 complete artifact inventory drifted.")
    return canonical_hash(
        {
            "schema_version": "oe_ppur_v3_complete_artifact_inventory_v1",
            "catalog_members": list(COMPLETE_CATALOG_MEMBERS),
            "internal_members": list(COMPLETE_INTERNAL_MEMBERS),
        }
    )


__all__: tuple[str, ...] = ()
