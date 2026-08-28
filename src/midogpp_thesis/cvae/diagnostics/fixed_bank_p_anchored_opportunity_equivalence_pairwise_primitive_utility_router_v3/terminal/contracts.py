"""Hash-only preterminal and aggregate-only terminal contracts for v3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
import math

from ....protocol import ProtocolError
from ..hashing import canonical_hash, require_sha256
from ..identity import EXPECTED_CASE_COUNT, EXPECTED_TERMINAL_CASE_INVENTORY_SHA256


_BOUNDARY_TOKEN = object()
_REQUEST_TOKEN = object()
_ATTESTATION_TOKEN = object()
ALLOWED_AGGREGATE_METRICS = (
    "selected_balanced_accuracy",
    "protected_p_balanced_accuracy",
    "selected_minus_p_balanced_accuracy",
    "selected_brier_score",
    "protected_p_brier_score",
    "p_minus_selected_brier_score",
    "selected_log_loss",
    "protected_p_log_loss",
    "p_minus_selected_log_loss",
    "routing_coverage",
    "top1_oracle_agreement",
    "spearman_rank_correlation",
    "normalized_oracle_gap",
    "exact_p_rate",
)
_FORBIDDEN_KEYS = (
    "label",
    "path",
    "row_id",
    "case_id",
    "sample_id",
    "prediction_vector",
)


@dataclass(frozen=True, slots=True)
class ArtifactOnlyPreterminalAttestationReceipt:
    sealed_ledger_receipt_hash: str
    artifact_file_sha256: str
    artifact_file_identity_sha256: str
    validator_runtime_sha256: str
    process_pid: int
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _ATTESTATION_TOKEN:
            raise ProtocolError("OE-PPUR v3 preterminal attestation bypassed validation.")
        for role in (
            "sealed_ledger_receipt_hash",
            "artifact_file_sha256",
            "artifact_file_identity_sha256",
            "validator_runtime_sha256",
        ):
            object.__setattr__(
                self, role, require_sha256(getattr(self, role), role.replace("_", " "))
            )
        if type(self.process_pid) is not int or self.process_pid <= 0:
            raise ProtocolError("OE-PPUR v3 attestation process identity drifted.")
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_artifact_only_preterminal_attestation_v1",
            "sealed_ledger_receipt_hash": self.sealed_ledger_receipt_hash,
            "artifact_file_sha256": self.artifact_file_sha256,
            "artifact_file_identity_sha256": self.artifact_file_identity_sha256,
            "validator_runtime_sha256": self.validator_runtime_sha256,
            "process_pid": self.process_pid,
            "artifact_only": True,
            "raw_path_present": False,
            "raw_labels_present": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


@dataclass(frozen=True, slots=True)
class GuardedPreterminalBoundary:
    seven_input_contract_hash: str
    source_seal_hash: str
    source_training_surface_receipt_hash: str
    decision_ledger_receipt_hash: str
    preterminal_attestation_hashes: tuple[str, str]
    case_inventory_sha256: str
    case_count: int
    exact_p_fallback_count: int
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _BOUNDARY_TOKEN:
            raise ProtocolError("OE-PPUR v3 terminal boundary bypassed sealing.")
        for role in (
            "seven_input_contract_hash",
            "source_seal_hash",
            "source_training_surface_receipt_hash",
            "decision_ledger_receipt_hash",
            "case_inventory_sha256",
        ):
            object.__setattr__(
                self, role, require_sha256(getattr(self, role), role.replace("_", " "))
            )
        attestations = tuple(
            require_sha256(value, "preterminal attestation hash")
            for value in self.preterminal_attestation_hashes
        )
        if (
            len(attestations) != 2
            or len(set(attestations)) != 2
            or self.case_inventory_sha256
            != EXPECTED_TERMINAL_CASE_INVENTORY_SHA256
            or self.case_count != EXPECTED_CASE_COUNT
            or type(self.exact_p_fallback_count) is not int
            or not 0 <= self.exact_p_fallback_count <= self.case_count
        ):
            raise ProtocolError("OE-PPUR v3 terminal boundary coverage drifted.")
        object.__setattr__(self, "preterminal_attestation_hashes", attestations)
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_guarded_preterminal_boundary_v1",
            "seven_input_contract_hash": self.seven_input_contract_hash,
            "source_seal_hash": self.source_seal_hash,
            "source_training_surface_receipt_hash": self.source_training_surface_receipt_hash,
            "decision_ledger_receipt_hash": self.decision_ledger_receipt_hash,
            "preterminal_attestation_hashes": list(self.preterminal_attestation_hashes),
            "case_inventory_sha256": self.case_inventory_sha256,
            "case_count": self.case_count,
            "exact_p_fallback_count": self.exact_p_fallback_count,
            "preterminal_decisions_sealed": True,
            "raw_paths_present": False,
            "raw_labels_present": False,
        }


@dataclass(frozen=True, slots=True)
class AggregateTerminalScoreRequest:
    boundary_receipt_hash: str
    decision_ledger_receipt_hash: str
    case_inventory_sha256: str
    case_count: int
    exact_p_fallback_count: int
    capability_hash: str
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _REQUEST_TOKEN:
            raise ProtocolError("OE-PPUR v3 terminal request bypassed capability.")
        for role in (
            "boundary_receipt_hash",
            "decision_ledger_receipt_hash",
            "case_inventory_sha256",
            "capability_hash",
        ):
            object.__setattr__(
                self, role, require_sha256(getattr(self, role), role.replace("_", " "))
            )
        if (
            self.case_inventory_sha256 != EXPECTED_TERMINAL_CASE_INVENTORY_SHA256
            or self.case_count != EXPECTED_CASE_COUNT
            or type(self.exact_p_fallback_count) is not int
            or not 0 <= self.exact_p_fallback_count <= self.case_count
        ):
            raise ProtocolError("OE-PPUR v3 terminal request coverage drifted.")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_aggregate_terminal_request_v1",
            "boundary_receipt_hash": self.boundary_receipt_hash,
            "decision_ledger_receipt_hash": self.decision_ledger_receipt_hash,
            "case_inventory_sha256": self.case_inventory_sha256,
            "case_count": self.case_count,
            "exact_p_fallback_count": self.exact_p_fallback_count,
            "capability_hash": self.capability_hash,
            "raw_paths_present": False,
            "raw_labels_present": False,
        }

    def __reduce__(self):  # pragma: no cover - explicit safety seam
        raise TypeError("OE-PPUR v3 terminal requests cannot be serialized.")


@dataclass(frozen=True, slots=True)
class AggregateOnlyTerminalReceipt:
    boundary_receipt_hash: str
    decision_ledger_receipt_hash: str
    evaluated_case_count: int
    routed_case_count: int
    exact_p_fallback_count: int
    aggregate_metrics: tuple[tuple[str, float], ...]
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        metrics = tuple((str(key), float(value)) for key, value in self.aggregate_metrics)
        if (
            self.evaluated_case_count != EXPECTED_CASE_COUNT
            or type(self.routed_case_count) is not int
            or not 0 <= self.routed_case_count <= self.evaluated_case_count
            or type(self.exact_p_fallback_count) is not int
            or self.routed_case_count + self.exact_p_fallback_count
            != self.evaluated_case_count
            or tuple(key for key, _ in metrics) != ALLOWED_AGGREGATE_METRICS
            or any(not math.isfinite(value) for _, value in metrics)
        ):
            raise ProtocolError("OE-PPUR v3 aggregate terminal receipt drifted.")
        object.__setattr__(self, "boundary_receipt_hash", require_sha256(self.boundary_receipt_hash, "terminal boundary hash"))
        object.__setattr__(self, "decision_ledger_receipt_hash", require_sha256(self.decision_ledger_receipt_hash, "terminal decision ledger hash"))
        object.__setattr__(self, "aggregate_metrics", metrics)
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))
        assert_aggregate_only_payload(self.to_payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_aggregate_only_terminal_receipt_v1",
            "boundary_receipt_hash": self.boundary_receipt_hash,
            "decision_ledger_receipt_hash": self.decision_ledger_receipt_hash,
            "evaluated_case_count": self.evaluated_case_count,
            "routed_case_count": self.routed_case_count,
            "exact_p_fallback_count": self.exact_p_fallback_count,
            "aggregate_metrics": {key: value for key, value in self.aggregate_metrics},
            "raw_paths_present": False,
            "raw_labels_present": False,
            "per_row_values_present": False,
            "per_case_values_present": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def seal_guarded_preterminal_boundary(
    *,
    seven_input_contract_hash: str,
    source_seal_hash: str,
    source_training_surface_receipt_hash: str,
    decision_ledger_receipt_hash: str,
    attestations: Sequence[ArtifactOnlyPreterminalAttestationReceipt],
    case_inventory_sha256: str,
    case_count: int,
    exact_p_fallback_count: int,
) -> GuardedPreterminalBoundary:
    """Seal hash-only decisions before a terminal reader can be constructed."""

    rows = tuple(attestations)
    if (
        len(rows) != 2
        or any(type(row) is not ArtifactOnlyPreterminalAttestationReceipt for row in rows)
        or len({row.receipt_hash for row in rows}) != 2
        or len({row.process_pid for row in rows}) != 2
        or any(row.sealed_ledger_receipt_hash != decision_ledger_receipt_hash for row in rows)
    ):
        raise ProtocolError(
            "OE-PPUR v3 terminal boundary requires two distinct artifact-only attestations."
        )
    return GuardedPreterminalBoundary(
        seven_input_contract_hash=seven_input_contract_hash,
        source_seal_hash=source_seal_hash,
        source_training_surface_receipt_hash=source_training_surface_receipt_hash,
        decision_ledger_receipt_hash=decision_ledger_receipt_hash,
        preterminal_attestation_hashes=tuple(row.receipt_hash for row in rows),
        case_inventory_sha256=case_inventory_sha256,
        case_count=case_count,
        exact_p_fallback_count=exact_p_fallback_count,
        _factory_token=_BOUNDARY_TOKEN,
    )


def issue_artifact_only_preterminal_attestation(
    *,
    sealed_ledger_receipt_hash: str,
    artifact_file_sha256: str,
    artifact_file_identity_sha256: str,
    validator_runtime_sha256: str,
    process_pid: int,
) -> ArtifactOnlyPreterminalAttestationReceipt:
    """Issue one hash-only receipt from an already-fresh validator process."""

    return ArtifactOnlyPreterminalAttestationReceipt(
        sealed_ledger_receipt_hash=sealed_ledger_receipt_hash,
        artifact_file_sha256=artifact_file_sha256,
        artifact_file_identity_sha256=artifact_file_identity_sha256,
        validator_runtime_sha256=validator_runtime_sha256,
        process_pid=process_pid,
        _factory_token=_ATTESTATION_TOKEN,
    )


def _issue_terminal_request(
    boundary: GuardedPreterminalBoundary,
    *,
    capability_hash: str,
) -> AggregateTerminalScoreRequest:
    return AggregateTerminalScoreRequest(
        boundary_receipt_hash=boundary.receipt_hash,
        decision_ledger_receipt_hash=boundary.decision_ledger_receipt_hash,
        case_inventory_sha256=boundary.case_inventory_sha256,
        case_count=boundary.case_count,
        exact_p_fallback_count=boundary.exact_p_fallback_count,
        capability_hash=capability_hash,
        _factory_token=_REQUEST_TOKEN,
    )


def assert_aggregate_only_payload(value: Mapping[str, object]) -> None:
    if not isinstance(value, Mapping):
        raise ProtocolError("OE-PPUR v3 terminal payload is not a mapping.")

    def walk(node: object) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                lowered = str(key).lower()
                if any(token in lowered for token in _FORBIDDEN_KEYS) and child is not False:
                    raise ProtocolError(
                        "OE-PPUR v3 terminal payload exposes labels, paths, or identities."
                    )
                walk(child)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for child in node:
                walk(child)

    walk(value)


__all__ = (
    "ALLOWED_AGGREGATE_METRICS",
    "AggregateOnlyTerminalReceipt",
    "AggregateTerminalScoreRequest",
    "ArtifactOnlyPreterminalAttestationReceipt",
    "GuardedPreterminalBoundary",
    "assert_aggregate_only_payload",
    "issue_artifact_only_preterminal_attestation",
    "seal_guarded_preterminal_boundary",
)
