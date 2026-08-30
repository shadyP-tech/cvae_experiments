"""Dependency-light schemas for terminal v4 authorization outcomes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256


OUTCOME_MEMBER = "outcome.json"
_OUTCOME_TOKEN = object()


@dataclass(frozen=True, slots=True)
class AuthorizationOutcomeReceipt:
    lease_path: Path
    status: str
    claim_hash: str
    evidence_hash: str
    terminal_run_state_receipt_hash: str | None
    final_bundle_receipt_hash: str | None
    artifact_inventory_hash: str | None
    lifecycle_lineage_hash: str | None
    outcome_hash: str
    completion_commit_hash: str | None = None
    complete_artifact_seal_receipt_hash: str | None = None
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _OUTCOME_TOKEN:
            raise ProtocolError("OE-PPUR v4 authorization outcome bypassed validation.")
        path = Path(self.lease_path)
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_dir()
            or self.status not in {"COMPLETE", "FAILED_EXHAUSTED"}
        ):
            raise ProtocolError("OE-PPUR v4 authorization outcome receipt drifted.")
        object.__setattr__(self, "lease_path", path)
        for role in ("claim_hash", "evidence_hash", "outcome_hash"):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )
        for role in (
            "terminal_run_state_receipt_hash",
            "final_bundle_receipt_hash",
            "artifact_inventory_hash",
            "lifecycle_lineage_hash",
            "completion_commit_hash",
            "complete_artifact_seal_receipt_hash",
        ):
            value = getattr(self, role)
            if value is not None:
                object.__setattr__(
                    self,
                    role,
                    require_sha256(value, role.replace("_", " ")),
                )
        if self.status == "COMPLETE" and any(
            getattr(self, role) is None
            for role in (
                "terminal_run_state_receipt_hash",
                "final_bundle_receipt_hash",
                "artifact_inventory_hash",
                "lifecycle_lineage_hash",
                "completion_commit_hash",
                "complete_artifact_seal_receipt_hash",
            )
        ):
            raise ProtocolError("OE-PPUR v4 complete outcome lacks whole-run evidence.")
        if self.status == "FAILED_EXHAUSTED" and any(
            getattr(self, role) is not None
            for role in (
                "final_bundle_receipt_hash",
                "artifact_inventory_hash",
                "lifecycle_lineage_hash",
                "completion_commit_hash",
                "complete_artifact_seal_receipt_hash",
            )
        ):
            raise ProtocolError("OE-PPUR v4 failed outcome carries success evidence.")
        object.__setattr__(self, "receipt_hash", canonical_hash(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_authorization_outcome_receipt_v1",
            "lease_path": self.lease_path.as_posix(),
            "status": self.status,
            "claim_hash": self.claim_hash,
            "evidence_hash": self.evidence_hash,
            "terminal_run_state_receipt_hash": self.terminal_run_state_receipt_hash,
            "final_bundle_receipt_hash": self.final_bundle_receipt_hash,
            "artifact_inventory_hash": self.artifact_inventory_hash,
            "lifecycle_lineage_hash": self.lifecycle_lineage_hash,
            "completion_commit_hash": self.completion_commit_hash,
            "complete_artifact_seal_receipt_hash": (
                self.complete_artifact_seal_receipt_hash
            ),
            "outcome_hash": self.outcome_hash,
        }


def build_authorization_outcome_payload(
    *,
    status: str,
    claim_hash: str,
    evidence_hash: str,
    terminal_run_state_receipt_hash: str | None,
    final_bundle_receipt_hash: str | None,
    artifact_inventory_hash: str | None,
    lifecycle_lineage_hash: str | None,
    completion_commit_hash: str | None,
    complete_artifact_seal_receipt_hash: str | None,
    error_class: str | None,
    recorded_at_utc: str,
) -> dict[str, object]:
    body = {
        "schema_version": "oe_ppur_v4_single_use_authorization_outcome_v1",
        "status": status,
        "claim_hash": claim_hash,
        "evidence_hash": evidence_hash,
        "terminal_run_state_receipt_hash": terminal_run_state_receipt_hash,
        "final_bundle_receipt_hash": final_bundle_receipt_hash,
        "artifact_inventory_hash": artifact_inventory_hash,
        "lifecycle_lineage_hash": lifecycle_lineage_hash,
        "completion_commit_hash": completion_commit_hash,
        "complete_artifact_seal_receipt_hash": (
            complete_artifact_seal_receipt_hash
        ),
        "error_class": error_class,
        "recorded_at_utc": recorded_at_utc,
        "authorization_exhausted": True,
        "authorization_restored": False,
        "recovery_allowed": False,
    }
    payload = {**body, "outcome_hash": canonical_hash(body)}
    # Parse through the schema before any writer receives the payload.  A
    # temporary path is not available here, so path-bound receipt issuance is
    # deliberately left to ``outcome_receipt`` after persistence.
    _validate_outcome_payload(payload)
    return payload


def outcome_receipt(
    lease_path: Path,
    payload: Mapping[str, object],
) -> AuthorizationOutcomeReceipt:
    normalized = _validate_outcome_payload(payload)
    return AuthorizationOutcomeReceipt(
        lease_path=Path(lease_path),
        status=str(normalized["status"]),
        claim_hash=str(normalized["claim_hash"]),
        evidence_hash=str(normalized["evidence_hash"]),
        terminal_run_state_receipt_hash=_optional_text(
            normalized["terminal_run_state_receipt_hash"]
        ),
        final_bundle_receipt_hash=_optional_text(
            normalized["final_bundle_receipt_hash"]
        ),
        artifact_inventory_hash=_optional_text(
            normalized["artifact_inventory_hash"]
        ),
        lifecycle_lineage_hash=_optional_text(normalized["lifecycle_lineage_hash"]),
        completion_commit_hash=_optional_text(normalized["completion_commit_hash"]),
        complete_artifact_seal_receipt_hash=_optional_text(
            normalized["complete_artifact_seal_receipt_hash"]
        ),
        outcome_hash=str(normalized["outcome_hash"]),
        _factory_token=_OUTCOME_TOKEN,
    )


def _validate_outcome_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    expected_keys = {
        "schema_version",
        "status",
        "claim_hash",
        "evidence_hash",
        "terminal_run_state_receipt_hash",
        "final_bundle_receipt_hash",
        "artifact_inventory_hash",
        "lifecycle_lineage_hash",
        "completion_commit_hash",
        "complete_artifact_seal_receipt_hash",
        "error_class",
        "recorded_at_utc",
        "authorization_exhausted",
        "authorization_restored",
        "recovery_allowed",
        "outcome_hash",
    }
    body = {key: value for key, value in payload.items() if key != "outcome_hash"}
    status = payload.get("status")
    if (
        set(payload) != expected_keys
        or payload.get("schema_version")
        != "oe_ppur_v4_single_use_authorization_outcome_v1"
        or status not in {"COMPLETE", "FAILED_EXHAUSTED"}
        or payload.get("outcome_hash") != canonical_hash(body)
        or payload.get("authorization_exhausted") is not True
        or payload.get("authorization_restored") is not False
        or payload.get("recovery_allowed") is not False
        or not isinstance(payload.get("recorded_at_utc"), str)
        or (status == "COMPLETE") != (payload.get("error_class") is None)
        or (
            status == "FAILED_EXHAUSTED"
            and not isinstance(payload.get("error_class"), str)
        )
        or (
            status == "FAILED_EXHAUSTED"
            and any(
                payload.get(role) is not None
                for role in (
                    "final_bundle_receipt_hash",
                    "artifact_inventory_hash",
                    "lifecycle_lineage_hash",
                    "completion_commit_hash",
                    "complete_artifact_seal_receipt_hash",
                )
            )
        )
    ):
        raise ProtocolError("OE-PPUR v4 authorization outcome drifted.")
    for role in ("claim_hash", "evidence_hash", "outcome_hash"):
        require_sha256(payload.get(role), role.replace("_", " "))
    evidence_roles = (
        "terminal_run_state_receipt_hash",
        "final_bundle_receipt_hash",
        "artifact_inventory_hash",
        "lifecycle_lineage_hash",
        "completion_commit_hash",
        "complete_artifact_seal_receipt_hash",
    )
    for role in evidence_roles:
        value = payload.get(role)
        if value is not None:
            require_sha256(value, role.replace("_", " "))
    if status == "COMPLETE" and any(payload.get(role) is None for role in evidence_roles):
        raise ProtocolError("OE-PPUR v4 complete outcome lacks whole-run evidence.")
    return dict(payload)


def safe_text(value: object) -> str:
    return " ".join(str(value).split())[:160]


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


__all__ = ("AuthorizationOutcomeReceipt", "OUTCOME_MEMBER")
