"""New, single-use execution authority for terminal HARP sensitivity v1."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import atomic_json, read_json, sha256_file
from .config import HarpStage90Config
from .identity import (
    AUTHORIZATION_SCOPE,
    EXPERIMENT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    authorization_input_binding_payload,
)


_LEASE_DIRECTORY = (
    ".authorization_lease__midogpp_oracle_uniform_b_v2_consumed_test_"
    "fixed_bank_harp_router_v1"
)


@dataclass(frozen=True, slots=True)
class HarpAuthorization:
    amendment_path: Path
    amendment_sha256: str
    amendment_hash: str


@dataclass(frozen=True, slots=True)
class HarpAuthorizationLease:
    root: Path
    lease_hash: str
    process_id: int


def repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def lease_path() -> Path:
    return (
        repository_root()
        / "artifacts/midogpp/90_oracles_and_diagnostics"
        / _LEASE_DIRECTORY
    )


def load_authorization(config: HarpStage90Config) -> HarpAuthorization:
    """Authenticate the new HARP-only amendment without mutating state."""

    if not isinstance(config, HarpStage90Config) or not config.execution_authorized:
        raise ProtocolError(
            "HARP Stage-90 execution is not authorized; a new HARP-specific "
            "single-use amendment is required."
        )
    expected = config.expected_execution_amendment_sha256
    if expected is None:
        raise ProtocolError("HARP Stage-90 execution amendment hash is absent.")
    path = config.resolved_path("execution_amendment_path")
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
        raise ProtocolError("HARP Stage-90 execution amendment bytes are absent or drifted.")
    raw = read_json(path)
    base = {key: value for key, value in raw.items() if key != "amendment_hash"}
    input_binding = _input_binding(config)
    if (
        set(raw)
        != {
            "schema_version",
            "experiment_id",
            "authorization_scope",
            "execution_authorized",
            "single_use",
            "consumed_test_reuse",
            "publication_status",
            "terminal_decision",
            "fresh_evidence",
            "predecessor_authority_reused",
            "predecessor_output_or_policy_used",
            "old_aggregate_utility_surface_used",
            "output_deletion_restores_authority",
            "authorized_input_binding",
            "amendment_hash",
        }
        or raw.get("schema_version")
        != "midogpp_harp_stage90_execution_amendment_v1"
        or raw.get("experiment_id") != EXPERIMENT_ID
        or raw.get("authorization_scope") != AUTHORIZATION_SCOPE
        or raw.get("execution_authorized") is not True
        or raw.get("single_use") is not True
        or raw.get("consumed_test_reuse") is not True
        or raw.get("publication_status") != PUBLICATION_STATUS
        or raw.get("terminal_decision") != TERMINAL_DECISION
        or raw.get("fresh_evidence") is not False
        or raw.get("predecessor_authority_reused") is not False
        or raw.get("predecessor_output_or_policy_used") is not False
        or raw.get("old_aggregate_utility_surface_used") is not False
        or raw.get("output_deletion_restores_authority") is not False
        or raw.get("authorized_input_binding") != input_binding
        or raw.get("amendment_hash") != canonical_hash(base)
    ):
        raise ProtocolError("HARP Stage-90 execution amendment failed authentication.")
    lease = lease_path()
    if lease.exists() or lease.is_symlink():
        raise ProtocolError("HARP Stage-90 single-use authorization is exhausted.")
    if not lease.parent.is_dir() or lease.parent.is_symlink():
        raise ProtocolError("HARP Stage-90 authorization lease parent is unsafe.")
    return HarpAuthorization(path, expected, str(raw["amendment_hash"]))


def _input_binding(config: HarpStage90Config) -> dict[str, object]:
    roles = (
        "expert_bank_lock_hash",
        "generation_lock_hash",
        "test_cache_content_sha256",
        "development_manifest_sha256",
        "evaluation_manifest_sha256",
        "parent_ledger_sha256",
    )
    values = {role: config.expected_hashes.get(role) for role in roles}
    if any(type(value) is not str or not value for value in values.values()):
        raise ProtocolError("HARP Stage-90 amendment input binding is incomplete.")
    return authorization_input_binding_payload(**values)  # type: ignore[arg-type]


def claim_authorization(
    authorization: HarpAuthorization, *, admission_hash: str
) -> HarpAuthorizationLease:
    """Atomically consume authority; the lease directory is the first mutation."""

    if not isinstance(authorization, HarpAuthorization):
        raise ProtocolError("HARP Stage-90 lease lacks typed authorization.")
    root = lease_path()
    try:
        root.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise ProtocolError("HARP Stage-90 single-use authorization is exhausted.") from exc
    except OSError as exc:
        raise ProtocolError("Cannot claim HARP Stage-90 authorization lease.") from exc
    payload = {
        "schema_version": "midogpp_harp_stage90_authorization_lease_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "CLAIMED_IN_PROGRESS",
        "process_id": os.getpid(),
        "admission_hash": admission_hash,
        "amendment_sha256": authorization.amendment_sha256,
        "amendment_hash": authorization.amendment_hash,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "authorization_exhausted": True,
        "recovery_allowed": False,
        "output_deletion_restores_authority": False,
    }
    sealed = {**payload, "lease_hash": canonical_hash(payload)}
    atomic_json(root / "lease.json", sealed)
    return HarpAuthorizationLease(root, str(sealed["lease_hash"]), os.getpid())


def finalize_authorization(
    lease: HarpAuthorizationLease, *, status: str, error: str | None = None
) -> HarpAuthorizationLease:
    if status not in {"COMPLETE_EXHAUSTED", "FAILED_EXHAUSTED"}:
        raise ProtocolError("HARP Stage-90 lease final status is invalid.")
    raw = read_json(lease.root / "lease.json")
    if raw.get("lease_hash") != lease.lease_hash or raw.get("process_id") != os.getpid():
        raise ProtocolError("HARP Stage-90 authorization lease was replaced.")
    base = {key: value for key, value in raw.items() if key not in {"lease_hash", "status"}}
    base.update(
        {
            "status": status,
            "predecessor_lease_hash": lease.lease_hash,
            "error": None if error is None else error[:2000],
        }
    )
    final_hash = canonical_hash(base)
    atomic_json(lease.root / "lease.json", {**base, "lease_hash": final_hash})
    return HarpAuthorizationLease(lease.root, final_hash, lease.process_id)


__all__ = (
    "HarpAuthorization",
    "HarpAuthorizationLease",
    "claim_authorization",
    "finalize_authorization",
    "lease_path",
    "load_authorization",
)
