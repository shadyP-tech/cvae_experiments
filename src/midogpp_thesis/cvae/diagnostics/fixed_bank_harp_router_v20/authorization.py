"""Separate, single-use authority for the fenced HARP v20 diagnostic."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
import os
from pathlib import Path
from types import MappingProxyType

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_bytes, canonical_hash, require_sha256
from ...runtime.artifact_io import atomic_json, read_json, sha256_file
from ...runtime.harp_v20_execution.action_capacity import (
    build_action_capacity_certificate,
    validate_action_capacity_certificate,
)
from .activation_lock import activation_lock
from .activation_paths import RepositoryBoundary
from .config import HarpStage90V20Config, INPUT_ARTIFACT_IDS
from .identity import (
    AUTHORIZATION_SCOPE,
    EXECUTION_REVISION,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    authorization_input_binding_payload,
)
from .source_seal import source_snapshot_identity


_LEASE_DIRECTORY = (
    ".authorization_lease__midogpp_oracle_uniform_b_v2_consumed_test_"
    "fixed_bank_harp_router_v20"
)
EXECUTION_AMENDMENT_ARTIFACT_ID = INPUT_ARTIFACT_IDS[-1]
EXECUTION_AMENDMENT_FILENAME = "harp_stage90_execution_amendment_v20.json"
EXECUTION_AMENDMENT_SCHEMA = "midogpp_harp_stage90_execution_amendment_v20"
SCIENTIFIC_CONTRACT_SCHEMA = "midogpp_harp_stage90_scientific_contract_v20"
AUTHORIZATION_BASIS = "explicit_user_authorization_for_harp_v20_terminal_consumed_test_diagnostic"
WORKSPACE_REGISTRATION_CONTRACT_SCHEMA = "midogpp_harp_stage90_workspace_registration_execution_contract_v20"
WORKSPACE_REGISTRY_RELATIVE_PATH = "experiments/midogpp/registry.yaml"
WORKSPACE_ARTIFACT_CATALOG_RELATIVE_PATH = "experiments/midogpp/artifact_catalog.yaml"
WORKSPACE_CONFIG_RELATIVE_PATH = (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_harp_router_v20.yaml"
)
WORKSPACE_PREPARATION_AUTHORITY_GATE = "harp_v20_consumed_test_execution_amendment_v1"
WORKSPACE_OUTPUT_CANONICAL_PATH = (
    "artifacts/midogpp/90_oracles_and_diagnostics/"
    "uniform_b_v2_consumed_test_fixed_bank_harp_router/v20"
)
WORKSPACE_AMENDMENT_RELATIVE_PATH = (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts/"
    "harp_router_v20/harp_stage90_execution_amendment_v20.json"
)
WORKSPACE_RUNNER_ARGV = (
    "{python}", "-m", "midogpp_thesis", "cvae-diagnostics",
    "fixed-bank-harp-router-v20", "--config", "{resolved_config}",
    "--artifact-root", f"output://{OUTPUT_ARTIFACT_ID}",
)
WORKSPACE_RUNNER_ENV = MappingProxyType({
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUDA_VISIBLE_DEVICES": "0,1",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": "0",
    "OMP_DYNAMIC": "FALSE", "MKL_DYNAMIC": "FALSE",
})


@dataclass(frozen=True, slots=True)
class HarpV20Authorization:
    amendment_path: Path
    amendment_sha256: str
    amendment_hash: str
    input_binding_hash: str
    scientific_contract_hash: str
    workspace_registration_execution_contract_hash: str
    source_snapshot_schema: str
    source_snapshot_manifest_sha256: str
    source_snapshot_tree_sha256: str
    source_snapshot_member_count: int


@dataclass(frozen=True, slots=True)
class HarpV20ExecutionAmendment:
    payload: Mapping[str, object]
    amendment_hash: str
    input_binding_hash: str
    scientific_contract_hash: str
    workspace_registration_execution_contract_hash: str
    source_snapshot_schema: str
    source_snapshot_manifest_sha256: str
    source_snapshot_tree_sha256: str
    source_snapshot_member_count: int


@dataclass(frozen=True, slots=True)
class HarpV20AuthorizationLease:
    root: Path
    lease_hash: str
    process_id: int


def repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def lease_path(repo_root: Path | None = None) -> Path:
    repository = repository_root() if repo_root is None else Path(repo_root).resolve()
    return repository / "artifacts/midogpp/90_oracles_and_diagnostics" / _LEASE_DIRECTORY


def authorization_input_binding(config: HarpStage90V20Config) -> dict[str, object]:
    if type(config) is not HarpStage90V20Config:
        raise ProtocolError("HARP v20 input binding requires a typed config.")
    roles = (
        "expert_bank_lock_hash", "generation_lock_hash", "test_cache_content_sha256",
        "development_manifest_sha256", "evaluation_manifest_sha256", "parent_ledger_sha256",
    )
    values = {role: config.expected_hashes.get(role) for role in roles}
    if any(type(value) is not str or not value for value in values.values()):
        raise ProtocolError("HARP v20 amendment input binding is incomplete.")
    return authorization_input_binding_payload(**values)  # type: ignore[arg-type]


def workspace_registration_execution_contract() -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": WORKSPACE_REGISTRATION_CONTRACT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "stage": "90_oracles_and_diagnostics", "status": "diagnostic",
        "claim_scope": "diagnostic_only", "config_path": WORKSPACE_CONFIG_RELATIVE_PATH,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "output_canonical_path": WORKSPACE_OUTPUT_CANONICAL_PATH,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "preparation_authority_gate": WORKSPACE_PREPARATION_AUTHORITY_GATE,
        "run_recovery_strategy": None,
        "runner_argv": list(WORKSPACE_RUNNER_ARGV),
        "runner_environment": dict(WORKSPACE_RUNNER_ENV),
    }
    return {**body, "workspace_registration_execution_contract_hash": canonical_hash(body)}


def validate_workspace_registration_execution_projection(
    projection: Mapping[str, object] | None,
) -> str:
    if not isinstance(projection, Mapping):
        raise ProtocolError("HARP v20 in-memory workspace registration projection is absent.")
    expected = workspace_registration_execution_contract()
    observed = {
        "schema_version": expected["schema_version"], **dict(projection),
        "workspace_registration_execution_contract_hash": expected[
            "workspace_registration_execution_contract_hash"
        ],
    }
    if canonical_bytes(observed) != canonical_bytes(expected):
        raise ProtocolError("HARP v20 workspace registration execution contract drifted.")
    return str(expected["workspace_registration_execution_contract_hash"])


def scientific_contract_payload(config: HarpStage90V20Config) -> dict[str, object]:
    if type(config) is not HarpStage90V20Config:
        raise ProtocolError("HARP v20 scientific contract requires typed config.")
    claim_boundary = {
        key: value for key, value in config.claim_boundary.items()
        if key not in {"claim_boundary_hash", "execution_authorized", "implementation_authorizes_execution"}
    }
    capacity = dict(
        build_action_capacity_certificate(
            centers=tuple(str(value) for value in config.protocol["centers"])
        )
    )
    validate_action_capacity_certificate(
        capacity,
        centers=tuple(str(value) for value in config.protocol["centers"]),
    )
    body: dict[str, object] = {
        "schema_version": SCIENTIFIC_CONTRACT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_revision": EXECUTION_REVISION,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "workspace_registration_execution_contract": workspace_registration_execution_contract(),
        "direct_input_artifact_ids": list(config.input_artifact_ids),
        "protocol": dict(config.protocol),
        "model": dict(config.model),
        "runtime": dict(config.runtime), "claim_boundary": claim_boundary,
        "action_capacity_certificate": capacity,
        "architecture_revision": True,
        "performance_profile_is_scientifically_invariant": True,
    }
    return {**body, "scientific_contract_hash": canonical_hash(body)}


def canonical_execution_amendment_payload(
    config: HarpStage90V20Config,
    *,
    authorization_basis: str,
    authorization_date: str,
    repo_root: Path | None = None,
) -> dict[str, object]:
    if type(config) is not HarpStage90V20Config or not config.execution_authorized:
        raise ProtocolError("HARP v20 canonical amendment requires an authorized typed config.")
    validate_activation_metadata(authorization_basis, authorization_date)
    binding = authorization_input_binding(config)
    scientific = scientific_contract_payload(config)
    registration = workspace_registration_execution_contract()
    body: dict[str, object] = {
        "schema_version": EXECUTION_AMENDMENT_SCHEMA,
        "experiment_id": EXPERIMENT_ID, "execution_revision": EXECUTION_REVISION,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "authorization_basis": authorization_basis,
        "authorization_date": authorization_date,
        "execution_authorized": True,
        "authorization_is_separate_from_implementation_request": True,
        "implementation_request_alone_authorizes_execution": False,
        "source_code_or_registration_alone_authorizes_execution": False,
        "single_use": True, "authorization_exhausted": False,
        "consumed_test_reuse": True,
        "direct_input_artifact_ids": list(config.input_artifact_ids),
        "publication_status": PUBLICATION_STATUS, "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False, "predecessor_authority_reused": False,
        "predecessor_output_or_policy_used": False,
        "old_aggregate_utility_surface_used": False,
        "predecessor_amendment_lease_output_cache_or_scratch_reused": False,
        "output_deletion_restores_authority": False,
        "authorized_input_binding": binding,
        "scientific_contract_hash": scientific["scientific_contract_hash"],
        "workspace_registration_execution_contract_hash": registration[
            "workspace_registration_execution_contract_hash"
        ],
        "source_snapshot_identity": dict(source_snapshot_identity(repo_root)),
    }
    return {**body, "amendment_hash": canonical_hash(body)}


def validate_execution_amendment_payload(
    value: object, config: HarpStage90V20Config, *, repo_root: Path | None = None,
) -> HarpV20ExecutionAmendment:
    if not isinstance(value, Mapping):
        raise ProtocolError("HARP v20 execution amendment must be an object.")
    basis = value.get("authorization_basis")
    activated_on = value.get("authorization_date")
    if type(basis) is not str or type(activated_on) is not str:
        raise ProtocolError("HARP v20 execution amendment lacks activation metadata.")
    expected = canonical_execution_amendment_payload(
        config,
        authorization_basis=basis,
        authorization_date=activated_on,
        repo_root=repo_root,
    )
    if canonical_bytes(value) != canonical_bytes(expected):
        raise ProtocolError("HARP v20 execution amendment failed authentication.")
    binding = expected["authorized_input_binding"]
    source = expected["source_snapshot_identity"]
    if not isinstance(binding, Mapping) or not isinstance(source, Mapping):
        raise ProtocolError("HARP v20 amendment reconstruction failed.")
    member_count = source.get("source_snapshot_member_count")
    snapshot_schema = source.get("source_snapshot_schema")
    if type(member_count) is not int or member_count < 1 or type(snapshot_schema) is not str:
        raise ProtocolError("HARP v20 source snapshot identity is invalid.")
    return HarpV20ExecutionAmendment(
        payload=MappingProxyType(dict(expected)),
        amendment_hash=require_sha256(expected["amendment_hash"], name="amendment_hash"),
        input_binding_hash=require_sha256(binding.get("input_binding_hash"), name="input_binding_hash"),
        scientific_contract_hash=require_sha256(expected["scientific_contract_hash"], name="scientific_contract_hash"),
        workspace_registration_execution_contract_hash=require_sha256(
            expected["workspace_registration_execution_contract_hash"],
            name="workspace_registration_execution_contract_hash"),
        source_snapshot_schema=snapshot_schema,
        source_snapshot_manifest_sha256=require_sha256(
            source.get("source_snapshot_manifest_sha256"), name="source_snapshot_manifest_sha256"),
        source_snapshot_tree_sha256=require_sha256(
            source.get("source_snapshot_tree_sha256"), name="source_snapshot_tree_sha256"),
        source_snapshot_member_count=member_count,
    )


def validate_activation_metadata(basis: str, activated_on: str) -> None:
    if basis != AUTHORIZATION_BASIS:
        raise ProtocolError("HARP v20 explicit authorization basis is absent or drifted.")
    try:
        parsed = date.fromisoformat(activated_on)
    except ValueError as exc:
        raise ProtocolError("HARP v20 authorization date must be strict YYYY-MM-DD.") from exc
    if parsed.isoformat() != activated_on:
        raise ProtocolError("HARP v20 authorization date must be canonical YYYY-MM-DD.")


def load_authorization(
    config: HarpStage90V20Config, *, repo_root: Path | None = None,
) -> HarpV20Authorization:
    if type(config) is not HarpStage90V20Config or not config.execution_authorized:
        raise ProtocolError("HARP v20 execution is not authorized; separate activation is required.")
    expected = config.expected_execution_amendment_sha256
    path = config.resolved_path("execution_amendment_path")
    if type(expected) is not str or not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
        raise ProtocolError("HARP v20 execution amendment bytes are absent or drifted.")
    amendment = validate_execution_amendment_payload(read_json(path), config, repo_root=repo_root)
    lease = lease_path(repo_root)
    if lease.exists() or lease.is_symlink():
        _validate_active_lease(
            lease,
            amendment_sha256=expected,
            amendment_hash=amendment.amendment_hash,
            input_binding_hash=amendment.input_binding_hash,
        )
    if not lease.parent.is_dir() or lease.parent.is_symlink():
        raise ProtocolError("HARP v20 authorization lease parent is unsafe.")
    return HarpV20Authorization(
        amendment_path=path, amendment_sha256=expected,
        amendment_hash=amendment.amendment_hash, input_binding_hash=amendment.input_binding_hash,
        scientific_contract_hash=amendment.scientific_contract_hash,
        workspace_registration_execution_contract_hash=amendment.workspace_registration_execution_contract_hash,
        source_snapshot_schema=amendment.source_snapshot_schema,
        source_snapshot_manifest_sha256=amendment.source_snapshot_manifest_sha256,
        source_snapshot_tree_sha256=amendment.source_snapshot_tree_sha256,
        source_snapshot_member_count=amendment.source_snapshot_member_count,
    )


def claim_authorization(
    authorization: HarpV20Authorization,
    *,
    admission_hash: str,
    repo_root: Path | None = None,
) -> HarpV20AuthorizationLease:
    if type(authorization) is not HarpV20Authorization:
        raise ProtocolError("HARP v20 lease lacks typed authorization.")
    admission = require_sha256(admission_hash, name="admission_hash")
    repository = repository_root() if repo_root is None else Path(repo_root).resolve()

    boundary = RepositoryBoundary.open(repository)
    with activation_lock(boundary):
        # This is intentionally inside the lock and immediately before the
        # lease operation. A delayed runner may not use an authority object
        # loaded before an activation supersession changed canonical state.
        current = _reload_canonical_authorization(boundary.resolved_root)
        if current != authorization:
            raise ProtocolError(
                "HARP v20 authorization changed between admission and lease claim."
            )
        root = lease_path(boundary.resolved_root)
        if root.exists() and not root.is_symlink():
            raw = _validate_active_lease(
                root,
                amendment_sha256=authorization.amendment_sha256,
                amendment_hash=authorization.amendment_hash,
                input_binding_hash=authorization.input_binding_hash,
                admission_hash=admission,
            )
            owner = int(raw["process_id"])
            if owner != os.getpid() and _process_is_alive(owner):
                raise ProtocolError(
                    "HARP v20 active authorization lease is owned by a live process."
                )
            return HarpV20AuthorizationLease(root, str(raw["lease_hash"]), os.getpid())
        try:
            root.mkdir(mode=0o700, parents=False, exist_ok=False)
        except FileExistsError as exc:
            raise ProtocolError("HARP v20 single-use authorization is exhausted.") from exc
        except OSError as exc:
            raise ProtocolError("Cannot claim HARP v20 authorization lease.") from exc
        payload = {
            "schema_version": "midogpp_harp_stage90_authorization_lease_v20",
            "experiment_id": EXPERIMENT_ID, "execution_revision": EXECUTION_REVISION,
            "status": "CLAIMED_IN_PROGRESS", "process_id": os.getpid(),
            "admission_hash": admission, "amendment_sha256": authorization.amendment_sha256,
            "amendment_hash": authorization.amendment_hash,
            "input_binding_hash": authorization.input_binding_hash,
            "scientific_contract_hash": authorization.scientific_contract_hash,
            "workspace_registration_execution_contract_hash": authorization.workspace_registration_execution_contract_hash,
            "source_snapshot_schema": authorization.source_snapshot_schema,
            "source_snapshot_manifest_sha256": authorization.source_snapshot_manifest_sha256,
            "source_snapshot_tree_sha256": authorization.source_snapshot_tree_sha256,
            "source_snapshot_member_count": authorization.source_snapshot_member_count,
            "authorization_scope": AUTHORIZATION_SCOPE, "authorization_exhausted": True,
            "recovery_allowed": True,
            "recovery_scope": "same_active_lease_label_free_only",
            "output_deletion_restores_authority": False,
        }
        sealed = {**payload, "lease_hash": canonical_hash(payload)}
        atomic_json(root / "lease.json", sealed)
        return HarpV20AuthorizationLease(root, str(sealed["lease_hash"]), os.getpid())


def _reload_canonical_authorization(repo_root: Path) -> HarpV20Authorization:
    """Re-authenticate every live authority-bearing surface at claim time."""

    from ....workspace.runtime import MidogppWorkspace, WorkspaceError
    from .config import load_config

    boundary = RepositoryBoundary.open(repo_root)
    config_path = boundary.member(
        WORKSPACE_CONFIG_RELATIVE_PATH,
        label="canonical authorization config",
        kind="file",
    )
    amendment_path = boundary.member(
        WORKSPACE_AMENDMENT_RELATIVE_PATH,
        label="canonical execution amendment",
        kind="file",
    )
    config = load_config(config_path)
    expected = config.expected_execution_amendment_sha256
    registry_path = boundary.member(
        WORKSPACE_REGISTRY_RELATIVE_PATH,
        label="registry",
        kind="file",
    )
    catalog_path = boundary.member(
        WORKSPACE_ARTIFACT_CATALOG_RELATIVE_PATH,
        label="artifact catalog",
        kind="file",
    )
    if (
        not config.execution_authorized
        or type(expected) is not str
        or sha256_file(amendment_path) != expected
    ):
        raise ProtocolError("HARP v20 canonical execution authority is inactive.")
    amendment = validate_execution_amendment_payload(
        read_json(amendment_path),
        config,
        repo_root=boundary.resolved_root,
    )
    try:
        receipt = MidogppWorkspace.load(
            boundary.resolved_root
        ).validate_preparation_authority(EXPERIMENT_ID)
    except WorkspaceError as exc:
        raise ProtocolError(
            "HARP v20 canonical registry authority failed authentication."
        ) from exc
    if (
        receipt is None
        or receipt.config_path != config_path
        or receipt.config_sha256 != sha256_file(config_path)
        or receipt.authority_path != amendment_path
        or receipt.authority_sha256 != expected
        or receipt.workspace_registration_contract_hash
        != amendment.workspace_registration_execution_contract_hash
        or receipt.registry_path != registry_path
        or receipt.registry_sha256 != sha256_file(registry_path)
        or receipt.artifact_catalog_path != catalog_path
        or receipt.artifact_catalog_sha256 != sha256_file(catalog_path)
    ):
        raise ProtocolError("HARP v20 canonical registry authority drifted.")
    return HarpV20Authorization(
        amendment_path=amendment_path,
        amendment_sha256=expected,
        amendment_hash=amendment.amendment_hash,
        input_binding_hash=amendment.input_binding_hash,
        scientific_contract_hash=amendment.scientific_contract_hash,
        workspace_registration_execution_contract_hash=(
            amendment.workspace_registration_execution_contract_hash
        ),
        source_snapshot_schema=amendment.source_snapshot_schema,
        source_snapshot_manifest_sha256=amendment.source_snapshot_manifest_sha256,
        source_snapshot_tree_sha256=amendment.source_snapshot_tree_sha256,
        source_snapshot_member_count=amendment.source_snapshot_member_count,
    )


def validate_active_recovery_surface(
    config: HarpStage90V20Config,
    amendment: HarpV20ExecutionAmendment,
    *,
    repo_root: Path,
) -> Mapping[str, object]:
    """Validate the only restartable state: same lease, before label access."""

    if type(config) is not HarpStage90V20Config or type(amendment) is not HarpV20ExecutionAmendment:
        raise ProtocolError("HARP v20 recovery authority is untyped.")
    lease = lease_path(repo_root)
    raw = _validate_active_lease(
        lease,
        amendment_sha256=str(config.expected_execution_amendment_sha256),
        amendment_hash=amendment.amendment_hash,
        input_binding_hash=amendment.input_binding_hash,
    )
    from .execution.admission import validate_pristine_or_label_free_recovery

    output_root = repo_root / WORKSPACE_OUTPUT_CANONICAL_PATH
    state = validate_pristine_or_label_free_recovery(
        output_root,
        admission_hash=str(raw["admission_hash"]),
    )
    if state not in {
        "PRISTINE",
        "ACTIVE_LEASE_PREJOURNAL_RECOVERY",
        "LABEL_FREE_RECOVERY",
    }:
        raise ProtocolError("HARP v20 active lease lacks its exact label-free output state.")
    return raw


def finalize_authorization(
    lease: HarpV20AuthorizationLease, *, status: str, error: str | None = None,
) -> HarpV20AuthorizationLease:
    if type(lease) is not HarpV20AuthorizationLease or status not in {"COMPLETE_EXHAUSTED", "FAILED_EXHAUSTED"}:
        raise ProtocolError("HARP v20 lease finalization is invalid.")
    raw = read_json(lease.root / "lease.json")
    if raw.get("lease_hash") != lease.lease_hash:
        raise ProtocolError("HARP v20 authorization lease was replaced.")
    base = {key: value for key, value in raw.items() if key not in {"lease_hash", "status"}}
    base.update({"status": status, "prior_lease_hash": lease.lease_hash,
                 "finalized_by_process_id": os.getpid(),
                 "error": None if error is None else error[:2000]})
    final_hash = canonical_hash(base)
    atomic_json(lease.root / "lease.json", {**base, "lease_hash": final_hash})
    return HarpV20AuthorizationLease(lease.root, final_hash, lease.process_id)


def _validate_active_lease(
    root: Path,
    *,
    amendment_sha256: str,
    amendment_hash: str,
    input_binding_hash: str,
    admission_hash: str | None = None,
) -> Mapping[str, object]:
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError("HARP v20 authorization lease is unsafe.")
    raw = read_json(root / "lease.json")
    if (
        raw.get("schema_version") != "midogpp_harp_stage90_authorization_lease_v20"
        or raw.get("experiment_id") != EXPERIMENT_ID
        or raw.get("execution_revision") != EXECUTION_REVISION
        or raw.get("status") != "CLAIMED_IN_PROGRESS"
        or raw.get("recovery_allowed") is not True
        or raw.get("recovery_scope") != "same_active_lease_label_free_only"
        or raw.get("amendment_sha256") != amendment_sha256
        or raw.get("amendment_hash") != amendment_hash
        or raw.get("input_binding_hash") != input_binding_hash
        or (admission_hash is not None and raw.get("admission_hash") != admission_hash)
        or type(raw.get("process_id")) is not int
        or type(raw.get("lease_hash")) is not str
        or raw.get("lease_hash")
        != canonical_hash({key: value for key, value in raw.items() if key != "lease_hash"})
    ):
        raise ProtocolError("HARP v20 single-use authorization is exhausted.")
    return raw


def _process_is_alive(process_id: int) -> bool:
    if process_id <= 0:
        return False
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


__all__ = (
    "AUTHORIZATION_BASIS", "EXECUTION_AMENDMENT_ARTIFACT_ID",
    "EXECUTION_AMENDMENT_FILENAME", "HarpV20Authorization", "HarpV20AuthorizationLease",
    "HarpV20ExecutionAmendment", "WORKSPACE_AMENDMENT_RELATIVE_PATH",
    "WORKSPACE_CONFIG_RELATIVE_PATH", "WORKSPACE_OUTPUT_CANONICAL_PATH",
    "WORKSPACE_PREPARATION_AUTHORITY_GATE", "authorization_input_binding",
    "canonical_execution_amendment_payload", "claim_authorization", "finalize_authorization",
    "lease_path", "load_authorization", "scientific_contract_payload",
    "validate_execution_amendment_payload", "validate_workspace_registration_execution_projection",
    "validate_activation_metadata", "validate_active_recovery_surface",
    "workspace_registration_execution_contract",
)
