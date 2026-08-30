"""Immutable structural boundary between sealed replay and run admission.

Replay owns rebuilding the workspace-backed evidence.  Admission owns opening
the one-shot execution edge.  This module is the only object-level dependency
between them: it projects the replay into hash- and path-bound immutable data,
without exposing replay implementation helpers to admission.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from pathlib import Path

from ....protocol import ProtocolError
from ..config import RouterV4Config, validate_workspace_sealed_config
from ..hashing import canonical_hash, require_sha256
from ..identity import LAUNCH_AUTHORIZATION_PHRASE
from .authority import (
    ExecutionLaunchAuthority,
    LoadedExecutionLaunchAuthority,
    build_execution_launch_authority,
)
from .inputs import (
    ResolvedDirectInput,
    hash_resolved_input_locations,
    validate_exact_resolved_input_bindings,
)


_CONTRACT_TOKEN = object()


@dataclass(frozen=True, slots=True)
class ReplayAuthorityBinding:
    workspace_snapshot_sha256: str
    workspace_plan_sha256: str
    authorization_amendment_sha256: str
    final_envelope_sha256: str
    seven_input_inventory_sha256: str
    topology_contract_sha256: str
    scientific_seals_sha256: str
    scientific_source_seal_sha256: str
    lifecycle_seal_sha256: str
    workstation_topology_sha256: str
    preflight_file_sha256: str
    resolved_input_contract_sha256: str
    envelope_admission_sha256: str
    input_manifest_file_sha256: str
    sealed_replay_receipt_hash: str
    _factory_token: InitVar[object | None] = None
    binding_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _CONTRACT_TOKEN:
            raise ProtocolError("OE-PPUR v4 replay authority binding is untyped.")
        for role in (
            "workspace_snapshot_sha256",
            "workspace_plan_sha256",
            "authorization_amendment_sha256",
            "final_envelope_sha256",
            "seven_input_inventory_sha256",
            "topology_contract_sha256",
            "scientific_seals_sha256",
            "scientific_source_seal_sha256",
            "lifecycle_seal_sha256",
            "workstation_topology_sha256",
            "preflight_file_sha256",
            "resolved_input_contract_sha256",
            "envelope_admission_sha256",
            "input_manifest_file_sha256",
            "sealed_replay_receipt_hash",
        ):
            digest = require_sha256(getattr(self, role), role.replace("_", " "))
            if digest == "0" * 64:
                raise ProtocolError("OE-PPUR v4 replay authority contains a placeholder.")
            object.__setattr__(self, role, digest)
        object.__setattr__(self, "binding_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "workspace_snapshot_sha256": self.workspace_snapshot_sha256,
            "workspace_plan_sha256": self.workspace_plan_sha256,
            "authorization_amendment_sha256": (
                self.authorization_amendment_sha256
            ),
            "final_envelope_sha256": self.final_envelope_sha256,
            "seven_input_inventory_sha256": self.seven_input_inventory_sha256,
            "topology_contract_sha256": self.topology_contract_sha256,
            "scientific_seals_sha256": self.scientific_seals_sha256,
            "scientific_source_seal_sha256": self.scientific_source_seal_sha256,
            "lifecycle_seal_sha256": self.lifecycle_seal_sha256,
            "workstation_topology_sha256": self.workstation_topology_sha256,
            "preflight_file_sha256": self.preflight_file_sha256,
            "resolved_input_contract_sha256": self.resolved_input_contract_sha256,
            "envelope_admission_sha256": self.envelope_admission_sha256,
            "input_manifest_file_sha256": self.input_manifest_file_sha256,
            "sealed_replay_receipt_hash": self.sealed_replay_receipt_hash,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "binding_hash": self.binding_hash}


@dataclass(frozen=True, slots=True)
class ReplayPathBinding:
    repository_root: Path
    preflight_path: Path
    artifact_root: Path
    scratch_root: Path
    lease_path: Path
    amendment_parent: Path
    resolved_config_path: Path
    input_manifest_path: Path
    final_envelope_path: Path
    _factory_token: InitVar[object | None] = None
    binding_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _CONTRACT_TOKEN:
            raise ProtocolError("OE-PPUR v4 replay path binding is untyped.")
        values = tuple(
            Path(getattr(self, role))
            for role in (
                "repository_root",
                "preflight_path",
                "artifact_root",
                "scratch_root",
                "lease_path",
                "amendment_parent",
                "resolved_config_path",
                "input_manifest_path",
                "final_envelope_path",
            )
        )
        if (
            not all(path.is_absolute() for path in values)
            or any(path.is_symlink() for path in values)
            or values[1].is_relative_to(values[0])
            or values[6] != values[2] / "config.resolved.yaml"
            or values[7] != values[2] / "provenance/input_artifacts.json"
            or values[8]
            != values[2] / "preparation/final_authorization_envelope.json"
        ):
            raise ProtocolError("OE-PPUR v4 replay path binding drifted.")
        for role, path in zip(
            (
                "repository_root",
                "preflight_path",
                "artifact_root",
                "scratch_root",
                "lease_path",
                "amendment_parent",
                "resolved_config_path",
                "input_manifest_path",
                "final_envelope_path",
            ),
            values,
            strict=True,
        ):
            object.__setattr__(self, role, path)
        object.__setattr__(self, "binding_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "repository_root": self.repository_root.as_posix(),
            "preflight_path": self.preflight_path.as_posix(),
            "artifact_root": self.artifact_root.as_posix(),
            "scratch_root": self.scratch_root.as_posix(),
            "lease_path": self.lease_path.as_posix(),
            "amendment_parent": self.amendment_parent.as_posix(),
            "resolved_config_path": self.resolved_config_path.as_posix(),
            "input_manifest_path": self.input_manifest_path.as_posix(),
            "final_envelope_path": self.final_envelope_path.as_posix(),
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "binding_hash": self.binding_hash}


@dataclass(frozen=True, slots=True)
class ReplayAdmissionContract:
    sealed_config: RouterV4Config
    input_bindings: tuple[ResolvedDirectInput, ...]
    authority: ReplayAuthorityBinding
    paths: ReplayPathBinding
    _factory_token: InitVar[object | None] = None
    contract_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if (
            _factory_token is not _CONTRACT_TOKEN
            or type(self.authority) is not ReplayAuthorityBinding
            or type(self.paths) is not ReplayPathBinding
        ):
            raise ProtocolError("OE-PPUR v4 replay admission contract is untyped.")
        config = validate_workspace_sealed_config(self.sealed_config)
        bindings = validate_exact_resolved_input_bindings(self.input_bindings)
        if (
            self.authority.workspace_plan_sha256 != config.workspace_plan_sha256
            or self.authority.authorization_amendment_sha256
            != config.authorization_amendment_sha256
            or self.authority.resolved_input_contract_sha256
            != hash_resolved_input_locations(bindings)
        ):
            raise ProtocolError("OE-PPUR v4 replay admission contract drifted.")
        object.__setattr__(self, "sealed_config", config)
        object.__setattr__(self, "input_bindings", bindings)
        object.__setattr__(self, "contract_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_replay_admission_contract_v1",
            "sealed_config_contract_hash": self.sealed_config.contract_hash,
            "input_binding_hashes": [row.binding_hash for row in self.input_bindings],
            "authority": self.authority.to_payload(),
            "paths": self.paths.to_payload(),
            "target_labels_opened": False,
            "filesystem_mutation_performed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "contract_hash": self.contract_hash}


def build_replay_admission_contract(
    *,
    sealed_config: RouterV4Config,
    input_bindings: tuple[ResolvedDirectInput, ...],
    repository_root: Path,
    preflight_path: Path,
    artifact_root: Path,
    scratch_root: Path,
    lease_path: Path,
    amendment_parent: Path,
    resolved_config_path: Path,
    input_manifest_path: Path,
    final_envelope_path: Path,
    workspace_snapshot_sha256: str,
    workspace_plan_sha256: str,
    authorization_amendment_sha256: str,
    final_envelope_sha256: str,
    seven_input_inventory_sha256: str,
    topology_contract_sha256: str,
    scientific_seals_sha256: str,
    scientific_source_seal_sha256: str,
    lifecycle_seal_sha256: str,
    workstation_topology_sha256: str,
    preflight_file_sha256: str,
    resolved_input_contract_sha256: str,
    envelope_admission_sha256: str,
    input_manifest_file_sha256: str,
    sealed_replay_receipt_hash: str,
) -> ReplayAdmissionContract:
    authority = ReplayAuthorityBinding(
        workspace_snapshot_sha256=workspace_snapshot_sha256,
        workspace_plan_sha256=workspace_plan_sha256,
        authorization_amendment_sha256=authorization_amendment_sha256,
        final_envelope_sha256=final_envelope_sha256,
        seven_input_inventory_sha256=seven_input_inventory_sha256,
        topology_contract_sha256=topology_contract_sha256,
        scientific_seals_sha256=scientific_seals_sha256,
        scientific_source_seal_sha256=scientific_source_seal_sha256,
        lifecycle_seal_sha256=lifecycle_seal_sha256,
        workstation_topology_sha256=workstation_topology_sha256,
        preflight_file_sha256=preflight_file_sha256,
        resolved_input_contract_sha256=resolved_input_contract_sha256,
        envelope_admission_sha256=envelope_admission_sha256,
        input_manifest_file_sha256=input_manifest_file_sha256,
        sealed_replay_receipt_hash=sealed_replay_receipt_hash,
        _factory_token=_CONTRACT_TOKEN,
    )
    paths = ReplayPathBinding(
        repository_root=repository_root,
        preflight_path=preflight_path,
        artifact_root=artifact_root,
        scratch_root=scratch_root,
        lease_path=lease_path,
        amendment_parent=amendment_parent,
        resolved_config_path=resolved_config_path,
        input_manifest_path=input_manifest_path,
        final_envelope_path=final_envelope_path,
        _factory_token=_CONTRACT_TOKEN,
    )
    return ReplayAdmissionContract(
        sealed_config=sealed_config,
        input_bindings=input_bindings,
        authority=authority,
        paths=paths,
        _factory_token=_CONTRACT_TOKEN,
    )


def require_replay_admission_contract(value: object) -> ReplayAdmissionContract:
    """Accept the exact contract or a replay carrying that exact capability."""

    if type(value) is ReplayAdmissionContract:
        return value
    contract = getattr(value, "admission_contract", None)
    if (
        type(contract) is not ReplayAdmissionContract
        or getattr(value, "receipt_hash", None)
        != contract.authority.sealed_replay_receipt_hash
    ):
        raise ProtocolError("OE-PPUR v4 run admission is untyped.")
    return contract


def build_launch_authority_from_contract(
    contract: ReplayAdmissionContract,
    *,
    authorization_phrase: str,
    authorization_nonce: str | None = None,
) -> ExecutionLaunchAuthority:
    if type(contract) is not ReplayAdmissionContract:
        raise ProtocolError("OE-PPUR v4 launch-authority replay is untyped.")
    bound = contract.authority
    return build_execution_launch_authority(
        authorization_phrase=authorization_phrase,
        workspace_snapshot_sha256=bound.workspace_snapshot_sha256,
        workspace_plan_sha256=bound.workspace_plan_sha256,
        authorization_amendment_sha256=bound.authorization_amendment_sha256,
        final_envelope_sha256=bound.final_envelope_sha256,
        seven_input_inventory_sha256=bound.seven_input_inventory_sha256,
        topology_contract_sha256=bound.topology_contract_sha256,
        scientific_seals_sha256=bound.scientific_seals_sha256,
        lifecycle_seal_sha256=bound.lifecycle_seal_sha256,
        workstation_topology_sha256=bound.workstation_topology_sha256,
        preflight_receipt_sha256=bound.preflight_file_sha256,
        authorization_nonce=authorization_nonce,
    )


def validate_contract_launch_authority(
    contract: ReplayAdmissionContract,
    loaded: LoadedExecutionLaunchAuthority,
) -> LoadedExecutionLaunchAuthority:
    if (
        type(contract) is not ReplayAdmissionContract
        or type(loaded) is not LoadedExecutionLaunchAuthority
    ):
        raise ProtocolError("OE-PPUR v4 launch-authority admission is untyped.")
    expected = build_launch_authority_from_contract(
        contract,
        authorization_phrase=LAUNCH_AUTHORIZATION_PHRASE,
        authorization_nonce=loaded.authority.authorization_nonce,
    )
    if loaded.authority != expected:
        raise ProtocolError("OE-PPUR v4 launch authority drifted from sealed replay.")
    path = loaded.path
    forbidden_roots = (
        contract.paths.repository_root,
        contract.paths.artifact_root,
        contract.paths.lease_path,
        contract.paths.scratch_root,
        contract.paths.amendment_parent,
    )
    if any(path == root or path.is_relative_to(root) for root in forbidden_roots):
        raise ProtocolError("OE-PPUR v4 launch authority path is unsafe.")
    return loaded


__all__ = (
    "ReplayAdmissionContract",
    "ReplayAuthorityBinding",
    "ReplayPathBinding",
    "build_launch_authority_from_contract",
    "build_replay_admission_contract",
    "require_replay_admission_contract",
    "validate_contract_launch_authority",
)
