"""Separate, explicit, single-use launch authority for OE-PPUR v4.

The workspace amendment is direct scientific input #7, but it deliberately
does not authorize execution.  This file defines the out-of-band capability
which may be issued only after that amendment and its prospective envelope have
been replayed.  The capability is a control-plane receipt, never an eighth
scientific input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import secrets

from ....protocol import ProtocolError
from ..hashing import canonical_bytes, canonical_hash, require_sha256
from ..identity import (
    EXPERIMENT_ID,
    LAUNCH_AUTHORIZATION_PHRASE,
    LAUNCH_AUTHORIZATION_SCOPE,
    LAUNCH_AUTHORITY_SCHEMA,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)


@dataclass(frozen=True, slots=True)
class ExecutionLaunchAuthority:
    workspace_snapshot_sha256: str
    workspace_plan_sha256: str
    authorization_amendment_sha256: str
    final_envelope_sha256: str
    seven_input_inventory_sha256: str
    topology_contract_sha256: str
    scientific_seals_sha256: str
    lifecycle_seal_sha256: str
    workstation_topology_sha256: str
    preflight_receipt_sha256: str
    authorization_nonce: str
    authorized_by: str = "user"
    authority_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for role in (
            "workspace_snapshot_sha256",
            "workspace_plan_sha256",
            "authorization_amendment_sha256",
            "final_envelope_sha256",
            "seven_input_inventory_sha256",
            "topology_contract_sha256",
            "scientific_seals_sha256",
            "lifecycle_seal_sha256",
            "workstation_topology_sha256",
            "preflight_receipt_sha256",
        ):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )
        nonce = require_sha256(self.authorization_nonce, "authorization nonce")
        if self.authorized_by != "user" or nonce == "0" * 64:
            raise ProtocolError("OE-PPUR v4 launch authority issuer drifted.")
        object.__setattr__(self, "authorization_nonce", nonce)
        object.__setattr__(self, "authority_hash", canonical_hash(self._body()))

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": LAUNCH_AUTHORITY_SCHEMA,
            "status": "AUTHORIZED_SINGLE_USE_NOT_CONSUMED",
            "experiment_id": EXPERIMENT_ID,
            "output_artifact_id": OUTPUT_ARTIFACT_ID,
            "authorization_scope": LAUNCH_AUTHORIZATION_SCOPE,
            "authorized_by": self.authorized_by,
            "execution_authorized": True,
            "consumed_test_reuse_authorized": True,
            "single_use_execution_identity": True,
            "authorization_exhausted": False,
            "cross_run_recovery_allowed": False,
            "workspace_snapshot_sha256": self.workspace_snapshot_sha256,
            "workspace_plan_sha256": self.workspace_plan_sha256,
            "authorization_amendment_sha256": self.authorization_amendment_sha256,
            "final_envelope_sha256": self.final_envelope_sha256,
            "seven_input_inventory_sha256": self.seven_input_inventory_sha256,
            "topology_contract_sha256": self.topology_contract_sha256,
            "scientific_seals_sha256": self.scientific_seals_sha256,
            "lifecycle_seal_sha256": self.lifecycle_seal_sha256,
            "workstation_topology_sha256": self.workstation_topology_sha256,
            "preflight_receipt_sha256": self.preflight_receipt_sha256,
            "authorization_phrase_sha256": hashlib.sha256(
                LAUNCH_AUTHORIZATION_PHRASE.encode("utf-8")
            ).hexdigest(),
            "authorization_nonce": self.authorization_nonce,
            "target_labels_open_only_after_durable_preterminal_attestation": True,
            "previous_stage90_operational_outputs_used": False,
            "previous_stage90_amendments_used": False,
            "previous_stage90_run_state_or_scratch_used": False,
            "fresh_evidence": False,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "may_feed_another_experiment": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._body(), "authority_hash": self.authority_hash}

    def canonical_file_bytes(self) -> bytes:
        return canonical_bytes(self.to_payload()) + b"\n"


@dataclass(frozen=True, slots=True)
class LoadedExecutionLaunchAuthority:
    authority: ExecutionLaunchAuthority
    path: Path
    file_sha256: str

    def __post_init__(self) -> None:
        path = Path(self.path)
        if (
            type(self.authority) is not ExecutionLaunchAuthority
            or not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ProtocolError("OE-PPUR v4 loaded launch authority drifted.")
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self, "file_sha256", require_sha256(self.file_sha256, "authority file")
        )


def build_execution_launch_authority(
    *,
    authorization_phrase: str,
    workspace_snapshot_sha256: str,
    workspace_plan_sha256: str,
    authorization_amendment_sha256: str,
    final_envelope_sha256: str,
    seven_input_inventory_sha256: str,
    topology_contract_sha256: str,
    scientific_seals_sha256: str,
    lifecycle_seal_sha256: str,
    workstation_topology_sha256: str,
    preflight_receipt_sha256: str,
    authorization_nonce: str | None = None,
) -> ExecutionLaunchAuthority:
    if authorization_phrase != LAUNCH_AUTHORIZATION_PHRASE:
        raise ProtocolError("OE-PPUR v4 explicit launch authorization phrase is absent.")
    return ExecutionLaunchAuthority(
        workspace_snapshot_sha256=workspace_snapshot_sha256,
        workspace_plan_sha256=workspace_plan_sha256,
        authorization_amendment_sha256=authorization_amendment_sha256,
        final_envelope_sha256=final_envelope_sha256,
        seven_input_inventory_sha256=seven_input_inventory_sha256,
        topology_contract_sha256=topology_contract_sha256,
        scientific_seals_sha256=scientific_seals_sha256,
        lifecycle_seal_sha256=lifecycle_seal_sha256,
        workstation_topology_sha256=workstation_topology_sha256,
        preflight_receipt_sha256=preflight_receipt_sha256,
        authorization_nonce=(
            secrets.token_hex(32)
            if authorization_nonce is None
            else authorization_nonce
        ),
    )


def load_execution_launch_authority(
    value: str | Path,
) -> LoadedExecutionLaunchAuthority:
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.abspath(path)) or path.is_symlink():
        raise ProtocolError("OE-PPUR v4 launch authority path is unsafe.")
    try:
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise ProtocolError("OE-PPUR v4 launch authority is unavailable.") from exc
    if (
        not path.is_file()
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ProtocolError("OE-PPUR v4 launch authority changed while read.")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("OE-PPUR v4 launch authority is not canonical JSON.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("OE-PPUR v4 launch authority payload is malformed.")
    authority = _authority_from_payload(payload)
    if raw != authority.canonical_file_bytes():
        raise ProtocolError("OE-PPUR v4 launch authority bytes are not canonical.")
    return LoadedExecutionLaunchAuthority(
        authority=authority,
        path=path,
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _authority_from_payload(payload: dict[str, object]) -> ExecutionLaunchAuthority:
    expected_keys = set(
        ExecutionLaunchAuthority(
            workspace_snapshot_sha256="1" * 64,
            workspace_plan_sha256="2" * 64,
            authorization_amendment_sha256="3" * 64,
            final_envelope_sha256="4" * 64,
            seven_input_inventory_sha256="5" * 64,
            topology_contract_sha256="6" * 64,
            scientific_seals_sha256="7" * 64,
            lifecycle_seal_sha256="8" * 64,
            workstation_topology_sha256="9" * 64,
            preflight_receipt_sha256="a" * 64,
            authorization_nonce="b" * 64,
        ).to_payload()
    )
    if set(payload) != expected_keys:
        raise ProtocolError("OE-PPUR v4 launch authority fields drifted.")
    authority = ExecutionLaunchAuthority(
        workspace_snapshot_sha256=str(payload.get("workspace_snapshot_sha256", "")),
        workspace_plan_sha256=str(payload.get("workspace_plan_sha256", "")),
        authorization_amendment_sha256=str(
            payload.get("authorization_amendment_sha256", "")
        ),
        final_envelope_sha256=str(payload.get("final_envelope_sha256", "")),
        seven_input_inventory_sha256=str(
            payload.get("seven_input_inventory_sha256", "")
        ),
        topology_contract_sha256=str(payload.get("topology_contract_sha256", "")),
        scientific_seals_sha256=str(payload.get("scientific_seals_sha256", "")),
        lifecycle_seal_sha256=str(payload.get("lifecycle_seal_sha256", "")),
        workstation_topology_sha256=str(
            payload.get("workstation_topology_sha256", "")
        ),
        preflight_receipt_sha256=str(payload.get("preflight_receipt_sha256", "")),
        authorization_nonce=str(payload.get("authorization_nonce", "")),
        authorized_by=str(payload.get("authorized_by", "")),
    )
    if payload != authority.to_payload():
        raise ProtocolError("OE-PPUR v4 launch authority semantics drifted.")
    return authority


__all__ = (
    "ExecutionLaunchAuthority",
    "LoadedExecutionLaunchAuthority",
    "build_execution_launch_authority",
    "load_execution_launch_authority",
)
