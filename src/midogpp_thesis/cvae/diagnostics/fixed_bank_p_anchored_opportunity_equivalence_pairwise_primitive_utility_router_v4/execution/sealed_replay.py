"""Read-only replay of the issued OE-PPUR v4 preparation commitments.

The preflight was produced while amendment input #7 was absent.  A launch
therefore cannot call the pristine-surface preflight again after that input is
issued.  Instead this module rebuilds the same candidate from live bytes,
checks the historical receipt exactly, and admits the prospective envelope
without creating the output, lease, or scratch surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

from ....protocol import ProtocolError
from ...oe_ppur_v4_preparation.hashing import bytes_sha256
from ...oe_ppur_v4_preparation.host import capture_workstation_topology
from ...oe_ppur_v4_preparation.inputs import inventory_existing_inputs
from ...oe_ppur_v4_preparation.snapshot import capture_workspace_snapshot
from ...oe_ppur_v4_preparation.validation import (
    PrepublicationValidationReceipt,
    PublicationSurfaceObservation,
    observe_publication_surfaces,
)
from ...oe_ppur_v4_preparation.workspace import (
    DEFAULT_SCRATCH_ROOT,
    WorkspacePreparationContext,
    build_workspace_preparation_context,
)
from ..admission import SealedEnvelopeAdmission
from ..config import (
    ResolvedV4ConfigBundle,
    RouterV4Config,
    build_workspace_sealed_config,
)
from ..hashing import canonical_bytes, canonical_hash, require_sha256
from .authority import (
    ExecutionLaunchAuthority,
    LoadedExecutionLaunchAuthority,
)
from .inputs import (
    SevenInputContractReceipt,
    build_runtime_seven_input_contract,
    resolved_inputs_from_inventory,
)
from .replay_contract import (
    ReplayAdmissionContract,
    build_launch_authority_from_contract,
    build_replay_admission_contract,
    validate_contract_launch_authority,
)


@dataclass(frozen=True, slots=True)
class SealedExecutionReplay:
    context: WorkspacePreparationContext
    preflight_path: Path
    preflight_file_sha256: str
    preflight_payload: Mapping[str, object]
    sealed_config: RouterV4Config
    input_contract: SevenInputContractReceipt
    envelope_admission: SealedEnvelopeAdmission
    receipt_hash: str = field(init=False)
    admission_contract: ReplayAdmissionContract = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.context) is not WorkspacePreparationContext
            or type(self.sealed_config) is not RouterV4Config
            or type(self.input_contract) is not SevenInputContractReceipt
            or type(self.envelope_admission) is not SealedEnvelopeAdmission
            or not isinstance(self.preflight_payload, Mapping)
        ):
            raise ProtocolError("OE-PPUR v4 sealed execution replay is untyped.")
        path = Path(self.preflight_path)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ProtocolError("OE-PPUR v4 preflight receipt path is unsafe.")
        object.__setattr__(self, "preflight_path", path)
        object.__setattr__(
            self,
            "preflight_file_sha256",
            require_sha256(self.preflight_file_sha256, "preflight receipt file"),
        )
        if (
            self.envelope_admission.config != self.sealed_config
            or self.input_contract.receipt_hash
            != self.sealed_config.seven_input_contract_hash
        ):
            raise ProtocolError("OE-PPUR v4 replay contract lineage drifted.")
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))
        candidate = self.context.candidate
        topology = candidate.plan.topology
        object.__setattr__(
            self,
            "admission_contract",
            build_replay_admission_contract(
                sealed_config=self.sealed_config,
                input_bindings=self.input_contract.resolved_inputs,
                repository_root=self.context.repository_root,
                preflight_path=self.preflight_path,
                artifact_root=topology.output_root,
                scratch_root=topology.scratch_root,
                lease_path=topology.lease_path,
                amendment_parent=topology.amendment_path.parent,
                resolved_config_path=topology.resolved_config_path,
                input_manifest_path=topology.input_manifest_path,
                final_envelope_path=topology.envelope_path,
                workspace_snapshot_sha256=candidate.plan.workspace.snapshot_hash,
                workspace_plan_sha256=candidate.plan.plan_hash,
                authorization_amendment_sha256=bytes_sha256(
                    candidate.amendment_raw
                ),
                final_envelope_sha256=bytes_sha256(candidate.envelope_raw),
                seven_input_inventory_sha256=(
                    candidate.envelope.inputs.inventory_hash
                ),
                topology_contract_sha256=topology.contract_hash,
                scientific_seals_sha256=candidate.plan.scientific.descriptor_hash,
                scientific_source_seal_sha256=(
                    candidate.plan.scientific.scientific_seal_sha256
                ),
                lifecycle_seal_sha256=(
                    candidate.plan.scientific.lifecycle_seal_sha256
                ),
                workstation_topology_sha256=(
                    candidate.plan.workstation.receipt_hash
                ),
                preflight_file_sha256=self.preflight_file_sha256,
                resolved_input_contract_sha256=(
                    self.input_contract.resolved_location_hash
                ),
                envelope_admission_sha256=self.envelope_admission.receipt_hash,
                input_manifest_file_sha256=(
                    candidate.envelope.realized_templates.input_manifest_sha256
                ),
                sealed_replay_receipt_hash=self.receipt_hash,
            ),
        )

    def _payload(self) -> dict[str, object]:
        candidate = self.context.candidate
        return {
            "schema_version": "oe_ppur_v4_sealed_execution_replay_v1",
            "candidate_hash": candidate.candidate_hash,
            "preflight_file_sha256": self.preflight_file_sha256,
            "workspace_snapshot_sha256": candidate.plan.workspace.snapshot_hash,
            "workspace_plan_sha256": candidate.plan.plan_hash,
            "authorization_amendment_sha256": bytes_sha256(
                candidate.amendment_raw
            ),
            "final_envelope_sha256": bytes_sha256(candidate.envelope_raw),
            "seven_input_inventory_sha256": candidate.envelope.inputs.inventory_hash,
            "topology_contract_sha256": candidate.plan.topology.contract_hash,
            "scientific_seals_sha256": candidate.plan.scientific.descriptor_hash,
            "lifecycle_seal_sha256": candidate.plan.scientific.lifecycle_seal_sha256,
            "workstation_topology_sha256": candidate.plan.workstation.receipt_hash,
            "resolved_input_contract_sha256": self.input_contract.resolved_location_hash,
            "envelope_admission_sha256": self.envelope_admission.receipt_hash,
            "filesystem_mutation_performed": False,
            "target_labels_opened": False,
            "launch_authority_consumed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def replay_sealed_execution(
    repository_root: str | Path,
    *,
    preflight_receipt_path: str | Path,
    scratch_root: str | Path = DEFAULT_SCRATCH_ROOT,
    host_id: str | None = None,
) -> SealedExecutionReplay:
    """Rebuild and authenticate the amendment-issued, prelaunch state."""

    context = build_workspace_preparation_context(
        repository_root,
        scratch_root=scratch_root,
        host_id=host_id,
    )
    candidate = context.candidate
    preflight_path = _external_regular_path(
        preflight_receipt_path,
        context=context,
        role="preflight receipt",
    )
    preflight_raw = _stable_read(preflight_path, role="preflight receipt")
    expected_receipt = _historical_prepublication_receipt(context)
    expected_payload = _historical_preflight_payload(context, expected_receipt)
    observed_payload = _decode_unique_object(preflight_raw, role="preflight receipt")
    if observed_payload != expected_payload:
        raise ProtocolError("OE-PPUR v4 historical preflight receipt drifted.")
    if preflight_raw != canonical_bytes(expected_payload) + b"\n":
        raise ProtocolError("OE-PPUR v4 historical preflight bytes are not canonical.")

    _replay_live_commitments(context)
    surfaces = observe_publication_surfaces(candidate.plan)
    expected_surfaces = PublicationSurfaceObservation(
        amendment_exists=True,
        amendment_sha256=bytes_sha256(candidate.amendment_raw),
        output_root_exists=False,
        envelope_exists=False,
        envelope_sha256=None,
        commit_marker_exists=False,
        commit_marker_sha256=None,
        lease_exists=False,
        scratch_root_exists=False,
        scratch_receipts_exist=False,
        topology_receipt_exists=False,
    )
    if surfaces != expected_surfaces:
        raise ProtocolError("OE-PPUR v4 amendment-issued surfaces are not pristine.")
    amendment_raw = _stable_read(
        candidate.plan.topology.amendment_path,
        role="authorization amendment",
    )
    if amendment_raw != candidate.amendment_raw:
        raise ProtocolError("OE-PPUR v4 issued amendment bytes drifted.")

    sealed_config = build_workspace_sealed_config(
        workspace_plan_sha256=candidate.plan.plan_hash,
        authorization_amendment_sha256=bytes_sha256(amendment_raw),
    )
    bindings = resolved_inputs_from_inventory(candidate.envelope.inputs)
    input_contract = build_runtime_seven_input_contract(bindings)
    admission = SealedEnvelopeAdmission(
        config=sealed_config,
        workspace_snapshot_sha256=candidate.plan.workspace.snapshot_hash,
        workspace_plan_sha256=candidate.plan.plan_hash,
        authorization_amendment_sha256=bytes_sha256(amendment_raw),
        final_envelope_sha256=bytes_sha256(candidate.envelope_raw),
        direct_input_artifact_ids=tuple(row.artifact_id for row in bindings),
        resolved_paths=tuple(row.path for row in bindings),
        topology_contract_sha256=candidate.plan.topology.contract_hash,
    )
    return SealedExecutionReplay(
        context=context,
        preflight_path=preflight_path,
        preflight_file_sha256=hashlib.sha256(preflight_raw).hexdigest(),
        preflight_payload=observed_payload,
        sealed_config=sealed_config,
        input_contract=input_contract,
        envelope_admission=admission,
    )


def build_launch_authority_from_replay(
    replay: SealedExecutionReplay,
    *,
    authorization_phrase: str,
    authorization_nonce: str | None = None,
) -> ExecutionLaunchAuthority:
    if type(replay) is not SealedExecutionReplay:
        raise ProtocolError("OE-PPUR v4 launch-authority replay is untyped.")
    return build_launch_authority_from_contract(
        replay.admission_contract,
        authorization_phrase=authorization_phrase,
        authorization_nonce=authorization_nonce,
    )


def validate_loaded_launch_authority(
    replay: SealedExecutionReplay,
    loaded: LoadedExecutionLaunchAuthority,
) -> LoadedExecutionLaunchAuthority:
    if (
        type(replay) is not SealedExecutionReplay
        or type(loaded) is not LoadedExecutionLaunchAuthority
    ):
        raise ProtocolError("OE-PPUR v4 launch-authority admission is untyped.")
    return validate_contract_launch_authority(replay.admission_contract, loaded)


def build_resolved_config_bundle(
    replay: SealedExecutionReplay,
    loaded: LoadedExecutionLaunchAuthority,
) -> ResolvedV4ConfigBundle:
    validate_loaded_launch_authority(replay, loaded)
    contract = replay.admission_contract
    topology = contract.paths
    return ResolvedV4ConfigBundle(
        config=contract.sealed_config,
        source_path=topology.resolved_config_path,
        artifact_root=topology.artifact_root,
        input_bindings=contract.input_bindings,
        input_manifest_path=topology.input_manifest_path,
        final_envelope_path=topology.final_envelope_path,
        workspace_snapshot_sha256=(contract.authority.workspace_snapshot_sha256),
        workspace_plan_sha256=contract.authority.workspace_plan_sha256,
        final_envelope_sha256=contract.authority.final_envelope_sha256,
        execution_launch_authority_sha256=loaded.file_sha256,
    )


def _historical_prepublication_receipt(
    context: WorkspacePreparationContext,
) -> PrepublicationValidationReceipt:
    candidate = context.candidate
    return PrepublicationValidationReceipt(
        candidate_hash=candidate.candidate_hash,
        workspace_snapshot_hash=candidate.plan.workspace.snapshot_hash,
        existing_input_inventory_hash=candidate.plan.existing_inputs.inventory_hash,
        plan_hash=candidate.plan.plan_hash,
        amendment_sha256=bytes_sha256(candidate.amendment_raw),
        final_envelope_sha256=bytes_sha256(candidate.envelope_raw),
        commit_marker_sha256=bytes_sha256(candidate.commit_marker_raw),
    )


def _historical_preflight_payload(
    context: WorkspacePreparationContext,
    receipt: PrepublicationValidationReceipt,
) -> dict[str, object]:
    return {
        **receipt.to_payload(),
        "receipt_hash": receipt.receipt_hash,
        "repository_root": context.repository_root.as_posix(),
        "pre_amendment_plan": context.candidate.plan.to_payload(),
        "pre_amendment_plan_sha256": context.candidate.plan.plan_hash,
        "predecessor_preservation_witness_sha256": (
            context.candidate.plan.predecessor.witness_hash
        ),
        "launch_authorized": False,
    }


def _replay_live_commitments(context: WorkspacePreparationContext) -> None:
    plan = context.candidate.plan
    if capture_workspace_snapshot(context.seal_spec) != plan.workspace:
        raise ProtocolError("OE-PPUR v4 workspace drifted after preflight.")
    if inventory_existing_inputs(context.input_specs) != plan.existing_inputs:
        raise ProtocolError("OE-PPUR v4 direct inputs drifted after preflight.")
    workstation = capture_workstation_topology(
        artifact_parent=plan.topology.canonical_output_parent,
        scratch_root=plan.topology.scratch_root,
    )
    if workstation != plan.workstation:
        raise ProtocolError("OE-PPUR v4 workstation topology drifted after preflight.")


def _external_regular_path(
    value: str | Path,
    *,
    context: WorkspacePreparationContext,
    role: str,
) -> Path:
    path = Path(value)
    candidate = Path(os.path.abspath(path))
    plan = context.candidate.plan
    forbidden_roots = (
        context.repository_root,
        plan.topology.output_root,
        plan.topology.lease_path,
        plan.topology.scratch_root,
        plan.topology.amendment_path.parent,
    )
    if (
        not path.is_absolute()
        or path != candidate
        or path.is_symlink()
        or not path.is_file()
        or any(path == root or path.is_relative_to(root) for root in forbidden_roots)
    ):
        raise ProtocolError(f"OE-PPUR v4 {role} path is unsafe.")
    return path


def _stable_read(path: Path, *, role: str) -> bytes:
    try:
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise ProtocolError(f"OE-PPUR v4 {role} is unavailable.") from exc
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or before.st_size != len(raw)
    ):
        raise ProtocolError(f"OE-PPUR v4 {role} changed while read.")
    return raw


def _decode_unique_object(raw: bytes, *, role: str) -> dict[str, object]:
    def unique(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise ValueError(key)
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolError(f"OE-PPUR v4 {role} is not unique JSON.") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"OE-PPUR v4 {role} is not a JSON object.")
    return value


__all__ = (
    "SealedExecutionReplay",
    "build_launch_authority_from_replay",
    "build_resolved_config_bundle",
    "replay_sealed_execution",
    "validate_loaded_launch_authority",
)
