"""Mutation-free full admission rehearsal for the OE-PPUR v4 real runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ....protocol import ProtocolError
from ..authorization_lease import assert_authorization_unclaimed
from ..capacity_preflight import preflight_resource_capacity
from ..execution.authority import load_execution_launch_authority
from ..execution.sealed_replay import (
    build_resolved_config_bundle,
    replay_sealed_execution,
)
from ..execution.services import ServicePreflightRequest
from ..hashing import canonical_hash, require_sha256
from ..physical import (
    load_label_free_test_frame,
    load_validated_upstream_inputs,
    project_workstation_topology,
)
from ..run_admission import admit_seven_input_execution
from ..service_factory import prepare_canonical_scientific_service_factory
from ..source_seal import build_source_seal
from ..source_supervision import load_immutable_source_training_surface


@dataclass(frozen=True, slots=True)
class RealLaunchDryRunReceipt:
    sealed_replay_receipt_hash: str
    seven_input_admission_hash: str
    launch_authority_file_sha256: str
    source_seal_hash: str
    source_surface_hash: str
    upstream_receipt_hash: str
    label_free_test_frame_hash: str
    workstation_receipt_hash: str
    resource_capacity_receipt_hash: str
    service_factory_identity_hash: str
    service_preflight_receipt_hash: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for role in (
            "sealed_replay_receipt_hash",
            "seven_input_admission_hash",
            "launch_authority_file_sha256",
            "source_seal_hash",
            "source_surface_hash",
            "upstream_receipt_hash",
            "label_free_test_frame_hash",
            "workstation_receipt_hash",
            "resource_capacity_receipt_hash",
            "service_factory_identity_hash",
            "service_preflight_receipt_hash",
        ):
            object.__setattr__(
                self, role, require_sha256(getattr(self, role), role.replace("_", " "))
            )
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_real_launch_dry_run_v1",
            "status": "PASS_READ_ONLY_NOT_LAUNCHED",
            "sealed_replay_receipt_hash": self.sealed_replay_receipt_hash,
            "seven_input_admission_hash": self.seven_input_admission_hash,
            "launch_authority_file_sha256": self.launch_authority_file_sha256,
            "source_seal_hash": self.source_seal_hash,
            "source_surface_hash": self.source_surface_hash,
            "upstream_receipt_hash": self.upstream_receipt_hash,
            "label_free_test_frame_hash": self.label_free_test_frame_hash,
            "workstation_receipt_hash": self.workstation_receipt_hash,
            "resource_capacity_receipt_hash": self.resource_capacity_receipt_hash,
            "service_factory_identity_hash": self.service_factory_identity_hash,
            "service_preflight_receipt_hash": self.service_preflight_receipt_hash,
            "scientific_input_count": 7,
            "launch_authority_is_scientific_input": False,
            "lease_claimed": False,
            "filesystem_mutation_performed": False,
            "target_labels_opened": False,
            "experiment_launched": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def dry_run_real_launch(
    repository_root: str | Path,
    *,
    preflight_receipt_path: str | Path,
    launch_authority_path: str | Path,
    scratch_root: str | Path,
    host_id: str | None = None,
) -> RealLaunchDryRunReceipt:
    replay = replay_sealed_execution(
        repository_root,
        preflight_receipt_path=preflight_receipt_path,
        scratch_root=scratch_root,
        host_id=host_id,
    )
    loaded = load_execution_launch_authority(launch_authority_path)
    bundle = build_resolved_config_bundle(replay, loaded)
    seal = build_source_seal(replay.context.repository_root)
    source_surface = load_immutable_source_training_surface(
        bundle.input_bindings[2].path
    )
    admission = admit_seven_input_execution(
        bundle,
        replay=replay.admission_contract,
        launch_authority=loaded,
        source_seal=seal,
        source_surface=source_surface,
        scratch_root=scratch_root,
    )
    upstream = load_validated_upstream_inputs(
        bundle.input_bindings[0].path,
        bundle.input_bindings[1].path,
    )
    frame = load_label_free_test_frame(bundle.input_bindings[3].path)
    workstation = project_workstation_topology(replay.context.candidate.plan.workstation)
    capacity = preflight_resource_capacity(admission.artifact_root, admission.scratch_root)
    factory = prepare_canonical_scientific_service_factory(
        bundle,
        source_seal=seal,
        source_surface=source_surface,
        admission=admission,
    )
    service = factory.build()
    service_preflight = service.preflight(
        ServicePreflightRequest(
            seven_input_contract_hash=bundle.config.seven_input_contract_hash,
            protocol_hash=bundle.config.protocol_hash,
            source_seal_hash=seal.combined_source_sha256,
            workstation_receipt_hash=workstation.receipt_hash,
        )
    )
    assert_authorization_unclaimed(admission.artifact_root, admission.scratch_root)
    return RealLaunchDryRunReceipt(
        sealed_replay_receipt_hash=replay.receipt_hash,
        seven_input_admission_hash=admission.receipt_hash,
        launch_authority_file_sha256=loaded.file_sha256,
        source_seal_hash=seal.combined_source_sha256,
        source_surface_hash=source_surface.surface_hash,
        upstream_receipt_hash=upstream.receipt_hash,
        label_free_test_frame_hash=frame.frame_hash,
        workstation_receipt_hash=workstation.receipt_hash,
        resource_capacity_receipt_hash=capacity.receipt_hash,
        service_factory_identity_hash=factory.identity.receipt_hash,
        service_preflight_receipt_hash=service_preflight.receipt_hash,
    )


__all__ = ("RealLaunchDryRunReceipt", "dry_run_real_launch")
