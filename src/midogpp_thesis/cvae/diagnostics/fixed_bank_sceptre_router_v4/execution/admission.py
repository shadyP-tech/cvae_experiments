"""Read-only launch admission for the one-shot SCEPTRE v4 execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from ....protocol import ProtocolError
from ..config import SceptreV4Config
from ..identity import EXPERIMENT_ID, canonical_hash, require_sha256
from ..protocol import validate_protocol_payload
from ..source_seal import validate_source_snapshot
from .authorization_lease import assert_authorization_unclaimed
from .inputs import ValidatedInputs, load_validated_inputs
from .scratch import ScratchLease, assert_scratch_absent, select_scratch
from .worker_runtime import (
    run_gpu_worker_runtime_smoke,
    validate_worker_runtime_smoke,
)
from .workspace_inputs import (
    validate_active_workspace_binding,
    validate_workspace_provenance,
)
from .workstation import run_workstation_preflight


InputLoader = Callable[[object], ValidatedInputs]


@dataclass(frozen=True, slots=True)
class ExecutionAdmission:
    artifact_root: Path
    scratch: ScratchLease
    input_binding_hash: str
    workspace_binding_hash: str
    workspace_provenance_hash: str
    workstation_preflight_hash: str
    worker_runtime_smoke_hash: str
    source_snapshot_manifest_sha256: str
    source_snapshot_tree_sha256: str
    execution_amendment_sha256: str
    admission_hash: str = ""

    def __post_init__(self) -> None:
        if not self.artifact_root.is_absolute() or not isinstance(
            self.scratch, ScratchLease
        ):
            raise ProtocolError("SCEPTRE v4 admission paths drifted.")
        for value, role in (
            (self.input_binding_hash, "input binding"),
            (self.workspace_binding_hash, "workspace binding"),
            (self.workspace_provenance_hash, "workspace provenance"),
            (self.workstation_preflight_hash, "workstation preflight"),
            (self.worker_runtime_smoke_hash, "worker smoke"),
            (self.source_snapshot_manifest_sha256, "source manifest"),
            (self.source_snapshot_tree_sha256, "source tree"),
            (self.execution_amendment_sha256, "execution amendment"),
        ):
            require_sha256(value, role)
        expected = canonical_hash(self._payload_without_hash())
        if self.admission_hash and self.admission_hash != expected:
            raise ProtocolError("SCEPTRE v4 admission hash drifted.")
        object.__setattr__(self, "admission_hash", expected)

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": "sceptre_v4_execution_admission_v1",
            "experiment_id": EXPERIMENT_ID,
            "artifact_root": str(self.artifact_root),
            "scratch_root": str(self.scratch.root),
            "scratch_role": self.scratch.role,
            "input_binding_hash": self.input_binding_hash,
            "workspace_binding_hash": self.workspace_binding_hash,
            "workspace_provenance_hash": self.workspace_provenance_hash,
            "workstation_preflight_hash": self.workstation_preflight_hash,
            "worker_runtime_smoke_hash": self.worker_runtime_smoke_hash,
            "source_snapshot_manifest_sha256": (
                self.source_snapshot_manifest_sha256
            ),
            "source_snapshot_tree_sha256": self.source_snapshot_tree_sha256,
            "execution_amendment_sha256": self.execution_amendment_sha256,
            "all_gates_read_only": True,
            "authorization_lease_claimed": False,
            "target_labels_opened": False,
            "filesystem_mutations": 0,
        }

    def to_payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {**self._payload_without_hash(), "admission_hash": self.admission_hash}
        )


def admit_execution(
    config: SceptreV4Config,
    *,
    artifact_root: str | Path,
    input_loader: InputLoader = load_validated_inputs,
    workspace_binding_loader: Callable[[object], Mapping[str, object]] = (
        validate_active_workspace_binding
    ),
    workspace_provenance_loader: Callable[
        [Path, object], Mapping[str, object]
    ] = validate_workspace_provenance,
    workstation_preflight_loader: Callable[..., Mapping[str, object]] = (
        run_workstation_preflight
    ),
    worker_smoke_loader: Callable[[], Mapping[str, object]] = (
        run_gpu_worker_runtime_smoke
    ),
) -> tuple[ExecutionAdmission, ValidatedInputs, Mapping[str, object], Mapping[str, object]]:
    """Run every costly/read-only gate before the irreversible lease mkdir."""

    if (
        not isinstance(config, SceptreV4Config)
        or config.experiment_id != EXPERIMENT_ID
        or config.execution_authorized is not True
        or config.experiment_id != str(config.protocol.get("experiment_id"))
    ):
        raise ProtocolError("SCEPTRE v4 executable config is not authorized.")
    validate_protocol_payload(config.protocol)
    expected_source = {
        key: config.source_provenance[key]
        for key in (
            "source_snapshot_schema",
            "source_snapshot_manifest_sha256",
            "source_snapshot_tree_sha256",
            "source_snapshot_member_count",
            "source_snapshot_member_pattern",
            "source_snapshot_excludes_bytecode_and_cache",
        )
    }
    validate_source_snapshot(expected_source)
    root = Path(artifact_root).resolve()
    _assert_pristine_output(root)
    assert_authorization_unclaimed()
    scratch = select_scratch(root, config.runtime)
    assert_scratch_absent(scratch)

    workspace_binding = dict(workspace_binding_loader(config))
    # The input loader authenticates the parent/execution amendment before it
    # opens any protected source-inner or consumed-test bytes.  Provenance
    # replay may hash those inputs, so it must follow that authority edge.
    validated = input_loader(config)
    if not isinstance(validated, ValidatedInputs):
        raise ProtocolError("SCEPTRE v4 input loader returned an untyped bundle.")
    workspace_provenance = dict(workspace_provenance_loader(root, config))
    preflight = dict(
        workstation_preflight_loader(
            root,
            scratch.root.parent,
            runtime=config.runtime,
        )
    )
    worker_smoke = dict(validate_worker_runtime_smoke(worker_smoke_loader()))
    input_binding = {
        "schema_version": "sceptre_v4_admitted_input_binding_v1",
        "config_hash": config.config_hash,
        "bank_lock_hash": config.expected_bank_lock_hash,
        "cache_binding_hash": validated.frame.cache_binding_hash,
        "test_cache_content_hash": config.expected_test_cache_content_hash,
        "test_cache_row_order_hash": config.expected_test_cache_row_order_hash,
        "generation_lock_hash": validated.generation_lock.generation_lock_hash,
        "source_inner_amendment_sha256": (
            validated.source_inner.amendment_sha256
        ),
        "execution_amendment_sha256": (
            config.expected_execution_amendment_sha256
        ),
        "parent_ledger_sha256": config.expected_test_consumption_ledger_sha256,
        "manifest_sha256": config.expected_manifest_sha256,
        "input_artifact_ids": list(config.input_artifact_ids),
        "target_labels_opened": False,
    }
    admission = ExecutionAdmission(
        artifact_root=root,
        scratch=scratch,
        input_binding_hash=canonical_hash(input_binding),
        workspace_binding_hash=canonical_hash(workspace_binding),
        workspace_provenance_hash=canonical_hash(workspace_provenance),
        workstation_preflight_hash=canonical_hash(preflight),
        worker_runtime_smoke_hash=str(worker_smoke["worker_runtime_smoke_hash"]),
        source_snapshot_manifest_sha256=str(
            config.source_provenance["source_snapshot_manifest_sha256"]
        ),
        source_snapshot_tree_sha256=str(
            config.source_provenance["source_snapshot_tree_sha256"]
        ),
        execution_amendment_sha256=config.expected_execution_amendment_sha256,
    )
    runtime = MappingProxyType(
        {
            "workstation_preflight": preflight,
            "worker_runtime_smoke": worker_smoke,
        }
    )
    return admission, validated, MappingProxyType(input_binding), runtime


def _assert_pristine_output(root: Path) -> None:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ProtocolError("SCEPTRE v4 prepared output root is absent or unsafe.")
    allowed_prepared_files = {
        "config.resolved.yaml",
        "provenance/input_artifacts.json",
    }
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ProtocolError("SCEPTRE v4 prepared output contains a symlink.")
        if path.is_file() and path.relative_to(root).as_posix() not in allowed_prepared_files:
            raise ProtocolError("SCEPTRE v4 output contains prior state or scientific bytes.")


__all__ = ("ExecutionAdmission", "admit_execution")
