"""Read-only execution and launch admission for one-shot SCEPTRE v3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from ...protocol import ProtocolError
from .authorization_lease import assert_authorization_unclaimed
from .config import SceptreV3Config, load_config
from .experiment_contracts import INPUT_ARTIFACT_IDS
from .identity import (
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    canonical_hash,
    require_sha256,
)
from .inputs import ValidatedInputs, assert_input_fence, load_validated_inputs
from .protocol import validate_protocol_payload
from .scratch import ScratchLease, assert_scratch_absent, select_scratch
from .source_seal import validate_source_snapshot
from .workspace_inputs import (
    validate_active_workspace_binding,
    validate_workspace_provenance,
)
from .workstation import validate_workstation_payload
from .worker_runtime import validate_worker_runtime_smoke


LAUNCH_FILES = frozenset({"config.resolved.yaml", "provenance/input_artifacts.json"})
WORKSPACE_DIRECTORIES = frozenset({"manifests", "provenance", "reports", "tables"})


@dataclass(frozen=True, slots=True)
class DryRunAdmission:
    artifact_root: Path
    scratch: ScratchLease
    authorization_lease_path: Path
    config_hash: str
    source_tree_sha256: str
    cache_binding_hash: str
    admission_hash: str

    def __post_init__(self) -> None:
        if (
            not self.artifact_root.is_absolute()
            or not self.authorization_lease_path.is_absolute()
        ):
            raise ProtocolError("SCEPTRE v3 dry-run admission path drifted.")
        for role, value in (
            ("config", self.config_hash),
            ("source tree", self.source_tree_sha256),
            ("cache binding", self.cache_binding_hash),
            ("admission", self.admission_hash),
        ):
            require_sha256(value, role)


@dataclass(frozen=True, slots=True)
class RuntimeLaunchAdmission:
    """In-memory binding of all pre-lease host and worker lifecycle gates."""

    read_only_admission_hash: str
    workstation_preflight_hash: str
    workspace_provenance_hash: str
    worker_runtime_smoke_hash: str
    admission_hash: str

    def __post_init__(self) -> None:
        for role, value in (
            ("read-only admission", self.read_only_admission_hash),
            ("workstation preflight", self.workstation_preflight_hash),
            ("workspace provenance", self.workspace_provenance_hash),
            ("worker runtime smoke", self.worker_runtime_smoke_hash),
            ("runtime launch admission", self.admission_hash),
        ):
            require_sha256(value, role)


def bind_runtime_launch_admission(
    admission: DryRunAdmission,
    *,
    workstation_preflight: Mapping[str, object],
    workspace_provenance: Mapping[str, object],
    worker_runtime_smoke: Mapping[str, object],
) -> RuntimeLaunchAdmission:
    """Bind the successful repeated-worker smoke before lease creation."""

    if not isinstance(admission, DryRunAdmission):
        raise ProtocolError("SCEPTRE v3 read-only admission receipt drifted.")
    preflight = dict(workstation_preflight)
    provenance = dict(workspace_provenance)
    smoke = dict(validate_worker_runtime_smoke(worker_runtime_smoke))
    if (
        preflight.get("status") != "PASS"
        or preflight.get("gpu_memory_allocated") is not False
        or smoke.get("filesystem_mutations") != 0
        or smoke.get("target_labels_opened") is not False
        or smoke.get("scientific_gpu_work_performed") is not False
    ):
        raise ProtocolError("SCEPTRE v3 runtime launch admission drifted.")
    preflight_hash = canonical_hash(preflight)
    provenance_hash = canonical_hash(provenance)
    smoke_hash = str(smoke["worker_runtime_smoke_hash"])
    base = {
        "schema_version": "sceptre_v3_runtime_launch_admission_v1",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "read_only_admission_hash": admission.admission_hash,
        "workstation_preflight_hash": preflight_hash,
        "workspace_provenance_hash": provenance_hash,
        "worker_runtime_smoke_hash": smoke_hash,
        "worker_smoke_used_exact_production_initializer": True,
        "scientific_source_generation_started": False,
        "target_labels_opened": False,
        "filesystem_mutations": 0,
        "authorization_lease_claimed": False,
    }
    return RuntimeLaunchAdmission(
        read_only_admission_hash=admission.admission_hash,
        workstation_preflight_hash=preflight_hash,
        workspace_provenance_hash=provenance_hash,
        worker_runtime_smoke_hash=smoke_hash,
        admission_hash=canonical_hash(base),
    )


def assert_execution_authorized(config: object) -> Mapping[str, object]:
    """Authenticate config, source, and amendment authority without mutation."""

    if not isinstance(config, SceptreV3Config):
        raise ProtocolError("SCEPTRE v3 requires its canonical config type.")
    if (
        config.experiment_id != EXPERIMENT_ID
        or config.output_artifact_id != OUTPUT_ARTIFACT_ID
        or tuple(config.input_artifact_ids) != INPUT_ARTIFACT_IDS
        or config.authorization_basis != AUTHORIZATION_BASIS
        or config.authorization_scope != AUTHORIZATION_SCOPE
        or config.execution_authorized is not True
        or config.source_path is None
    ):
        raise ProtocolError("SCEPTRE v3 execution identity drifted.")
    source = Path(config.source_path)
    if source.is_symlink() or not source.is_file():
        raise ProtocolError("SCEPTRE v3 canonical config snapshot is unsafe.")
    reloaded = load_config(source)
    if reloaded != config or reloaded.contract_hash != config.contract_hash:
        raise ProtocolError("SCEPTRE v3 canonical config snapshot drifted.")
    assert_input_fence(config)
    validate_protocol_payload(config.protocol)
    validate_workstation_payload(config.runtime)
    claim = dict(config.claim_boundary)
    if (
        claim.get("execution_authorized") is not True
        or claim.get("single_use_execution_identity") is not True
        or claim.get("authorization_exhausted") is not False
        or claim.get("fresh_evidence") is not False
        or claim.get("routing_success_claimed") is not False
        or claim.get("nelbo_compatibility_claimed") is not False
        or claim.get("may_feed_another_experiment") is not False
    ):
        raise ProtocolError("SCEPTRE v3 executable claim boundary drifted.")
    source_receipt = validate_source_snapshot(
        expected_manifest_sha256=config.expected_source_snapshot_manifest_sha256,
        expected_tree_sha256=config.expected_source_snapshot_tree_sha256,
        expected_member_count=config.expected_source_snapshot_member_count,
    )
    return MappingProxyType(
        {
            "status": "PASS",
            "experiment_id": EXPERIMENT_ID,
            "output_artifact_id": OUTPUT_ARTIFACT_ID,
            "authorization_basis": AUTHORIZATION_BASIS,
            "authorization_scope": AUTHORIZATION_SCOPE,
            "execution_amendment_sha256": (
                config.expected_execution_amendment_sha256
            ),
            "source_snapshot_manifest_sha256": source_receipt[
                "source_snapshot_manifest_sha256"
            ],
            "source_snapshot_tree_sha256": source_receipt[
                "source_snapshot_tree_sha256"
            ],
            "source_snapshot_member_count": source_receipt[
                "source_snapshot_member_count"
            ],
            "fresh_evidence": False,
            "routing_success_claimed": False,
        }
    )


def dry_run_admission(
    config: object,
    *,
    repository_root: Path | None = None,
    require_workspace_binding: bool = True,
    input_loader: Callable[[object], ValidatedInputs] = load_validated_inputs,
) -> DryRunAdmission:
    """Run every launch gate and return a receipt; perform no writes."""

    authority = assert_execution_authorized(config)
    if not isinstance(config, SceptreV3Config):  # narrows for static tooling
        raise ProtocolError("SCEPTRE v3 config type drifted.")
    root = Path(config.artifact_root)
    assert_pristine_output(root, config)
    validated = input_loader(config)
    if not isinstance(validated, ValidatedInputs):
        raise ProtocolError("SCEPTRE v3 input validator receipt drifted.")
    if require_workspace_binding:
        validate_active_workspace_binding(config)
        validate_workspace_provenance(root, config)
    elif (root / "provenance/input_artifacts.json").exists():
        # A present workspace provenance claim is never silently ignored.
        validate_workspace_provenance(root, config)
    scratch = select_scratch(root, config.runtime)
    assert_scratch_absent(scratch)
    lease_path = assert_authorization_unclaimed(repository_root)
    base = {
        "schema_version": "sceptre_v3_read_only_admission_v1",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "config_hash": config.contract_hash,
        "source_tree_sha256": authority["source_snapshot_tree_sha256"],
        "cache_binding_hash": validated.frame.cache_binding_hash,
        "generation_lock_hash": validated.generation_lock.generation_lock_hash,
        "source_inner_amendment_sha256": (
            validated.source_inner.amendment_sha256
        ),
        "execution_amendment_sha256": config.expected_execution_amendment_sha256,
        "scratch_role": scratch.role,
        "authorization_lease_name": lease_path.name,
        "all_eight_inputs_validated": True,
        "target_labels_opened": False,
        "filesystem_mutations": 0,
    }
    return DryRunAdmission(
        artifact_root=root.resolve(),
        scratch=scratch,
        authorization_lease_path=lease_path,
        config_hash=config.contract_hash,
        source_tree_sha256=str(authority["source_snapshot_tree_sha256"]),
        cache_binding_hash=validated.frame.cache_binding_hash,
        admission_hash=canonical_hash(base),
    )


def assert_pristine_output(root: Path, config: object) -> None:
    target = Path(root)
    if target.is_symlink() or not target.is_dir():
        raise ProtocolError("SCEPTRE v3 workspace-prepared root is absent or unsafe.")
    source = Path(getattr(config, "source_path"))
    config_path = target / "config.resolved.yaml"
    provenance = target / "provenance/input_artifacts.json"
    if (
        config_path.is_symlink()
        or not config_path.is_file()
        or provenance.is_symlink()
        or not provenance.is_file()
        or source.resolve() != config_path.resolve()
    ):
        raise ProtocolError("SCEPTRE v3 launch files are absent or unsafe.")
    members = tuple(target.rglob("*"))
    if any(path.is_symlink() for path in members):
        raise ProtocolError("SCEPTRE v3 pre-BEGIN tree contains a symlink.")
    files = {path.relative_to(target).as_posix() for path in members if path.is_file()}
    directories = {
        path.relative_to(target).as_posix() for path in members if path.is_dir()
    }
    other = tuple(path for path in members if not path.is_file() and not path.is_dir())
    if (
        files != set(LAUNCH_FILES)
        or directories != set(WORKSPACE_DIRECTORIES)
        or other
    ):
        raise ProtocolError(
            "SCEPTRE v3 output contains partial, foreign, or prior-run state."
        )


__all__ = (
    "DryRunAdmission",
    "RuntimeLaunchAdmission",
    "LAUNCH_FILES",
    "WORKSPACE_DIRECTORIES",
    "assert_execution_authorized",
    "assert_pristine_output",
    "bind_runtime_launch_admission",
    "dry_run_admission",
)
