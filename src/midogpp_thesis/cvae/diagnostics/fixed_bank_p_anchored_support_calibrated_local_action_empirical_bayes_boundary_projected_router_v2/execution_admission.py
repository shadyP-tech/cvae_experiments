"""Mutation-free single-use execution admission for SCALE-BP v2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .authorization_lease import assert_authorization_unclaimed
from .config import ScaleBPV2Config
from .experiment_contracts import validate_authorization_amendment
from .hashing import canonical_hash
from .identity import (
    DIRECT_INPUT_ARTIFACT_IDS,
    EXPECTED_PARENT_LEDGER_SHA256,
    EXPECTED_TEST_MANIFEST_SHA256,
    EXPERIMENT_ID,
    GovernanceError,
    OUTPUT_ARTIFACT_ID,
)
from .protocol import (
    validate_protocol_payload,
    validate_terminal_claim_firewall,
)
from .source_fence import SourceFenceReceipt, validate_source_fence
from .source_snapshot import package_source_root, validate_source_snapshot
from .workstation import canonical_workstation_payload
from .workspace_manifest import validate_workspace_manifest


ADMISSION_SCHEMA = "scale_bp_v2_single_use_execution_admission_v1"


@dataclass(frozen=True, slots=True)
class ExecutionAdmissionReceipt:
    status: str
    experiment_id: str
    output_artifact_id: str
    config_contract_hash: str
    source_fence_receipt_hash: str
    source_snapshot_manifest_sha256: str
    source_snapshot_tree_sha256: str
    source_snapshot_member_count: int
    direct_input_binding_hash: str
    artifact_root: str
    scratch_root: str
    authorization_lease_path: str
    receipt_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": ADMISSION_SCHEMA,
            "status": self.status,
            "experiment_id": self.experiment_id,
            "output_artifact_id": self.output_artifact_id,
            "config_contract_hash": self.config_contract_hash,
            "source_fence_receipt_hash": self.source_fence_receipt_hash,
            "source_snapshot_manifest_sha256": (
                self.source_snapshot_manifest_sha256
            ),
            "source_snapshot_tree_sha256": self.source_snapshot_tree_sha256,
            "source_snapshot_member_count": self.source_snapshot_member_count,
            "direct_input_binding_hash": self.direct_input_binding_hash,
            "artifact_root": self.artifact_root,
            "scratch_root": self.scratch_root,
            "authorization_lease_path": self.authorization_lease_path,
            "single_use_execution_identity": True,
            "consumed_test_reuse_authorized": True,
            "predecessor_state_used": False,
            "mutation_performed": False,
            "receipt_hash": self.receipt_hash,
        }


def run_read_only_source_preflight() -> SourceFenceReceipt:
    return validate_source_fence()


def admit_single_use_execution(
    config: ScaleBPV2Config,
    artifact_root: str | Path,
    scratch_root: str | Path,
) -> ExecutionAdmissionReceipt:
    """Admit exactly one pristine v2 run without creating any filesystem state."""

    if not isinstance(config, ScaleBPV2Config):
        raise GovernanceError("SCALE-BP v2 admission requires its sealed config.")
    if (
        config.experiment_id != EXPERIMENT_ID
        or config.output_artifact_id != OUTPUT_ARTIFACT_ID
        or tuple(config.direct_input_artifact_ids) != DIRECT_INPUT_ARTIFACT_IDS
        or config.execution_authorized is not True
        or config.consumed_test_reuse_authorized is not True
    ):
        raise GovernanceError("SCALE-BP v2 execution identity drifted.")
    validate_protocol_payload(config.protocol)
    validate_terminal_claim_firewall(config.claim_boundary)
    _validate_runtime(config.runtime, config.scratch_root)

    artifact = _canonical_requested_root(
        artifact_root, expected=config.artifact_root, role="artifact root"
    )
    scratch = _canonical_requested_root(
        scratch_root, expected=config.scratch_root, role="scratch root"
    )
    if artifact == scratch or _is_within(artifact, scratch) or _is_within(scratch, artifact):
        raise GovernanceError("SCALE-BP v2 output and scratch roots overlap.")
    _require_pristine_or_workspace_launch_root(artifact, config=config)
    _require_pristine_root(scratch, "scratch root")

    source_fence = run_read_only_source_preflight()
    source_snapshot = validate_source_snapshot(
        expected_manifest_sha256=config.expected_source_snapshot_manifest_sha256,
        expected_tree_sha256=config.expected_source_snapshot_tree_sha256,
        expected_member_count=config.expected_source_snapshot_member_count,
    )

    inputs = (
        ("expert_bank", config.expert_bank_root, "directory"),
        ("generation_lock", config.generation_lock_root, "directory"),
        ("test_cache", config.test_cache_root, "directory"),
        ("test_manifest", config.test_manifest_path, "file"),
        ("parent_ledger", config.test_consumption_ledger_path, "file"),
        ("authorization_amendment", config.ledger_amendment_path, "file"),
    )
    resolved_inputs: list[dict[str, object]] = []
    observed_paths: set[Path] = set()
    for role, raw_path, kind in inputs:
        path = _resolved_existing_input(raw_path, role=role, kind=kind)
        if path in observed_paths or _is_within(path, artifact) or _is_within(path, scratch):
            raise GovernanceError("SCALE-BP v2 direct input topology drifted.")
        observed_paths.add(path)
        _reject_predecessor_input_path(path, role)
        row: dict[str, object] = {"role": role, "path": str(path), "kind": kind}
        if kind == "file":
            row["sha256"] = _sha256_file(path)
        resolved_inputs.append(row)

    by_role = {str(row["role"]): row for row in resolved_inputs}
    if by_role["test_manifest"].get("sha256") != EXPECTED_TEST_MANIFEST_SHA256:
        raise GovernanceError("SCALE-BP v2 test manifest binding drifted.")
    if by_role["parent_ledger"].get("sha256") != EXPECTED_PARENT_LEDGER_SHA256:
        raise GovernanceError("SCALE-BP v2 parent ledger binding drifted.")
    if by_role["authorization_amendment"].get("sha256") != (
        config.expected_authorization_amendment_sha256
    ):
        raise GovernanceError("SCALE-BP v2 authorization amendment hash drifted.")
    _validate_authorization_amendment(
        config.ledger_amendment_path,
        config=config,
    )

    direct_input_binding_hash = canonical_hash(
        {
            "schema_version": "scale_bp_v2_direct_input_binding_v1",
            "artifact_ids": list(DIRECT_INPUT_ARTIFACT_IDS),
            "inputs": resolved_inputs,
            "predecessor_artifacts_used": False,
        }
    )
    authorization_lease = assert_authorization_unclaimed(
        artifact,
        scratch,
        forbidden_paths=(
            package_source_root(),
            config.source_path,
            *(Path(str(row["path"])) for row in resolved_inputs),
        ),
    )
    body = {
        "schema_version": ADMISSION_SCHEMA,
        "status": "ADMITTED_SINGLE_USE",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "config_contract_hash": config.contract_hash,
        "source_fence_receipt_hash": source_fence.receipt_hash,
        "source_snapshot_manifest_sha256": source_snapshot.manifest_sha256,
        "source_snapshot_tree_sha256": source_snapshot.tree_sha256,
        "source_snapshot_member_count": source_snapshot.member_count,
        "direct_input_binding_hash": direct_input_binding_hash,
        "artifact_root": str(artifact),
        "scratch_root": str(scratch),
        "authorization_lease_path": str(authorization_lease),
        "single_use_execution_identity": True,
        "consumed_test_reuse_authorized": True,
        "predecessor_state_used": False,
        "mutation_performed": False,
    }
    receipt_hash = canonical_hash(body)
    return ExecutionAdmissionReceipt(
        status="ADMITTED_SINGLE_USE",
        experiment_id=EXPERIMENT_ID,
        output_artifact_id=OUTPUT_ARTIFACT_ID,
        config_contract_hash=config.contract_hash,
        source_fence_receipt_hash=source_fence.receipt_hash,
        source_snapshot_manifest_sha256=source_snapshot.manifest_sha256,
        source_snapshot_tree_sha256=source_snapshot.tree_sha256,
        source_snapshot_member_count=source_snapshot.member_count,
        direct_input_binding_hash=direct_input_binding_hash,
        artifact_root=str(artifact),
        scratch_root=str(scratch),
        authorization_lease_path=str(authorization_lease),
        receipt_hash=receipt_hash,
    )


assert_execution_authorized = admit_single_use_execution


def _validate_runtime(runtime: Mapping[str, object], scratch_root: Path) -> None:
    if not isinstance(runtime, Mapping):
        raise GovernanceError("SCALE-BP v2 runtime admission is malformed.")
    plan = canonical_workstation_payload()
    if any(runtime.get(key) != value for key, value in plan.items()):
        raise GovernanceError("SCALE-BP v2 runtime admission drifted.")
    if (
        runtime.get("scratch_root") != str(scratch_root)
        or runtime.get("cross_run_recovery_allowed") is not False
        or runtime.get("terminal_recovery_allowed") is not False
        or runtime.get("nested_process_pools_allowed") is not False
    ):
        raise GovernanceError("SCALE-BP v2 recovery/topology gate drifted.")


def _validate_authorization_amendment(
    path: Path, *, config: ScaleBPV2Config
) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GovernanceError(
            "Cannot read SCALE-BP v2 authorization amendment."
        ) from exc
    if not isinstance(payload, dict) or payload.get("parent_sha256") != (
        EXPECTED_PARENT_LEDGER_SHA256
    ):
        raise GovernanceError("SCALE-BP v2 authorization amendment drifted.")
    validate_authorization_amendment(
        payload,
        expected_source_manifest_sha256=(
            config.expected_source_snapshot_manifest_sha256
        ),
        expected_source_tree_sha256=config.expected_source_snapshot_tree_sha256,
        expected_source_member_count=config.expected_source_snapshot_member_count,
    )


def _canonical_requested_root(
    value: str | Path, *, expected: Path, role: str
) -> Path:
    path = Path(value)
    if not path.is_absolute() or path == Path(path.anchor):
        raise GovernanceError(f"SCALE-BP v2 {role} is unsafe.")
    resolved = path.resolve(strict=False)
    if resolved != expected.resolve(strict=False):
        raise GovernanceError(f"SCALE-BP v2 {role} differs from its config.")
    return resolved


def _require_pristine_root(path: Path, role: str) -> None:
    if path.exists() or path.is_symlink():
        raise GovernanceError(
            f"SCALE-BP v2 {role} already exists; single-use execution refused."
        )
    current = path.parent
    while not current.exists() and current.parent != current:
        if current.is_symlink():
            raise GovernanceError(f"SCALE-BP v2 {role} parent is unsafe.")
        current = current.parent
    if not current.is_dir() or current.is_symlink():
        raise GovernanceError(f"SCALE-BP v2 {role} parent is unsafe.")


def _require_pristine_or_workspace_launch_root(
    path: Path, *, config: ScaleBPV2Config
) -> None:
    """Accept either absence or the workspace runtime's exact launch envelope.

    ``midogpp_thesis workspace run`` resolves the config and records the exact
    input manifest before invoking the experiment process.  Those two files do
    not consume the scientific authorization; every other existing byte does.
    """

    if not path.exists() and not path.is_symlink():
        _require_pristine_root(path, "artifact root")
        return
    if path.is_symlink() or not path.is_dir():
        raise GovernanceError("SCALE-BP v2 artifact root is unsafe.")
    expected_config = path / "config.resolved.yaml"
    expected_manifest = path / "provenance/input_artifacts.json"
    try:
        members = tuple(path.rglob("*"))
        actual_files = tuple(
            sorted(
                member.relative_to(path).as_posix()
                for member in members
                if member.is_file()
            )
        )
        actual_directories = tuple(
            sorted(
                member.relative_to(path).as_posix()
                for member in members
                if member.is_dir()
            )
        )
        if any(member.is_symlink() for member in members):
            raise GovernanceError("SCALE-BP v2 workspace launch root contains a symlink.")
        if actual_files != (
            "config.resolved.yaml",
            "provenance/input_artifacts.json",
        ) or actual_directories != (
            "manifests",
            "provenance",
            "reports",
            "tables",
        ):
            raise GovernanceError(
                "SCALE-BP v2 artifact root is not an exact workspace launch envelope."
            )
        if config.source_path.resolve(strict=True) != expected_config.resolve(strict=True):
            raise GovernanceError("SCALE-BP v2 resolved launch config binding drifted.")
        payload = json.loads(expected_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GovernanceError("SCALE-BP v2 workspace launch envelope is unreadable.") from exc
    if not isinstance(payload, Mapping):
        raise GovernanceError("SCALE-BP v2 workspace launch manifest is malformed.")
    validate_workspace_manifest(payload)


def _resolved_existing_input(value: Path, *, role: str, kind: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise GovernanceError(f"SCALE-BP v2 {role} input was not resolved.")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise GovernanceError(f"SCALE-BP v2 {role} input is absent.") from exc
    if path.is_symlink() or (kind == "file" and not resolved.is_file()) or (
        kind == "directory" and not resolved.is_dir()
    ):
        raise GovernanceError(f"SCALE-BP v2 {role} input is unsafe.")
    return resolved


def _reject_predecessor_input_path(path: Path, role: str) -> None:
    lowered = path.as_posix().casefold()
    if role in {"expert_bank", "generation_lock"}:
        return
    if any(
        fragment in lowered
        for fragment in (
            "/90_oracles_and_diagnostics/",
            "_router_v1/",
            "_amendment_v1",
            "/scratch/",
            "/checkpoints/",
            "/run_state",
        )
    ):
        raise GovernanceError("SCALE-BP v2 predecessor input path detected.")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise GovernanceError("Cannot hash SCALE-BP v2 direct input.") from exc
    return digest.hexdigest()


__all__ = (
    "ADMISSION_SCHEMA",
    "ExecutionAdmissionReceipt",
    "admit_single_use_execution",
    "assert_execution_authorized",
    "run_read_only_source_preflight",
)
