"""Read-only launch admission followed by a durable P-DCAPS v2 run lock."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
from typing import Iterator

from ....protocol import ProtocolError
from .execution_admission import assert_v2_execution_authorized
from .identity import AUTHORIZATION_BASIS, EXPERIMENT_ID, OUTPUT_ARTIFACT_ID, require_sha256
from .scratch import ScratchLease, select_scratch
from .workspace_inputs import validate_workspace_provenance


LAUNCH_FILES = frozenset(
    {"config.resolved.yaml", "provenance/input_artifacts.json"}
)
WORKSPACE_DIRECTORIES = frozenset(
    {"manifests", "provenance", "reports", "tables"}
)
RUN_STATE_MEMBER = "reports/run_state.json"
LOCK_MEMBER = ".run.lock"


@dataclass(frozen=True)
class ReadOnlyRunAdmission:
    """Receipt proving all authority and launch checks preceded mutation."""

    root: Path
    config_contract_hash: str
    source_snapshot_tree_sha256: str
    scratch_root: Path
    scratch_role: str

    def __post_init__(self) -> None:
        root = Path(self.root)
        scratch = Path(self.scratch_root)
        require_sha256(self.config_contract_hash, "v2 admitted config hash")
        require_sha256(self.source_snapshot_tree_sha256, "v2 admitted source tree")
        if (
            not root.is_absolute()
            or not scratch.is_absolute()
            or self.scratch_role not in {"dedicated_local", "artifact_parent"}
        ):
            raise ProtocolError("P-DCAPS v2 run-admission receipt drifted.")
        object.__setattr__(self, "root", root.resolve())
        object.__setattr__(self, "scratch_root", scratch)


def assert_read_only_run_admission(
    config: object, *, root: Path
) -> ReadOnlyRunAdmission:
    """Perform every authority check before a lock or scratch path is created."""

    authorization = assert_v2_execution_authorized(config)
    target = Path(root)
    reject_predecessor_execution(config)
    assert_workspace_resolved_paths(config, root=target)
    assert_launch_files(target, config)
    assert_no_partial_state(target)
    validate_workspace_provenance(target, config)
    scratch = select_scratch(target, getattr(config, "runtime"))
    return ReadOnlyRunAdmission(
        target.resolve(),
        str(getattr(config, "contract_hash")),
        str(authorization["source_snapshot_tree_sha256"]),
        scratch.root,
        scratch.role,
    )


def reject_predecessor_execution(config: object) -> None:
    identity = str(getattr(config, "experiment_id", ""))
    output = str(getattr(config, "output_artifact_id", ""))
    if identity != EXPERIMENT_ID or output != OUTPUT_ARTIFACT_ID:
        raise ProtocolError("P-DCAPS v2 requires its fresh execution identity.")


def assert_launch_files(root: Path, config: object) -> None:
    target = Path(root)
    if target.is_symlink() or not target.is_dir():
        raise ProtocolError("P-DCAPS v2 workspace-prepared root is absent or unsafe.")
    config_path = target / "config.resolved.yaml"
    provenance = target / "provenance/input_artifacts.json"
    if (
        config_path.is_symlink()
        or not config_path.is_file()
        or provenance.is_symlink()
        or not provenance.is_file()
        or Path(getattr(config, "source_path")).resolve() != config_path.resolve()
    ):
        raise ProtocolError("P-DCAPS v2 launch files are absent or unsafe.")


def assert_workspace_resolved_paths(config: object, *, root: Path) -> None:
    target = Path(root)
    values = (
        target,
        getattr(config, "artifact_root"),
        getattr(config, "expert_bank_root"),
        getattr(config, "generation_lock_root"),
        getattr(config, "test_cache_root"),
        getattr(config, "test_manifest_path"),
        getattr(config, "test_consumption_ledger_path"),
        getattr(config, "ledger_amendment_path"),
    )
    if (
        any(not Path(value).is_absolute() for value in values)
        or target.resolve() != Path(getattr(config, "artifact_root")).resolve()
    ):
        raise ProtocolError("P-DCAPS v2 requires workspace-resolved paths.")


def reject_existing_run_state(root: Path) -> None:
    path = Path(root) / RUN_STATE_MEMBER
    if path.exists() or path.is_symlink():
        raise ProtocolError(
            "P-DCAPS v2 cross-run recovery is forbidden; run state already exists."
        )


def assert_no_partial_state(root: Path) -> None:
    """Accept only the normal workspace-prepared empty-directory skeleton."""

    target = Path(root)
    if target.is_symlink() or not target.is_dir():
        raise ProtocolError("P-DCAPS v2 output root is absent or unsafe.")
    members = tuple(target.rglob("*"))
    if any(path.is_symlink() for path in members):
        raise ProtocolError("P-DCAPS v2 pre-BEGIN tree contains a symlink.")
    observed_files = {
        path.relative_to(target).as_posix() for path in members if path.is_file()
    }
    observed_directories = {
        path.relative_to(target).as_posix() for path in members if path.is_dir()
    }
    other = tuple(path for path in members if not path.is_file() and not path.is_dir())
    if (
        observed_files != set(LAUNCH_FILES)
        or observed_directories != set(WORKSPACE_DIRECTORIES)
        or other
    ):
        raise ProtocolError(
            "P-DCAPS v2 pre-BEGIN output contains partial, foreign, or prior-run state."
        )
    reject_existing_run_state(target)


@contextmanager
def exclusive_run_lock(
    root: Path, *, admission: ReadOnlyRunAdmission
) -> Iterator[None]:
    """Create the first run-owned member only after read-only admission passed."""

    target = Path(root).resolve()
    if (
        not isinstance(admission, ReadOnlyRunAdmission)
        or admission.root != target
        or admission.scratch_root.exists()
        or admission.scratch_root.is_symlink()
    ):
        raise ProtocolError("P-DCAPS v2 run lock lacks a valid admission receipt.")
    # Recheck immediately before the first mutation to close the admission race.
    assert_no_partial_state(target)
    path = target / LOCK_MEMBER
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError("P-DCAPS v2 diagnostic is already running.") from exc
        os.ftruncate(descriptor, 0)
        os.write(
            descriptor,
            (
                f"pid={os.getpid()}\n"
                f"experiment_id={EXPERIMENT_ID}\n"
                f"authorization_basis={AUTHORIZATION_BASIS}\n"
                f"config_contract_hash={admission.config_contract_hash}\n"
                f"source_snapshot_tree_sha256={admission.source_snapshot_tree_sha256}\n"
            ).encode("ascii"),
        )
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


__all__ = (
    "LAUNCH_FILES",
    "LOCK_MEMBER",
    "RUN_STATE_MEMBER",
    "ReadOnlyRunAdmission",
    "WORKSPACE_DIRECTORIES",
    "assert_launch_files",
    "assert_no_partial_state",
    "assert_read_only_run_admission",
    "assert_workspace_resolved_paths",
    "exclusive_run_lock",
    "reject_existing_run_state",
    "reject_predecessor_execution",
)
