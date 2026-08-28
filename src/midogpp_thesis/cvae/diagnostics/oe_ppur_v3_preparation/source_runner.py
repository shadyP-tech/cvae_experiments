"""Canonical workstation runner for OE-PPUR v3 source input materialization."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import os
from pathlib import Path
from typing import Iterator

from midogpp_thesis.workspace.runtime import MidogppWorkspace

from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.capacity_preflight import (
    ResourceCapacityReceipt,
    preflight_resource_capacity,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.hashing import (
    canonical_hash,
    require_sha256,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.identity import (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    SOURCE_SUPERVISION_ARTIFACT_ID,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.lifecycle_source_seal import (
    build_lifecycle_source_seal,
    validate_lifecycle_source_seal,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.run_paths import (
    assert_no_symlink_chain,
    paths_overlap,
    validate_absolute_path,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_bundle.constants import (
    SOURCE_CACHE_ARTIFACT_ID,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_production.orchestrator import (
    SourceProductionResult,
    produce_source_supervision_bundle,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_production.resume import (
    fsync_directory,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_seal import (
    build_source_seal,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.workstation import (
    WorkstationPlanReceipt,
    preflight_workstation,
)
from ...protocol import ProtocolError
from .source_receipt import (
    SourceArtifactReceipt,
    validate_materialized_source_artifact,
)


@dataclass(frozen=True, slots=True)
class SourceMaterializationResult:
    artifact_root: Path
    status: str
    artifact: SourceArtifactReceipt
    lifecycle_source_seal_sha256: str
    lifecycle_source_seal_receipt_hash: str
    producer_result_hash: str | None
    producer_bundle_receipt_hash: str | None
    workstation_receipt_hash: str | None
    capacity_receipt_hash: str | None
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        root = Path(self.artifact_root)
        if (
            not root.is_absolute()
            or root == Path(root.anchor)
            or self.status
            not in {"PRODUCED_AND_VALIDATED", "EXISTING_ARTIFACT_REVALIDATED"}
            or not isinstance(self.artifact, SourceArtifactReceipt)
        ):
            raise ProtocolError("OE-PPUR v3 source materialization result drifted.")
        guarded = (
            "producer_result_hash",
            "producer_bundle_receipt_hash",
            "workstation_receipt_hash",
            "capacity_receipt_hash",
        )
        for name in (
            "lifecycle_source_seal_sha256",
            "lifecycle_source_seal_receipt_hash",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), name.replace("_", " ")),
            )
        if self.status == "PRODUCED_AND_VALIDATED":
            if any(getattr(self, name) is None for name in guarded):
                raise ProtocolError("OE-PPUR v3 fresh source result is incomplete.")
            for name in guarded:
                object.__setattr__(
                    self,
                    name,
                    require_sha256(getattr(self, name), name.replace("_", " ")),
                )
        elif any(getattr(self, name) is not None for name in guarded):
            raise ProtocolError("OE-PPUR v3 existing source result drifted.")
        object.__setattr__(self, "artifact_root", root)
        object.__setattr__(self, "result_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_source_materialization_result_v2",
            "status": self.status,
            "artifact_id": SOURCE_SUPERVISION_ARTIFACT_ID,
            "artifact_root": self.artifact_root.as_posix(),
            "artifact_receipt": self.artifact.to_payload(),
            "lifecycle_source_seal_sha256": (
                self.lifecycle_source_seal_sha256
            ),
            "lifecycle_source_seal_receipt_hash": (
                self.lifecycle_source_seal_receipt_hash
            ),
            "producer_result_hash": self.producer_result_hash,
            "producer_bundle_receipt_hash": self.producer_bundle_receipt_hash,
            "workstation_receipt_hash": self.workstation_receipt_hash,
            "capacity_receipt_hash": self.capacity_receipt_hash,
            "target_rows_present": False,
            "target_labels_used": False,
            "consumed_test_authorized": False,
            "experiment_launched": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "result_hash": self.result_hash}


def materialize_source_input(
    *,
    scratch_root: str | Path,
) -> SourceMaterializationResult:
    """Produce or reconstructively validate the sole canonical input #3."""

    live_seal = build_source_seal()
    repository = Path(live_seal.repository_root)
    lifecycle = build_lifecycle_source_seal(repository)
    if Path(lifecycle.repository_root) != repository:
        raise ProtocolError("OE-PPUR v3 source lifecycle repository drifted.")
    workspace = MidogppWorkspace.load(repository)
    workspace.validate()
    if workspace.repo_root != repository:
        raise ProtocolError("OE-PPUR v3 source workspace root drifted.")
    bank = _resolve_canonical_source_artifact(
        workspace, EXPERT_BANK_ARTIFACT_ID, require_exists=True
    )
    generation = _resolve_canonical_source_artifact(
        workspace, GENERATION_LOCK_ARTIFACT_ID, require_exists=True
    )
    cache = _resolve_canonical_source_artifact(
        workspace, SOURCE_CACHE_ARTIFACT_ID, require_exists=True
    )
    output = _resolve_canonical_source_artifact(
        workspace,
        SOURCE_SUPERVISION_ARTIFACT_ID,
        for_output=True,
        require_exists=False,
    )
    scratch = _validate_source_paths(
        scratch_root,
        inputs=(bank, generation, cache),
        output=output,
    )
    # Atomic publication makes an existing bundle immutable.  Revalidate it
    # without creating even the preparation-lock inode.
    if output.exists() or output.is_symlink():
        artifact = validate_materialized_source_artifact(output)
        lifecycle = validate_lifecycle_source_seal(
            lifecycle,
            expected_sha256=lifecycle.lifecycle_source_sha256,
        )
        return SourceMaterializationResult(
            artifact_root=output,
            status="EXISTING_ARTIFACT_REVALIDATED",
            artifact=artifact,
            lifecycle_source_seal_sha256=lifecycle.lifecycle_source_sha256,
            lifecycle_source_seal_receipt_hash=lifecycle.receipt_hash,
            producer_result_hash=None,
            producer_bundle_receipt_hash=None,
            workstation_receipt_hash=None,
            capacity_receipt_hash=None,
        )
    anchor = _nearest_existing_directory(output.parent)
    with _exclusive_preparation_lock(anchor):
        # Another conforming producer may have published between the first
        # absence check and lock acquisition.
        if output.exists() or output.is_symlink():
            artifact = validate_materialized_source_artifact(output)
            lifecycle = validate_lifecycle_source_seal(
                lifecycle,
                expected_sha256=lifecycle.lifecycle_source_sha256,
            )
            return SourceMaterializationResult(
                artifact_root=output,
                status="EXISTING_ARTIFACT_REVALIDATED",
                artifact=artifact,
                lifecycle_source_seal_sha256=lifecycle.lifecycle_source_sha256,
                lifecycle_source_seal_receipt_hash=lifecycle.receipt_hash,
                producer_result_hash=None,
                producer_bundle_receipt_hash=None,
                workstation_receipt_hash=None,
                capacity_receipt_hash=None,
            )
        workstation = preflight_workstation()
        capacity = preflight_resource_capacity(anchor, scratch)
        _create_canonical_output_parent(output.parent, stop=anchor)
        produced = produce_source_supervision_bundle(
            source_cache_root=cache,
            expert_bank_root=bank,
            generation_lock_root=generation,
            output_root=output,
            scratch_parent=scratch.parent,
            resumable_work_root=scratch,
            expected_producer_source_seal_sha256=(
                live_seal.combined_source_sha256
            ),
        )
        artifact = validate_materialized_source_artifact(output)
        _validate_fresh_result(
            produced,
            artifact=artifact,
            workstation=workstation,
            capacity=capacity,
        )
        lifecycle = validate_lifecycle_source_seal(
            lifecycle,
            expected_sha256=lifecycle.lifecycle_source_sha256,
        )
        return SourceMaterializationResult(
            artifact_root=output,
            status="PRODUCED_AND_VALIDATED",
            artifact=artifact,
            lifecycle_source_seal_sha256=lifecycle.lifecycle_source_sha256,
            lifecycle_source_seal_receipt_hash=lifecycle.receipt_hash,
            producer_result_hash=produced.result_hash,
            producer_bundle_receipt_hash=(
                produced.bundle.production_receipt.receipt_hash
            ),
            workstation_receipt_hash=workstation.receipt_hash,
            capacity_receipt_hash=capacity.receipt_hash,
        )


def _validate_fresh_result(
    produced: SourceProductionResult,
    *,
    artifact: SourceArtifactReceipt,
    workstation: WorkstationPlanReceipt,
    capacity: ResourceCapacityReceipt,
) -> None:
    if (
        not isinstance(produced, SourceProductionResult)
        or not isinstance(workstation, WorkstationPlanReceipt)
        or not isinstance(capacity, ResourceCapacityReceipt)
        or produced.bundle.production_receipt.physical_receipt_sha256
        != artifact.content_sha256
        or produced.producer_source_seal_sha256
        != artifact.producer_source_seal_sha256
        or produced.bundle.surface.receipt.row_order_sha256
        != artifact.row_order_sha256
        or produced.bundle.surface.receipt.compiler_recomputation_receipt_sha256
        != artifact.compiler_recomputation_receipt_sha256
    ):
        raise ProtocolError("OE-PPUR v3 fresh source parse-back drifted.")


def _validate_source_paths(
    scratch_root: str | Path,
    *,
    inputs: tuple[Path, ...],
    output: Path,
) -> Path:
    scratch = validate_absolute_path(scratch_root, role="source preparation scratch")
    output_path = validate_absolute_path(output, role="source supervision output")
    assert_no_symlink_chain(scratch, allow_missing_leaf=True)
    assert_no_symlink_chain(output_path, allow_missing_leaf=True)
    rows = tuple(Path(value) for value in inputs)
    if len(rows) != 3 or len(set(rows)) != 3:
        raise ProtocolError("OE-PPUR v3 source preparation input identity drifted.")
    for row in rows:
        assert_no_symlink_chain(row)
        if not row.is_dir() or row.is_symlink():
            raise ProtocolError("OE-PPUR v3 source preparation input is unsafe.")
        if paths_overlap(row, output_path) or paths_overlap(row, scratch):
            raise ProtocolError("OE-PPUR v3 source preparation path overlap detected.")
    if paths_overlap(output_path, scratch):
        raise ProtocolError("OE-PPUR v3 source output overlaps scratch.")
    parent = scratch.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ProtocolError("OE-PPUR v3 source scratch parent is unsafe.")
    return scratch


def _resolve_canonical_source_artifact(
    workspace: MidogppWorkspace,
    artifact_id: str,
    *,
    for_output: bool = False,
    require_exists: bool,
) -> Path:
    """Reject catalog fallbacks and symlink-collapsed canonical locations."""

    try:
        entry = workspace.artifacts[artifact_id]
    except (AttributeError, KeyError) as exc:
        raise ProtocolError("OE-PPUR v3 source catalog identity is absent.") from exc
    relative = Path(entry.canonical_path or "")
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise ProtocolError("OE-PPUR v3 source canonical path drifted.")
    declared = workspace.repo_root / relative
    assert_no_symlink_chain(declared, allow_missing_leaf=not require_exists)
    observed = workspace.resolve_artifact(
        artifact_id,
        for_output=for_output,
        require_exists=require_exists,
    )
    if observed != declared:
        raise ProtocolError("OE-PPUR v3 source artifact used a noncanonical path.")
    if require_exists and (observed.is_symlink() or not observed.is_dir()):
        raise ProtocolError("OE-PPUR v3 canonical source artifact is unsafe.")
    return observed


def _nearest_existing_directory(path: Path) -> Path:
    current = Path(path)
    while not current.exists():
        if current == current.parent:
            raise ProtocolError("OE-PPUR v3 source output filesystem is absent.")
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise ProtocolError("OE-PPUR v3 source output ancestor is unsafe.")
    assert_no_symlink_chain(current)
    return current


def _create_canonical_output_parent(path: Path, *, stop: Path) -> None:
    parent = Path(path)
    anchor = Path(stop)
    try:
        parent.relative_to(anchor)
    except ValueError as exc:
        raise ProtocolError("OE-PPUR v3 source output parent escaped its anchor.") from exc
    current = anchor
    for part in parent.relative_to(anchor).parts:
        current = current / part
        if current.exists():
            if current.is_symlink() or not current.is_dir():
                raise ProtocolError("OE-PPUR v3 source output parent is unsafe.")
            continue
        current.mkdir(mode=0o750, exist_ok=False)
        fsync_directory(current.parent)
    fsync_directory(parent)


@contextmanager
def _exclusive_preparation_lock(anchor: Path) -> Iterator[None]:
    lock_path = anchor / ".oe_ppur_v3_source_preparation.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o640,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError(
                "OE-PPUR v3 source preparation is already active."
            ) from exc
        os.fsync(descriptor)
        fsync_directory(anchor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


__all__ = ("SourceMaterializationResult", "materialize_source_input")
