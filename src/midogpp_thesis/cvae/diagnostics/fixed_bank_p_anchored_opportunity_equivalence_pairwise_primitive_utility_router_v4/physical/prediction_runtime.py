"""Sealed v4 physical B/U/A1 materialization without the 3-thread facade."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
from typing import Mapping

from ....generation.contracts import GenerationLock
from ....protocol import ProtocolError
from ....runtime.fixed_bank_a1_prediction_contracts import (
    CHECKPOINT_DIRECTORY,
    PREDICTION_ARRAY_MEMBER,
    PREDICTION_INDEX_MEMBER,
    PREDICTION_SEAL_MEMBER,
    GlobalPredictionSeal,
    validate_action_library,
)
from ....runtime.fixed_bank_a1_prediction_planning import (
    build_prediction_tasks,
    write_target_scratch,
)
from ....runtime.fixed_bank_a1_prediction_store import (
    cells_from_checkpoints,
    compute_store_hash,
    load_global_prediction_seal,
    write_prediction_store,
)
from ....runtime.frozen_source_streams import (
    FrozenSourceStreamCache,
    materialize_frozen_source_streams,
)
from ..hashing import canonical_hash, require_sha256
from ..identity import (
    CENTERS,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_CASE_COUNT,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_TEST_ROW_COUNT,
)
from .actions import action_library_by_target
from .cpu_pool import execute_prediction_tasks_one_thread
from .frame import LabelFreeTestFrame
from .runtime_config import PhysicalRuntimeConfig, physical_runtime_payload
from .topology import CPU_WORKER_ENVIRONMENT


PHASE_ORDER = (
    "two_persistent_gpu_source_materialization",
    "four_spawn_cpu_prediction",
)


@dataclass(frozen=True, slots=True)
class MaterializedPhysicalInputs:
    source_cache: FrozenSourceStreamCache
    prediction: GlobalPredictionSeal
    partition_hash: str
    source_root: Path
    prediction_root: Path

    def __post_init__(self) -> None:
        if (
            type(self.source_cache) is not FrozenSourceStreamCache
            or type(self.prediction) is not GlobalPredictionSeal
            or require_sha256(self.partition_hash, "physical partition hash")
            != self.partition_hash
            or not self.source_root.is_absolute()
            or not self.prediction_root.is_absolute()
            or self.source_root.is_symlink()
            or self.prediction_root.is_symlink()
        ):
            raise ProtocolError("OE-PPUR v4 physical materialization receipt drifted.")


def physical_partition_hash(frame: LabelFreeTestFrame) -> str:
    if type(frame) is not LabelFreeTestFrame:
        raise ProtocolError("OE-PPUR v4 physical partition frame is untyped.")
    return canonical_hash(
        {
            "schema_version": "oe_ppur_v4_physical_partition_v1",
            "rows": [
                {
                    "target_center": row.center,
                    "case_id": row.case_id,
                    "sample_id": row.sample_id,
                }
                for row in frame.rows
            ],
            "row_count": EXPECTED_TEST_ROW_COUNT,
            "case_count": EXPECTED_CASE_COUNT,
            "labels_used": False,
        }
    )


def materialize_physical_inputs(
    config: PhysicalRuntimeConfig,
    generation_lock: GenerationLock,
    frame: LabelFreeTestFrame,
    *,
    artifact_root: Path,
    scratch_root: Path,
) -> MaterializedPhysicalInputs:
    """Generate 81 source streams, then fit 810 cells in a disjoint CPU phase."""

    if (
        type(config) is not PhysicalRuntimeConfig
        or dict(config.runtime) != physical_runtime_payload()
        or type(generation_lock) is not GenerationLock
        or generation_lock.generation_lock_hash != EXPECTED_GENERATION_LOCK_HASH
        or generation_lock.bank_lock_hash != EXPECTED_BANK_LOCK_HASH
        or type(frame) is not LabelFreeTestFrame
    ):
        raise ProtocolError("OE-PPUR v4 physical input lineage drifted.")
    artifact = _existing_plain_directory(artifact_root, role="artifact")
    scratch_parent = _existing_plain_directory(scratch_root, role="scratch")
    source_root = _fresh_directory(artifact / "physical/source_streams")
    prediction_root = _fresh_directory(artifact / "physical/predictions")
    work_root = _fresh_directory(scratch_parent / "oe_ppur_v4_prediction_work")
    partition = physical_partition_hash(frame)

    source = _materialize_source_phase(
        config,
        generation_lock,
        root=source_root,
    )
    for name, value in CPU_WORKER_ENVIRONMENT.items():
        os.environ[name] = value
    payload, library_hash = validate_action_library(action_library_by_target())
    target_binding = frame.cache_binding_hash
    target_scratch = write_target_scratch(
        work_root,
        frame,
        partition,
        target_binding,
    )
    tasks = build_prediction_tasks(
        config,
        source,
        target_scratch,
        payload,
        library_hash,
        partition,
        work_root,
    )
    completed = execute_prediction_tasks_one_thread(tasks)
    cells = cells_from_checkpoints(tasks, completed)
    rows = {
        center: tuple(
            str(value) for value in target_scratch["row_ids_by_center"][center]
        )
        for center in CENTERS
    }
    cases = {
        center: tuple(
            str(value) for value in target_scratch["case_ids_by_center"][center]
        )
        for center in CENTERS
    }
    store_hash = compute_store_hash(
        cells,
        rows,
        cases,
        source.lock_hash,
        library_hash,
        target_binding,
    )
    write_prediction_store(
        prediction_root,
        cells,
        rows,
        cases,
        config.contract_hash,
        partition,
        source.lock_hash,
        library_hash,
        target_binding,
        store_hash,
    )
    prediction = load_global_prediction_seal(
        prediction_root,
        expected_config_hash=config.contract_hash,
        expected_partition_hash=partition,
        expected_source_lock_hash=source.lock_hash,
        expected_action_library_hash=library_hash,
        expected_target_cache_binding_hash=target_binding,
    )
    _delete_owned_work_root(work_root)
    return MaterializedPhysicalInputs(
        source,
        prediction,
        partition,
        source_root,
        prediction_root,
    )


def _materialize_source_phase(
    config: PhysicalRuntimeConfig,
    generation_lock: GenerationLock,
    *,
    root: Path,
) -> FrozenSourceStreamCache:
    if (
        tuple(config.runtime.get("generation_devices", ()))
        != ("cuda:0", "cuda:1")
        or config.runtime.get("source_workers_per_device") != 1
        or config.runtime.get("generation_workers_per_device") != 1
    ):
        raise ProtocolError("OE-PPUR v4 two-device source scheduling drifted.")
    return materialize_frozen_source_streams(config, generation_lock, root=root)


def _existing_plain_directory(value: Path, *, role: str) -> Path:
    candidate = Path(os.path.abspath(Path(value)))
    _reject_symlink_chain(candidate, role=role)
    try:
        resolved = Path(value).resolve(strict=True)
    except OSError as exc:
        raise ProtocolError(f"OE-PPUR v4 {role} root is absent.") from exc
    if (
        resolved != candidate
        or not resolved.is_dir()
        or resolved.is_symlink()
        or resolved == Path(resolved.anchor)
    ):
        raise ProtocolError(f"OE-PPUR v4 {role} root is unsafe.")
    return resolved


def _reject_symlink_chain(path: Path, *, role: str) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ProtocolError(f"OE-PPUR v4 {role} path contains a symlink.")
        if current == current.parent:
            return
        current = current.parent


def _fresh_directory(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise ProtocolError("OE-PPUR v4 physical state is not fresh.")
    current = path.parent
    while current != current.parent and not current.exists():
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise ProtocolError("OE-PPUR v4 physical parent chain is unsafe.")
    path.mkdir(parents=True, exist_ok=False)
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_dir():
        raise ProtocolError("OE-PPUR v4 physical directory creation drifted.")
    return resolved


def _delete_owned_work_root(work_root: Path) -> None:
    checkpoint = work_root / CHECKPOINT_DIRECTORY
    if work_root.is_symlink() or not work_root.is_dir():
        raise ProtocolError("OE-PPUR v4 prediction work root is unsafe.")
    allowed_files = {
        "target_scratch.json",
        "target_embeddings.npy",
    }
    centers = r"(?:0|1|2|3|5|6|7|8|9)"
    seeds = r"(?:17|42|101)"
    task_pattern = re.compile(
        rf"tasks/target_{centers}_train_{seeds}_generation_{seeds}\.(?:json|npz)"
    )
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ProtocolError("OE-PPUR v4 prediction checkpoint root is unsafe.")
    observed = set()
    for path in checkpoint.rglob("*"):
        if path.is_symlink():
            raise ProtocolError("OE-PPUR v4 prediction work tree contains a symlink.")
        if path.is_file():
            relative = path.relative_to(checkpoint).as_posix()
            if relative not in allowed_files and task_pattern.fullmatch(relative) is None:
                raise ProtocolError("OE-PPUR v4 prediction work tree has unknown members.")
            observed.add(relative)
        elif not path.is_dir():
            raise ProtocolError("OE-PPUR v4 prediction work tree is unsafe.")
    if not allowed_files <= observed:
        raise ProtocolError("OE-PPUR v4 prediction work tree is incomplete.")
    expected_final = {
        PREDICTION_ARRAY_MEMBER,
        PREDICTION_INDEX_MEMBER,
        PREDICTION_SEAL_MEMBER,
    }
    if any((work_root / member).exists() for member in expected_final):
        raise ProtocolError("OE-PPUR v4 final predictions escaped the artifact root.")
    shutil.rmtree(checkpoint)
    work_root.rmdir()


__all__ = (
    "MaterializedPhysicalInputs",
    "PHASE_ORDER",
    "materialize_physical_inputs",
    "physical_partition_hash",
)
