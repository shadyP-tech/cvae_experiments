"""Small facade for neutral exact-nine fixed-bank A1 predictions."""

from __future__ import annotations

from pathlib import Path
import os
import re
import shutil
from typing import Mapping, Sequence

from ..protocol import ProtocolError
from .fixed_bank_a1_prediction_contracts import (
    ACTION_COUNT_PER_TARGET,
    CHECKPOINT_DIRECTORY,
    EXPECTED_CELL_COUNT,
    EXPECTED_TASK_COUNT,
    GlobalPredictionSeal,
    PREDICTION_ARRAY_MEMBER,
    PREDICTION_INDEX_MEMBER,
    PREDICTION_SEAL_MEMBER,
    PredictionCell,
    PredictionConfig,
    PredictionStore,
    assert_runtime,
    validate_action_library,
)
from .fixed_bank_a1_prediction_planning import (
    build_prediction_tasks,
    write_target_scratch,
)
from .fixed_bank_a1_prediction_store import (
    cells_from_checkpoints,
    compute_store_hash,
    load_global_prediction_seal,
    write_prediction_store,
)
from .fixed_bank_a1_prediction_worker import execute_or_resume_prediction_tasks
from .frozen_source_streams import FrozenSourceStreamCache


def materialize_fixed_bank_a1_action_predictions(
    config: PredictionConfig,
    source_cache: FrozenSourceStreamCache,
    frame: object,
    *,
    partition_hash: str,
    action_library: Mapping[str, Sequence[object]],
    root: Path,
    scratch_root: Path | None = None,
) -> GlobalPredictionSeal:
    """Fit, resume, consolidate, and globally seal the exact 810 cells."""

    assert_runtime(config.runtime)
    payload, library_hash = validate_action_library(action_library)
    target_binding = str(getattr(frame, "cache_binding_hash"))
    final_paths = tuple(
        root / member
        for member in (
            PREDICTION_ARRAY_MEMBER,
            PREDICTION_INDEX_MEMBER,
            PREDICTION_SEAL_MEMBER,
        )
    )
    present = tuple(path.is_file() for path in final_paths)
    if any(path.is_symlink() for path in final_paths):
        raise ProtocolError("Fixed-bank A1 final prediction trio contains a symlink.")
    if all(present):
        result = load_global_prediction_seal(
            root,
            expected_config_hash=config.contract_hash,
            expected_partition_hash=partition_hash,
            expected_source_lock_hash=source_cache.lock_hash,
            expected_action_library_hash=library_hash,
            expected_target_cache_binding_hash=target_binding,
        )
        _cleanup_owned_scratch(scratch_root or root, root=root)
        return result
    if present not in {
        (False, False, False),
        (True, False, False),
        (True, True, False),
    }:
        raise ProtocolError("Fixed-bank A1 final prediction trio is an unsafe partial state.")
    work_root = Path(scratch_root or root)
    _validate_checkpoint_tree(work_root / CHECKPOINT_DIRECTORY)
    scratch = write_target_scratch(
        work_root, frame, partition_hash, target_binding
    )
    tasks = build_prediction_tasks(
        config,
        source_cache,
        scratch,
        payload,
        library_hash,
        partition_hash,
        work_root,
    )
    completed = execute_or_resume_prediction_tasks(tasks, workers=4)
    cells = cells_from_checkpoints(tasks, completed)
    rows = {
        center: tuple(str(value) for value in scratch["row_ids_by_center"][center])
        for center in source_cache.lock_payload.get("centers", ())
    }
    # Neutral frozen-source locks predate a centers field; preserve canonical
    # order from the prediction cells in that case.
    if not rows:
        from ..expert_bank.uniform_b_v2_promotion.contracts import CENTERS

        rows = {
            center: tuple(
                str(value) for value in scratch["row_ids_by_center"][center]
            )
            for center in CENTERS
        }
    cases = {
        center: tuple(str(value) for value in scratch["case_ids_by_center"][center])
        for center in rows
    }
    store_hash = compute_store_hash(
        cells,
        rows,
        cases,
        source_cache.lock_hash,
        library_hash,
        target_binding,
    )
    write_prediction_store(
        root,
        cells,
        rows,
        cases,
        config.contract_hash,
        partition_hash,
        source_cache.lock_hash,
        library_hash,
        target_binding,
        store_hash,
    )
    result = load_global_prediction_seal(
        root,
        expected_config_hash=config.contract_hash,
        expected_partition_hash=partition_hash,
        expected_source_lock_hash=source_cache.lock_hash,
        expected_action_library_hash=library_hash,
        expected_target_cache_binding_hash=target_binding,
    )
    _cleanup_owned_scratch(work_root, root=root)
    return result


def _cleanup_owned_scratch(scratch_root: Path, *, root: Path) -> None:
    directory = Path(scratch_root) / CHECKPOINT_DIRECTORY
    if not directory.exists():
        return
    if directory.is_symlink() or not directory.is_dir():
        raise ProtocolError("Refusing to clean unsafe fixed-bank A1 scratch.")
    _validate_checkpoint_tree(directory)
    if Path(scratch_root).resolve() == root.resolve():
        # The artifact-local fallback is still package-owned and has the same
        # exact checkpoint namespace.
        pass
    shutil.rmtree(directory)


def _validate_checkpoint_tree(directory: Path) -> None:
    if not directory.exists():
        if directory.is_symlink():
            raise ProtocolError("Fixed-bank A1 checkpoint root is a dangling symlink.")
        return
    if directory.is_symlink() or not directory.is_dir():
        raise ProtocolError("Fixed-bank A1 checkpoint root is unsafe.")
    observed: set[str] = set()
    directories: set[str] = set()
    temporary: list[Path] = []
    for base, names, files in os.walk(directory, followlinks=False):
        parent = Path(base)
        for name in (*names, *files):
            if (parent / name).is_symlink():
                raise ProtocolError("Fixed-bank A1 checkpoint tree contains a symlink.")
        directories.update((parent / name).relative_to(directory).as_posix() for name in names)
        for name in files:
            path = parent / name
            relative = path.relative_to(directory).as_posix()
            if re.fullmatch(r".+\.[1-9][0-9]*\.tmp", relative):
                temporary.append(path)
            else:
                observed.add(relative)
    centers = r"(?:0|1|2|3|5|6|7|8|9)"
    seeds = r"(?:17|42|101)"
    allowed = (
        r"target_scratch\.json",
        r"target_embeddings\.npy",
        rf"tasks/target_{centers}_train_{seeds}_generation_{seeds}\.(?:json|npz)",
    )
    for path in temporary:
        relative = path.relative_to(directory).as_posix()
        base_member = re.sub(r"\.[1-9][0-9]*\.tmp$", "", relative)
        if not any(re.fullmatch(pattern, base_member) for pattern in allowed):
            raise ProtocolError("Fixed-bank A1 scratch contains an unknown atomic temp.")
        path.unlink()
    unknown = sorted(
        member for member in observed
        if not any(re.fullmatch(pattern, member) for pattern in allowed)
    )
    if unknown:
        raise ProtocolError(f"Fixed-bank A1 checkpoint tree has unknown members: {unknown}.")
    if not directories <= {"tasks"}:
        raise ProtocolError("Fixed-bank A1 checkpoint tree has unknown directories.")
    scratch = {"target_scratch.json", "target_embeddings.npy"}
    if observed & scratch and observed & scratch not in ({"target_embeddings.npy"}, scratch):
        raise ProtocolError("Fixed-bank A1 target scratch pair is partial.")
    tasks = observed - scratch
    stems: dict[str, set[str]] = {}
    for member in tasks:
        stem, suffix = member.rsplit(".", 1)
        stems.setdefault(stem, set()).add(suffix)
    if any(suffixes not in ({"npz"}, {"json", "npz"}) for suffixes in stems.values()):
        raise ProtocolError("Fixed-bank A1 checkpoint action pair is unsafe.")


__all__ = (
    "ACTION_COUNT_PER_TARGET",
    "EXPECTED_CELL_COUNT",
    "EXPECTED_TASK_COUNT",
    "GlobalPredictionSeal",
    "PredictionCell",
    "PredictionStore",
    "load_global_prediction_seal",
    "materialize_fixed_bank_a1_action_predictions",
)
