"""Canonical 81-task planning for the label-free target action probe."""

from __future__ import annotations

from pathlib import Path

from ...protocol import ProtocolError
from ..exact_tail_utility_surface.config import CLASSIFIER
from ..exact_tail_utility_surface.source_contracts import GeneratedDevelopmentCache
from ..residual_topup.hashing import canonical_sha256
from ..utility_aligned_identities import CENTERS
from .action_probe_checkpoint import sha256_file
from .action_probe_contracts import (
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    ActionProbeRuntime,
    ActionProbeTask,
)
from .inputs import TargetSupportInputs


EXPECTED_ACTION_PROBE_TASK_COUNT = 81


def support_partition_hash(inputs: TargetSupportInputs, target: str) -> str:
    """Bind one target's ordered fresh support rows to its parent reservation."""

    rows = inputs.rows_by_target[target]
    return canonical_sha256(
        {
            "schema_version": "midogpp_target_support_partition_bridge_v1",
            "parent_reservation_hash": inputs.reservation_hash,
            "target": target,
            "case_ids": sorted(set(inputs.case_ids_by_target[target])),
            "ordered_sample_ids": [row.sample_id for row in rows],
            "ordered_case_ids": [row.case_id for row in rows],
        }
    )


def build_action_probe_tasks(
    inputs: TargetSupportInputs,
    generated: GeneratedDevelopmentCache,
    *,
    checkpoint_root: Path,
    runtime: ActionProbeRuntime,
) -> tuple[ActionProbeTask, ...]:
    """Plan target x training-seed x generation-seed work without labels."""

    source_by_key = generated.source_by_key
    tasks: list[ActionProbeTask] = []
    ordinal = 0
    for target in CENTERS:
        candidate_sources = tuple(source for source in CENTERS if source != target)
        support_path = inputs.support_array_path_by_target[target]
        rows = inputs.rows_by_target[target]
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                records = tuple(
                    source_by_key.get((source, training_seed, generation_seed))
                    for source in candidate_sources
                )
                if any(record is None for record in records):
                    raise ProtocolError(
                        "Target-support action probe is missing a generated source cell."
                    )
                resolved = tuple(record for record in records if record is not None)
                paths = {
                    record.source_center: str(
                        _safe_source_member(generated.root, record.relative_path)
                    )
                    for record in resolved
                }
                hashes = {
                    record.source_center: record.file_sha256 for record in resolved
                }
                tasks.append(
                    ActionProbeTask(
                        task_ordinal=ordinal,
                        target_id=target,
                        training_seed=training_seed,
                        generation_seed=generation_seed,
                        candidate_sources=candidate_sources,
                        support_array_path=str(support_path),
                        support_file_sha256=sha256_file(support_path),
                        support_partition_hash=support_partition_hash(inputs, target),
                        support_case_ids=tuple(row.case_id for row in rows),
                        support_sample_ids=tuple(row.sample_id for row in rows),
                        source_array_path_by_source=paths,
                        source_file_sha256_by_source=hashes,
                        generated_cache_hash=generated.cache_hash,
                        classifier_payload=CLASSIFIER.to_payload(),
                        runtime=runtime,
                        checkpoint_root=str(checkpoint_root),
                    )
                )
                ordinal += 1
    if (
        runtime.task_count != EXPECTED_ACTION_PROBE_TASK_COUNT
        or runtime.fit_count != EXPECTED_ACTION_PROBE_TASK_COUNT * 9
        or len(tasks) != runtime.task_count
        or tuple(task.task_ordinal for task in tasks)
        != tuple(range(runtime.task_count))
        or len({task.task_hash for task in tasks}) != len(tasks)
        or any(task.runtime != runtime for task in tasks)
    ):
        raise ProtocolError("Target-support action-probe task grid drifted.")
    return tuple(tasks)


def _safe_source_member(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ProtocolError("Target-support generated source member is unsafe.")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ProtocolError(
            "Target-support generated source member escaped its cache."
        ) from exc
    if not resolved.is_file():
        raise ProtocolError("Target-support generated source member is absent.")
    return resolved


__all__ = (
    "EXPECTED_ACTION_PROBE_TASK_COUNT",
    "build_action_probe_tasks",
    "support_partition_hash",
)
