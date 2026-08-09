"""Exact-nine ensemble-first aggregation for target-support action probes."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from ..residual_topup.hashing import array_sha256, canonical_sha256
from ..utility_aligned.ensemble_contracts import (
    ENSEMBLE_SEED_KEYS,
    SeedProbabilityVector,
)
from ..utility_aligned.ensemble_endpoint import support_action_probability_shift
from ..utility_aligned_identities import CENTERS
from .action_probe_checkpoint import load_checkpoint_probabilities
from .action_probe_contracts import (
    ActionProbeCheckpoint,
    ActionProbeTask,
    TargetSupportActionShiftRow,
)


def build_action_shift_rows(
    tasks: Sequence[ActionProbeTask],
    checkpoints: Sequence[ActionProbeCheckpoint],
) -> tuple[TargetSupportActionShiftRow, ...]:
    """Collapse all target checkpoints into case-level exact-nine scalars."""

    ordered_tasks = tuple(tasks)
    ordered_checkpoints = tuple(checkpoints)
    if len(ordered_tasks) != len(ordered_checkpoints):
        raise ProtocolError("Target-support action-probe task/checkpoint count drifted.")
    if any(
        checkpoint.task_hash != task.task_hash
        for task, checkpoint in zip(
            ordered_tasks, ordered_checkpoints, strict=True
        )
    ):
        raise ProtocolError("Target-support action-probe checkpoint order drifted.")
    rows: list[TargetSupportActionShiftRow] = []
    for target in CENTERS:
        indexed = tuple(
            (task, checkpoint)
            for task, checkpoint in zip(
                ordered_tasks, ordered_checkpoints, strict=True
            )
            if task.target_id == target
        )
        rows.extend(
            build_task_action_shift_rows(
                tuple(task for task, _ in indexed),
                tuple(checkpoint for _, checkpoint in indexed),
            )
        )
    ordered_rows = tuple(sorted(rows, key=lambda row: row.row_key))
    expected_minimum = 9 * 8 * 9 * 8
    if (
        len(ordered_rows) < expected_minimum
        or len({row.row_key for row in ordered_rows}) != len(ordered_rows)
    ):
        raise ProtocolError("Target-support action-shift row grid drifted.")
    return ordered_rows


def build_task_action_shift_rows(
    tasks: Sequence[ActionProbeTask],
    checkpoints: Sequence[ActionProbeCheckpoint],
) -> tuple[TargetSupportActionShiftRow, ...]:
    """Aggregate one target's canonical nine tasks ensemble-first by case."""

    seed_tasks = tuple(tasks)
    seed_checkpoints = tuple(checkpoints)
    if (
        len(seed_tasks) != len(ENSEMBLE_SEED_KEYS)
        or len(seed_checkpoints) != len(ENSEMBLE_SEED_KEYS)
        or tuple(
            (task.training_seed, task.generation_seed) for task in seed_tasks
        )
        != ENSEMBLE_SEED_KEYS
        or any(
            checkpoint.task_hash != task.task_hash
            for task, checkpoint in zip(
                seed_tasks, seed_checkpoints, strict=True
            )
        )
    ):
        raise ProtocolError(
            "Target-support case aggregation requires canonical exact-nine tasks."
        )
    task = seed_tasks[0]
    if any(
        (
            other.target_id,
            other.candidate_sources,
            other.support_partition_hash,
            other.support_case_ids,
            other.support_sample_ids,
        )
        != (
            task.target_id,
            task.candidate_sources,
            task.support_partition_hash,
            task.support_case_ids,
            task.support_sample_ids,
        )
        for other in seed_tasks[1:]
    ):
        raise ProtocolError(
            "Target-support exact-nine tasks disagree on support geometry."
        )
    probability_cells = tuple(
        load_checkpoint_probabilities(seed_task, checkpoint)
        for seed_task, checkpoint in zip(
            seed_tasks, seed_checkpoints, strict=True
        )
    )
    case_ids = tuple(sorted(set(task.support_case_ids)))
    if len(case_ids) < 8:
        raise ProtocolError(
            "Target-support action shifts require eight independent cases."
        )
    rows: list[TargetSupportActionShiftRow] = []
    for source_ordinal, source in enumerate(task.candidate_sources, start=1):
        for case_id in case_ids:
            indices = np.asarray(
                [
                    ordinal
                    for ordinal, value in enumerate(task.support_case_ids)
                    if value == case_id
                ],
                dtype=np.int64,
            )
            if not len(indices):
                raise ProtocolError("Target-support action-shift case is empty.")
            sample_ids = [task.support_sample_ids[int(index)] for index in indices]
            case_identity_hash = canonical_sha256(
                {
                    "schema_version": (
                        "midogpp_utility_aligned_target_support_case_rows_v1"
                    ),
                    "outer_target_id": task.target_id,
                    "case_id": case_id,
                    "ordered_sample_ids": sample_ids,
                }
            )
            base_vectors = _case_vectors(
                seed_tasks,
                probability_cells,
                action_ordinal=0,
                indices=indices,
                case_identity_hash=case_identity_hash,
            )
            tail_vectors = _case_vectors(
                seed_tasks,
                probability_cells,
                action_ordinal=source_ordinal,
                indices=indices,
                case_identity_hash=case_identity_hash,
            )
            aggregate = support_action_probability_shift(base_vectors, tail_vectors)
            for ordinal, (seed_task, base_vector, tail_vector) in enumerate(
                zip(seed_tasks, base_vectors, tail_vectors, strict=True)
            ):
                rows.append(
                    TargetSupportActionShiftRow(
                        outer_target_id=task.target_id,
                        query_id=task.target_id,
                        candidate_source=source,
                        training_seed=seed_task.training_seed,
                        generation_seed=seed_task.generation_seed,
                        case_id=case_id,
                        support_partition_hash=task.support_partition_hash,
                        case_row_identity_hash=case_identity_hash,
                        support_row_count=len(indices),
                        base_probability_sha256=(
                            base_vector.prediction_provenance_hash
                        ),
                        tail_probability_sha256=(
                            tail_vector.prediction_provenance_hash
                        ),
                        base_component_vector_hash=base_vector.vector_hash,
                        tail_component_vector_hash=tail_vector.vector_hash,
                        descriptive_seed_mean_absolute_positive_probability_shift=(
                            aggregate.per_seed_mean_absolute_shifts[ordinal]
                        ),
                        case_ensemble_mean_absolute_positive_probability_shift=(
                            aggregate.value
                        ),
                        case_base_ensemble_probability_sha256=(
                            aggregate.base_ensemble_probability_hash
                        ),
                        case_tail_ensemble_probability_sha256=(
                            aggregate.tail_ensemble_probability_hash
                        ),
                        case_ensemble_absolute_difference_sha256=(
                            aggregate.ensemble_absolute_difference_hash
                        ),
                        case_ensemble_shift_hash=aggregate.shift_hash,
                    )
                )
    return tuple(sorted(rows, key=lambda row: row.row_key))


def _case_vectors(
    tasks: tuple[ActionProbeTask, ...],
    probability_cells: tuple[np.ndarray, ...],
    *,
    action_ordinal: int,
    indices: np.ndarray,
    case_identity_hash: str,
) -> tuple[SeedProbabilityVector, ...]:
    vectors: list[SeedProbabilityVector] = []
    for task, probabilities in zip(tasks, probability_cells, strict=True):
        values = np.ascontiguousarray(
            probabilities[action_ordinal, indices], dtype=np.float32
        )
        vectors.append(
            SeedProbabilityVector(
                training_seed=task.training_seed,
                generation_seed=task.generation_seed,
                row_identity_hash=case_identity_hash,
                prediction_provenance_hash=array_sha256(values),
                positive_class_probabilities=values,
            )
        )
    return tuple(vectors)


__all__ = ("build_action_shift_rows", "build_task_action_shift_rows")
