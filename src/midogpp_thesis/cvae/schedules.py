"""Deterministic class -> case -> row schedules for fixed-step CVAE training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..common.hashing import stable_hash
from .protocol import ProtocolError


@dataclass(frozen=True)
class BalancedSchedule:
    """Materialized balanced batches and their exact exposure audit."""

    batches: object
    step_hashes: tuple[str, ...]
    stream_hash: str
    row_exposure: Mapping[str, int]
    case_class_exposure: Mapping[str, int]


def build_balanced_schedule(
    labels: Sequence[int],
    case_ids: Sequence[str],
    sample_ids: Sequence[str],
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> BalancedSchedule:
    """Sample an exact 50/50 class schedule via class -> case -> row draws."""

    import numpy as np

    if steps <= 0 or batch_size <= 0 or batch_size % 2:
        raise ProtocolError("Balanced schedule requires positive steps and even batch size.")
    y = np.asarray(labels, dtype=np.int64)
    cases = np.asarray(case_ids, dtype=str)
    samples = np.asarray(sample_ids, dtype=str)
    if len(y) == 0 or len(y) != len(cases) or len(y) != len(samples):
        raise ProtocolError("Balanced schedule arrays must be aligned and nonempty.")
    groups: dict[int, dict[str, np.ndarray]] = {}
    for label in (0, 1):
        groups[label] = {}
        for case in sorted(set(cases[y == label].tolist())):
            indices = np.flatnonzero((y == label) & (cases == case))
            if len(indices):
                groups[label][str(case)] = indices
        if not groups[label]:
            raise ProtocolError(f"Balanced schedule has no cases for class {label}.")
    rng = np.random.default_rng(int(seed))
    batches = np.empty((steps, batch_size), dtype=np.int64)
    step_hashes: list[str] = []
    row_exposure = {str(sample): 0 for sample in samples.tolist()}
    case_exposure = {
        f"{label}:{case}": 0
        for label in (0, 1)
        for case in groups[label]
    }
    half = batch_size // 2
    for step in range(steps):
        selected: list[int] = []
        for label in (0, 1):
            eligible_cases = tuple(groups[label])
            chosen_cases = rng.choice(eligible_cases, size=half, replace=True)
            for case in chosen_cases:
                candidates = groups[label][str(case)]
                row = int(candidates[int(rng.integers(0, len(candidates)))])
                selected.append(row)
                row_exposure[str(samples[row])] += 1
                case_exposure[f"{label}:{case}"] += 1
        permutation = rng.permutation(batch_size)
        batch = np.asarray(selected, dtype=np.int64)[permutation]
        if int((y[batch] == 0).sum()) != half or int((y[batch] == 1).sum()) != half:
            raise ProtocolError("Balanced schedule failed its exact class quota.")
        batches[step] = batch
        step_hashes.append(
            stable_hash(
                {
                    "step": step + 1,
                    "sample_ids": [str(samples[index]) for index in batch],
                }
            )
        )
    stream_hash = stable_hash(
        {
            "schema_version": "midogpp_case_class_batch_stream_v1",
            "steps": steps,
            "batch_size": batch_size,
            "seed": int(seed),
            "step_hashes": step_hashes,
        }
    )
    return BalancedSchedule(
        batches=batches,
        step_hashes=tuple(step_hashes),
        stream_hash=stream_hash,
        row_exposure=row_exposure,
        case_class_exposure=case_exposure,
    )


def build_fold_fixed_schedule(
    labels: Sequence[int],
    case_ids: Sequence[str],
    sample_ids: Sequence[str],
    *,
    steps: int,
    batch_size: int,
    center: str,
    fit_row_hash: str,
    recipe_version: str,
) -> BalancedSchedule:
    """Materialize a schedule independent of candidate and initialization.

    The key deliberately omits candidate and training seed. It is therefore a
    common-random-number control for one-draw and antithetic estimators.
    """

    key = stable_hash(
        {
            "schema_version": "midogpp_b_fold_fixed_schedule_v1",
            "center": str(center),
            "fit_row_hash": str(fit_row_hash),
            "recipe_version": str(recipe_version),
        }
    )
    seed = int(key[:16], 16) % (2**31 - 1)
    return build_balanced_schedule(
        labels,
        case_ids,
        sample_ids,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
    )


__all__ = (
    "BalancedSchedule",
    "build_balanced_schedule",
    "build_fold_fixed_schedule",
)
