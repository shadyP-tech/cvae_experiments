"""Exact deterministic class -> case -> row training schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError


@dataclass(frozen=True)
class BalancedSchedule:
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
