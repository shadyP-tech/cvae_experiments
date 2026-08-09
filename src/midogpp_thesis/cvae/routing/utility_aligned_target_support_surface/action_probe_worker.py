"""Spawn-safe CPU worker for label-free target-support action probing."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from ....real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    fit_logistic_classifier,
)
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from ..exact_tail_utility_surface.config import CLASSIFIER
from .action_probe_checkpoint import write_action_probe_checkpoint
from .action_probe_contracts import (
    SOURCE_ROWS_PER_CLASS,
    TARGET_BASE_PER_SOURCE,
    TARGET_TAIL_PER_SELECTED_SOURCE,
    ActionProbeCheckpoint,
    ActionProbeTask,
)


def action_probe_worker(task: ActionProbeTask) -> ActionProbeCheckpoint:
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError("Target-support action probing requires threadpoolctl.") from exc
    support = _load_support(
        Path(task.support_array_path),
        len(task.support_case_ids),
        expected_sha256=task.support_file_sha256,
    )
    sources = {
        source: _load_source(
            Path(task.source_array_path_by_source[source]),
            expected_sha256=task.source_file_sha256_by_source[source],
        )
        for source in task.candidate_sources
    }
    spec = _classifier_from_payload(task.classifier_payload)
    probabilities: list[np.ndarray] = []
    with threadpool_limits(limits=task.runtime.threads_per_worker):
        for selected_source in (None, *task.candidate_sources):
            train_embeddings, train_labels = compose_target_action(
                sources,
                selected_source=selected_source,
            )
            fitted = fit_logistic_classifier(
                train_embeddings,
                train_labels,
                support,
                spec=spec,
            )
            raw = np.asarray(fitted.probabilities, dtype=np.float64)
            if (
                tuple(int(value) for value in fitted.classes) != (0, 1)
                or raw.shape != (len(support), 2)
                or not np.isfinite(raw).all()
                or not np.allclose(raw.sum(axis=1), 1.0, rtol=0.0, atol=1e-7)
                or not fitted.converged
                or fitted.classifier_config_hash != spec.config_hash
                or not fitted.scaler_state_hash
            ):
                raise ProtocolError("Target-support action-probe classifier drifted.")
            probabilities.append(raw[:, 1].astype(np.float32, copy=False))
    return write_action_probe_checkpoint(
        task,
        np.ascontiguousarray(np.stack(probabilities), dtype=np.float32),
    )


def compose_target_action(
    sources: Mapping[str, np.ndarray],
    *,
    selected_source: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    source_order = tuple(sources)
    if selected_source is not None and selected_source not in source_order:
        raise ProtocolError("Target-support selected source escaped its candidate set.")
    values: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for label in (0, 1):
        start = label * SOURCE_ROWS_PER_CLASS
        for source in source_order:
            count = TARGET_BASE_PER_SOURCE + (
                TARGET_TAIL_PER_SELECTED_SOURCE if source == selected_source else 0
            )
            stop = start + count
            if stop > (label + 1) * SOURCE_ROWS_PER_CLASS:
                raise ProtocolError("Target-support action exceeded source capacity.")
            values.append(np.asarray(sources[source][start:stop], dtype=np.float32))
            labels.append(np.full(count, label, dtype=np.uint8))
    embeddings = np.ascontiguousarray(np.concatenate(values), dtype=np.float32)
    truth = np.ascontiguousarray(np.concatenate(labels), dtype=np.uint8)
    expected_per_class = (
        len(source_order) * TARGET_BASE_PER_SOURCE
        + (0 if selected_source is None else TARGET_TAIL_PER_SELECTED_SOURCE)
    )
    if (
        embeddings.shape != (2 * expected_per_class, COMMON_OUTPUT_DIM)
        or truth.shape != (2 * expected_per_class,)
        or not np.isfinite(embeddings).all()
    ):
        raise ProtocolError("Target-support action composition drifted.")
    return embeddings, truth


def _load_support(
    path: Path,
    row_count: int,
    *,
    expected_sha256: str,
) -> np.ndarray:
    from .action_probe_checkpoint import sha256_file

    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ProtocolError("Target-support action-probe support bytes drifted.")
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("Target-support action-probe input is unreadable.") from exc
    if (
        value.shape != (row_count, COMMON_OUTPUT_DIM)
        or value.dtype != np.float32
        or not np.isfinite(value).all()
    ):
        raise ProtocolError("Target-support action-probe support geometry drifted.")
    return value


def _load_source(path: Path, *, expected_sha256: str) -> np.ndarray:
    from .action_probe_checkpoint import sha256_file

    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ProtocolError("Target-support action-probe source bytes drifted.")
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("Target-support action-probe source is unreadable.") from exc
    if (
        value.shape != (2 * SOURCE_ROWS_PER_CLASS, COMMON_OUTPUT_DIM)
        or value.dtype != np.float32
        or not np.isfinite(value).all()
    ):
        raise ProtocolError("Target-support action-probe source geometry drifted.")
    return value


def _classifier_from_payload(raw: Mapping[str, object]) -> ClassifierSpec:
    try:
        spec = ClassifierSpec(
            family=str(raw["family"]),
            C=float(raw["C"]),
            penalty=str(raw["penalty"]),
            solver=str(raw["solver"]),
            max_iter=int(raw["max_iter"]),
            class_weight=None if raw["class_weight"] is None else str(raw["class_weight"]),
            random_state=int(raw["random_state"]),
            l1_ratio=None if raw["l1_ratio"] is None else float(raw["l1_ratio"]),
            threshold_policy=str(raw["threshold_policy"]),
            scaler_fit=str(raw["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Target-support classifier payload is malformed.") from exc
    if spec.to_payload() != CLASSIFIER.to_payload():
        raise ProtocolError("Target-support classifier contract drifted.")
    return spec


__all__ = ("action_probe_worker", "compose_target_action")
