"""Deterministic bounded parallelism for pure TSTR scoring tasks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...protocol import ProtocolError
from .composition import ComposedSynthetic
from .evaluation import TSTRDiagnostic, evaluate_tstr_diagnostic


@dataclass(frozen=True)
class ScoredComposition:
    mode: str
    selected_source: str
    synthetic: ComposedSynthetic
    diagnostic: TSTRDiagnostic


@dataclass(frozen=True)
class _ScoringTask:
    mode: str
    selected_source: str
    synthetic: ComposedSynthetic
    eval_embeddings: np.ndarray
    eval_labels: np.ndarray
    classifier_spec: ClassifierSpec


class DeterministicScoringPool:
    """Reuse one bounded thread pool and always return submission order."""

    def __init__(self, workers: int) -> None:
        if int(workers) < 1:
            raise ProtocolError("Scoring workers must be positive.")
        self.workers = int(workers)
        self._executor = (
            None
            if self.workers == 1
            else ThreadPoolExecutor(
                max_workers=self.workers,
                thread_name_prefix="uniform-b-score",
            )
        )

    def score(
        self,
        sealed: Sequence[tuple[str, str, ComposedSynthetic]],
        eval_embeddings: np.ndarray,
        eval_labels: np.ndarray,
        *,
        classifier_spec: ClassifierSpec,
    ) -> tuple[ScoredComposition, ...]:
        tasks = tuple(
            _ScoringTask(
                mode=str(mode),
                selected_source=str(selected),
                synthetic=synthetic,
                eval_embeddings=eval_embeddings,
                eval_labels=eval_labels,
                classifier_spec=classifier_spec,
            )
            for mode, selected, synthetic in sealed
        )
        if not tasks:
            return ()
        if self._executor is None:
            results = tuple(_score_task(task) for task in tasks)
        else:
            # executor.map preserves input order even when completion order differs.
            results = tuple(self._executor.map(_score_task, tasks))
        if tuple((item.mode, item.selected_source) for item in results) != tuple(
            (task.mode, task.selected_source) for task in tasks
        ):
            raise ProtocolError("Parallel scoring changed deterministic result order.")
        return results

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __enter__(self) -> "DeterministicScoringPool":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _score_task(task: _ScoringTask) -> ScoredComposition:
    diagnostic = evaluate_tstr_diagnostic(
        task.synthetic,
        task.eval_embeddings,
        task.eval_labels,
        classifier_spec=task.classifier_spec,
    )
    return ScoredComposition(
        mode=task.mode,
        selected_source=task.selected_source,
        synthetic=task.synthetic,
        diagnostic=diagnostic,
    )


__all__ = ("DeterministicScoringPool", "ScoredComposition")
