from __future__ import annotations

import numpy as np

from midogpp_thesis.cvae.task_fisher import fit_task_fisher_metric
from midogpp_thesis.real_features.classifier_reference.classifiers import ClassifierSpec


def test_task_fisher_is_trace_normalized_and_rank_one() -> None:
    rng = np.random.default_rng(17)
    labels = np.asarray([0] * 40 + [1] * 40)
    embeddings = rng.normal(size=(80, 6))
    embeddings[:, 0] += labels * 2.5
    result = fit_task_fisher_metric(
        embeddings,
        labels,
        spec=ClassifierSpec(C=1.0, random_state=23),
        alpha=1.0,
    )
    assert result.valid
    assert result.rank <= 1
    assert np.isclose(np.trace(result.metric), 6.0)
    assert np.linalg.eigvalsh(result.metric).min() >= 0.0


def test_task_fisher_alpha_zero_is_identity() -> None:
    labels = [0, 0, 1, 1]
    embeddings = [[-2.0, 0.0], [-1.0, 1.0], [1.0, 0.0], [2.0, -1.0]]
    result = fit_task_fisher_metric(
        embeddings,
        labels,
        spec=ClassifierSpec(C=1.0, random_state=23),
        alpha=0.0,
    )
    assert result.valid
    assert np.allclose(result.metric, np.eye(2))
