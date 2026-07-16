from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from midogpp_thesis.cvae.preservation.source_inner_studies.contracts import (
    FISHER_ALPHAS,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.fisher_runner import (
    _fit_raw_and_derived_fisher,
)
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


def test_shrunk_task_fisher_reuses_one_raw_state_and_matches_formula() -> None:
    rng = np.random.default_rng(101)
    labels = np.asarray([0] * 80 + [1] * 80)
    embeddings = rng.normal(size=(160, 128))
    embeddings[:, 0] += labels * 2.0
    prepared = SimpleNamespace(
        outer="0",
        inner="1",
        x_fit=embeddings,
        y_fit=labels,
        fit_centers=("2", "3", "5", "6", "7", "8", "9"),
        source_ids=tuple(f"source-{index}" for index in range(len(labels))),
        frame=SimpleNamespace(state_hash="frame-hash"),
        spec=ClassifierSpec(C=1.0, random_state=23),
    )
    config = SimpleNamespace(
        alphas=FISHER_ALPHAS,
        raw_fisher_fit_scope="shared_per_outer_inner",
    )

    raw_record, derived = _fit_raw_and_derived_fisher(prepared, config=config)

    assert raw_record["valid"] is True
    assert derived[0.0]["metric"] is None
    raw = np.asarray(raw_record["raw_fisher"])
    normalized = 128.0 * raw / np.trace(raw)
    for alpha in FISHER_ALPHAS[1:]:
        expected = (np.eye(128) + alpha * normalized) / (1.0 + alpha)
        assert np.allclose(derived[alpha]["metric"], expected)
        assert np.isclose(np.trace(derived[alpha]["metric"]), 128.0)
        assert derived[alpha]["raw_fisher_state_hash"] == raw_record[
            "raw_fisher_state_hash"
        ]
        assert np.isclose(derived[alpha]["directional_ratio"], 1.0 + 128.0 * alpha)
