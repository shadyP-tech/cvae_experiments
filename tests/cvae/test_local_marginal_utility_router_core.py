from __future__ import annotations

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.local_marginal_utility import (
    ENERGY_FEATURE_NAMES,
    boost_action_id,
    build_energy_feature_matrix,
    build_perturbation_library,
    fit_cluster_weighted_ridge,
    legal_candidate_sources,
    nested_loqdo_predictions,
    paired_marginal_utility,
    robust_local_utility_weights,
)


def _sources(count: int) -> tuple[str, ...]:
    return tuple(f"center_{index}" for index in range(count))


@pytest.mark.parametrize(
    ("count", "total", "control_count", "boosted_count", "other_count", "effective"),
    (
        (7, 1008, 144, 252, 126, 6.4),
        (8, 1024, 128, 240, 112, 512.0 / 71.0),
    ),
)
def test_perturbation_library_is_integer_exact_and_deterministic(
    count: int,
    total: int,
    control_count: int,
    boosted_count: int,
    other_count: int,
    effective: float,
) -> None:
    sources = _sources(count)
    plans = build_perturbation_library(tuple(reversed(sources)), total_per_class=total)

    assert plans[0].action_id == "control"
    assert plans[0].candidate_sources == sources
    assert plans[0].allocations_per_class == {
        source: control_count for source in sources
    }
    assert len(plans) == count + 1
    for source, plan in zip(sources, plans[1:], strict=True):
        assert plan.action_id == boost_action_id(source)
        assert plan.boosted_source == source
        assert plan.allocations_per_class[source] == boosted_count
        assert all(
            count_value == (boosted_count if candidate == source else other_count)
            for candidate, count_value in plan.allocations_per_class.items()
        )
        assert sum(plan.allocations_per_class.values()) == total
        assert plan.maximum_source_weight <= 0.25
        assert plan.effective_source_count == pytest.approx(effective)


def test_candidate_geometry_excludes_outer_target_and_query() -> None:
    universe = _sources(9)
    candidates = legal_candidate_sources(
        tuple(reversed(universe)), outer_target="center_8", query_cluster="center_3"
    )
    assert candidates == tuple(
        source for source in universe if source not in {"center_8", "center_3"}
    )
    assert len(candidates) == 7
    with pytest.raises(ProtocolError):
        legal_candidate_sources(universe, outer_target="center_3", query_cluster="center_3")


def test_paired_marginal_utility_uses_locked_one_eighth_step() -> None:
    assert paired_marginal_utility(0.78, 0.77) == pytest.approx(0.08)
    vector = paired_marginal_utility([0.78, 0.75], [0.77, 0.76])
    assert np.allclose(vector, [0.08, -0.08])
    with pytest.raises(ProtocolError):
        paired_marginal_utility(0.78, 0.77, epsilon=0.25)


def test_energy_features_are_label_free_keyed_and_deterministic() -> None:
    energies = {
        "q2": {"c": 2.0, "a": 1.0, "b": 1.0},
        "q1": {"b": 4.0, "a": 3.0},
    }
    first = build_energy_feature_matrix(energies)
    second = build_energy_feature_matrix(dict(reversed(tuple(energies.items()))))

    assert first.row_keys == (("q1", "a"), ("q1", "b"), ("q2", "a"), ("q2", "b"), ("q2", "c"))
    assert first.feature_names[: len(ENERGY_FEATURE_NAMES)] == ENERGY_FEATURE_NAMES
    assert np.array_equal(first.values, second.values)
    # Exact energy ties receive the same average rank.
    assert first.values[2, 3] == first.values[3, 3] == pytest.approx(0.25)
    with pytest.raises(ProtocolError):
        build_energy_feature_matrix(
            energies,
            candidate_sources_by_query={"q1": ("a",), "q2": ("a", "b", "c")},
        )


def test_cluster_weighted_ridge_and_nested_loqdo_are_finite_and_query_heldout() -> None:
    rows: list[list[float]] = []
    utility: list[float] = []
    clusters: list[str] = []
    for query_index in range(4):
        for source_index in range(query_index + 2):
            x0 = float(source_index - query_index / 2.0)
            x1 = float(query_index)
            rows.append([x0, x1])
            utility.append(1.5 * x0 - 0.25 * x1 + 0.1)
            clusters.append(f"q{query_index}")
    matrix = np.asarray(rows, dtype=np.float64)
    response = np.asarray(utility, dtype=np.float64)
    model = fit_cluster_weighted_ridge(
        matrix,
        response,
        clusters,
        alpha=0.01,
        feature_names=("compatibility", "query_axis"),
    )
    prediction = model.predict_with_uncertainty(matrix[:3])
    assert prediction.mean.shape == (3,)
    assert prediction.covariance.shape == (3, 3)
    assert np.linalg.eigvalsh(prediction.covariance).min() >= -1e-12

    result = nested_loqdo_predictions(
        matrix,
        response,
        clusters,
        alphas=(0.01, 0.1),
        feature_names=("compatibility", "query_axis"),
    )
    assert np.isfinite(result.predictions).all()
    assert np.isfinite(result.standard_errors).all()
    assert result.query_equal_mean_squared_error >= 0.0
    for fold in result.folds:
        assert fold.heldout_query_cluster not in fold.model.training_query_clusters
        assert fold.selected_alpha in {0.01, 0.1}


def test_strict_nested_loqdo_excludes_domain_from_query_and_source_roles() -> None:
    domains = tuple(f"q{index}" for index in range(8))
    rows: list[list[float]] = []
    utility: list[float] = []
    query_clusters: list[str] = []
    source_clusters: list[str] = []
    for query_index, query in enumerate(domains):
        for source_index, source in enumerate(domains):
            if source == query:
                continue
            for seed_index in range(9):
                rows.append([float(query_index), float(source_index), float(seed_index)])
                utility.append(float(source_index - query_index) + seed_index / 100.0)
                query_clusters.append(query)
                source_clusters.append(source)

    result = nested_loqdo_predictions(
        np.asarray(rows, dtype=np.float64),
        np.asarray(utility, dtype=np.float64),
        query_clusters,
        source_clusters=source_clusters,
        alphas=(0.1,),
        feature_names=("query_feature", "source_feature", "seed_feature"),
    )

    assert len(rows) == 8 * 7 * 9
    for fold in result.folds:
        assert fold.strict_source_domain_exclusion
        assert len(fold.heldout_row_indices) == 7 * 9 == 63
        assert fold.model.observation_count == 7 * 6 * 9 == 378
        assert fold.heldout_query_cluster not in fold.model.training_query_clusters
        assert fold.heldout_query_cluster not in fold.training_source_clusters

    leaked_sources = list(source_clusters)
    leaked_sources[0] = query_clusters[0]
    with pytest.raises(ProtocolError, match="held-out domain as source"):
        nested_loqdo_predictions(
            np.asarray(rows, dtype=np.float64),
            np.asarray(utility, dtype=np.float64),
            query_clusters,
            source_clusters=leaked_sources,
            alphas=(0.1,),
            feature_names=("query_feature", "source_feature", "seed_feature"),
        )


def test_robust_optimizer_returns_feasible_dense_nonuniform_solution() -> None:
    sources = _sources(7)
    marginal = {
        source: value
        for source, value in zip(sources, (3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0), strict=True)
    }
    solution = robust_local_utility_weights(
        marginal,
        np.eye(7, dtype=np.float64) * 0.01,
        kappa=0.1,
        l2_penalty=0.01,
    )
    weights = np.asarray([solution.weights[source] for source in sources])

    assert not solution.used_uniform_fallback
    assert solution.objective_value > 0.0
    assert weights.sum() == pytest.approx(1.0, abs=1e-10)
    assert weights.min() >= -1e-10
    assert weights.max() <= 0.25 + 1e-9
    assert np.dot(weights, weights) <= 1.0 / 6.0 + 1e-9
    assert solution.effective_source_count >= 6.0 - 1e-8


def test_robust_optimizer_has_exact_uniform_fallback_and_rejects_bad_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import midogpp_thesis.cvae.routing.local_marginal_utility.optimizer as optimizer

    sources = _sources(8)
    marginal = {source: 1.0 for source in sources}
    solution = robust_local_utility_weights(marginal, np.eye(8))
    assert solution.used_uniform_fallback
    assert solution.fallback_reason == "nonpositive_robust_gain_uniform"
    assert solution.weights == {source: 0.125 for source in sources}
    assert solution.delta == {source: 0.0 for source in sources}

    non_psd = np.eye(8)
    non_psd[0, 0] = -1.0
    with pytest.raises(ProtocolError, match="positive semidefinite"):
        robust_local_utility_weights(marginal, non_psd)
    nonfinite = np.eye(8)
    nonfinite[0, 0] = np.nan
    with pytest.raises(ProtocolError, match="finite"):
        robust_local_utility_weights(marginal, nonfinite)

    def fail_solver(*args: object, **kwargs: object) -> object:
        raise RuntimeError("forced failure")

    monkeypatch.setattr(optimizer, "minimize", fail_solver)
    failed = optimizer.robust_local_utility_weights(
        {source: float(index) for index, source in enumerate(sources)}, np.eye(8)
    )
    assert failed.used_uniform_fallback
    assert failed.fallback_reason == "solver_failure_uniform"
    assert failed.weights == {source: 0.125 for source in sources}
