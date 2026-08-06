from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from midogpp_thesis.cvae.generation_samplers import FULL_SAMPLER
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.dense_residual_soft_router import (
    CALIBRATION_SEMANTICS,
    CLASS_PRIOR,
    ENERGY_SEMANTICS,
    ReplicaKey,
    assert_outer_query_source_exclusions,
    build_hamilton_allocation,
    calibrate_own_source_energies,
    compose_prefix_blocks,
    deterministic_case_partitions,
    gaussian_kl_diagonal_to_full,
    hamilton_allocate,
    residual_soft_weights,
    score_variational_compatibility,
)


class _IdentityFrame:
    input_dim = 2
    output_dim = 2

    def transform(self, values: object) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)

    def inverse_transform(self, values: object) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)


class _ToyCVAE(torch.nn.Module):
    input_dim = 2
    latent_dim = 2
    n_classes = 2

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def encode(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        offset = y.to(dtype=x.dtype).view(-1, 1) * 0.25
        return x * 0.5 + offset + self.anchor * 0.0, torch.zeros_like(x)

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del y
        return z + self.anchor * 0.0


def _expert(*, realized_family: str = FULL_SAMPLER) -> SimpleNamespace:
    states = {
        label: SimpleNamespace(
            realized_family=realized_family,
            mean=np.full(2, 0.25 * label, dtype=np.float64),
            covariance=np.eye(2, dtype=np.float64),
        )
        for label in (0, 1)
    }
    return SimpleNamespace(
        source_center="1",
        training_seed=17,
        model=_ToyCVAE(),
        source_frame=SimpleNamespace(frame=_IdentityFrame()),
        sampler=SimpleNamespace(latent_dim=2, classes=states),
    )


def _replica_case_map(
    candidates: tuple[str, ...],
    seeds: tuple[int, ...],
    value_factory: object,
) -> dict[ReplicaKey, tuple[float, ...]]:
    return {
        ReplicaKey(source, seed): tuple(value_factory(source, seed))  # type: ignore[operator]
        for source in candidates
        for seed in seeds
    }


def test_case_partition_is_deterministic_label_blind_and_whole_case() -> None:
    samples = ("s0", "s1", "s2", "s3", "s4", "s5")
    cases = ("a", "a", "b", "c", "c", "d")
    first = deterministic_case_partitions(
        samples,
        cases,
        target_center="7",
        support_case_count=2,
        split_seed=20260806,
    )
    order = (5, 2, 0, 4, 1, 3)
    reordered = deterministic_case_partitions(
        tuple(samples[index] for index in order),
        tuple(cases[index] for index in order),
        target_center="7",
        support_case_count=2,
    )
    assert first.support_cases == reordered.support_cases
    assert first.split_seed == 20260806
    assert set(first.support_cases).isdisjoint(first.evaluation_cases)
    assert set(first.support_sample_ids).isdisjoint(first.evaluation_sample_ids)
    assert {
        cases[index] for index in first.support_indices
    } == set(first.support_cases)
    assert {
        cases[index] for index in first.evaluation_indices
    } == set(first.evaluation_cases)
    with pytest.raises(ProtocolError, match="non-boolean integer"):
        deterministic_case_partitions(
            samples, cases, target_center="7", split_seed=True
        )


def test_outer_query_exclusions_require_complete_nested_pool() -> None:
    centers = tuple(str(value) for value in range(9))
    candidates = tuple(value for value in centers if value not in {"0", "2"})
    assert assert_outer_query_source_exclusions(
        outer_target="0",
        query_center="2",
        candidate_sources=reversed(candidates),
        all_centers=centers,
    ) == candidates
    with pytest.raises(ProtocolError, match="Outer-target"):
        assert_outer_query_source_exclusions(
            outer_target="0", query_center="2", candidate_sources=("0", "1")
        )
    with pytest.raises(ProtocolError, match="query != outer"):
        assert_outer_query_source_exclusions(
            outer_target="2", query_center="2", candidate_sources=("0", "1")
        )


def test_analytic_diagonal_to_full_gaussian_kl() -> None:
    mu = np.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=np.float64)
    logvar = np.zeros_like(mu)
    observed = gaussian_kl_diagonal_to_full(
        mu,
        logvar,
        np.zeros(2, dtype=np.float64),
        np.eye(2, dtype=np.float64),
    )
    np.testing.assert_allclose(observed, (0.0, 0.5), atol=1e-12)

    covariance = np.diag((2.0, 4.0))
    expected = 0.5 * (0.5 + 0.25 - 2.0 + np.log(8.0))
    observed_full = gaussian_kl_diagonal_to_full(
        np.zeros((1, 2)), np.zeros((1, 2)), np.zeros(2), covariance
    )
    np.testing.assert_allclose(observed_full, (expected,), atol=1e-12)
    with pytest.raises(ProtocolError, match="positive definite"):
        gaussian_kl_diagonal_to_full(
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            np.zeros(2),
            np.asarray(((1.0, 0.0), (0.0, 0.0))),
        )


def test_compatibility_energy_is_class_marginal_and_case_equal() -> None:
    embeddings = np.asarray(((0.0, 0.0), (2.0, 0.0), (0.0, 2.0)))
    score = score_variational_compatibility(
        _expert(), embeddings, ("case-a", "case-a", "case-b")
    )
    assert score.energy_semantics == ENERGY_SEMANTICS
    assert score.class_prior == CLASS_PRIOR
    assert score.exact_nelbo is False
    assert score.labels_consumed is False
    assert score.per_row.shape == (3,)
    assert score.per_row.flags.writeable is False
    assert set(score.per_class_energy) == {0, 1}
    assert score.case_equal_mean == pytest.approx(
        np.mean((score.per_case["case-a"], score.per_case["case-b"]))
    )
    assert score.case_equal_mean != pytest.approx(float(score.per_row.mean()))
    with pytest.raises(ProtocolError, match="promoted full PS Gaussian"):
        score_variational_compatibility(
            _expert(realized_family="standard_normal"),
            embeddings,
            ("case-a", "case-a", "case-b"),
        )


def test_own_source_calibration_is_robust_and_uses_all_three_seeds() -> None:
    candidates = ("1", "2")
    seeds = (17, 42, 101)
    query = _replica_case_map(
        candidates,
        seeds,
        lambda source, seed: (3.0, 3.0) if source == "1" else (2.0, 2.0),
    )
    own = _replica_case_map(
        candidates,
        seeds,
        lambda source, seed: (1.0, 2.0, 3.0) if source == "1" else (2.0, 2.0),
    )
    result = calibrate_own_source_energies(
        query,
        own,
        candidate_sources=candidates,
        training_seeds=seeds,
        scale_floor=1e-6,
    )
    assert result.calibration_semantics == CALIBRATION_SEMANTICS
    assert len(result.replicas) == 6
    assert result.mean_z_by_source["1"] == pytest.approx(1.0 / 1.4826)
    assert result.mean_z_by_source["2"] == 0.0
    assert {
        row.scale_source for row in result.replicas if row.source_center == "1"
    } == {"scaled_mad"}
    assert {
        row.scale_source for row in result.replicas if row.source_center == "2"
    } == {"fixed_floor_fallback"}
    with pytest.raises(ProtocolError, match="every declared"):
        calibrate_own_source_energies(
            {key: value for key, value in query.items() if key.training_seed != 101},
            own,
            candidate_sources=candidates,
            training_seeds=seeds,
        )
    with pytest.raises(ProtocolError, match="frozen three-seed"):
        calibrate_own_source_energies(
            {
                key: value
                for key, value in query.items()
                if key.training_seed != 101
            },
            {
                key: value
                for key, value in own.items()
                if key.training_seed != 101
            },
            candidate_sources=candidates,
            training_seeds=(17, 42),
        )


def test_residual_weights_are_exact_uniform_for_rho_zero_and_ties() -> None:
    scores = {str(index): float(index) for index in range(8)}
    rho_zero = residual_soft_weights(scores, rho=0.0)
    expected = {str(index): 0.125 for index in range(8)}
    assert rho_zero.weights == expected
    tied = residual_soft_weights(
        {str(index): 4.0 for index in range(8)}, rho=0.5
    )
    assert tied.weights == expected
    assert tied.applied_rho == 0.0


def test_residual_weights_automatically_preserve_density() -> None:
    scores = {str(index): (0.0 if index == 0 else 50.0) for index in range(8)}
    result = residual_soft_weights(scores, rho=0.5)
    assert 0.0 < result.applied_rho < 0.5
    assert sum(result.weights.values()) == pytest.approx(1.0)
    assert max(result.weights.values()) <= 0.25 + 1e-12
    assert min(result.weights.values()) > 0.0
    assert result.effective_source_count >= 6.0 - 1e-10
    assert result.active_constraints


def test_positive_hamilton_allocation_has_canonical_ties_and_fixed_total() -> None:
    uniform = hamilton_allocate(
        {str(index): 1.0 for index in reversed(range(8))}, total=1024
    )
    assert uniform == {str(index): 128 for index in range(8)}
    tied = build_hamilton_allocation({"c": 1.0, "a": 1.0, "b": 1.0}, total=8)
    assert tied.allocations == {"a": 3, "b": 3, "c": 2}
    assert tied.remainder_order == ("a", "b", "c")
    sparse = hamilton_allocate({"a": 1.0, "b": 0.0, "c": 0.0}, total=9)
    assert sum(sparse.values()) == 9
    assert all(value > 0 for value in sparse.values())
    with pytest.raises(ProtocolError, match="positive lower bound"):
        hamilton_allocate({"a": 1.0, "b": 1.0}, total=4, minimum_per_source=0)


def test_prefix_composition_is_canonical_deterministic_and_prefix_only() -> None:
    def block(source_offset: float) -> dict[str, np.ndarray]:
        class_zero = np.asarray(
            [[source_offset + index, 0.0] for index in range(5)], dtype=np.float32
        )
        class_one = np.asarray(
            [[source_offset + index, 1.0] for index in range(5)], dtype=np.float32
        )
        return {
            "embeddings": np.concatenate((class_zero, class_one), axis=0),
            "labels": np.asarray([0] * 5 + [1] * 5, dtype=np.int64),
        }

    blocks = {"b": block(100.0), "a": block(0.0)}
    first = compose_prefix_blocks(
        blocks,
        {"a": 3, "b": 2},
        shuffle_seed_by_class={0: 11, 1: 12},
        total_per_class=5,
    )
    second = compose_prefix_blocks(
        {"a": blocks["a"], "b": blocks["b"]},
        {"b": 2, "a": 3},
        shuffle_seed_by_class={"1": 12, "0": 11},
        total_per_class=5,
    )
    np.testing.assert_array_equal(first.embeddings, second.embeddings)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert first.source_order == ("a", "b")
    assert first.composition_hash == second.composition_hash
    assert first.source_by_row.count("a") == 6
    assert first.source_by_row.count("b") == 4
    assert np.array_equal(first.labels, np.asarray([0] * 5 + [1] * 5))

    mutated_suffix = block(0.0)
    mutated_suffix["embeddings"][[3, 4, 8, 9]] += 9999.0
    suffix_changed = compose_prefix_blocks(
        {"a": mutated_suffix, "b": blocks["b"]},
        {"a": 3, "b": 2},
        shuffle_seed_by_class={0: 11, 1: 12},
        total_per_class=5,
    )
    np.testing.assert_array_equal(first.embeddings, suffix_changed.embeddings)
    assert first.composition_hash == suffix_changed.composition_hash
