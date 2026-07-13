from __future__ import annotations

import numpy as np

from midogpp_thesis.cvae.generation_samplers import (
    DIAGONAL_SAMPLER,
    FULL_SAMPLER,
    STANDARD_SAMPLER,
    aggregate_posterior_moments,
    fit_aggregate_posterior_sampler,
    sample_latents,
)


def test_total_moments_match_row_then_posterior_monte_carlo() -> None:
    rng = np.random.default_rng(3)
    mu = rng.normal(size=(12, 3))
    logvar = rng.normal(loc=-0.7, scale=0.2, size=(12, 3))
    mean, covariance, _, _ = aggregate_posterior_moments(mu, logvar)
    indices = rng.integers(0, len(mu), size=200_000)
    draws = mu[indices] + rng.normal(size=(len(indices), 3)) * np.exp(0.5 * logvar[indices])
    assert np.allclose(draws.mean(axis=0), mean, atol=0.02)
    assert np.allclose(np.cov(draws, rowvar=False, bias=True), covariance, atol=0.04)


def test_full_sampler_is_deterministic_and_records_realized_families() -> None:
    rng = np.random.default_rng(7)
    labels = np.asarray([0] * 70 + [1] * 70)
    mu = rng.normal(size=(140, 4)) + labels[:, None] * 0.5
    logvar = np.full_like(mu, -1.0)
    sampler = fit_aggregate_posterior_sampler(
        mu,
        logvar,
        labels,
        family=FULL_SAMPLER,
        source_row_hash="rows",
    )
    assert set(sampler.realized_family_by_class().values()).issubset({FULL_SAMPLER, DIAGONAL_SAMPLER, STANDARD_SAMPLER})
    first = sample_latents(sampler, [0, 1, 0, 1], seed=42)
    second = sample_latents(sampler, [0, 1, 0, 1], seed=42)
    assert np.array_equal(first, second)


def test_small_classes_fail_closed_to_standard() -> None:
    mu = np.zeros((4, 2))
    logvar = np.zeros((4, 2))
    labels = [0, 0, 1, 1]
    sampler = fit_aggregate_posterior_sampler(
        mu,
        logvar,
        labels,
        family=DIAGONAL_SAMPLER,
        source_row_hash="rows",
        min_class_count=3,
    )
    assert sampler.realized_family_by_class() == {"0": STANDARD_SAMPLER, "1": STANDARD_SAMPLER}
    assert not sampler.requested_family_realized_for_both_classes
