from __future__ import annotations

import numpy as np
import pytest
import torch

from midogpp_thesis.cvae.generation_samplers import standard_normal_sampler
from midogpp_thesis.cvae.objectives import ISOTROPIC_OBJECTIVE, TASK_FISHER_OBJECTIVE
from midogpp_thesis.cvae.preservation.representations import (
    decode_means,
    encode_posterior,
    posterior_samples,
    sampler_decodes,
)
from midogpp_thesis.cvae.training import (
    TrainingKey,
    TrainingVariant,
    train_cvae,
    training_variant_hash,
)


def test_stochastic_training_is_key_deterministic_and_aggregates_epochs() -> None:
    rng = np.random.default_rng(5)
    labels = np.asarray([0, 1] * 8)
    embeddings = rng.normal(size=(16, 4)).astype("float32")
    embeddings[:, 0] += labels
    variant = TrainingVariant(
        objective_id=ISOTROPIC_OBJECTIVE,
        hidden_dim=8,
        latent_dim=2,
        train_epochs=2,
        batch_size=4,
        kl_warmup_epochs=1,
    )
    key = TrainingKey(
        fit_centers=("0", "1"),
        fit_row_hash="rows",
        objective_id=variant.objective_id,
        training_seed=17,
        frame_hash="frame",
        dataset_contract_hash="manifest",
        feature_cache_hash="features",
        backbone_output_frame_id="virchow2:2560",
        protocol_hash="protocol",
        code_version="test",
        variant_hash=training_variant_hash(variant),
        stochastic_pairing_hash="paired-stream",
        objective_context_hash="none",
    )
    first = train_cvae(embeddings, labels, variant=variant, training_key=key)
    second = train_cvae(embeddings, labels, variant=variant, training_key=key)
    assert first.checkpoint_hash == second.checkpoint_hash
    assert len(first.diagnostics) == 2
    assert all(row["n_rows"] == 16 for row in first.diagnostics)


def test_task_fisher_and_isotropic_arms_share_initialization_and_stochastic_streams() -> None:
    labels = np.asarray([0, 1] * 8)
    embeddings = np.random.default_rng(9).normal(size=(16, 4)).astype("float32")
    isotropic = TrainingVariant(hidden_dim=8, latent_dim=2, train_epochs=1, batch_size=4)
    task = TrainingVariant(
        objective_id=TASK_FISHER_OBJECTIVE,
        hidden_dim=8,
        latent_dim=2,
        train_epochs=1,
        batch_size=4,
        alpha=1.0,
    )

    def key(variant: TrainingVariant, context: str) -> TrainingKey:
        return TrainingKey(
            fit_centers=("0", "1"),
            fit_row_hash="rows",
            objective_id=variant.objective_id,
            training_seed=17,
            frame_hash="frame",
            dataset_contract_hash="manifest",
            feature_cache_hash="features",
            backbone_output_frame_id="virchow2:pca",
            protocol_hash="protocol",
            code_version="test",
            variant_hash=training_variant_hash(variant),
            stochastic_pairing_hash="same-paired-stream",
            objective_context_hash=context,
        )

    arm_a = train_cvae(embeddings, labels, variant=isotropic, training_key=key(isotropic, "none"))
    arm_b = train_cvae(
        embeddings,
        labels,
        variant=task,
        training_key=key(task, "fisher"),
        task_metric=np.eye(4),
    )
    assert arm_a.initialization_hash == arm_b.initialization_hash
    assert arm_a.stochastic_stream_hash == arm_b.stochastic_stream_hash


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA replay requires a GPU")
def test_cuda_replay_and_all_representation_builders_are_deterministic() -> None:
    labels = np.asarray([0, 1] * 8)
    embeddings = np.random.default_rng(13).normal(size=(16, 4)).astype("float32")
    isotropic = TrainingVariant(hidden_dim=8, latent_dim=2, train_epochs=2, batch_size=4)
    task = TrainingVariant(
        objective_id=TASK_FISHER_OBJECTIVE,
        hidden_dim=8,
        latent_dim=2,
        train_epochs=2,
        batch_size=4,
        alpha=1.0,
    )

    def key(variant: TrainingVariant, context: str) -> TrainingKey:
        return TrainingKey(
            fit_centers=("0", "1"),
            fit_row_hash="cuda-rows",
            objective_id=variant.objective_id,
            training_seed=17,
            frame_hash="cuda-frame",
            dataset_contract_hash="manifest",
            feature_cache_hash="features",
            backbone_output_frame_id="virchow2:pca",
            protocol_hash="cuda-protocol",
            code_version="test",
            variant_hash=training_variant_hash(variant),
            stochastic_pairing_hash="cuda-paired-stream",
            objective_context_hash=context,
        )

    first = train_cvae(
        embeddings,
        labels,
        variant=isotropic,
        training_key=key(isotropic, "none"),
        device="cuda:0",
    )
    replay = train_cvae(
        embeddings,
        labels,
        variant=isotropic,
        training_key=key(isotropic, "none"),
        device="cuda:0",
    )
    task_runtime = train_cvae(
        embeddings,
        labels,
        variant=task,
        training_key=key(task, "fisher"),
        task_metric=np.eye(4),
        device="cuda:0",
    )
    assert first.device == replay.device == task_runtime.device == "cuda:0"
    assert first.checkpoint_hash == replay.checkpoint_hash
    assert first.initialization_hash == task_runtime.initialization_hash
    assert first.stochastic_stream_hash == task_runtime.stochastic_stream_hash

    sampler = standard_normal_sampler(latent_dim=2, source_row_hash="cuda-rows")
    decoded, mu, logvar = decode_means(first, embeddings, labels)
    posterior = posterior_samples(first, embeddings, labels, seed=17)[0]
    posterior_replay = posterior_samples(first, embeddings, labels, seed=17)[0]
    prior = sampler_decodes(first, sampler, labels, seed=17)
    prior_replay = sampler_decodes(first, sampler, labels, seed=17)
    encoded_mu, encoded_logvar = encode_posterior(first, embeddings, labels)
    for value in (decoded, mu, logvar, posterior, prior, encoded_mu, encoded_logvar):
        assert np.asarray(value).shape[0] == len(labels)
        assert np.isfinite(value).all()
    np.testing.assert_array_equal(posterior, posterior_replay)
    np.testing.assert_array_equal(prior, prior_replay)
