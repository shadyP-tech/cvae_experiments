from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from midogpp_thesis.cvae.objectives import ISOTROPIC_OBJECTIVE
from midogpp_thesis.cvae.generation_samplers import (
    DIAGONAL_SAMPLER,
    fit_aggregate_posterior_sampler,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.checkpoint_store import (
    StudyCheckpointStore,
    validate_study_checkpoint_index,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.contracts import (
    LEARNED_CONDITIONAL_DIAGONAL_PRIOR,
    LEARNED_PRIOR_MODE,
    LEARNED_PRIOR_MODEL_FAMILY,
    STANDARD_MODEL_FAMILY,
    STANDARD_NORMAL_PRIOR,
    StudyTrainingKey,
    StudyTrainingVariant,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.training import (
    encode_runtime,
    paired_epsilon,
    state_key_partitions,
    train_study_cvae,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.prior_runner import (
    _learned_prior_record,
    _learned_prior_sampler_rows,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.prior_validation import (
    _validate_prior_state_record,
)


def _variant(*, learned: bool) -> StudyTrainingVariant:
    return StudyTrainingVariant(
        study_mode=LEARNED_PRIOR_MODE,
        study_version="v2",
        model_family=(
            LEARNED_PRIOR_MODEL_FAMILY if learned else STANDARD_MODEL_FAMILY
        ),
        prior_family=(
            LEARNED_CONDITIONAL_DIAGONAL_PRIOR
            if learned
            else STANDARD_NORMAL_PRIOR
        ),
        objective_id=ISOTROPIC_OBJECTIVE,
        alpha=0.0,
        raw_fisher_state_hash="none",
        objective_context_hash="none",
        hidden_dim=8,
        latent_dim=3,
        num_hidden_layers=2,
        train_epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        beta_final=1e-3,
        kl_warmup_epochs=1,
        network_gradient_clip_norm=5.0,
        prior_learning_rate_multiplier=1.0,
        prior_weight_decay=0.0,
        prior_gradient_clip_norm=5.0,
    )


def _key(variant: StudyTrainingVariant) -> StudyTrainingKey:
    return StudyTrainingKey(
        study_id="learned_conditional_prior_source_inner_v2",
        study_version="v2",
        outer_target_center="0",
        inner_pseudo_target_center="1",
        fit_centers=("2", "3", "5", "6", "7", "8", "9"),
        fit_row_hash="fit-row-hash",
        frame_hash="frame-hash",
        feature_cache_hash="feature-cache-hash",
        manifest_hash="manifest-hash",
        protocol_hash="protocol-hash",
        training_seed=17,
        variant=variant,
    )


def _training_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(19)
    labels = np.asarray([0] * 16 + [1] * 16, dtype=np.int64)
    embeddings = rng.normal(size=(32, 4)).astype(np.float32)
    embeddings[:, 0] += labels.astype(np.float32) * 1.5
    return embeddings, labels


def test_a_and_e_pair_shared_initialization_and_training_stream() -> None:
    embeddings, labels = _training_data()
    standard_variant = _variant(learned=False)
    learned_variant = _variant(learned=True)

    runtime_a = train_study_cvae(
        embeddings,
        labels,
        variant=standard_variant,
        training_key=_key(standard_variant),
        model_family=STANDARD_MODEL_FAMILY,
        device="cpu",
    )
    runtime_e = train_study_cvae(
        embeddings,
        labels,
        variant=learned_variant,
        training_key=_key(learned_variant),
        model_family=LEARNED_PRIOR_MODEL_FAMILY,
        device="cpu",
    )

    assert runtime_a.shared_initialization_hash == runtime_e.shared_initialization_hash
    assert runtime_a.training_stream_hash == runtime_e.training_stream_hash
    assert runtime_a.training_key.hash != runtime_e.training_key.hash
    assert state_key_partitions(runtime_a.model)["prior"] == []
    assert state_key_partitions(runtime_e.model)["prior"] == [
        "latent_prior.prior_mu",
        "latent_prior.prior_rho",
    ]
    assert all(torch.isfinite(parameter).all() for parameter in runtime_e.model.parameters())
    assert len(runtime_e.diagnostics) == 2
    assert all(
        {
            "prior_mu_min",
            "prior_mu_max",
            "effective_logvar_min",
            "effective_logvar_max",
            "prior_std_min",
            "prior_std_max",
            "prior_saturation_count",
            "prior_saturated",
        }.issubset(row)
        for row in runtime_e.diagnostics
    )
    mu_a, logvar_a = encode_runtime(runtime_a, embeddings, labels)
    sampler_c = fit_aggregate_posterior_sampler(
        mu_a,
        logvar_a,
        labels,
        family=DIAGONAL_SAMPLER,
        source_row_hash="fit-row-hash",
        min_class_count=2,
    )
    mu_e, logvar_e = encode_runtime(runtime_e, embeddings, labels)
    record, integrity_valid, _ = _learned_prior_record(
        runtime_e,
        posterior_mu=mu_e,
        posterior_logvar=logvar_e,
        labels=labels,
        sampler_c=sampler_c,
        outer="0",
        inner="1",
        training_seed=17,
    )
    assert integrity_valid
    assert set(record["posterior_sufficient_statistics_by_class"]) == {"0", "1"}
    assert set(record["kl_to_learned_prior_by_class"]) == {"0", "1"}
    assert len(record["prior_training_trajectory"]) == 2
    assert len(record["final_prior_partition_hash"]) == 16
    sampler_rows = _learned_prior_sampler_rows(record)
    assert len(sampler_rows) == 2
    assert {row["class_label"] for row in sampler_rows} == {0, 1}
    assert all(row["sampler_state_hash"] == record["state_hash"] for row in sampler_rows)

    # The derived log-variance can differ by a few float32 ulps between CPU
    # and GPU tanh kernels. It remains audit data, but must not alter the
    # canonical parameter-partition identity shared with the checkpoint.
    record["state"]["effective_logvar"][0][0] += 5e-9
    _validate_prior_state_record(
        record,
        latent_dim=learned_variant.latent_dim,
        train_epochs=learned_variant.train_epochs,
    )


def test_learned_prior_checkpoint_round_trip_is_strict_and_content_addressed(
    tmp_path,
) -> None:
    embeddings, labels = _training_data()
    variant = replace(_variant(learned=True), train_epochs=1)
    key = _key(variant)
    runtime = train_study_cvae(
        embeddings,
        labels,
        variant=variant,
        training_key=key,
        model_family=LEARNED_PRIOR_MODEL_FAMILY,
        device="cpu",
    )

    store = StudyCheckpointStore(tmp_path)
    record = store.save(runtime)
    store.write_indices()
    loaded = StudyCheckpointStore(tmp_path).load(
        training_key=key,
        variant=variant,
        input_dim=embeddings.shape[1],
        device="cpu",
    )

    assert loaded is not None
    assert loaded.resumed_from_checkpoint
    assert loaded.checkpoint_hash == runtime.checkpoint_hash
    assert record["training_key_hash"] == key.hash
    assert record["prior_partition_hash"] != "none"
    assert validate_study_checkpoint_index(tmp_path)["n_unique_training_keys"] == 1


def test_evaluation_epsilon_is_training_seed_and_arm_neutral() -> None:
    labels = [0, 1, 0, 1, 1, 0]
    first, first_hash = paired_epsilon(
        study_id="learned_conditional_prior_source_inner_v2",
        outer_target_center="0",
        inner_pseudo_target_center="1",
        generation_seed=17,
        labels=labels,
        latent_dim=3,
        stream="prior_generation",
    )
    repeated, repeated_hash = paired_epsilon(
        study_id="learned_conditional_prior_source_inner_v2",
        outer_target_center="0",
        inner_pseudo_target_center="1",
        generation_seed=17,
        labels=labels,
        latent_dim=3,
        stream="prior_generation",
    )
    changed, changed_hash = paired_epsilon(
        study_id="learned_conditional_prior_source_inner_v2",
        outer_target_center="0",
        inner_pseudo_target_center="1",
        generation_seed=42,
        labels=labels,
        latent_dim=3,
        stream="prior_generation",
    )

    assert np.array_equal(first, repeated)
    assert first_hash == repeated_hash
    assert not np.array_equal(first, changed)
    assert first_hash != changed_hash
