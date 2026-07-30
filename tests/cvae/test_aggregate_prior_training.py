from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from midogpp_thesis.cvae.models import AggregateMatchedMixturePriorCVAE
from midogpp_thesis.cvae.preservation.aggregate_prior_study.config import (
    load_aggregate_prior_study_config,
)
from midogpp_thesis.cvae.preservation.aggregate_prior_study.contracts import (
    ARMS,
    SourceExpertTrainingKey,
)
from midogpp_thesis.cvae.preservation.aggregate_prior_study.training import (
    train_source_expert_panel,
)


CONFIG = Path(
    "experiments/midogpp/stages/20_cvae_preservation/configs/"
    "aggregate_posterior_mixture_geco_source_inner_v3.yaml"
)


def test_four_arms_share_warmup_and_stream_and_freeze_mixture() -> None:
    base = load_aggregate_prior_study_config(CONFIG)
    config = replace(
        base,
        device="cpu",
        pca_dim=6,
        latent_dim=3,
        hidden_dim=8,
        warmup_epochs=1,
        continuation_epochs=3,
        batch_size=16,
        kl_warmup_epochs=1,
        refit_interval_epochs=1,
        final_stabilization_epochs=1,
        minimum_component_rows=2,
        minimum_component_cases=1,
    )
    rng = np.random.default_rng(12)
    labels = np.asarray([0] * 40 + [1] * 40)
    modes = np.asarray([0] * 20 + [1] * 20 + [0] * 20 + [1] * 20)
    embeddings = (
        rng.normal(scale=0.2, size=(80, 6))
        + modes[:, None] * np.asarray([2.5, -1.5, 0.8, 0.0, 0.4, -0.2])
        + labels[:, None] * 0.3
    ).astype(np.float32)
    keys = {
        arm: SourceExpertTrainingKey(
            source_center="2",
            training_seed=17,
            arm=arm,
            source_row_hash="rows",
            source_case_hash="cases",
            source_frame_hash="frame",
            manifest_hash="manifest",
            feature_cache_hash="cache",
            protocol_hash="protocol",
            config_hash=config.contract_hash,
        )
        for arm in ARMS
    }
    runtimes = train_source_expert_panel(
        embeddings,
        labels,
        [f"case-{index}" for index in range(80)],
        config=config,
        training_keys=keys,
    )
    assert tuple(runtimes) == ARMS
    assert len({runtime.warmup_checkpoint_hash for runtime in runtimes.values()}) == 1
    assert len({runtime.training_stream_hash for runtime in runtimes.values()}) == 1
    assert runtimes["SF"].geco_state is None
    assert runtimes["KF"].geco_state is None
    assert runtimes["SG"].geco_state is not None
    assert runtimes["KG"].geco_state is not None
    for arm in ("KF", "KG"):
        model = runtimes[arm].model
        assert isinstance(model, AggregateMatchedMixturePriorCVAE)
        assert all(
            not parameter.requires_grad and parameter.grad is None
            for parameter in model.latent_prior.parameters()
        )
        assert [
            int(row["after_continuation_epoch"])
            for row in runtimes[arm].mixture_refit_records
        ] == [0, 1, 2]
