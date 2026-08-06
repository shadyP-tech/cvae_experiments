from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from midogpp_thesis.cvae.generation_samplers import FULL_SAMPLER, STANDARD_SAMPLER
from midogpp_thesis.cvae.models import ClassConditionedCVAE
from midogpp_thesis.cvae.preservation.cli import build_parser
from midogpp_thesis.cvae.preservation.uniform_b_optimized_prior.config import (
    load_optimized_prior_config,
)
from midogpp_thesis.cvae.preservation.uniform_b_optimized_prior.contracts import (
    ARMS, EXPERIMENT_ID, P0, PS, OptimizedTrainingKey, legal_sources,
)
from midogpp_thesis.cvae.preservation.uniform_b_optimized_prior.core import (
    fit_source_sampler, load_checkpoint, save_checkpoint, train_optimized_checkpoint,
)
from midogpp_thesis.cvae.preservation.uniform_b_task_geometry.composition import (
    compose_generated_blocks,
)
from midogpp_thesis.cvae.preservation.uniform_b_task_geometry.generation import (
    GeneratedBlock,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/midogpp/stages/20_cvae_preservation/configs/uniform_b_geco_aggregate_prior_union_source_inner_v2.yaml"


def test_v2_config_locks_capacity_composition_and_runtime_outside_contract() -> None:
    config = load_optimized_prior_config(CONFIG)
    assert config.pca_output_dim == 256
    assert config.hidden_dim == 1024
    assert config.latent_dim == 64
    assert config.num_hidden_layers == 3
    assert config.total_steps == 4000
    assert config.arms == ARMS
    assert config.total_generation_per_class == 1024
    assert config.classifier_c == 0.01
    assert config.runtime_training_devices == ("cuda:0", "cuda:1")
    assert config.runtime_scoring_workers == 12
    changed_runtime = replace(
        config,
        runtime_training_devices=("cpu",),
        runtime_scoring_workers=1,
    )
    assert changed_runtime.contract_hash == config.contract_hash


def test_v2_cli_surface_is_registered() -> None:
    parsed = build_parser().parse_args(
        ["source-inner-uniform-b-geco-aggregate-prior-union", "--config", str(CONFIG)]
    )
    assert parsed.surface == "source-inner-uniform-b-geco-aggregate-prior-union"


def test_cvae_depth_generalization_preserves_v1_state_layout() -> None:
    v1 = ClassConditionedCVAE(128, hidden_dim=32, latent_dim=8, num_hidden_layers=2)
    assert tuple(v1.state_dict()) == (
        "encoder.0.weight", "encoder.0.bias", "encoder.2.weight", "encoder.2.bias",
        "fc_mu.weight", "fc_mu.bias", "fc_logvar.weight", "fc_logvar.bias",
        "decoder.0.weight", "decoder.0.bias", "decoder.2.weight", "decoder.2.bias",
        "decoder.4.weight", "decoder.4.bias",
    )
    v2 = ClassConditionedCVAE(256, hidden_dim=48, latent_dim=12, num_hidden_layers=3)
    x = torch.randn(6, 256)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    reconstruction, mu, logvar = v2(x, y)
    assert reconstruction.shape == (6, 256)
    assert mu.shape == logvar.shape == (6, 12)
    assert "encoder.4.weight" in v2.state_dict()
    assert "decoder.6.weight" in v2.state_dict()


def test_three_layer_keyed_training_checkpoint_round_trip(tmp_path: Path) -> None:
    config = replace(
        load_optimized_prior_config(CONFIG),
        hidden_dim=32,
        latent_dim=8,
        batch_size=32,
        warmup_steps=1,
        total_steps=2,
    )
    labels = tuple([0] * 32 + [1] * 32)
    case_ids = tuple(f"case-{index}" for index in range(64))
    sample_ids = tuple(f"sample-{index}" for index in range(64))
    projected = np.random.default_rng(3).normal(size=(64, 256)).astype(np.float32)
    key = OptimizedTrainingKey(
        source_center="0", training_seed=17, source_row_hash="rows",
        source_case_hash="cases", frame_hash="frame",
        manifest_hash="manifest", feature_cache_hash="cache",
        config_hash=config.contract_hash,
    )
    runtime = train_optimized_checkpoint(
        projected, labels, case_ids, sample_ids,
        source_identity_hash="source", config=config, training_key=key,
        device="cpu",
    )
    assert runtime.state.completed_step == 2
    assert runtime.state.model.num_hidden_layers == 3
    record = save_checkpoint(tmp_path, runtime)
    restored = load_checkpoint(tmp_path, key.hash, config, device="cpu")
    assert restored is not None
    state, loaded_record = restored
    assert state.model.num_hidden_layers == 3
    assert loaded_record["checkpoint_hash"] == record["checkpoint_hash"]


def test_shrinkage_prior_uses_all_or_none_class_fallback() -> None:
    config = load_optimized_prior_config(CONFIG)
    model = ClassConditionedCVAE(
        256,
        hidden_dim=32,
        latent_dim=64,
        num_hidden_layers=3,
    )
    rng = np.random.default_rng(7)
    projected = rng.normal(size=(90, 256)).astype(np.float32)
    labels = np.asarray([0] * 70 + [1] * 20, dtype=np.int64)
    fitted, effective, _, _ = fit_source_sampler(
        model,
        projected,
        labels,
        source_row_hash="source_rows",
        config=replace(config, hidden_dim=32),
        device="cpu",
    )
    assert fitted.requested_family == FULL_SAMPLER
    assert fitted.classes[0].realized_family == FULL_SAMPLER
    assert fitted.classes[1].realized_family == STANDARD_SAMPLER
    assert not fitted.requested_family_realized_for_both_classes
    assert effective.requested_family == STANDARD_SAMPLER
    assert all(
        state.realized_family == STANDARD_SAMPLER
        for state in effective.classes.values()
    )


def _block(source: str, arm: str) -> GeneratedBlock:
    per_class = 16
    value = float(int(source) + (0.25 if arm == PS else 0.0))
    embeddings = np.full((2 * per_class, 3840), value, dtype=np.float32)
    labels = np.asarray([0] * per_class + [1] * per_class, dtype=np.int64)
    return GeneratedBlock(
        source_center=source,
        arm=arm,
        training_seed=17,
        generation_seed=42,
        embeddings=embeddings,
        labels=labels,
        per_class=per_class,
        checkpoint_hash=f"checkpoint-{source}",
        frame_hash=f"frame-{source}",
        stream_hash=f"stream-{source}-{arm}",
    )


def test_legal_union_has_seven_sources_fixed_budget_and_paired_shuffle() -> None:
    centers = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
    sources = legal_sources(centers, outer_center="0", inner_center="1")
    assert len(sources) == 7
    p0 = compose_generated_blocks(
        {source: _block(source, P0) for source in sources},
        mode="union_equal_total",
        base_per_class=70,
        shuffle_seed=123,
    )
    ps = compose_generated_blocks(
        {source: _block(source, PS) for source in sources},
        mode="union_equal_total",
        base_per_class=70,
        shuffle_seed=123,
    )
    assert p0.embeddings.shape == ps.embeddings.shape == (140, 3840)
    assert all(counts == {0: 10, 1: 10} for counts in p0.source_counts.values())
    assert p0.shuffle_hash == ps.shuffle_hash


def test_registry_and_catalog_bind_v2_non_consumable_protocol() -> None:
    registry = yaml.safe_load((ROOT / "experiments/midogpp/registry.yaml").read_text())
    experiments = registry["experiments"]
    entry = next(item for item in experiments if item["experiment_id"] == EXPERIMENT_ID)
    assert entry["runner"]["environment"]["MIDOGPP_OPTIMIZED_PRIOR_TRAINING_DEVICES"] == "cuda:0,cuda:1"
    assert entry["runner"]["environment"]["MIDOGPP_OPTIMIZED_PRIOR_SCORING_WORKERS"] == "12"
    catalog = yaml.safe_load((ROOT / "experiments/midogpp/artifact_catalog.yaml").read_text())
    artifact = next(
        item for item in catalog["artifacts"]
        if item["artifact_id"] == "midogpp_output_cvae_uniform_b_geco_aggregate_prior_union_source_inner_v2"
    )
    assert artifact["may_feed_recipe_selection"] is False
    assert artifact["may_feed_deployable_selection"] is False
    assert "reports/validation_report.json" in artifact["required_files"]
