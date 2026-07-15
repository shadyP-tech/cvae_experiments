from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from midogpp_thesis.cvae.generation_samplers import (
    DIAGONAL_SAMPLER,
    FULL_SAMPLER,
    STANDARD_SAMPLER,
)
from midogpp_thesis.cvae.preservation.prior_recovery_config import (
    SAMPLER_FALLBACK_POLICY,
    SAMPLER_VIABILITY_POLICY,
    STABILITY_CONSENSUS_RULE,
    OuterPriorRecoveryConfig,
    PriorRecoveryConfig,
    SourceInnerPriorRecoveryConfig,
    SourceInnerStabilityConfig,
)
from midogpp_thesis.cvae.training import TrainingVariant


def prior_recovery_config(
    *,
    mode: str,
    artifact_root: Path,
    manifest: Path,
    cache: Path,
    reference: Path | None = None,
    locks: Path | None = None,
) -> PriorRecoveryConfig:
    base = TrainingVariant(
        hidden_dim=12,
        latent_dim=3,
        train_epochs=1,
        batch_size=16,
        kl_warmup_epochs=1,
    )
    common = dict(
        name=f"test_{mode}",
        artifact_root=artifact_root,
        manifest_path=manifest,
        feature_cache_path=cache,
        heldout_centers=("0",),
        expected_feature_dim=6,
        pca_dim=4,
        selection_training_seed=42,
        generation_seeds=(17,),
        device="cpu",
        isotropic_variant=base,
        task_fisher_variant=TrainingVariant(
            **(
                base.to_payload()
                | {"objective_id": "stochastic_task_fisher_v1", "alpha": 1.0}
            )
        ),
        sampler_min_class_count=2,
        sampler_max_condition_number=1e6,
        sampler_families=(STANDARD_SAMPLER, DIAGONAL_SAMPLER, FULL_SAMPLER),
        sampler_fallback_policy=SAMPLER_FALLBACK_POLICY,
        sampler_viability_policy=SAMPLER_VIABILITY_POLICY,
        gate_min_ratio_improvement=0.04,
        gate_min_inner_wins=1,
        sampler_tie_margin=0.01,
        task_increment_min_ratio=0.01,
        safety_max_bacc_regression=0.01,
        minimum_real_bacc=0.55,
        code_version="test",
    )
    if mode == "source_inner":
        return SourceInnerPriorRecoveryConfig(**common)
    if mode == "stability":
        return SourceInnerStabilityConfig(
            **(
                common
                | {
                    "name": "test_source_inner_training_seed_stability",
                    "selection_training_seed": 17,
                    "generation_seeds": (17, 42),
                    "code_version": "test_stability",
                }
            ),
            training_seeds=(17, 42),
            consensus_rule_id=STABILITY_CONSENSUS_RULE,
            child_code_version="test",
        )
    if mode != "outer" or reference is None or locks is None:
        raise ValueError("Outer test config requires reference and lock roots.")
    return OuterPriorRecoveryConfig(
        **common,
        reference_artifact_root=reference,
        recipe_lock_artifact_root=locks,
        training_seeds=(17,),
        positive_claim_min_ratio=0.80,
        positive_claim_min_center_wins=1,
    )


def write_prior_recovery_fixture(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    manifest = root / "midogpp_manifest.csv"
    cache = root / "virchow2_midogpp_train.npz"
    rows: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    embeddings = []
    rng = np.random.default_rng(31)
    index = 0
    for center_index, center in enumerate(("0", "1", "2", "3")):
        for local in range(16):
            label = local % 2
            sample_id = f"s{index}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "case_id": f"case{index}",
                    "label": label,
                    "split": "train",
                    "center": center,
                }
            )
            metadata.append(
                {"sample_id": sample_id, "label": label, "center": center, "split": "train"}
            )
            vector = rng.normal(scale=0.4, size=6)
            vector[0] += label * 4.0 + center_index * 0.03
            embeddings.append(vector)
            index += 1
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(
        cache,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        metadata_json=json.dumps(metadata),
        feature_extractor_json=json.dumps(
            {"backbone_type": "virchow2", "dataset": "midogpp"}
        ),
    )
    return manifest, cache
