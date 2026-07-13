from __future__ import annotations

from pathlib import Path

import yaml

from midogpp_thesis.cvae.preservation.tuned_classifier import (
    parse_midogpp_tuned_classifier_preservation_config,
)


def test_canonical_tuned_preservation_config_preserves_locked_protocol() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / (
        "experiments/midogpp/stages/20_cvae_preservation/configs/"
        "tuned_classifier_preservation_v1.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["experiment"]["artifact_root"] = (
        "artifacts/midogpp/20_cvae_preservation/"
        "virchow2_cvae_midogpp_tuned_classifier_preservation_v1/seed42"
    )
    payload["inputs"]["manifest_path"] = (
        "datasets/midogpp/contract/annotation_patch_v1/manifest.csv"
    )
    payload["inputs"]["feature_cache_path"] = (
        "datasets/midogpp/derived/features/virchow2/annotation_patch_xyxy/"
        "seed42/embeddings/train.pt"
    )
    payload["inputs"]["real_feature_reference_artifact_root"] = (
        "artifacts/midogpp/10_real_feature_reference/"
        "real_feature_threshold_both_annotation_patch_xyxy_virchow2_seed42"
    )

    cfg = parse_midogpp_tuned_classifier_preservation_config(payload, base_dir=repo_root)

    assert cfg.name == "virchow2_cvae_midogpp_tuned_classifier_preservation_v1"
    assert cfg.variant.variant_id == "pca128_beta001"
    assert cfg.variant.pca_dim == 128
    assert cfg.experiment_seed == 42
    assert cfg.heldout_centers is None
    assert cfg.expected_reference_manifest_hash == (
        "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
    )
    assert cfg.expected_reference_feature_cache_hash == (
        "f6608e513fb2d06671e3ec117b093a85d58530b77b1fae44a3be1680d9feabd2"
    )
