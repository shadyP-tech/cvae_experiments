from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from midogpp_thesis.cvae.preservation.tuned_classifier import (
    EXPERIMENT_NAME,
    load_midogpp_tuned_classifier_preservation_config,
)
from midogpp_thesis.cvae.preservation.tuned_reference import (
    load_tuned_classifier_reference,
)
from midogpp_thesis.real_features.classifier_reference.artifacts import stable_hash


def test_reference_import_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing tuned real-feature reference outputs"):
        load_tuned_classifier_reference(tmp_path / "missing")


def test_reference_import_rejects_unsafe_flags(tmp_path: Path) -> None:
    root = _write_reference_artifact(tmp_path / "reference", centers=("0",), selection_used_target_labels=True)
    with pytest.raises(ValueError, match="selection_used_target_labels"):
        load_tuned_classifier_reference(root)


def test_config_locks_primary_variant_and_reference_root(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
experiment:
  name: {EXPERIMENT_NAME}
inputs:
  manifest_path: manifest.csv
  feature_cache_path: train.npz
  real_feature_reference_artifact_root: reference
  allow_npz_test_cache: true
variant:
  variant_id: pca128_beta001
  pca_dim: 128
  latent_dim: 32
""",
        encoding="utf-8",
    )
    cfg = load_midogpp_tuned_classifier_preservation_config(config)
    assert cfg.variant.variant_id == "pca128_beta001"
    assert cfg.real_feature_reference_artifact_root == config.resolve().parents[2] / "reference"

    bad = config.read_text(encoding="utf-8").replace("pca128_beta001", "pca64_beta001")
    config.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="variant_id must remain"):
        load_midogpp_tuned_classifier_preservation_config(config)


def test_cli_smoke_writes_tuned_preservation_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    pytest.importorskip("torch")

    import numpy as np

    manifest = tmp_path / "manifest.csv"
    cache = tmp_path / "train.npz"
    reference = _write_reference_artifact(tmp_path / "reference", centers=("0", "1"))
    artifact_root = tmp_path / "artifacts"
    config = tmp_path / "config.yaml"
    _write_manifest(manifest)
    labels = np.array([idx % 2 for idx in range(32)])
    rng = np.random.default_rng(17)
    embeddings = rng.normal(size=(32, 8)).astype("float32")
    embeddings[:, 0] += labels * 2.0
    metadata = [{"sample_id": f"s{idx}", "label": int(labels[idx])} for idx in range(32)]
    np.savez(
        cache,
        embeddings=embeddings,
        metadata_json=json.dumps(metadata),
        feature_extractor_json=json.dumps({"backbone_type": "virchow2"}),
    )
    config.write_text(
        f"""
experiment:
  name: {EXPERIMENT_NAME}
  artifact_root: {artifact_root}
inputs:
  manifest_path: {manifest}
  feature_cache_path: {cache}
  real_feature_reference_artifact_root: {reference}
  allow_npz_test_cache: true
positive_label: 1
run:
  experiment_seed: 42
  heldout_centers: ["0", "1"]
variant:
  variant_id: pca128_beta001
  pca_dim: 128
  latent_dim: 32
  hidden_dim: 512
  num_hidden_layers: 2
  train_epochs: 1
  batch_size: 8
  learning_rate: 0.001
  weight_decay: 0.0001
  beta_final: 0.001
  kl_warmup_epochs: 1
validity_thresholds:
  min_fit: 4
  min_eval: 4
  min_fit_pos: 2
  min_fit_neg: 2
  min_eval_pos: 2
  min_eval_neg: 2
  min_fit_cases: 2
  min_eval_cases: 2
bootstrap:
  reps: 5
  seed: 9
reference_validation:
  real_reference_min_bacc_for_ratio: 0.55
  ci_low_threshold: 0.50
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "midogpp_thesis",
            "cvae-preservation",
            "tuned-classifier",
            "--config",
            str(config),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for rel in (
        "tables/tuned_preservation_metrics.csv",
        "tables/imported_real_tuned_reference.csv",
        "tables/reconstruction_diagnostics.csv",
        "tables/training_diagnostics.csv",
        "tables/identity_overlap_audit.csv",
        "tables/predictions.csv",
        "manifests/protocol_manifest.json",
        "reports/leakage_report.json",
        "reports/decision_report.md",
    ):
        assert (artifact_root / rel).exists()
    metrics = list(csv.DictReader((artifact_root / "tables/tuned_preservation_metrics.csv").open()))
    assert {row["representation_role"] for row in metrics if row["aggregation_level"] == "seed"} == {
        "real_pca128_reference",
        "decode_mu_fit_to_real_eval",
        "posterior_sample_fit_to_real_eval",
        "prior_sample_fit_to_real_eval",
    }
    assert len({row["selected_classifier_config_hash"] for row in metrics}) == 2


def _write_manifest(path: Path) -> None:
    rows = []
    for idx in range(32):
        center = "0" if idx < 16 else "1"
        rows.append(
            {
                "sample_id": f"s{idx}",
                "case_id": f"case{idx // 2}",
                "label": idx % 2,
                "split": "train",
                "center": center,
                "image_path": f"img{idx}.png",
            }
        )
    _write_csv(path, rows)


def _write_reference_artifact(
    root: Path,
    *,
    centers: tuple[str, ...],
    selection_used_target_labels: bool = False,
) -> Path:
    spec = {
        "family": "sklearn_logistic_regression",
        "C": 0.01,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 2000,
        "class_weight": None,
        "random_state": 23,
        "l1_ratio": None,
        "threshold_policy": "predict",
        "scaler_fit": "synthetic_train_only",
    }
    protocol = {
        "schema_version": "midogpp_real_feature_source_only_classifier_reference_v1",
        "claim_scope": "real_feature_transfer_only",
        "heldout_centers": list(centers),
        "manifest_hash": "manifest-hash",
        "feature_cache_hash": "feature-hash",
        "protocol_hash": "protocol-hash",
        "selection_used_target_labels": selection_used_target_labels,
        "fit_used_target_center": False,
        "generated_embeddings_used": False,
        "cvae_checkpoint_used": False,
        "source_summary_manifest_used": False,
        "is_router": False,
        "probabilities_calibrated": False,
    }
    leakage = {
        **protocol,
        "status": "PASS",
        "target_labels_used_for_scoring_only": True,
    }
    rows = []
    for center in centers:
        row_spec = dict(spec)
        if center == "1":
            row_spec["class_weight"] = "balanced"
        rows.append(
            {
                "schema_version": "midogpp_real_feature_classifier_results_v1",
                "method": "source_inner_tuned",
                "experiment_seed": 42,
                "classifier_seed": 23,
                "heldout_center": center,
                "train_centers": "[]",
                "n_train": 16,
                "n_eval": 16,
                "classifier_grid_hash": "grid",
                "selected_classifier_config_hash": stable_hash(row_spec),
                "selected_classifier_spec": json.dumps(row_spec, sort_keys=True),
                "selection_source": "source_inner_lodo",
                "source_inner_mean_bacc": 0.7,
                "source_inner_min_bacc": 0.6,
                "source_inner_std_bacc": 0.01,
                "source_inner_n_centers": 1,
                "heldout_bacc": 0.75,
                "heldout_macro_f1": 0.74,
                "converged": "true",
                "n_iter": "[10]",
                "status": "ok",
                "error_message": "",
                "feature_cache_hash": "feature-hash",
                "manifest_hash": "manifest-hash",
                "target_eval_labels_used_for_scoring_only": "true",
                "selection_used_target_labels": str(selection_used_target_labels).lower(),
                "fit_used_target_center": "false",
                "generated_embeddings_used": "false",
                "cvae_checkpoint_used": "false",
                "source_summary_manifest_used": "false",
                "is_router": "false",
                "claim_scope": "real_feature_transfer_only",
                "probabilities_calibrated": "false",
            }
        )
    (root / "tables").mkdir(parents=True)
    (root / "manifests").mkdir(parents=True)
    (root / "reports").mkdir(parents=True)
    _write_csv(root / "tables" / "classifier_tuned_source_results.csv", rows)
    (root / "manifests" / "protocol_manifest.json").write_text(json.dumps(protocol), encoding="utf-8")
    (root / "reports" / "leakage_provenance_report.json").write_text(json.dumps(leakage), encoding="utf-8")
    return root


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
