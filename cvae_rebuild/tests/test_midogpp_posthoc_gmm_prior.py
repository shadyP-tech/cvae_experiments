from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.protocol import ProtocolError
from experiments.midogpp.midogpp_pca128_posthoc_gmm_prior import (
    BALANCED_CONTROL,
    CLASS_PRIOR_SHUFFLED_CONTROL,
    EXPERIMENT_NAME,
    GATE_REQUIRED_LABEL,
    LABEL_PERMUTED_GMM_CONTROL,
    PRIMARY_METHOD,
    RANDOM_LATENT_CONTROL,
    load_midogpp_posthoc_gmm_prior_config,
    parse_midogpp_posthoc_gmm_prior_config,
    run_midogpp_posthoc_gmm_prior,
)


def test_config_locks_primary_method_and_posterior_mu_semantics() -> None:
    cfg = parse_midogpp_posthoc_gmm_prior_config(_config_payload(), base_dir=Path("."))

    assert cfg.name == EXPERIMENT_NAME
    assert cfg.primary_method == PRIMARY_METHOD
    assert cfg.primary_variant.variant_id == "pca128_beta001"
    assert cfg.primary_variant.pca_dim == 128
    assert cfg.primary_variant.latent_dim == 32
    assert cfg.no_rejection is True

    bad = _config_payload()
    bad["run"]["primary_method"] = "choose_best_after_eval"
    with pytest.raises(ProtocolError, match="primary_method"):
        parse_midogpp_posthoc_gmm_prior_config(bad, base_dir=Path("."))


def test_missing_gate_feasibility_label_blocks_audit(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    manifest, cache, split_manifest, gate_root, config = _write_smoke_inputs(tmp_path, gate_labels=["PCA128_CVAE_STRONG_PRESERVATION"])
    del manifest, cache, split_manifest

    cfg = load_midogpp_posthoc_gmm_prior_config(config)
    assert cfg.preservation_gate_artifact_root == gate_root
    with pytest.raises(ProtocolError, match=GATE_REQUIRED_LABEL):
        run_midogpp_posthoc_gmm_prior(cfg)


def test_gate_provenance_mismatch_blocks_audit(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    manifest, cache, split_manifest, gate_root, config = _write_smoke_inputs(tmp_path)
    del manifest, cache, split_manifest
    manifest_json = gate_root / "manifests" / "protocol_manifest.json"
    payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    payload["feature_cache_path"] = str(tmp_path / "wrong-cache.pt")
    manifest_json.write_text(json.dumps(payload), encoding="utf-8")

    cfg = load_midogpp_posthoc_gmm_prior_config(config)
    with pytest.raises(ProtocolError, match="provenance_mismatch"):
        run_midogpp_posthoc_gmm_prior(cfg)


def test_cli_smoke_writes_required_artifacts_and_controls(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    pytest.importorskip("torch")

    _, _, _, _, config = _write_smoke_inputs(tmp_path)
    artifact_root = tmp_path / "override_artifacts"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "src/cli.py",
            "diagnose-midogpp-pca128-posthoc-gmm-prior",
            "--config",
            str(config),
            "--artifact-root",
            str(artifact_root),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    assert str(artifact_root) in result.stdout

    expected = [
        artifact_root / "tables" / "posthoc_gmm_prior_metrics.csv",
        artifact_root / "tables" / "gmm_parameter_diagnostics.csv",
        artifact_root / "tables" / "negative_control_metrics.csv",
        artifact_root / "tables" / "predictions.csv",
        artifact_root / "manifests" / "protocol_manifest.json",
        artifact_root / "manifests" / "model_manifest.csv",
        artifact_root / "reports" / "leakage_report.json",
        artifact_root / "reports" / "decision_report.md",
    ]
    for path in expected:
        assert path.exists(), path

    metrics = _read_csv(artifact_root / "tables" / "posthoc_gmm_prior_metrics.csv")
    assert any(row["method_role"] == PRIMARY_METHOD for row in metrics)
    assert all(row["method_role"] != PRIMARY_METHOD or row["adoption_eligible"] in {"true", "false"} for row in metrics)

    controls = _read_csv(artifact_root / "tables" / "negative_control_metrics.csv")
    emitted = {row["method_role"] for row in controls}
    assert LABEL_PERMUTED_GMM_CONTROL in emitted
    assert CLASS_PRIOR_SHUFFLED_CONTROL in emitted
    assert RANDOM_LATENT_CONTROL in emitted

    protocol = json.loads((artifact_root / "manifests" / "protocol_manifest.json").read_text(encoding="utf-8"))
    assert protocol["posterior_source"] == "encoder_mu_fit_rows_only"
    assert protocol["gmm_fit_scope"] == "fit_rows_only"
    assert protocol["held_out_eval_role"] == "final_scoring_only"
    assert "routing" in protocol["claim_boundary"]["forbidden"].lower()


def _write_smoke_inputs(
    tmp_path: Path,
    *,
    gate_labels: list[str] | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    import numpy as np

    manifest = tmp_path / "manifest.csv"
    cache = tmp_path / "train.npz"
    split_manifest = tmp_path / "split_manifest.csv"
    gate_root = tmp_path / "gate"
    config = tmp_path / "config.yaml"
    artifact_root = tmp_path / "artifacts"
    _write_midog_manifest(manifest)
    rng = np.random.default_rng(123)
    labels = np.array([idx % 2 for idx in range(24)])
    embeddings = rng.normal(size=(24, 8)).astype("float32")
    embeddings[:, 0] += labels * 1.25
    metadata = [{"sample_id": f"s{idx}", "label": int(labels[idx])} for idx in range(24)]
    np.savez(
        cache,
        embeddings=embeddings,
        metadata_json=json.dumps(metadata),
        feature_extractor_json=json.dumps({"backbone_type": "virchow2"}),
    )
    _write_csv(
        split_manifest,
        [
            *[_split_row(BALANCED_CONTROL, 1, "fit", idx, f"s{idx}") for idx in range(16)],
            *[_split_row(BALANCED_CONTROL, 1, "eval", idx, f"s{idx}") for idx in range(16, 24)],
        ],
    )
    _write_fake_gate(
        gate_root,
        manifest=manifest,
        cache=cache,
        split_manifest=split_manifest,
        labels=gate_labels or [GATE_REQUIRED_LABEL, "LATENT_CLASS_SIGNAL_DOMINATES_CONDITION_WARNING"],
    )
    config.write_text(
        f"""
experiment:
  name: {EXPERIMENT_NAME}
  artifact_root: {artifact_root}
inputs:
  manifest_path: {manifest}
  feature_cache_path: {cache}
  signal_split_manifest_path: {split_manifest}
  preservation_gate_artifact_root: {gate_root}
  allow_npz_test_cache: true
positive_label: 1
run:
  controls: [{BALANCED_CONTROL}]
  primary_method: {PRIMARY_METHOD}
  gate_required_label: {GATE_REQUIRED_LABEL}
  allow_condition_warning: true
  class_prior_policy: balanced
  synthetic_budget: match_fit
  split_seeds: [1]
primary_variant:
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
gmm_prior:
  components: 2
  covariance_type: diag
  reg_covar: 0.0001
  n_init: 1
  max_iter: 50
  min_per_class_n: 4
  min_samples_per_component: 2
  fallback_to_single_gaussian: true
  no_rejection: true
downstream:
  generation_seeds: [13]
  classifier_seeds: [17]
bootstrap:
  reps: 5
  seed: 7
decision_thresholds:
  real_gate_min_bacc: 0.50
  ci_low_threshold: 0.00
""",
        encoding="utf-8",
    )
    return manifest, cache, split_manifest, gate_root, config


def _write_fake_gate(
    gate_root: Path,
    *,
    manifest: Path,
    cache: Path,
    split_manifest: Path,
    labels: list[str],
) -> None:
    (gate_root / "reports").mkdir(parents=True, exist_ok=True)
    (gate_root / "manifests").mkdir(parents=True, exist_ok=True)
    (gate_root / "reports" / "decision_report.md").write_text(
        "# Gate\n\n" f"- Decision labels: `{', '.join(labels)}`\n",
        encoding="utf-8",
    )
    (gate_root / "reports" / "leakage_report.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    (gate_root / "manifests" / "protocol_manifest.json").write_text(
        json.dumps(
            {
                "manifest_path": str(manifest),
                "feature_cache_path": str(cache),
                "signal_split_manifest_path": str(split_manifest),
                "train_rows": 24,
                "split_count": 1,
            }
        ),
        encoding="utf-8",
    )


def _config_payload() -> dict[str, object]:
    return {
        "experiment": {"name": EXPERIMENT_NAME, "artifact_root": "artifacts"},
        "inputs": {
            "manifest_path": "manifest.csv",
            "feature_cache_path": "train.pt",
            "signal_split_manifest_path": "split.csv",
            "preservation_gate_artifact_root": "gate",
        },
        "run": {
            "controls": [BALANCED_CONTROL],
            "primary_method": PRIMARY_METHOD,
            "class_prior_policy": "balanced",
            "generation_seeds": [13],
            "classifier_seeds": [17],
        },
        "primary_variant": {
            "variant_id": "pca128_beta001",
            "pca_dim": 128,
            "latent_dim": 32,
            "hidden_dim": 512,
            "num_hidden_layers": 2,
            "beta_final": 0.001,
        },
        "gmm_prior": {
            "components": 2,
            "covariance_type": "diag",
            "reg_covar": 0.0001,
            "fallback_to_single_gaussian": True,
            "no_rejection": True,
        },
        "downstream": {"generation_seeds": [13], "classifier_seeds": [17]},
    }


def _write_midog_manifest(path: Path) -> None:
    rows = []
    for idx in range(24):
        rows.append(
            {
                "sample_id": f"s{idx}",
                "case_id": f"case{idx}",
                "label": idx % 2,
                "split": "train",
                "image_path": f"img{idx}.png",
                "tumor_type": "mast_cell_tumor" if idx % 3 else "other",
                "scanner_model": "scanner",
                "lab_or_origin": "lab",
                "species": "dog",
            }
        )
    _write_csv(path, rows)


def _split_row(control: str, seed: int, subset: str, idx: int, sample_id: str) -> dict[str, object]:
    return {
        "control_name": control,
        "domain_name": "",
        "split_seed": seed,
        "subset": subset,
        "feature_row_index": idx,
        "sample_id": sample_id,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
