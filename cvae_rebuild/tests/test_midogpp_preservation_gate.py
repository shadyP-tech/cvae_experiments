from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.midogpp.midogpp_preservation_gate import (
    BALANCED_CONTROL,
    CVAE_DECODED_ROW_SHUFFLE,
    DECODE_MU,
    EXPERIMENT_NAME,
    PCA256_DIAGNOSTIC_VARIANT,
    PRIMARY_VARIANT,
    REAL_FEATURE_ROW_SHUFFLE,
    REAL_LABEL_PERMUTATION,
    REAL_PCA128_REFERENCE,
    WITHIN_TUMOR_CONTROL,
    MidogPPPreservationGateConfig,
    VariantConfig,
    _decision_labels,
    _gate_above_chance,
    parse_midogpp_preservation_gate_config,
)
from experiments.midogpp.midogpp_condition_audit import TRUE_TRAIN_TRUE_ENCODE_PERMUTED_DECODE


def test_config_locks_pca128_primary_and_pca256_diagnostic() -> None:
    cfg = parse_midogpp_preservation_gate_config(_config_payload(), base_dir=Path("."))
    assert cfg.primary_variant == PRIMARY_VARIANT
    assert {variant.variant_id for variant in cfg.variants} == {
        "pca64_beta001",
        PRIMARY_VARIANT,
        PCA256_DIAGNOSTIC_VARIANT,
    }

    bad = _config_payload()
    bad["run"] = {"primary_variant": "pca256_beta001"}
    with pytest.raises(ValueError, match="Primary gate variant"):
        parse_midogpp_preservation_gate_config(bad, base_dir=Path("."))


def test_gate_above_chance_uses_strict_bacc_threshold() -> None:
    cfg = _cfg()
    assert not _gate_above_chance(cfg, _summary(PRIMARY_VARIANT, DECODE_MU, bacc=0.60, ci_low=0.55))
    assert _gate_above_chance(cfg, _summary(PRIMARY_VARIANT, DECODE_MU, bacc=0.601, ci_low=0.55))


def test_decision_pass_requires_within_tumor_support_for_gmm_feasibility() -> None:
    cfg = _cfg()
    metrics = [
        _summary(PRIMARY_VARIANT, REAL_PCA128_REFERENCE, bacc=0.66, ci_low=0.55),
        _summary(PRIMARY_VARIANT, DECODE_MU, bacc=0.65, ci_low=0.54, ratio=0.94),
        _summary(PCA256_DIAGNOSTIC_VARIANT, DECODE_MU, bacc=0.69, ci_low=0.56),
        _summary(PRIMARY_VARIANT, DECODE_MU, control=WITHIN_TUMOR_CONTROL, domain="tumor-a", bacc=0.64, ci_low=0.53),
        _summary(PRIMARY_VARIANT, DECODE_MU, control=WITHIN_TUMOR_CONTROL, domain="tumor-b", bacc=0.59, ci_low=0.52),
    ]
    labels = _decision_labels(
        cfg,
        metrics,
        negatives=_clean_negatives(),
        condition_rows=[],
        leakage={"status": "PASS"},
    )
    assert "PCA128_CVAE_DECODE_PRESERVATION_PASS" in labels
    assert "GMM_FEASIBILITY_ALLOWED_NEXT" not in labels
    assert "DO_NOT_RUN_GMM_YET" in labels

    metrics.append(
        _summary(PRIMARY_VARIANT, DECODE_MU, control=WITHIN_TUMOR_CONTROL, domain="tumor-c", bacc=0.64, ci_low=0.53)
    )
    labels = _decision_labels(
        cfg,
        metrics,
        negatives=_clean_negatives(),
        condition_rows=[],
        leakage={"status": "PASS"},
    )
    assert "GMM_FEASIBILITY_ALLOWED_NEXT" in labels
    assert "DO_NOT_RUN_GMM_YET" not in labels


def test_condition_permutation_above_chance_is_warning_not_leakage() -> None:
    cfg = _cfg()
    metrics = [
        _summary(PRIMARY_VARIANT, REAL_PCA128_REFERENCE, bacc=0.66, ci_low=0.55),
        _summary(PRIMARY_VARIANT, DECODE_MU, bacc=0.65, ci_low=0.54, ratio=0.94),
        _summary(PRIMARY_VARIANT, DECODE_MU, control=WITHIN_TUMOR_CONTROL, domain="tumor-a", bacc=0.64, ci_low=0.53),
        _summary(PRIMARY_VARIANT, DECODE_MU, control=WITHIN_TUMOR_CONTROL, domain="tumor-b", bacc=0.65, ci_low=0.54),
    ]
    condition_rows = [
        _summary(
            PRIMARY_VARIANT,
            TRUE_TRAIN_TRUE_ENCODE_PERMUTED_DECODE,
            bacc=0.64,
            ci_low=0.53,
            condition_row_id=TRUE_TRAIN_TRUE_ENCODE_PERMUTED_DECODE,
        )
    ]
    labels = _decision_labels(
        cfg,
        metrics,
        negatives=_clean_negatives(),
        condition_rows=condition_rows,
        leakage={"status": "PASS"},
    )
    assert "LATENT_CLASS_SIGNAL_DOMINATES_CONDITION_WARNING" in labels
    assert "LEAKAGE_OR_ALIGNMENT_FAILURE_SUSPECT" not in labels
    assert "GMM_FEASIBILITY_ALLOWED_NEXT" in labels


def test_negative_control_above_chance_blocks_gate() -> None:
    cfg = _cfg()
    bad_negatives = _clean_negatives()
    bad_negatives[0]["bacc"] = 0.62
    bad_negatives[0]["ci_low"] = 0.51
    labels = _decision_labels(
        cfg,
        metrics=[
            _summary(PRIMARY_VARIANT, REAL_PCA128_REFERENCE, bacc=0.66, ci_low=0.55),
            _summary(PRIMARY_VARIANT, DECODE_MU, bacc=0.65, ci_low=0.54, ratio=0.94),
        ],
        negatives=bad_negatives,
        condition_rows=[],
        leakage={"status": "PASS"},
    )
    assert labels == ["LEAKAGE_OR_ALIGNMENT_FAILURE_SUSPECT", "DO_NOT_RUN_GMM_YET"]


def test_cli_smoke_writes_required_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    pytest.importorskip("torch")

    import numpy as np

    root = tmp_path
    manifest = root / "manifest.csv"
    cache = root / "train.npz"
    split_manifest = root / "split_manifest.csv"
    config = root / "config.yaml"
    artifact_root = root / "artifacts"
    _write_midog_manifest(manifest)
    rng = np.random.default_rng(123)
    labels = np.array([idx % 2 for idx in range(24)])
    embeddings = rng.normal(size=(24, 8)).astype("float32")
    embeddings[:, 0] += labels * 1.5
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
    config.write_text(
        f"""
experiment:
  name: {EXPERIMENT_NAME}
  artifact_root: {artifact_root}
inputs:
  manifest_path: {manifest}
  feature_cache_path: {cache}
  signal_split_manifest_path: {split_manifest}
  signal_decision_report_path:
  allow_npz_test_cache: true
positive_label: 1
run:
  controls: [{BALANCED_CONTROL}]
  primary_variant: pca128_beta001
  split_seeds: [1]
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
  seed: 7
decision_thresholds:
  gate_min_bacc: 0.60
  ci_low_threshold: 0.50
  preservation_pass_ratio: 0.80
  preservation_strong_ratio: 0.90
  within_tumor_min_above_fraction: 0.60
  pca256_stronger_delta: 0.03
variants:
  - variant_id: pca64_beta001
    pca_dim: 64
    latent_dim: 16
    hidden_dim: 512
    num_hidden_layers: 2
    train_epochs: 1
    batch_size: 8
    learning_rate: 0.001
    weight_decay: 0.0001
    beta_final: 0.001
    kl_warmup_epochs: 1
  - variant_id: pca128_beta001
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
  - variant_id: pca256_beta001
    pca_dim: 256
    latent_dim: 64
    hidden_dim: 512
    num_hidden_layers: 2
    train_epochs: 1
    batch_size: 8
    learning_rate: 0.001
    weight_decay: 0.0001
    beta_final: 0.001
    kl_warmup_epochs: 1
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli",
            "diagnose-midogpp-preservation-gate-pca128",
            "--config",
            str(config),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for rel in (
        "tables/preservation_gate_metrics.csv",
        "tables/pca_capacity_context.csv",
        "tables/condition_warning_matrix.csv",
        "tables/reconstruction_diagnostics.csv",
        "tables/training_diagnostics.csv",
        "tables/negative_control_metrics.csv",
        "tables/identity_overlap_audit.csv",
        "tables/predictions.csv",
        "manifests/protocol_manifest.json",
        "manifests/model_manifest.csv",
        "reports/leakage_report.json",
        "reports/decision_report.md",
    ):
        assert (artifact_root / rel).exists()
    report = (artifact_root / "reports" / "decision_report.md").read_text(encoding="utf-8")
    assert "pca128 CVAE Preservation Gate" in report
    assert "not GMM composition" in report


def _cfg() -> MidogPPPreservationGateConfig:
    return MidogPPPreservationGateConfig(
        name=EXPERIMENT_NAME,
        artifact_root=Path("unused"),
        manifest_path=Path("unused"),
        feature_cache_path=Path("unused"),
        signal_split_manifest_path=Path("unused"),
        signal_decision_report_path=None,
        positive_label=1,
        controls=(BALANCED_CONTROL, WITHIN_TUMOR_CONTROL),
        variants=(
            VariantConfig("pca64_beta001", 64, 16),
            VariantConfig(PRIMARY_VARIANT, 128, 32),
            VariantConfig(PCA256_DIAGNOSTIC_VARIANT, 256, 64),
        ),
        primary_variant=PRIMARY_VARIANT,
        split_seeds=None,
        bootstrap_reps=10,
        bootstrap_seed=1,
        allow_npz_test_cache=True,
        min_fit=4,
        min_eval=4,
        min_fit_pos=2,
        min_fit_neg=2,
        min_eval_pos=2,
        min_eval_neg=2,
        min_fit_cases=2,
        min_eval_cases=2,
        gate_min_bacc=0.60,
        ci_low_threshold=0.50,
        preservation_pass_ratio=0.80,
        preservation_strong_ratio=0.90,
        within_tumor_min_above_fraction=0.60,
        pca256_stronger_delta=0.03,
    )


def _summary(
    variant: str,
    role: str,
    *,
    bacc: float,
    ci_low: float,
    control: str = BALANCED_CONTROL,
    domain: str = "",
    ratio: float | str = "",
    condition_row_id: str = "",
) -> dict[str, object]:
    return {
        "aggregation_level": "summary",
        "control_name": control,
        "domain_name": domain,
        "variant_id": variant,
        "representation_role": role,
        "condition_row_id": condition_row_id,
        "status": "valid",
        "bacc": bacc,
        "ci_low": ci_low,
        "ci_high": max(ci_low, bacc + 0.05),
        "recall_pos": 0.5,
        "preservation_ratio_vs_real_frame": ratio,
    }


def _clean_negatives() -> list[dict[str, object]]:
    return [
        _summary(PRIMARY_VARIANT, REAL_LABEL_PERMUTATION, bacc=0.50, ci_low=0.45),
        _summary(PRIMARY_VARIANT, REAL_FEATURE_ROW_SHUFFLE, bacc=0.51, ci_low=0.44),
        _summary(PRIMARY_VARIANT, CVAE_DECODED_ROW_SHUFFLE, bacc=0.52, ci_low=0.46),
    ]


def _config_payload() -> dict[str, object]:
    return {
        "experiment": {"name": EXPERIMENT_NAME},
        "inputs": {
            "manifest_path": "manifest.csv",
            "feature_cache_path": "train.pt",
            "signal_split_manifest_path": "split_manifest.csv",
            "allow_npz_test_cache": True,
        },
        "run": {"primary_variant": PRIMARY_VARIANT},
        "variants": [
            {"variant_id": "pca64_beta001", "pca_dim": 64, "latent_dim": 16},
            {"variant_id": PRIMARY_VARIANT, "pca_dim": 128, "latent_dim": 32},
            {"variant_id": PCA256_DIAGNOSTIC_VARIANT, "pca_dim": 256, "latent_dim": 64},
        ],
    }


def _split_row(control: str, seed: int, subset: str, idx: int, sample_id: str) -> dict[str, object]:
    return {
        "control_name": control,
        "domain_name": "",
        "split_seed": seed,
        "subset": subset,
        "row_index": idx,
        "feature_row_index": idx,
        "sample_id": sample_id,
        "case_id": f"case{idx}",
        "image_path": f"img{idx}.png",
        "label": idx % 2,
        "tumor_type": "tumor-a" if idx < 12 else "tumor-b",
        "scanner_model": "scanner",
        "lab_or_origin": "lab",
        "species": "species",
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
                "tumor_type": "tumor-a" if idx < 12 else "tumor-b",
                "scanner_model": "scanner",
                "lab_or_origin": "lab",
                "species": "species",
            }
        )
    _write_csv(path, rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
