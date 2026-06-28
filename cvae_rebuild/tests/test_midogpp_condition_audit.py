from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.midogpp.midogpp_condition_audit import (
    BALANCED_CONTROL,
    CONDITION_ROW_IDS,
    FULL_DIM_VARIANT,
    PERMUTED_LABELS,
    PERMUTED_TRAIN_PERMUTED_ENCODE_PERMUTED_DECODE,
    PERMUTED_TRAIN_TRUE_ENCODE_TRUE_DECODE,
    REAL_FULL_DIM_REFERENCE,
    REAL_PCA64_REFERENCE,
    TRUE_LABELS,
    TRUE_TRAIN_TRUE_ENCODE_PERMUTED_DECODE,
    TRUE_TRAIN_TRUE_ENCODE_TRUE_DECODE,
    WITHIN_TUMOR_CONTROL,
    MidogPPConditionAuditConfig,
    SplitSpec,
    VariantConfig,
    _condition_rows,
    _decision_labels,
    _decoder_label_conditioning_row,
    _without_preservation_pass,
    parse_midogpp_condition_audit_config,
)


def test_condition_rows_are_exact_and_independently_controlled() -> None:
    true_runtime = SimpleNamespace(name="true")
    permuted_runtime = SimpleNamespace(name="permuted")
    rows = _condition_rows(true_runtime, permuted_runtime, [0, 1, 0, 1], [1, 0, 1, 0])
    assert tuple(str(row["condition_row_id"]) for row in rows) == CONDITION_ROW_IDS
    by_id = {str(row["condition_row_id"]): row for row in rows}

    assert by_id[TRUE_TRAIN_TRUE_ENCODE_TRUE_DECODE]["train_condition_labels"] == TRUE_LABELS
    assert by_id[TRUE_TRAIN_TRUE_ENCODE_TRUE_DECODE]["encode_condition_labels"] == TRUE_LABELS
    assert by_id[TRUE_TRAIN_TRUE_ENCODE_TRUE_DECODE]["decode_condition_labels"] == TRUE_LABELS
    assert by_id[TRUE_TRAIN_TRUE_ENCODE_TRUE_DECODE]["runtime"] is true_runtime

    assert by_id[PERMUTED_TRAIN_TRUE_ENCODE_TRUE_DECODE]["train_condition_labels"] == PERMUTED_LABELS
    assert by_id[PERMUTED_TRAIN_TRUE_ENCODE_TRUE_DECODE]["encode_condition_labels"] == TRUE_LABELS
    assert by_id[PERMUTED_TRAIN_TRUE_ENCODE_TRUE_DECODE]["decode_condition_labels"] == TRUE_LABELS
    assert by_id[PERMUTED_TRAIN_TRUE_ENCODE_TRUE_DECODE]["runtime"] is permuted_runtime

    assert by_id[TRUE_TRAIN_TRUE_ENCODE_PERMUTED_DECODE]["train_condition_labels"] == TRUE_LABELS
    assert by_id[TRUE_TRAIN_TRUE_ENCODE_PERMUTED_DECODE]["encode_condition_labels"] == TRUE_LABELS
    assert by_id[TRUE_TRAIN_TRUE_ENCODE_PERMUTED_DECODE]["decode_condition_labels"] == PERMUTED_LABELS

    assert by_id[PERMUTED_TRAIN_PERMUTED_ENCODE_PERMUTED_DECODE]["train_condition_labels"] == PERMUTED_LABELS
    assert by_id[PERMUTED_TRAIN_PERMUTED_ENCODE_PERMUTED_DECODE]["encode_condition_labels"] == PERMUTED_LABELS
    assert by_id[PERMUTED_TRAIN_PERMUTED_ENCODE_PERMUTED_DECODE]["decode_condition_labels"] == PERMUTED_LABELS


def test_diagnostic_labels_use_locked_thresholds_and_never_emit_preservation_pass() -> None:
    cfg = _cfg()
    pca_rows = [
        _summary_metric(FULL_DIM_VARIANT, REAL_FULL_DIM_REFERENCE, bacc=0.72, ci_low=0.60),
        _summary_metric("pca64_beta001", REAL_PCA64_REFERENCE, bacc=0.58, ci_low=0.52),
        _summary_metric("pca128_beta001", "real_pca128_reference", bacc=0.66, ci_low=0.55),
        _summary_metric("pca256_beta001", "real_pca256_reference", bacc=0.68, ci_low=0.56),
    ]
    condition_rows = [
        _condition_summary("pca64_beta001", TRUE_TRAIN_TRUE_ENCODE_TRUE_DECODE, bacc=0.70, ci_low=0.58),
        _condition_summary("pca64_beta001", TRUE_TRAIN_TRUE_ENCODE_PERMUTED_DECODE, bacc=0.69, ci_low=0.57),
        _condition_summary("pca64_beta001", PERMUTED_TRAIN_TRUE_ENCODE_TRUE_DECODE, bacc=0.62, ci_low=0.53),
    ]
    decoder_rows = [
        {
            "aggregation_level": "summary",
            "control_name": BALANCED_CONTROL,
            "domain_name": "",
            "variant_id": "pca64_beta001",
            "weak_conditioning": "true",
        }
    ]
    labels = _decision_labels(cfg, pca_rows, condition_rows, decoder_rows, {"status": "PASS"})
    assert "CONDITION_AUDIT_PROTOCOL_CLEAN" in labels
    assert "PCA64_CAPACITY_BOTTLENECK" in labels
    assert "PCA128_CAPACITY_RECOVERY" in labels
    assert "PCA256_CAPACITY_RECOVERY" in labels
    assert "CONDITION_PERMUTATION_CONTROL_REPRODUCED" in labels
    assert "LATENT_CLASS_SIGNAL_DOMINATES_CONDITION" in labels
    assert "DECODER_LABEL_CONDITIONING_WEAK" in labels
    assert all(
        "PRESERVATION_PASS" not in label
        and "PRESERVATION_SANITY_PASS" not in label
        and "CVAE_PRESERVATION" not in label
        for label in labels
    )


def test_without_preservation_pass_filters_forbidden_labels() -> None:
    labels = _without_preservation_pass(["CVAE_PRESERVATION_SANITY_PASS", "PCA128_CAPACITY_RECOVERY"])
    assert labels == ["PCA128_CAPACITY_RECOVERY"]


def test_same_latent_label_swap_uses_fit_rows_only() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("torch")

    import numpy as np
    import torch

    class FakeModel:
        latent_dim = 2

        def __init__(self) -> None:
            self.encoded_labels: list[int] | None = None

        def encode(self, x, y):
            self.encoded_labels = [int(value) for value in y.detach().cpu().numpy().tolist()]
            return x, torch.zeros_like(x)

        def decode(self, z, y):
            return z + y.float().unsqueeze(1) * 0.1

    model = FakeModel()
    runtime = SimpleNamespace(
        variant=VariantConfig("pca64_beta001", 64, 16),
        fit_x=np.asarray([[0.0, 0.0], [1.0, 1.0], [0.2, 0.1], [1.2, 1.1]], dtype="float32"),
        model=model,
    )
    spec = SplitSpec(BALANCED_CONTROL, "", 7, (0, 1, 2, 3), (4, 5))
    row = _decoder_label_conditioning_row(_cfg(), spec, runtime, [0, 1, 0, 1])
    assert model.encoded_labels == [0, 1, 0, 1]
    assert row["status"] == "valid"
    assert float(row["label_swap_l2"]) > 0.0


def test_config_requires_condition_audit_name_and_pca256_variant() -> None:
    base = _config_payload()
    cfg = parse_midogpp_condition_audit_config(base, base_dir=Path("."))
    assert cfg.name == "virchow2_cvae_midogpp_preservation_condition_audit_v1"
    assert {variant.variant_id for variant in cfg.variants} == {
        "pca64_beta001",
        "pca128_beta001",
        "pca256_beta001",
    }

    bad = _config_payload()
    bad["variants"] = bad["variants"][:2]
    with pytest.raises(ValueError, match="pca256_beta001"):
        parse_midogpp_condition_audit_config(bad, base_dir=Path("."))


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
  name: virchow2_cvae_midogpp_preservation_condition_audit_v1
  artifact_root: {artifact_root}
inputs:
  manifest_path: {manifest}
  feature_cache_path: {cache}
  signal_split_manifest_path: {split_manifest}
  allow_npz_test_cache: true
positive_label: 1
run:
  controls: [{BALANCED_CONTROL}]
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
  real_gate_min_bacc: 0.60
  ci_low_threshold: 0.50
  close_bacc_delta: 0.03
  close_ratio_threshold: 0.90
  weak_conditioning_ratio_threshold: 0.25
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
            "diagnose-midogpp-preservation-condition-audit",
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
        "tables/pca_capacity_audit.csv",
        "tables/condition_permutation_matrix.csv",
        "tables/decoder_label_conditioning_audit.csv",
        "tables/downstream_condition_metrics.csv",
        "tables/reconstruction_diagnostics.csv",
        "tables/training_diagnostics.csv",
        "tables/identity_overlap_audit.csv",
        "tables/predictions.csv",
        "manifests/protocol_manifest.json",
        "manifests/model_manifest.csv",
        "reports/leakage_report.json",
        "reports/decision_report.md",
    ):
        assert (artifact_root / rel).exists()
    decision_text = (artifact_root / "reports" / "decision_report.md").read_text(encoding="utf-8")
    assert "CVAE preservation PASS" in decision_text
    assert "Decision labels" in decision_text


def _cfg() -> MidogPPConditionAuditConfig:
    return MidogPPConditionAuditConfig(
        name="virchow2_cvae_midogpp_preservation_condition_audit_v1",
        artifact_root=Path("unused"),
        manifest_path=Path("unused"),
        feature_cache_path=Path("unused"),
        signal_split_manifest_path=Path("unused"),
        positive_label=1,
        controls=(BALANCED_CONTROL, WITHIN_TUMOR_CONTROL),
        variants=(
            VariantConfig("pca64_beta001", 64, 16),
            VariantConfig("pca128_beta001", 128, 32),
            VariantConfig("pca256_beta001", 256, 64),
        ),
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
        real_gate_min_bacc=0.60,
        ci_low_threshold=0.50,
        close_bacc_delta=0.03,
        close_ratio_threshold=0.90,
        weak_conditioning_ratio_threshold=0.25,
    )


def _summary_metric(variant: str, role: str, *, bacc: float, ci_low: float) -> dict[str, object]:
    return {
        "aggregation_level": "summary",
        "control_name": BALANCED_CONTROL,
        "domain_name": "",
        "variant_id": variant,
        "representation_role": role,
        "status": "valid",
        "bacc": bacc,
        "ci_low": ci_low,
        "ci_high": max(ci_low, bacc + 0.05),
        "recall_pos": 0.5,
        "above_chance": "true" if bacc >= 0.60 and ci_low > 0.50 else "false",
    }


def _condition_summary(variant: str, condition_row_id: str, *, bacc: float, ci_low: float) -> dict[str, object]:
    row = _summary_metric(variant, condition_row_id, bacc=bacc, ci_low=ci_low)
    row["condition_row_id"] = condition_row_id
    return row


def _config_payload() -> dict[str, object]:
    return {
        "experiment": {"name": "virchow2_cvae_midogpp_preservation_condition_audit_v1"},
        "inputs": {
            "manifest_path": "manifest.csv",
            "feature_cache_path": "train.pt",
            "signal_split_manifest_path": "split_manifest.csv",
            "allow_npz_test_cache": True,
        },
        "variants": [
            {"variant_id": "pca64_beta001", "pca_dim": 64, "latent_dim": 16},
            {"variant_id": "pca128_beta001", "pca_dim": 128, "latent_dim": 32},
            {"variant_id": "pca256_beta001", "pca_dim": 256, "latent_dim": 64},
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
