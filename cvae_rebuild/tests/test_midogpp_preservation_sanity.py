from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cvae_rebuild.midogpp_preservation_sanity import (
    BALANCED_CONTROL,
    CVAE_CONDITION_LABEL_PERMUTATION,
    CVAE_DECODED_ROW_SHUFFLE,
    DECODE_MU,
    REAL_FEATURE_ROW_SHUFFLE,
    PRIMARY_VARIANT,
    RAW_VARIANT,
    REAL_FRAME_REFERENCE,
    REAL_LABEL_PERMUTATION,
    REAL_RAW_REFERENCE,
    VALID_STATUS,
    ManifestRow,
    SplitSpec,
    _chance_corrected_ratio,
    _decision_labels,
    _identity_audit_rows,
    _read_split_manifest,
    _real_baseline_suitability,
)


def test_split_manifest_parsing_preserves_indices_and_rejects_mismatch(tmp_path: Path) -> None:
    rows = tuple(
        ManifestRow(i, i, f"s{i}", f"c{i}", i % 2, "train", {"image_path": f"p{i}.png"})
        for i in range(4)
    )
    split_path = tmp_path / "split_manifest.csv"
    _write_csv(
        split_path,
        [
            _split_row(BALANCED_CONTROL, 7, "fit", 0, "s0"),
            _split_row(BALANCED_CONTROL, 7, "fit", 1, "s1"),
            _split_row(BALANCED_CONTROL, 7, "eval", 2, "s2"),
            _split_row(BALANCED_CONTROL, 7, "eval", 3, "s3"),
        ],
    )
    specs = _read_split_manifest(split_path, rows, controls=(BALANCED_CONTROL,))
    assert specs == (SplitSpec(BALANCED_CONTROL, "", 7, (0, 1), (2, 3)),)

    bad_path = tmp_path / "bad_split_manifest.csv"
    _write_csv(bad_path, [_split_row(BALANCED_CONTROL, 7, "fit", 0, "wrong")])
    with pytest.raises(ValueError, match="Split manifest row mismatch"):
        _read_split_manifest(bad_path, rows, controls=(BALANCED_CONTROL,))


def test_identity_audit_detects_case_and_feature_overlap() -> None:
    rows = (
        ManifestRow(0, 0, "s0", "case0", 0, "train", {"image_path": "a.png"}),
        ManifestRow(1, 1, "s1", "case0", 1, "train", {"image_path": "b.png"}),
    )
    spec = SplitSpec(BALANCED_CONTROL, "", 1, (0,), (0, 1))
    audit = _identity_audit_rows(rows, spec)
    failures = {row["identity_field"]: row for row in audit if row["status"] == "FAIL"}
    assert "feature_row_index" in failures
    assert "case_id" in failures


def test_real_baseline_suitability_blocks_prior_shortcut() -> None:
    cfg = _cfg()
    metrics = [
        _summary_metric(BALANCED_CONTROL, "", PRIMARY_VARIANT, REAL_FRAME_REFERENCE, bacc=0.66, ci_low=0.55),
        _summary_metric("within_tumor_case_disjoint_control", "tumor-a", PRIMARY_VARIANT, REAL_FRAME_REFERENCE, bacc=0.66, ci_low=0.55),
        _summary_metric("within_tumor_case_disjoint_control", "tumor-b", PRIMARY_VARIANT, REAL_FRAME_REFERENCE, bacc=0.67, ci_low=0.56),
    ]
    rows, suitable = _real_baseline_suitability(
        cfg,
        metrics,
        negatives=[
            _negative_summary(PRIMARY_VARIANT, REAL_LABEL_PERMUTATION),
            _negative_summary(PRIMARY_VARIANT, REAL_FEATURE_ROW_SHUFFLE),
        ],
        prior={"provided": True, "shortcut_suspect": True, "path": "decision_report.md"},
    )
    assert not suitable
    shortcut = [row for row in rows if row["criterion"] == "prior_signal_control_not_shortcut_suspect"][0]
    assert shortcut["passed"] == "false"


def test_decision_uses_primary_variant_not_best_diagnostic_variant() -> None:
    cfg = _cfg()
    metrics = [
        _summary_metric(BALANCED_CONTROL, "", RAW_VARIANT, REAL_RAW_REFERENCE, bacc=0.70, ci_low=0.58),
        _summary_metric(BALANCED_CONTROL, "", PRIMARY_VARIANT, REAL_FRAME_REFERENCE, bacc=0.68, ci_low=0.56),
        _summary_metric(BALANCED_CONTROL, "", PRIMARY_VARIANT, DECODE_MU, bacc=0.55, ci_low=0.49),
        _summary_metric(BALANCED_CONTROL, "", "pca128_beta001", DECODE_MU, bacc=0.67, ci_low=0.56),
    ]
    labels = _decision_labels(
        cfg,
        metrics,
        negatives=[
            _negative_summary(PRIMARY_VARIANT, REAL_LABEL_PERMUTATION),
            _negative_summary(PRIMARY_VARIANT, REAL_FEATURE_ROW_SHUFFLE),
            _negative_summary(PRIMARY_VARIANT, CVAE_CONDITION_LABEL_PERMUTATION),
            _negative_summary(PRIMARY_VARIANT, CVAE_DECODED_ROW_SHUFFLE),
        ],
        leakage={"status": "PASS"},
        real_suitable=True,
        prior={"shortcut_suspect": False},
    )
    assert "CVAE_PRESERVATION_SANITY_PASS" not in labels
    assert "CVAE_RECONSTRUCTION_BOTTLENECK" in labels


def test_chance_corrected_preservation_ratio() -> None:
    assert _chance_corrected_ratio(0.64, 0.70) == pytest.approx(0.70)


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
    embeddings = rng.normal(size=(24, 8)).astype("float32")
    labels = np.array([idx % 2 for idx in range(24)])
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
  name: virchow2_cvae_midogpp_preservation_sanity_v1
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
  primary_variant: pca64_beta001
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
  cvae_gate_min_bacc: 0.60
  ci_low_threshold: 0.50
  preservation_pass_ratio: 0.80
  preservation_strong_ratio: 0.90
  within_tumor_min_above_fraction: 0.60
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
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cvae_rebuild.cli",
            "diagnose-midogpp-preservation-sanity",
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
        "tables/preservation_metrics.csv",
        "tables/variant_gap_summary.csv",
        "tables/real_baseline_suitability.csv",
        "tables/negative_control_metrics.csv",
        "manifests/protocol_manifest.json",
        "manifests/model_manifest.csv",
        "reports/leakage_report.json",
        "reports/decision_report.md",
    ):
        assert (artifact_root / rel).exists()


def _cfg():
    from cvae_rebuild.midogpp_preservation_sanity import MidogPPPreservationSanityConfig, VariantConfig

    return MidogPPPreservationSanityConfig(
        name="virchow2_cvae_midogpp_preservation_sanity_v1",
        artifact_root=Path("unused"),
        manifest_path=Path("unused"),
        feature_cache_path=Path("unused"),
        signal_split_manifest_path=Path("unused"),
        signal_decision_report_path=None,
        positive_label=1,
        controls=(BALANCED_CONTROL,),
        variants=(VariantConfig(PRIMARY_VARIANT, 64, 16),),
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
        real_gate_min_bacc=0.60,
        cvae_gate_min_bacc=0.60,
        ci_low_threshold=0.50,
        preservation_pass_ratio=0.80,
        preservation_strong_ratio=0.90,
        within_tumor_min_above_fraction=0.60,
    )


def _summary_metric(control: str, domain: str, variant: str, role: str, *, bacc: float, ci_low: float) -> dict[str, object]:
    return {
        "aggregation_level": "summary",
        "control_name": control,
        "domain_name": domain,
        "variant_id": variant,
        "representation_role": role,
        "status": VALID_STATUS,
        "bacc": bacc,
        "ci_low": ci_low,
        "ci_high": max(ci_low, bacc + 0.05),
        "recall_pos": 0.5,
        "above_chance": "true" if bacc >= 0.60 and ci_low > 0.50 else "false",
    }


def _negative_summary(variant: str, role: str) -> dict[str, object]:
    row = _summary_metric(BALANCED_CONTROL, "", variant, role, bacc=0.50, ci_low=0.45)
    row["ci_high"] = 0.55
    row["near_chance"] = "true"
    row["above_chance"] = "false"
    return row


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
