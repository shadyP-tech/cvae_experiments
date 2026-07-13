from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

from midogpp_thesis.real_features.sail.features import write_npz_cache
from midogpp_thesis.real_features.sail.midogpp_multiaxis import (
    MidogPPMultiAxisConfig,
    load_midogpp_multiaxis_config,
    run_midogpp_multiaxis_baseline,
)


def test_midogpp_multiaxis_run_outputs_protocol_artifacts_and_mlp_skip(tmp_path: Path) -> None:
    manifest, cache = _write_multiaxis_fixture(tmp_path, per_class=6)
    config = MidogPPMultiAxisConfig(
        manifest_path=str(manifest),
        feature_cache_path=str(cache),
        artifacts_root="artifacts/midogpp",
        bootstrap_reps=50,
        mlp_seeds=(42,),
        allow_npz_test_cache=True,
    )

    result = run_midogpp_multiaxis_baseline(config=config, repo_root=tmp_path)

    for path in result.output_paths.values():
        assert path.exists()
    protocol = json.loads(result.output_paths["protocol_manifest"].read_text())
    assert protocol["positive_label"] == 1
    assert protocol["feature_dim"] == 2
    assert protocol["cache_building_in_scope"] is False

    leakage = json.loads(result.output_paths["leakage_report"].read_text())
    assert leakage["status"] == "PASS"
    assert leakage["target_labels_used_for_fitting"] is False
    assert leakage["target_labels_used_for_scoring_only"] is True

    metrics = _read_csv(result.output_paths["per_axis_domain_metrics"])
    tumor_logistic = [
        row for row in metrics if row["axis"] == "tumor_type" and row["model_type"] == "logistic_regression"
    ]
    assert len(tumor_logistic) == 3
    assert {row["status"] for row in tumor_logistic} == {"valid"}
    assert all(float(row["n_source_pos"]) >= 10 for row in tumor_logistic)
    assert all(float(row["n_eval_pos"]) >= 5 for row in tumor_logistic)
    assert all(row["precision_pos"] != "" for row in tumor_logistic)
    assert all(row["ci_low"] != "" and row["ci_high"] != "" for row in tumor_logistic)

    mlp_rows = [row for row in metrics if row["axis"] == "tumor_type" and row["model_type"] == "mlp"]
    assert mlp_rows
    assert {row["status"] for row in mlp_rows} == {"skipped_insufficient_support_for_mlp"}

    summary = _read_csv(result.output_paths["axis_summary"])
    tumor_summary = [
        row for row in summary if row["axis"] == "tumor_type" and row["model_type"] == "logistic_regression"
    ][0]
    assert tumor_summary["decision_valid"] == "true"
    assert tumor_summary["global_failure_gate_axis"] == "true"

    species_summary = [
        row for row in summary if row["axis"] == "species" and row["model_type"] == "logistic_regression"
    ][0]
    assert species_summary["axis_high_confounding"] == "true"
    assert species_summary["global_failure_gate_axis"] == "false"


def test_midogpp_multiaxis_class_minimum_statuses_are_specific(tmp_path: Path) -> None:
    manifest, cache = _write_multiaxis_fixture(tmp_path, per_class=6, weak_domain="T2", weak_domain_pos=4)
    config = MidogPPMultiAxisConfig(
        manifest_path=str(manifest),
        feature_cache_path=str(cache),
        artifacts_root="artifacts/weak",
        bootstrap_reps=10,
        mlp_seeds=(),
        allow_npz_test_cache=True,
    )

    result = run_midogpp_multiaxis_baseline(config=config, repo_root=tmp_path)

    metrics = _read_csv(result.output_paths["per_axis_domain_metrics"])
    weak = [
        row
        for row in metrics
        if row["axis"] == "tumor_type"
        and row["heldout_domain_name"] == "T2"
        and row["model_type"] == "logistic_regression"
    ][0]
    assert weak["status"] == "invalid_too_few_eval_pos"
    assert "n_eval_pos=4" in weak["error_message"]


def test_midogpp_multiaxis_case_overlap_is_protocol_failure(tmp_path: Path) -> None:
    manifest, cache = _write_multiaxis_fixture(tmp_path, per_class=6, shared_case_across_tumors=True)
    config = MidogPPMultiAxisConfig(
        manifest_path=str(manifest),
        feature_cache_path=str(cache),
        artifacts_root="artifacts/overlap",
        bootstrap_reps=10,
        mlp_seeds=(),
        allow_npz_test_cache=True,
    )

    result = run_midogpp_multiaxis_baseline(config=config, repo_root=tmp_path)

    metrics = _read_csv(result.output_paths["per_axis_domain_metrics"])
    tumor_rows = [
        row for row in metrics if row["axis"] == "tumor_type" and row["model_type"] == "logistic_regression"
    ]
    assert any(row["status"] == "protocol_failed_case_overlap" for row in tumor_rows)
    leakage = json.loads(result.output_paths["leakage_report"].read_text())
    assert leakage["status"] == "FOLD_PROTOCOL_FAILURES_REPORTED"
    overlaps = _read_csv(result.output_paths["source_target_identity_overlap"])
    assert any(int(row["case_overlap_count"]) > 0 for row in overlaps if row["axis"] == "tumor_type")


def test_midogpp_multiaxis_config_and_cli_smoke(tmp_path: Path) -> None:
    manifest, cache = _write_multiaxis_fixture(tmp_path, per_class=6)
    config_path = tmp_path / "midogpp.yaml"
    config_path.write_text(
        f"""
experiment:
  name: midogpp_virchow2_real_feature_multiaxis_baseline
inputs:
  manifest_path: {manifest}
  feature_cache_path: {cache}
  allow_npz_test_cache: true
positive_label: 1
label_mapping:
  "0": non_mitotic
  "1": mitotic
validity_thresholds:
  min_source: 20
  min_eval: 10
  min_source_pos: 10
  min_source_neg: 10
  min_eval_pos: 5
  min_eval_neg: 5
mlp_sensitivity:
  seeds: []
bootstrap:
  reps: 10
output:
  artifacts_root: {tmp_path / "artifacts" / "cli"}
""",
        encoding="utf-8",
    )

    loaded = load_midogpp_multiaxis_config(config_path)
    assert loaded.manifest_path == str(manifest)
    assert loaded.allow_npz_test_cache is True

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "midogpp_thesis",
            "real-features",
            "run-midogpp-multiaxis",
            "--config",
            str(config_path),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "midogpp_virchow2_real_feature_multiaxis_complete"
    assert Path(payload["outputs"]["decision_report"]).exists()


def _write_multiaxis_fixture(
    root: Path,
    *,
    per_class: int,
    weak_domain: str | None = None,
    weak_domain_pos: int | None = None,
    shared_case_across_tumors: bool = False,
) -> tuple[Path, Path]:
    import numpy as np

    manifest = root / "manifest.csv"
    cache = root / "cache" / "virchow2" / "seed42" / "embeddings" / "train.npz"
    domains = [
        ("T0", "S0", "L0", "human"),
        ("T1", "S1", "L1", "dog"),
        ("T2", "S2", "L2", "human"),
    ]
    rows = []
    embeddings = []
    cache_metadata = []
    for domain_idx, (tumor, scanner, lab, species) in enumerate(domains):
        for label in (0, 1):
            count = int(weak_domain_pos) if weak_domain == tumor and label == 1 and weak_domain_pos is not None else per_class
            for idx in range(count):
                sample_id = f"{tumor}_y{label}_{idx}"
                case_id = f"case_{tumor}_y{label}_{idx}"
                if shared_case_across_tumors and idx == 0 and label == 1 and tumor in {"T0", "T1"}:
                    case_id = "shared_case_pos"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "case_id": case_id,
                        "image_path": f"dummy/{sample_id}.png",
                        "label": str(label),
                        "split": "train",
                        "scanner_model": scanner,
                        "tumor_type": tumor,
                        "lab_or_origin": lab,
                        "species": species,
                    }
                )
                base = np.array([float(label) * 2.0 + domain_idx * 0.05, float(label) * -1.0 + domain_idx * 0.05])
                embeddings.append(base + np.array([idx * 0.001, -idx * 0.001]))
                cache_metadata.append(
                    {
                        "sample_id": sample_id,
                        "label": label,
                        "split": "train",
                        "center": scanner,
                    }
                )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "case_id",
                "image_path",
                "label",
                "split",
                "scanner_model",
                "tumor_type",
                "lab_or_origin",
                "species",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    write_npz_cache(cache, np.asarray(embeddings, dtype=float), cache_metadata)
    return manifest, cache


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))
