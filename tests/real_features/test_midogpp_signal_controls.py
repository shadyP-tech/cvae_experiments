from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

from midogpp_thesis.real_features.sail.features import write_npz_cache
from midogpp_thesis.real_features.sail.midogpp_multiaxis import _read_manifest
from midogpp_thesis.real_features.sail.midogpp_signal_controls import (
    BALANCED_CONTROL,
    LOGISTIC_MODEL_TYPE,
    MidogPPSignalControlsConfig,
    POOLED_CONTROL,
    VALID_STATUS,
    WITHIN_TUMOR_CONTROL,
    _case_cluster_bacc_ci,
    _case_disjoint_split,
    _split_counts,
    _split_status,
    _tumor_class_balanced_indices,
    load_midogpp_signal_controls_config,
    run_midogpp_signal_controls,
)


def test_midogpp_signal_controls_run_outputs_protocol_artifacts(tmp_path: Path) -> None:
    manifest, cache = _write_signal_fixture(tmp_path, per_class=30)
    config = MidogPPSignalControlsConfig(
        manifest_path=str(manifest),
        feature_cache_path=str(cache),
        artifacts_root="artifacts/signal",
        split_seeds=(42, 43),
        bootstrap_reps=50,
        mlp_seeds=(),
        allow_npz_test_cache=True,
        prior_lodo_axis_summary_path=None,
    )

    result = run_midogpp_signal_controls(config=config, repo_root=tmp_path)

    for path in result.output_paths.values():
        assert path.exists()
    protocol = json.loads(result.output_paths["protocol_manifest"].read_text())
    assert protocol["feature_dim"] == 3
    assert protocol["threshold_policy"] == "fixed_0.5_classifier_rule_not_calibrated_probability"
    assert protocol["cache_building_in_scope"] is False

    leakage = json.loads(result.output_paths["leakage_report"].read_text())
    assert leakage["fit_only_standardization"] is True
    assert leakage["threshold_tuned_on_eval"] is False

    controls = _read_csv(result.output_paths["control_metrics"])
    summaries = [row for row in controls if row["aggregation_level"] == "summary" and row["model_type"] == LOGISTIC_MODEL_TYPE]
    assert {row["control_name"] for row in summaries} == {POOLED_CONTROL, BALANCED_CONTROL}
    assert all(row["ci_method"] == "case_cluster_conservative_seed_aggregate" for row in summaries)
    assert all(row["valid_seed_count"] == "2" for row in summaries)

    domains = _read_csv(result.output_paths["domain_control_metrics"])
    domain_summaries = [
        row for row in domains if row["aggregation_level"] == "summary" and row["model_type"] == LOGISTIC_MODEL_TYPE
    ]
    assert len(domain_summaries) == 3
    assert {row["control_name"] for row in domain_summaries} == {WITHIN_TUMOR_CONTROL}

    negatives = _read_csv(result.output_paths["negative_control_metrics"])
    negative_summaries = [row for row in negatives if row["aggregation_level"] == "summary"]
    assert negative_summaries
    assert all(row["bacc_delta_vs_real"] != "" for row in negative_summaries)

    audit = _read_csv(result.output_paths["identity_overlap_audit"])
    assert audit
    assert {row["status"] for row in audit} == {"PASS"}


def test_tumor_class_balanced_split_balances_primary_axis_and_class(tmp_path: Path) -> None:
    manifest, _cache = _write_signal_fixture(tmp_path, per_class=25)
    rows = tuple(row for row in _read_manifest(manifest, positive_label=1) if row.split == "train")
    config = MidogPPSignalControlsConfig(
        manifest_path=str(manifest),
        feature_cache_path="unused",
        split_seeds=(42,),
        mlp_seeds=(),
        allow_npz_test_cache=True,
    )

    fit_idx, eval_idx = _case_disjoint_split(rows, tuple(range(len(rows))), config=config, seed=42)
    balanced_fit, balanced_eval = _tumor_class_balanced_indices(rows, fit_idx, eval_idx, seed=42)

    for indices in (balanced_fit, balanced_eval):
        counts: dict[tuple[str, int], int] = {}
        for idx in indices:
            key = (str(rows[idx].metadata["tumor_type"]), int(rows[idx].label))
            counts[key] = counts.get(key, 0) + 1
        assert len(set(counts.values())) == 1
        assert len(counts) == 6


def test_case_level_support_gate_and_mlp_skip_are_specific(tmp_path: Path) -> None:
    manifest, _cache = _write_signal_fixture(tmp_path, per_class=12, collapse_eval_pos_case=True)
    rows = tuple(row for row in _read_manifest(manifest, positive_label=1) if row.split == "train")
    config = MidogPPSignalControlsConfig(
        manifest_path=str(manifest),
        feature_cache_path="unused",
        split_seeds=(42,),
        bootstrap_reps=10,
        mlp_seeds=(42,),
        allow_npz_test_cache=True,
    )
    fit_idx = tuple(idx for idx, row in enumerate(rows) if str(row.metadata["tumor_type"]) != "T0")
    eval_idx = tuple(idx for idx, row in enumerate(rows) if str(row.metadata["tumor_type"]) == "T0")

    status, error = _split_status(config, rows, fit_idx, eval_idx, model_type=LOGISTIC_MODEL_TYPE)

    assert status == "invalid_too_few_eval_pos_cases"
    assert "n_eval_pos_cases=1" in error

    mlp_manifest, _mlp_cache = _write_signal_fixture(tmp_path / "mlp", per_class=15)
    mlp_rows = tuple(row for row in _read_manifest(mlp_manifest, positive_label=1) if row.split == "train")
    pooled_fit, pooled_eval = _case_disjoint_split(mlp_rows, tuple(range(len(mlp_rows))), config=config, seed=42)
    mlp_status, mlp_error = _split_status(config, mlp_rows, pooled_fit, pooled_eval, model_type="mlp")
    assert mlp_status == "skipped_insufficient_support_for_mlp"
    assert "n_fit=" in mlp_error


def test_case_cluster_bootstrap_ci_uses_case_clusters() -> None:
    y_true = [0, 0, 1, 1, 0, 1]
    y_pred = [0, 1, 1, 1, 0, 0]
    cases = ["c0", "c0", "c1", "c1", "c2", "c3"]

    low, high, method = _case_cluster_bacc_ci(y_true, y_pred, cases, reps=100, seed=7)

    assert method == "case_cluster"
    assert 0.0 <= low <= high <= 1.0


def test_midogpp_signal_controls_config_and_cli_smoke(tmp_path: Path) -> None:
    manifest, cache = _write_signal_fixture(tmp_path, per_class=20)
    config_path = tmp_path / "signal.yaml"
    config_path.write_text(
        f"""
experiment:
  name: midogpp_virchow2_real_feature_signal_controls
inputs:
  manifest_path: {manifest}
  feature_cache_path: {cache}
  allow_npz_test_cache: true
positive_label: 1
label_mapping:
  "0": non_mitotic
  "1": mitotic
split:
  seeds: [42]
  eval_fraction: 0.2
validity_thresholds:
  min_fit: 20
  min_eval: 10
  min_fit_pos: 10
  min_fit_neg: 10
  min_eval_pos: 5
  min_eval_neg: 5
  min_fit_pos_cases: 3
  min_fit_neg_cases: 3
  min_eval_pos_cases: 2
  min_eval_neg_cases: 2
mlp_sensitivity:
  seeds: []
bootstrap:
  reps: 10
prior_lodo:
  axis_summary_path:
output:
  artifacts_root: {tmp_path / "artifacts" / "cli"}
""",
        encoding="utf-8",
    )

    loaded = load_midogpp_signal_controls_config(config_path)
    assert loaded.manifest_path == str(manifest)
    assert loaded.prior_lodo_axis_summary_path is None

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "midogpp_thesis",
            "real-features",
            "run-midogpp-signal-controls",
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
    assert payload["status"] == "midogpp_virchow2_real_feature_signal_controls_complete"
    assert Path(payload["outputs"]["decision_report"]).exists()


def _write_signal_fixture(
    root: Path,
    *,
    per_class: int,
    collapse_eval_pos_case: bool = False,
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
            for idx in range(per_class):
                sample_id = f"{tumor}_y{label}_{idx}"
                case_id = f"case_{tumor}_y{label}_{idx}"
                if collapse_eval_pos_case and tumor == "T0" and label == 1:
                    case_id = "collapsed_T0_pos_case"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "case_id": case_id,
                        "image_path": f"dummy/{sample_id}.png",
                        "annotation_id": f"ann_{sample_id}",
                        "bbox_x": str(idx),
                        "bbox_y": str(idx + 1),
                        "bbox_w": "32",
                        "bbox_h": "32",
                        "patch_center_x": str(100 + idx),
                        "patch_center_y": str(200 + idx),
                        "label": str(label),
                        "split": "train",
                        "scanner_model": scanner,
                        "tumor_type": tumor,
                        "lab_or_origin": lab,
                        "species": species,
                    }
                )
                base = np.array([float(label) * 4.0, float(label) * -3.0, float(domain_idx) * 0.1])
                embeddings.append(base + np.array([idx * 0.001, -idx * 0.001, 0.0]))
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
                "annotation_id",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "patch_center_x",
                "patch_center_y",
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
