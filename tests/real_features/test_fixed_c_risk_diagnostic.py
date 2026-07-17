from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from midogpp_thesis.real_features.classifier_reference.fixed_c_risk_diagnostic import (
    FixedCRiskDiagnosticConfig,
    compute_risk_weights,
    load_fixed_c_risk_config,
    run_fixed_c_risk_diagnostic,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError
from midogpp_thesis.real_features.classifier_reference.schemas.fixed_c_risk_diagnostic import (
    FIXED_CLASSIFIER_CONFIG_HASH,
    RISK_POLICY_FORMULAS,
    RISK_POLICY_IDS,
    assert_fixed_c_risk_artifacts,
)


def test_exact_risk_weight_formulas_sum_to_n() -> None:
    labels = (0, 0, 0, 1, 0, 1)
    domains = ("a", "a", "a", "a", "b", "b")
    sample_ids = tuple(f"s{index}" for index in range(len(labels)))
    plans = {
        policy: compute_risk_weights(
            labels, domains, policy, sample_ids=sample_ids
        )
        for policy in RISK_POLICY_IDS
    }
    assert plans["pooled"].weights == pytest.approx((1, 1, 1, 1, 1, 1))
    assert plans["global_class"].group_weights == pytest.approx(
        {"class=0": 6 / 8, "class=1": 6 / 4}
    )
    assert plans["domain"].group_weights == pytest.approx(
        {"domain=a": 6 / 8, "domain=b": 6 / 4}
    )
    assert plans["domain_class"].group_weights == pytest.approx(
        {
            "domain=a|class=0": 6 / 12,
            "domain=a|class=1": 6 / 4,
            "domain=b|class=0": 6 / 4,
            "domain=b|class=1": 6 / 4,
        }
    )
    for policy, plan in plans.items():
        assert plan.formula == RISK_POLICY_FORMULAS[policy]
        assert sum(plan.weights) == pytest.approx(len(labels))
        assert min(plan.weights) > 0.0


def test_domain_class_zero_cell_fails_closed() -> None:
    with pytest.raises(ProtocolError, match="domain×class cells are missing"):
        compute_risk_weights(
            (0, 1, 0),
            ("a", "a", "b"),
            "domain_class",
            sample_ids=("s0", "s1", "s2"),
        )


def test_partial_fixture_writes_and_validates_full_generated_bundle(
    tmp_path: Path,
) -> None:
    manifest, cache = _write_fixture(tmp_path / "midogpp_fixture")
    root = run_fixed_c_risk_diagnostic(
        FixedCRiskDiagnosticConfig(
            name="fixed_c_risk_diagnostic_v1",
            artifact_root=tmp_path / "artifact",
            manifest_path=manifest,
            feature_cache_path=cache,
            heldout_centers=("0", "1", "2"),
            expected_feature_dim=4,
            allow_partial_test_coverage=True,
        )
    )
    assert_fixed_c_risk_artifacts(root)
    generated = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert generated == {
        "manifests/frozen_protocol_snapshot.json",
        "manifests/protocol_manifest.json",
        "reports/leakage_provenance_report.json",
        "reports/diagnostic_summary.json",
        "reports/diagnostic_report.md",
        "reports/runtime_summary.json",
        "tables/fixed_c_risk_results.csv",
        "tables/fixed_c_risk_predictions.csv",
        "tables/fixed_c_risk_weight_audit.csv",
        "tables/fixed_c_risk_paired_comparison.csv",
    }
    results = _read_csv(root / "tables/fixed_c_risk_results.csv")
    audits = _read_csv(root / "tables/fixed_c_risk_weight_audit.csv")
    paired = _read_csv(root / "tables/fixed_c_risk_paired_comparison.csv")
    assert len(results) == 12
    assert len(audits) == 12
    assert len(paired) == 3
    assert {
        (row["heldout_center"], row["risk_policy_id"]) for row in results
    } == {
        (heldout, policy)
        for heldout in ("0", "1", "2")
        for policy in RISK_POLICY_IDS
    }
    assert {row["contrast_id"] for row in paired} == {
        "domain_class_minus_pooled"
    }
    assert {row["fixed_classifier_config_hash"] for row in results} == {
        FIXED_CLASSIFIER_CONFIG_HASH
    }
    assert all(row["target_rows_used"] == "false" for row in audits)


def test_production_mode_rejects_partial_coverage(tmp_path: Path) -> None:
    manifest, cache = _write_fixture(tmp_path / "midogpp_fixture")
    with pytest.raises(ProtocolError, match="Production fixed-C risk runtime locks"):
        run_fixed_c_risk_diagnostic(
            FixedCRiskDiagnosticConfig(
                name="fixed_c_risk_diagnostic_v1",
                artifact_root=tmp_path / "artifact",
                manifest_path=manifest,
                feature_cache_path=cache,
                heldout_centers=("0", "1", "2"),
                expected_feature_dim=4,
            )
        )


def test_config_drift_is_rejected(tmp_path: Path) -> None:
    source = Path(
        "experiments/midogpp/stages/10_real_feature_reference/configs/"
        "fixed_c_risk_diagnostic_v1.yaml"
    )
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["classifier"]["C"] = 0.1
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="classifier config/hash drifted"):
        load_fixed_c_risk_config(drifted)


def test_prediction_and_weight_tampering_are_rejected(tmp_path: Path) -> None:
    manifest, cache = _write_fixture(tmp_path / "midogpp_fixture")
    root = run_fixed_c_risk_diagnostic(
        FixedCRiskDiagnosticConfig(
            name="fixed_c_risk_diagnostic_v1",
            artifact_root=tmp_path / "artifact",
            manifest_path=manifest,
            feature_cache_path=cache,
            heldout_centers=("0", "1", "2"),
            expected_feature_dim=4,
            allow_partial_test_coverage=True,
        )
    )
    changed_prediction = tmp_path / "changed_prediction"
    shutil.copytree(root, changed_prediction)
    prediction_path = changed_prediction / "tables/fixed_c_risk_predictions.csv"
    prediction_rows = _read_csv(prediction_path)
    prediction_rows[0]["y_pred"] = str(1 - int(prediction_rows[0]["y_pred"]))
    _write_csv(prediction_path, prediction_rows)
    with pytest.raises(ProtocolError, match="bundle hash mismatch"):
        assert_fixed_c_risk_artifacts(changed_prediction)

    changed_weight = tmp_path / "changed_weight"
    shutil.copytree(root, changed_weight)
    audit_path = changed_weight / "tables/fixed_c_risk_weight_audit.csv"
    audit_rows = _read_csv(audit_path)
    audit_rows[0]["weight_vector_hash"] = "tampered"
    _write_csv(audit_path, audit_rows)
    with pytest.raises(ProtocolError, match="bundle hash mismatch"):
        assert_fixed_c_risk_artifacts(changed_weight)


def _write_fixture(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    manifest = root / "midogpp_manifest.csv"
    cache = root / "virchow2_midogpp_train.npz"
    rows: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    embeddings: list[np.ndarray] = []
    rng = np.random.default_rng(17)
    index = 0
    for center_index, center in enumerate(("0", "1", "2")):
        for local in range(12):
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
                {
                    "sample_id": sample_id,
                    "label": label,
                    "center": center,
                    "split": "train",
                }
            )
            vector = rng.normal(size=4)
            vector[0] += (2.0 * label) + (0.05 * center_index)
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
