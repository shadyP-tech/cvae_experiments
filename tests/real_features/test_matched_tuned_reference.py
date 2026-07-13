from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from midogpp_thesis.cvae.preservation.tuned_reference import load_tuned_classifier_reference
from midogpp_thesis.cvae.preservation.splits import source_only_frame
from midogpp_thesis.real_features.classifier_reference.classifiers import ClassifierSpec
from midogpp_thesis.real_features.classifier_reference.artifacts import stable_hash
from midogpp_thesis.real_features.classifier_reference.matched_reference import (
    CANONICAL_GRID_HASH,
    MatchedReferenceConfig,
    canonical_matched_reference_specs,
    run_matched_reference,
    select_nested_predict_spec,
    select_nested_predict_spec_source_only,
)
from midogpp_thesis.real_features.classifier_reference.midogpp_real_feature_classifier import (
    load_midogpp_real_feature_frame,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError
from midogpp_thesis.real_features.classifier_reference.schemas.matched_reference import (
    _validate_workspace_provenance,
    assert_matched_reference_artifacts,
    matched_reference_bundle_hash,
)


def test_canonical_matched_reference_grid_is_frozen() -> None:
    specs = canonical_matched_reference_specs()
    assert len(specs) == 20
    from midogpp_thesis.real_features.classifier_reference.classifiers import classifier_grid_hash

    assert classifier_grid_hash(specs) == CANONICAL_GRID_HASH


def test_nested_predict_selection_excludes_outer_and_inner(tmp_path: Path) -> None:
    manifest, cache = _write_fixture(tmp_path / "midogpp_nested")
    frame = load_midogpp_real_feature_frame(
        manifest_path=manifest,
        feature_cache_path=cache,
        expected_feature_dim=4,
    )
    result = select_nested_predict_spec(
        frame,
        outer_target_center="0",
        inner_pseudo_target_center="1",
        candidate_specs=(
            ClassifierSpec(C=0.1, random_state=23),
            ClassifierSpec(C=1.0, random_state=23),
        ),
    )
    assert result.inner_pseudo_target_center == "1"
    for row in result.candidate_rows:
        assert json.loads(str(row["excluded_centers"])) == ["0", "1"]
        assert row["fit_used_outer_target_center"] == "false"
        assert row["fit_used_inner_pseudo_target_center"] == "false"

    source_frame = source_only_frame(frame, outer_target_center="0")
    isolated = select_nested_predict_spec_source_only(
        source_frame,
        outer_target_center="0",
        inner_pseudo_target_center="1",
        candidate_specs=(ClassifierSpec(C=0.1, random_state=23),),
    )
    assert "0" not in source_frame.eligible_centers
    assert isolated.inner_pseudo_target_center == "1"


def test_matched_reference_artifact_is_importable_without_mutating_v1(tmp_path: Path) -> None:
    manifest, cache = _write_fixture(tmp_path / "midogpp_reference")
    root = run_matched_reference(
        MatchedReferenceConfig(
            name="eligible_tuned_real_reference_v2",
            artifact_root=tmp_path / "artifact",
            manifest_path=manifest,
            feature_cache_path=cache,
            heldout_centers=("0", "1"),
            expected_feature_dim=4,
            allow_partial_test_coverage=True,
        )
    )
    imported = load_tuned_classifier_reference(root, required_centers=("0", "1"))
    assert imported.protocol["schema_version"] == "midogpp_eligible_tuned_real_reference_v2"
    assert imported.rows_by_center["0"].selected_classifier_spec.threshold_policy == "predict"
    rows = list(csv.DictReader((root / "tables/classifier_tuned_source_results.csv").open()))
    assert {row["method"] for row in rows} == {"source_inner_tuned_predict"}

    changed_score = tmp_path / "changed-score"
    shutil.copytree(root, changed_score)
    score_path = changed_score / "tables/classifier_tuned_source_results.csv"
    score_rows = _read_csv(score_path)
    score_rows[0]["heldout_bacc"] = "0.123"
    _write_csv(score_path, score_rows)
    _rebind_matched_bundle(changed_score)
    with pytest.raises(ProtocolError, match="heldout BACC does not recompute"):
        assert_matched_reference_artifacts(changed_score)

    changed_selection = tmp_path / "changed-selection"
    shutil.copytree(root, changed_selection)
    tuning_path = changed_selection / "tables/source_inner_classifier_tuning.csv"
    tuning_rows = _read_csv(tuning_path)
    center_rows = [row for row in tuning_rows if row["outer_target_center"] == "0"]
    selected = next(row for row in center_rows if row["selected"] == "true")
    replacement = next(row for row in center_rows if row["selected"] == "false")
    selected["selected"] = "false"
    replacement["selected"] = "true"
    _write_csv(tuning_path, tuning_rows)
    _rebind_matched_bundle(changed_selection)
    with pytest.raises(ProtocolError, match="deterministic selection"):
        assert_matched_reference_artifacts(changed_selection)


def test_complete_matched_reference_binds_workspace_inputs(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    (root / "provenance").mkdir(parents=True)
    manifest = tmp_path / "manifest.csv"
    cache = tmp_path / "train.pt"
    manifest.write_text("sample_id\nrow-1\n", encoding="utf-8")
    cache.write_bytes(b"feature-cache")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    cache_hash = hashlib.sha256(cache.read_bytes()).hexdigest()
    protocol = {
        "coverage_mode": "complete",
        "experiment_name": "eligible_tuned_real_reference_v2",
        "experiment_seed": 42,
        "classifier_seed": 23,
        "classifier_grid_hash": CANONICAL_GRID_HASH,
        "manifest_hash": manifest_hash,
        "feature_cache_hash": cache_hash,
    }
    config = {
        "experiment": {"name": "eligible_tuned_real_reference_v2"},
        "inputs": {
            "manifest_path": str(manifest),
            "feature_cache_path": str(cache),
        },
        "run": {
            "experiment_seed": 42,
            "classifier_seed": 23,
            "heldout_centers": "all",
        },
        "classifier_grid": {
            "expected_candidate_count": 20,
            "expected_grid_hash": CANONICAL_GRID_HASH,
            "threshold_policy": "predict",
        },
    }
    (root / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    def artifact_row(artifact_id: str, relative: str, digest: str) -> dict[str, object]:
        return {
            "artifact_id": artifact_id,
            "exists": True,
            "semantic_identities_are_file_hashes": False,
            "file_integrity": {
                "status": "HASHES_RECORDED_NO_EXPECTATIONS",
                "files": [
                    {
                        "path": relative,
                        "exists": True,
                        "computed": {"sha256": digest},
                        "expected": None,
                    }
                ],
            },
        }

    provenance = {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": "midogpp.real_feature.eligible_tuned_predict_reference.v2",
        "stage": "10_real_feature_reference",
        "claim_scope": "real_feature_transfer_only",
        "selection_used_target_eval_artifacts": False,
        "input_artifacts": [
            artifact_row(
                "midogpp_dataset_contract_annotation_patch_v1",
                "manifest.csv",
                manifest_hash,
            ),
            artifact_row(
                "midogpp_virchow2_xyxy_feature_cache_seed42",
                "embeddings/train.pt",
                cache_hash,
            ),
        ],
    }
    provenance_path = root / "provenance/input_artifacts.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    _validate_workspace_provenance(root, protocol)

    provenance["input_artifacts"][1]["file_integrity"]["files"][0]["computed"]["sha256"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ProtocolError, match="workspace input hashes"):
        _validate_workspace_provenance(root, protocol)


def _write_fixture(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    manifest = root / "midogpp_manifest.csv"
    cache = root / "virchow2_midogpp_train.npz"
    rows = []
    metadata = []
    embeddings = []
    rng = np.random.default_rng(11)
    index = 0
    for center_index, center in enumerate(("0", "1", "2", "3")):
        for local in range(10):
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
            metadata.append({"sample_id": sample_id, "label": label, "center": center, "split": "train"})
            vector = rng.normal(size=4)
            vector[0] += label * 2.0 + center_index * 0.05
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
        feature_extractor_json=json.dumps({"backbone_type": "virchow2", "dataset": "midogpp"}),
    )
    return manifest, cache


def _rebind_matched_bundle(root: Path) -> None:
    tuning_path = root / "tables/source_inner_classifier_tuning.csv"
    result_path = root / "tables/classifier_tuned_source_results.csv"
    prediction_path = root / "tables/classifier_tuned_predictions.csv"
    tuning_rows = _read_csv(tuning_path)
    result_rows = _read_csv(result_path)
    prediction_rows = _read_csv(prediction_path)
    protocol_path = root / "manifests/protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["reference_bundle_hash"] = matched_reference_bundle_hash(
        tuning_rows,
        result_rows,
        prediction_rows,
    )
    protocol.pop("protocol_hash", None)
    protocol["protocol_hash"] = stable_hash(protocol)
    for row in (*result_rows, *prediction_rows):
        row["protocol_hash"] = protocol["protocol_hash"]
    _write_csv(result_path, result_rows)
    _write_csv(prediction_path, prediction_rows)
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    leakage_path = root / "reports/leakage_provenance_report.json"
    leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    leakage.update(protocol)
    leakage_path.write_text(
        json.dumps(leakage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
