from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from midogpp_real_feature_gate.runner import RunConfig, SourceInnerReliabilityConfig, run_gate, run_source_inner_reliability
from midogpp_real_feature_gate.validation import validate_artifact_bundle


def test_run_gate_writes_valid_artifact_bundle(tmp_path: Path) -> None:
    manifest, cache = _write_fixture(tmp_path, per_class=3)
    result = run_gate(
        RunConfig(
            manifest_path=manifest,
            feature_cache_path=cache,
            artifact_root=tmp_path / "artifacts",
            repo_root=Path(__file__).resolve().parents[2],
            min_source=8,
            min_eval=4,
            allow_npz_test_cache=True,
        )
    )

    validate_artifact_bundle(tmp_path / "artifacts")
    assert result.output_paths["matrix"].exists()
    assert result.output_paths["predictions"].exists()
    assert result.decision_labels

    matrix = _read_csv(result.output_paths["matrix"])
    source_rows = [row for row in matrix if row["row_role"] == "source_only_transfer"]
    assert source_rows
    assert all(row["adoption_eligible"] == "true" for row in source_rows)
    assert all(row["diagnostic_only"] == "false" for row in source_rows)
    assert all(row["fit_used_target_center"] == "false" for row in source_rows)
    assert all(row["selection_used_target_labels"] == "false" for row in source_rows)
    assert all(row["target_eval_labels_used_for_scoring_only"] == "true" for row in source_rows)

    pooled_rows = [row for row in matrix if row["row_role"] == "pooled_diagnostic_ceiling"]
    assert pooled_rows
    assert all(row["adoption_eligible"] == "false" for row in pooled_rows)
    assert all(row["diagnostic_only"] == "true" for row in pooled_rows)

    protocol = json.loads(result.output_paths["protocol_manifest"].read_text(encoding="utf-8"))
    assert protocol["schema_version"] == "midogpp_real_feature_transfer_ceiling_v1"
    assert protocol["sail_reference"]["runtime_dependency"] is False


def test_quarantine_center_is_diagnostic_only(tmp_path: Path) -> None:
    manifest, cache = _write_fixture(tmp_path, per_class=3, include_center4=True)
    result = run_gate(
        RunConfig(
            manifest_path=manifest,
            feature_cache_path=cache,
            artifact_root=tmp_path / "artifacts",
            repo_root=Path(__file__).resolve().parents[2],
            min_source=8,
            min_eval=4,
            allow_npz_test_cache=True,
        )
    )

    matrix = _read_csv(result.output_paths["matrix"])
    center4 = [row for row in matrix if row["heldout_center"] == "4"]
    assert center4
    assert {row["row_role"] for row in center4} == {"pooled_diagnostic_ceiling"}
    assert all(row["claim_role"] == "quarantine_only" for row in center4)
    assert all(row["adoption_eligible"] == "false" for row in center4)


def test_run_source_inner_reliability_writes_protocol_artifacts(tmp_path: Path) -> None:
    manifest, cache = _write_fixture(tmp_path, per_class=4)
    result = run_source_inner_reliability(
        SourceInnerReliabilityConfig(
            manifest_path=manifest,
            feature_cache_path=cache,
            artifact_root=tmp_path / "source_inner_artifacts",
            repo_root=Path(__file__).resolve().parents[2],
            min_source=4,
            min_eval=4,
            allow_npz_test_cache=True,
        )
    )

    for key in (
        "source_inner_reliability",
        "ensemble_weights",
        "member_predictions_manifest",
        "ensemble_results",
        "ensemble_predictions",
        "protocol_manifest",
        "leakage_provenance_report",
    ):
        assert result.output_paths[key].exists()

    weights = _read_csv(result.output_paths["ensemble_weights"])
    assert weights
    assert all(row["selection_source"] == "source_inner" for row in weights)
    assert all(row["selection_used_target_labels"] == "false" for row in weights)
    assert all(row["fit_used_target_center"] == "false" for row in weights)
    by_target_role: dict[tuple[str, str], float] = {}
    for row in weights:
        key = (row["heldout_center"], row["row_role"])
        by_target_role[key] = by_target_role.get(key, 0.0) + float(row["w_i_utility"])
        assert row["expert_center"] != row["heldout_center"]
    assert all(total == pytest.approx(1.0) for total in by_target_role.values())

    reliability = _read_csv(result.output_paths["source_inner_reliability"])
    excluded = [row for row in reliability if row["pseudo_target_center"] == row["expert_center"]]
    assert excluded
    assert all(row["eligible"] == "false" for row in excluded)
    assert all(row["fallback_reason"] == "pseudo_target_expert_excluded" for row in excluded)

    results = _read_csv(result.output_paths["ensemble_results"])
    roles = {row["row_role"] for row in results}
    assert {"source_inner_weighted_ensemble", "uniform_dense_ensemble"}.issubset(roles)
    assert all(row["selection_used_target_labels"] == "false" for row in results)
    assert all(row["target_eval_labels_used_for_scoring_only"] == "true" for row in results)


def _write_fixture(root: Path, *, per_class: int, include_center4: bool = False) -> tuple[Path, Path]:
    import numpy as np

    manifest = root / "manifest.csv"
    cache = root / "cache.npz"
    centers = ["0", "1", "2"]
    if include_center4:
        centers.append("4")
    rows = []
    embeddings = []
    metadata = []
    for center_idx, center in enumerate(centers):
        for label in (0, 1):
            for item_idx in range(per_class):
                sample_id = f"c{center}_y{label}_{item_idx}"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "case_id": f"case_{sample_id}",
                        "label": str(label),
                        "split": "train",
                        "center": center,
                        "tumor_type": f"tumor_{center_idx % 2}",
                    }
                )
                embeddings.append([float(label) * 2.0 + center_idx * 0.01, float(label) * -1.0])
                metadata.append({"sample_id": sample_id, "label": label, "split": "train"})

    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "case_id", "label", "split", "center", "tumor_type"])
        writer.writeheader()
        writer.writerows(rows)
    np.savez(
        cache,
        embeddings=np.asarray(embeddings, dtype=float),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    return manifest, cache


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
