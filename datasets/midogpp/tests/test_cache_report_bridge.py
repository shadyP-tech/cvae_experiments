from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from midogpp_contract.cache_report import (  # noqa: E402
    CacheReportError,
    build_cache_domain_report,
    format_cache_domain_report,
)


def test_missing_optional_cache_report_does_not_fail(tmp_path: Path) -> None:
    artifact = _write_bridge_artifact(tmp_path)

    report = build_cache_domain_report(artifact)

    assert report["cache_report"]["provided"] is False
    assert report["cache_report"]["warnings"] == ["optional_cache_report_absent"]
    assert report["eligible_domain_ids"] == [0, 1]


def test_duplicate_domain_id_fails(tmp_path: Path) -> None:
    artifact = _write_bridge_artifact(tmp_path)
    mapping = json.loads((artifact / "domain_mapping.json").read_text(encoding="utf-8"))
    mapping["domains"][1]["domain_id"] = "0"
    (artifact / "domain_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(CacheReportError, match="duplicate domain_id"):
        build_cache_domain_report(artifact)


def test_eligible_domain_missing_from_manifest_fails(tmp_path: Path) -> None:
    artifact = _write_bridge_artifact(tmp_path)
    rows = _read_csv(artifact / "manifest.csv")
    _write_csv(artifact / "manifest.csv", [row for row in rows if row["domain_id"] != "1"])

    with pytest.raises(CacheReportError, match="zero manifest samples"):
        build_cache_domain_report(artifact)


def test_manifest_containing_ineligible_domain_warns_not_fails(tmp_path: Path) -> None:
    artifact = _write_bridge_artifact(tmp_path)
    cache_report = _write_cache_report(tmp_path, {"train": 5, "test": 5, "val": 2})

    report = build_cache_domain_report(artifact, cache_report_path=cache_report)

    assert report["cache_report"]["provided"] is True
    assert report["hints"]["sail"]["candidate_centers"] == [0, 1]
    assert any("manifest_contains_ineligible_domains: 2" in warning for warning in report["warnings"])


def test_eligible_ids_are_derived_from_feasibility_not_mapping(tmp_path: Path) -> None:
    artifact = _write_bridge_artifact(tmp_path)

    report = build_cache_domain_report(artifact)

    assert [row["domain_id"] for row in report["mapped_domains"]] == [0, 1, 2]
    assert report["eligible_domain_ids"] == [0, 1]


def test_json_schema_stable(tmp_path: Path) -> None:
    artifact = _write_bridge_artifact(tmp_path)

    report = build_cache_domain_report(artifact)

    assert tuple(report) == (
        "schema_version",
        "artifact_root",
        "domain_axis",
        "mapped_domains",
        "eligible_domain_ids",
        "ineligible_domains",
        "manifest_counts",
        "cache_report",
        "hints",
        "warnings",
    )
    assert report["schema_version"] == "midogpp_cache_report_v1"
    assert set(report["hints"]) == {"sail", "c63"}
    assert report["hints"]["c63"]["blocked"] is True


def test_text_output_says_c63_is_not_experiment_ready(tmp_path: Path) -> None:
    artifact = _write_bridge_artifact(tmp_path)

    text = format_cache_domain_report(build_cache_domain_report(artifact))

    assert "C6.3 hint" in text
    assert "blocked: true" in text
    assert "blocked until positive-union logic is generalized beyond the Camelyon 5-domain assumption" in text


def test_no_yaml_written(tmp_path: Path) -> None:
    artifact = _write_bridge_artifact(tmp_path)

    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*.yaml"))
    build_cache_domain_report(artifact)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*.yaml"))

    assert after == before


def test_no_torch_pt_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = _write_bridge_artifact(tmp_path)
    cache_report = _write_cache_report(tmp_path, {"train": 5, "test": 5, "val": 2})
    original_open = Path.open

    def guarded_open(self: Path, *args, **kwargs):
        if self.suffix == ".pt":
            raise AssertionError(f"bridge must not open .pt cache files: {self}")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    report = build_cache_domain_report(artifact, cache_report_path=cache_report)

    assert report["cache_report"]["provided"] is True


def test_cache_split_count_mismatch_fails(tmp_path: Path) -> None:
    artifact = _write_bridge_artifact(tmp_path)
    cache_report = _write_cache_report(tmp_path, {"train": 5, "test": 4, "val": 2})

    with pytest.raises(CacheReportError, match="split_counts contradict"):
        build_cache_domain_report(artifact, cache_report_path=cache_report)


def _write_bridge_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "datasets/midogpp/artifacts/midogpp_annotation_patch_v1"
    artifact.mkdir(parents=True)
    axis = "tumor_type|lab_or_origin|scanner_model"
    manifest_rows = _manifest_rows()
    feasibility_rows = _feasibility_rows(axis)
    mapping = {
        "schema_version": "midogpp_domain_mapping_v1",
        "domain_axis": axis,
        "domain_name_to_id": {f"domain_{idx}": str(idx) for idx in range(3)},
        "domains": [
            {"domain_id": "0", "domain_name": "domain_0", "n_cases": 5, "n_rows": 5},
            {"domain_id": "1", "domain_name": "domain_1", "n_cases": 5, "n_rows": 5},
            {"domain_id": "2", "domain_name": "domain_2", "n_cases": 2, "n_rows": 2},
        ],
    }
    contract = {
        "schema_version": "midogpp_annotation_patch_dataset_contract_v1",
        "artifact_name": "midogpp_annotation_patch_v1",
        "status": "pass",
        "domain_policy": {"selected_domain_axis": axis},
    }
    (artifact / "dataset_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (artifact / "domain_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    _write_csv(artifact / "manifest.csv", manifest_rows)
    _write_csv(artifact / "domain_feasibility.csv", feasibility_rows)
    return artifact


def _write_cache_report(tmp_path: Path, split_counts: dict[str, int]) -> Path:
    path = tmp_path / "sail/artifacts/pathology_embeddings_midogpp_annotation_patch_v1/virchow2/seed42/reports/cache_builder_report.json"
    payload = {
        "schema_version": "sail_virchow2_cache_builder_report_v1",
        "status": "complete",
        "backbone_name": "virchow2",
        "split_counts": split_counts,
        "output_paths": {
            "train": "sail/artifacts/pathology_embeddings_midogpp_annotation_patch_v1/virchow2/seed42/embeddings/train.pt",
            "val": "sail/artifacts/pathology_embeddings_midogpp_annotation_patch_v1/virchow2/seed42/embeddings/val.pt",
            "test": "sail/artifacts/pathology_embeddings_midogpp_annotation_patch_v1/virchow2/seed42/embeddings/test.pt",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _manifest_rows() -> list[dict[str, str]]:
    rows = []
    row_id = 0
    for domain_id in ("0", "1"):
        for split, labels in (("train", (1, 0)), ("test", (1, 0)), ("val", (1,))):
            for label in labels:
                rows.append(_manifest_row(row_id, domain_id, split, label))
                row_id += 1
    for split, label in (("train", 1), ("test", 0)):
        rows.append(_manifest_row(row_id, "2", split, label))
        row_id += 1
    return rows


def _manifest_row(row_id: int, domain_id: str, split: str, label: int) -> dict[str, str]:
    return {
        "sample_id": f"s{row_id}",
        "case_id": f"case_{domain_id}_{row_id}",
        "image_path": f"patches/s{row_id}.jpg",
        "label": str(label),
        "split": split,
        "domain_axis": "tumor_type|lab_or_origin|scanner_model",
        "domain_name": f"domain_{domain_id}",
        "domain_id": domain_id,
        "center": domain_id,
        "magnification": domain_id,
    }


def _feasibility_rows(axis: str) -> list[dict[str, str]]:
    return [
        _feasibility_row(axis, "0", True, "", 5, 5, 2, 2, 1, 1, 1, 1),
        _feasibility_row(axis, "1", True, "", 5, 5, 2, 2, 1, 1, 1, 1),
        _feasibility_row(axis, "2", False, "train_negatives<1;eval_positives<1", 2, 2, 1, 1, 1, 0, 0, 1),
    ]


def _feasibility_row(
    axis: str,
    domain_id: str,
    eligible: bool,
    reasons: str,
    total_rows: int,
    total_cases: int,
    train_cases: int,
    eval_cases: int,
    train_pos: int,
    train_neg: int,
    eval_pos: int,
    eval_neg: int,
) -> dict[str, str]:
    return {
        "domain_axis": axis,
        "domain_name": f"domain_{domain_id}",
        "domain_id_for_axis": domain_id,
        "total_rows": str(total_rows),
        "total_cases": str(total_cases),
        "train_cases": str(train_cases),
        "eval_cases": str(eval_cases),
        "train_positives": str(train_pos),
        "train_negatives": str(train_neg),
        "eval_positives": str(eval_pos),
        "eval_negatives": str(eval_neg),
        "eligible": str(eligible),
        "ineligible_reasons": reasons,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
