from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from midogpp_thesis.data.contract.builder import BuilderConfig, build_contract
from midogpp_thesis.data.contract.validation import ValidationError, validate_contract


SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets/midogpp/schemas/dataset_contract.schema.json"
)


def test_annotation_patch_contract_builds_valid_sail_ready_artifact(tmp_path: Path) -> None:
    raw = _make_midogpp_fixture(tmp_path, domains=3, cases_per_domain=4)
    config = _config(tmp_path, raw, min_eligible_domains=2)

    result = build_contract(config, overwrite=True)

    assert result.status == "pass"
    validation = validate_contract(config.artifact_root, repo_root=tmp_path, schema_path=SCHEMA_PATH)
    assert validation["status"] == "PASS"

    manifest_rows = _read_csv(config.artifact_root / "manifest.csv")
    positive_rows = [row for row in manifest_rows if row["label"] == "1"]
    negative_rows = [row for row in manifest_rows if row["label"] == "0"]
    expected_positives = 3 * 4

    assert len(positive_rows) == expected_positives
    assert len({row["annotation_id"] for row in positive_rows}) == expected_positives
    assert len(negative_rows) == expected_positives
    assert {row["negative_match_scope"] for row in negative_rows} == {"same_case"}
    assert all(row["negative_match_scope"].startswith("positive_") for row in positive_rows)

    _assert_case_disjoint(manifest_rows)
    _assert_sail_columns(manifest_rows)
    assert all(not Path(row["image_path"]).is_absolute() for row in manifest_rows)
    assert all((tmp_path / row["image_path"]).exists() for row in manifest_rows)

    feasibility_rows = _read_csv(config.artifact_root / "domain_feasibility.csv")
    reported_axes = {row["domain_axis"] for row in feasibility_rows}
    assert reported_axes == {"scanner_model", "tumor_type", "tumor_type|lab_or_origin|scanner_model"}

    first_manifest = (config.artifact_root / "manifest.csv").read_text(encoding="utf-8")
    build_contract(config, overwrite=True)
    second_manifest = (config.artifact_root / "manifest.csv").read_text(encoding="utf-8")
    assert second_manifest == first_manifest


def test_composite_axis_blocks_when_too_few_pseudo_domains_are_eligible(tmp_path: Path) -> None:
    raw = _make_midogpp_fixture(tmp_path, domains=3, cases_per_domain=4)
    config = _config(tmp_path, raw, min_eligible_domains=6)

    result = build_contract(config, overwrite=True)

    assert result.status == "blocked_insufficient_eligible_domains"
    contract = json.loads((config.artifact_root / "dataset_contract.json").read_text(encoding="utf-8"))
    assert contract["domain_policy"]["eligible_domain_count"] == 3
    assert contract["domain_policy"]["final_axis_frozen"] is False
    with pytest.raises(ValidationError):
        validate_contract(config.artifact_root, repo_root=tmp_path, schema_path=SCHEMA_PATH)


def test_ineligible_domains_are_marked_with_reasons(tmp_path: Path) -> None:
    raw = _make_midogpp_fixture(tmp_path, domains=2, cases_per_domain=3)
    config = _config(
        tmp_path,
        raw,
        min_eligible_domains=1,
        thresholds={
            "total_cases_min": 20,
            "train_cases_min": 10,
            "eval_cases_min": 10,
            "train_positives_min": 50,
            "train_negatives_min": 50,
            "eval_positives_min": 20,
            "eval_negatives_min": 20,
        },
    )

    result = build_contract(config, overwrite=True)

    assert result.status == "blocked_insufficient_eligible_domains"
    feasibility_rows = _read_csv(config.artifact_root / "domain_feasibility.csv")
    ineligible = [row for row in feasibility_rows if row["eligible"] == "False"]
    assert ineligible
    assert all(row["ineligible_reasons"] for row in ineligible)


def test_same_domain_negative_fallback_never_crosses_split(tmp_path: Path) -> None:
    raw = _make_midogpp_fixture(
        tmp_path,
        domains=1,
        cases_per_domain=2,
        per_case_annotations={
            "domain0_case0": ("positive",),
            "domain0_case1": ("negative",),
        },
    )
    config = _config(
        tmp_path,
        raw,
        min_eligible_domains=1,
        split_fractions={"train": 1.0, "val": 0.0, "test": 0.0},
    )

    build_contract(config, overwrite=True)

    manifest_rows = _read_csv(config.artifact_root / "manifest.csv")
    negative_rows = [row for row in manifest_rows if row["label"] == "0"]
    assert len(negative_rows) == 1
    assert negative_rows[0]["negative_match_scope"] == "same_domain_same_split"
    assert {row["split"] for row in manifest_rows} == {"train"}
    leakage = json.loads((config.artifact_root / "leakage_report.json").read_text(encoding="utf-8"))
    assert leakage["negative_cross_split_violations"] == []


def _make_midogpp_fixture(
    repo_root: Path,
    *,
    domains: int,
    cases_per_domain: int,
    per_case_annotations: dict[str, tuple[str, ...]] | None = None,
) -> Path:
    raw = repo_root / "raw_midogpp"
    images_dir = raw / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows: list[dict[str, str]] = []
    images_json: list[dict[str, object]] = []
    annotations_json: list[dict[str, object]] = []
    image_id = 1
    annotation_id = 1

    for domain_idx in range(domains):
        for case_idx in range(cases_per_domain):
            case_id = f"domain{domain_idx}_case{case_idx}"
            filename = f"{case_id}.png"
            image_path = images_dir / filename
            Image.new("RGB", (64, 64), color=(80 + domain_idx * 20, 90 + case_idx * 10, 120)).save(image_path)
            metadata_rows.append(
                {
                    "image_path": f"images/{filename}",
                    "case_id": case_id,
                    "scanner_model": f"scanner_{domain_idx}",
                    "lab_or_origin": f"lab_{domain_idx}",
                    "tumor_type": f"tumor_{domain_idx}",
                    "species": "human",
                    "resolution": "0.25",
                }
            )
            images_json.append({"id": image_id, "file_name": filename})
            annotation_kinds = per_case_annotations.get(case_id, ("positive", "negative")) if per_case_annotations else ("positive", "negative")
            for ann_idx, kind in enumerate(annotation_kinds):
                is_positive = kind == "positive"
                annotations_json.append(
                    {
                        "id": f"ann_{annotation_id}",
                        "image_id": image_id,
                        "category_id": 1 if is_positive else 2,
                        "bbox": [8 + ann_idx * 20, 8 + ann_idx * 20, 10, 10],
                    }
                )
                annotation_id += 1
            image_id += 1

    _write_csv(raw / "metadata.csv", metadata_rows)
    payload = {
        "images": images_json,
        "annotations": annotations_json,
        "categories": [{"id": 1, "name": "mitotic figure"}, {"id": 2, "name": "hard negative"}],
    }
    (raw / "annotations.json").write_text(json.dumps(payload), encoding="utf-8")
    return raw


def _config(
    repo_root: Path,
    raw: Path,
    *,
    min_eligible_domains: int,
    thresholds: dict[str, int] | None = None,
    split_fractions: dict[str, float] | None = None,
) -> BuilderConfig:
    artifact_root = repo_root / "datasets/midogpp/contract/annotation_patch_v1"
    return BuilderConfig.from_mapping(
        {
            "artifact": {"name": "midogpp_annotation_patch_v1", "root": artifact_root},
            "inputs": {"root": raw, "metadata": raw / "metadata.csv", "annotations": raw / "annotations.json"},
            "patches": {
                "patch_dir": artifact_root / "patches_224",
                "patch_size": 16,
                "image_quality": 90,
                "bbox_format": "coco_xywh",
            },
            "sampling": {
                "positive_policy": "all_valid_mitotic_annotations",
                "negative_policy": "matched_1_to_1",
                "negative_ratio": 1.0,
                "negative_seed": 7,
            },
            "split": {"seed": 13, "fractions": split_fractions or {"train": 0.50, "val": 0.25, "test": 0.25}},
            "domain": {
                "candidate_axes": ["scanner_model", "tumor_type", "tumor_type|lab_or_origin|scanner_model"],
                "preferred_axis": "tumor_type|lab_or_origin|scanner_model",
                "final_axis_policy": "auto_tumor_lab_scanner",
                "min_eligible_domains": min_eligible_domains,
            },
            "eligibility": thresholds
            or {
                "total_cases_min": 2,
                "train_cases_min": 1,
                "eval_cases_min": 1,
                "train_positives_min": 1,
                "train_negatives_min": 1,
                "eval_positives_min": 1,
                "eval_negatives_min": 1,
            },
        },
        repo_root=repo_root,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _assert_case_disjoint(rows: list[dict[str, str]]) -> None:
    case_to_splits: dict[str, set[str]] = {}
    for row in rows:
        case_to_splits.setdefault(row["case_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in case_to_splits.values())


def _assert_sail_columns(rows: list[dict[str, str]]) -> None:
    required = {"sample_id", "image_path", "label", "split", "center", "magnification"}
    assert rows
    assert required.issubset(rows[0])
    assert all(row["center"] == row["domain_id"] for row in rows)
    assert all(row["magnification"] == row["domain_id"] for row in rows)
