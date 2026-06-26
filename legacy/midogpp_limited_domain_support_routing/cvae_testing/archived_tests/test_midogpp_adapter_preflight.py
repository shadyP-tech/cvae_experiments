from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SUPPORT_ROOT = REPO_ROOT / "cvae_support_routing"
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cvae_support_routing.scripts.preflight.preflight_midogpp_scanner import (
    build_confounding_rows,
    build_fold_rows,
)
from src.config.load_config import load_config
from src.config.schema import validate_config
from src.data.datasets.midogpp import MidogPPRecord, prepare_midogpp_records


def _write_midogpp_fixture(root: Path, *, missing_group: bool = False) -> Path:
    images = root / "images"
    images.mkdir(parents=True)
    metadata_path = root / "metadata.csv"
    fieldnames = [
        "image_path",
        "case_id",
        "scanner_model",
        "scanner_vendor",
        "lab_or_origin",
        "tumor_type",
        "species",
        "resolution",
        "label",
    ]
    rows = []
    for scanner_idx, scanner in enumerate(["Hamamatsu XR", "Leica CS2", "3DHistech Pannoramic"]):
        for group_idx in range(6):
            filename = f"s{scanner_idx}_g{group_idx}.png"
            (images / filename).touch()
            rows.append(
                {
                    "image_path": f"images/{filename}",
                    "case_id": "" if missing_group and scanner_idx == 0 and group_idx == 0 else f"case_{scanner_idx}_{group_idx}",
                    "scanner_model": scanner,
                    "scanner_vendor": scanner.split()[0],
                    "lab_or_origin": f"lab_{scanner_idx % 2}",
                    "tumor_type": f"tumor_{group_idx % 2}",
                    "species": "human",
                    "resolution": "0.25",
                    "label": str(group_idx % 2),
                }
            )
    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return metadata_path


def test_midogpp_adapter_maps_scanner_models_and_preserves_metadata(tmp_path: Path) -> None:
    metadata_path = _write_midogpp_fixture(tmp_path)
    records, report = prepare_midogpp_records(
        root=tmp_path,
        extensions=[".png"],
        split={"train": 0.50, "val": 0.17, "test": 0.33},
        cap_per_domain=None,
        seed=42,
        require_patient_ids=True,
        metadata_file=str(metadata_path),
        midogpp_domain_axis="scanner_model",
        split_domain_caps={"train": 3, "val": 1, "test": 2},
        configured_domains=[0, 1, 2],
    )

    assert sorted({int(rec.magnification) for rec in records}) == [0, 1, 2]
    assert all(isinstance(rec, MidogPPRecord) for rec in records)
    assert all(rec.scanner_model for rec in records)
    assert all(rec.lab_or_origin for rec in records)
    assert all(rec.tumor_type for rec in records)
    preflight = report["midogpp_preflight"]
    assert preflight["domain_axis_used"] == "scanner_model"
    assert preflight["n_domains"] == 3
    assert preflight["domain_id_source"] == "scanner_model"
    assert set(preflight["domain_id_to_raw_scanner_label"]) == {"0", "1", "2"}
    assert report["patient_overlap"] == {"train_val": [], "train_test": [], "val_test": []}


def test_midogpp_adapter_merges_json_database_level_scanner_metadata(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    images = []
    databases = []
    scanners = ["Hamamatsu XR", "Leica CS2", "3DHistech Pannoramic"]
    for scanner_idx, scanner in enumerate(scanners):
        databases.append(
            {
                "id": f"db_{scanner_idx}",
                "scanner_model": scanner,
                "scanner_vendor": scanner.split()[0],
                "lab_or_origin": f"lab_{scanner_idx}",
                "tumor_type": f"tumor_{scanner_idx}",
                "species": "human",
                "resolution": "0.25",
            }
        )
        for group_idx in range(6):
            filename = f"json_s{scanner_idx}_g{group_idx}.png"
            (images_dir / filename).touch()
            images.append(
                {
                    "id": f"img_{scanner_idx}_{group_idx}",
                    "file_name": f"images/{filename}",
                    "database_id": f"db_{scanner_idx}",
                    "case_id": f"json_case_{scanner_idx}_{group_idx}",
                }
            )
    metadata_path = tmp_path / "MIDOG++.json"
    metadata_path.write_text(json.dumps({"databases": databases, "images": images}), encoding="utf-8")

    records, report = prepare_midogpp_records(
        root=tmp_path,
        extensions=[".png"],
        split={"train": 0.50, "val": 0.17, "test": 0.33},
        cap_per_domain=None,
        seed=42,
        require_patient_ids=True,
        metadata_file=str(metadata_path),
        midogpp_domain_axis="scanner_model",
        split_domain_caps={"train": 3, "val": 1, "test": 2},
        configured_domains=[0, 1, 2],
    )

    assert len(records) == 18
    assert sorted({rec.scanner_model for rec in records}) == sorted(scanners)
    assert all(rec.lab_or_origin for rec in records)
    assert report["midogpp_preflight"]["n_domains"] == 3


def test_midogpp_adapter_requires_group_ids(tmp_path: Path) -> None:
    metadata_path = _write_midogpp_fixture(tmp_path, missing_group=True)
    with pytest.raises(ValueError, match="without group IDs"):
        prepare_midogpp_records(
            root=tmp_path,
            extensions=[".png"],
            split={"train": 0.50, "val": 0.17, "test": 0.33},
            cap_per_domain=None,
            seed=42,
            require_patient_ids=True,
            metadata_file=str(metadata_path),
            midogpp_domain_axis="scanner_model",
            split_domain_caps={"train": 3, "val": 1, "test": 2},
            configured_domains=[0, 1, 2],
        )


def test_midogpp_preflight_classifies_confounding_and_group_feasibility() -> None:
    records = []
    for domain in [0, 1, 2]:
        for idx in range(64):
            records.append(
                MidogPPRecord(
                    sample_id=f"{domain}_{idx}",
                    image_path=f"/tmp/{domain}_{idx}.png",
                    label=idx % 2,
                    label_name=str(idx % 2),
                    magnification=domain,
                    domain_name=f"scanner_{domain}",
                    patient_id=f"case_{domain}_{idx}",
                    split="test",
                    scanner_model=f"scanner_{domain}",
                    scanner_vendor=f"vendor_{domain}",
                    scanner_family=f"vendor_{domain}",
                    lab_or_origin=f"lab_{idx % 4}",
                    tumor_type=f"tumor_{idx % 4}",
                    species=f"species_{idx % 4}",
                    resolution="0.25",
                    resolution_bin="<=0.30",
                )
            )

    confounding = build_confounding_rows(records)
    folds = build_fold_rows(records, [4, 8, 16, 32])

    assert confounding
    assert {row["scanner_domain"] for row in confounding} == {0, 1, 2}
    assert all(row["target_group_count"] == 64 for row in folds)
    assert all(row["group_feasibility"] == "preferred" for row in folds)
    assert all(row["fold_classification"] == "clean-ish scanner fold" for row in folds)
    assert all(float(row["tumor_effective_count"]) > 1.0 for row in folds)


def test_midogpp_support_config_is_protocol_locked() -> None:
    path = (
        SUPPORT_ROOT
        / "configs"
        / "experiments"
        / "midogpp"
        / "midogpp_scanner_support_estimated_utility_routing_v1.yaml"
    )
    cfg = load_config(path)
    validate_config(cfg)
    assert cfg["data"]["dataset_type"] == "midogpp"
    assert cfg["data"]["midogpp_domain_axis"] == "scanner_model"
    assert cfg["data"]["dataset_domain_semantics"] == "midogpp_scanner"
    support_cfg = cfg["learned_utility"]["support_response_routing"]
    assert support_cfg["support_sizes"] == [4, 8, 16, 32]
    assert support_cfg["support_seeds"] == [17, 23, 31]
    assert support_cfg["sampling_policies"] == ["random"]
    assert support_cfg["support_utility"]["require_unlabeled_support"] is True
