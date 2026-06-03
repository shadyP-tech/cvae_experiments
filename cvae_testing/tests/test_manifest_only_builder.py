from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import manifest_only


@dataclass(frozen=True)
class DummyRecord:
    sample_id: str
    image_path: str
    label: int
    label_name: str
    magnification: int
    domain_name: str
    patient_id: str
    split: str


def _cfg(output_root: Path) -> dict:
    return {
        "seed": 0,
        "experiment": {
            "dataset_name": "camelyon17",
            "name": "camelyon17_support_estimated_utility_routing_v2",
            "mode": "learned_utility_routing",
        },
        "data": {
            "root": "unused",
            "image_extensions": [".png"],
            "magnifications": [0, 1, 2, 3, 4],
            "split": {"train": 0.7, "val": 0.15, "test": 0.15},
        },
        "output": {"root": str(output_root)},
    }


def _records(*, omit_domain3_test: bool = False) -> list[DummyRecord]:
    rows: list[DummyRecord] = []
    for domain in [0, 1, 2, 3, 4]:
        for split in ["train", "val", "test"]:
            if omit_domain3_test and domain == 3 and split == "test":
                continue
            rows.append(
                DummyRecord(
                    sample_id=f"seed_sample_c{domain}_{split}",
                    image_path=f"/tmp/camelyon17/c{domain}_{split}.png",
                    label=int(split == "test"),
                    label_name="tumor" if split == "test" else "normal",
                    magnification=domain,
                    domain_name=f"center_{domain}",
                    patient_id=f"patient_{domain}_{split}",
                    split=split,
                )
            )
    return rows


def _write_manifest(records: list[DummyRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "image_path", "label", "label_name", "magnification", "domain", "patient_id", "split"]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "sample_id": rec.sample_id,
                    "image_path": rec.image_path,
                    "label": rec.label,
                    "label_name": rec.label_name,
                    "magnification": rec.magnification,
                    "domain": rec.domain_name,
                    "patient_id": rec.patient_id,
                    "split": rec.split,
                }
            )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_manifest_only_builder_writes_split_artifacts_without_training(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path / "outputs")
    monkeypatch.setattr(manifest_only, "load_config", lambda path: dict(cfg))
    monkeypatch.setattr(
        manifest_only,
        "prepare_dataset_records",
        lambda project_root, config: (_records(), {"patient_overlap": {}, "duplicate_paths": []}),
    )
    monkeypatch.setattr(manifest_only, "write_manifest", _write_manifest)

    result = manifest_only.materialize_manifest_only_run(
        project_root=tmp_path / "cvae_testing",
        config_path=tmp_path / "config.yaml",
        seed=50,
        run_id="support_utility_v2_seed50",
    )

    assert result.samples_manifest.exists()
    assert result.split_manifest.exists()
    assert result.leakage_report.exists()
    assert result.manifest_only_report.exists()
    assert result.config_resolved.exists()
    assert not any((result.run_root / "checkpoints").glob("*"))
    assert not any((result.run_root / "embeddings").glob("*"))

    rows = _read_csv(result.samples_manifest)
    assert len(rows) == 15
    assert {"sample_id", "image_path", "label", "magnification", "domain", "split"}.issubset(rows[0])

    report = json.loads(result.manifest_only_report.read_text(encoding="utf-8"))
    assert report["manifest_only"] is True
    assert report["training_executed"] is False
    assert report["embedding_extraction_executed"] is False
    assert report["routing_or_selection_executed"] is False
    assert report["target_labels_used_for_selection"] is False
    assert report["split_counts"] == {"train": 5, "val": 5, "test": 5}


def test_manifest_only_builder_refuses_existing_outputs_without_overwrite(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path / "outputs")
    monkeypatch.setattr(manifest_only, "load_config", lambda path: dict(cfg))
    monkeypatch.setattr(
        manifest_only,
        "prepare_dataset_records",
        lambda project_root, config: (_records(), {"patient_overlap": {}, "duplicate_paths": []}),
    )
    monkeypatch.setattr(manifest_only, "write_manifest", _write_manifest)

    kwargs = {
        "project_root": tmp_path / "cvae_testing",
        "config_path": tmp_path / "config.yaml",
        "seed": 50,
        "run_id": "support_utility_v2_seed50",
    }
    manifest_only.materialize_manifest_only_run(**kwargs)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        manifest_only.materialize_manifest_only_run(**kwargs)

    rerun = manifest_only.materialize_manifest_only_run(**kwargs, overwrite=True)
    assert rerun.samples_manifest.exists()


def test_manifest_only_builder_preserves_domain_split_coverage_guard(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path / "outputs")
    monkeypatch.setattr(manifest_only, "load_config", lambda path: dict(cfg))
    monkeypatch.setattr(
        manifest_only,
        "prepare_dataset_records",
        lambda project_root, config: (_records(omit_domain3_test=True), {"patient_overlap": {}, "duplicate_paths": []}),
    )
    monkeypatch.setattr(manifest_only, "write_manifest", _write_manifest)

    with pytest.raises(RuntimeError, match="Missing test domains: \\[3\\]"):
        manifest_only.materialize_manifest_only_run(
            project_root=tmp_path / "cvae_testing",
            config_path=tmp_path / "config.yaml",
            seed=50,
            run_id="support_utility_v2_seed50",
        )
