from __future__ import annotations

import csv
from pathlib import Path

from midogpp_thesis.real_features.sail.features import read_manifest_rows_by_split


def test_read_manifest_resolves_repo_relative_image_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    image_path = tmp_path / "datasets/midogpp/contract/annotation_patch_v1/patches_224/sample.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake image bytes")
    manifest_path = tmp_path / "datasets/midogpp/contract/annotation_patch_v1/manifest.csv"
    _write_manifest(
        manifest_path,
        "datasets/midogpp/contract/annotation_patch_v1/patches_224/sample.jpg",
    )

    rows = read_manifest_rows_by_split(manifest_path, splits=("train",))

    assert rows["train"][0]["image_path"] == str(image_path)


def test_read_manifest_preserves_manifest_relative_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    artifact_root = tmp_path / "artifact"
    image_path = artifact_root / "patches/sample.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake image bytes")
    manifest_path = artifact_root / "manifest.csv"
    _write_manifest(manifest_path, "patches/sample.jpg")

    rows = read_manifest_rows_by_split(manifest_path, splits=("train",))

    assert rows["train"][0]["image_path"] == str(image_path)


def test_read_manifest_relocates_frozen_annotation_patch_v1_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    canonical = (
        tmp_path
        / "datasets/midogpp/contract/annotation_patch_v1/patches_224/sample.jpg"
    )
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"fake image bytes")
    manifest = tmp_path / "datasets/midogpp/contract/annotation_patch_v1/manifest.csv"
    _write_manifest(
        manifest,
        "datasets/midogpp/artifacts/midogpp_annotation_patch_v1/patches_224/sample.jpg",
    )

    rows = read_manifest_rows_by_split(manifest, splits=("train",))

    assert rows["train"][0]["image_path"] == str(canonical)


def _write_manifest(path: Path, image_path: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "image_path", "label", "split", "center", "magnification"])
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample",
                "image_path": image_path,
                "label": "1",
                "split": "train",
                "center": "0",
                "magnification": "0",
            }
        )
