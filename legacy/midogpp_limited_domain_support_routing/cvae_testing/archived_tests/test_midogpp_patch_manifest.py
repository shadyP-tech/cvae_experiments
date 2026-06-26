from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_midogpp_patch_manifest import build_patch_manifest


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_build_midogpp_patch_manifest_preserves_scanner_groups(tmp_path: Path) -> None:
    root = tmp_path / "MIDOGpp"
    images = root / "images"
    images.mkdir(parents=True)

    metadata_rows = []
    annotation_images = []
    annotations = []
    scanners = ["3D Histech", "Hamamatsu XR", "Aperio CS2"]
    ann_id = 1
    for idx, scanner in enumerate(scanners, start=1):
        filename = f"{idx:03d}.tiff"
        arr = np.full((128, 128, 3), 40 * idx, dtype=np.uint8)
        Image.fromarray(arr).save(images / filename)
        metadata_rows.append(
            {
                "image_path": f"images/{filename}",
                "case_id": f"{idx:03d}",
                "scanner_model": scanner,
                "scanner_vendor": "",
                "scanner_family": "",
                "lab_or_origin": f"lab_{idx}",
                "tumor_type": f"tumor_{idx}",
                "species": "Human",
                "split": "",
                "resolution": "unknown",
                "label": "0",
            }
        )
        annotation_images.append({"id": idx, "file_name": filename})
        for category_id, x in [(1, 32), (2, 96)]:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": category_id,
                    "bbox": [x - 5, 64 - 5, x + 5, 64 + 5],
                }
            )
            ann_id += 1

    metadata = root / "midogpp_scanner_metadata_full_resplit.csv"
    _write_csv(metadata, metadata_rows)
    ann_path = root / "databases" / "MIDOG++.json"
    ann_path.parent.mkdir()
    ann_path.write_text(
        json.dumps(
            {
                "images": annotation_images,
                "annotations": annotations,
                "categories": [
                    {"id": 1, "name": "mitotic figure"},
                    {"id": 2, "name": "not mitotic figure"},
                ],
            }
        ),
        encoding="utf-8",
    )

    out_meta = root / "midogpp_scanner_patch_metadata.csv"
    report = build_patch_manifest(
        root=root,
        metadata_path=metadata,
        annotations_path=ann_path,
        out_dir=root / "patches_64",
        output_metadata=out_meta,
        report_path=tmp_path / "patch_report.json",
        patch_size=64,
        patches_per_slide=2,
        seed=42,
        overwrite=True,
    )

    rows = list(csv.DictReader(out_meta.open(newline="", encoding="utf-8")))
    assert len(rows) == 6
    assert sorted({row["scanner_model"] for row in rows}) == sorted(scanners)
    assert sorted({row["case_id"] for row in rows}) == ["001", "002", "003"]
    assert {row["label"] for row in rows} == {"0", "1"}
    assert all((root / row["image_path"]).exists() for row in rows)
    assert report["group_counts_by_scanner"] == {scanner: 1 for scanner in scanners}
    assert report["n_skipped_slides"] == 0
