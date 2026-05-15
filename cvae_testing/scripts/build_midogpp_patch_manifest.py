#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image


def _clean(value: object) -> str:
    text = str(value if value is not None else "").strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _safe_stem(value: str) -> str:
    stem = Path(str(value)).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "sample"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {path}")
        return [dict(row) for row in reader]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            key_s = str(key)
            if key_s not in seen:
                seen.add(key_s)
                fieldnames.append(key_s)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_midogpp_annotations(path: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError(f"MIDOG++ annotation JSON must contain list images/annotations: {path}")

    id_to_file: Dict[int, str] = {}
    for image in images:
        if not isinstance(image, Mapping):
            continue
        try:
            image_id = int(image.get("id"))
        except Exception:
            continue
        filename = _clean(image.get("file_name", image.get("filename", "")))
        if filename:
            id_to_file[image_id] = filename

    category_names: Dict[int, str] = {}
    for category in categories if isinstance(categories, list) else []:
        if not isinstance(category, Mapping):
            continue
        try:
            category_id = int(category.get("id"))
        except Exception:
            continue
        category_names[category_id] = _clean(category.get("name", ""))

    by_file: Dict[str, List[Dict[str, Any]]] = {}
    for ann in annotations:
        if not isinstance(ann, Mapping):
            continue
        try:
            image_id = int(ann.get("image_id"))
        except Exception:
            continue
        filename = id_to_file.get(image_id, "")
        if not filename:
            continue
        row = dict(ann)
        try:
            category_id = int(row.get("category_id", 0))
        except Exception:
            category_id = 0
        row["label"] = _label_from_category(category_id, category_names.get(category_id, ""))
        row["label_name"] = category_names.get(category_id, str(category_id)) or str(category_id)
        by_file.setdefault(Path(filename).name, []).append(row)
        by_file.setdefault(Path(filename).stem, []).append(row)
    return by_file, category_names


def _label_from_category(category_id: int, category_name: str) -> int:
    text = str(category_name).strip().lower()
    if text.startswith("not ") or "not mitotic" in text or "non-mitotic" in text:
        return 0
    if "mitotic" in text or "mitosis" in text:
        return 1
    return 1 if int(category_id) == 1 else 0


def _bbox_center(bbox: object) -> Tuple[float, float] | None:
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) < 4:
        return None
    try:
        x0, y0, x2, y2 = [float(v) for v in list(bbox)[:4]]
    except Exception:
        return None
    if x2 > x0 and y2 > y0:
        return (x0 + x2) / 2.0, (y0 + y2) / 2.0
    if x2 > 0 and y2 > 0:
        return x0 + x2 / 2.0, y0 + y2 / 2.0
    return None


def _normalize_image_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[0] in {3, 4} and arr.shape[-1] not in {3, 4}:
            arr = np.moveaxis(arr, 0, -1)
        arr = arr[..., :3]
    else:
        raise ValueError(f"Unsupported image array shape: {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32, copy=False)
        finite = np.isfinite(arr)
        if not finite.any():
            arr = np.zeros(arr.shape, dtype=np.uint8)
        else:
            min_v = float(arr[finite].min())
            max_v = float(arr[finite].max())
            if max_v <= 1.0 and min_v >= 0.0:
                arr = arr * 255.0
            elif max_v > 255.0 or min_v < 0.0:
                arr = 255.0 * (arr - min_v) / max(max_v - min_v, 1.0e-6)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _read_image(path: Path) -> np.ndarray:
    try:
        import tifffile

        return _normalize_image_array(tifffile.imread(path))
    except Exception:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"))


def _crop_centered(arr: np.ndarray, center_x: float, center_y: float, patch_size: int) -> Image.Image:
    h, w = int(arr.shape[0]), int(arr.shape[1])
    size = int(patch_size)
    half = size // 2
    cx = int(round(float(center_x)))
    cy = int(round(float(center_y)))
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + size
    y1 = y0 + size

    src_x0 = max(x0, 0)
    src_y0 = max(y0, 0)
    src_x1 = min(x1, w)
    src_y1 = min(y1, h)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        canvas[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = arr[
            src_y0:src_y1, src_x0:src_x1
        ]
    return Image.fromarray(canvas, mode="RGB")


def _choose_annotations(
    annotations: Sequence[Mapping[str, Any]],
    *,
    patches_per_slide: int,
    rng: random.Random,
) -> List[Mapping[str, Any]]:
    valid = [ann for ann in annotations if _bbox_center(ann.get("bbox")) is not None]
    if not valid:
        return []
    by_label: Dict[int, List[Mapping[str, Any]]] = {}
    for ann in valid:
        by_label.setdefault(int(ann.get("label", 0)), []).append(ann)
    for vals in by_label.values():
        rng.shuffle(vals)

    selected: List[Mapping[str, Any]] = []
    labels = sorted(by_label)
    if len(labels) >= 2:
        per_label = max(1, int(patches_per_slide) // len(labels))
        for label in labels:
            selected.extend(by_label[label][:per_label])

    if len(selected) < int(patches_per_slide):
        selected_ids = {id(ann) for ann in selected}
        remaining = [ann for ann in valid if id(ann) not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: int(patches_per_slide) - len(selected)])

    return selected[: int(patches_per_slide)]


def _scanner_vendor(scanner_model: str, fallback: str = "") -> str:
    fallback = _clean(fallback)
    if fallback:
        return fallback
    lowered = scanner_model.lower()
    if "hamamatsu" in lowered:
        return "hamamatsu"
    if "aperio" in lowered:
        return "aperio"
    if "3d histech" in lowered or "3dhistech" in lowered:
        return "3dhistech"
    return ""


def _metadata_for_patch(
    source: Mapping[str, str],
    *,
    patch_rel_path: str,
    sample_id: str,
    annotation: Mapping[str, Any],
    patch_x: int,
    patch_y: int,
    patch_size: int,
) -> Dict[str, Any]:
    scanner_model = _clean(source.get("scanner_model", source.get("Scanner", "")))
    row: Dict[str, Any] = dict(source)
    row.update(
        {
            "sample_id": sample_id,
            "image_path": patch_rel_path,
            "case_id": _clean(source.get("case_id", source.get("slide_id", Path(source.get("image_path", "")).stem))),
            "patient_id": _clean(source.get("case_id", source.get("patient_id", Path(source.get("image_path", "")).stem))),
            "scanner_model": scanner_model,
            "scanner_vendor": _scanner_vendor(scanner_model, source.get("scanner_vendor", "")),
            "scanner_family": _clean(source.get("scanner_family", "")) or _scanner_vendor(scanner_model, ""),
            "lab_or_origin": _clean(source.get("lab_or_origin", source.get("Origin", ""))),
            "tumor_type": _clean(source.get("tumor_type", source.get("Tumor", ""))),
            "species": _clean(source.get("species", source.get("Species", ""))),
            "resolution": _clean(source.get("resolution", "")) or "unknown",
            "label": int(annotation.get("label", 0)),
            "label_name": _clean(annotation.get("label_name", "")) or str(annotation.get("label", "")),
            "annotation_id": _clean(annotation.get("id", "")),
            "patch_center_x": int(patch_x),
            "patch_center_y": int(patch_y),
            "patch_size": int(patch_size),
        }
    )
    return row


def build_patch_manifest(
    *,
    root: Path,
    metadata_path: Path,
    annotations_path: Path,
    out_dir: Path,
    output_metadata: Path,
    report_path: Path,
    patch_size: int,
    patches_per_slide: int,
    seed: int,
    overwrite: bool,
) -> Dict[str, Any]:
    rows = _read_csv(metadata_path)
    annotations_by_file, _category_names = _load_midogpp_annotations(annotations_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(seed))

    output_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    scanner_groups: Dict[str, set[str]] = {}

    for source in rows:
        rel_image = _clean(source.get("image_path", source.get("file_name", source.get("filename", ""))))
        if not rel_image:
            skipped.append({"image_path": "", "reason": "missing_image_path"})
            continue
        image_path = root / rel_image
        case_id = _clean(source.get("case_id", Path(rel_image).stem)) or Path(rel_image).stem
        scanner_model = _clean(source.get("scanner_model", source.get("Scanner", "")))
        anns = annotations_by_file.get(Path(rel_image).name) or annotations_by_file.get(Path(rel_image).stem) or []
        chosen = _choose_annotations(anns, patches_per_slide=int(patches_per_slide), rng=rng)
        if not chosen:
            skipped.append({"image_path": rel_image, "case_id": case_id, "reason": "no_valid_annotations"})
            continue
        try:
            arr = _read_image(image_path)
        except Exception as exc:
            skipped.append({"image_path": rel_image, "case_id": case_id, "reason": f"image_read_failed:{type(exc).__name__}:{exc}"})
            continue

        scanner_groups.setdefault(scanner_model, set()).add(case_id)
        for patch_idx, ann in enumerate(chosen):
            center = _bbox_center(ann.get("bbox"))
            if center is None:
                continue
            cx, cy = center
            patch_name = f"{_safe_stem(case_id)}_ann{_safe_stem(str(ann.get('id', patch_idx)))}_p{patch_idx:02d}.jpg"
            patch_path = out_dir / patch_name
            patch_rel = str(patch_path.relative_to(root))
            if overwrite or not patch_path.exists():
                patch = _crop_centered(arr, cx, cy, int(patch_size))
                patch.save(patch_path, quality=92)
            sample_id = f"{_safe_stem(case_id)}__ann{_safe_stem(str(ann.get('id', patch_idx)))}__p{patch_idx:02d}"
            output_rows.append(
                _metadata_for_patch(
                    source,
                    patch_rel_path=patch_rel,
                    sample_id=sample_id,
                    annotation=ann,
                    patch_x=int(round(cx)),
                    patch_y=int(round(cy)),
                    patch_size=int(patch_size),
                )
            )

    _write_csv(output_metadata, output_rows)
    skipped_path = report_path.with_name(report_path.stem + "_skipped.csv")
    _write_csv(skipped_path, skipped)

    patch_counts_by_scanner: Dict[str, int] = {}
    group_counts_by_scanner: Dict[str, int] = {}
    for row in output_rows:
        scanner = str(row.get("scanner_model", ""))
        case_id = str(row.get("case_id", ""))
        patch_counts_by_scanner[scanner] = patch_counts_by_scanner.get(scanner, 0) + 1
        scanner_groups.setdefault(scanner, set()).add(case_id)
    for scanner, groups in scanner_groups.items():
        group_counts_by_scanner[scanner] = len(groups)

    report = {
        "metadata_path": str(metadata_path),
        "annotations_path": str(annotations_path),
        "output_metadata": str(output_metadata),
        "patch_dir": str(out_dir),
        "patch_size": int(patch_size),
        "patches_per_slide": int(patches_per_slide),
        "n_source_rows": int(len(rows)),
        "n_patch_rows": int(len(output_rows)),
        "n_skipped_slides": int(len(skipped)),
        "patch_counts_by_scanner": patch_counts_by_scanner,
        "group_counts_by_scanner": group_counts_by_scanner,
        "skipped_csv": str(skipped_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MIDOG++ annotation-centered patch manifest.")
    parser.add_argument("--root", type=Path, default=Path("data/MIDOGpp"))
    parser.add_argument("--metadata", type=Path, default=Path("data/MIDOGpp/midogpp_scanner_metadata_full_resplit.csv"))
    parser.add_argument("--annotations", type=Path, default=Path("data/MIDOGpp/databases/MIDOG++.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/MIDOGpp/patches_224"))
    parser.add_argument("--output-metadata", type=Path, default=Path("data/MIDOGpp/midogpp_scanner_patch_metadata.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/comparison_tables/midogpp_scanner_patch_manifest_report.json"))
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--patches-per-slide", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if int(args.patch_size) <= 0:
        raise ValueError("--patch-size must be positive")
    if int(args.patches_per_slide) <= 0:
        raise ValueError("--patches-per-slide must be positive")

    report = build_patch_manifest(
        root=args.root,
        metadata_path=args.metadata,
        annotations_path=args.annotations,
        out_dir=args.out_dir,
        output_metadata=args.output_metadata,
        report_path=args.report,
        patch_size=int(args.patch_size),
        patches_per_slide=int(args.patches_per_slide),
        seed=int(args.seed),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
