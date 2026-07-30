"""No-training audit of nonlinear rescues, regressions, and the hard core."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageFilter


def build_error_audit(
    primary: Sequence[Mapping[str, str]],
    baseline: Sequence[Mapping[str, str]],
    manifest_by_id: Mapping[str, Mapping[str, str]],
    c_predictions: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    baseline_by_id = {row["sample_id"]: row for row in baseline}
    rows = []
    for nonlinear in primary:
        sample_id = nonlinear["sample_id"]
        linear = baseline_by_id[sample_id]
        truth = int(nonlinear["y_true"])
        linear_correct = int(linear["y_pred"]) == truth
        nonlinear_correct = int(nonlinear["y_pred"]) == truth
        category = {
            (False, True): "nonlinear_rescue",
            (True, False): "nonlinear_regression",
            (False, False): "shared_hard_core",
            (True, True): "shared_correct",
        }[(linear_correct, nonlinear_correct)]
        if category == "shared_correct":
            continue
        meta = manifest_by_id[sample_id]
        bbox_w = float(meta["bbox_w"])
        bbox_h = float(meta["bbox_h"])
        offset_x = float(meta["bbox_x"]) + bbox_w / 2.0 - float(meta["patch_center_x"])
        offset_y = float(meta["bbox_y"]) + bbox_h / 2.0 - float(meta["patch_center_y"])
        quality = _quality(Path(meta["image_path"]))
        c_row = c_predictions.get(sample_id)
        rows.append(
            {
                "schema_version": "midogpp_uniform_bplus_error_audit_v1",
                "sample_id": sample_id,
                "case_id": nonlinear["case_id"],
                "center": nonlinear["center"],
                "label": truth,
                "category": category,
                "linear_pred": int(linear["y_pred"]),
                "linear_prob": float(linear["prob_pos"]),
                "linear_margin": abs(float(linear["prob_pos"]) - 0.5),
                "nonlinear_pred": int(nonlinear["y_pred"]),
                "nonlinear_prob": float(nonlinear["prob_pos"]),
                "nonlinear_margin": abs(float(nonlinear["prob_pos"]) - 0.5),
                "nonlinear_confident": abs(float(nonlinear["prob_pos"]) - 0.5) >= 0.4,
                "scanner_model": meta["scanner_model"],
                "lab_or_origin": meta["lab_or_origin"],
                "tumor_type": meta["tumor_type"],
                "negative_match_scope": meta["negative_match_scope"],
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "bbox_area": bbox_w * bbox_h,
                "bbox_aspect_ratio": bbox_w / max(bbox_h, 1.0),
                "annotation_offset_x": offset_x,
                "annotation_offset_y": offset_y,
                "annotation_offset_radius": float(np.hypot(offset_x, offset_y)),
                **quality,
                "c_prediction_available": c_row is not None,
                "c_correct": (
                    int(c_row["prediction"]) == truth if c_row is not None else ""
                ),
            }
        )
    return rows, _summaries(rows)


def _quality(path: Path) -> dict[str, float]:
    try:
        with Image.open(path) as image:
            gray = np.asarray(image.convert("L"), dtype=np.float32)
            edges = np.asarray(
                image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32
            )
    except (FileNotFoundError, OSError):
        return {
            "quality_brightness": float("nan"),
            "quality_contrast": float("nan"),
            "quality_edge_energy": float("nan"),
        }
    return {
        "quality_brightness": float(np.mean(gray)),
        "quality_contrast": float(np.std(gray)),
        "quality_edge_energy": float(np.mean(np.abs(edges))),
    }


def _summaries(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    dimensions = ("center", "label", "scanner_model", "tumor_type")
    for dimension in dimensions:
        values = sorted({str(row[dimension]) for row in rows})
        for value in values:
            selected = [row for row in rows if str(row[dimension]) == value]
            counts = defaultdict(int)
            for row in selected:
                counts[str(row["category"])] += 1
            output.append(
                {
                    "schema_version": "midogpp_uniform_bplus_error_group_v1",
                    "dimension": dimension,
                    "value": value,
                    "n_error_exchange": len(selected),
                    "nonlinear_rescue": counts["nonlinear_rescue"],
                    "nonlinear_regression": counts["nonlinear_regression"],
                    "shared_hard_core": counts["shared_hard_core"],
                    "net_rescue": counts["nonlinear_rescue"]
                    - counts["nonlinear_regression"],
                    "mean_nonlinear_margin": float(
                        np.mean([float(row["nonlinear_margin"]) for row in selected])
                    ),
                    "confident_regressions": sum(
                        row["category"] == "nonlinear_regression"
                        and bool(row["nonlinear_confident"])
                        for row in selected
                    ),
                }
            )
    return output
