"""Immutable clipped-bbox, in-bounds annotation-local contract v3."""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.data.features.virchow2_tokens import (
    normalized_position_to_window_start,
)

from .config_v3 import (
    CONTRACT_SCHEMA_VERSION,
    PROFILE_ID,
    PhysicalMultiscaleV3BuildConfig,
)
from .contract_inputs import (
    assert_raw_source_identity,
    load_contract_inputs,
    relative_to_repo,
    sha256_file,
)
from .geometry import (
    clipped_annotation_bbox_anchor,
    physical_crop_geometry_in_bounds,
)
from .resolution_audit import MppAudit, audit_tiff_mpp


V3_CONTRACT_FILES = (
    "config.resolved.yaml",
    "physical_multiscale_contract.json",
    "physical_multiscale_manifest.csv",
    "resolution_audit.csv",
    "source_geometry_audit.json",
    "row_alignment_report.json",
    "leakage_report.json",
)


def audit_physical_multiscale_sources_v3(
    config: PhysicalMultiscaleV3BuildConfig,
    *,
    report_path: str | Path | None = None,
) -> Mapping[str, object]:
    """Audit every required TIFF and v3 geometry without publishing a contract."""

    collected = _collect(config, include_raw_hashes=False)
    report = _audit_report(config, collected)
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, report)
    if report["status"] != "PASS":
        raise ValueError(
            "Physical multiscale clipped-bbox v3 source audit failed: "
            + json.dumps(report, sort_keys=True)
        )
    return report


def build_physical_multiscale_contract_v3(
    config: PhysicalMultiscaleV3BuildConfig,
) -> Path:
    """Build, independently validate, and atomically publish the v3 contract."""

    from midogpp_thesis.common.staged_directory import staged_directory

    from .contract_validation_v3 import validate_contract_bundle_v3

    with staged_directory(config.contract_root) as stage:
        _build_contract_v3_in_place(config, stage)
        validate_contract_bundle_v3(
            stage,
            verify_raw_files=True,
            expected_config=config,
        )
    return config.contract_root


def _build_contract_v3_in_place(
    config: PhysicalMultiscaleV3BuildConfig,
    root: Path,
) -> None:
    if not root.is_dir() or any(root.iterdir()):
        raise FileExistsError(
            f"Physical multiscale v3 staging root is not empty: {root}"
        )
    collected = _collect(config, include_raw_hashes=True)
    audit_report = _audit_report(config, collected)
    if audit_report["status"] != "PASS":
        raise ValueError(
            "Physical multiscale clipped-bbox v3 contract preflight failed: "
            + json.dumps(audit_report, sort_keys=True)
        )
    rows = list(collected["rows"])
    slide_audits = collected["slide_audits"]
    assert isinstance(slide_audits, Mapping)
    sample_ids = [str(row["sample_id"]) for row in rows]
    identity_hashes = _identity_hashes(rows)
    geometry_policy = geometry_policy_v3(config)
    contract_hash = physical_contract_hash_v3(
        canonical_cache_sha256=str(collected["canonical_cache_sha256"]),
        manifest_rows=rows,
        fov_um=config.fov_um,
        geometry_policy=geometry_policy,
        identity_hashes=identity_hashes,
    )
    _write_csv(root / "physical_multiscale_manifest.csv", rows)
    resolution_rows = [
        {
            "raw_tiff_path": relative_to_repo(path, config.repo_root),
            "raw_tiff_sha256": entry["raw_hash"],
            **asdict(entry["audit"]),
            "status": "PASS",
        }
        for path, entry in sorted(slide_audits.items(), key=lambda item: str(item[0]))
    ]
    _write_csv(root / "resolution_audit.csv", resolution_rows)
    _write_json(root / "source_geometry_audit.json", audit_report)
    _write_json(
        root / "physical_multiscale_contract.json",
        {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "status": "PASS",
            "profile_id": PROFILE_ID,
            "contract_hash": contract_hash,
            "artifact_dataset": "MIDOG++",
            "claim_dataset": "MIDOG++",
            "split": "train",
            "eligible_centers": list(config.eligible_centers),
            "excluded_centers": ["4"],
            "canonical_cache_sha256": collected["canonical_cache_sha256"],
            "row_count": len(rows),
            "slide_count": len(slide_audits),
            "fov_um": list(config.fov_um),
            "geometry_policy": geometry_policy,
            "identity_hashes": identity_hashes,
            "target_labels_used_for_extraction": False,
            "claim_firewall": claim_firewall_v3(),
        },
    )
    _write_json(
        root / "row_alignment_report.json",
        {
            "schema_version": "midogpp_physical_multiscale_row_alignment_v3",
            "status": "PASS",
            "profile_id": PROFILE_ID,
            "row_count": len(rows),
            "sample_id_order_hash": stable_hash(sample_ids),
            "identity_hashes": identity_hashes,
            "center_4_present": False,
            "canonical_order_exact": True,
            "geometry_driven_row_exclusion": False,
        },
    )
    _write_json(
        root / "leakage_report.json",
        {
            "schema_version": "midogpp_physical_multiscale_leakage_v3",
            "status": "PASS",
            "split": "train",
            "target_labels_used_for_geometry": False,
            "target_metrics_used_for_geometry": False,
            "center_identity_used_for_geometry": False,
            "post_label_row_exclusion": False,
            "geometry_driven_row_exclusion": False,
            "label_stratified_geometry_summary_is_audit_only": True,
            "mechanical_pass_criteria_are_label_blind": True,
            "claim_firewall": claim_firewall_v3(),
        },
    )
    (root / "config.resolved.yaml").write_text(
        yaml.safe_dump(_resolved_config_payload(config), sort_keys=False),
        encoding="utf-8",
    )


def geometry_policy_v3(
    config: PhysicalMultiscaleV3BuildConfig,
) -> dict[str, object]:
    return {
        "crop_policy": "deterministic_axiswise_in_bounds_translation",
        "annotation_anchor": "clipped_axis_aligned_bbox_continuous_centroid",
        "annotation_anchor_policy_id": config.annotation_anchor_policy_id,
        "bbox_convention": "continuous_half_open_xyxy",
        "image_intersection": (
            "[max(0,x0),min(W,x1)) x [max(0,y0),min(H,y1))"
        ),
        "minimum_clipped_bbox_area_fraction": (
            config.minimum_clipped_bbox_area_fraction
        ),
        "anchor_delta_definition": (
            "clipped_bbox_centroid_minus_original_bbox_centroid"
        ),
        "coordinate_origin": "level_0_top_left_pixel_edge",
        "pixel_rounding": "round_half_up",
        "out_of_bounds_padding_used": False,
        "realized_padding_pixel_count": 0,
        "crop_size_changed": False,
        "resize_interpolation": config.resize_interpolation,
        "resize_antialias": config.resize_antialias,
        "output_size_px": config.output_size_px,
        "token_window_start": "axis_start=clamp(floor(16*p_axis-2),0,12)",
        "token_window_shape": [4, 4],
        "geometry_uses_labels": False,
        "geometry_uses_center_identity": False,
    }


def claim_firewall_v3() -> dict[str, bool]:
    return {
        "feature_extraction_stochastic": False,
        "geometry_uses_labels": False,
        "geometry_uses_center_identity": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "uses_likelihood": False,
        "uses_nelbo": False,
        "uses_latent_prior": False,
        "uses_posterior": False,
        "uses_mixture_model": False,
        "uses_experts": False,
        "performs_expert_aggregation": False,
        "uses_generative_sampling": False,
    }


def physical_contract_hash_v3(
    *,
    canonical_cache_sha256: str,
    manifest_rows: Sequence[Mapping[str, object]],
    fov_um: Sequence[float],
    geometry_policy: Mapping[str, object],
    identity_hashes: Mapping[str, object],
) -> str:
    """Bind the ordered cohort, raw/clipped bboxes, crops, and token starts."""

    normalized_rows = [
        {str(key): "" if value is None else str(value) for key, value in row.items()}
        for row in manifest_rows
    ]
    return stable_hash(
        {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "profile_id": PROFILE_ID,
            "canonical_cache_sha256": str(canonical_cache_sha256),
            "manifest_rows": normalized_rows,
            "fov_um": [float(value) for value in fov_um],
            "geometry_policy": dict(geometry_policy),
            "identity_hashes": dict(identity_hashes),
        }
    )


def _collect(
    config: PhysicalMultiscaleV3BuildConfig,
    *,
    include_raw_hashes: bool,
) -> dict[str, object]:
    inputs = load_contract_inputs(config)
    if len(inputs.selected_metadata) != config.expected_row_count:
        raise ValueError(
            f"Physical multiscale v3 row count drift: "
            f"expected={config.expected_row_count}, actual={len(inputs.selected_metadata)}"
        )
    slide_audits: dict[Path, dict[str, Any]] = {}
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for row_index, metadata in enumerate(inputs.selected_metadata):
        sample_id = str(metadata["sample_id"])
        manifest = inputs.manifest_by_id[sample_id]
        case_id = str(metadata.get("case_id", manifest.get("case_id", "")))
        source = inputs.source_by_case.get(case_id)
        if source is None:
            raise ValueError(f"No unique raw TIFF mapping for case_id={case_id!r}")
        assert_raw_source_identity(manifest, source)
        raw_path = Path(str(source["raw_path"]))
        if raw_path not in slide_audits:
            audit = audit_tiff_mpp(
                raw_path,
                mpp_min=config.mpp_min,
                mpp_max=config.mpp_max,
                anisotropy_relative_max=config.anisotropy_relative_max,
                dual_source_relative_max=config.dual_source_relative_max,
            )
            slide_audits[raw_path] = {
                "audit": audit,
                "raw_hash": sha256_file(raw_path) if include_raw_hashes else "",
            }
        entry = slide_audits[raw_path]
        audit = entry["audit"]
        assert isinstance(audit, MppAudit)
        try:
            anchor = clipped_annotation_bbox_anchor(
                bbox_x=float(manifest["bbox_x"]),
                bbox_y=float(manifest["bbox_y"]),
                bbox_w=float(manifest["bbox_w"]),
                bbox_h=float(manifest["bbox_h"]),
                image_width=audit.width,
                image_height=audit.height,
                minimum_clipped_area_fraction=(
                    config.minimum_clipped_bbox_area_fraction
                ),
            )
            scale_geometry: dict[str, object] = {}
            for fov in config.fov_um:
                geometry = physical_crop_geometry_in_bounds(
                    anchor_x=anchor.anchor_x,
                    anchor_y=anchor.anchor_y,
                    fov_um=fov,
                    mpp_x=audit.mpp_x,
                    mpp_y=audit.mpp_y,
                    image_width=audit.width,
                    image_height=audit.height,
                )
                token_row, token_col = normalized_position_to_window_start(
                    x=geometry.p_x,
                    y=geometry.p_y,
                )
                scale_geometry[_fov_key(fov)] = {
                    **asdict(geometry),
                    "token_start_row": token_row,
                    "token_start_col": token_col,
                }
        except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
            if len(failures) < 20:
                failures.append(
                    {
                        "sample_id": sample_id,
                        "case_id": case_id,
                        "error": str(exc),
                    }
                )
            continue
        rows.append(
            {
                "row_index": row_index,
                "sample_id": sample_id,
                "case_id": case_id,
                "annotation_id": str(manifest.get("annotation_id", "")),
                "label": int(float(str(manifest["label"]))),
                "split": "train",
                "center": str(metadata.get("center", manifest.get("center", ""))),
                "raw_tiff_path": relative_to_repo(raw_path, config.repo_root),
                "raw_tiff_sha256": entry["raw_hash"],
                "level": 0,
                "orientation": audit.orientation,
                "image_width": audit.width,
                "image_height": audit.height,
                "bbox_x": float(manifest["bbox_x"]),
                "bbox_y": float(manifest["bbox_y"]),
                "bbox_w": float(manifest["bbox_w"]),
                "bbox_h": float(manifest["bbox_h"]),
                **asdict(anchor),
                "mpp_x": audit.mpp_x,
                "mpp_y": audit.mpp_y,
                "mpp_source": audit.source,
                "scale_geometry_json": json.dumps(
                    scale_geometry,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    if not failures:
        if len(slide_audits) != config.expected_slide_count:
            raise ValueError(
                f"Physical multiscale v3 slide count drift: "
                f"expected={config.expected_slide_count}, actual={len(slide_audits)}"
            )
        if [str(row["sample_id"]) for row in rows] != [
            str(row["sample_id"]) for row in inputs.selected_metadata
        ]:
            raise ValueError(
                "Physical multiscale v3 is not bijective with canonical row order."
            )
    return {
        "rows": tuple(rows),
        "slide_audits": slide_audits,
        "canonical_cache_sha256": inputs.canonical.cache_sha256,
        "failures": tuple(failures),
    }


def _audit_report(
    config: PhysicalMultiscaleV3BuildConfig,
    collected: Mapping[str, object],
) -> dict[str, object]:
    rows = list(collected["rows"])  # type: ignore[arg-type]
    slide_audits = collected["slide_audits"]
    failures = list(collected.get("failures", ()))  # type: ignore[arg-type]
    assert isinstance(slide_audits, Mapping)
    clipped_rows = [row for row in rows if bool(row["was_clipped"])]
    original_center_outside = [
        row
        for row in rows
        if not (
            0.0 <= float(row["original_centroid_x"]) < int(row["image_width"])
            and 0.0 <= float(row["original_centroid_y"]) < int(row["image_height"])
        )
    ]
    shifted_samples: set[str] = set()
    counts_by_fov = {f"{fov:g}um": 0 for fov in config.fov_um}
    shifted_by_center = {center: 0 for center in config.eligible_centers}
    shifted_by_label = {"0": 0, "1": 0}
    maximum_shift_px = 0
    maximum_shift_fraction = 0.0
    direction_counts = {"left": 0, "right": 0, "up": 0, "down": 0}
    for row in rows:
        sample_shifted = False
        geometries = json.loads(str(row["scale_geometry_json"]))
        for fov_key, geometry in geometries.items():
            shift_x = int(geometry["shift_x"])
            shift_y = int(geometry["shift_y"])
            side = int(geometry["side_px"])
            if shift_x or shift_y:
                sample_shifted = True
                counts_by_fov[fov_key] += 1
            maximum_shift_px = max(maximum_shift_px, abs(shift_x), abs(shift_y))
            maximum_shift_fraction = max(
                maximum_shift_fraction,
                abs(shift_x) / side,
                abs(shift_y) / side,
            )
            direction_counts["right"] += int(shift_x > 0)
            direction_counts["left"] += int(shift_x < 0)
            direction_counts["down"] += int(shift_y > 0)
            direction_counts["up"] += int(shift_y < 0)
        if sample_shifted:
            shifted_samples.add(str(row["sample_id"]))
            shifted_by_center[str(row["center"])] += 1
            shifted_by_label[str(int(row["label"]))] += 1
    audits = [
        entry["audit"] for entry in slide_audits.values()  # type: ignore[union-attr]
    ]
    mpp_values = [
        value for audit in audits for value in (audit.mpp_x, audit.mpp_y)
    ]
    fractions = [float(row["clipped_area_fraction"]) for row in clipped_rows]
    delta_x = [float(row["anchor_delta_x"]) for row in clipped_rows]
    delta_y = [float(row["anchor_delta_y"]) for row in clipped_rows]
    mechanical_pass = (
        not failures
        and len(rows) == config.expected_row_count
        and len(slide_audits) == config.expected_slide_count
        and len(clipped_rows) == config.expected_clipped_bbox_count
    )
    return {
        "schema_version": "midogpp_physical_multiscale_source_audit_v3",
        "status": "PASS" if mechanical_pass else "FAIL",
        "profile_id": PROFILE_ID,
        "row_count": len(rows),
        "expected_row_count": config.expected_row_count,
        "slide_count": len(slide_audits),
        "expected_slide_count": config.expected_slide_count,
        "clipped_bbox_count": len(clipped_rows),
        "expected_clipped_bbox_count": config.expected_clipped_bbox_count,
        "clipped_bbox_counts_by_center": _counts(clipped_rows, "center"),
        "clipped_bbox_counts_by_label": _counts(clipped_rows, "label"),
        "clipped_bbox_area_fraction_summary": _distribution(fractions),
        "anchor_delta_x_summary_px": _distribution(delta_x),
        "anchor_delta_y_summary_px": _distribution(delta_y),
        "maximum_absolute_anchor_delta_px": max(
            [abs(value) for value in (*delta_x, *delta_y)],
            default=0.0,
        ),
        "original_bbox_centroid_out_of_bounds_count": len(
            original_center_outside
        ),
        "original_bbox_centroid_out_of_bounds_examples": [
            {
                "sample_id": row["sample_id"],
                "original_centroid_x": row["original_centroid_x"],
                "original_centroid_y": row["original_centroid_y"],
                "anchor_x": row["anchor_x"],
                "anchor_y": row["anchor_y"],
                "clipped_area_fraction": row["clipped_area_fraction"],
            }
            for row in original_center_outside[:20]
        ],
        "minimum_required_clipped_bbox_area_fraction": (
            config.minimum_clipped_bbox_area_fraction
        ),
        "minimum_mpp": min(mpp_values) if mpp_values else None,
        "maximum_mpp": max(mpp_values) if mpp_values else None,
        "all_orientations_top_left": all(audit.orientation == 1 for audit in audits),
        "all_required_tiff_headers_validated": not failures,
        "out_of_bounds_pixels_synthesized": False,
        "realized_padding_pixel_count": 0,
        "shifted_sample_count": len(shifted_samples),
        "shifted_sample_fraction": (
            len(shifted_samples) / len(rows) if rows else None
        ),
        "shifted_crop_counts_by_fov": counts_by_fov,
        "shifted_sample_counts_by_center": shifted_by_center,
        "shifted_sample_counts_by_label": shifted_by_label,
        "shift_direction_counts": direction_counts,
        "maximum_axis_shift_px": maximum_shift_px,
        "maximum_axis_shift_fraction": maximum_shift_fraction,
        "geometry_driven_row_exclusion": False,
        "mechanical_pass_criteria_are_label_blind": True,
        "center_and_label_stratified_counts_are_audit_only": True,
        "failure_examples": failures,
    }


def _identity_hashes(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    return {
        "sample_id_order": stable_hash([row["sample_id"] for row in rows]),
        "case_id_order": stable_hash([row["case_id"] for row in rows]),
        "slide_id_order": stable_hash([row["raw_tiff_path"] for row in rows]),
        "center_order": stable_hash([row["center"] for row in rows]),
        "label_order": stable_hash([row["label"] for row in rows]),
        "partition_identity": stable_hash(
            [
                (
                    row["sample_id"],
                    row["case_id"],
                    row["raw_tiff_path"],
                    row["center"],
                    row["split"],
                )
                for row in rows
            ]
        ),
    }


def _resolved_config_payload(
    config: PhysicalMultiscaleV3BuildConfig,
) -> Mapping[str, object]:
    return {
        "artifact": {"name": PROFILE_ID},
        "source_config_path": str(config.config_path),
        "inputs": {
            "raw_root": str(config.raw_root),
            "raw_metadata_path": str(config.raw_metadata_path),
            "base_manifest_path": str(config.base_manifest_path),
            "canonical_cache_path": str(config.canonical_cache_path),
            "canonical_reference_root": str(config.canonical_reference_root),
        },
        "outputs": {
            "contract_root": str(config.contract_root),
            "cache_bundle_root": str(config.cache_bundle_root),
            "b_cache_root": str(config.b_cache_root),
            "c_cache_root": str(config.c_cache_root),
        },
        "physical_scale": {
            "fov_um": list(config.fov_um),
            "mpp_min": config.mpp_min,
            "mpp_max": config.mpp_max,
            "anisotropy_relative_max": config.anisotropy_relative_max,
            "dual_source_relative_max": config.dual_source_relative_max,
        },
        "geometry": geometry_policy_v3(config),
        "bridge": {
            "minimum_cosine": config.bridge_minimum_cosine,
            "maximum_relative_l2": config.bridge_maximum_relative_l2,
            "minimum_prediction_agreement": config.bridge_minimum_prediction_agreement,
            "maximum_absolute_equal_center_bacc_delta": (
                config.bridge_maximum_equal_center_bacc_delta
            ),
        },
        "model": {
            "model_ref": config.model_ref,
            "model_revision": config.model_revision,
            "expected_model_config_sha256": config.expected_model_config_sha256,
            "expected_checkpoint_file_sha256": (
                config.expected_checkpoint_file_sha256
            ),
            "expected_state_dict_sha256": config.expected_state_dict_sha256,
            "expected_preprocessing_config_hash": (
                config.expected_preprocessing_config_hash
            ),
        },
        "runtime_identity": {
            "timm": config.expected_timm_version,
            "torch": config.expected_torch_version,
            "pillow": config.expected_pillow_version,
            "pyvips": config.expected_pyvips_version,
            "libvips": config.expected_libvips_version,
        },
        "run": {
            "eligible_centers": list(config.eligible_centers),
            "experiment_seed": config.experiment_seed,
            "device": config.device,
            "batch_size": config.batch_size,
            "require_tiled_reader": config.require_tiled_reader,
            "required_slide_reader_backend": (
                config.required_slide_reader_backend
            ),
            "expected_row_count": config.expected_row_count,
            "expected_slide_count": config.expected_slide_count,
            "expected_clipped_bbox_count": config.expected_clipped_bbox_count,
        },
    }


def _counts(
    rows: Sequence[Mapping[str, object]],
    key: str,
) -> dict[str, int]:
    values = sorted({str(row[key]) for row in rows}, key=lambda value: int(value))
    return {value: sum(str(row[key]) == value for row in rows) for value in values}


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p25": None,
            "median": None,
            "p75": None,
            "maximum": None,
            "mean": None,
        }
    ordered = sorted(float(value) for value in values)

    def at(q: float) -> float:
        return ordered[int(math.floor((len(ordered) - 1) * q))]

    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "p25": at(0.25),
        "median": at(0.5),
        "p75": at(0.75),
        "maximum": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _fov_key(value: float) -> str:
    return f"{value:g}um"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    columns = tuple(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
