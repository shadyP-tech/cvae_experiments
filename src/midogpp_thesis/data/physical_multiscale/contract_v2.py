"""Immutable in-bounds physical multiscale annotation-local contract v2."""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from midogpp_thesis.common.hashing import stable_hash

from .config_v2 import (
    CONTRACT_SCHEMA_VERSION,
    PROFILE_ID,
    PhysicalMultiscaleV2BuildConfig,
)
from .contract_inputs import (
    assert_raw_source_identity,
    load_contract_inputs,
    relative_to_repo,
    sha256_file,
)
from .geometry import physical_crop_geometry_v2
from .resolution_audit import MppAudit, audit_tiff_mpp


V2_CONTRACT_FILES = (
    "config.resolved.yaml",
    "physical_multiscale_contract.json",
    "physical_multiscale_manifest.csv",
    "resolution_audit.csv",
    "source_geometry_audit.json",
    "row_alignment_report.json",
    "leakage_report.json",
)


def audit_physical_multiscale_sources_v2(
    config: PhysicalMultiscaleV2BuildConfig,
    *,
    report_path: str | Path | None = None,
) -> Mapping[str, object]:
    """Audit every required TIFF and v2 crop without publishing contract bytes."""

    collected = _collect(config, include_raw_hashes=False)
    report = _audit_report(config, collected)
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, report)
    if report["status"] != "PASS":
        raise ValueError(
            "Physical multiscale annotation-local v2 source audit failed: "
            + json.dumps(report, sort_keys=True)
        )
    return report


def build_physical_multiscale_contract_v2(
    config: PhysicalMultiscaleV2BuildConfig,
) -> Path:
    """Build, validate, and atomically publish the v2 physical contract."""

    from midogpp_thesis.common.staged_directory import staged_directory

    from .contract_validation import validate_contract_bundle_v2

    with staged_directory(config.contract_root) as stage:
        _build_contract_v2_in_place(config, stage)
        validate_contract_bundle_v2(
            stage,
            verify_raw_files=True,
            expected_config=config,
        )
    return config.contract_root


def _build_contract_v2_in_place(
    config: PhysicalMultiscaleV2BuildConfig,
    root: Path,
) -> None:
    root.mkdir(parents=True, exist_ok=False)
    collected = _collect(config, include_raw_hashes=True)
    audit_report = _audit_report(config, collected)
    if audit_report["status"] != "PASS":
        raise ValueError(
            "Physical multiscale annotation-local v2 contract preflight failed: "
            + json.dumps(audit_report, sort_keys=True)
        )
    rows = list(collected["rows"])
    slide_audits = collected["slide_audits"]
    assert isinstance(slide_audits, Mapping)
    sample_ids = [str(row["sample_id"]) for row in rows]
    identity_hashes = _identity_hashes(rows)
    geometry_policy = geometry_policy_v2(config)
    contract_hash = physical_contract_hash_v2(
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
            "claim_firewall": claim_firewall_v2(),
        },
    )
    _write_json(
        root / "row_alignment_report.json",
        {
            "schema_version": "midogpp_physical_multiscale_row_alignment_v2",
            "status": "PASS",
            "profile_id": PROFILE_ID,
            "row_count": len(rows),
            "sample_id_order_hash": stable_hash(sample_ids),
            "identity_hashes": identity_hashes,
            "center_4_present": False,
            "canonical_order_exact": True,
        },
    )
    _write_json(
        root / "leakage_report.json",
        {
            "schema_version": "midogpp_physical_multiscale_leakage_v2",
            "status": "PASS",
            "split": "train",
            "target_labels_used_for_geometry": False,
            "target_metrics_used_for_geometry": False,
            "center_identity_used_for_geometry": False,
            "post_label_row_exclusion": False,
            "label_stratified_shift_summary_is_audit_only": True,
            "mechanical_pass_criteria_are_label_blind": True,
            "claim_firewall": claim_firewall_v2(),
        },
    )
    (root / "config.resolved.yaml").write_text(
        yaml.safe_dump(_resolved_config_payload(config), sort_keys=False),
        encoding="utf-8",
    )


def geometry_policy_v2(
    config: PhysicalMultiscaleV2BuildConfig,
) -> dict[str, object]:
    return {
        "crop_policy": "deterministic_axiswise_in_bounds_translation",
        "annotation_anchor": "canonical_xyxy_continuous_center",
        "coordinate_origin": "level_0_top_left_pixel_edge",
        "pixel_rounding": "round_half_up",
        "out_of_bounds_padding_used": False,
        "realized_padding_pixel_count": 0,
        "crop_size_changed": False,
        "resize_interpolation": config.resize_interpolation,
        "resize_antialias": config.resize_antialias,
        "output_size_px": config.output_size_px,
        "geometry_uses_labels": False,
        "geometry_uses_center_identity": False,
    }


def claim_firewall_v2() -> dict[str, bool]:
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
        "uses_generative_sampling": False,
    }


def physical_contract_hash_v2(
    *,
    canonical_cache_sha256: str,
    manifest_rows: Sequence[Mapping[str, object]],
    fov_um: Sequence[float],
    geometry_policy: Mapping[str, object],
    identity_hashes: Mapping[str, object],
) -> str:
    """Bind every cohort, raw-source, requested, and realized geometry scalar."""

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
    config: PhysicalMultiscaleV2BuildConfig,
    *,
    include_raw_hashes: bool,
) -> dict[str, object]:
    inputs = load_contract_inputs(config)
    if len(inputs.selected_metadata) != config.expected_row_count:
        raise ValueError(
            f"Physical multiscale v2 row count drift: "
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
        anchor_x = float(manifest["patch_center_x"])
        anchor_y = float(manifest["patch_center_y"])
        scale_geometry: dict[str, object] = {}
        for fov in config.fov_um:
            try:
                geometry = physical_crop_geometry_v2(
                    anchor_x=anchor_x,
                    anchor_y=anchor_y,
                    fov_um=fov,
                    mpp_x=audit.mpp_x,
                    mpp_y=audit.mpp_y,
                    image_width=audit.width,
                    image_height=audit.height,
                )
            except ValueError as exc:
                if len(failures) < 20:
                    failures.append(
                        {
                            "sample_id": sample_id,
                            "case_id": case_id,
                            "fov_um": fov,
                            "error": str(exc),
                        }
                    )
                continue
            scale_geometry[_fov_key(fov)] = asdict(geometry)
        if len(scale_geometry) != len(config.fov_um):
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
                "anchor_x": anchor_x,
                "anchor_y": anchor_y,
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
    if failures:
        return {
            "rows": tuple(rows),
            "slide_audits": slide_audits,
            "canonical_cache_sha256": inputs.canonical.cache_sha256,
            "failures": tuple(failures),
        }
    if len(slide_audits) != config.expected_slide_count:
        raise ValueError(
            f"Physical multiscale v2 slide count drift: "
            f"expected={config.expected_slide_count}, actual={len(slide_audits)}"
        )
    if [str(row["sample_id"]) for row in rows] != [
        str(row["sample_id"]) for row in inputs.selected_metadata
    ]:
        raise ValueError("Physical multiscale v2 is not bijective with canonical row order.")
    return {
        "rows": tuple(rows),
        "slide_audits": slide_audits,
        "canonical_cache_sha256": inputs.canonical.cache_sha256,
        "failures": (),
    }


def _audit_report(
    config: PhysicalMultiscaleV2BuildConfig,
    collected: Mapping[str, object],
) -> dict[str, object]:
    rows = list(collected["rows"])  # type: ignore[arg-type]
    slide_audits = collected["slide_audits"]
    failures = list(collected.get("failures", ()))  # type: ignore[arg-type]
    assert isinstance(slide_audits, Mapping)
    shifted_samples: set[str] = set()
    counts_by_fov = {f"{fov:g}um": 0 for fov in config.fov_um}
    counts_by_center = {center: 0 for center in config.eligible_centers}
    counts_by_label = {"0": 0, "1": 0}
    maximum_shift_px = 0
    maximum_shift_fraction = 0.0
    direction_counts = {
        "left": 0,
        "right": 0,
        "up": 0,
        "down": 0,
    }
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
            sample_id = str(row["sample_id"])
            shifted_samples.add(sample_id)
            counts_by_center[str(row["center"])] += 1
            counts_by_label[str(int(row["label"]))] += 1
    audits = [
        entry["audit"] for entry in slide_audits.values()  # type: ignore[union-attr]
    ]
    mpp_values = [
        value
        for audit in audits
        for value in (audit.mpp_x, audit.mpp_y)
    ]
    mechanical_pass = (
        not failures
        and len(rows) == config.expected_row_count
        and len(slide_audits) == config.expected_slide_count
    )
    return {
        "schema_version": "midogpp_physical_multiscale_source_audit_v2",
        "status": "PASS" if mechanical_pass else "FAIL",
        "profile_id": PROFILE_ID,
        "row_count": len(rows),
        "expected_row_count": config.expected_row_count,
        "slide_count": len(slide_audits),
        "expected_slide_count": config.expected_slide_count,
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
        "shifted_sample_counts_by_center": counts_by_center,
        "shifted_sample_counts_by_label": counts_by_label,
        "shift_direction_counts": direction_counts,
        "maximum_axis_shift_px": maximum_shift_px,
        "maximum_axis_shift_fraction": maximum_shift_fraction,
        "mechanical_pass_criteria_are_label_blind": True,
        "label_stratified_counts_are_audit_only": True,
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
    config: PhysicalMultiscaleV2BuildConfig,
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
        "geometry": geometry_policy_v2(config),
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
            "expected_checkpoint_file_sha256": config.expected_checkpoint_file_sha256,
            "expected_state_dict_sha256": config.expected_state_dict_sha256,
            "expected_preprocessing_config_hash": (
                config.expected_preprocessing_config_hash
            ),
        },
        "runtime_identity": {
            "timm": config.expected_timm_version,
            "torch": config.expected_torch_version,
            "pillow": config.expected_pillow_version,
        },
        "run": {
            "eligible_centers": list(config.eligible_centers),
            "experiment_seed": config.experiment_seed,
            "device": config.device,
            "batch_size": config.batch_size,
            "require_tiled_reader": config.require_tiled_reader,
            "expected_row_count": config.expected_row_count,
            "expected_slide_count": config.expected_slide_count,
        },
        "claim_firewall": claim_firewall_v2(),
    }


def _fov_key(value: float) -> str:
    return f"{int(value) if float(value).is_integer() else value:g}um"


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
