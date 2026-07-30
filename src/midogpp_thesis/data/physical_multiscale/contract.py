"""Immutable sample-to-slide physical extraction contract builder."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from midogpp_thesis.common.hashing import stable_hash
from .config import PhysicalMultiscaleBuildConfig
from .contract_inputs import (
    assert_raw_source_identity as _assert_raw_source_identity,
    load_contract_inputs,
    relative_to_repo as _relative_to_repo,
    sha256_file as _sha256_file,
)
from .geometry import physical_crop_geometry
from .resolution_audit import MppAudit, audit_tiff_mpp


CONTRACT_FILES = (
    "config.resolved.yaml",
    "physical_multiscale_contract.json",
    "physical_multiscale_manifest.csv",
    "resolution_audit.csv",
    "row_alignment_report.json",
    "leakage_report.json",
)


def audit_physical_multiscale_sources(
    config: PhysicalMultiscaleBuildConfig,
) -> Mapping[str, object]:
    """Read every required level-0 TIFF header without creating contract bytes."""

    inputs = load_contract_inputs(config)
    canonical = inputs.canonical
    manifest_by_id = inputs.manifest_by_id
    source_by_case = inputs.source_by_case
    required_slides: dict[Path, MppAudit] = {}
    selected_rows = 0
    maximum_padding_fraction = 0.0
    padding_violations: list[dict[str, object]] = []
    padding_violation_sample_ids: set[str] = set()
    violation_counts_by_fov = {f"{fov:g}um": 0 for fov in config.fov_um}
    violation_sample_counts_by_center = {
        center: 0 for center in config.eligible_centers
    }
    violation_sample_counts_by_label = {"0": 0, "1": 0}
    for metadata in inputs.selected_metadata:
        sample_id = str(metadata.get("sample_id", ""))
        manifest = manifest_by_id.get(sample_id)
        if manifest is None:
            raise ValueError(f"Canonical cache sample is absent from base manifest: {sample_id}")
        center = str(metadata.get("center", manifest.get("center", "")))
        case_id = str(metadata.get("case_id", manifest.get("case_id", "")))
        source = source_by_case.get(case_id)
        if source is None:
            raise ValueError(f"No unique raw TIFF mapping for case_id={case_id!r}")
        _assert_raw_source_identity(manifest, source)
        raw_path = Path(str(source["raw_path"]))
        if raw_path not in required_slides:
            required_slides[raw_path] = audit_tiff_mpp(
                raw_path,
                mpp_min=config.mpp_min,
                mpp_max=config.mpp_max,
                anisotropy_relative_max=config.anisotropy_relative_max,
                dual_source_relative_max=config.dual_source_relative_max,
            )
        audit = required_slides[raw_path]
        center_x = float(manifest["patch_center_x"])
        center_y = float(manifest["patch_center_y"])
        sample_violated = False
        for fov in config.fov_um:
            geometry = physical_crop_geometry(
                center_x=center_x,
                center_y=center_y,
                fov_um=fov,
                mpp_x=audit.mpp_x,
                mpp_y=audit.mpp_y,
                image_width=audit.width,
                image_height=audit.height,
            )
            maximum_padding_fraction = max(
                maximum_padding_fraction,
                geometry.padding_fraction,
            )
            if geometry.padding_fraction > config.padding_fraction_max:
                sample_violated = True
                fov_key = f"{fov:g}um"
                violation_counts_by_fov[fov_key] += 1
                if len(padding_violations) < 20:
                    padding_violations.append(
                        {
                            "sample_id": sample_id,
                            "case_id": case_id,
                            "fov_um": fov,
                            "padding_fraction": geometry.padding_fraction,
                            "padding_fraction_max": config.padding_fraction_max,
                        }
                    )
        if sample_violated:
            padding_violation_sample_ids.add(sample_id)
            violation_sample_counts_by_center[center] += 1
            violation_sample_counts_by_label[
                str(int(float(str(manifest["label"]))))
            ] += 1
        selected_rows += 1
    if selected_rows == 0 or not required_slides:
        raise ValueError("Physical source audit selected no eligible train rows or slides.")
    mpp_values = [
        value
        for audit in required_slides.values()
        for value in (audit.mpp_x, audit.mpp_y)
    ]
    report = {
        "schema_version": "midogpp_physical_multiscale_source_audit_v1",
        "status": "FAIL" if padding_violations else "PASS",
        "row_count": selected_rows,
        "slide_count": len(required_slides),
        "minimum_mpp": min(mpp_values),
        "maximum_mpp": max(mpp_values),
        "all_orientations_top_left": all(
            audit.orientation == 1 for audit in required_slides.values()
        ),
        "all_required_tiff_headers_validated": True,
        "maximum_padding_fraction": maximum_padding_fraction,
        "padding_violation_count": sum(violation_counts_by_fov.values()),
        "padding_violation_sample_count": len(padding_violation_sample_ids),
        "padding_violation_sample_fraction": (
            len(padding_violation_sample_ids) / float(selected_rows)
        ),
        "padding_violation_counts_by_fov": violation_counts_by_fov,
        "padding_violation_sample_counts_by_center": (
            violation_sample_counts_by_center
        ),
        "padding_violation_sample_counts_by_label": (
            violation_sample_counts_by_label
        ),
        "padding_violation_examples": padding_violations,
    }
    if padding_violations:
        raise ValueError(
            "Physical source geometry audit failed: "
            + json.dumps(report, sort_keys=True)
        )
    return report


def build_physical_multiscale_contract(
    config: PhysicalMultiscaleBuildConfig,
) -> Path:
    """Bind canonical eligible train rows bijectively to raw level-0 TIFF geometry."""

    if config.contract_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable physical contract: {config.contract_root}"
        )
    inputs = load_contract_inputs(config)
    canonical = inputs.canonical
    manifest_by_id = inputs.manifest_by_id
    source_by_case = inputs.source_by_case
    selected_metadata = inputs.selected_metadata

    slide_audits: dict[Path, tuple[str, MppAudit]] = {}
    sidecar_rows: list[dict[str, object]] = []
    for row_index, metadata in enumerate(selected_metadata):
        sample_id = str(metadata["sample_id"])
        manifest = manifest_by_id[sample_id]
        case_id = str(metadata.get("case_id", manifest.get("case_id", "")))
        source = source_by_case.get(case_id)
        if source is None:
            raise ValueError(f"No unique raw TIFF mapping for case_id={case_id!r}")
        _assert_raw_source_identity(manifest, source)
        raw_path = Path(str(source["raw_path"]))
        if raw_path not in slide_audits:
            audit = audit_tiff_mpp(
                raw_path,
                mpp_min=config.mpp_min,
                mpp_max=config.mpp_max,
                anisotropy_relative_max=config.anisotropy_relative_max,
                dual_source_relative_max=config.dual_source_relative_max,
            )
            slide_audits[raw_path] = (_sha256_file(raw_path), audit)
        raw_hash, audit = slide_audits[raw_path]
        center_x = float(manifest["patch_center_x"])
        center_y = float(manifest["patch_center_y"])
        scale_geometry = {}
        for fov in config.fov_um:
            geometry = physical_crop_geometry(
                center_x=center_x,
                center_y=center_y,
                fov_um=fov,
                mpp_x=audit.mpp_x,
                mpp_y=audit.mpp_y,
                image_width=audit.width,
                image_height=audit.height,
            )
            if geometry.padding_fraction > config.padding_fraction_max:
                raise ValueError(
                    f"{sample_id}: padding fraction {geometry.padding_fraction:.6f} exceeds "
                    f"{config.padding_fraction_max:.6f} for FOV={fov}"
                )
            scale_geometry[_fov_key(fov)] = geometry.__dict__
        sidecar_rows.append(
            {
                "row_index": row_index,
                "sample_id": sample_id,
                "case_id": case_id,
                "annotation_id": str(manifest.get("annotation_id", "")),
                "label": int(float(str(manifest["label"]))),
                "split": "train",
                "center": str(metadata.get("center", manifest.get("center", ""))),
                "raw_tiff_path": _relative_to_repo(raw_path, config.repo_root),
                "raw_tiff_sha256": raw_hash,
                "level": 0,
                "orientation": audit.orientation,
                "image_width": audit.width,
                "image_height": audit.height,
                "center_x": center_x,
                "center_y": center_y,
                "mpp_x": audit.mpp_x,
                "mpp_y": audit.mpp_y,
                "mpp_source": audit.source,
                "scale_geometry_json": json.dumps(
                    scale_geometry, sort_keys=True, separators=(",", ":")
                ),
            }
        )

    sample_ids = [str(row["sample_id"]) for row in sidecar_rows]
    if sample_ids != [
        str(row["sample_id"]) for row in selected_metadata
    ] or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Physical contract is not bijective with canonical sample row order.")
    geometry_policy = {
        "padding_fraction_max": config.padding_fraction_max,
        "padding_rgb": list(config.padding_rgb),
        "resize_interpolation": config.resize_interpolation,
        "resize_antialias": config.resize_antialias,
        "coordinate_origin": "level_0_top_left",
        "pixel_rounding": "round_half_up",
    }
    contract_hash = physical_contract_hash(
        canonical_cache_sha256=canonical.cache_sha256,
        manifest_rows=sidecar_rows,
        fov_um=config.fov_um,
        geometry_policy=geometry_policy,
    )
    config.contract_root.mkdir(parents=True, exist_ok=False)
    _write_csv(config.contract_root / "physical_multiscale_manifest.csv", sidecar_rows)
    resolution_rows = [
        {
            "raw_tiff_path": _relative_to_repo(path, config.repo_root),
            "raw_tiff_sha256": raw_hash,
            **audit.__dict__,
            "status": "PASS",
        }
        for path, (raw_hash, audit) in sorted(
            slide_audits.items(), key=lambda item: str(item[0])
        )
    ]
    _write_csv(config.contract_root / "resolution_audit.csv", resolution_rows)
    _write_json(
        config.contract_root / "physical_multiscale_contract.json",
        {
            "schema_version": "midogpp_physical_multiscale_contract_v1",
            "status": "PASS",
            "contract_hash": contract_hash,
            "artifact_dataset": "MIDOG++",
            "claim_dataset": "MIDOG++",
            "split": "train",
            "eligible_centers": list(config.eligible_centers),
            "excluded_centers": ["4"],
            "canonical_cache_sha256": canonical.cache_sha256,
            "row_count": len(sidecar_rows),
            "slide_count": len(slide_audits),
            "fov_um": list(config.fov_um),
            "geometry_policy": geometry_policy,
            "target_labels_used_for_extraction": False,
        },
    )
    _write_json(
        config.contract_root / "row_alignment_report.json",
        {
            "schema_version": "midogpp_physical_multiscale_row_alignment_v1",
            "status": "PASS",
            "row_count": len(sidecar_rows),
            "sample_id_order_hash": stable_hash(sample_ids),
            "center_4_present": False,
            "canonical_order_exact": True,
        },
    )
    _write_json(
        config.contract_root / "leakage_report.json",
        {
            "schema_version": "midogpp_physical_multiscale_leakage_v1",
            "status": "PASS",
            "split": "train",
            "target_labels_used_for_geometry": False,
            "target_metrics_used_for_geometry": False,
            "post_label_row_exclusion": False,
        },
    )
    (config.contract_root / "config.resolved.yaml").write_text(
        yaml.safe_dump(
            _resolved_config_payload(config),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config.contract_root


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    columns = tuple(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fov_key(value: float) -> str:
    return f"{int(value) if float(value).is_integer() else value:g}um"


def physical_contract_hash(
    *,
    canonical_cache_sha256: str,
    manifest_rows: Sequence[Mapping[str, object]],
    fov_um: Sequence[float],
    geometry_policy: Mapping[str, object],
) -> str:
    """Bind every geometry-bearing manifest scalar to the contract identity."""

    normalized_rows = [
        {
            str(key): "" if value is None else str(value)
            for key, value in row.items()
        }
        for row in manifest_rows
    ]
    return stable_hash(
        {
            "canonical_cache_sha256": str(canonical_cache_sha256),
            "manifest_rows": normalized_rows,
            "fov_um": [float(value) for value in fov_um],
            "geometry_policy": dict(geometry_policy),
        }
    )


def _resolved_config_payload(
    config: PhysicalMultiscaleBuildConfig,
) -> Mapping[str, object]:
    return {
        "artifact": {"name": "physical_multiscale_center_pooling_pilot_v1"},
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
        "geometry": {
            "padding_fraction_max": config.padding_fraction_max,
            "padding_rgb": list(config.padding_rgb),
            "resize_interpolation": config.resize_interpolation,
            "resize_antialias": config.resize_antialias,
        },
        "bridge": {
            "minimum_cosine": config.bridge_minimum_cosine,
            "maximum_relative_l2": config.bridge_maximum_relative_l2,
            "minimum_prediction_agreement": (
                config.bridge_minimum_prediction_agreement
            ),
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
        "run": {
            "eligible_centers": list(config.eligible_centers),
            "experiment_seed": config.experiment_seed,
            "device": config.device,
            "batch_size": config.batch_size,
            "require_tiled_reader": config.require_tiled_reader,
        },
    }
