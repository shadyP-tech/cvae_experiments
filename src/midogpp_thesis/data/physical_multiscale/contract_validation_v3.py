"""Independent validation for the clipped-bbox annotation-local v3 contract."""

from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

import yaml

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.common.midogpp import MIDOGPP_ELIGIBLE_CENTERS
from midogpp_thesis.data.features.virchow2_tokens import (
    normalized_position_to_window_start,
)

from .config_v3 import (
    CONTRACT_SCHEMA_VERSION,
    PROFILE_ID,
    PhysicalMultiscaleV3BuildConfig,
)
from .contract_inputs import load_contract_inputs
from .contract_v3 import (
    V3_CONTRACT_FILES,
    claim_firewall_v3,
    geometry_policy_v3,
    physical_contract_hash_v3,
)
from .geometry import (
    clipped_annotation_bbox_anchor,
    physical_crop_geometry_in_bounds,
)


DEFAULT_CONTRACT_SCHEMA_V3 = (
    Path(__file__).resolve().parents[4]
    / "datasets"
    / "midogpp"
    / "schemas"
    / "physical_multiscale_clipped_bbox_annotation_local_pooling_contract_v3.schema.json"
)


def validate_contract_document_v3(
    contract: Mapping[str, object],
    *,
    schema_path: str | Path = DEFAULT_CONTRACT_SCHEMA_V3,
) -> None:
    try:
        from jsonschema import Draft202012Validator  # type: ignore
        from jsonschema.exceptions import ValidationError  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Physical contract validation requires jsonschema.") from exc
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    try:
        Draft202012Validator(schema).validate(dict(contract))
    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise ValueError(
            f"Physical multiscale v3 contract schema failed at {location}: "
            f"{exc.message}"
        ) from exc


def validate_contract_bundle_v3(
    root: str | Path,
    *,
    verify_raw_files: bool,
    expected_config: PhysicalMultiscaleV3BuildConfig | None = None,
) -> Mapping[str, object]:
    """Reconstruct every bbox, crop, token, identity, and content invariant."""

    path = Path(root)
    _require(path, V3_CONTRACT_FILES)
    contract = _json(path / "physical_multiscale_contract.json")
    validate_contract_document_v3(contract)
    rows = _csv(path / "physical_multiscale_manifest.csv")
    resolution_rows = _csv(path / "resolution_audit.csv")
    audit = _json(path / "source_geometry_audit.json")
    alignment = _json(path / "row_alignment_report.json")
    leakage = _json(path / "leakage_report.json")
    resolved_config = _yaml(path / "config.resolved.yaml")
    geometry_policy = contract.get("geometry_policy")
    identity_hashes = contract.get("identity_hashes")
    if not isinstance(geometry_policy, Mapping) or not isinstance(identity_hashes, Mapping):
        raise ValueError("Physical multiscale v3 contract identity is malformed.")
    sample_ids = [row["sample_id"] for row in rows]
    centers = tuple(sorted(set(row["center"] for row in rows), key=int))
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or contract.get("profile_id") != PROFILE_ID
        or contract.get("status") != "PASS"
        or audit.get("status") != "PASS"
        or alignment.get("status") != "PASS"
        or leakage.get("status") != "PASS"
        or len(rows) != int(contract.get("row_count", -1))
        or len(rows) != 9648
        or len(sample_ids) != len(set(sample_ids))
        or centers != MIDOGPP_ELIGIBLE_CENTERS
        or "4" in centers
        or any(row.get("split") != "train" for row in rows)
        or contract.get("claim_firewall") != claim_firewall_v3()
        or leakage.get("mechanical_pass_criteria_are_label_blind") is not True
        or leakage.get("label_stratified_geometry_summary_is_audit_only") is not True
        or leakage.get("geometry_driven_row_exclusion") is not False
        or alignment.get("geometry_driven_row_exclusion") is not False
    ):
        raise ValueError("Physical multiscale v3 contract status/cohort firewall failed.")
    if expected_config is not None and dict(geometry_policy) != geometry_policy_v3(
        expected_config
    ):
        raise ValueError("Physical multiscale v3 geometry policy drifted.")

    expected_identity = _identity_hashes(rows)
    if dict(identity_hashes) != expected_identity:
        raise ValueError("Physical multiscale v3 ordered cohort identity drifted.")
    _validate_resolved_config(
        resolved_config,
        geometry_policy=geometry_policy,
        expected_config=expected_config,
    )
    _validate_against_canonical_inputs(
        rows,
        expected_config,
        declared_cache_sha256=str(contract["canonical_cache_sha256"]),
    )

    resolution_by_path = {row["raw_tiff_path"]: row for row in resolution_rows}
    if len(resolution_by_path) != len(resolution_rows):
        raise ValueError("Physical multiscale v3 resolution audit duplicates TIFF paths.")
    manifest_hashes: dict[str, str] = {}
    clipped_count = 0
    for row in rows:
        raw_path = row["raw_tiff_path"]
        raw_hash = row["raw_tiff_sha256"]
        if raw_path in manifest_hashes and manifest_hashes[raw_path] != raw_hash:
            raise ValueError("Physical multiscale v3 assigns multiple hashes to one TIFF.")
        manifest_hashes[raw_path] = raw_hash
        anchor = clipped_annotation_bbox_anchor(
            bbox_x=float(row["bbox_x"]),
            bbox_y=float(row["bbox_y"]),
            bbox_w=float(row["bbox_w"]),
            bbox_h=float(row["bbox_h"]),
            image_width=int(row["image_width"]),
            image_height=int(row["image_height"]),
            minimum_clipped_area_fraction=float(
                geometry_policy["minimum_clipped_bbox_area_fraction"]
            ),
        )
        anchor_payload = asdict(anchor)
        for key, expected_value in anchor_payload.items():
            actual_value: object
            if isinstance(expected_value, bool):
                actual_value = _bool(row[key])
            elif isinstance(expected_value, float):
                actual_value = float(row[key])
            else:
                actual_value = row[key]
            if actual_value != expected_value:
                raise ValueError(
                    f"Physical multiscale v3 clipped-bbox recomputation drift: "
                    f"{row['sample_id']} {key}"
                )
        clipped_count += int(anchor.was_clipped)
        geometries = json.loads(row["scale_geometry_json"])
        if set(geometries) != {"28um", "56um", "112um"}:
            raise ValueError("Physical multiscale v3 has incomplete FOV geometry.")
        for key, fov in (("28um", 28.0), ("56um", 56.0), ("112um", 112.0)):
            crop = physical_crop_geometry_in_bounds(
                anchor_x=anchor.anchor_x,
                anchor_y=anchor.anchor_y,
                fov_um=fov,
                mpp_x=float(row["mpp_x"]),
                mpp_y=float(row["mpp_y"]),
                image_width=int(row["image_width"]),
                image_height=int(row["image_height"]),
            )
            token_row, token_col = normalized_position_to_window_start(
                x=crop.p_x,
                y=crop.p_y,
            )
            expected_geometry = {
                **asdict(crop),
                "token_start_row": token_row,
                "token_start_col": token_col,
            }
            if geometries[key] != expected_geometry:
                raise ValueError(
                    f"Physical multiscale v3 crop/token recomputation drift: "
                    f"{row['sample_id']} {key}"
                )
            if (
                int(crop.realized_x1) - int(crop.realized_x0) != int(crop.side_px)
                or int(crop.realized_y1) - int(crop.realized_y0)
                != int(crop.side_px)
                or any(
                    int(getattr(crop, name)) != 0
                    for name in ("pad_left", "pad_top", "pad_right", "pad_bottom")
                )
                or crop.padding_fraction != 0.0
            ):
                raise ValueError("Physical multiscale v3 exact-square invariant failed.")
    if clipped_count != 84:
        raise ValueError(
            f"Physical multiscale v3 clipped bbox count drifted: {clipped_count}."
        )
    _validate_regression_anchors(rows)
    if (
        int(audit.get("clipped_bbox_count", -1)) != clipped_count
        or int(audit.get("expected_clipped_bbox_count", -1)) != 84
        or audit.get("geometry_driven_row_exclusion") is not False
        or audit.get("mechanical_pass_criteria_are_label_blind") is not True
        or audit.get("center_and_label_stratified_counts_are_audit_only") is not True
        or audit.get("out_of_bounds_pixels_synthesized") is not False
        or int(audit.get("realized_padding_pixel_count", -1)) != 0
    ):
        raise ValueError("Physical multiscale v3 source audit invariants drifted.")
    if set(manifest_hashes) != set(resolution_by_path):
        raise ValueError("Physical multiscale v3 resolution coverage drifted.")
    if (
        int(contract.get("slide_count", -1)) != len(manifest_hashes)
        or len(manifest_hashes) != 216
    ):
        raise ValueError("Physical multiscale v3 slide count drifted.")

    if verify_raw_files:
        repo_root = (
            expected_config.repo_root
            if expected_config is not None
            else Path(__file__).resolve().parents[4]
        )
        for raw_path, expected_hash in manifest_hashes.items():
            candidate = Path(raw_path)
            resolved = candidate if candidate.is_absolute() else repo_root / candidate
            audit_row = resolution_by_path[raw_path]
            if (
                not resolved.is_file()
                or _sha256(resolved) != expected_hash
                or audit_row.get("raw_tiff_sha256") != expected_hash
                or audit_row.get("status") != "PASS"
                or int(audit_row.get("orientation", -1)) != 1
            ):
                raise ValueError(
                    f"Physical multiscale v3 TIFF lineage failed: {raw_path}"
                )

    expected_hash = physical_contract_hash_v3(
        canonical_cache_sha256=str(contract["canonical_cache_sha256"]),
        manifest_rows=rows,
        fov_um=tuple(float(value) for value in contract["fov_um"]),
        geometry_policy=geometry_policy,
        identity_hashes=identity_hashes,
    )
    if (
        contract.get("contract_hash") != expected_hash
        or alignment.get("sample_id_order_hash") != stable_hash(sample_ids)
        or alignment.get("identity_hashes") != identity_hashes
        or alignment.get("center_4_present") is not False
        or alignment.get("canonical_order_exact") is not True
    ):
        raise ValueError("Physical multiscale v3 contract hash/alignment drifted.")
    if expected_config is not None and (
        len(rows) != expected_config.expected_row_count
        or len(manifest_hashes) != expected_config.expected_slide_count
        or clipped_count != expected_config.expected_clipped_bbox_count
    ):
        raise ValueError("Physical multiscale v3 expected cohort cardinality drifted.")
    return {
        "status": "PASS",
        "profile_id": PROFILE_ID,
        "row_count": len(rows),
        "slide_count": len(manifest_hashes),
        "clipped_bbox_count": clipped_count,
        "contract_hash": expected_hash,
    }


def _validate_against_canonical_inputs(
    rows: list[dict[str, str]],
    config: PhysicalMultiscaleV3BuildConfig | None,
    *,
    declared_cache_sha256: str,
) -> None:
    if config is None:
        return
    inputs = load_contract_inputs(config)
    if declared_cache_sha256 != inputs.canonical.cache_sha256:
        raise ValueError(
            "Physical multiscale v3 canonical cache SHA256 differs from "
            "the configured cache bytes."
        )
    expected_ids = [str(row["sample_id"]) for row in inputs.selected_metadata]
    if [row["sample_id"] for row in rows] != expected_ids:
        raise ValueError("Physical multiscale v3 canonical cohort/order differs.")
    for row in rows:
        source = inputs.manifest_by_id[row["sample_id"]]
        for field in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"):
            if float(row[field]) != float(source[field]):
                raise ValueError(
                    f"Physical multiscale v3 raw bbox differs: "
                    f"{row['sample_id']} {field}"
                )


def _validate_resolved_config(
    payload: Mapping[str, object],
    *,
    geometry_policy: Mapping[str, object],
    expected_config: PhysicalMultiscaleV3BuildConfig | None,
) -> None:
    artifact = payload.get("artifact")
    geometry = payload.get("geometry")
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("name") != PROFILE_ID
        or not isinstance(geometry, Mapping)
        or dict(geometry) != dict(geometry_policy)
    ):
        raise ValueError("Physical multiscale v3 resolved config identity drifted.")
    if expected_config is None:
        return
    expected_sections: dict[str, Mapping[str, object]] = {
        "inputs": {
            "raw_root": str(expected_config.raw_root),
            "raw_metadata_path": str(expected_config.raw_metadata_path),
            "base_manifest_path": str(expected_config.base_manifest_path),
            "canonical_cache_path": str(expected_config.canonical_cache_path),
            "canonical_reference_root": str(
                expected_config.canonical_reference_root
            ),
        },
        "outputs": {
            "contract_root": str(expected_config.contract_root),
            "cache_bundle_root": str(expected_config.cache_bundle_root),
            "b_cache_root": str(expected_config.b_cache_root),
            "c_cache_root": str(expected_config.c_cache_root),
        },
        "physical_scale": {
            "fov_um": list(expected_config.fov_um),
            "mpp_min": expected_config.mpp_min,
            "mpp_max": expected_config.mpp_max,
            "anisotropy_relative_max": (
                expected_config.anisotropy_relative_max
            ),
            "dual_source_relative_max": (
                expected_config.dual_source_relative_max
            ),
        },
        "bridge": {
            "minimum_cosine": expected_config.bridge_minimum_cosine,
            "maximum_relative_l2": expected_config.bridge_maximum_relative_l2,
            "minimum_prediction_agreement": (
                expected_config.bridge_minimum_prediction_agreement
            ),
            "maximum_absolute_equal_center_bacc_delta": (
                expected_config.bridge_maximum_equal_center_bacc_delta
            ),
        },
        "model": {
            "model_ref": expected_config.model_ref,
            "model_revision": expected_config.model_revision,
            "expected_model_config_sha256": (
                expected_config.expected_model_config_sha256
            ),
            "expected_checkpoint_file_sha256": (
                expected_config.expected_checkpoint_file_sha256
            ),
            "expected_state_dict_sha256": (
                expected_config.expected_state_dict_sha256
            ),
            "expected_preprocessing_config_hash": (
                expected_config.expected_preprocessing_config_hash
            ),
        },
        "runtime_identity": {
            "timm": expected_config.expected_timm_version,
            "torch": expected_config.expected_torch_version,
            "pillow": expected_config.expected_pillow_version,
            "pyvips": expected_config.expected_pyvips_version,
            "libvips": expected_config.expected_libvips_version,
        },
        "run": {
            "eligible_centers": list(expected_config.eligible_centers),
            "experiment_seed": expected_config.experiment_seed,
            "device": expected_config.device,
            "batch_size": expected_config.batch_size,
            "require_tiled_reader": expected_config.require_tiled_reader,
            "required_slide_reader_backend": (
                expected_config.required_slide_reader_backend
            ),
            "expected_row_count": expected_config.expected_row_count,
            "expected_slide_count": expected_config.expected_slide_count,
            "expected_clipped_bbox_count": (
                expected_config.expected_clipped_bbox_count
            ),
        },
    }
    for section, expected in expected_sections.items():
        actual = payload.get(section)
        if not isinstance(actual, Mapping) or dict(actual) != dict(expected):
            raise ValueError(
                f"Physical multiscale v3 resolved config section drifted: "
                f"{section}."
            )


def _validate_regression_anchors(rows: list[dict[str, str]]) -> None:
    by_id = {row["sample_id"]: row for row in rows}
    expected = {
        "305__305__ann15363__y1": (11.5, 2416.0, 0.46),
        "309__309__ann15728__y0": (4211.0, 10.0, 0.4),
    }
    for sample_id, values in expected.items():
        row = by_id.get(sample_id)
        if row is None or (
            float(row["anchor_x"]),
            float(row["anchor_y"]),
            float(row["clipped_area_fraction"]),
        ) != values:
            raise ValueError(
                f"Physical multiscale v3 regression anchor drifted: {sample_id}."
            )


def _identity_hashes(rows: list[dict[str, str]]) -> dict[str, str]:
    return {
        "sample_id_order": stable_hash([row["sample_id"] for row in rows]),
        "case_id_order": stable_hash([row["case_id"] for row in rows]),
        "slide_id_order": stable_hash([row["raw_tiff_path"] for row in rows]),
        "center_order": stable_hash([row["center"] for row in rows]),
        "label_order": stable_hash([int(row["label"]) for row in rows]),
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


def _bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Expected canonical bool serialization, got {value!r}.")


def _require(root: Path, relatives: tuple[str, ...]) -> None:
    missing = [relative for relative in relatives if not (root / relative).is_file()]
    if missing:
        raise ValueError(f"Physical multiscale v3 contract is missing files: {missing}")


def _json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _yaml(path: Path) -> Mapping[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
