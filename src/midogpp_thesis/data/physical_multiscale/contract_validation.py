"""Independent validators for versioned physical multiscale contracts."""

from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.common.midogpp import MIDOGPP_ELIGIBLE_CENTERS

from .config_v2 import CONTRACT_SCHEMA_VERSION, PROFILE_ID
from .contract_v2 import (
    V2_CONTRACT_FILES,
    claim_firewall_v2,
    physical_contract_hash_v2,
)
from .geometry import physical_crop_geometry_v2

if TYPE_CHECKING:
    from .config_v2 import PhysicalMultiscaleV2BuildConfig


DEFAULT_CONTRACT_SCHEMA_V2 = (
    Path(__file__).resolve().parents[4]
    / "datasets"
    / "midogpp"
    / "schemas"
    / "physical_multiscale_annotation_local_pooling_contract_v2.schema.json"
)


def validate_contract_document_v2(
    contract: Mapping[str, object],
    *,
    schema_path: str | Path = DEFAULT_CONTRACT_SCHEMA_V2,
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
            f"Physical multiscale v2 contract schema failed at {location}: {exc.message}"
        ) from exc


def validate_contract_bundle_v2(
    root: str | Path,
    *,
    verify_raw_files: bool,
    expected_config: "PhysicalMultiscaleV2BuildConfig | None" = None,
) -> Mapping[str, object]:
    path = Path(root)
    _require(path, V2_CONTRACT_FILES)
    contract = _json(path / "physical_multiscale_contract.json")
    validate_contract_document_v2(contract)
    rows = _csv(path / "physical_multiscale_manifest.csv")
    resolution_rows = _csv(path / "resolution_audit.csv")
    audit = _json(path / "source_geometry_audit.json")
    alignment = _json(path / "row_alignment_report.json")
    leakage = _json(path / "leakage_report.json")
    geometry_policy = contract.get("geometry_policy")
    identity_hashes = contract.get("identity_hashes")
    if not isinstance(geometry_policy, Mapping) or not isinstance(identity_hashes, Mapping):
        raise ValueError("Physical multiscale v2 contract identity is malformed.")
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
        or len(sample_ids) != len(set(sample_ids))
        or centers != MIDOGPP_ELIGIBLE_CENTERS
        or "4" in centers
        or any(row.get("split") != "train" for row in rows)
        or contract.get("claim_firewall") != claim_firewall_v2()
        or leakage.get("mechanical_pass_criteria_are_label_blind") is not True
        or leakage.get("label_stratified_shift_summary_is_audit_only") is not True
    ):
        raise ValueError("Physical multiscale v2 contract status/cohort firewall failed.")
    expected_identity = _identity_hashes(rows)
    if dict(identity_hashes) != expected_identity:
        raise ValueError("Physical multiscale v2 ordered cohort identity drifted.")
    resolution_by_path = {row["raw_tiff_path"]: row for row in resolution_rows}
    if len(resolution_by_path) != len(resolution_rows):
        raise ValueError("Physical multiscale v2 resolution audit duplicates TIFF paths.")
    manifest_hashes: dict[str, str] = {}
    for row in rows:
        raw_path = row["raw_tiff_path"]
        raw_hash = row["raw_tiff_sha256"]
        if raw_path in manifest_hashes and manifest_hashes[raw_path] != raw_hash:
            raise ValueError("Physical multiscale v2 assigns multiple hashes to one TIFF.")
        manifest_hashes[raw_path] = raw_hash
        geometries = json.loads(row["scale_geometry_json"])
        if set(geometries) != {"28um", "56um", "112um"}:
            raise ValueError("Physical multiscale v2 has incomplete FOV geometry.")
        for key, fov in (("28um", 28.0), ("56um", 56.0), ("112um", 112.0)):
            expected = asdict(
                physical_crop_geometry_v2(
                    anchor_x=float(row["anchor_x"]),
                    anchor_y=float(row["anchor_y"]),
                    fov_um=fov,
                    mpp_x=float(row["mpp_x"]),
                    mpp_y=float(row["mpp_y"]),
                    image_width=int(row["image_width"]),
                    image_height=int(row["image_height"]),
                )
            )
            if geometries[key] != expected:
                raise ValueError(
                    f"Physical multiscale v2 geometry recomputation drift: "
                    f"{row['sample_id']} {key}"
                )
            if (
                int(expected["realized_x1"]) - int(expected["realized_x0"])
                != int(expected["side_px"])
                or int(expected["realized_y1"]) - int(expected["realized_y0"])
                != int(expected["side_px"])
                or any(int(expected[name]) != 0 for name in (
                    "pad_left",
                    "pad_top",
                    "pad_right",
                    "pad_bottom",
                ))
                or float(expected["padding_fraction"]) != 0.0
            ):
                raise ValueError("Physical multiscale v2 exact-square invariant failed.")
    if set(manifest_hashes) != set(resolution_by_path):
        raise ValueError("Physical multiscale v2 resolution coverage drifted.")
    if int(contract.get("slide_count", -1)) != len(manifest_hashes):
        raise ValueError("Physical multiscale v2 slide count drifted.")
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
                    f"Physical multiscale v2 TIFF lineage failed: {raw_path}"
                )
    expected_hash = physical_contract_hash_v2(
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
        raise ValueError("Physical multiscale v2 contract hash/alignment drifted.")
    if expected_config is not None and (
        len(rows) != expected_config.expected_row_count
        or len(manifest_hashes) != expected_config.expected_slide_count
    ):
        raise ValueError("Physical multiscale v2 expected cohort cardinality drifted.")
    return {
        "status": "PASS",
        "profile_id": PROFILE_ID,
        "row_count": len(rows),
        "slide_count": len(manifest_hashes),
        "contract_hash": expected_hash,
    }


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


def _require(root: Path, relatives: tuple[str, ...]) -> None:
    missing = [relative for relative in relatives if not (root / relative).is_file()]
    if missing:
        raise ValueError(f"Physical multiscale v2 contract is missing files: {missing}")


def _json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
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
