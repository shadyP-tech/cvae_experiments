"""Independent validation for the frozen padding-capable v1 contract."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.common.midogpp import MIDOGPP_ELIGIBLE_CENTERS

from .contract import physical_contract_hash


DEFAULT_CONTRACT_SCHEMA = (
    Path(__file__).resolve().parents[4]
    / "datasets"
    / "midogpp"
    / "schemas"
    / "physical_multiscale_center_pooling_contract.schema.json"
)


def validate_contract_bundle(
    root: str | Path,
    *,
    schema_path: str | Path = DEFAULT_CONTRACT_SCHEMA,
) -> Mapping[str, object]:
    path = Path(root)
    required = (
        "config.resolved.yaml",
        "physical_multiscale_contract.json",
        "physical_multiscale_manifest.csv",
        "resolution_audit.csv",
        "row_alignment_report.json",
        "leakage_report.json",
    )
    _require(path, required)
    contract = _json(path / "physical_multiscale_contract.json")
    validate_contract_document(contract, schema_path=schema_path)
    alignment = _json(path / "row_alignment_report.json")
    leakage = _json(path / "leakage_report.json")
    rows = _csv(path / "physical_multiscale_manifest.csv")
    resolution_rows = _csv(path / "resolution_audit.csv")
    sample_ids = [row["sample_id"] for row in rows]
    centers = tuple(sorted(set(row["center"] for row in rows), key=int))
    geometry_policy = contract.get("geometry_policy")
    if not isinstance(geometry_policy, Mapping):
        raise ValueError("Physical contract geometry policy must be a mapping.")
    expected_contract_hash = physical_contract_hash(
        canonical_cache_sha256=str(contract.get("canonical_cache_sha256", "")),
        manifest_rows=rows,
        fov_um=tuple(float(value) for value in contract.get("fov_um", ())),
        geometry_policy=geometry_policy,
    )
    resolution_by_path = {row["raw_tiff_path"]: row for row in resolution_rows}
    if len(resolution_by_path) != len(resolution_rows):
        raise ValueError("Physical resolution audit contains duplicate TIFF paths.")
    manifest_slide_hashes: dict[str, str] = {}
    for row in rows:
        raw_path = row["raw_tiff_path"]
        raw_hash = row["raw_tiff_sha256"]
        if raw_path in manifest_slide_hashes and manifest_slide_hashes[raw_path] != raw_hash:
            raise ValueError("Physical manifest assigns multiple hashes to one TIFF.")
        manifest_slide_hashes[raw_path] = raw_hash
        geometries = json.loads(row["scale_geometry_json"])
        if set(geometries) != {"28um", "56um", "112um"}:
            raise ValueError("Physical manifest has incomplete scale geometry.")
        if any(
            float(geometry["padding_fraction"])
            > float(geometry_policy["padding_fraction_max"])
            for geometry in geometries.values()
        ):
            raise ValueError("Physical manifest geometry exceeds padding policy.")
    if set(manifest_slide_hashes) != set(resolution_by_path):
        raise ValueError("Physical resolution audit does not cover exact manifest slides.")
    repo_root = Path(__file__).resolve().parents[4]
    for raw_path, expected_hash in manifest_slide_hashes.items():
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
            raise ValueError(f"Physical TIFF lineage failed validation: {raw_path}")
    if (
        contract.get("status") != "PASS"
        or alignment.get("status") != "PASS"
        or leakage.get("status") != "PASS"
        or len(rows) != int(contract.get("row_count", -1))
        or len(sample_ids) != len(set(sample_ids))
        or centers != MIDOGPP_ELIGIBLE_CENTERS
        or "4" in centers
        or contract.get("contract_hash") != expected_contract_hash
        or contract.get("slide_count") != len(manifest_slide_hashes)
        or alignment.get("row_count") != len(rows)
        or alignment.get("sample_id_order_hash") != stable_hash(sample_ids)
        or alignment.get("center_4_present") is not False
        or alignment.get("canonical_order_exact") is not True
        or leakage.get("split") != "train"
        or leakage.get("target_labels_used_for_geometry") is not False
        or leakage.get("target_metrics_used_for_geometry") is not False
        or leakage.get("post_label_row_exclusion") is not False
        or any(row.get("split") != "train" for row in rows)
    ):
        raise ValueError("Physical multiscale contract bundle failed validation.")
    return {"status": "PASS", "row_count": len(rows), "centers": list(centers)}


def validate_contract_document(
    contract: Mapping[str, object],
    *,
    schema_path: str | Path = DEFAULT_CONTRACT_SCHEMA,
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
            f"Physical multiscale contract schema failed at {location}: {exc.message}"
        ) from exc


def _require(root: Path, relatives: tuple[str, ...]) -> None:
    missing = [relative for relative in relatives if not (root / relative).is_file()]
    if missing:
        raise ValueError(f"Physical multiscale bundle is missing files: {missing}")


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
