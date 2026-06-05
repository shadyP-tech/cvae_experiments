from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .builder import MANIFEST_COLUMNS
from .reporting import read_csv_rows, read_json


SAIL_COMPATIBILITY_COLUMNS = ("sample_id", "image_path", "label", "split", "center", "magnification")
REQUIRED_ARTIFACT_FILES = (
    "manifest.csv",
    "domain_mapping.json",
    "split_manifest.csv",
    "domain_feasibility.csv",
    "class_balance_by_domain.csv",
    "leakage_report.json",
    "dataset_contract.json",
)


class ValidationError(RuntimeError):
    pass


def validate_contract(
    artifact_root: str | Path,
    *,
    schema_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(artifact_root)
    repo = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    _require_files(root, REQUIRED_ARTIFACT_FILES)

    manifest_rows, manifest_fields = read_csv_rows(root / "manifest.csv")
    feasibility_rows, feasibility_fields = read_csv_rows(root / "domain_feasibility.csv")
    split_rows, _split_fields = read_csv_rows(root / "split_manifest.csv")
    contract = read_json(root / "dataset_contract.json")
    domain_mapping = read_json(root / "domain_mapping.json")
    leakage_report = read_json(root / "leakage_report.json")

    _validate_schema(contract, schema_path=schema_path, repo_root=repo)
    _require_columns(manifest_fields, MANIFEST_COLUMNS, "manifest.csv")
    _require_columns(manifest_fields, SAIL_COMPATIBILITY_COLUMNS, "manifest.csv")
    _require_columns(feasibility_fields, ("domain_axis", "domain_name", "eligible", "ineligible_reasons"), "domain_feasibility.csv")

    errors: list[str] = []
    errors.extend(_manifest_errors(manifest_rows, repo_root=repo))
    errors.extend(_split_overlap_errors(manifest_rows))
    errors.extend(_split_manifest_errors(split_rows))
    errors.extend(_domain_mapping_errors(manifest_rows, domain_mapping))
    errors.extend(_feasibility_errors(feasibility_rows, contract))

    if str(leakage_report.get("status", "")).upper() != "PASS":
        errors.append("leakage_report.status is not PASS")
    if contract.get("status") != "pass":
        errors.append(f"dataset_contract.status is {contract.get('status')!r}, expected 'pass'")
    domain_policy = contract.get("domain_policy", {}) if isinstance(contract.get("domain_policy"), Mapping) else {}
    if not bool(domain_policy.get("final_axis_frozen", False)):
        errors.append("selected composite domain axis is not frozen")

    if errors:
        raise ValidationError("; ".join(errors))

    return {
        "status": "PASS",
        "artifact_root": str(root),
        "manifest_rows": len(manifest_rows),
        "domain_axis": domain_policy.get("selected_domain_axis", ""),
        "eligible_domain_count": domain_policy.get("eligible_domain_count", 0),
    }


def _require_files(root: Path, filenames: Sequence[str]) -> None:
    missing = [name for name in filenames if not (root / name).exists()]
    if missing:
        raise ValidationError(f"Missing required artifact files: {missing}")


def _require_columns(found: Sequence[str], required: Sequence[str], filename: str) -> None:
    missing = sorted(set(required).difference(found))
    if missing:
        raise ValidationError(f"{filename} missing required columns: {missing}")


def _validate_schema(contract: Mapping[str, Any], *, schema_path: str | Path | None, repo_root: Path) -> None:
    path = Path(schema_path) if schema_path is not None else repo_root / "datasets/midogpp/schemas/dataset_contract.schema.json"
    if not path.exists():
        return
    schema = json.loads(path.read_text(encoding="utf-8"))
    required = schema.get("required", [])
    if isinstance(required, list):
        missing = [key for key in required if key not in contract]
        if missing:
            raise ValidationError(f"dataset_contract.json missing schema-required keys: {missing}")


def _manifest_errors(rows: Sequence[Mapping[str, str]], *, repo_root: Path) -> list[str]:
    errors: list[str] = []
    sample_ids = [row.get("sample_id", "") for row in rows]
    duplicates = sorted({sid for sid in sample_ids if sample_ids.count(sid) > 1})
    if duplicates:
        errors.append(f"duplicate sample_id values: {duplicates[:5]}")
    if not rows:
        errors.append("manifest.csv has no rows")

    split_counts = {"train": 0, "val": 0, "test": 0}
    for idx, row in enumerate(rows):
        image_path = Path(str(row.get("image_path", "")))
        if image_path.is_absolute():
            errors.append(f"row {idx} image_path is absolute")
        elif not (repo_root / image_path).exists():
            errors.append(f"row {idx} image_path does not exist relative to repo root: {image_path}")
        split = str(row.get("split", "")).strip()
        if split not in split_counts:
            errors.append(f"row {idx} invalid split={split!r}")
        else:
            split_counts[split] += 1
        try:
            label = int(float(str(row.get("label", ""))))
        except Exception:
            errors.append(f"row {idx} label is not numeric")
            label = -1
        if label not in {0, 1}:
            errors.append(f"row {idx} label must be 0 or 1")
        domain_id = str(row.get("domain_id", "")).strip()
        if not domain_id:
            errors.append(f"row {idx} missing domain_id")
        if str(row.get("center", "")).strip() != domain_id:
            errors.append(f"row {idx} center must equal domain_id")
        if str(row.get("magnification", "")).strip() != domain_id:
            errors.append(f"row {idx} magnification must equal domain_id")
        if label == 0 and str(row.get("negative_match_scope", "")).strip() not in {"same_case", "same_domain_same_split"}:
            errors.append(f"row {idx} negative row has invalid negative_match_scope")
    for split, count in split_counts.items():
        if count == 0:
            errors.append(f"manifest.csv has no rows for split={split}")
    return errors


def _split_overlap_errors(rows: Sequence[Mapping[str, str]]) -> list[str]:
    case_to_splits: dict[str, set[str]] = {}
    for row in rows:
        case_to_splits.setdefault(str(row.get("case_id", "")), set()).add(str(row.get("split", "")))
    overlapping = {case_id: sorted(splits) for case_id, splits in case_to_splits.items() if len(splits) > 1}
    if overlapping:
        return [f"case_id overlap across splits: {dict(list(overlapping.items())[:5])}"]
    return []


def _split_manifest_errors(rows: Sequence[Mapping[str, str]]) -> list[str]:
    case_to_splits: dict[str, set[str]] = {}
    for row in rows:
        case_to_splits.setdefault(str(row.get("case_id", "")), set()).add(str(row.get("split", "")))
    overlapping = {case_id: sorted(splits) for case_id, splits in case_to_splits.items() if len(splits) > 1}
    if overlapping:
        return [f"split_manifest.csv has repeated case_id across splits: {dict(list(overlapping.items())[:5])}"]
    return []


def _domain_mapping_errors(rows: Sequence[Mapping[str, str]], mapping: Mapping[str, Any]) -> list[str]:
    domains = mapping.get("domains", [])
    if not isinstance(domains, list):
        return ["domain_mapping.json domains must be a list"]
    mapped_ids = {str(row.get("domain_id", "")) for row in domains if isinstance(row, Mapping)}
    manifest_ids = {str(row.get("domain_id", "")) for row in rows}
    missing = sorted(manifest_ids.difference(mapped_ids))
    if missing:
        return [f"manifest domain_ids missing from domain_mapping.json: {missing}"]
    return []


def _feasibility_errors(rows: Sequence[Mapping[str, str]], contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    domain_policy = contract.get("domain_policy", {}) if isinstance(contract.get("domain_policy"), Mapping) else {}
    candidate_axes = [str(axis) for axis in domain_policy.get("candidate_axes", [])]
    reported_axes = {str(row.get("domain_axis", "")) for row in rows}
    missing_axes = sorted(set(candidate_axes).difference(reported_axes))
    if missing_axes:
        errors.append(f"domain_feasibility.csv missing candidate axes: {missing_axes}")
    for idx, row in enumerate(rows):
        eligible = str(row.get("eligible", "")).strip().lower() in {"true", "1", "yes"}
        reasons = str(row.get("ineligible_reasons", "")).strip()
        if not eligible and not reasons:
            errors.append(f"domain_feasibility.csv row {idx} is ineligible without explicit reasons")
    selected_axis = str(domain_policy.get("selected_domain_axis", ""))
    selected_rows = [row for row in rows if str(row.get("domain_axis", "")) == selected_axis]
    eligible_count = sum(1 for row in selected_rows if str(row.get("eligible", "")).strip().lower() in {"true", "1", "yes"})
    if int(domain_policy.get("eligible_domain_count", -1)) != eligible_count:
        errors.append("domain_policy.eligible_domain_count does not match domain_feasibility.csv")
    return errors
