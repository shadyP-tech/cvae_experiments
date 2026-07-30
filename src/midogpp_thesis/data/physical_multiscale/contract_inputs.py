"""Shared raw-source and canonical-row inputs for physical multiscale contracts."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from midogpp_thesis.data.features.cache_io import load_cache_rows


@dataclass(frozen=True)
class PhysicalContractInputs:
    canonical: Any
    manifest_rows: tuple[Mapping[str, str], ...]
    manifest_by_id: Mapping[str, Mapping[str, str]]
    source_by_case: Mapping[str, Mapping[str, object]]
    selected_metadata: tuple[Mapping[str, object], ...]


def load_contract_inputs(config: Any) -> PhysicalContractInputs:
    """Resolve the exact eligible-train cohort shared by v1/v2 builders."""

    canonical = load_cache_rows(config.canonical_cache_path, expected_dim=2560)
    manifest_rows = tuple(read_csv(config.base_manifest_path))
    assert_cache_manifest_identity(
        canonical.metadata,
        manifest_rows,
        eligible_centers=config.eligible_centers,
    )
    manifest_by_id = unique_by(manifest_rows, "sample_id", "base manifest")
    source_by_case = load_raw_sources(config.raw_metadata_path, config.raw_root)
    selected: list[Mapping[str, object]] = []
    for metadata in canonical.metadata:
        sample_id = str(metadata.get("sample_id", ""))
        manifest = manifest_by_id.get(sample_id)
        if manifest is None:
            raise ValueError(
                f"Canonical cache sample is absent from base manifest: {sample_id}"
            )
        center = str(metadata.get("center", manifest.get("center", "")))
        split = str(metadata.get("split", manifest.get("split", ""))).lower()
        if center == "4":
            continue
        if center not in config.eligible_centers or split != "train":
            raise ValueError(
                f"Canonical cache contains unexpected production row {sample_id}: "
                f"center={center}, split={split}"
            )
        selected.append(metadata)
    if not selected:
        raise ValueError("Physical contract selected no eligible canonical train rows.")
    return PhysicalContractInputs(
        canonical=canonical,
        manifest_rows=manifest_rows,
        manifest_by_id=manifest_by_id,
        source_by_case=source_by_case,
        selected_metadata=tuple(selected),
    )


def load_raw_sources(
    metadata_path: Path,
    raw_root: Path,
) -> dict[str, dict[str, object]]:
    rows = read_csv(metadata_path)
    by_case: dict[str, dict[str, object]] = {}
    duplicates: set[str] = set()
    for row in rows:
        image_ref = first(
            row,
            ("image_path", "file_name", "filename", "File", "Image", "image"),
        )
        if not image_ref:
            continue
        case_id = first(
            row,
            (
                "case_id",
                "specimen_id",
                "slide_id",
                "patient_id",
                "filename",
                "file_name",
                "image_path",
            ),
        )
        case_id = str(case_id or Path(image_ref).stem).strip()
        raw_path = resolve_raw_path(raw_root, image_ref)
        if case_id in by_case and Path(str(by_case[case_id]["raw_path"])) != raw_path:
            duplicates.add(case_id)
        by_case[case_id] = {"raw_path": raw_path, "metadata": dict(row)}
    if duplicates:
        raise ValueError(
            f"Raw metadata maps cases to multiple TIFFs: {sorted(duplicates)[:5]}"
        )
    return by_case


def assert_cache_manifest_identity(
    cache_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
    *,
    eligible_centers: Sequence[str],
) -> None:
    eligible = {str(center) for center in eligible_centers}
    cache_selected = {
        str(row.get("sample_id", "")): row
        for row in cache_rows
        if str(row.get("center", "")) in eligible
        and str(row.get("split", "")).lower() == "train"
    }
    manifest_selected = {
        str(row.get("sample_id", "")): row
        for row in manifest_rows
        if str(row.get("center", "")) in eligible
        and str(row.get("split", "")).lower() == "train"
    }
    if (
        not cache_selected
        or len(cache_selected)
        != sum(
            str(row.get("center", "")) in eligible
            and str(row.get("split", "")).lower() == "train"
            for row in cache_rows
        )
        or len(manifest_selected)
        != sum(
            str(row.get("center", "")) in eligible
            and str(row.get("split", "")).lower() == "train"
            for row in manifest_rows
        )
        or set(cache_selected) != set(manifest_selected)
    ):
        raise ValueError(
            "Canonical cache and base manifest eligible-train sample sets differ."
        )
    for sample_id, cache_row in cache_selected.items():
        manifest_row = manifest_selected[sample_id]
        cache_identity = (
            str(cache_row.get("case_id", "")),
            str(cache_row.get("center", "")),
            str(cache_row.get("split", "")).lower(),
            int(float(str(cache_row.get("label", -1)))),
        )
        manifest_identity = (
            str(manifest_row.get("case_id", "")),
            str(manifest_row.get("center", "")),
            str(manifest_row.get("split", "")).lower(),
            int(float(str(manifest_row.get("label", -1)))),
        )
        if cache_identity != manifest_identity:
            raise ValueError(
                f"Canonical cache/base manifest identity differs: {sample_id}"
            )


def assert_raw_source_identity(
    manifest: Mapping[str, object],
    source: Mapping[str, object],
) -> None:
    raw_metadata = source.get("metadata")
    if not isinstance(raw_metadata, Mapping):
        raise ValueError("Raw TIFF source metadata must be a mapping.")
    case_id = str(manifest.get("case_id", ""))
    raw_case_id = str(raw_metadata.get("case_id", ""))
    raw_path = Path(str(source.get("raw_path", "")))
    if not case_id or raw_case_id != case_id or raw_path.stem != case_id:
        raise ValueError(f"Raw TIFF case/slide identity differs for case {case_id!r}.")
    for field in ("scanner_model", "lab_or_origin", "tumor_type", "species"):
        manifest_value = str(manifest.get(field, "")).strip()
        raw_value = str(raw_metadata.get(field, "")).strip()
        if manifest_value and raw_value and manifest_value != raw_value:
            raise ValueError(
                f"Raw TIFF/base manifest {field} differs for case {case_id!r}."
            )


def resolve_raw_path(raw_root: Path, image_ref: str) -> Path:
    raw = Path(str(image_ref).strip())
    candidates = (raw, raw_root / raw, raw_root / "images" / raw, raw_root / "tiffs" / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"Raw TIFF referenced by metadata is missing: {image_ref!r}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def unique_by(
    rows: Sequence[Mapping[str, object]],
    key: str,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if not value or value in out:
            raise ValueError(f"{label} contains empty or duplicated {key}: {value!r}")
        out[value] = row
    return out


def first(row: Mapping[str, object], names: Sequence[str]) -> str:
    for name in names:
        value = str(row.get(name, "") or "").strip()
        if value:
            return value
    return ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())
