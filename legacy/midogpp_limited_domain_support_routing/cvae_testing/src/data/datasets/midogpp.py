from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from src.data.datasets.breakhis import BreakHisRecord, cap_samples_per_domain, leakage_report
from src.data.shared_split import image_level_split_indices, split_groups


@dataclass(frozen=True)
class MidogPPRecord(BreakHisRecord):
    scanner_model: str = ""
    scanner_vendor: str = ""
    scanner_family: str = ""
    lab_or_origin: str = ""
    tumor_type: str = ""
    species: str = ""
    stain_or_preparation: str = ""
    resolution: str = ""
    resolution_bin: str = ""
    midogpp_domain_axis: str = "scanner_model"
    domain_axis_used: str = "scanner_model"
    raw_scanner_label: str = ""
    domain_id_source: str = ""


def _find_col(fieldnames: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    by_lower = {str(name).strip().lower(): str(name) for name in fieldnames}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in by_lower:
            return by_lower[key]
    return None


def _clean(value: object) -> str:
    text = str(value if value is not None else "").strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _value(row: Mapping[str, Any], fieldnames: Iterable[str], candidates: Iterable[str]) -> str:
    col = _find_col(fieldnames, candidates)
    return _clean(row.get(col, "")) if col is not None else ""


def _parse_split(value: object) -> Optional[str]:
    raw = _clean(value).lower()
    if raw in {"train", "tr", "training", "0"}:
        return "train"
    if raw in {"val", "valid", "validation", "1"}:
        return "val"
    if raw in {"test", "te", "testing", "2", "3"}:
        return "test"
    return None


def _label_from_row(row: Mapping[str, Any], fieldnames: Iterable[str]) -> Tuple[int, str]:
    label_col = _find_col(fieldnames, ["label", "target", "y", "class", "has_mitosis", "is_mitotic"])
    if label_col is None:
        return 0, "unknown"
    raw = _clean(row.get(label_col, ""))
    if not raw:
        return 0, "unknown"
    lowered = raw.lower()
    if lowered in {"mitotic", "mitosis", "positive", "true", "yes"}:
        return 1, raw
    if lowered in {"non-mitotic", "non_mitotic", "negative", "false", "no"}:
        return 0, raw
    try:
        value = int(float(raw))
    except ValueError:
        return 0, raw
    return value, str(value)


def _resolution_bin(raw_resolution: str) -> str:
    text = _clean(raw_resolution).lower()
    if not text:
        return ""
    numeric = ""
    for char in text:
        if char.isdigit() or char == ".":
            numeric += char
        elif numeric:
            break
    if numeric:
        try:
            value = float(numeric)
        except ValueError:
            value = -1.0
        if value > 0:
            if value <= 0.30:
                return "<=0.30"
            if value <= 0.50:
                return "0.31-0.50"
            return ">0.50"
    return text


def _scanner_vendor(scanner_model: str, fallback: str = "") -> str:
    vendor = _clean(fallback)
    if vendor:
        return vendor
    lowered = scanner_model.lower()
    for candidate in ["hamamatsu", "leica", "aperio", "3dhistech", "philips", "zeiss"]:
        if candidate in lowered:
            return candidate
    return ""


def _candidate_image_paths(root: Path, raw_path: str, image_id: str, extensions: Iterable[str]) -> List[Path]:
    candidates: List[Path] = []
    if raw_path:
        p = Path(raw_path)
        candidates.append(p if p.is_absolute() else root / p)
        candidates.append(root / "images" / raw_path)
    if image_id:
        stem = Path(image_id).stem
        for ext in extensions:
            candidates.append(root / f"{stem}{ext}")
            candidates.append(root / "images" / f"{stem}{ext}")
    return candidates


def _resolve_image_path(root: Path, raw_path: str, image_id: str, extensions: Iterable[str]) -> Optional[Path]:
    ext_set = {str(ext).lower() for ext in extensions}
    for candidate in _candidate_image_paths(root, raw_path, image_id, extensions):
        if candidate.suffix.lower() not in ext_set:
            continue
        if candidate.exists():
            return candidate
    return None


def _read_csv_rows(metadata_path: Path) -> List[Dict[str, Any]]:
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty MIDOG++ metadata file: {metadata_path}")
        return [dict(row) for row in reader]


def _database_lookup(payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw = (
        payload.get("databases")
        or payload.get("database")
        or payload.get("datasets")
        or payload.get("domains")
    )
    lookup: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, Mapping):
        iterable = raw.items()
        for key, value in iterable:
            if isinstance(value, Mapping):
                row = dict(value)
            else:
                row = {"database_label": value}
            row.setdefault("database_id", key)
            for alias in [key, row.get("id"), row.get("database_id"), row.get("name"), row.get("domain")]:
                text = _clean(alias)
                if text:
                    lookup[text] = row
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            row = dict(item)
            for alias in [
                row.get("id"),
                row.get("database_id"),
                row.get("dataset_id"),
                row.get("domain_id"),
                row.get("name"),
                row.get("domain"),
            ]:
                text = _clean(alias)
                if text:
                    lookup[text] = row
    return lookup


def _merge_image_database_metadata(payload: Mapping[str, Any], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup = _database_lookup(payload)
    if not lookup:
        return rows
    merged_rows: List[Dict[str, Any]] = []
    link_keys = [
        "database_id",
        "database",
        "dataset_id",
        "domain_id",
        "domain",
        "source_database",
    ]
    for row in rows:
        database_row: Dict[str, Any] = {}
        for key in link_keys:
            value = _clean(row.get(key, ""))
            if value and value in lookup:
                database_row = lookup[value]
                break
        merged_rows.append({**database_row, **row} if database_row else row)
    return merged_rows


def _read_json_rows(metadata_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        raise ValueError(f"Unsupported MIDOG++ JSON metadata shape: {metadata_path}")
    images = payload.get("images")
    if isinstance(images, list):
        rows = [dict(row) for row in images if isinstance(row, Mapping)]
        return _merge_image_database_metadata(payload, rows)
    rows = payload.get("rows") or payload.get("records") or payload.get("data")
    if isinstance(rows, list):
        parsed = [dict(row) for row in rows if isinstance(row, Mapping)]
        return _merge_image_database_metadata(payload, parsed)
    raise ValueError(
        f"Could not find MIDOG++ rows in JSON metadata {metadata_path}; expected images/rows/records/data"
    )


def _read_metadata_rows(metadata_path: Path) -> List[Dict[str, Any]]:
    suffix = metadata_path.suffix.lower()
    if suffix == ".json":
        return _read_json_rows(metadata_path)
    if suffix == ".csv":
        return _read_csv_rows(metadata_path)
    raise ValueError(f"Unsupported MIDOG++ metadata file extension: {metadata_path.suffix}")


def _resolve_metadata_path(root: Path, metadata_file: str | None) -> Path:
    candidates = []
    if metadata_file:
        p = Path(metadata_file)
        candidates.append(p if p.is_absolute() else root / p)
    candidates.extend(
        [
            root / "metadata.csv",
            root / "MIDOG++.csv",
            root / "databases" / "MIDOG++.json",
            root / "MIDOG++.json",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"MIDOG++ metadata file not found. Tried: {tried}")


def _build_records_from_rows(
    *,
    root: Path,
    metadata_path: Path,
    rows: List[Dict[str, Any]],
    extensions: Iterable[str],
    domain_axis: str,
    require_group_ids: bool,
) -> Tuple[List[MidogPPRecord], Dict[str, Any]]:
    if domain_axis != "scanner_model":
        raise ValueError("MIDOG++ v1 supports only data.midogpp_domain_axis='scanner_model'")
    if not rows:
        raise ValueError(f"No MIDOG++ metadata rows found in {metadata_path}")

    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    scanner_col = _find_col(
        fieldnames,
        [
            "scanner_model",
            "scanner",
            "scanner_name",
            "scanner_type",
            "scan_device",
            "digitization_device",
            "device",
        ],
    )
    if scanner_col is None:
        raise ValueError("MIDOG++ metadata must contain a scanner model column for scanner_model axis")

    raw_scanners = sorted({_clean(row.get(scanner_col, "")) for row in rows if _clean(row.get(scanner_col, ""))})
    if len(raw_scanners) < 3:
        raise ValueError(f"MIDOG++ scanner_model axis requires at least 3 scanner domains, got {len(raw_scanners)}")
    scanner_to_domain = {scanner: idx for idx, scanner in enumerate(raw_scanners)}

    path_col = _find_col(fieldnames, ["image_path", "filepath", "path", "file", "filename", "file_name"])
    id_col = _find_col(fieldnames, ["sample_id", "image_id", "id", "case_id", "slide_id"])
    group_col = _find_col(fieldnames, ["case_id", "case", "specimen_id", "slide_id", "patient_id", "image_id", "id"])
    split_col = _find_col(fieldnames, ["split", "fold", "set"])

    missing_scanner = 0
    missing_group = 0
    skipped_missing_image = 0
    records: List[MidogPPRecord] = []
    for row_idx, row in enumerate(rows):
        scanner_model = _clean(row.get(scanner_col, ""))
        if not scanner_model:
            missing_scanner += 1
            continue
        group_id = _clean(row.get(group_col, "")) if group_col is not None else ""
        if not group_id:
            missing_group += 1
            if require_group_ids:
                continue
        raw_path = _clean(row.get(path_col, "")) if path_col is not None else ""
        image_id = _clean(row.get(id_col, "")) if id_col is not None else f"midogpp_{row_idx}"
        image_path = _resolve_image_path(root, raw_path, image_id, extensions)
        if image_path is None:
            skipped_missing_image += 1
            continue

        label, label_name = _label_from_row(row, fieldnames)
        vendor = _scanner_vendor(
            scanner_model,
            _value(row, fieldnames, ["scanner_vendor", "vendor", "manufacturer"]),
        )
        scanner_family = _value(row, fieldnames, ["scanner_family", "scanner_vendor_family", "scanner_type"])
        if not scanner_family:
            scanner_family = vendor
        resolution = _value(row, fieldnames, ["resolution", "mpp", "microns_per_pixel", "pixel_size"])
        domain_id = int(scanner_to_domain[scanner_model])
        records.append(
            MidogPPRecord(
                sample_id=Path(image_path).stem if not image_id else image_id,
                image_path=str(image_path),
                label=int(label),
                label_name=str(label_name),
                magnification=domain_id,
                domain_name=scanner_model,
                patient_id=group_id or None,
                split=_parse_split(row.get(split_col, "")) if split_col is not None else "",
                scanner_model=scanner_model,
                scanner_vendor=vendor,
                scanner_family=scanner_family,
                lab_or_origin=_value(row, fieldnames, ["lab_or_origin", "laboratory", "lab", "origin", "source"]),
                tumor_type=_value(row, fieldnames, ["tumor_type", "tumour_type", "cancer_type", "diagnosis", "entity"]),
                species=_value(row, fieldnames, ["species"]),
                stain_or_preparation=_value(row, fieldnames, ["stain", "staining", "preparation", "slide_preparation"]),
                resolution=resolution,
                resolution_bin=_resolution_bin(resolution),
                raw_scanner_label=scanner_model,
                domain_id_source=str(scanner_col),
            )
        )

    if missing_scanner:
        raise ValueError(f"MIDOG++ metadata has {missing_scanner} rows without scanner IDs")
    if require_group_ids and missing_group:
        raise ValueError(f"MIDOG++ metadata has {missing_group} rows without group IDs")
    if not records:
        raise RuntimeError(
            "No MIDOG++ images were found after metadata parsing. "
            f"metadata={metadata_path}, skipped_missing_image={skipped_missing_image}"
        )

    observed = sorted({int(rec.magnification) for rec in records})
    report = {
        "metadata_file": str(metadata_path),
        "domain_axis_used": "scanner_model",
        "domain_id_source": str(scanner_col),
        "domain_id_to_raw_scanner_label": {str(v): k for k, v in scanner_to_domain.items()},
        "n_domains": int(len(scanner_to_domain)),
        "n_cases_per_domain": {
            str(domain): int(sum(1 for rec in records if int(rec.magnification) == int(domain)))
            for domain in observed
        },
        "n_groups_per_domain": {
            str(domain): int(
                len({str(rec.patient_id) for rec in records if int(rec.magnification) == int(domain)})
            )
            for domain in observed
        },
        "skipped_missing_image_count": int(skipped_missing_image),
    }
    return records, report


def _assign_split_by_domain_groups(
    records: List[MidogPPRecord],
    split: Dict[str, float],
    seed: int,
    require_group_ids: bool,
) -> Tuple[List[MidogPPRecord], Dict[str, str]]:
    rng = random.Random(seed)
    limitations: Dict[str, str] = {}
    out: List[MidogPPRecord] = []
    by_domain: Dict[int, List[MidogPPRecord]] = {}
    for rec in records:
        by_domain.setdefault(int(rec.magnification), []).append(rec)

    for domain, domain_records in by_domain.items():
        with_group = [r for r in domain_records if r.patient_id is not None]
        without_group = [r for r in domain_records if r.patient_id is None]
        if require_group_ids and without_group:
            raise ValueError(f"Missing MIDOG++ group IDs for scanner domain {domain}")

        by_group: Dict[str, List[MidogPPRecord]] = {}
        for rec in with_group:
            by_group.setdefault(str(rec.patient_id), []).append(rec)
        if by_group:
            group_ids = sorted(by_group)
            group_splits = split_groups(group_ids, split, rng)
            group_to_split = {
                gid: split_name
                for split_name, ids in group_splits.items()
                for gid in ids
            }
            for gid, recs in by_group.items():
                split_name = group_to_split[gid]
                for rec in recs:
                    out.append(MidogPPRecord(**{**rec.__dict__, "split": split_name}))

        if without_group:
            limitations[f"domain_{domain}"] = "Some group IDs unavailable; image-level split used."
            split_map = image_level_split_indices(len(without_group), split=split, rng=rng)
            for split_name, indices in split_map.items():
                for idx in indices:
                    rec = without_group[int(idx)]
                    out.append(MidogPPRecord(**{**rec.__dict__, "split": split_name}))
    return out, limitations


def _assert_no_group_leakage(records: List[MidogPPRecord]) -> None:
    report = leakage_report(records)
    overlaps = report.get("patient_overlap", {})
    if isinstance(overlaps, Mapping) and any(overlaps.values()):
        raise ValueError(f"MIDOG++ group overlap across train/val/test detected: {overlaps}")


def _assert_domain_split_coverage(records: List[MidogPPRecord], *, stage: str) -> None:
    required = {"train", "val", "test"}
    by_domain: Dict[int, set[str]] = {}
    for rec in records:
        by_domain.setdefault(int(rec.magnification), set()).add(str(rec.split))
    missing = {
        int(domain): sorted(required - splits)
        for domain, splits in sorted(by_domain.items())
        if not required.issubset(splits)
    }
    if missing:
        raise ValueError(
            "MIDOG++ train/val/test must be represented for every scanner-model domain "
            f"after {stage}; missing={missing}"
        )


def prepare_midogpp_records(
    root: Path,
    extensions: Iterable[str],
    split: Dict[str, float],
    cap_per_domain: int | None,
    seed: int,
    require_patient_ids: bool,
    metadata_file: str | None = None,
    midogpp_domain_axis: str = "scanner_model",
    split_domain_caps: Dict[str, int] | None = None,
    configured_domains: Iterable[int] | None = None,
) -> Tuple[List[MidogPPRecord], Dict[str, object]]:
    metadata_path = _resolve_metadata_path(root, metadata_file)
    rows = _read_metadata_rows(metadata_path)
    records, preflight = _build_records_from_rows(
        root=root,
        metadata_path=metadata_path,
        rows=rows,
        extensions=extensions,
        domain_axis=str(midogpp_domain_axis).strip().lower(),
        require_group_ids=bool(require_patient_ids),
    )

    observed = sorted({int(rec.magnification) for rec in records})
    configured = sorted(int(v) for v in configured_domains or observed)
    if configured != observed:
        raise ValueError(f"MIDOG++ configured scanner domains {configured} do not match observed {observed}")

    if records and all(rec.split in {"train", "val", "test"} for rec in records):
        split_records = records
        limitations: Dict[str, str] = {}
    else:
        split_records, limitations = _assign_split_by_domain_groups(
            records=records,
            split=split,
            seed=int(seed),
            require_group_ids=bool(require_patient_ids),
        )
    _assert_domain_split_coverage(split_records, stage="split assignment")
    _assert_no_group_leakage(split_records)

    capped, cap_report = cap_samples_per_domain(
        split_records,
        cap_per_domain=cap_per_domain,
        seed=int(seed),
        split_domain_caps=split_domain_caps,
    )
    _assert_domain_split_coverage(capped, stage="sample capping")
    _assert_no_group_leakage(capped)
    report = leakage_report(capped)
    report["limitations"] = limitations
    report["split_cap_accounting"] = cap_report
    report["midogpp_preflight"] = preflight
    return capped, report
