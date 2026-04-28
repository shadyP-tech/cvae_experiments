from __future__ import annotations

import csv
from pathlib import Path
import random
from typing import Dict, Iterable, List, Optional, Tuple

from src.data.datasets.breakhis import BreakHisRecord, cap_samples_per_domain, leakage_report
from src.data.shared_split import image_level_split_indices, split_groups


def _find_col(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    lowered = {name.lower(): name for name in fieldnames}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    return None


def _parse_split(value: str) -> Optional[str]:
    raw = str(value).strip().lower()
    if raw in {"train", "tr", "0"}:
        return "train"
    if raw in {"val", "valid", "validation", "1"}:
        return "val"
    if raw in {"test", "te", "2", "3"}:
        return "test"
    return None


def _assign_split_site_group(
    records: List[BreakHisRecord],
    split: Dict[str, float],
    seed: int,
    require_patient_ids: bool,
) -> Tuple[List[BreakHisRecord], Dict[str, str]]:
    rng = random.Random(seed)
    limitations: Dict[str, str] = {}

    by_domain: Dict[int, List[BreakHisRecord]] = {}
    for rec in records:
        by_domain.setdefault(int(rec.magnification), []).append(rec)

    out: List[BreakHisRecord] = []
    for domain, domain_records in by_domain.items():
        with_group = [r for r in domain_records if r.patient_id is not None]
        without_group = [r for r in domain_records if r.patient_id is None]

        if require_patient_ids and without_group:
            raise ValueError(f"Missing grouping IDs for domain {domain} while require_patient_ids=true")

        if with_group:
            by_group: Dict[str, List[BreakHisRecord]] = {}
            for rec in with_group:
                by_group.setdefault(rec.patient_id or "", []).append(rec)

            # Coarse class stratification by group label.
            group_label = {gid: recs[0].label for gid, recs in by_group.items()}
            neg_ids = [gid for gid, lbl in group_label.items() if lbl == 0]
            pos_ids = [gid for gid, lbl in group_label.items() if lbl == 1]

            neg_split = split_groups(neg_ids, split, rng)
            pos_split = split_groups(pos_ids, split, rng)

            group_to_split: Dict[str, str] = {}
            for s in ["train", "val", "test"]:
                for gid in neg_split[s] + pos_split[s]:
                    group_to_split[gid] = s

            for gid, recs in by_group.items():
                s = group_to_split[gid]
                for rec in recs:
                    out.append(BreakHisRecord(**{**rec.__dict__, "split": s}))

        if without_group:
            limitations[f"domain_{domain}"] = "Some grouping IDs unavailable; image-level split used."
            split_map = image_level_split_indices(len(without_group), split=split, rng=rng)
            for s, arr in split_map.items():
                for i in arr:
                    rec = without_group[i]
                    out.append(BreakHisRecord(**{**rec.__dict__, "split": s}))

    return out, limitations


def _resolve_image_path_from_row(root: Path, row: Dict[str, str], path_col: Optional[str]) -> Path:
    if path_col is not None and row.get(path_col):
        p = Path(row[path_col])
        return p if p.is_absolute() else root / p

    # Common Camelyon17-WILDS metadata columns.
    patient = row.get("patient")
    node = row.get("node")
    x_coord = row.get("x_coord")
    y_coord = row.get("y_coord")
    if all(v is not None and str(v).strip() != "" for v in [patient, node, x_coord, y_coord]):
        patient_int = int(patient)
        node_int = int(node)
        x_int = int(x_coord)
        y_int = int(y_coord)
        folder = root / "patches" / f"patient_{patient_int:03d}_node_{node_int}"
        name = f"patch_patient_{patient_int:03d}_node_{node_int}_x_{x_int}_y_{y_int}.png"
        return folder / name

    raise ValueError(
        "Could not resolve image path from metadata row. "
        "Provide a path-like column (image_path/path/filepath/filename) or WILDS columns "
        "(patient,node,x_coord,y_coord)."
    )


def prepare_camelyon17_records(
    root: Path,
    extensions: Iterable[str],
    split: Dict[str, float],
    cap_per_domain: int | None,
    seed: int,
    require_patient_ids: bool,
    domain_field: str = "center",
    metadata_file: str = "metadata.csv",
    use_metadata_split: bool = False,
    split_domain_caps: Dict[str, int] | None = None,
) -> Tuple[List[BreakHisRecord], Dict[str, object]]:
    metadata_path = root / metadata_file
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Camelyon17 metadata not found at {metadata_path}. "
            "Set data.metadata_file in config or place metadata.csv under dataset root."
        )

    with metadata_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty metadata file: {metadata_path}")
        fieldnames = list(reader.fieldnames)

        label_col = _find_col(fieldnames, ["label", "tumor", "target", "y"])
        patient_col = _find_col(fieldnames, ["patient", "patient_id", "case_id"])
        slide_col = _find_col(fieldnames, ["slide", "slide_id", "slide_index"])
        domain_col = _find_col(fieldnames, [domain_field, "center", "hospital", "site", "domain"])
        split_col = _find_col(fieldnames, ["split", "fold"])
        path_col = _find_col(fieldnames, ["image_path", "filepath", "path", "file", "filename"])

        if label_col is None:
            raise ValueError("Could not find label column in Camelyon17 metadata.")
        if domain_col is None:
            raise ValueError("Could not find domain column in Camelyon17 metadata.")

        ext_set = {e.lower() for e in extensions}
        records: List[BreakHisRecord] = []
        for idx, row in enumerate(reader):
            image_path = _resolve_image_path_from_row(root, row, path_col)
            if image_path.suffix.lower() not in ext_set:
                continue
            if not image_path.exists():
                continue

            label = int(float(row[label_col]))
            domain_value = int(float(row[domain_col]))
            patient_raw = row.get(patient_col) if patient_col is not None else None
            slide_raw = row.get(slide_col) if slide_col is not None else None
            patient_raw = patient_raw.strip() if patient_raw is not None else None
            slide_raw = slide_raw.strip() if slide_raw is not None else None

            # Prefer slide-level grouping when slide ID is available to avoid slide leakage.
            if patient_raw is not None and patient_raw != "" and slide_raw is not None and slide_raw != "":
                group_id = f"patient_{patient_raw}__slide_{slide_raw}"
            elif patient_raw is not None and patient_raw != "":
                group_id = f"patient_{patient_raw}"
            else:
                group_id = None

            assigned_split = _parse_split(row.get(split_col, "")) if split_col else None
            sample_id = image_path.stem if image_path.stem else f"cam17_{idx}"

            records.append(
                BreakHisRecord(
                    sample_id=sample_id,
                    image_path=str(image_path),
                    label=label,
                    label_name="tumor" if label == 1 else "normal",
                    magnification=domain_value,
                    domain_name=f"center_{domain_value}",
                    patient_id=group_id,
                    split=assigned_split or "",
                )
            )

    # Optional: honor dataset-provided split if explicitly requested.
    if use_metadata_split and records and all(r.split in {"train", "val", "test"} for r in records):
        split_records = records
        limitations: Dict[str, str] = {}
    else:
        split_records, limitations = _assign_split_site_group(
            records=records,
            split=split,
            seed=seed,
            require_patient_ids=require_patient_ids,
        )

    capped, cap_report = cap_samples_per_domain(
        split_records,
        cap_per_domain=cap_per_domain,
        seed=seed,
        split_domain_caps=split_domain_caps,
    )
    report = leakage_report(capped)
    report["limitations"] = limitations
    report["metadata_file"] = str(metadata_path)
    report["split_cap_accounting"] = cap_report
    return capped, report
