#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SUPPORT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SUPPORT_ROOT.parent
PROJECT_ROOT = REPO_ROOT / "cvae_testing"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.load_config import load_config
from src.config.schema import validate_config
from src.data.registry import prepare_dataset_records


CONF_THRESHOLD = 0.70
SPECIES_THRESHOLD = 0.90
FULL_MIN_GROUPS = 48
PREFERRED_MIN_GROUPS = 64
MIN_EVAL_GROUPS = 16


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if str(key) not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _counts(values: Iterable[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        text = str(value or "").strip() or "missing"
        out[text] = out.get(text, 0) + 1
    return out


def _dominant_fraction(values: Iterable[str]) -> float:
    counts = _counts(values)
    total = sum(counts.values())
    return float(max(counts.values()) / total) if total else 0.0


def _effective_count(values: Iterable[str]) -> float:
    counts = _counts(values)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = [float(count) / float(total) for count in counts.values()]
    denom = sum(p * p for p in probs)
    return float(1.0 / denom) if denom > 0 else 0.0


def _rec_value(rec: object, field: str) -> str:
    return str(getattr(rec, field, "") or "").strip()


def build_confounding_rows(records: Sequence[object]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, str, str, str, str], List[object]] = {}
    for rec in records:
        key = (
            str(getattr(rec, "magnification")),
            _rec_value(rec, "tumor_type") or "missing",
            _rec_value(rec, "lab_or_origin") or "missing",
            _rec_value(rec, "species") or "missing",
            _rec_value(rec, "resolution_bin") or _rec_value(rec, "resolution") or "missing",
        )
        groups.setdefault(key, []).append(rec)
    rows: List[Dict[str, Any]] = []
    for (domain, tumor, lab, species, resolution), recs in sorted(groups.items(), key=lambda item: item[0]):
        rows.append(
            {
                "scanner_domain": int(domain),
                "scanner_model": _rec_value(recs[0], "scanner_model") or _rec_value(recs[0], "domain_name"),
                "tumor_type": tumor,
                "lab_or_origin": lab,
                "species": species,
                "resolution": resolution,
                "n_cases": int(len({str(getattr(rec, "patient_id", "")) for rec in recs})),
                "n_groups": int(len({str(getattr(rec, "patient_id", "")) for rec in recs})),
                "n_images": int(len(recs)),
            }
        )
    return rows


def build_fold_rows(records: Sequence[object], support_sizes: Sequence[int]) -> List[Dict[str, Any]]:
    domains = sorted({int(getattr(rec, "magnification")) for rec in records})
    max_support = max(int(v) for v in support_sizes)
    rows: List[Dict[str, Any]] = []
    for domain in domains:
        target = [rec for rec in records if int(getattr(rec, "magnification")) == int(domain)]
        groups = {str(getattr(rec, "patient_id", "")) for rec in target}
        n_groups = len(groups)
        feasible_sizes = [
            int(size)
            for size in support_sizes
            if int(n_groups) >= int(size) + int(MIN_EVAL_GROUPS)
        ]
        tumor_values = [_rec_value(rec, "tumor_type") for rec in target]
        lab_values = [_rec_value(rec, "lab_or_origin") for rec in target]
        species_values = [_rec_value(rec, "species") for rec in target]
        resolution_values = [_rec_value(rec, "resolution_bin") or _rec_value(rec, "resolution") for rec in target]
        dominant_tumor = _dominant_fraction(tumor_values)
        dominant_lab = _dominant_fraction(lab_values)
        dominant_species = _dominant_fraction(species_values)
        missing_conf = any(
            not value
            for values in [tumor_values, lab_values, species_values, resolution_values]
            for value in values
        )
        full_feasible = int(n_groups >= FULL_MIN_GROUPS)
        if not full_feasible or missing_conf:
            classification = "invalid for scanner-specific claim"
        elif dominant_tumor > CONF_THRESHOLD or dominant_lab > CONF_THRESHOLD or dominant_species > SPECIES_THRESHOLD:
            classification = "mixed scanner/tumor/lab fold"
        else:
            classification = "clean-ish scanner fold"
        rows.append(
            {
                "heldout_scanner": int(domain),
                "heldout_scanner_model": _rec_value(target[0], "scanner_model") or _rec_value(target[0], "domain_name"),
                "source_scanners": "|".join(str(d) for d in domains if int(d) != int(domain)),
                "candidate_expert_count": int(max(len(domains) - 1, 0)),
                "target_group_count": int(n_groups),
                "support_feasible_sizes": "|".join(str(size) for size in feasible_sizes),
                "eval_group_count_after_support": int(max(n_groups - max_support, 0)),
                "dominant_tumor_fraction": dominant_tumor,
                "dominant_lab_fraction": dominant_lab,
                "dominant_species_fraction": dominant_species,
                "tumor_effective_count": _effective_count(tumor_values),
                "lab_effective_count": _effective_count(lab_values),
                "species_effective_count": _effective_count(species_values),
                "resolution_effective_count": _effective_count(resolution_values),
                "group_feasibility": "preferred" if n_groups >= PREFERRED_MIN_GROUPS else ("feasible_high_variance" if full_feasible else "smoke_only"),
                "fold_classification": classification,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight MIDOG++ scanner-model support-NELBO stress test.")
    parser.add_argument(
        "--config",
        type=Path,
        default=SUPPORT_ROOT
        / "configs"
        / "experiments"
        / "midogpp"
        / "midogpp_scanner_support_estimated_utility_routing_v1.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=SUPPORT_ROOT / "artifacts" / "comparison_tables")
    parser.add_argument("--output-prefix", default="midogpp_scanner_support_estimated_utility_routing_v1")
    parser.add_argument("--require-full-feasible", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)
    records, leakage = prepare_dataset_records(PROJECT_ROOT, cfg)
    support_sizes = [
        int(v)
        for v in cfg["learned_utility"]["support_response_routing"].get("support_sizes", [4, 8, 16, 32])
    ]

    confounding_rows = build_confounding_rows(records)
    fold_rows = build_fold_rows(records, support_sizes)
    preflight = dict(leakage.get("midogpp_preflight", {}))
    invalid_folds = [row for row in fold_rows if str(row["fold_classification"]).startswith("invalid")]
    full_feasible = bool(fold_rows) and not invalid_folds and all(
        int(row["target_group_count"]) >= FULL_MIN_GROUPS for row in fold_rows
    )
    status = "pass" if full_feasible else "diagnostic_only"
    payload = {
        "status": status,
        "thesis_facing": bool(full_feasible),
        "dataset_type": "midogpp",
        "domain_axis_used": "scanner_model",
        "n_records": int(len(records)),
        "n_domains": int(preflight.get("n_domains", 0)),
        "domain_id_source": str(preflight.get("domain_id_source", "")),
        "domain_id_to_raw_scanner_label": preflight.get("domain_id_to_raw_scanner_label", {}),
        "n_cases_per_domain": preflight.get("n_cases_per_domain", {}),
        "n_groups_per_domain": preflight.get("n_groups_per_domain", {}),
        "full_min_groups": FULL_MIN_GROUPS,
        "preferred_min_groups": PREFERRED_MIN_GROUPS,
        "min_eval_groups": MIN_EVAL_GROUPS,
        "invalid_fold_count": int(len(invalid_folds)),
        "folds": fold_rows,
        "leakage_report": leakage,
    }

    output_dir = args.output_dir
    prefix = str(args.output_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{prefix}_preflight.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / f"{prefix}_scanner_confounding_table.csv", confounding_rows)
    _write_csv(output_dir / f"{prefix}_scanner_fold_feasibility.csv", fold_rows)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_full_feasible and not full_feasible:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
