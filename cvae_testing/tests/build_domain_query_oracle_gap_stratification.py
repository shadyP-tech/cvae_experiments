#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from statistics import median
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    if not rows:
        raise RuntimeError(f"CSV has no rows: {path}")
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(clean) / len(clean)) if clean else 0.0


def _std(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return 0.0
    mu = _mean(clean)
    return float(math.sqrt(sum((v - mu) ** 2 for v in clean) / len(clean)))


def _quantile(values: Sequence[float], q: float) -> float:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    pos = float(q) * float(len(clean) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    weight = pos - lo
    return float((1.0 - weight) * clean[lo] + weight * clean[hi])


def _infer_label(row: Mapping[str, object]) -> Tuple[str, str]:
    label = str(row.get("label", "") or "").strip()
    label_name = str(row.get("label_name", "") or "").strip().lower()
    if label and label_name:
        return label, label_name

    text = f"{row.get('image_path', '')} {row.get('sample_id', '')}".lower()
    inferred_label = label
    inferred_name = label_name
    if not inferred_name:
        if "benign" in text or re.search(r"(^|[_\-\s])b([_\-\s]|$)", text):
            inferred_label = inferred_label or "0"
            inferred_name = "benign"
        elif "malignant" in text or re.search(r"(^|[_\-\s])m([_\-\s]|$)", text):
            inferred_label = inferred_label or "1"
            inferred_name = "malignant"
    if not inferred_name and inferred_label:
        inferred_name = "benign" if inferred_label == "0" else "malignant" if inferred_label == "1" else ""
    return inferred_label, inferred_name


def _infer_patient_id(row: Mapping[str, object]) -> str:
    patient = str(row.get("patient_id", "") or "").strip()
    if patient:
        return patient
    raw = str(row.get("sample_id", "") or "").strip()
    if not raw:
        raw = Path(str(row.get("image_path", "") or "")).stem
    stem = Path(raw).stem
    match = re.match(r"(.+?)-(40|100|200|400)-\d+$", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.match(r"(.+?)-\d+$", stem)
    if match:
        return match.group(1)
    return ""


def _enrich_sample_rows(rows: Sequence[dict]) -> List[dict]:
    enriched: List[dict] = []
    for row in rows:
        out = dict(row)
        label, label_name = _infer_label(out)
        patient_id = _infer_patient_id(out)
        out["label"] = label
        out["label_name"] = label_name
        out["patient_id"] = patient_id
        out["patient_available"] = int(bool(patient_id))
        enriched.append(out)
    return enriched


def _group_rows(rows: Iterable[Mapping[str, object]], keys: Sequence[str]) -> Dict[Tuple[str, ...], List[Mapping[str, object]]]:
    groups: Dict[Tuple[str, ...], List[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(str(row.get(k, "") or "") for k in keys)
        groups.setdefault(key, []).append(row)
    return groups


def _expert_distribution(rows: Sequence[Mapping[str, object]]) -> Tuple[str, float, float, int]:
    counts: Dict[str, int] = {}
    candidate_keys: set[str] = set()
    for row in rows:
        expert = str(row.get("per_query_oracle_expert", "") or "")
        if expert:
            counts[expert] = counts.get(expert, 0) + 1
        try:
            candidate_keys.update(str(k) for k in json.loads(str(row.get("nelbo_by_expert_json", "{}"))).keys())
        except Exception:
            pass

    total = int(sum(counts.values()))
    if total <= 0:
        return "{}", 0.0, 0.0, int(len(candidate_keys))
    modal_share = max(counts.values()) / float(total)
    switch_rate = 1.0 - modal_share
    probs = [count / float(total) for count in counts.values() if count > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    n_options = max(len(candidate_keys), len(counts))
    entropy_norm = entropy / math.log(n_options) if n_options > 1 else 0.0
    return json.dumps(counts, sort_keys=True, separators=(",", ":")), float(switch_rate), float(entropy_norm), int(n_options)


def _low_margin_share(
    rows: Sequence[Mapping[str, object]],
    *,
    abs_threshold: float,
    rel_threshold: float,
) -> float:
    flags: List[int] = []
    for row in rows:
        margin = _to_float(row.get("per_query_oracle_margin", 0.0))
        threshold = float(abs_threshold)
        try:
            values = [float(v) for v in json.loads(str(row.get("nelbo_by_expert_json", "{}"))).values()]
            if values:
                threshold = max(threshold, float(rel_threshold) * (max(values) - min(values)))
        except Exception:
            pass
        flags.append(int(margin <= threshold))
    return _mean(flags)


def _sample_summary_rows(
    rows: Sequence[Mapping[str, object]],
    keys: Sequence[str],
    *,
    low_margin_abs_threshold: float,
    low_margin_rel_threshold: float,
) -> List[dict]:
    out: List[dict] = []
    for key, vals in sorted(_group_rows(rows, keys).items()):
        gaps = [_to_float(v.get("fixed_to_query_sample_gap", 0.0)) for v in vals]
        margins = [_to_float(v.get("per_query_oracle_margin", 0.0)) for v in vals]
        ranks = [_to_float(v.get("fixed_domain_oracle_sample_rank", 0.0)) for v in vals]
        counts_json, switch_rate, entropy_norm, n_options = _expert_distribution(vals)
        row: Dict[str, object] = {k: value for k, value in zip(keys, key)}
        row.update(
            {
                "n_samples": int(len(vals)),
                "n_runs": int(len(set(str(v.get("run_id", "")) for v in vals))),
                "n_patients": int(len(set(str(v.get("patient_id", "")) for v in vals if str(v.get("patient_id", ""))))),
                "n_candidate_experts_observed": int(n_options),
                "fixed_to_query_sample_gap_mean": _mean(gaps),
                "fixed_to_query_sample_gap_std": _std(gaps),
                "fixed_to_query_sample_gap_median": float(median(gaps)) if gaps else 0.0,
                "fixed_to_query_sample_gap_q75": _quantile(gaps, 0.75),
                "fixed_to_query_sample_gap_q90": _quantile(gaps, 0.90),
                "per_query_oracle_margin_mean": _mean(margins),
                "per_query_oracle_margin_median": float(median(margins)) if margins else 0.0,
                "low_margin_share": _low_margin_share(
                    vals,
                    abs_threshold=float(low_margin_abs_threshold),
                    rel_threshold=float(low_margin_rel_threshold),
                ),
                "fixed_domain_oracle_sample_rank_mean": _mean(ranks),
                "fixed_domain_oracle_rank1_share": _mean([1 if int(r) == 1 else 0 for r in ranks]),
                "fixed_domain_oracle_rank2_or_worse_share": _mean([1 if int(r) >= 2 else 0 for r in ranks]),
                "fixed_domain_oracle_rank3_share": _mean([1 if int(r) >= 3 else 0 for r in ranks]),
                "per_query_oracle_switch_rate": switch_rate,
                "per_query_expert_entropy_normalized": entropy_norm,
                "per_query_selected_expert_counts_json": counts_json,
            }
        )
        out.append(row)
    return out


def _fold_summary_rows(rows: Sequence[Mapping[str, object]], keys: Sequence[str]) -> List[dict]:
    metrics = [
        "fixed_to_query_oracle_gap",
        "normalized_fixed_to_query_oracle_gap",
        "per_query_expert_entropy_normalized",
        "per_query_oracle_switch_rate",
        "low_margin_share",
        "fixed_domain_oracle_sample_rank_mean",
        "metadata_ordinal_excluded_normalized_gap_to_fixed_oracle",
    ]
    out: List[dict] = []
    for key, vals in sorted(_group_rows(rows, keys).items()):
        row: Dict[str, object] = {k: value for k, value in zip(keys, key)}
        row.update(
            {
                "n_folds": int(len(vals)),
                "n_runs": int(len(set(str(v.get("run_id", "")) for v in vals))),
                "n_target_samples": int(sum(int(float(v.get("n_target_samples", 0) or 0)) for v in vals)),
            }
        )
        for metric in metrics:
            arr = [_to_float(v.get(metric, 0.0)) for v in vals]
            row[f"{metric}_mean"] = _mean(arr)
            row[f"{metric}_std"] = _std(arr)
        out.append(row)
    return out


def _write_md(path: Path, outputs: Mapping[str, Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Domain-Query Oracle Gap Stratification",
        "",
        "These tables stratify the per-sample oracle-headroom diagnostic by target domain, class label, and patient.",
        "",
        "## Outputs",
        "",
    ]
    for label, out_path in outputs.items():
        lines.append(f"- {label}: `{out_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _as_abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stratified summaries for domain-query oracle-gap outputs.")
    parser.add_argument(
        "--raw",
        default="results/comparison_tables/domain_query_oracle_gap_loqdo_breakhis_raw.csv",
    )
    parser.add_argument(
        "--per-sample",
        default="results/comparison_tables/domain_query_oracle_gap_loqdo_breakhis_per_sample.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="results/comparison_tables/domain_query_oracle_gap_stratification",
    )
    parser.add_argument(
        "--summary-md-out",
        default="results/summaries/domain_query_oracle_gap_stratification_breakhis_summary.md",
    )
    parser.add_argument("--low-margin-abs-threshold", type=float, default=1.0e-8)
    parser.add_argument("--low-margin-rel-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_rows = _read_csv(_as_abs(str(args.raw)))
    sample_rows = _enrich_sample_rows(_read_csv(_as_abs(str(args.per_sample))))
    out_dir = _as_abs(str(args.out_dir))

    outputs = {
        "fold_by_target_domain": out_dir / "fold_by_target_domain.csv",
        "fold_by_backbone_target_domain": out_dir / "fold_by_backbone_target_domain.csv",
        "sample_by_target_domain": out_dir / "sample_by_target_domain.csv",
        "sample_by_target_domain_label": out_dir / "sample_by_target_domain_label.csv",
        "sample_by_target_domain_patient": out_dir / "sample_by_target_domain_patient.csv",
    }

    _write_csv(outputs["fold_by_target_domain"], _fold_summary_rows(raw_rows, ["target_domain"]))
    _write_csv(outputs["fold_by_backbone_target_domain"], _fold_summary_rows(raw_rows, ["backbone_type", "target_domain"]))
    _write_csv(
        outputs["sample_by_target_domain"],
        _sample_summary_rows(
            sample_rows,
            ["target_domain"],
            low_margin_abs_threshold=float(args.low_margin_abs_threshold),
            low_margin_rel_threshold=float(args.low_margin_rel_threshold),
        ),
    )
    _write_csv(
        outputs["sample_by_target_domain_label"],
        _sample_summary_rows(
            sample_rows,
            ["target_domain", "label", "label_name"],
            low_margin_abs_threshold=float(args.low_margin_abs_threshold),
            low_margin_rel_threshold=float(args.low_margin_rel_threshold),
        ),
    )
    patient_rows = [r for r in sample_rows if str(r.get("patient_id", ""))]
    _write_csv(
        outputs["sample_by_target_domain_patient"],
        _sample_summary_rows(
            patient_rows,
            ["target_domain", "patient_id", "label", "label_name"],
            low_margin_abs_threshold=float(args.low_margin_abs_threshold),
            low_margin_rel_threshold=float(args.low_margin_rel_threshold),
        ),
    )
    _write_md(_as_abs(str(args.summary_md_out)), outputs)
    for label, out_path in outputs.items():
        print(f"Wrote {label}: {out_path}")
    print(f"Wrote summary: {_as_abs(str(args.summary_md_out))}")


if __name__ == "__main__":
    main()
