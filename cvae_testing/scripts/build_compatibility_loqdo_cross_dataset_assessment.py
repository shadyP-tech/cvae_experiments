#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def _read_table(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Decision table not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    if not rows:
        raise RuntimeError(f"Decision table is empty: {path}")
    return rows


def _to_float(v: object) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _find_best_nonbaseline(rows: List[dict]) -> Optional[dict]:
    cand = [r for r in rows if str(r.get("method_key", "")) != "metadata_routing"]
    if not cand:
        return None

    tier_rank = {"strong_pass": 0, "weak_pass": 1, "fail": 2, "baseline": 3}

    def _key(r: dict):
        return (
            tier_rank.get(str(r.get("tier", "fail")), 9),
            -_to_float(r.get("oracle_gap_reduction_vs_metadata_mean", 0.0)),
            -_to_float(r.get("spearman_uplift_vs_metadata_mean", 0.0)),
            -_to_float(r.get("top1_uplift_vs_metadata_mean", 0.0)),
        )

    return sorted(cand, key=_key)[0]


def _has_any_pass(rows: List[dict]) -> bool:
    return any(str(r.get("tier", "")) in {"strong_pass", "weak_pass"} for r in rows)


def _classify(breakhis_rows: List[dict], camelyon_rows: List[dict]) -> Dict[str, object]:
    b_pass = _has_any_pass(breakhis_rows)
    c_pass = _has_any_pass(camelyon_rows)

    b_best = _find_best_nonbaseline(breakhis_rows)
    c_best = _find_best_nonbaseline(camelyon_rows)

    if b_pass and c_pass:
        label = "cross_dataset_actionable_success"
        rationale = "At least one method reached pass tier on both datasets under locked instability gates."
    elif b_pass != c_pass:
        label = "dataset_sensitive_behavior"
        rationale = "Pass-tier behavior appears on only one dataset under the same protocol."
    else:
        # Neither passed.
        b_gap = _to_float((b_best or {}).get("oracle_gap_reduction_vs_metadata_mean", 0.0))
        c_gap = _to_float((c_best or {}).get("oracle_gap_reduction_vs_metadata_mean", 0.0))
        b_sp = _to_float((b_best or {}).get("spearman_uplift_vs_metadata_mean", 0.0))
        c_sp = _to_float((c_best or {}).get("spearman_uplift_vs_metadata_mean", 0.0))

        if (b_sp > 0.0 or c_sp > 0.0) and (b_gap <= 0.0 or c_gap <= 0.0):
            label = "non_actionable_ranking_only"
            rationale = "Some ranking uplift appears, but utility-gap reduction does not hold across datasets."
        else:
            label = "utility_signal_not_recoverable_current_observables"
            rationale = "No method reaches pass-tier and utility-aligned translation remains absent/unstable across datasets."

    return {
        "classification": label,
        "rationale": rationale,
        "breakhis_best_method": (b_best or {}).get("method_key"),
        "camelyon17_best_method": (c_best or {}).get("method_key"),
        "breakhis_best": b_best,
        "camelyon17_best": c_best,
    }


def _write_md(path: Path, payload: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# Cross-Dataset Assessment: LOQDO Utility-Compatible Learning")
    lines.append("")
    lines.append(f"- Classification: {payload.get('classification')}")
    lines.append(f"- Rationale: {payload.get('rationale')}")
    lines.append(f"- BreakHis best method: {payload.get('breakhis_best_method')}")
    lines.append(f"- Camelyon17 best method: {payload.get('camelyon17_best_method')}")
    lines.append("")

    for dataset_key in ["breakhis_best", "camelyon17_best"]:
        rec = payload.get(dataset_key)
        if not isinstance(rec, dict):
            continue
        lines.append(f"## {dataset_key}")
        lines.append(f"- method_key: {rec.get('method_key')}")
        lines.append(f"- tier: {rec.get('tier')}")
        lines.append(f"- top1_uplift_vs_metadata_mean: {rec.get('top1_uplift_vs_metadata_mean')}")
        lines.append(f"- spearman_uplift_vs_metadata_mean: {rec.get('spearman_uplift_vs_metadata_mean')}")
        lines.append(f"- oracle_gap_reduction_vs_metadata_mean: {rec.get('oracle_gap_reduction_vs_metadata_mean')}")
        lines.append(f"- improving_run_count: {rec.get('improving_run_count')}")
        lines.append(f"- instability_breach: {rec.get('instability_breach')}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-dataset assessment for LOQDO compatibility decision tables.")
    p.add_argument("--breakhis-csv", type=Path, required=True)
    p.add_argument("--camelyon17-csv", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    args = p.parse_args()

    b_rows = _read_table(args.breakhis_csv)
    c_rows = _read_table(args.camelyon17_csv)
    payload = _classify(b_rows, c_rows)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_md(args.output_md, payload)

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
