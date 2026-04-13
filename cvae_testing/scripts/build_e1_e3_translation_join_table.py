#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _to_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _to_int(v: object, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _pick_best_compat_method(rows: List[dict]) -> Optional[dict]:
    candidates = [r for r in rows if str(r.get("method_key", "")) != "metadata_routing"]
    if not candidates:
        return None

    tier_rank = {"strong_pass": 0, "weak_pass": 1, "fail": 2, "baseline": 3}

    def _key(r: dict) -> Tuple[int, float, float, int]:
        return (
            int(tier_rank.get(str(r.get("tier", "fail")), 9)),
            -_to_float(r.get("oracle_gap_reduction_vs_metadata_mean", 0.0)),
            -_to_float(r.get("spearman_uplift_vs_metadata_mean", 0.0)),
            _to_int(r.get("instability_breach", 0), 0),
        )

    return sorted(candidates, key=_key)[0]


def _extract_metrics_from_report(path: Path) -> dict:
    payload = _read_json(path)
    m = payload.get("metrics", {})
    return {
        "best_expert_true_utility_nelbo": _to_float(m.get("best_expert_true_utility_nelbo", 0.0)),
        "top1_oracle_hit_true_utility": _to_float(m.get("top1_oracle_hit_true_utility", 0.0)),
        "spearman_with_true_utility": _to_float(m.get("spearman_with_true_utility", 0.0)),
        "routed_to_global_gap": _to_float(m.get("routed_to_global_gap", 0.0)),
        "routed_to_true_oracle_gap": _to_float(m.get("routed_to_true_oracle_gap", 0.0)),
        "routed_to_global_gap_norm_abs_median": _to_float(m.get("routed_to_global_gap_norm_abs_median", 0.0)),
        "hard_metadata_routing_nelbo": _to_float(m.get("hard_metadata_routing_nelbo", 0.0)),
        "global_cvae_nelbo": _to_float(m.get("global_cvae_nelbo", 0.0)),
        "routing_selection_accuracy": _to_float(m.get("routing_selection_accuracy", 0.0)),
    }


def _aggregate_system_metrics(decision_rows: List[dict]) -> dict:
    base_metrics: List[dict] = []
    cond_metrics: List[dict] = []
    seeds: List[int] = []

    for row in decision_rows:
        seed = _to_int(row.get("seed", 0), 0)
        base_report = Path(str(row.get("baseline_report", "")))
        cond_report = Path(str(row.get("conditioned_report", "")))
        if not base_report.exists() or not cond_report.exists():
            raise FileNotFoundError(f"Missing report in decision row for seed={seed}")
        seeds.append(seed)
        base_metrics.append(_extract_metrics_from_report(base_report))
        cond_metrics.append(_extract_metrics_from_report(cond_report))

    keys = [
        "best_expert_true_utility_nelbo",
        "top1_oracle_hit_true_utility",
        "spearman_with_true_utility",
        "routed_to_global_gap",
        "routed_to_true_oracle_gap",
        "routed_to_global_gap_norm_abs_median",
        "hard_metadata_routing_nelbo",
        "global_cvae_nelbo",
        "routing_selection_accuracy",
    ]

    out: Dict[str, object] = {
        "n_seeds": int(len(seeds)),
        "seeds": sorted(seeds),
    }

    for k in keys:
        base_vals = [_to_float(m.get(k, 0.0)) for m in base_metrics]
        cond_vals = [_to_float(m.get(k, 0.0)) for m in cond_metrics]
        base_mean = _mean(base_vals)
        cond_mean = _mean(cond_vals)
        out[f"baseline_{k}_mean"] = float(base_mean)
        out[f"conditioned_{k}_mean"] = float(cond_mean)
        out[f"delta_{k}_conditioned_minus_baseline"] = float(cond_mean - base_mean)
        if abs(base_mean) > 1e-12:
            out[f"relative_delta_{k}"] = float((cond_mean - base_mean) / abs(base_mean))
        else:
            out[f"relative_delta_{k}"] = 0.0

    # Positive values indicate conditioned reduced gap relative to baseline.
    out["routed_to_global_gap_reduction_mean"] = float(
        _to_float(out["baseline_routed_to_global_gap_mean"]) - _to_float(out["conditioned_routed_to_global_gap_mean"])
    )
    out["routed_to_true_oracle_gap_reduction_mean"] = float(
        _to_float(out["baseline_routed_to_true_oracle_gap_mean"])
        - _to_float(out["conditioned_routed_to_true_oracle_gap_mean"])
    )
    return out


def _translation_join_status(legacy_row: dict, compat_row: dict, compat_cross_label: str) -> str:
    e1_pass = bool(legacy_row.get("legacy_e1_pass", 0))
    e2_pass = bool(legacy_row.get("legacy_e2_pass", 0))
    e3_pass = bool(legacy_row.get("legacy_e3_pass", 0))
    compat_tier = str(compat_row.get("compat_best_tier", "fail"))
    compat_improving = _to_int(compat_row.get("compat_best_improving_run_count", 0), 0)

    if e1_pass and e3_pass and compat_tier in {"strong_pass", "weak_pass"}:
        return "actionable_translation"
    if (e2_pass and (not e1_pass) and (not e3_pass)) or compat_cross_label == "non_actionable_ranking_only":
        return "non_actionable_ranking_only"
    if (not e1_pass) and (not e3_pass) and compat_tier == "fail" and compat_improving <= 0:
        return "translation_not_supported"
    return "mixed_or_unstable"


def _build_dataset_row(
    *,
    dataset_name: str,
    legacy_summary: dict,
    legacy_decision_rows: List[dict],
    compat_rows: List[dict],
    legacy_cross_label: str,
    compat_cross_label: str,
) -> dict:
    gates = legacy_summary.get("gates", {})
    e1 = gates.get("e1", {})
    e2 = gates.get("e2", {})
    e3 = gates.get("e3", {})
    uniform = legacy_summary.get("uniformization", {})
    best_compat = _pick_best_compat_method(compat_rows) or {}
    sys_metrics = _aggregate_system_metrics(legacy_decision_rows)

    row: Dict[str, object] = {
        "dataset_name": dataset_name,
        "legacy_taxonomy_label": str(legacy_summary.get("taxonomy_label", "unknown")),
        "legacy_e1_pass": int(bool(e1.get("pass", False))),
        "legacy_e2_pass": int(bool(e2.get("pass", False))),
        "legacy_e3_pass": int(bool(e3.get("pass", False))),
        "legacy_e1_median_relative_delta": _to_float(e1.get("median_relative_delta", 0.0)),
        "legacy_e2_median_top1_uplift": _to_float(e2.get("median_top1_uplift", 0.0)),
        "legacy_e2_median_spearman_uplift": _to_float(e2.get("median_spearman_uplift", 0.0)),
        "legacy_e3_median_relative_gap_reduction": _to_float(e3.get("median_relative_gap_reduction", 0.0)),
        "legacy_e3_median_abs_norm_gap": _to_float(e3.get("median_abs_normalized_gap", 0.0)),
        "legacy_uniformization_flag": int(bool(uniform.get("flag", False))),
        "compat_best_method_key": str(best_compat.get("method_key", "none")),
        "compat_best_tier": str(best_compat.get("tier", "none")),
        "compat_best_top1_uplift_vs_metadata_mean": _to_float(
            best_compat.get("top1_uplift_vs_metadata_mean", 0.0)
        ),
        "compat_best_spearman_uplift_vs_metadata_mean": _to_float(
            best_compat.get("spearman_uplift_vs_metadata_mean", 0.0)
        ),
        "compat_best_oracle_gap_reduction_vs_metadata_mean": _to_float(
            best_compat.get("oracle_gap_reduction_vs_metadata_mean", 0.0)
        ),
        "compat_best_improving_run_count": _to_int(best_compat.get("improving_run_count", 0), 0),
        "compat_best_instability_breach": _to_int(best_compat.get("instability_breach", 0), 0),
        "legacy_cross_dataset_label": legacy_cross_label,
        "compat_cross_dataset_label": compat_cross_label,
    }
    row.update(sys_metrics)

    row["translation_join_status"] = _translation_join_status(
        legacy_row=row,
        compat_row=row,
        compat_cross_label=str(compat_cross_label),
    )
    return row


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: List[dict], payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# E1/E3 Translation-Join Assessment")
    lines.append("")
    lines.append("This table joins legacy translation gates (E1/E3), LOQDO compatibility outcomes, and routed system metrics.")
    lines.append("")
    lines.append(f"- legacy cross-dataset label: {payload.get('legacy_cross_dataset_label')}")
    lines.append(f"- compatibility cross-dataset label: {payload.get('compat_cross_dataset_label')}")
    lines.append("")
    lines.append(
        "| Dataset | Translation status | E1 pass | E3 pass | E1 median rel delta | E3 median rel gap reduction | Compat best method | Compat tier | Compat gap reduction vs metadata | Conditioned-Baseline top1 | Conditioned-Baseline spearman | Routed-global gap reduction | Routed-oracle gap reduction |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {} | {} | {} | {} | {:.6f} | {:.6f} | {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                r["dataset_name"],
                r["translation_join_status"],
                int(r["legacy_e1_pass"]),
                int(r["legacy_e3_pass"]),
                _to_float(r["legacy_e1_median_relative_delta"]),
                _to_float(r["legacy_e3_median_relative_gap_reduction"]),
                r["compat_best_method_key"],
                r["compat_best_tier"],
                _to_float(r["compat_best_oracle_gap_reduction_vs_metadata_mean"]),
                _to_float(r["delta_top1_oracle_hit_true_utility_conditioned_minus_baseline"]),
                _to_float(r["delta_spearman_with_true_utility_conditioned_minus_baseline"]),
                _to_float(r["routed_to_global_gap_reduction_mean"]),
                _to_float(r["routed_to_true_oracle_gap_reduction_mean"]),
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- E1 improvement requires lower (more negative) relative delta in best-expert utility NELBO.")
    lines.append("- E3 improvement is positive routed/global gap reduction.")
    lines.append("- Compatibility method rows are selected from non-baseline candidates using tier, then utility-gap reduction, then Spearman uplift.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Build joined E1/E3 translation table with routed and compatibility metrics.")
    p.add_argument(
        "--legacy-breakhis-summary-json",
        type=Path,
        default=Path("results/comparison_tables/legacy_conditioning_decision_summary.json"),
    )
    p.add_argument(
        "--legacy-camelyon-summary-json",
        type=Path,
        default=Path("results/comparison_tables/legacy_conditioning_camelyon17_decision_summary.json"),
    )
    p.add_argument(
        "--legacy-breakhis-decision-csv",
        type=Path,
        default=Path("results/comparison_tables/legacy_conditioning_decision_table.csv"),
    )
    p.add_argument(
        "--legacy-camelyon-decision-csv",
        type=Path,
        default=Path("results/comparison_tables/legacy_conditioning_camelyon17_decision_table.csv"),
    )
    p.add_argument(
        "--compat-breakhis-csv",
        type=Path,
        default=Path("results/comparison_tables/compatibility_loqdo_breakhis_decision_table.csv"),
    )
    p.add_argument(
        "--compat-camelyon-csv",
        type=Path,
        default=Path("results/comparison_tables/compatibility_loqdo_camelyon17_decision_table.csv"),
    )
    p.add_argument(
        "--legacy-cross-json",
        type=Path,
        default=Path("results/comparison_tables/legacy_conditioning_cross_dataset_assessment.json"),
    )
    p.add_argument(
        "--compat-cross-json",
        type=Path,
        default=Path("results/comparison_tables/compatibility_loqdo_cross_dataset_assessment.json"),
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/comparison_tables/e1_e3_translation_join_table.csv"),
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/comparison_tables/e1_e3_translation_join_table.json"),
    )
    p.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/summaries/e1_e3_translation_join_table.md"),
    )
    args = p.parse_args()

    legacy_breakhis_summary = _read_json(args.legacy_breakhis_summary_json)
    legacy_camelyon_summary = _read_json(args.legacy_camelyon_summary_json)
    legacy_breakhis_rows = _read_csv(args.legacy_breakhis_decision_csv)
    legacy_camelyon_rows = _read_csv(args.legacy_camelyon_decision_csv)
    compat_breakhis_rows = _read_csv(args.compat_breakhis_csv)
    compat_camelyon_rows = _read_csv(args.compat_camelyon_csv)
    legacy_cross = _read_json(args.legacy_cross_json)
    compat_cross = _read_json(args.compat_cross_json)

    legacy_cross_label = str(legacy_cross.get("classification", {}).get("label", "unknown"))
    compat_cross_label = str(compat_cross.get("classification", "unknown"))

    rows: List[dict] = []
    rows.append(
        _build_dataset_row(
            dataset_name="breakhis",
            legacy_summary=legacy_breakhis_summary,
            legacy_decision_rows=legacy_breakhis_rows,
            compat_rows=compat_breakhis_rows,
            legacy_cross_label=legacy_cross_label,
            compat_cross_label=compat_cross_label,
        )
    )
    rows.append(
        _build_dataset_row(
            dataset_name="camelyon17",
            legacy_summary=legacy_camelyon_summary,
            legacy_decision_rows=legacy_camelyon_rows,
            compat_rows=compat_camelyon_rows,
            legacy_cross_label=legacy_cross_label,
            compat_cross_label=compat_cross_label,
        )
    )

    _write_csv(args.output_csv, rows)

    payload = {
        "legacy_cross_dataset_label": legacy_cross_label,
        "compat_cross_dataset_label": compat_cross_label,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(args.output_md, rows, payload)

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
