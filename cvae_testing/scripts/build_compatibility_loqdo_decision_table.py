#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return 0.0, 0.0
    mu = float(mean(clean))
    var = float(sum((v - mu) ** 2 for v in clean) / len(clean))
    return mu, math.sqrt(max(var, 0.0))


def _sign_inconsistency_count(values: Sequence[float]) -> int:
    pos = any(float(v) > 1e-12 for v in values)
    neg = any(float(v) < -1e-12 for v in values)
    return 1 if (pos and neg) else 0


def _tier(
    *,
    improving_run_count: int,
    min_improving_runs: int,
    spearman_uplift_mean: float,
    top1_uplift_mean: float,
    gap_reduction_mean: float,
    normalized_gap_reduction_mean: float,
    strong: Dict[str, float],
    weak: Dict[str, float],
    instability_breach: bool,
) -> str:
    if instability_breach:
        return "fail"

    strong_ok = (
        improving_run_count >= int(min_improving_runs)
        and spearman_uplift_mean >= float(strong["spearman_uplift_min"])
        and top1_uplift_mean >= float(strong["top1_uplift_min"])
        and gap_reduction_mean >= float(strong["oracle_gap_reduction_min"])
        and normalized_gap_reduction_mean >= float(strong["normalized_oracle_gap_reduction_min"])
    )
    if strong_ok:
        return "strong_pass"

    weak_ok = (
        improving_run_count >= int(min_improving_runs)
        and spearman_uplift_mean >= float(weak["spearman_uplift_min"])
        and top1_uplift_mean >= float(weak["top1_uplift_min"])
        and gap_reduction_mean >= float(weak["oracle_gap_reduction_min"])
        and normalized_gap_reduction_mean >= float(weak["normalized_oracle_gap_reduction_min"])
    )
    if weak_ok:
        return "weak_pass"

    return "fail"


def _read_raw(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]

    if not rows:
        raise RuntimeError(f"Raw CSV has no rows: {path}")
    return rows


def _run_key(r: dict) -> Tuple[str, str, str, str]:
    return (
        str(r.get("dataset_name", "")),
        str(r.get("backbone_type", "")),
        str(r.get("run_id", "")),
        str(r.get("variant", "")),
    )


def _method_key(r: dict) -> str:
    method = str(r.get("method", ""))
    feature_set = str(r.get("feature_set", ""))
    probe_mode = str(r.get("probe_feature_mode", "off"))
    interaction_mode = str(r.get("interaction_feature_mode", "off"))
    arm = str(r.get("disentanglement_arm", "default"))
    if method == "metadata_routing":
        return "metadata_routing"
    return f"{method}__{feature_set}__probe_{probe_mode}__interact_{interaction_mode}__arm_{arm}"


def _aggregate_per_run(rows: Sequence[dict]) -> Dict[Tuple[str, str, str, str, str], dict]:
    groups: Dict[Tuple[str, str, str, str, str], List[dict]] = {}
    for r in rows:
        key = _run_key(r) + (_method_key(r),)
        groups.setdefault(key, []).append(r)

    out: Dict[Tuple[str, str, str, str, str], dict] = {}
    for key, vals in groups.items():
        dataset_name, backbone_type, run_id, variant, method_key = key
        top1 = [_to_float(v.get("top1_agreement_with_best_expert", 0.0)) for v in vals]
        spearman = [_to_float(v.get("spearman_similarity_vs_neg_nelbo", 0.0)) for v in vals]
        gap = [_to_float(v.get("metadata_to_oracle_gap", 0.0)) for v in vals]
        norm_gap = [_to_float(v.get("normalized_metadata_to_oracle_gap", 0.0)) for v in vals]
        cal = [_to_float(v.get("calibration_error_bin10", 0.0)) for v in vals]
        margin = [_to_float(v.get("top1_margin", 0.0)) for v in vals]

        out[key] = {
            "dataset_name": dataset_name,
            "backbone_type": backbone_type,
            "run_id": run_id,
            "variant": variant,
            "method_key": method_key,
            "n_folds": int(len(vals)),
            "top1": float(mean(top1)) if top1 else 0.0,
            "spearman": float(mean(spearman)) if spearman else 0.0,
            "oracle_gap": float(mean(gap)) if gap else 0.0,
            "normalized_oracle_gap": float(mean(norm_gap)) if norm_gap else 0.0,
            "calibration_error": float(mean(cal)) if cal else 0.0,
            "top1_margin": float(mean(margin)) if margin else 0.0,
        }
    return out


def _select_rows(rows: Sequence[dict], only_feature_set_b: bool) -> List[dict]:
    selected: List[dict] = []
    for r in rows:
        method = str(r.get("method", ""))
        feature_set = str(r.get("feature_set", ""))
        if method == "metadata_routing":
            selected.append(r)
            continue
        if only_feature_set_b and feature_set != "B":
            continue
        selected.append(r)
    return selected


def _aggregate_methods(
    rows: Sequence[dict],
    *,
    uplift_reference_method: str,
    min_improving_runs: int,
    strong: Dict[str, float],
    weak: Dict[str, float],
    instability_std_threshold: float,
    instability_sign_inconsistency_min_count: int,
    max_calibration_error_mean: float,
    calibration_reduction_min: float,
) -> Tuple[List[dict], Dict[str, object]]:
    per_run = _aggregate_per_run(rows)

    by_run: Dict[Tuple[str, str, str, str], Dict[str, dict]] = {}
    for key, rec in per_run.items():
        run_key = key[:4]
        by_run.setdefault(run_key, {})[str(rec["method_key"])] = rec

    method_records: Dict[str, List[dict]] = {}
    for run_key, methods in by_run.items():
        baseline = methods.get(str(uplift_reference_method))
        if baseline is None:
            continue
        for mkey, rec in methods.items():
            row = dict(rec)
            row["top1_uplift_vs_metadata"] = float(rec["top1"] - baseline["top1"])
            row["spearman_uplift_vs_metadata"] = float(rec["spearman"] - baseline["spearman"])
            row["oracle_gap_reduction_vs_metadata"] = float(baseline["oracle_gap"] - rec["oracle_gap"])
            row["normalized_oracle_gap_reduction_vs_metadata"] = float(
                baseline["normalized_oracle_gap"] - rec["normalized_oracle_gap"]
            )
            row["calibration_error_reduction_vs_metadata"] = float(
                baseline["calibration_error"] - rec["calibration_error"]
            )
            method_records.setdefault(mkey, []).append(row)

    out_rows: List[dict] = []
    for method_key in sorted(method_records.keys()):
        vals = method_records[method_key]
        n_runs = int(len(vals))

        top1_vals = [float(v["top1"]) for v in vals]
        spearman_vals = [float(v["spearman"]) for v in vals]
        gap_vals = [float(v["oracle_gap"]) for v in vals]
        norm_gap_vals = [float(v["normalized_oracle_gap"]) for v in vals]
        cal_vals = [float(v["calibration_error"]) for v in vals]
        margin_vals = [float(v["top1_margin"]) for v in vals]

        top1_uplifts = [float(v["top1_uplift_vs_metadata"]) for v in vals]
        spearman_uplifts = [float(v["spearman_uplift_vs_metadata"]) for v in vals]
        gap_reductions = [float(v["oracle_gap_reduction_vs_metadata"]) for v in vals]
        norm_gap_reductions = [float(v["normalized_oracle_gap_reduction_vs_metadata"]) for v in vals]
        cal_reductions = [float(v["calibration_error_reduction_vs_metadata"]) for v in vals]

        top1_mean, top1_std = _mean_std(top1_vals)
        spearman_mean, spearman_std = _mean_std(spearman_vals)
        gap_mean, gap_std = _mean_std(gap_vals)
        norm_gap_mean, norm_gap_std = _mean_std(norm_gap_vals)
        cal_mean, cal_std = _mean_std(cal_vals)
        margin_mean, margin_std = _mean_std(margin_vals)

        top1_uplift_mean, top1_uplift_std = _mean_std(top1_uplifts)
        spearman_uplift_mean, spearman_uplift_std = _mean_std(spearman_uplifts)
        gap_reduction_mean, gap_reduction_std = _mean_std(gap_reductions)
        norm_gap_reduction_mean, norm_gap_reduction_std = _mean_std(norm_gap_reductions)
        cal_reduction_mean, cal_reduction_std = _mean_std(cal_reductions)

        improving_run_count = sum(
            1
            for i in range(n_runs)
            if (
                top1_uplifts[i] > 0.0
                and spearman_uplifts[i] > 0.0
                and gap_reductions[i] > 0.0
                and norm_gap_reductions[i] > 0.0
            )
        )

        std_breach = bool(
            top1_uplift_std > float(instability_std_threshold)
            or spearman_uplift_std > float(instability_std_threshold)
            or gap_reduction_std > float(instability_std_threshold)
            or norm_gap_reduction_std > float(instability_std_threshold)
        )
        sign_inconsistency_count = (
            _sign_inconsistency_count(top1_uplifts)
            + _sign_inconsistency_count(spearman_uplifts)
            + _sign_inconsistency_count(gap_reductions)
            + _sign_inconsistency_count(norm_gap_reductions)
        )
        sign_breach = bool(sign_inconsistency_count >= int(instability_sign_inconsistency_min_count))
        instability_breach = bool(std_breach or sign_breach)

        if method_key == str(uplift_reference_method):
            tier = "baseline"
        else:
            tier = _tier(
                improving_run_count=int(improving_run_count),
                min_improving_runs=int(min_improving_runs),
                spearman_uplift_mean=float(spearman_uplift_mean),
                top1_uplift_mean=float(top1_uplift_mean),
                gap_reduction_mean=float(gap_reduction_mean),
                normalized_gap_reduction_mean=float(norm_gap_reduction_mean),
                strong=strong,
                weak=weak,
                instability_breach=instability_breach,
            )

        joint_top1_gap_guardrail_pass = bool(
            top1_uplift_mean > 0.0 and gap_reduction_mean > 0.0 and norm_gap_reduction_mean > 0.0
        )
        uncertainty_calibration_gate_pass = bool(
            cal_mean <= float(max_calibration_error_mean)
            and cal_reduction_mean >= float(calibration_reduction_min)
            and not instability_breach
        )
        adoption_gate_pass_proxy = bool(
            method_key != str(uplift_reference_method)
            and joint_top1_gap_guardrail_pass
            and uncertainty_calibration_gate_pass
            and spearman_uplift_mean > 0.0
        )

        out_rows.append(
            {
                "method_key": method_key,
                "n_runs": n_runs,
                "top1_mean": top1_mean,
                "top1_std": top1_std,
                "spearman_mean": spearman_mean,
                "spearman_std": spearman_std,
                "oracle_gap_mean": gap_mean,
                "oracle_gap_std": gap_std,
                "normalized_oracle_gap_mean": norm_gap_mean,
                "normalized_oracle_gap_std": norm_gap_std,
                "calibration_error_mean": cal_mean,
                "calibration_error_std": cal_std,
                "top1_margin_mean": margin_mean,
                "top1_margin_std": margin_std,
                "top1_uplift_vs_metadata_mean": top1_uplift_mean,
                "top1_uplift_vs_metadata_std": top1_uplift_std,
                "spearman_uplift_vs_metadata_mean": spearman_uplift_mean,
                "spearman_uplift_vs_metadata_std": spearman_uplift_std,
                "oracle_gap_reduction_vs_metadata_mean": gap_reduction_mean,
                "oracle_gap_reduction_vs_metadata_std": gap_reduction_std,
                "normalized_oracle_gap_reduction_vs_metadata_mean": norm_gap_reduction_mean,
                "normalized_oracle_gap_reduction_vs_metadata_std": norm_gap_reduction_std,
                "calibration_error_reduction_vs_metadata_mean": cal_reduction_mean,
                "calibration_error_reduction_vs_metadata_std": cal_reduction_std,
                "improving_run_count": int(improving_run_count),
                "instability_std_breach": int(std_breach),
                "instability_sign_inconsistency_count": int(sign_inconsistency_count),
                "instability_breach": int(instability_breach),
                "joint_top1_gap_guardrail_pass": int(joint_top1_gap_guardrail_pass),
                "uncertainty_calibration_gate_pass": int(uncertainty_calibration_gate_pass),
                "adoption_gate_pass_proxy": int(adoption_gate_pass_proxy),
                "tier": tier,
            }
        )

    out_rows.sort(key=lambda r: (str(r["tier"]), -float(r["spearman_uplift_vs_metadata_mean"]), str(r["method_key"])))

    summary = {
        "total_methods": int(len(out_rows)),
        "strong_pass_count": int(sum(1 for r in out_rows if str(r["tier"]) == "strong_pass")),
        "weak_pass_count": int(sum(1 for r in out_rows if str(r["tier"]) == "weak_pass")),
        "fail_count": int(sum(1 for r in out_rows if str(r["tier"]) == "fail")),
        "baseline_count": int(sum(1 for r in out_rows if str(r["tier"]) == "baseline")),
    }
    return out_rows, summary


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: Sequence[dict], summary: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# LOQDO Compatibility Decision Table")
    lines.append("")
    lines.append(f"- Total methods: {int(summary['total_methods'])}")
    lines.append(f"- Strong pass: {int(summary['strong_pass_count'])}")
    lines.append(f"- Weak pass: {int(summary['weak_pass_count'])}")
    lines.append(f"- Fail: {int(summary['fail_count'])}")
    lines.append("")
    lines.append("| Method | Tier | Runs | Top1 mean+-std | Spearman mean+-std | Gap mean+-std | NormGap mean+-std | CalErr mean+-std | Top1 uplift | Spearman uplift | Gap reduction | NormGap reduction | CalErr reduction | Joint guardrail | Cal gate | Adoption gate |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {} | {} | {} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {} | {} | {} |".format(
                r["method_key"],
                r["tier"],
                int(r["n_runs"]),
                float(r["top1_mean"]),
                float(r["top1_std"]),
                float(r["spearman_mean"]),
                float(r["spearman_std"]),
                float(r["oracle_gap_mean"]),
                float(r["oracle_gap_std"]),
                float(r["normalized_oracle_gap_mean"]),
                float(r["normalized_oracle_gap_std"]),
                float(r["calibration_error_mean"]),
                float(r["calibration_error_std"]),
                float(r["top1_uplift_vs_metadata_mean"]),
                float(r["top1_uplift_vs_metadata_std"]),
                float(r["spearman_uplift_vs_metadata_mean"]),
                float(r["spearman_uplift_vs_metadata_std"]),
                float(r["oracle_gap_reduction_vs_metadata_mean"]),
                float(r["oracle_gap_reduction_vs_metadata_std"]),
                float(r["normalized_oracle_gap_reduction_vs_metadata_mean"]),
                float(r["normalized_oracle_gap_reduction_vs_metadata_std"]),
                float(r["calibration_error_reduction_vs_metadata_mean"]),
                float(r["calibration_error_reduction_vs_metadata_std"]),
                int(r["joint_top1_gap_guardrail_pass"]),
                int(r["uncertainty_calibration_gate_pass"]),
                int(r["adoption_gate_pass_proxy"]),
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Build decision-grade table from LOQDO raw CSV outputs.")
    p.add_argument("--raw-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    p.add_argument("--uplift-reference-method", type=str, default="metadata_routing")
    p.add_argument("--only-feature-set-b", action="store_true")
    p.add_argument("--min-improving-runs", type=int, default=6)
    p.add_argument("--strong-spearman-uplift-min", type=float, default=0.05)
    p.add_argument("--strong-top1-uplift-min", type=float, default=0.10)
    p.add_argument("--strong-gap-reduction-min", type=float, default=0.005)
    p.add_argument("--strong-normalized-gap-reduction-min", type=float, default=0.01)
    p.add_argument("--weak-spearman-uplift-min", type=float, default=0.025)
    p.add_argument("--weak-top1-uplift-min", type=float, default=0.05)
    p.add_argument("--weak-gap-reduction-min", type=float, default=0.0025)
    p.add_argument("--weak-normalized-gap-reduction-min", type=float, default=0.005)
    p.add_argument("--max-calibration-error-mean", type=float, default=0.20)
    p.add_argument("--calibration-reduction-min", type=float, default=0.0)
    p.add_argument("--instability-std-threshold", type=float, default=0.05)
    p.add_argument("--instability-sign-inconsistency-min-count", type=int, default=2)
    args = p.parse_args()

    raw_rows = _read_raw(args.raw_csv)
    selected_rows = _select_rows(raw_rows, only_feature_set_b=bool(args.only_feature_set_b))

    strong = {
        "spearman_uplift_min": float(args.strong_spearman_uplift_min),
        "top1_uplift_min": float(args.strong_top1_uplift_min),
        "oracle_gap_reduction_min": float(args.strong_gap_reduction_min),
        "normalized_oracle_gap_reduction_min": float(args.strong_normalized_gap_reduction_min),
    }
    weak = {
        "spearman_uplift_min": float(args.weak_spearman_uplift_min),
        "top1_uplift_min": float(args.weak_top1_uplift_min),
        "oracle_gap_reduction_min": float(args.weak_gap_reduction_min),
        "normalized_oracle_gap_reduction_min": float(args.weak_normalized_gap_reduction_min),
    }

    out_rows, summary = _aggregate_methods(
        selected_rows,
        uplift_reference_method=str(args.uplift_reference_method),
        min_improving_runs=int(args.min_improving_runs),
        strong=strong,
        weak=weak,
        instability_std_threshold=float(args.instability_std_threshold),
        instability_sign_inconsistency_min_count=int(args.instability_sign_inconsistency_min_count),
        max_calibration_error_mean=float(args.max_calibration_error_mean),
        calibration_reduction_min=float(args.calibration_reduction_min),
    )

    _write_csv(args.output_csv, out_rows)
    _write_md(args.output_md, out_rows, summary)

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
