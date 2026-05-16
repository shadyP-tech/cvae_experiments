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


def _to_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


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


def _is_deployable_method_key(method_key: str) -> bool:
    mk = str(method_key)
    blocked_tokens = [
        "response_indirect_shuffled",
        "response_target_adjacent_diagnostic",
        "response_oracle_diagnostic",
        "control_only",
        "diagnostic",
        "target_adjacent",
        "oracle_diagnostic",
        "oracle_eval_mean_cheat",
        "oracle_pairwise_rank_cheat",
        "semi_oracle_support_mean",
        "semi_oracle_support_riskaware",
        "__arm_baseline",
    ]
    return not any(tok in mk for tok in blocked_tokens)


def _find_best_nonbaseline_deployable(rows: List[dict]) -> Optional[dict]:
    cand = [
        r
        for r in rows
        if str(r.get("method_key", "")) != "metadata_routing"
        and _is_deployable_method_key(str(r.get("method_key", "")))
    ]
    if not cand:
        return None

    tier_rank = {"strong_pass": 0, "weak_pass": 1, "fail": 2, "baseline": 3}

    def _key(r: dict):
        return (
            tier_rank.get(str(r.get("tier", "fail")), 9),
            -_to_float(r.get("oracle_gap_reduction_vs_metadata_mean", 0.0)),
            -_to_float(r.get("normalized_oracle_gap_reduction_vs_metadata_mean", 0.0)),
            -_to_float(r.get("spearman_uplift_vs_metadata_mean", 0.0)),
            -_to_float(r.get("top1_uplift_vs_metadata_mean", 0.0)),
        )

    return sorted(cand, key=_key)[0]


def _is_transfer_success(
    row: Optional[dict],
    *,
    min_top1_uplift: float,
    min_norm_gap_reduction: float,
    max_calibration_error_mean: float,
    require_adoption_gate_proxy: bool,
    allow_derived_fallback: bool,
) -> bool:
    if row is None:
        return False

    raw_gate = row.get("adoption_gate_pass_proxy")
    has_adoption_gate = raw_gate is not None and str(raw_gate).strip() != ""
    if bool(require_adoption_gate_proxy):
        if has_adoption_gate:
            return int(_to_float(raw_gate, 0.0)) == 1
        if not bool(allow_derived_fallback):
            return False

    top1_uplift = _to_float(row.get("top1_uplift_vs_metadata_mean", 0.0))
    norm_gap_reduction = _to_float(row.get("normalized_oracle_gap_reduction_vs_metadata_mean", 0.0))
    instability_breach = int(_to_float(row.get("instability_breach", 1)))
    cal_mean = _to_float(row.get("calibration_error_mean", 1.0))
    return bool(
        top1_uplift >= float(min_top1_uplift)
        and norm_gap_reduction >= float(min_norm_gap_reduction)
        and instability_breach == 0
        and cal_mean <= float(max_calibration_error_mean)
    )


def _has_any_pass(rows: List[dict]) -> bool:
    return any(str(r.get("tier", "")) in {"strong_pass", "weak_pass"} for r in rows)


def _classify(
    breakhis_rows: List[dict],
    camelyon_rows: List[dict],
    *,
    min_top1_uplift: float,
    min_norm_gap_reduction: float,
    max_calibration_error_mean: float,
    include_diagnostic_oracle_methods: bool,
    require_adoption_gate_proxy: bool,
    allow_derived_fallback: bool,
) -> Dict[str, object]:
    b_pass = _has_any_pass(breakhis_rows)
    c_pass = _has_any_pass(camelyon_rows)

    b_best_all = _find_best_nonbaseline(breakhis_rows)
    c_best_all = _find_best_nonbaseline(camelyon_rows)
    if include_diagnostic_oracle_methods:
        b_best = b_best_all
        c_best = c_best_all
    else:
        b_best = _find_best_nonbaseline_deployable(breakhis_rows)
        c_best = _find_best_nonbaseline_deployable(camelyon_rows)

    b_transfer_ok = _is_transfer_success(
        b_best,
        min_top1_uplift=min_top1_uplift,
        min_norm_gap_reduction=min_norm_gap_reduction,
        max_calibration_error_mean=max_calibration_error_mean,
        require_adoption_gate_proxy=bool(require_adoption_gate_proxy),
        allow_derived_fallback=bool(allow_derived_fallback),
    )
    c_transfer_ok = _is_transfer_success(
        c_best,
        min_top1_uplift=min_top1_uplift,
        min_norm_gap_reduction=min_norm_gap_reduction,
        max_calibration_error_mean=max_calibration_error_mean,
        require_adoption_gate_proxy=bool(require_adoption_gate_proxy),
        allow_derived_fallback=bool(allow_derived_fallback),
    )
    transfer_success = bool(b_transfer_ok and c_transfer_ok)

    if transfer_success:
        label = "cross_dataset_transfer_success"
        rationale = (
            "Best non-baseline methods satisfy transfer gate on both datasets: "
            "Top1 uplift, normalized-gap reduction, instability=0, and bounded calibration error."
        )
    elif b_pass and c_pass:
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
        "include_diagnostic_oracle_methods": bool(include_diagnostic_oracle_methods),
        "require_adoption_gate_proxy": bool(require_adoption_gate_proxy),
        "allow_derived_fallback": bool(allow_derived_fallback),
        "transfer_success": transfer_success,
        "breakhis_transfer_gate_pass": b_transfer_ok,
        "camelyon17_transfer_gate_pass": c_transfer_ok,
        "transfer_gate": {
            "min_top1_uplift": float(min_top1_uplift),
            "min_normalized_gap_reduction": float(min_norm_gap_reduction),
            "max_calibration_error_mean": float(max_calibration_error_mean),
            "required_instability_breach": 0,
            "required_adoption_gate_pass_proxy": 1 if bool(require_adoption_gate_proxy) else 0,
            "allow_derived_fallback_when_missing": bool(allow_derived_fallback),
        },
        "breakhis_best_method": (b_best or {}).get("method_key"),
        "camelyon17_best_method": (c_best or {}).get("method_key"),
        "breakhis_best_method_all_candidates": (b_best_all or {}).get("method_key"),
        "camelyon17_best_method_all_candidates": (c_best_all or {}).get("method_key"),
        "breakhis_best": b_best,
        "camelyon17_best": c_best,
        "breakhis_best_all_candidates": b_best_all,
        "camelyon17_best_all_candidates": c_best_all,
    }


def _write_md(path: Path, payload: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# Cross-Dataset Assessment: LOQDO Utility-Compatible Learning")
    lines.append("")
    lines.append(f"- Classification: {payload.get('classification')}")
    lines.append(f"- Rationale: {payload.get('rationale')}")
    lines.append(f"- Transfer success: {payload.get('transfer_success')}")
    lines.append(f"- BreakHis transfer gate pass: {payload.get('breakhis_transfer_gate_pass')}")
    lines.append(f"- Camelyon17 transfer gate pass: {payload.get('camelyon17_transfer_gate_pass')}")
    lines.append(f"- Include diagnostic oracle methods: {payload.get('include_diagnostic_oracle_methods')}")
    lines.append(f"- Require adoption_gate_pass_proxy: {payload.get('require_adoption_gate_proxy')}")
    lines.append(f"- Allow derived fallback when missing: {payload.get('allow_derived_fallback')}")
    lines.append(f"- BreakHis best method: {payload.get('breakhis_best_method')}")
    lines.append(f"- Camelyon17 best method: {payload.get('camelyon17_best_method')}")
    lines.append(f"- BreakHis best (all candidates): {payload.get('breakhis_best_method_all_candidates')}")
    lines.append(f"- Camelyon17 best (all candidates): {payload.get('camelyon17_best_method_all_candidates')}")
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
        lines.append(f"- normalized_oracle_gap_reduction_vs_metadata_mean: {rec.get('normalized_oracle_gap_reduction_vs_metadata_mean')}")
        lines.append(f"- calibration_error_mean: {rec.get('calibration_error_mean')}")
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
    p.add_argument("--min-top1-uplift", type=float, default=0.0)
    p.add_argument("--min-normalized-gap-reduction", type=float, default=0.0)
    p.add_argument("--max-calibration-error-mean", type=float, default=0.20)
    p.add_argument(
        "--include-diagnostic-oracle-methods",
        action="store_true",
        help="If set, oracle/semi-oracle diagnostic methods are allowed as best-candidate selectors.",
    )
    p.add_argument(
        "--disable-require-adoption-gate-proxy",
        action="store_true",
        help="Disable strict coupling to adoption_gate_pass_proxy in decision tables.",
    )
    p.add_argument(
        "--allow-derived-fallback",
        action="store_true",
        help="If adoption_gate_pass_proxy is missing in a decision table row, allow derived transfer checks.",
    )
    args = p.parse_args()

    b_rows = _read_table(args.breakhis_csv)
    c_rows = _read_table(args.camelyon17_csv)
    payload = _classify(
        b_rows,
        c_rows,
        min_top1_uplift=float(args.min_top1_uplift),
        min_norm_gap_reduction=float(args.min_normalized_gap_reduction),
        max_calibration_error_mean=float(args.max_calibration_error_mean),
        include_diagnostic_oracle_methods=bool(args.include_diagnostic_oracle_methods),
        require_adoption_gate_proxy=not bool(args.disable_require_adoption_gate_proxy),
        allow_derived_fallback=bool(args.allow_derived_fallback),
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_md(args.output_md, payload)

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
