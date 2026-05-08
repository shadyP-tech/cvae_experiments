#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from statistics import mean
from typing import Dict, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.feature_regimes import get_feature_regime, serialize_feature_list
from scripts.compatibility_stability import (
    CATASTROPHIC_GAP_PCT_REDUCTION_MIN,
    CATASTROPHIC_SPEARMAN_UPLIFT_MIN,
    CATASTROPHIC_TOP1_UPLIFT_MIN,
    GAP_PCT_CI_LOWER_TOLERANCE,
    LEGACY_STD_POLICY,
    SIGN_CI_POLICY,
    SPEARMAN_CI_LOWER_TOLERANCE,
    TOP1_CI_LOWER_TOLERANCE,
    evaluate_sign_ci_stability,
    mean_std as _shared_mean_std,
    sign_inconsistency_count as _shared_sign_inconsistency_count,
    validate_decision_policy_version,
)


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    return _shared_mean_std(values)


def _sign_inconsistency_count(values: Sequence[float]) -> int:
    return _shared_sign_inconsistency_count(values)


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
    feature_regime = str(r.get("feature_regime", "") or feature_set)
    probe_mode = str(r.get("probe_feature_mode", "off"))
    interaction_mode = str(r.get("interaction_feature_mode", "off"))
    arm = str(r.get("disentanglement_arm", "default"))
    if method == "metadata_routing":
        return "metadata_routing"
    return f"{feature_regime}::{method}__{feature_set}__probe_{probe_mode}__interact_{interaction_mode}__arm_{arm}"


def _row_feature_regime(row: dict) -> str:
    raw = str(row.get("feature_regime", "")).strip()
    if raw:
        return raw
    method = str(row.get("method", ""))
    feature_set = str(row.get("feature_set", ""))
    response_mode = str(row.get("response_feature_mode", "off"))
    if method == "metadata_routing":
        return "static_metadata"
    if response_mode == "on":
        return "response_indirect"
    if feature_set == "A":
        return "static_metadata"
    if feature_set == "B":
        return "static_combined"
    return feature_set or "unknown"


def _split_feature_names(raw: object) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [item for item in text.split("|") if item]


def _decision_blocked_terms_for_feature(feature_name: str) -> List[str]:
    """Return deployability veto terms for feature names.

    The matrix builder owns regime-aware static/response isolation. At decision
    time, feature-name scanning is a final guard against utility leakage, not a
    reason to reject deployable static metadata/embedding features.
    """

    name = str(feature_name)
    terms = []
    for term in ["nelbo", "recon_mean", "kl_mean"]:
        if term in name:
            terms.append(term)
    if name in {"query_id", "expert_id", "domain_id", "source_domain", "target_domain", "oracle_utility"}:
        terms.append(name)
    if name.startswith("oracle_") or name.startswith("target_"):
        terms.append(name.split("_", 1)[0] + "_")
    return terms


def _veto_for_method(method_key: str, vals: Sequence[dict]) -> Tuple[bool, str, List[str], List[str], str]:
    regimes = sorted(set(_row_feature_regime(v) for v in vals))
    diagnostic = False
    control = False
    adoption_eligible = False
    reasons: List[str] = []
    blocked_features: List[str] = []
    blocked_terms: List[str] = []

    for regime_name in regimes:
        try:
            regime = get_feature_regime(regime_name)
        except Exception:
            reasons.append("unknown_feature_regime")
            continue
        diagnostic = diagnostic or bool(regime.diagnostic_only)
        control = control or bool(regime.control_only)
        adoption_eligible = adoption_eligible or bool(regime.adoption_eligible)

    if diagnostic:
        reasons.append("diagnostic_or_target_derived_features")
    if control:
        reasons.append("control_only")
    if not adoption_eligible and method_key != "metadata_routing":
        reasons.append("not_adoption_eligible")

    for row in vals:
        for field in _split_feature_names(row.get("feature_names", "")):
            terms = _decision_blocked_terms_for_feature(field)
            if terms:
                blocked_features.append(field)
                blocked_terms.extend(terms)
        for field in _split_feature_names(row.get("blocked_features", "")):
            blocked_features.append(field)
        for term in _split_feature_names(row.get("blocked_feature_terms", "")):
            blocked_terms.append(term)

    for token_source in [method_key] + regimes:
        lowered = str(token_source).lower()
        for token in ["diagnostic", "target_adjacent", "oracle"]:
            if token in lowered:
                reasons.append("diagnostic_or_target_derived_features")
        for term in ["nelbo", "recon_mean", "kl_mean"]:
            if term in lowered:
                blocked_terms.append(term)

    if blocked_features or blocked_terms:
        reasons.append("blocked_features")

    vetoed = bool(reasons)
    return (
        vetoed,
        "|".join(sorted(set(reasons))) if reasons else "",
        sorted(set(blocked_features)),
        sorted(set(blocked_terms)),
        "|".join(regimes),
    )


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
        feature_names = "|".join(str(v.get("feature_names", "")) for v in vals if str(v.get("feature_names", "")))
        blocked_features = "|".join(str(v.get("blocked_features", "")) for v in vals if str(v.get("blocked_features", "")))
        blocked_terms = "|".join(
            str(v.get("blocked_feature_terms", "")) for v in vals if str(v.get("blocked_feature_terms", ""))
        )

        out[key] = {
            "dataset_name": dataset_name,
            "backbone_type": backbone_type,
            "run_id": run_id,
            "variant": variant,
            "method_key": method_key,
            "feature_regime": _row_feature_regime(vals[0]),
            "feature_names": "|".join(sorted(set(feature_names.split("|")))) if feature_names else "",
            "blocked_features": "|".join(sorted(set(blocked_features.split("|")))) if blocked_features else "",
            "blocked_feature_terms": "|".join(sorted(set(blocked_terms.split("|")))) if blocked_terms else "",
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


def _granular_key(r: dict) -> Tuple[str, str, str, str, str]:
    fold = str(r.get("heldout_query_domain", "") or r.get("query_domain", "") or r.get("fold_id", ""))
    return _run_key(r) + (fold,)


def _paired_granular_uplifts(
    rows: Sequence[dict],
    *,
    uplift_reference_method: str,
) -> Dict[str, List[dict]]:
    by_group: Dict[Tuple[str, str, str, str, str], Dict[str, dict]] = {}
    for row in rows:
        by_group.setdefault(_granular_key(row), {})[_method_key(row)] = row

    out: Dict[str, List[dict]] = {}
    for group_key, methods in by_group.items():
        baseline = methods.get(str(uplift_reference_method))
        if baseline is None:
            continue
        for method_key, row in methods.items():
            out.setdefault(method_key, []).append(
                {
                    "group_key": "|".join(group_key),
                    "query_domain": group_key[-1],
                    "top1_uplift": _to_float(row.get("top1_agreement_with_best_expert", 0.0))
                    - _to_float(baseline.get("top1_agreement_with_best_expert", 0.0)),
                    "spearman_uplift": _to_float(row.get("spearman_similarity_vs_neg_nelbo", 0.0))
                    - _to_float(baseline.get("spearman_similarity_vs_neg_nelbo", 0.0)),
                    "gap_reduction": _to_float(baseline.get("metadata_to_oracle_gap", 0.0))
                    - _to_float(row.get("metadata_to_oracle_gap", 0.0)),
                    "normalized_gap_reduction": _to_float(
                        baseline.get("normalized_metadata_to_oracle_gap", 0.0)
                    )
                    - _to_float(row.get("normalized_metadata_to_oracle_gap", 0.0)),
                }
            )
    return out


def _catastrophic_regression_report(granular_uplifts: Sequence[dict]) -> Dict[str, object]:
    report: Dict[str, object] = {
        "catastrophic_regression_breach": 0,
        "catastrophic_regression_metric": "",
        "catastrophic_regression_query_domain": "",
        "catastrophic_regression_group_key": "",
        "worst_domain_top1_uplift": 0.0,
        "worst_domain_spearman_uplift": 0.0,
        "worst_domain_gap_reduction": 0.0,
    }
    if not granular_uplifts:
        return report
    worst_top1 = min(granular_uplifts, key=lambda r: float(r["top1_uplift"]))
    worst_spearman = min(granular_uplifts, key=lambda r: float(r["spearman_uplift"]))
    worst_gap = min(granular_uplifts, key=lambda r: float(r["gap_reduction"]))
    report.update(
        {
            "worst_domain_top1_uplift": float(worst_top1["top1_uplift"]),
            "worst_domain_spearman_uplift": float(worst_spearman["spearman_uplift"]),
            "worst_domain_gap_reduction": float(worst_gap["gap_reduction"]),
        }
    )
    breaches = [
        ("top1", worst_top1, CATASTROPHIC_TOP1_UPLIFT_MIN, "top1_uplift"),
        ("spearman", worst_spearman, CATASTROPHIC_SPEARMAN_UPLIFT_MIN, "spearman_uplift"),
        ("gap", worst_gap, CATASTROPHIC_GAP_PCT_REDUCTION_MIN, "gap_reduction"),
    ]
    for metric, row, threshold, key in breaches:
        if float(row[key]) < float(threshold):
            report.update(
                {
                    "catastrophic_regression_breach": 1,
                    "catastrophic_regression_metric": str(metric),
                    "catastrophic_regression_query_domain": str(row.get("query_domain", "")),
                    "catastrophic_regression_group_key": str(row.get("group_key", "")),
                }
            )
            break
    return report


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
    decision_policy_version: str = LEGACY_STD_POLICY,
    top1_uplift_std_threshold: float = 0.05,
    spearman_uplift_std_threshold: float = 0.05,
    gap_reduction_std_threshold: float = 0.005,
    normalized_gap_reduction_std_threshold: float = 0.01,
    min_positive_fraction: float = 0.67,
    ci_level: float = 0.95,
    ci_bootstrap_reps: int = 10000,
    ci_bootstrap_seed: int = 1337,
) -> Tuple[List[dict], Dict[str, object]]:
    decision_policy_version = validate_decision_policy_version(decision_policy_version)
    per_run = _aggregate_per_run(rows)
    granular_uplifts_by_method = _paired_granular_uplifts(
        rows,
        uplift_reference_method=str(uplift_reference_method),
    )

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
        vetoed, veto_reason, blocked_features, blocked_terms, feature_regimes = _veto_for_method(method_key, vals)

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

        if decision_policy_version == LEGACY_STD_POLICY:
            resolved_top1_std_threshold = float(instability_std_threshold)
            resolved_spearman_std_threshold = float(instability_std_threshold)
            resolved_gap_std_threshold = float(instability_std_threshold)
            resolved_norm_gap_std_threshold = float(instability_std_threshold)
        else:
            resolved_top1_std_threshold = float(top1_uplift_std_threshold)
            resolved_spearman_std_threshold = float(spearman_uplift_std_threshold)
            resolved_gap_std_threshold = float(gap_reduction_std_threshold)
            resolved_norm_gap_std_threshold = float(normalized_gap_reduction_std_threshold)

        top1_std_breach = bool(top1_uplift_std > resolved_top1_std_threshold)
        spearman_std_breach = bool(spearman_uplift_std > resolved_spearman_std_threshold)
        gap_std_breach = bool(gap_reduction_std > resolved_gap_std_threshold)
        norm_gap_std_breach = bool(norm_gap_reduction_std > resolved_norm_gap_std_threshold)
        std_breach = bool(top1_std_breach or spearman_std_breach or gap_std_breach or norm_gap_std_breach)
        sign_inconsistency_count = (
            _sign_inconsistency_count(top1_uplifts)
            + _sign_inconsistency_count(spearman_uplifts)
            + _sign_inconsistency_count(gap_reductions)
            + _sign_inconsistency_count(norm_gap_reductions)
        )
        sign_breach = bool(sign_inconsistency_count >= int(instability_sign_inconsistency_min_count))
        legacy_instability_breach = bool(std_breach or sign_breach)

        granular_uplifts = granular_uplifts_by_method.get(method_key, [])
        granular_top1 = [float(r["top1_uplift"]) for r in granular_uplifts] or top1_uplifts
        granular_spearman = [float(r["spearman_uplift"]) for r in granular_uplifts] or spearman_uplifts
        granular_gap = [float(r["gap_reduction"]) for r in granular_uplifts] or gap_reductions
        granular_norm_gap = [
            float(r["normalized_gap_reduction"]) for r in granular_uplifts
        ] or norm_gap_reductions
        catastrophic_report = _catastrophic_regression_report(granular_uplifts)
        catastrophic_breach = bool(int(catastrophic_report["catastrophic_regression_breach"]))
        ci_source = "domain" if granular_uplifts else ("seed_descriptive" if n_runs <= 3 else "seed")
        sign_ci_report = evaluate_sign_ci_stability(
            metric_values={
                "top1": granular_top1,
                "spearman": granular_spearman,
                "gap": granular_gap,
                "normalized_gap": granular_norm_gap,
            },
            metric_means={
                "top1": top1_uplift_mean,
                "spearman": spearman_uplift_mean,
                "gap": gap_reduction_mean,
                "normalized_gap": norm_gap_reduction_mean,
            },
            min_improving_runs=int(min_improving_runs),
            min_positive_fraction=float(min_positive_fraction),
            ci_level=float(ci_level),
            ci_bootstrap_reps=int(ci_bootstrap_reps),
            ci_bootstrap_seed=int(ci_bootstrap_seed),
            ci_source=str(ci_source),
            ci_lower_tolerances={
                "top1": TOP1_CI_LOWER_TOLERANCE,
                "spearman": SPEARMAN_CI_LOWER_TOLERANCE,
                "gap": GAP_PCT_CI_LOWER_TOLERANCE,
                "normalized_gap": 0.0,
            },
            catastrophic_regression_breach=catastrophic_breach,
            regression_check_missing=False,
        )
        if decision_policy_version == LEGACY_STD_POLICY:
            instability_breach = bool(legacy_instability_breach)
            tier_improving_count = int(improving_run_count)
            tier_min_improving = int(min_improving_runs)
        else:
            instability_breach = not bool(int(sign_ci_report["sign_ci_stability_pass"]))
            tier_improving_count = int(sign_ci_report["gap_positive_count"])
            tier_min_improving = int(sign_ci_report["positive_observation_threshold"])

        if method_key == str(uplift_reference_method):
            tier = "baseline"
        else:
            tier = _tier(
                improving_run_count=int(tier_improving_count),
                min_improving_runs=int(tier_min_improving),
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
            and not vetoed
        )

        out_rows.append(
            {
                "decision_policy_version": str(decision_policy_version),
                "method_key": method_key,
                "feature_regime": feature_regimes,
                "adoption_eligible": int(not vetoed and method_key != str(uplift_reference_method)),
                "diagnostic_only": int("diagnostic_or_target_derived_features" in veto_reason),
                "control_only": int("control_only" in veto_reason),
                "veto_reason": veto_reason,
                "blocked_features": serialize_feature_list(blocked_features),
                "blocked_feature_terms": serialize_feature_list(blocked_terms),
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
                "top1_uplift_std_threshold": float(resolved_top1_std_threshold),
                "spearman_uplift_std_threshold": float(resolved_spearman_std_threshold),
                "gap_reduction_std_threshold": float(resolved_gap_std_threshold),
                "normalized_gap_reduction_std_threshold": float(resolved_norm_gap_std_threshold),
                "top1_uplift_std_breach": int(top1_std_breach),
                "spearman_uplift_std_breach": int(spearman_std_breach),
                "gap_reduction_std_breach": int(gap_std_breach),
                "normalized_gap_reduction_std_breach": int(norm_gap_std_breach),
                "instability_std_breach": int(std_breach),
                "instability_sign_inconsistency_count": int(sign_inconsistency_count),
                "instability_breach": int(instability_breach),
                "granular_uplift_rows": int(len(granular_uplifts)),
                **catastrophic_report,
                "positive_observation_count": int(sign_ci_report["positive_observation_count"]),
                "positive_observation_threshold": int(sign_ci_report["positive_observation_threshold"]),
                "top1_positive_count": int(sign_ci_report["top1_positive_count"]),
                "spearman_positive_count": int(sign_ci_report["spearman_positive_count"]),
                "gap_positive_count": int(sign_ci_report["gap_positive_count"]),
                "normalized_gap_positive_count": int(sign_ci_report["normalized_gap_positive_count"]),
                "ci_source": str(sign_ci_report["ci_source"]),
                "ci_hard_gate_applied": int(sign_ci_report["ci_hard_gate_applied"]),
                "top1_ci_low": float(sign_ci_report["top1_ci_low"]),
                "top1_ci_high": float(sign_ci_report["top1_ci_high"]),
                "spearman_ci_low": float(sign_ci_report["spearman_ci_low"]),
                "spearman_ci_high": float(sign_ci_report["spearman_ci_high"]),
                "gap_ci_low": float(sign_ci_report["gap_ci_low"]),
                "gap_ci_high": float(sign_ci_report["gap_ci_high"]),
                "normalized_gap_ci_low": float(sign_ci_report["normalized_gap_ci_low"]),
                "normalized_gap_ci_high": float(sign_ci_report["normalized_gap_ci_high"]),
                "sign_ci_stability_pass": int(sign_ci_report["sign_ci_stability_pass"]),
                "joint_top1_gap_guardrail_pass": int(joint_top1_gap_guardrail_pass),
                "uncertainty_calibration_gate_pass": int(uncertainty_calibration_gate_pass),
                "adoption_gate_pass_proxy": int(adoption_gate_pass_proxy),
                "tier": tier,
            }
        )

    out_rows.sort(key=lambda r: (str(r["tier"]), -float(r["spearman_uplift_vs_metadata_mean"]), str(r["method_key"])))

    summary = {
        "decision_policy_version": str(decision_policy_version),
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
    lines.append(f"- Decision policy version: {summary['decision_policy_version']}")
    lines.append(f"- Total methods: {int(summary['total_methods'])}")
    lines.append(f"- Strong pass: {int(summary['strong_pass_count'])}")
    lines.append(f"- Weak pass: {int(summary['weak_pass_count'])}")
    lines.append(f"- Fail: {int(summary['fail_count'])}")
    lines.append("")
    lines.append("| Method | Regime | Tier | Veto | Runs | Top1 mean+-std | Spearman mean+-std | Gap mean+-std | NormGap mean+-std | CalErr mean+-std | Top1 uplift | Spearman uplift | Gap reduction | NormGap reduction | CalErr reduction | Positive threshold | CI source | Catastrophic | Joint guardrail | Cal gate | Adoption gate |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {} | {} | {} | {} | {} | {} |".format(
                r["method_key"],
                r.get("feature_regime", ""),
                r["tier"],
                r.get("veto_reason", ""),
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
                int(r["positive_observation_threshold"]),
                str(r["ci_source"]),
                int(r["catastrophic_regression_breach"]),
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
    p.add_argument(
        "--decision-policy-version",
        type=str,
        default=SIGN_CI_POLICY,
        choices=[LEGACY_STD_POLICY, SIGN_CI_POLICY],
    )
    p.add_argument("--top1-uplift-std-threshold", type=float, default=0.05)
    p.add_argument("--spearman-uplift-std-threshold", type=float, default=0.05)
    p.add_argument("--gap-reduction-std-threshold", type=float, default=0.005)
    p.add_argument("--normalized-gap-reduction-std-threshold", type=float, default=0.01)
    p.add_argument("--min-positive-fraction", type=float, default=0.67)
    p.add_argument("--ci-level", type=float, default=0.95)
    p.add_argument("--ci-bootstrap-reps", type=int, default=10000)
    p.add_argument("--ci-bootstrap-seed", type=int, default=1337)
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
        decision_policy_version=str(args.decision_policy_version),
        top1_uplift_std_threshold=float(args.top1_uplift_std_threshold),
        spearman_uplift_std_threshold=float(args.spearman_uplift_std_threshold),
        gap_reduction_std_threshold=float(args.gap_reduction_std_threshold),
        normalized_gap_reduction_std_threshold=float(args.normalized_gap_reduction_std_threshold),
        min_positive_fraction=float(args.min_positive_fraction),
        ci_level=float(args.ci_level),
        ci_bootstrap_reps=int(args.ci_bootstrap_reps),
        ci_bootstrap_seed=int(args.ci_bootstrap_seed),
    )

    _write_csv(args.output_csv, out_rows)
    _write_md(args.output_md, out_rows, summary)

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
