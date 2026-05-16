#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compatibility_stability import (  # noqa: E402
    CATASTROPHIC_GAP_PCT_REDUCTION_MIN,
    CATASTROPHIC_SPEARMAN_UPLIFT_MIN,
    CATASTROPHIC_TOP1_UPLIFT_MIN,
    GAP_PCT_CI_LOWER_TOLERANCE,
    LEGACY_STD_POLICY,
    SIGN_CI_POLICY,
    SPEARMAN_CI_LOWER_TOLERANCE,
    TOP1_CI_LOWER_TOLERANCE,
    evaluate_sign_ci_stability,
    finite_values,
    mean_std as _shared_mean_std,
    sign_inconsistency_count as _shared_sign_inconsistency_count,
    validate_decision_policy_version,
)


EXPECTED_PROTOCOL_VERSION = "learned_utility_loqdo_candidate_exclusion_v2"
REQUIRED_METHOD_POLICY_FIELDS = {
    "protocol_version",
    "method_role",
    "adoption_eligible",
    "diagnostic_only",
    "routing_uses_eval_nelbo",
    "routing_uses_eval_domain_statistics",
}


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    return _shared_mean_std(values)


def _load_manifest(manifest_path: Path) -> List[Path]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    paths: List[Path] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"Manifest entry does not exist: {p}")
        paths.append(p)

    if not paths:
        raise RuntimeError("Manifest is empty; no result json files found.")
    return paths


def _seed_from_path(path: Path, fallback: int) -> int:
    text = str(path)
    m = re.search(r"seed(\d+)", text)
    if m is not None:
        return int(m.group(1))
    return int(fallback)


def _sign_inconsistency_count(values: Sequence[float]) -> int:
    return _shared_sign_inconsistency_count(values)


def _validate_result_protocol(path: Path, payload: Dict[str, Any]) -> str:
    protocol_version = str(payload.get("protocol_version", "")).strip()
    contract = payload.get("protocol_contract", {})
    if not isinstance(contract, dict):
        raise RuntimeError(f"Result JSON has invalid protocol_contract: {path}")

    contract_version = str(contract.get("protocol_version", "")).strip()
    if protocol_version != EXPECTED_PROTOCOL_VERSION or contract_version != EXPECTED_PROTOCOL_VERSION:
        raise RuntimeError(
            "Decision table requires learned utility LOQDO v2 artifacts. "
            f"Expected protocol_version={EXPECTED_PROTOCOL_VERSION}; "
            f"got top_level={protocol_version or '<missing>'} "
            f"contract={contract_version or '<missing>'} in {path}"
        )
    return protocol_version


def _validate_method_policy_fields(path: Path, method: str, metrics: Dict[str, Any]) -> None:
    missing = sorted(field for field in REQUIRED_METHOD_POLICY_FIELDS if field not in metrics)
    if missing:
        raise RuntimeError(
            f"Method '{method}' in {path} is missing required v2 method policy fields: {missing}"
        )

    method_protocol = str(metrics.get("protocol_version", "")).strip()
    if method_protocol != EXPECTED_PROTOCOL_VERSION:
        raise RuntimeError(
            f"Method '{method}' in {path} has protocol_version={method_protocol or '<missing>'}; "
            f"expected {EXPECTED_PROTOCOL_VERSION}"
        )


def _is_selectable_method(row: Dict[str, Any], uplift_reference_method: str) -> bool:
    if str(row.get("method", "")) == str(uplift_reference_method):
        return False
    return bool(
        _to_int(row.get("adoption_eligible", 0)) == 1
        and _to_int(row.get("diagnostic_only", 0)) == 0
        and _to_int(row.get("routing_uses_eval_nelbo", 0)) == 0
        and _to_int(row.get("routing_uses_eval_domain_statistics", 0)) == 0
    )


def _tier(
    *,
    improving_seed_count: int,
    min_improving_seeds: int,
    spearman_uplift_mean: float,
    top1_uplift_mean: float,
    gap_reduction_mean: float,
    strong: Dict[str, float],
    weak: Dict[str, float],
    instability_breach: bool,
) -> str:
    if instability_breach:
        return "fail"

    strong_ok = (
        improving_seed_count >= int(min_improving_seeds)
        and spearman_uplift_mean >= float(strong["spearman_uplift_min"])
        and top1_uplift_mean >= float(strong["top1_uplift_min"])
        and gap_reduction_mean >= float(strong["oracle_gap_pct_reduction_min"])
    )
    if strong_ok:
        return "strong_pass"

    weak_ok = (
        improving_seed_count >= int(min_improving_seeds)
        and spearman_uplift_mean >= float(weak["spearman_uplift_min"])
        and top1_uplift_mean >= float(weak["top1_uplift_min"])
        and gap_reduction_mean >= float(weak["oracle_gap_pct_reduction_min"])
    )
    if weak_ok:
        return "weak_pass"

    return "fail"


def _read_rows(
    result_paths: Sequence[Path],
    uplift_reference_method: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_protocol_versions: set[str] = set()
    for idx, path in enumerate(result_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seen_protocol_versions.add(_validate_result_protocol(path, payload))
        metrics_by_method = payload.get("metrics_by_method", {})
        if not isinstance(metrics_by_method, dict) or not metrics_by_method:
            continue
        if str(uplift_reference_method) not in metrics_by_method:
            raise RuntimeError(
                f"uplift_reference_method='{uplift_reference_method}' is missing from metrics_by_method in {path}"
            )

        seed = _seed_from_path(path, fallback=idx)
        baseline = metrics_by_method.get(uplift_reference_method, {})
        b_top1 = _to_float((baseline or {}).get("top1_oracle_hit", 0.0))
        b_spearman = _to_float((baseline or {}).get("spearman", 0.0))
        b_gap_pct = _to_float((baseline or {}).get("mean_oracle_gap_pct", 0.0))

        for method, m in metrics_by_method.items():
            mm = m or {}
            if not isinstance(mm, dict):
                raise RuntimeError(f"Metrics for method '{method}' in {path} must be a dictionary")
            _validate_method_policy_fields(path, str(method), mm)
            top1 = _to_float(mm.get("top1_oracle_hit", 0.0))
            spearman = _to_float(mm.get("spearman", 0.0))
            gap_pct = _to_float(mm.get("mean_oracle_gap_pct", 0.0))
            rows.append(
                {
                    "seed": int(seed),
                    "protocol_version": str(mm["protocol_version"]),
                    "method": str(method),
                    "method_role": str(mm["method_role"]),
                    "adoption_eligible": _to_int(mm["adoption_eligible"]),
                    "diagnostic_only": _to_int(mm["diagnostic_only"]),
                    "routing_uses_eval_nelbo": _to_int(mm["routing_uses_eval_nelbo"]),
                    "routing_uses_eval_domain_statistics": _to_int(
                        mm["routing_uses_eval_domain_statistics"]
                    ),
                    "top1_oracle_hit": top1,
                    "spearman": spearman,
                    "mean_oracle_gap_pct": gap_pct,
                    "top1_uplift_vs_metadata": float(top1 - b_top1),
                    "spearman_uplift_vs_metadata": float(spearman - b_spearman),
                    "oracle_gap_pct_reduction_vs_metadata": float(b_gap_pct - gap_pct),
                    "source_json": str(path),
                    "decision_policy_version": str(mm.get("decision_policy_version", "")),
                    "residual_policy_version": str(mm.get("residual_policy_version", "")),
                    "threshold_selection_policy": str(mm.get("threshold_selection_policy", "")),
                    "feature_set": str(mm.get("feature_set", "")),
                    "residual_variant": str(mm.get("residual_variant", "")),
                    "selected_tau": str(mm.get("selected_tau", "")),
                    "selected_by_inner_validation": _to_int(mm.get("selected_by_inner_validation", 0)),
                    "adoption_selected_method": str(mm.get("adoption_selected_method", "")),
                    "harmful_override_max": str(mm.get("harmful_override_max", "")),
                    "allow_calibrated_adoption": str(mm.get("allow_calibrated_adoption", "")),
                    "fallback_used": str(mm.get("fallback_used", "")),
                }
            )
    if len(seen_protocol_versions) > 1:
        raise RuntimeError(f"Mixed protocol versions in manifest are not allowed: {sorted(seen_protocol_versions)}")
    return rows


def _read_domain_breakdown_rows(source_json: str) -> Tuple[List[Dict[str, Any]], bool]:
    path = Path(str(source_json))
    if not path.exists():
        return [], True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [], True
    artifacts = payload.get("artifacts", {})
    artifact_name = ""
    if isinstance(artifacts, dict):
        artifact_name = str(artifacts.get("domain_breakdown", "")).strip()
    if not artifact_name:
        artifact_name = "learned_utility_domain_breakdown.csv"

    domain_path = path.parent / artifact_name
    if not domain_path.exists():
        return [], True
    with domain_path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)], False


def _paired_domain_uplifts(
    *,
    source_paths: Sequence[str],
    method: str,
    uplift_reference_method: str,
    domain_cache: Dict[str, Tuple[List[Dict[str, Any]], bool]],
) -> Tuple[List[Dict[str, Any]], bool]:
    paired: List[Dict[str, Any]] = []
    missing = False
    for source_json in sorted(set(str(p) for p in source_paths)):
        if source_json not in domain_cache:
            domain_cache[source_json] = _read_domain_breakdown_rows(source_json)
        rows, source_missing = domain_cache[source_json]
        if source_missing:
            missing = True
            continue
        by_key = {
            (str(r.get("method", "")), int(float(r.get("query_domain", 0)))): r
            for r in rows
            if str(r.get("method", ""))
        }
        query_domains = sorted(
            q for m, q in by_key.keys() if m == str(uplift_reference_method)
        )
        for q in query_domains:
            base = by_key.get((str(uplift_reference_method), q))
            cand = by_key.get((str(method), q))
            if base is None or cand is None:
                missing = True
                continue
            paired.append(
                {
                    "source_json": source_json,
                    "query_domain": int(q),
                    "top1_uplift": _to_float(cand.get("top1_oracle_hit", 0.0))
                    - _to_float(base.get("top1_oracle_hit", 0.0)),
                    "spearman_uplift": _to_float(cand.get("spearman", 0.0))
                    - _to_float(base.get("spearman", 0.0)),
                    "gap_pct_reduction": _to_float(base.get("mean_oracle_gap_pct", 0.0))
                    - _to_float(cand.get("mean_oracle_gap_pct", 0.0)),
                }
            )
    return paired, bool(missing)


def _catastrophic_regression_report(domain_uplifts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "catastrophic_regression_breach": 0,
        "catastrophic_regression_metric": "",
        "catastrophic_regression_query_domain": "",
        "catastrophic_regression_source_json": "",
        "worst_domain_top1_uplift": 0.0,
        "worst_domain_spearman_uplift": 0.0,
        "worst_domain_gap_pct_reduction": 0.0,
    }
    if not domain_uplifts:
        return report

    worst_top1 = min(domain_uplifts, key=lambda r: float(r["top1_uplift"]))
    worst_spearman = min(domain_uplifts, key=lambda r: float(r["spearman_uplift"]))
    worst_gap = min(domain_uplifts, key=lambda r: float(r["gap_pct_reduction"]))
    report.update(
        {
            "worst_domain_top1_uplift": float(worst_top1["top1_uplift"]),
            "worst_domain_spearman_uplift": float(worst_spearman["spearman_uplift"]),
            "worst_domain_gap_pct_reduction": float(worst_gap["gap_pct_reduction"]),
        }
    )

    breaches = [
        ("top1", worst_top1, CATASTROPHIC_TOP1_UPLIFT_MIN, "top1_uplift"),
        ("spearman", worst_spearman, CATASTROPHIC_SPEARMAN_UPLIFT_MIN, "spearman_uplift"),
        ("gap_pct", worst_gap, CATASTROPHIC_GAP_PCT_REDUCTION_MIN, "gap_pct_reduction"),
    ]
    breached = [
        (metric, row, key)
        for metric, row, threshold, key in breaches
        if float(row[key]) < float(threshold)
    ]
    if breached:
        metric, row, _key = breached[0]
        report.update(
            {
                "catastrophic_regression_breach": 1,
                "catastrophic_regression_metric": str(metric),
                "catastrophic_regression_query_domain": str(row.get("query_domain", "")),
                "catastrophic_regression_source_json": str(row.get("source_json", "")),
            }
        )
    return report


def _ci_source_for_values(domain_uplifts: Sequence[Dict[str, Any]], seed_values: Sequence[float]) -> str:
    if domain_uplifts:
        return "domain"
    return "seed_descriptive" if len(finite_values(seed_values)) <= 3 else "seed"


def _aggregate(
    rows: Sequence[Dict[str, Any]],
    uplift_reference_method: str,
    min_improving_seeds: int,
    strong: Dict[str, float],
    weak: Dict[str, float],
    instability_std_threshold: float,
    instability_sign_inconsistency_min_count: int,
    decision_policy_version: str = LEGACY_STD_POLICY,
    top1_uplift_std_threshold: float = 0.05,
    spearman_uplift_std_threshold: float = 0.05,
    gap_pct_reduction_std_threshold: float = 3.0,
    min_positive_fraction: float = 0.67,
    ci_level: float = 0.95,
    ci_bootstrap_reps: int = 10000,
    ci_bootstrap_seed: int = 1337,
    allow_missing_domain_breakdown_as_diagnostic: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    decision_policy_version = validate_decision_policy_version(decision_policy_version)
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(str(r["method"]), []).append(r)

    out_rows: List[Dict[str, Any]] = []
    domain_cache: Dict[str, Tuple[List[Dict[str, Any]], bool]] = {}
    for method in sorted(by_method.keys()):
        method_rows = sorted(by_method[method], key=lambda r: int(r["seed"]))
        seeds = [int(r["seed"]) for r in method_rows]

        top1_vals = [float(r["top1_oracle_hit"]) for r in method_rows]
        spearman_vals = [float(r["spearman"]) for r in method_rows]
        gap_vals = [float(r["mean_oracle_gap_pct"]) for r in method_rows]

        top1_uplifts = [float(r["top1_uplift_vs_metadata"]) for r in method_rows]
        spearman_uplifts = [float(r["spearman_uplift_vs_metadata"]) for r in method_rows]
        gap_reductions = [float(r["oracle_gap_pct_reduction_vs_metadata"]) for r in method_rows]

        top1_mean, top1_std = _mean_std(top1_vals)
        spearman_mean, spearman_std = _mean_std(spearman_vals)
        gap_mean, gap_std = _mean_std(gap_vals)

        top1_uplift_mean, top1_uplift_std = _mean_std(top1_uplifts)
        spearman_uplift_mean, spearman_uplift_std = _mean_std(spearman_uplifts)
        gap_reduction_mean, gap_reduction_std = _mean_std(gap_reductions)

        improving_seed_count = sum(
            1
            for i in range(len(method_rows))
            if top1_uplifts[i] > 0.0 and spearman_uplifts[i] > 0.0 and gap_reductions[i] > 0.0
        )

        if decision_policy_version == LEGACY_STD_POLICY:
            resolved_top1_std_threshold = float(instability_std_threshold)
            resolved_spearman_std_threshold = float(instability_std_threshold)
            resolved_gap_std_threshold = float(instability_std_threshold)
        else:
            resolved_top1_std_threshold = float(top1_uplift_std_threshold)
            resolved_spearman_std_threshold = float(spearman_uplift_std_threshold)
            resolved_gap_std_threshold = float(gap_pct_reduction_std_threshold)

        top1_std_breach = bool(top1_uplift_std > resolved_top1_std_threshold)
        spearman_std_breach = bool(spearman_uplift_std > resolved_spearman_std_threshold)
        gap_std_breach = bool(gap_reduction_std > resolved_gap_std_threshold)
        std_breach = bool(top1_std_breach or spearman_std_breach or gap_std_breach)
        sign_inconsistency_count = (
            _sign_inconsistency_count(top1_uplifts)
            + _sign_inconsistency_count(spearman_uplifts)
            + _sign_inconsistency_count(gap_reductions)
        )
        sign_breach = bool(sign_inconsistency_count >= int(instability_sign_inconsistency_min_count))
        legacy_instability_breach = bool(std_breach or sign_breach)

        base = method_rows[0]
        base_selectable = _is_selectable_method(base, uplift_reference_method=str(uplift_reference_method))

        domain_uplifts: List[Dict[str, Any]] = []
        domain_missing = False
        if decision_policy_version == SIGN_CI_POLICY and base_selectable:
            domain_uplifts, domain_missing = _paired_domain_uplifts(
                source_paths=[str(r.get("source_json", "")) for r in method_rows],
                method=str(method),
                uplift_reference_method=str(uplift_reference_method),
                domain_cache=domain_cache,
            )

        regression_check_missing = bool(
            decision_policy_version == SIGN_CI_POLICY and base_selectable and domain_missing
        )
        evidence_only = bool(regression_check_missing and allow_missing_domain_breakdown_as_diagnostic)
        selectable = bool(base_selectable and not evidence_only)
        instability_gate_applied = bool(selectable)

        catastrophic_report = _catastrophic_regression_report(domain_uplifts)
        catastrophic_breach = bool(int(catastrophic_report["catastrophic_regression_breach"]))

        granular_top1 = [float(r["top1_uplift"]) for r in domain_uplifts] or top1_uplifts
        granular_spearman = [float(r["spearman_uplift"]) for r in domain_uplifts] or spearman_uplifts
        granular_gap = [float(r["gap_pct_reduction"]) for r in domain_uplifts] or gap_reductions
        ci_source = _ci_source_for_values(domain_uplifts, top1_uplifts)
        sign_ci_report = evaluate_sign_ci_stability(
            metric_values={
                "top1": granular_top1,
                "spearman": granular_spearman,
                "gap_pct": granular_gap,
            },
            metric_means={
                "top1": float(top1_uplift_mean),
                "spearman": float(spearman_uplift_mean),
                "gap_pct": float(gap_reduction_mean),
            },
            min_improving_runs=int(min_improving_seeds),
            min_positive_fraction=float(min_positive_fraction),
            ci_level=float(ci_level),
            ci_bootstrap_reps=int(ci_bootstrap_reps),
            ci_bootstrap_seed=int(ci_bootstrap_seed),
            ci_source=str(ci_source),
            ci_lower_tolerances={
                "top1": TOP1_CI_LOWER_TOLERANCE,
                "spearman": SPEARMAN_CI_LOWER_TOLERANCE,
                "gap_pct": GAP_PCT_CI_LOWER_TOLERANCE,
            },
            catastrophic_regression_breach=catastrophic_breach,
            regression_check_missing=bool(regression_check_missing),
        )
        sign_ci_breach = not bool(int(sign_ci_report["sign_ci_stability_pass"]))
        raw_instability_breach = bool(legacy_instability_breach)
        if decision_policy_version == LEGACY_STD_POLICY:
            instability_breach = bool(legacy_instability_breach and instability_gate_applied)
            tier_improving_count = int(improving_seed_count)
            tier_min_improving = int(min_improving_seeds)
        else:
            instability_breach = bool(sign_ci_breach and instability_gate_applied)
            tier_improving_count = int(sign_ci_report["gap_pct_positive_count"])
            tier_min_improving = int(sign_ci_report["positive_observation_threshold"])

        if method == str(uplift_reference_method):
            tier = "baseline"
        elif evidence_only:
            tier = "needs_evidence"
        elif not selectable:
            tier = "reference_only"
        else:
            tier = _tier(
                improving_seed_count=int(tier_improving_count),
                min_improving_seeds=int(tier_min_improving),
                spearman_uplift_mean=float(spearman_uplift_mean),
                top1_uplift_mean=float(top1_uplift_mean),
                gap_reduction_mean=float(gap_reduction_mean),
                strong=strong,
                weak=weak,
                instability_breach=instability_breach,
            )
            if (
                str(base.get("residual_policy_version", "")) == "metadata_residual_safe_override_v2"
                and str(tier) == "fail"
                and not catastrophic_breach
                and not instability_breach
                and float(top1_uplift_mean) >= 0.0
                and float(gap_reduction_mean) >= 0.0
            ):
                tier = "safe_no_gain"

        out_rows.append(
            {
                "protocol_version": str(base.get("protocol_version", "")),
                "decision_policy_version": str(decision_policy_version),
                "method": method,
                "method_role": str(base.get("method_role", "")),
                "adoption_eligible": _to_int(base.get("adoption_eligible", 0)),
                "diagnostic_only": _to_int(base.get("diagnostic_only", 0)),
                "routing_uses_eval_nelbo": _to_int(base.get("routing_uses_eval_nelbo", 0)),
                "routing_uses_eval_domain_statistics": _to_int(
                    base.get("routing_uses_eval_domain_statistics", 0)
                ),
                "residual_policy_version": str(base.get("residual_policy_version", "")),
                "threshold_selection_policy": str(base.get("threshold_selection_policy", "")),
                "feature_set": str(base.get("feature_set", "")),
                "residual_variant": str(base.get("residual_variant", "")),
                "selected_tau": str(base.get("selected_tau", "")),
                "selected_by_inner_validation": _to_int(base.get("selected_by_inner_validation", 0)),
                "adoption_selected_method": str(base.get("adoption_selected_method", "")),
                "harmful_override_max": str(base.get("harmful_override_max", "")),
                "allow_calibrated_adoption": str(base.get("allow_calibrated_adoption", "")),
                "fallback_used": str(base.get("fallback_used", "")),
                "selection_eligible": int(selectable),
                "n_seeds": int(len(seeds)),
                "seeds": ",".join(str(s) for s in seeds),
                "top1_oracle_hit_mean": float(top1_mean),
                "top1_oracle_hit_std": float(top1_std),
                "spearman_mean": float(spearman_mean),
                "spearman_std": float(spearman_std),
                "mean_oracle_gap_pct_mean": float(gap_mean),
                "mean_oracle_gap_pct_std": float(gap_std),
                "top1_uplift_vs_metadata_mean": float(top1_uplift_mean),
                "top1_uplift_vs_metadata_std": float(top1_uplift_std),
                "spearman_uplift_vs_metadata_mean": float(spearman_uplift_mean),
                "spearman_uplift_vs_metadata_std": float(spearman_uplift_std),
                "oracle_gap_pct_reduction_vs_metadata_mean": float(gap_reduction_mean),
                "oracle_gap_pct_reduction_vs_metadata_std": float(gap_reduction_std),
                "improving_seed_count": int(improving_seed_count),
                "instability_std_threshold": float(instability_std_threshold),
                "top1_uplift_std_threshold": float(resolved_top1_std_threshold),
                "spearman_uplift_std_threshold": float(resolved_spearman_std_threshold),
                "gap_pct_reduction_std_threshold": float(resolved_gap_std_threshold),
                "top1_uplift_std_breach": int(top1_std_breach),
                "spearman_uplift_std_breach": int(spearman_std_breach),
                "gap_pct_reduction_std_breach": int(gap_std_breach),
                "instability_std_breach": int(std_breach),
                "instability_sign_inconsistency_count": int(sign_inconsistency_count),
                "instability_sign_inconsistency_min_count": int(instability_sign_inconsistency_min_count),
                "raw_instability_breach": int(raw_instability_breach),
                "instability_gate_applied": int(instability_gate_applied),
                "instability_breach": int(instability_breach),
                "regression_check_missing": int(regression_check_missing),
                "domain_breakdown_rows": int(len(domain_uplifts)),
                "allow_missing_domain_breakdown_as_diagnostic": int(
                    allow_missing_domain_breakdown_as_diagnostic
                ),
                **catastrophic_report,
                "positive_observation_count": int(sign_ci_report["positive_observation_count"]),
                "positive_observation_threshold": int(sign_ci_report["positive_observation_threshold"]),
                "top1_positive_count": int(sign_ci_report["top1_positive_count"]),
                "spearman_positive_count": int(sign_ci_report["spearman_positive_count"]),
                "gap_pct_positive_count": int(sign_ci_report["gap_pct_positive_count"]),
                "ci_source": str(sign_ci_report["ci_source"]),
                "ci_hard_gate_applied": int(sign_ci_report["ci_hard_gate_applied"]),
                "top1_ci_low": float(sign_ci_report["top1_ci_low"]),
                "top1_ci_high": float(sign_ci_report["top1_ci_high"]),
                "spearman_ci_low": float(sign_ci_report["spearman_ci_low"]),
                "spearman_ci_high": float(sign_ci_report["spearman_ci_high"]),
                "gap_pct_ci_low": float(sign_ci_report["gap_pct_ci_low"]),
                "gap_pct_ci_high": float(sign_ci_report["gap_pct_ci_high"]),
                "sign_ci_stability_pass": int(sign_ci_report["sign_ci_stability_pass"]),
                "tier": str(tier),
            }
        )

    candidates = [
        r
        for r in out_rows
        if int(r["selection_eligible"]) == 1
    ]
    strong_candidates = [r for r in candidates if str(r["tier"]) == "strong_pass"]
    weak_candidates = [r for r in candidates if str(r["tier"]) == "weak_pass"]

    selected_method = str(uplift_reference_method)
    overall_tier = "fail"
    if strong_candidates:
        selected_method = str(
            max(strong_candidates, key=lambda r: float(r["oracle_gap_pct_reduction_vs_metadata_mean"]))["method"]
        )
        overall_tier = "strong_pass"
    elif weak_candidates:
        selected_method = str(
            max(weak_candidates, key=lambda r: float(r["oracle_gap_pct_reduction_vs_metadata_mean"]))["method"]
        )
        overall_tier = "weak_pass"

    for r in out_rows:
        if str(r["method"]) == str(uplift_reference_method):
            r["decision"] = "baseline_reference"
        elif str(r.get("tier", "")) == "needs_evidence":
            r["decision"] = "NEEDS_EVIDENCE"
        elif str(r["method"]) == selected_method:
            r["decision"] = "selected"
        else:
            r["decision"] = "not_selected"

    summary = {
        "decision_policy_version": str(decision_policy_version),
        "uplift_reference_method": str(uplift_reference_method),
        "overall_tier": str(overall_tier),
        "selected_method": str(selected_method),
        "min_improving_seeds": int(min_improving_seeds),
        "strong_thresholds": strong,
        "weak_thresholds": weak,
        "instability": {
            "std_threshold": float(instability_std_threshold),
            "top1_uplift_std_threshold": float(top1_uplift_std_threshold),
            "spearman_uplift_std_threshold": float(spearman_uplift_std_threshold),
            "gap_pct_reduction_std_threshold": float(gap_pct_reduction_std_threshold),
            "sign_inconsistency_min_count": int(instability_sign_inconsistency_min_count),
            "min_positive_fraction": float(min_positive_fraction),
            "ci_level": float(ci_level),
            "ci_bootstrap_reps": int(ci_bootstrap_reps),
            "ci_bootstrap_seed": int(ci_bootstrap_seed),
            "ci_lower_tolerances": {
                "top1": TOP1_CI_LOWER_TOLERANCE,
                "spearman": SPEARMAN_CI_LOWER_TOLERANCE,
                "gap_pct": GAP_PCT_CI_LOWER_TOLERANCE,
            },
            "catastrophic_regression_thresholds": {
                "top1": CATASTROPHIC_TOP1_UPLIFT_MIN,
                "spearman": CATASTROPHIC_SPEARMAN_UPLIFT_MIN,
                "gap_pct": CATASTROPHIC_GAP_PCT_REDUCTION_MIN,
            },
            "allow_missing_domain_breakdown_as_diagnostic": bool(
                allow_missing_domain_breakdown_as_diagnostic
            ),
        },
    }
    return out_rows, summary


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Compatibility Decision Table\n\n")
        f.write("- decision_policy_version: {}\n".format(summary["decision_policy_version"]))
        f.write("- uplift_reference_method: {}\n".format(summary["uplift_reference_method"]))
        f.write("- overall_tier: {}\n".format(summary["overall_tier"]))
        f.write("- selected_method: {}\n".format(summary["selected_method"]))
        f.write("- min_improving_seeds: {}\n".format(summary["min_improving_seeds"]))
        f.write(
            "- instability: std_threshold={} top1_std_threshold={} spearman_std_threshold={} gap_pct_std_threshold={} sign_inconsistency_min_count={}\n".format(
                summary["instability"]["std_threshold"],
                summary["instability"]["top1_uplift_std_threshold"],
                summary["instability"]["spearman_uplift_std_threshold"],
                summary["instability"]["gap_pct_reduction_std_threshold"],
                summary["instability"]["sign_inconsistency_min_count"],
            )
        )
        f.write("\n")
        f.write(
            "| method | role | eligible | decision | tier | n_seeds | top1 | spearman | mean_oracle_gap_pct | "
            "top1_uplift_vs_metadata | spearman_uplift_vs_metadata | gap_pct_reduction_vs_metadata | "
            "improving_seed_count | positive_threshold | ci_source | ci_low_top1/spearman/gap | "
            "catastrophic_regression | regression_check_missing | raw_instability_breach | instability_gate_applied | instability_breach |\n"
        )
        f.write("|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                "| {} | {} | {} | {} | {} | {} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | "
                "{:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {} | {} | {} | "
                "{:.4f}/{:.4f}/{:.4f} | {} | {} | {} | {} | {} |\n".format(
                    r["method"],
                    r["method_role"],
                    int(r["selection_eligible"]),
                    r.get("decision", "not_selected"),
                    r["tier"],
                    r["n_seeds"],
                    float(r["top1_oracle_hit_mean"]),
                    float(r["top1_oracle_hit_std"]),
                    float(r["spearman_mean"]),
                    float(r["spearman_std"]),
                    float(r["mean_oracle_gap_pct_mean"]),
                    float(r["mean_oracle_gap_pct_std"]),
                    float(r["top1_uplift_vs_metadata_mean"]),
                    float(r["top1_uplift_vs_metadata_std"]),
                    float(r["spearman_uplift_vs_metadata_mean"]),
                    float(r["spearman_uplift_vs_metadata_std"]),
                    float(r["oracle_gap_pct_reduction_vs_metadata_mean"]),
                    float(r["oracle_gap_pct_reduction_vs_metadata_std"]),
                    int(r["improving_seed_count"]),
                    int(r["positive_observation_threshold"]),
                    str(r["ci_source"]),
                    float(r["top1_ci_low"]),
                    float(r["spearman_ci_low"]),
                    float(r["gap_pct_ci_low"]),
                    int(r["catastrophic_regression_breach"]),
                    int(r["regression_check_missing"]),
                    int(r["raw_instability_breach"]),
                    int(r["instability_gate_applied"]),
                    int(r["instability_breach"]),
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compatibility decision table from learned utility run manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/comparison_tables/compatibility_run_manifest.txt"),
    )
    parser.add_argument("--uplift-reference-method", type=str, default="metadata_routing")
    parser.add_argument("--min-improving-seeds", type=int, default=2)
    parser.add_argument("--strong-spearman-uplift-min", type=float, default=0.05)
    parser.add_argument("--strong-top1-uplift-min", type=float, default=0.10)
    parser.add_argument("--strong-gap-pct-reduction-min", type=float, default=5.0)
    parser.add_argument("--weak-spearman-uplift-min", type=float, default=0.025)
    parser.add_argument("--weak-top1-uplift-min", type=float, default=0.05)
    parser.add_argument("--weak-gap-pct-reduction-min", type=float, default=2.5)
    parser.add_argument("--instability-std-threshold", type=float, default=0.05)
    parser.add_argument("--instability-sign-inconsistency-min-count", type=int, default=2)
    parser.add_argument(
        "--decision-policy-version",
        type=str,
        default=SIGN_CI_POLICY,
        choices=[LEGACY_STD_POLICY, SIGN_CI_POLICY],
    )
    parser.add_argument("--top1-uplift-std-threshold", type=float, default=0.05)
    parser.add_argument("--spearman-uplift-std-threshold", type=float, default=0.05)
    parser.add_argument("--gap-pct-reduction-std-threshold", type=float, default=3.0)
    parser.add_argument("--min-positive-fraction", type=float, default=0.67)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--ci-bootstrap-reps", type=int, default=10000)
    parser.add_argument("--ci-bootstrap-seed", type=int, default=1337)
    parser.add_argument("--allow-missing-domain-breakdown-as-diagnostic", action="store_true")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    decision_policy_version = validate_decision_policy_version(args.decision_policy_version)
    if args.output_csv is None:
        csv_name = (
            "compatibility_decision_table.csv"
            if decision_policy_version == LEGACY_STD_POLICY
            else "compatibility_decision_table_sign_ci_v2.csv"
        )
        args.output_csv = Path("results/comparison_tables") / csv_name
    if args.output_md is None:
        md_name = (
            "compatibility_decision_table.md"
            if decision_policy_version == LEGACY_STD_POLICY
            else "compatibility_decision_table_sign_ci_v2.md"
        )
        args.output_md = Path("results/summaries") / md_name

    result_paths = _load_manifest(args.manifest)
    rows = _read_rows(result_paths, uplift_reference_method=str(args.uplift_reference_method))
    if not rows:
        raise RuntimeError("No rows could be read from result json files.")

    strong = {
        "spearman_uplift_min": float(args.strong_spearman_uplift_min),
        "top1_uplift_min": float(args.strong_top1_uplift_min),
        "oracle_gap_pct_reduction_min": float(args.strong_gap_pct_reduction_min),
    }
    weak = {
        "spearman_uplift_min": float(args.weak_spearman_uplift_min),
        "top1_uplift_min": float(args.weak_top1_uplift_min),
        "oracle_gap_pct_reduction_min": float(args.weak_gap_pct_reduction_min),
    }

    out_rows, summary = _aggregate(
        rows=rows,
        uplift_reference_method=str(args.uplift_reference_method),
        min_improving_seeds=int(args.min_improving_seeds),
        strong=strong,
        weak=weak,
        instability_std_threshold=float(args.instability_std_threshold),
        instability_sign_inconsistency_min_count=int(args.instability_sign_inconsistency_min_count),
        decision_policy_version=str(decision_policy_version),
        top1_uplift_std_threshold=float(args.top1_uplift_std_threshold),
        spearman_uplift_std_threshold=float(args.spearman_uplift_std_threshold),
        gap_pct_reduction_std_threshold=float(args.gap_pct_reduction_std_threshold),
        min_positive_fraction=float(args.min_positive_fraction),
        ci_level=float(args.ci_level),
        ci_bootstrap_reps=int(args.ci_bootstrap_reps),
        ci_bootstrap_seed=int(args.ci_bootstrap_seed),
        allow_missing_domain_breakdown_as_diagnostic=bool(
            args.allow_missing_domain_breakdown_as_diagnostic
        ),
    )

    _write_csv(args.output_csv, out_rows)
    _write_md(args.output_md, out_rows, summary)

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
