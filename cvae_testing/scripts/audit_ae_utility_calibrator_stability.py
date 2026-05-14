#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PRIMARY_METHOD = "ae_utility_calibrated_safe_override_v1"
PRIMARY_METHOD_V2 = "ae_utility_calibrated_consensus_safe_override_v2"
PRIMARY_METHODS = {PRIMARY_METHOD, PRIMARY_METHOD_V2}
AE_ARGMIN_METHOD = "ae_argmin_zscore"
GLOBAL_BASELINES = {
    "metadata_routing",
    "metadata_ae_residual_safe_override_v1",
    "pairwise_ranker_ae_combined",
    "ae_first_margin_gated_v1",
}

THRESHOLDS = {
    "top1_drop_vs_ae_argmin_abs_max": 0.02,
    "spearman_drop_vs_ae_argmin_abs_max": 0.03,
    "gap_pct_degradation_vs_ae_argmin_max": 1.0,
    "center_dominance_share_max": 0.50,
    "min_centers_improved_for_stable_pass": 4,
    "min_seed_domain_units_improved_for_stable_pass": 12,
    "min_selected_override_precision": 0.50,
    "min_active_override_rate_for_pass": 0.20,
    "min_active_override_rate_for_weak_pass": 0.10,
    "seed_dominance_share_max": 0.50,
}


@dataclass(frozen=True)
class RunArtifact:
    result_path: Path
    reports_dir: Path
    dataset: str
    seed: str
    run_id: str


def _read_manifest(path: Path) -> List[Path]:
    with path.open("r", encoding="utf-8") as f:
        rows = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    out: List[Path] = []
    for row in rows:
        p = Path(row)
        out.append(p if p.is_absolute() else path.parent.parent.parent / p)
    return out


def _read_csv(path: Path, *, required: bool = True) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        if required:
            raise FileNotFoundError(path)
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json(path: Path, *, required: bool = True) -> Dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def _mean(values: Iterable[float], default: float = float("nan")) -> float:
    vals = _finite(values)
    return float(mean(vals)) if vals else float(default)


def _sum(values: Iterable[float]) -> float:
    return float(sum(_finite(values)))


def _dataset_from_path(path: Path) -> str:
    text = str(path).lower()
    if "camelyon17" in text:
        return "camelyon17"
    if "breakhis" in text:
        return "breakhis"
    return "unknown"


def _seed_from_path(path: Path) -> str:
    match = re.search(r"seed(\d+)", str(path))
    return match.group(1) if match else "unknown"


def _run_id_from_path(path: Path) -> str:
    if path.parent.name == "reports":
        return path.parent.parent.name
    return path.parent.name


def _load_runs(manifest: Path, dataset: str) -> List[RunArtifact]:
    paths = _read_manifest(manifest)
    runs: List[RunArtifact] = []
    for path in paths:
        if _dataset_from_path(path) != str(dataset):
            continue
        reports_dir = path.parent if path.parent.name == "reports" else path.parent / "reports"
        runs.append(
            RunArtifact(
                result_path=path,
                reports_dir=reports_dir,
                dataset=str(dataset),
                seed=_seed_from_path(path),
                run_id=_run_id_from_path(path),
            )
        )
    if not runs:
        raise RuntimeError(f"No result paths for dataset={dataset} were found in {manifest}")
    return runs


def _by_method_domain(rows: Sequence[Mapping[str, str]]) -> Dict[tuple[str, str], Mapping[str, str]]:
    return {(str(row.get("method", "")), str(row.get("query_domain", ""))): row for row in rows}


def _policy_by_domain(rows: Sequence[Mapping[str, str]], method: str = PRIMARY_METHOD) -> Dict[str, Mapping[str, str]]:
    return {
        str(row.get("fold_query_domain", row.get("query_domain", ""))): row
        for row in rows
        if str(row.get("method", "")) == method
    }


def _primary_method_for_rows(rows: Sequence[Mapping[str, str]]) -> str:
    methods = {str(row.get("method", "")) for row in rows}
    return PRIMARY_METHOD_V2 if PRIMARY_METHOD_V2 in methods else PRIMARY_METHOD


def _best_non_oracle_baseline(
    by_key: Mapping[tuple[str, str], Mapping[str, str]],
    domain: str,
) -> Mapping[str, str] | None:
    candidates = [
        by_key[(method, domain)]
        for method in GLOBAL_BASELINES
        if (method, domain) in by_key
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: (_float(row.get("mean_oracle_gap_pct")), str(row.get("method"))))[0]


def build_seed_domain_metrics(runs: Sequence[RunArtifact]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for run in runs:
        domain_rows = _read_csv(run.reports_dir / "learned_utility_domain_breakdown.csv")
        policy_rows = _read_csv(run.reports_dir / "ae_utility_calibrator_policy_audit.csv")
        precision_rows = _read_csv(run.reports_dir / "ae_utility_calibrator_override_precision.csv", required=False)
        primary_method = _primary_method_for_rows(domain_rows)
        by_key = _by_method_domain(domain_rows)
        policies = _policy_by_domain(policy_rows, method=primary_method)
        precision = _policy_by_domain(precision_rows, method=primary_method)
        domains = sorted({domain for method, domain in by_key if method == primary_method}, key=lambda x: int(float(x)))
        for domain in domains:
            primary = by_key[(primary_method, domain)]
            ae = by_key.get((AE_ARGMIN_METHOD, domain))
            best = _best_non_oracle_baseline(by_key, domain)
            policy = policies.get(domain, {})
            precision_row = precision.get(domain, {})
            if ae is None or best is None:
                continue
            row = {
                "dataset": run.dataset,
                "seed": run.seed,
                "run_id": run.run_id,
                "heldout_center": int(float(domain)),
                "result_path": str(run.result_path),
                "primary_top1": _float(primary.get("top1_oracle_hit")),
                "primary_spearman": _float(primary.get("spearman")),
                "primary_mean_oracle_gap_pct": _float(primary.get("mean_oracle_gap_pct")),
                "ae_argmin_top1": _float(ae.get("top1_oracle_hit")),
                "ae_argmin_spearman": _float(ae.get("spearman")),
                "ae_argmin_mean_oracle_gap_pct": _float(ae.get("mean_oracle_gap_pct")),
                "best_non_oracle_baseline": str(best.get("method", "")),
                "best_non_oracle_top1": _float(best.get("top1_oracle_hit")),
                "best_non_oracle_spearman": _float(best.get("spearman")),
                "best_non_oracle_mean_oracle_gap_pct": _float(best.get("mean_oracle_gap_pct")),
                "active_override_rate": _float(policy.get("active_override_rate")),
                "selected_override_precision": _float(
                    policy.get("selected_override_precision", precision_row.get("selected_override_precision"))
                ),
                "harmful_vs_ae_argmin_rate": _float(policy.get("harmful_vs_ae_argmin_rate")),
                "improving_vs_ae_argmin_rate": _float(policy.get("improving_vs_ae_argmin_rate")),
                "override_capture_rate": _float(
                    policy.get("override_capture_rate", precision_row.get("override_capture_rate"))
                ),
                "oracle_improvable_query_rate": _float(policy.get("oracle_improvable_query_rate")),
                "oracle_headroom_vs_ae_argmin": _float(policy.get("oracle_headroom_vs_ae_argmin")),
                "selected_delta_threshold": str(policy.get("selected_delta_threshold", "")),
                "selected_margin_threshold": str(policy.get("selected_margin_threshold", "")),
                "selected_feature_set": str(policy.get("feature_set", "")),
            }
            row["top1_delta_vs_ae_argmin"] = row["primary_top1"] - row["ae_argmin_top1"]
            row["spearman_delta_vs_ae_argmin"] = row["primary_spearman"] - row["ae_argmin_spearman"]
            row["gap_pct_reduction_vs_ae_argmin"] = (
                row["ae_argmin_mean_oracle_gap_pct"] - row["primary_mean_oracle_gap_pct"]
            )
            row["top1_delta_vs_best_non_oracle_baseline"] = row["primary_top1"] - row["best_non_oracle_top1"]
            row["spearman_delta_vs_best_non_oracle_baseline"] = (
                row["primary_spearman"] - row["best_non_oracle_spearman"]
            )
            row["gap_pct_reduction_vs_best_non_oracle_baseline"] = (
                row["best_non_oracle_mean_oracle_gap_pct"] - row["primary_mean_oracle_gap_pct"]
            )
            row["material_degradation_vs_ae_argmin"] = int(
                -float(row["top1_delta_vs_ae_argmin"]) > THRESHOLDS["top1_drop_vs_ae_argmin_abs_max"]
                or -float(row["spearman_delta_vs_ae_argmin"]) > THRESHOLDS["spearman_drop_vs_ae_argmin_abs_max"]
                or -float(row["gap_pct_reduction_vs_ae_argmin"]) > THRESHOLDS["gap_pct_degradation_vs_ae_argmin_max"]
            )
            out.append(row)
    return out


def _group(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, List[Mapping[str, Any]]]:
    out: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row.get(key, "")), []).append(row)
    return out


def _stability_rows(rows: Sequence[Mapping[str, Any]], key: str, label: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    by_key = _group(rows, key)
    for value, vals in sorted(by_key.items(), key=lambda item: int(float(item[0])) if item[0] else -1):
        positive_gain = _sum(max(_float(row.get("gap_pct_reduction_vs_ae_argmin"), 0.0), 0.0) for row in vals)
        out.append(
            {
                label: int(float(value)) if str(value).replace(".", "", 1).isdigit() else value,
                "n_units": len(vals),
                "mean_top1_delta_vs_ae_argmin": _mean(_float(r.get("top1_delta_vs_ae_argmin")) for r in vals),
                "mean_spearman_delta_vs_ae_argmin": _mean(_float(r.get("spearman_delta_vs_ae_argmin")) for r in vals),
                "mean_gap_pct_reduction_vs_ae_argmin": _mean(
                    _float(r.get("gap_pct_reduction_vs_ae_argmin")) for r in vals
                ),
                "positive_gap_units": int(
                    sum(1 for r in vals if _float(r.get("gap_pct_reduction_vs_ae_argmin")) > 0.0)
                ),
                "material_degradation_units": int(sum(_int(r.get("material_degradation_vs_ae_argmin")) for r in vals)),
                "mean_active_override_rate": _mean(_float(r.get("active_override_rate")) for r in vals),
                "mean_selected_override_precision": _mean(_float(r.get("selected_override_precision")) for r in vals),
                "mean_override_capture_rate": _mean(_float(r.get("override_capture_rate")) for r in vals),
                "positive_gap_reduction_sum": positive_gain,
            }
        )
    total_positive = _sum(row["positive_gap_reduction_sum"] for row in out)
    for row in out:
        row["positive_gap_reduction_share"] = (
            float(row["positive_gap_reduction_sum"]) / total_positive if total_positive > 0.0 else float("nan")
        )
    return out


def threshold_selection_rows(runs: Sequence[RunArtifact]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for run in runs:
        rows = _read_csv(run.reports_dir / "ae_utility_calibrator_source_inner_validation.csv")
        for row in rows:
            if str(row.get("method", "")) not in PRIMARY_METHODS:
                continue
            if _int(row.get("selected_by_source_inner_validation")) != 1:
                continue
            out.append(
                {
                    "dataset": run.dataset,
                    "seed": run.seed,
                    "run_id": run.run_id,
                    "heldout_center": _int(row.get("fold_query_domain")),
                    "source_inner_pseudo_query_domain": _int(row.get("source_inner_pseudo_query_domain")),
                    "feature_set": row.get("feature_set", ""),
                    "delta_threshold": row.get("delta_threshold", ""),
                    "margin_threshold": row.get("margin_threshold", ""),
                    "selected_feature_set": row.get("selected_feature_set", ""),
                    "selected_delta_threshold": row.get("selected_delta_threshold", ""),
                    "selected_margin_threshold": row.get("selected_margin_threshold", ""),
                    "macro_top1_oracle_hit": _float(row.get("macro_top1_oracle_hit")),
                    "macro_mean_oracle_gap_pct": _float(row.get("macro_mean_oracle_gap_pct")),
                    "macro_active_override_rate": _float(row.get("macro_active_override_rate")),
                    "macro_selected_override_precision": _float(row.get("macro_selected_override_precision")),
                    "heldout_target_nelbo_used_for_selection": _int(row.get("heldout_target_nelbo_used_for_selection")),
                }
            )
    return out


def headroom_capture_rows(runs: Sequence[RunArtifact]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for run in runs:
        raw_policy_rows = _read_csv(run.reports_dir / "ae_utility_calibrator_policy_audit.csv")
        policy_rows = _policy_by_domain(raw_policy_rows, method=_primary_method_for_rows(raw_policy_rows))
        for domain, row in sorted(policy_rows.items(), key=lambda item: int(float(item[0]))):
            out.append(
                {
                    "dataset": run.dataset,
                    "seed": run.seed,
                    "run_id": run.run_id,
                    "heldout_center": _int(domain),
                    "oracle_headroom_vs_ae_argmin": _float(row.get("oracle_headroom_vs_ae_argmin")),
                    "oracle_improvable_query_rate": _float(row.get("oracle_improvable_query_rate")),
                    "override_capture_rate": _float(row.get("override_capture_rate")),
                    "active_override_rate": _float(row.get("active_override_rate")),
                    "net_gain_vs_ae_argmin": _float(row.get("net_gain_vs_ae_argmin")),
                }
            )
    return out


def override_precision_rows(runs: Sequence[RunArtifact]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for run in runs:
        rows = _read_csv(run.reports_dir / "ae_utility_calibrator_override_precision.csv")
        for row in rows:
            if str(row.get("method", "")) not in PRIMARY_METHODS:
                continue
            out.append(
                {
                    "dataset": run.dataset,
                    "seed": run.seed,
                    "run_id": run.run_id,
                    "heldout_center": _int(row.get("fold_query_domain")),
                    "active_overrides": _int(row.get("active_overrides")),
                    "selected_override_precision": _float(row.get("selected_override_precision")),
                    "active_override_rate": _float(row.get("active_override_rate")),
                    "override_capture_rate": _float(row.get("override_capture_rate")),
                }
            )
    return out


def _empty_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 0


def _flag(condition: bool) -> int:
    return 1 if bool(condition) else 0


def leakage_provenance_rows(runs: Sequence[RunArtifact]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for run in runs:
        leakage = _json(run.reports_dir / "leakage_report.json")
        provenance = _json(run.reports_dir / "support_free_ae_provenance.json")
        policy_rows = _read_csv(run.reports_dir / "ae_utility_calibrator_policy_audit.csv")
        overlap_path = run.reports_dir / "support_free_ae_overlap_audit.csv"
        overlap_rows = _read_csv(overlap_path, required=False)

        patient_overlap = leakage.get("patient_overlap", {}) if isinstance(leakage, dict) else {}
        checks = {
            "patient_train_test_overlap_zero": _flag(_empty_list(patient_overlap.get("train_test"))),
            "patient_val_test_overlap_zero": _flag(_empty_list(patient_overlap.get("val_test"))),
            "duplicate_paths_zero": _flag(_empty_list(leakage.get("duplicate_paths"))),
            "target_support_used_zero": _flag(_int(provenance.get("target_support_used"), -1) == 0),
            "target_labels_used_zero": _flag(_int(provenance.get("target_labels_used"), -1) == 0),
            "target_domain_normalization_statistics_used_zero": _flag(
                _int(provenance.get("target_domain_normalization_statistics_used"), -1) == 0
            ),
            "target_ae_excluded": _flag(_int(provenance.get("target_ae_excluded"), -1) == 1),
            "source_inner_self_ae_excluded": _flag(_int(provenance.get("source_inner_self_ae_excluded"), -1) == 1),
            "policy_heldout_target_nelbo_not_used": _flag(
                bool(policy_rows)
                and all(_int(row.get("heldout_target_nelbo_used_for_selection"), -1) == 0 for row in policy_rows)
            ),
            "policy_target_ae_excluded": _flag(
                bool(policy_rows) and all(_int(row.get("excluded_target_ae"), -1) == 1 for row in policy_rows)
            ),
            "policy_target_cvae_excluded": _flag(
                bool(policy_rows) and all(_int(row.get("excluded_target_cvae"), -1) == 1 for row in policy_rows)
            ),
            "policy_pseudo_query_ae_excluded": _flag(
                bool(policy_rows) and all(_int(row.get("excluded_pseudo_query_ae"), -1) == 1 for row in policy_rows)
            ),
            "policy_pseudo_query_cvae_excluded": _flag(
                bool(policy_rows) and all(_int(row.get("excluded_pseudo_query_cvae"), -1) == 1 for row in policy_rows)
            ),
        }
        if overlap_rows:
            checks["overlap_audit_present"] = 1
            checks["ae_train_query_overlap_zero"] = _flag(
                all(_int(row.get("ae_train_query_overlap_count"), -1) == 0 for row in overlap_rows)
            )
            checks["ae_val_query_overlap_zero"] = _flag(
                all(_int(row.get("ae_val_query_overlap_count"), -1) == 0 for row in overlap_rows)
            )
        else:
            checks["overlap_audit_present"] = 0
            checks["ae_train_query_overlap_zero"] = 0
            checks["ae_val_query_overlap_zero"] = 0

        failed = [key for key, value in checks.items() if int(value) != 1]
        if failed and failed == ["overlap_audit_present", "ae_train_query_overlap_zero", "ae_val_query_overlap_zero"]:
            status = "NEEDS EVIDENCE"
        elif failed:
            status = "FAIL"
        else:
            status = "PASS"
        out.append(
            {
                "dataset": run.dataset,
                "seed": run.seed,
                "run_id": run.run_id,
                "provenance_status": status,
                "failed_checks": "|".join(failed),
                **checks,
            }
        )
    return out


def _aggregate_verdict(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "REJECTED"
    gap = _mean(_float(row.get("gap_pct_reduction_vs_ae_argmin")) for row in rows)
    top1 = _mean(_float(row.get("top1_delta_vs_ae_argmin")) for row in rows)
    spearman = _mean(_float(row.get("spearman_delta_vs_ae_argmin")) for row in rows)
    active = _mean(_float(row.get("active_override_rate")) for row in rows)
    precision = _mean(_float(row.get("selected_override_precision")) for row in rows)
    harmful = _mean(_float(row.get("harmful_vs_ae_argmin_rate")) for row in rows)
    improving = _mean(_float(row.get("improving_vs_ae_argmin_rate")) for row in rows)
    no_material = (
        -top1 <= THRESHOLDS["top1_drop_vs_ae_argmin_abs_max"]
        and -spearman <= THRESHOLDS["spearman_drop_vs_ae_argmin_abs_max"]
        and -gap <= THRESHOLDS["gap_pct_degradation_vs_ae_argmin_max"]
    )
    if (
        gap > 0.0
        and top1 >= 0.0
        and spearman >= 0.0
        and active >= THRESHOLDS["min_active_override_rate_for_pass"]
        and precision > THRESHOLDS["min_selected_override_precision"]
        and harmful <= improving
        and no_material
    ):
        return "PASS"
    if (
        gap > 0.0
        and no_material
        and active >= THRESHOLDS["min_active_override_rate_for_weak_pass"]
        and precision >= THRESHOLDS["min_selected_override_precision"]
        and harmful <= improving
    ):
        return "WEAK PASS"
    if gap > 0.0 or active > 0.0 or spearman > 0.0:
        return "DIAGNOSTIC ONLY"
    return "FAIL"


def leave_one_center_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    centers = sorted({int(row["heldout_center"]) for row in rows})
    out: List[Dict[str, Any]] = []
    for center in centers:
        kept = [row for row in rows if int(row["heldout_center"]) != int(center)]
        out.append(
            {
                "removed_heldout_center": int(center),
                "remaining_units": len(kept),
                "verdict_without_center": _aggregate_verdict(kept),
                "mean_gap_pct_reduction_vs_ae_argmin_without_center": _mean(
                    _float(row.get("gap_pct_reduction_vs_ae_argmin")) for row in kept
                ),
                "mean_top1_delta_vs_ae_argmin_without_center": _mean(
                    _float(row.get("top1_delta_vs_ae_argmin")) for row in kept
                ),
                "mean_spearman_delta_vs_ae_argmin_without_center": _mean(
                    _float(row.get("spearman_delta_vs_ae_argmin")) for row in kept
                ),
            }
        )
    return out


def summarize_stability(
    *,
    seed_domain_rows: Sequence[Mapping[str, Any]],
    per_domain_rows: Sequence[Mapping[str, Any]],
    per_seed_rows: Sequence[Mapping[str, Any]],
    leave_one_rows: Sequence[Mapping[str, Any]],
    provenance_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    aggregate_verdict = _aggregate_verdict(seed_domain_rows)
    provenance_statuses = {str(row.get("provenance_status", "")) for row in provenance_rows}
    if "FAIL" in provenance_statuses:
        final = "REJECTED"
    elif "NEEDS EVIDENCE" in provenance_statuses:
        final = "DIAGNOSTIC ONLY"
    else:
        positive_center_count = sum(
            1 for row in per_domain_rows if _float(row.get("mean_gap_pct_reduction_vs_ae_argmin")) > 0.0
        )
        positive_seed_count = sum(
            1 for row in per_seed_rows if _float(row.get("mean_gap_pct_reduction_vs_ae_argmin")) > 0.0
        )
        positive_units = sum(
            1 for row in seed_domain_rows if _float(row.get("gap_pct_reduction_vs_ae_argmin")) > 0.0
        )
        center_dominance = max(
            _finite(_float(row.get("positive_gap_reduction_share")) for row in per_domain_rows) or [0.0]
        )
        seed_dominance = max(
            _finite(_float(row.get("positive_gap_reduction_share")) for row in per_seed_rows) or [0.0]
        )
        one_center_dominated = bool(center_dominance > THRESHOLDS["center_dominance_share_max"])
        leave_one_fragile = any(str(row.get("verdict_without_center")) not in {"PASS", "WEAK PASS"} for row in leave_one_rows)
        seed_sensitive = bool(
            positive_seed_count < len(per_seed_rows)
            or seed_dominance > THRESHOLDS["seed_dominance_share_max"]
            or any(_int(row.get("material_degradation_units")) > 0 for row in per_seed_rows)
        )
        center_material_degradation = any(_int(row.get("material_degradation_units")) > 0 for row in per_domain_rows)
        precision_ok = all(
            (
                _float(row.get("selected_override_precision")) >= THRESHOLDS["min_selected_override_precision"]
                if _float(row.get("active_override_rate"), 0.0) > 0.0
                else True
            )
            for row in seed_domain_rows
        )
        if (
            aggregate_verdict == "PASS"
            and positive_seed_count == len(per_seed_rows)
            and positive_center_count >= THRESHOLDS["min_centers_improved_for_stable_pass"]
            and positive_units >= THRESHOLDS["min_seed_domain_units_improved_for_stable_pass"]
            and not center_material_degradation
            and not one_center_dominated
            and not leave_one_fragile
            and not seed_sensitive
            and precision_ok
        ):
            final = "STABLE PASS"
        elif aggregate_verdict == "PASS" and (one_center_dominated or leave_one_fragile):
            final = "CENTER-SENSITIVE PASS"
        elif aggregate_verdict == "PASS" and seed_sensitive:
            final = "SEED-SENSITIVE PASS"
        elif aggregate_verdict in {"PASS", "WEAK PASS", "DIAGNOSTIC ONLY"}:
            final = "DIAGNOSTIC ONLY"
        else:
            final = "FAIL"

    return {
        "protocol_status": "PASS" if provenance_statuses == {"PASS"} else ("FAIL" if "FAIL" in provenance_statuses else "NEEDS EVIDENCE"),
        "aggregate_verdict": aggregate_verdict,
        "stability_interpretation": final,
        "thresholds": THRESHOLDS,
        "n_seed_domain_units": len(seed_domain_rows),
        "n_seeds": len({row.get("seed") for row in seed_domain_rows}),
        "n_heldout_centers": len({row.get("heldout_center") for row in seed_domain_rows}),
        "positive_seed_domain_units": sum(
            1 for row in seed_domain_rows if _float(row.get("gap_pct_reduction_vs_ae_argmin")) > 0.0
        ),
        "positive_centers": sum(
            1 for row in per_domain_rows if _float(row.get("mean_gap_pct_reduction_vs_ae_argmin")) > 0.0
        ),
        "positive_seeds": sum(
            1 for row in per_seed_rows if _float(row.get("mean_gap_pct_reduction_vs_ae_argmin")) > 0.0
        ),
        "max_center_positive_gap_reduction_share": max(
            _finite(_float(row.get("positive_gap_reduction_share")) for row in per_domain_rows) or [float("nan")]
        ),
        "max_seed_positive_gap_reduction_share": max(
            _finite(_float(row.get("positive_gap_reduction_share")) for row in per_seed_rows) or [float("nan")]
        ),
        "leave_one_center_fragile": int(
            any(str(row.get("verdict_without_center")) not in {"PASS", "WEAK PASS"} for row in leave_one_rows)
        ),
        "provenance_statuses": sorted(provenance_statuses),
    }


def _write_markdown(path: Path, summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Camelyon17 AE Utility Calibrator Stability Audit",
        "",
        f"- Protocol status: `{summary.get('protocol_status')}`",
        f"- Aggregate verdict: `{summary.get('aggregate_verdict')}`",
        f"- Stability interpretation: `{summary.get('stability_interpretation')}`",
        f"- Seed-domain units: `{summary.get('n_seed_domain_units')}`",
        f"- Positive seed-domain units: `{summary.get('positive_seed_domain_units')}`",
        f"- Max center gain share: `{_float(summary.get('max_center_positive_gap_reduction_share')):.4f}`",
        f"- Leave-one-center fragile: `{summary.get('leave_one_center_fragile')}`",
        "",
        "| seed | center | gap reduction vs AE | top1 delta vs AE | spearman delta vs AE | active override | precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {heldout_center} | {gap:.4f} | {top1:.4f} | {spearman:.4f} | {active:.4f} | {precision:.4f} |".format(
                seed=row.get("seed"),
                heldout_center=int(row.get("heldout_center")),
                gap=_float(row.get("gap_pct_reduction_vs_ae_argmin")),
                top1=_float(row.get("top1_delta_vs_ae_argmin")),
                spearman=_float(row.get("spearman_delta_vs_ae_argmin")),
                active=_float(row.get("active_override_rate")),
                precision=_float(row.get("selected_override_precision")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(output_dir: Path, dataset: str, rows: Sequence[Mapping[str, Any]], per_domain: Sequence[Mapping[str, Any]], threshold_rows: Sequence[Mapping[str, Any]]) -> List[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: List[str] = []

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(output_dir / name)
        plt.close()
        artifacts.append(name)

    labels = [f"{row.get('seed')}-c{row.get('heldout_center')}" for row in rows]
    x = list(range(len(rows)))
    if rows:
        plt.figure(figsize=(max(8, len(rows) * 0.45), 4))
        plt.bar(x, [_float(row.get("gap_pct_reduction_vs_ae_argmin")) for row in rows])
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xticks(x, labels, rotation=70, ha="right")
        plt.ylabel("Gap pct reduction vs AE argmin")
        save(f"ae_utility_calibrator_{dataset}_gap_reduction_by_seed_domain.png")

        plt.figure(figsize=(max(8, len(rows) * 0.45), 4))
        plt.bar(x, [_float(row.get("top1_delta_vs_ae_argmin")) for row in rows])
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xticks(x, labels, rotation=70, ha="right")
        plt.ylabel("Top1 delta vs AE argmin")
        save(f"ae_utility_calibrator_{dataset}_top1_delta_by_seed_domain.png")

    if per_domain:
        centers = [str(row.get("heldout_center")) for row in per_domain]
        x2 = list(range(len(per_domain)))
        plt.figure(figsize=(7, 4))
        plt.bar(x2, [_float(row.get("mean_override_capture_rate")) for row in per_domain])
        plt.xticks(x2, centers)
        plt.ylabel("Mean override capture rate")
        save(f"ae_utility_calibrator_{dataset}_headroom_capture_by_domain.png")

        plt.figure(figsize=(7, 4))
        plt.bar(x2, [_float(row.get("mean_selected_override_precision")) for row in per_domain])
        plt.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
        plt.xticks(x2, centers)
        plt.ylabel("Mean selected override precision")
        save(f"ae_utility_calibrator_{dataset}_override_precision_by_domain.png")

    if threshold_rows:
        selected = {(str(r.get("seed")), int(r.get("heldout_center"))): str(r.get("selected_delta_threshold")) for r in threshold_rows}
        seeds = sorted({seed for seed, _center in selected}, key=lambda v: int(v) if str(v).isdigit() else str(v))
        centers = sorted({center for _seed, center in selected})
        values = [[_float(selected.get((seed, center), "nan")) for center in centers] for seed in seeds]
        plt.figure(figsize=(7, 4))
        plt.imshow(values, aspect="auto")
        plt.xticks(range(len(centers)), centers)
        plt.yticks(range(len(seeds)), seeds)
        plt.xlabel("Heldout center")
        plt.ylabel("Seed")
        plt.colorbar(label="Selected delta threshold")
        save(f"ae_utility_calibrator_{dataset}_threshold_selection_heatmap.png")
    return artifacts


def build_outputs(
    *,
    manifest: Path,
    dataset: str,
    output_dir: Path,
    summary_md: Path,
) -> Dict[str, Any]:
    runs = _load_runs(manifest, dataset)
    seed_domain = build_seed_domain_metrics(runs)
    per_domain = _stability_rows(seed_domain, "heldout_center", "heldout_center")
    per_seed = _stability_rows(seed_domain, "seed", "seed")
    threshold_rows = threshold_selection_rows(runs)
    headroom_rows = headroom_capture_rows(runs)
    precision_rows = override_precision_rows(runs)
    provenance_rows = leakage_provenance_rows(runs)
    loo_rows = leave_one_center_rows(seed_domain)
    summary = summarize_stability(
        seed_domain_rows=seed_domain,
        per_domain_rows=per_domain,
        per_seed_rows=per_seed,
        leave_one_rows=loo_rows,
        provenance_rows=provenance_rows,
    )
    plots = _write_plots(output_dir, dataset, seed_domain, per_domain, threshold_rows)
    summary["plot_artifacts"] = plots

    prefix = f"ae_utility_calibrator_{dataset}"
    outputs = {
        "seed_domain_metrics": output_dir / f"{prefix}_seed_domain_metrics.csv",
        "per_domain_stability": output_dir / f"{prefix}_per_domain_stability.csv",
        "per_seed_stability": output_dir / f"{prefix}_per_seed_stability.csv",
        "threshold_selection_by_fold": output_dir / f"{prefix}_threshold_selection_by_fold.csv",
        "headroom_capture": output_dir / f"{prefix}_headroom_capture.csv",
        "override_precision_by_domain": output_dir / f"{prefix}_override_precision_by_domain.csv",
        "leakage_provenance_audit": output_dir / f"{prefix}_leakage_provenance_audit.csv",
        "leave_one_center_sensitivity": output_dir / f"{prefix}_leave_one_center_sensitivity.csv",
        "summary_json": output_dir / f"{prefix}_stability_summary.json",
        "summary_md": summary_md,
    }
    _write_csv(outputs["seed_domain_metrics"], seed_domain)
    _write_csv(outputs["per_domain_stability"], per_domain)
    _write_csv(outputs["per_seed_stability"], per_seed)
    _write_csv(outputs["threshold_selection_by_fold"], threshold_rows)
    _write_csv(outputs["headroom_capture"], headroom_rows)
    _write_csv(outputs["override_precision_by_domain"], precision_rows)
    _write_csv(outputs["leakage_provenance_audit"], provenance_rows)
    _write_csv(outputs["leave_one_center_sensitivity"], loo_rows)
    outputs["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["summary_json"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_markdown(outputs["summary_md"], summary, seed_domain)
    return {
        "summary": summary,
        "outputs": {key: str(value) for key, value in outputs.items()},
        "n_runs": len(runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AE utility calibrator stability across seeds and heldout domains.")
    parser.add_argument("--manifest", type=Path, default=Path("results/comparison_tables/ae_utility_calibrator_run_manifest.txt"))
    parser.add_argument("--dataset", type=str, default="camelyon17")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/comparison_tables/ae_utility_calibrator_stability"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=None,
        help="Markdown summary path. Defaults to results/summaries/ae_utility_calibrator_<dataset>_stability_audit.md",
    )
    args = parser.parse_args()
    summary_md = args.summary_md or Path("results/summaries") / f"ae_utility_calibrator_{args.dataset}_stability_audit.md"
    result = build_outputs(
        manifest=args.manifest,
        dataset=str(args.dataset),
        output_dir=args.output_dir,
        summary_md=summary_md,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
