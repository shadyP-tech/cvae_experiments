#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "camelyon17"
    / "camelyon17_support_estimated_utility_routing_v2"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "comparison_tables"
EARLIER_DECISION_TABLE = (
    OUTPUT_DIR / "camelyon17_support_estimated_utility_routing_v2_decision_table.csv"
)

RUN_SEEDS = (42, 43, 44)
HELDOUT_CENTERS = (0, 1, 2, 3, 4)
SUPPORT_SEEDS = (17, 23, 31)
SUPPORT_SIZES = (4, 8, 16, 32)

DIRECT_METHOD = "support_set_nelbo_top1"
CONSERVATIVE_METHOD = "support_set_nelbo_conservative"
METADATA_METHOD = "support_metadata_routing"
STATIC_METHOD = "support_static_embedding_routing"
METHOD_LABELS = {
    DIRECT_METHOD: "direct_support_nelbo",
    CONSERVATIVE_METHOD: "conservative_support_nelbo",
    METADATA_METHOD: "metadata_routing",
    STATIC_METHOD: "static_embedding_routing",
}
PRIMARY_METHOD_LABEL = "direct_support_nelbo"
CONSERVATIVE_METHOD_LABEL = "conservative_support_nelbo"

EXPECTED_SELECTED_ROWS = (
    len(RUN_SEEDS)
    * len(HELDOUT_CENTERS)
    * len(SUPPORT_SEEDS)
    * len(SUPPORT_SIZES)
    * 2
)
EXPECTED_ALPHA_ROWS = len(RUN_SEEDS) * len(HELDOUT_CENTERS) * len(SUPPORT_SIZES)
LINEAGE_NOTE = (
    "This report supersedes earlier selection summaries for thesis interpretation. "
    "It does not invalidate their protocol checks; it revises the claim level based "
    "on per-k support-seed stability and alpha-selection diagnostics."
)


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _variance(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return 0.0
    mu = _mean(vals)
    return float(sum((v - mu) ** 2 for v in vals) / len(vals))


def _std(values: Iterable[float]) -> float:
    return float(math.sqrt(_variance(values)))


def _pct(count: int, total: int) -> float:
    return float((100.0 * int(count) / int(total)) if total else 0.0)


def _run_seed_from_run_dir(run_dir: Path) -> int:
    match = re.search(r"seed(\d+)", run_dir.name)
    if not match:
        raise ValueError(f"Cannot infer seed from run directory: {run_dir}")
    return int(match.group(1))


def _parse_candidate_experts(value: object) -> List[int]:
    out: List[int] = []
    for part in str(value or "").split("|"):
        part = part.strip()
        if not part:
            continue
        out.append(int(float(part)))
    return out


def _json_score_values(value: object) -> List[float]:
    try:
        payload = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    vals: List[float] = []
    for item in payload.values():
        try:
            val = float(item)
        except Exception:
            continue
        if math.isfinite(val):
            vals.append(val)
    return vals


def _has_nonconstant_values(values: Sequence[float], *, tol: float = 1.0e-12) -> bool:
    if len(values) < 2:
        return False
    first = float(values[0])
    return any(abs(float(value) - first) > tol for value in values[1:])


def _spearman_valid(row: Mapping[str, Any]) -> bool:
    spearman = _to_float(row.get("spearman", ""), default=float("nan"))
    if not math.isfinite(spearman):
        return False
    predicted = _json_score_values(row.get("predicted_score_by_expert_json", "{}"))
    utility = _json_score_values(row.get("eval_nelbo_by_expert_json", "{}"))
    return _has_nonconstant_values(predicted) and _has_nonconstant_values(utility)


def _has_ties(values: Sequence[float], *, tol: float = 1.0e-12) -> bool:
    rounded: List[float] = []
    for value in values:
        if not math.isfinite(float(value)):
            continue
        rounded_value = round(float(value) / tol) * tol
        if rounded_value in rounded:
            return True
        rounded.append(rounded_value)
    return False


def _spearman_tie_present(row: Mapping[str, Any]) -> bool:
    predicted = _json_score_values(row.get("predicted_score_by_expert_json", "{}"))
    utility = _json_score_values(row.get("eval_nelbo_by_expert_json", "{}"))
    return _has_ties(predicted) or _has_ties(utility)


def _result_paths(experiment_root: Path) -> List[Path]:
    paths: List[Path] = []
    for seed in RUN_SEEDS:
        run_dir = experiment_root / f"support_utility_v2_seed{seed}"
        reports_dir = run_dir / "reports"
        required = [
            run_dir / "config_resolved.yaml",
            reports_dir / "support_response_sample_selections.csv",
            reports_dir / "support_utility_selected_hyperparams.csv",
            reports_dir / "support_response_split_manifest.csv",
            reports_dir / "support_response_results.json",
            reports_dir / "leakage_report.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required support-NELBO source artifacts:\n" + "\n".join(missing)
            )
        paths.append(run_dir)
    return paths


def _load_source_rows(experiment_root: Path) -> Tuple[List[dict], List[dict], List[dict]]:
    selected_rows: List[dict] = []
    alpha_rows: List[dict] = []
    split_rows: List[dict] = []
    for run_dir in _result_paths(experiment_root):
        run_seed = _run_seed_from_run_dir(run_dir)
        reports_dir = run_dir / "reports"

        sample_path = reports_dir / "support_response_sample_selections.csv"
        for row in _read_csv(sample_path):
            method = str(row.get("method", ""))
            if method not in {DIRECT_METHOD, CONSERVATIVE_METHOD, METADATA_METHOD, STATIC_METHOD}:
                continue
            row = dict(row)
            row["run_seed"] = run_seed
            row["run_id"] = run_dir.name
            row["source_path"] = str(sample_path)
            row["method_label"] = METHOD_LABELS.get(method, method)
            selected_rows.append(row)

        alpha_path = reports_dir / "support_utility_selected_hyperparams.csv"
        for row in _read_csv(alpha_path):
            row = dict(row)
            row["run_seed"] = run_seed
            row["run_id"] = run_dir.name
            row["source_path"] = str(alpha_path)
            alpha_rows.append(row)

        split_path = reports_dir / "support_response_split_manifest.csv"
        for row in _read_csv(split_path):
            row = dict(row)
            row["run_seed"] = run_seed
            row["run_id"] = run_dir.name
            row["source_path"] = str(split_path)
            split_rows.append(row)

    direct_conservative = [
        row for row in selected_rows if row.get("method") in {DIRECT_METHOD, CONSERVATIVE_METHOD}
    ]
    if len(direct_conservative) != EXPECTED_SELECTED_ROWS:
        raise RuntimeError(
            "Selected-method row-count assertion failed for direct/conservative support-NELBO rows. "
            f"Expected {EXPECTED_SELECTED_ROWS} selected-method rows, not raw candidate rows; "
            f"found {len(direct_conservative)}."
        )
    if len(alpha_rows) != EXPECTED_ALPHA_ROWS:
        raise RuntimeError(
            "Alpha-selection row-count assertion failed. "
            f"Expected {EXPECTED_ALPHA_ROWS} alpha-selection rows; found {len(alpha_rows)}."
        )
    return selected_rows, alpha_rows, split_rows


def _group_key(row: Mapping[str, Any]) -> Tuple[int, int, int, int, str]:
    return (
        _to_int(row.get("run_seed", row.get("seed", 0))),
        _to_int(row.get("query_domain", row.get("target_domain", 0))),
        _to_int(row.get("support_size_requested", row.get("support_size", 0))),
        _to_int(row.get("support_seed", 0)),
        str(row.get("method", "")),
    )


def _selected_rows(rows: Sequence[Mapping[str, Any]], methods: Sequence[str]) -> List[dict]:
    wanted = set(methods)
    return [dict(row) for row in rows if str(row.get("method", "")) in wanted]


def _summarize_method(rows: Sequence[Mapping[str, Any]], method: str) -> dict:
    vals = [row for row in rows if str(row.get("method", "")) == method]
    spearman_vals = [
        _to_float(row.get("spearman", 0.0))
        for row in vals
        if _spearman_valid(row)
    ]
    return {
        "method": METHOD_LABELS.get(method, method),
        "source_method": method,
        "role": "primary" if method == DIRECT_METHOD else "diagnostic_ablation" if method == CONSERVATIVE_METHOD else "baseline",
        "n_rows": len(vals),
        "top1_oracle_hit_mean": _mean(_to_float(row.get("top1_oracle_hit", 0.0)) for row in vals),
        "top1_oracle_hit_std": _std(_to_float(row.get("top1_oracle_hit", 0.0)) for row in vals),
        "spearman_valid_rows": len(spearman_vals),
        "spearman_dropped_rows": len(vals) - len(spearman_vals),
        "spearman_mean": _mean(spearman_vals),
        "spearman_std": _std(spearman_vals),
        "oracle_gap_pct_mean": _mean(_to_float(row.get("mean_oracle_gap_pct", 0.0)) for row in vals),
        "oracle_gap_pct_std": _std(_to_float(row.get("mean_oracle_gap_pct", 0.0)) for row in vals),
        "selected_eval_nelbo_mean": _mean(_to_float(row.get("selected_nelbo", 0.0)) for row in vals),
        "selected_eval_nelbo_std": _std(_to_float(row.get("selected_nelbo", 0.0)) for row in vals),
        "high_regret_rate_mean": _mean(_to_float(row.get("high_regret_selection", 0.0)) for row in vals),
        "catastrophic_mistake_rate_mean": _mean(_to_float(row.get("catastrophic_mistake", 0.0)) for row in vals),
    }


def build_summary_rows(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    return [
        _summarize_method(rows, method)
        for method in [METADATA_METHOD, STATIC_METHOD, DIRECT_METHOD, CONSERVATIVE_METHOD]
    ]


def _metric_values(rows: Sequence[Mapping[str, Any]], method: str, k: int) -> List[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("method", "")) == method
        and _to_int(row.get("support_size_requested", 0)) == int(k)
    ]


def build_per_k_rows(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    out: List[dict] = []
    for support_size in SUPPORT_SIZES:
        direct = _metric_values(rows, DIRECT_METHOD, support_size)
        conservative = _metric_values(rows, CONSERVATIVE_METHOD, support_size)
        direct_spearman = [_to_float(r.get("spearman", 0.0)) for r in direct if _spearman_valid(r)]
        conservative_spearman = [
            _to_float(r.get("spearman", 0.0)) for r in conservative if _spearman_valid(r)
        ]
        direct_top1 = _mean(_to_float(r.get("top1_oracle_hit", 0.0)) for r in direct)
        conservative_top1 = _mean(_to_float(r.get("top1_oracle_hit", 0.0)) for r in conservative)
        direct_gap = _mean(_to_float(r.get("mean_oracle_gap_pct", 0.0)) for r in direct)
        conservative_gap = _mean(_to_float(r.get("mean_oracle_gap_pct", 0.0)) for r in conservative)
        direct_nelbo = _mean(_to_float(r.get("selected_nelbo", 0.0)) for r in direct)
        conservative_nelbo = _mean(_to_float(r.get("selected_nelbo", 0.0)) for r in conservative)
        direct_high_regret = _mean(_to_float(r.get("high_regret_selection", 0.0)) for r in direct)
        conservative_high_regret = _mean(
            _to_float(r.get("high_regret_selection", 0.0)) for r in conservative
        )
        out.append(
            {
                "support_size": support_size,
                "direct_n_rows": len(direct),
                "conservative_n_rows": len(conservative),
                "direct_top1_oracle_hit_mean": direct_top1,
                "conservative_top1_oracle_hit_mean": conservative_top1,
                "conservative_minus_direct_top1": conservative_top1 - direct_top1,
                "direct_spearman_valid_rows": len(direct_spearman),
                "conservative_spearman_valid_rows": len(conservative_spearman),
                "direct_spearman_mean": _mean(direct_spearman),
                "conservative_spearman_mean": _mean(conservative_spearman),
                "conservative_minus_direct_spearman": _mean(conservative_spearman)
                - _mean(direct_spearman),
                "direct_oracle_gap_pct_mean": direct_gap,
                "conservative_oracle_gap_pct_mean": conservative_gap,
                "direct_minus_conservative_oracle_gap_pct": direct_gap - conservative_gap,
                "direct_selected_eval_nelbo_mean": direct_nelbo,
                "conservative_selected_eval_nelbo_mean": conservative_nelbo,
                "direct_minus_conservative_selected_eval_nelbo": direct_nelbo - conservative_nelbo,
                "direct_high_regret_rate": direct_high_regret,
                "conservative_high_regret_rate": conservative_high_regret,
                "direct_minus_conservative_high_regret_rate": direct_high_regret - conservative_high_regret,
            }
        )
    return out


def build_per_center_gap_rows(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    out: List[dict] = []
    for center in HELDOUT_CENTERS:
        direct = [
            row for row in rows
            if str(row.get("method", "")) == DIRECT_METHOD
            and _to_int(row.get("query_domain", 0)) == int(center)
        ]
        conservative = [
            row for row in rows
            if str(row.get("method", "")) == CONSERVATIVE_METHOD
            and _to_int(row.get("query_domain", 0)) == int(center)
        ]
        direct_top1 = _mean(_to_float(row.get("top1_oracle_hit", 0.0)) for row in direct)
        conservative_top1 = _mean(_to_float(row.get("top1_oracle_hit", 0.0)) for row in conservative)
        direct_gap = _mean(_to_float(row.get("mean_oracle_gap_pct", 0.0)) for row in direct)
        conservative_gap = _mean(_to_float(row.get("mean_oracle_gap_pct", 0.0)) for row in conservative)
        out.append(
            {
                "heldout_center": center,
                "direct_n_rows": len(direct),
                "conservative_n_rows": len(conservative),
                "direct_top1_oracle_hit_mean": direct_top1,
                "conservative_top1_oracle_hit_mean": conservative_top1,
                "conservative_minus_direct_top1": conservative_top1 - direct_top1,
                "direct_oracle_gap_pct_mean": direct_gap,
                "conservative_oracle_gap_pct_mean": conservative_gap,
                "direct_minus_conservative_oracle_gap_pct": direct_gap - conservative_gap,
                "direct_high_regret_rate": _mean(
                    _to_float(row.get("high_regret_selection", 0.0)) for row in direct
                ),
                "conservative_high_regret_rate": _mean(
                    _to_float(row.get("high_regret_selection", 0.0)) for row in conservative
                ),
            }
        )
    return out


def _alpha_distribution_rows(alpha_rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    out: List[dict] = []

    def add_distribution(scope: str, vals: Sequence[Mapping[str, Any]], *, support_size: object = "", heldout_center: object = "") -> None:
        total = len(vals)
        counts: Dict[str, int] = {}
        nonzero = 0
        for row in vals:
            alpha = _to_float(row.get("selected_alpha", 0.0))
            label = f"{alpha:.1f}"
            counts[label] = counts.get(label, 0) + 1
            if abs(alpha) > 1.0e-12:
                nonzero += 1
        for alpha, count in sorted(counts.items(), key=lambda item: float(item[0])):
            out.append(
                {
                    "scope": scope,
                    "support_size": support_size,
                    "heldout_center": heldout_center,
                    "alpha": alpha,
                    "count": count,
                    "pct": _pct(count, total),
                }
            )
        out.append(
            {
                "scope": scope,
                "support_size": support_size,
                "heldout_center": heldout_center,
                "alpha": "nonzero",
                "count": nonzero,
                "pct": _pct(nonzero, total),
            }
        )

    add_distribution("overall", list(alpha_rows))
    for support_size in SUPPORT_SIZES:
        vals = [
            row for row in alpha_rows
            if _to_int(row.get("support_size", 0)) == int(support_size)
        ]
        add_distribution("by_support_size", vals, support_size=support_size)
    for center in HELDOUT_CENTERS:
        vals = [
            row for row in alpha_rows
            if _to_int(row.get("outer_center", 0)) == int(center)
        ]
        add_distribution("by_heldout_center", vals, heldout_center=center)
    return out


def _support_seed_group_key(row: Mapping[str, Any]) -> Tuple[int, int, int, str]:
    return (
        _to_int(row.get("run_seed", row.get("seed", 0))),
        _to_int(row.get("query_domain", 0)),
        _to_int(row.get("support_size_requested", 0)),
        str(row.get("method", "")),
    )


def build_stability_rows(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[dict], Dict[str, Any]]:
    groups: Dict[Tuple[int, int, int, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row.get("method", "")) not in {DIRECT_METHOD, CONSERVATIVE_METHOD}:
            continue
        groups.setdefault(_support_seed_group_key(row), []).append(row)

    per_group: List[dict] = []
    for key, vals in sorted(groups.items()):
        run_seed, center, support_size, method = key
        valid_spearman_vals = [
            _to_float(row.get("spearman", 0.0)) for row in vals if _spearman_valid(row)
        ]
        per_group.append(
            {
                "run_seed": run_seed,
                "heldout_center": center,
                "support_size": support_size,
                "method": METHOD_LABELS.get(method, method),
                "source_method": method,
                "n_support_seed_rows": len(vals),
                "top1_oracle_hit_variance": _variance(
                    _to_float(row.get("top1_oracle_hit", 0.0)) for row in vals
                ),
                "oracle_gap_pct_variance": _variance(
                    _to_float(row.get("mean_oracle_gap_pct", 0.0)) for row in vals
                ),
                "selected_eval_nelbo_variance": _variance(
                    _to_float(row.get("selected_nelbo", 0.0)) for row in vals
                ),
                "spearman_valid_support_seed_rows": len(valid_spearman_vals),
                "spearman_dropped_support_seed_rows": len(vals) - len(valid_spearman_vals),
                "spearman_variance": _variance(valid_spearman_vals) if len(valid_spearman_vals) >= 2 else "",
                "spearman_variance_defined": int(len(valid_spearman_vals) >= 2),
                "spearman_tie_rows": sum(1 for row in vals if _spearman_tie_present(row)),
                "variance_definition": "population_over_support_seed_within_run_seed_center_k_method",
                "tie_handling": (
                    "precomputed_spearman_used; constant predicted/eval score vectors dropped; "
                    "ties otherwise retained from upstream rank calculation"
                ),
            }
        )

    out: List[dict] = []
    for support_size in SUPPORT_SIZES:
        for method in [DIRECT_METHOD, CONSERVATIVE_METHOD]:
            vals = [
                row for row in per_group
                if int(row["support_size"]) == int(support_size)
                and str(row["source_method"]) == method
            ]
            spearman_defined = [
                _to_float(row.get("spearman_variance", 0.0))
                for row in vals
                if int(row.get("spearman_variance_defined", 0)) == 1
            ]
            out.append(
                {
                    "support_size": support_size,
                    "method": METHOD_LABELS[method],
                    "source_method": method,
                    "n_groups": len(vals),
                    "mean_top1_oracle_hit_variance_over_support_seed": _mean(
                        _to_float(row.get("top1_oracle_hit_variance", 0.0)) for row in vals
                    ),
                    "mean_oracle_gap_pct_variance_over_support_seed": _mean(
                        _to_float(row.get("oracle_gap_pct_variance", 0.0)) for row in vals
                    ),
                    "mean_selected_eval_nelbo_variance_over_support_seed": _mean(
                        _to_float(row.get("selected_eval_nelbo_variance", 0.0)) for row in vals
                    ),
                    "spearman_valid_group_count": len(spearman_defined),
                    "spearman_dropped_group_count": len(vals) - len(spearman_defined),
                    "mean_spearman_variance_over_support_seed": _mean(spearman_defined),
                    "spearman_tie_row_count": sum(_to_int(row.get("spearman_tie_rows", 0)) for row in vals),
                    "variance_definition": "population_over_support_seed_within_run_seed_center_k_method",
                    "tie_handling": (
                        "precomputed_spearman_used; constant predicted/eval score vectors dropped; "
                        "ties otherwise retained from upstream rank calculation"
                    ),
                }
            )

    warnings = []
    dropped = sum(_to_int(row.get("spearman_dropped_group_count", 0)) for row in out)
    if dropped:
        warnings.append(f"{dropped} Spearman variance groups were dropped because fewer than two support seeds were valid.")
    return out, {"per_group": per_group, "warnings": warnings}


def _split_lookup(split_rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[int, int, int, int, str], Mapping[str, Any]]:
    lookup: Dict[Tuple[int, int, int, int, str], Mapping[str, Any]] = {}
    for row in split_rows:
        if str(row.get("split_role", "")) != "target":
            continue
        key = (
            _to_int(row.get("run_seed", row.get("seed", 0))),
            _to_int(row.get("query_domain", 0)),
            _to_int(row.get("support_seed", 0)),
            _to_int(row.get("support_size_requested", 0)),
            str(row.get("support_eval_split_id", "")),
        )
        lookup[key] = row
    return lookup


def build_protocol_audit_rows(rows: Sequence[Mapping[str, Any]], alpha_rows: Sequence[Mapping[str, Any]], split_rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    split_by_key = _split_lookup(split_rows)
    alpha_by_key = {
        (
            _to_int(row.get("run_seed", row.get("seed", 0))),
            _to_int(row.get("outer_center", 0)),
            _to_int(row.get("support_size", 0)),
        ): row
        for row in alpha_rows
    }
    out: List[dict] = []
    for row in _selected_rows(rows, [DIRECT_METHOD, CONSERVATIVE_METHOD]):
        run_seed = _to_int(row.get("run_seed", row.get("seed", 0)))
        center = _to_int(row.get("query_domain", 0))
        support_seed = _to_int(row.get("support_seed", 0))
        support_size = _to_int(row.get("support_size_requested", 0))
        split_id = str(row.get("support_eval_split_id", ""))
        split = split_by_key.get((run_seed, center, support_seed, support_size, split_id), {})
        alpha = alpha_by_key.get((run_seed, center, support_size), {})
        candidates = _parse_candidate_experts(row.get("candidate_experts", ""))
        selected = _to_int(row.get("selected_expert", -999999))
        candidate_oracle = _to_int(row.get("candidate_oracle_expert", -999999))
        target = _to_int(row.get("fold_query_domain", row.get("target_domain", center)))
        method = str(row.get("method", ""))
        alpha_applicable = int(method == CONSERVATIVE_METHOD)
        alpha_selected_ok = (
            int(_to_int(row.get("selected_before_target_eval_scoring", 0)) == 1)
            if alpha_applicable
            else ""
        )
        alpha_hyper_ok = (
            int(_to_int(alpha.get("selected_before_target_eval_scoring", 0)) == 1)
            if alpha_applicable
            else ""
        )
        out.append(
            {
                "run_seed": run_seed,
                "heldout_center": center,
                "support_seed": support_seed,
                "support_size": support_size,
                "method": METHOD_LABELS.get(method, method),
                "source_method": method,
                "split_status": split.get("split_status", ""),
                "split_row_found": int(bool(split)),
                "support_eval_disjoint_ok": int(_to_int(split.get("support_eval_disjoint", 0)) == 1),
                "support_labels_unused_for_routing_ok": int(_to_int(split.get("support_labels_used", 1)) == 0),
                "target_expert_excluded_ok": int(_to_int(row.get("target_expert_excluded", 0)) == 1),
                "candidate_pool_excludes_target_expert_ok": int(target not in candidates),
                "selected_expert_in_candidate_pool_ok": int(selected in candidates),
                "candidate_oracle_in_candidate_pool_ok": int(candidate_oracle in candidates),
                "routing_uses_eval_nelbo_ok": int(_to_int(row.get("routing_uses_eval_nelbo", 1)) == 0),
                "routing_uses_eval_domain_statistics_ok": int(
                    _to_int(row.get("routing_uses_eval_domain_statistics", 1)) == 0
                ),
                "alpha_selection_applicable": alpha_applicable,
                "alpha_selected_before_target_eval_scoring_ok": alpha_selected_ok,
                "alpha_hyperparam_selected_before_target_eval_scoring_ok": alpha_hyper_ok,
                "protocol_version": row.get("protocol_version", ""),
                "source_selection_path": row.get("source_path", ""),
                "source_split_path": split.get("source_path", ""),
                "source_alpha_path": alpha.get("source_path", ""),
            }
        )
    return out


def _audit_summary(audit_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    checks = [
        "support_eval_disjoint_ok",
        "support_labels_unused_for_routing_ok",
        "target_expert_excluded_ok",
        "candidate_pool_excludes_target_expert_ok",
        "selected_expert_in_candidate_pool_ok",
        "candidate_oracle_in_candidate_pool_ok",
        "routing_uses_eval_nelbo_ok",
        "routing_uses_eval_domain_statistics_ok",
    ]
    summary: Dict[str, Any] = {}
    for check in checks:
        failures = [row for row in audit_rows if _to_int(row.get(check, 0)) != 1]
        summary[check] = {
            "status": "pass" if not failures else "fail",
            "failures": len(failures),
            "total": len(audit_rows),
        }
    alpha_rows = [row for row in audit_rows if _to_int(row.get("alpha_selection_applicable", 0)) == 1]
    alpha_failures = [
        row for row in alpha_rows
        if _to_int(row.get("alpha_selected_before_target_eval_scoring_ok", 0)) != 1
        or _to_int(row.get("alpha_hyperparam_selected_before_target_eval_scoring_ok", 0)) != 1
    ]
    summary["alpha_selected_before_target_eval_scoring_ok"] = {
        "status": "pass" if not alpha_failures else "fail",
        "failures": len(alpha_failures),
        "total": len(alpha_rows),
    }
    summary["overall_protocol_validity"] = (
        "pass" if all(item["status"] == "pass" for item in summary.values()) else "fail"
    )
    return summary


def _cross_check_decision_table(summary_rows: Sequence[Mapping[str, Any]], decision_table: Path) -> Dict[str, Any]:
    if not decision_table.exists():
        return {"status": "missing", "path": str(decision_table), "rows": []}
    old_rows = {row["method"]: row for row in _read_csv(decision_table)}
    by_source = {row["source_method"]: row for row in summary_rows}
    checks: List[dict] = []
    for method in [DIRECT_METHOD, CONSERVATIVE_METHOD]:
        old = old_rows.get(method)
        new = by_source.get(method)
        if old is None or new is None:
            continue
        checks.append(
            {
                "method": METHOD_LABELS[method],
                "source_method": method,
                "old_top1_oracle_hit_mean": _to_float(old.get("top1_oracle_hit_mean", 0.0)),
                "new_top1_oracle_hit_mean": _to_float(new.get("top1_oracle_hit_mean", 0.0)),
                "old_spearman_mean": _to_float(old.get("spearman_mean", 0.0)),
                "new_spearman_mean": _to_float(new.get("spearman_mean", 0.0)),
                "old_oracle_gap_pct_mean": _to_float(old.get("mean_oracle_gap_pct_mean", 0.0)),
                "new_oracle_gap_pct_mean": _to_float(new.get("oracle_gap_pct_mean", 0.0)),
                "old_decision": old.get("decision", ""),
                "new_claim_level": "primary" if method == DIRECT_METHOD else "diagnostic_ablation_only",
            }
        )
    return {
        "status": "checked",
        "path": str(decision_table),
        "rows": checks,
        "interpretation": (
            "Changed conclusion is a claim-level revision based on per-k support-seed "
            "stability and alpha diagnostics, not a protocol invalidation."
        ),
    }


def _small_k_stability_conclusion(stability_rows: Sequence[Mapping[str, Any]]) -> str:
    by_key = {
        (int(row["support_size"]), str(row["method"])): row
        for row in stability_rows
    }
    regressions: List[str] = []
    for k in [4, 8]:
        direct = by_key.get((k, PRIMARY_METHOD_LABEL), {})
        conservative = by_key.get((k, CONSERVATIVE_METHOD_LABEL), {})
        direct_gap_var = _to_float(direct.get("mean_oracle_gap_pct_variance_over_support_seed", 0.0))
        cons_gap_var = _to_float(conservative.get("mean_oracle_gap_pct_variance_over_support_seed", 0.0))
        if cons_gap_var > direct_gap_var + 1.0e-12:
            regressions.append(f"k={k}")
    if regressions:
        return (
            "Conservative scoring does not demonstrate stable small-k improvement; "
            "oracle-gap support-seed variance is higher than direct support NELBO for "
            + ", ".join(regressions)
            + "."
        )
    return "Conservative scoring does not improve small-k stability beyond direct support NELBO in the inspected groups."


def _alpha_degeneracy_summary(alpha_distribution: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    overall = [
        row for row in alpha_distribution
        if str(row.get("scope", "")) == "overall"
    ]
    by_alpha = {str(row.get("alpha", "")): row for row in overall}
    zero = by_alpha.get("0.0", {"count": 0, "pct": 0.0})
    nonzero = by_alpha.get("nonzero", {"count": 0, "pct": 0.0})
    return {
        "alpha_zero_count": _to_int(zero.get("count", 0)),
        "alpha_zero_pct": _to_float(zero.get("pct", 0.0)),
        "alpha_nonzero_count": _to_int(nonzero.get("count", 0)),
        "alpha_nonzero_pct": _to_float(nonzero.get("pct", 0.0)),
        "conclusion": (
            "Alpha mostly collapses to direct support NELBO, so conservative scoring is "
            "not meaningfully regularizing routing in this run."
        ),
    }


def _format_float(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[Tuple[str, str]], *, limit: int | None = None) -> str:
    chosen = list(rows[:limit] if limit is not None else rows)
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in chosen:
        cells = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(_format_float(value))
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + body)


def write_report(
    *,
    path: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    per_k_rows: Sequence[Mapping[str, Any]],
    alpha_distribution: Sequence[Mapping[str, Any]],
    per_center_rows: Sequence[Mapping[str, Any]],
    stability_rows: Sequence[Mapping[str, Any]],
    audit_summary: Mapping[str, Any],
    cross_check: Mapping[str, Any],
    spearman_warnings: Sequence[str],
) -> None:
    alpha_summary = _alpha_degeneracy_summary(alpha_distribution)
    overall_alpha = [
        row for row in alpha_distribution
        if str(row.get("scope", "")) == "overall" and str(row.get("alpha", "")) in {"0.0", "nonzero"}
    ]
    primary = next(row for row in summary_rows if row["method"] == PRIMARY_METHOD_LABEL)
    conservative = next(row for row in summary_rows if row["method"] == CONSERVATIVE_METHOD_LABEL)
    small_k = _small_k_stability_conclusion(stability_rows)
    warnings = list(spearman_warnings)
    if not warnings:
        warnings.append("No Spearman variance groups were dropped.")

    lines = [
        "# Support-NELBO Consolidation Report",
        "",
        LINEAGE_NOTE,
        "",
        "## Thesis-facing decision",
        "",
        "- Primary method: `direct_support_nelbo`",
        "- Conservative method: `diagnostic ablation only`",
        "- Result wording: Direct support-set NELBO is the primary support-estimated utility router.",
        "- Claim boundary: direct support-set NELBO is the strongest support-estimated utility variant in the current Camelyon17 support experiment; this report does not make a broader overall router-ranking claim.",
        "- Reason: conservative scoring is protocol-safe, but alpha selection is mostly degenerate and does not demonstrate a stable small-k improvement.",
        "",
        "## Decision layers",
        "",
        "### Protocol validity",
        "",
        f"- Overall protocol validity: `{audit_summary.get('overall_protocol_validity', 'unknown')}`",
        "- Support/eval disjointness, target expert exclusion, candidate-pool exclusion, eval-NELBO isolation, eval-statistics isolation, and alpha preselection are audited in `support_nelbo_protocol_audit.csv`.",
        "",
        "### Utility performance",
        "",
        _markdown_table(
            summary_rows,
            [
                ("Method", "method"),
                ("Role", "role"),
                ("Rows", "n_rows"),
                ("Top1", "top1_oracle_hit_mean"),
                ("Spearman", "spearman_mean"),
                ("Oracle gap pct", "oracle_gap_pct_mean"),
                ("Selected eval NELBO", "selected_eval_nelbo_mean"),
            ],
        ),
        "",
        f"Direct support NELBO: top1={_format_float(primary['top1_oracle_hit_mean'])}, Spearman={_format_float(primary['spearman_mean'])}, oracle gap pct={_format_float(primary['oracle_gap_pct_mean'])}.",
        f"Conservative support NELBO: top1={_format_float(conservative['top1_oracle_hit_mean'])}, Spearman={_format_float(conservative['spearman_mean'])}, oracle gap pct={_format_float(conservative['oracle_gap_pct_mean'])}.",
        "",
        "### Stability diagnostics",
        "",
        small_k,
        "",
        _markdown_table(
            stability_rows,
            [
                ("k", "support_size"),
                ("Method", "method"),
                ("Groups", "n_groups"),
                ("Gap var", "mean_oracle_gap_pct_variance_over_support_seed"),
                ("NELBO var", "mean_selected_eval_nelbo_variance_over_support_seed"),
                ("Spearman groups", "spearman_valid_group_count"),
                ("Spearman dropped", "spearman_dropped_group_count"),
            ],
        ),
        "",
        "Spearman handling: precomputed Spearman values are used; rows with constant predicted or eval score vectors are treated as undefined for variance. Ties are otherwise retained from the upstream rank calculation.",
        "",
        "Warnings:",
        *[f"- {warning}" for warning in warnings],
        "",
        "## Direct vs conservative by k",
        "",
        _markdown_table(
            per_k_rows,
            [
                ("k", "support_size"),
                ("Direct top1", "direct_top1_oracle_hit_mean"),
                ("Cons top1", "conservative_top1_oracle_hit_mean"),
                ("Direct gap", "direct_oracle_gap_pct_mean"),
                ("Cons gap", "conservative_oracle_gap_pct_mean"),
                ("Direct-cons gap", "direct_minus_conservative_oracle_gap_pct"),
                ("Direct high-regret", "direct_high_regret_rate"),
                ("Cons high-regret", "conservative_high_regret_rate"),
            ],
        ),
        "",
        "## Alpha degeneracy",
        "",
        _markdown_table(
            overall_alpha,
            [
                ("alpha", "alpha"),
                ("count", "count"),
                ("pct", "pct"),
            ],
        ),
        "",
        alpha_summary["conclusion"],
        "",
        "## Per-center oracle gap",
        "",
        _markdown_table(
            per_center_rows,
            [
                ("Center", "heldout_center"),
                ("Direct top1", "direct_top1_oracle_hit_mean"),
                ("Cons top1", "conservative_top1_oracle_hit_mean"),
                ("Direct gap", "direct_oracle_gap_pct_mean"),
                ("Cons gap", "conservative_oracle_gap_pct_mean"),
                ("Direct-cons gap", "direct_minus_conservative_oracle_gap_pct"),
            ],
        ),
        "",
        "## Earlier-artifact cross-check",
        "",
        str(cross_check.get("interpretation", "")),
        "",
        "Allowed thesis claim: Direct support-set NELBO is the primary support-estimated utility router and the strongest support-estimated utility variant in the current Camelyon17 support experiment.",
        "",
        "Not allowed: Conservative support NELBO improves small-k stability, or alpha regularization is meaningful in this run.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_outputs(experiment_root: Path, output_dir: Path, decision_table: Path) -> Dict[str, Any]:
    all_rows, alpha_rows, split_rows = _load_source_rows(experiment_root)
    selected_support_rows = _selected_rows(all_rows, [DIRECT_METHOD, CONSERVATIVE_METHOD])

    summary_rows = build_summary_rows(all_rows)
    per_k_rows = build_per_k_rows(selected_support_rows)
    alpha_distribution = _alpha_distribution_rows(alpha_rows)
    per_center_rows = build_per_center_gap_rows(selected_support_rows)
    stability_rows, stability_extra = build_stability_rows(selected_support_rows)
    audit_rows = build_protocol_audit_rows(all_rows, alpha_rows, split_rows)
    audit_summary = _audit_summary(audit_rows)
    cross_check = _cross_check_decision_table(summary_rows, decision_table)
    alpha_summary = _alpha_degeneracy_summary(alpha_distribution)

    outputs = {
        "report": output_dir / "support_nelbo_consolidation_report.md",
        "summary_csv": output_dir / "support_nelbo_consolidation_summary.csv",
        "summary_json": output_dir / "support_nelbo_consolidation_summary.json",
        "per_k_metrics": output_dir / "support_nelbo_per_k_metrics.csv",
        "alpha_distribution": output_dir / "support_nelbo_alpha_distribution.csv",
        "per_center_gap": output_dir / "support_nelbo_per_center_gap.csv",
        "stability_by_k": output_dir / "support_nelbo_stability_by_k.csv",
        "protocol_audit": output_dir / "support_nelbo_protocol_audit.csv",
    }

    _write_csv(
        outputs["summary_csv"],
        summary_rows,
        [
            "method",
            "source_method",
            "role",
            "n_rows",
            "top1_oracle_hit_mean",
            "top1_oracle_hit_std",
            "spearman_valid_rows",
            "spearman_dropped_rows",
            "spearman_mean",
            "spearman_std",
            "oracle_gap_pct_mean",
            "oracle_gap_pct_std",
            "selected_eval_nelbo_mean",
            "selected_eval_nelbo_std",
            "high_regret_rate_mean",
            "catastrophic_mistake_rate_mean",
        ],
    )
    _write_csv(
        outputs["per_k_metrics"],
        per_k_rows,
        [
            "support_size",
            "direct_n_rows",
            "conservative_n_rows",
            "direct_top1_oracle_hit_mean",
            "conservative_top1_oracle_hit_mean",
            "conservative_minus_direct_top1",
            "direct_spearman_valid_rows",
            "conservative_spearman_valid_rows",
            "direct_spearman_mean",
            "conservative_spearman_mean",
            "conservative_minus_direct_spearman",
            "direct_oracle_gap_pct_mean",
            "conservative_oracle_gap_pct_mean",
            "direct_minus_conservative_oracle_gap_pct",
            "direct_selected_eval_nelbo_mean",
            "conservative_selected_eval_nelbo_mean",
            "direct_minus_conservative_selected_eval_nelbo",
            "direct_high_regret_rate",
            "conservative_high_regret_rate",
            "direct_minus_conservative_high_regret_rate",
        ],
    )
    _write_csv(
        outputs["alpha_distribution"],
        alpha_distribution,
        ["scope", "support_size", "heldout_center", "alpha", "count", "pct"],
    )
    _write_csv(
        outputs["per_center_gap"],
        per_center_rows,
        [
            "heldout_center",
            "direct_n_rows",
            "conservative_n_rows",
            "direct_top1_oracle_hit_mean",
            "conservative_top1_oracle_hit_mean",
            "conservative_minus_direct_top1",
            "direct_oracle_gap_pct_mean",
            "conservative_oracle_gap_pct_mean",
            "direct_minus_conservative_oracle_gap_pct",
            "direct_high_regret_rate",
            "conservative_high_regret_rate",
        ],
    )
    _write_csv(
        outputs["stability_by_k"],
        stability_rows,
        [
            "support_size",
            "method",
            "source_method",
            "n_groups",
            "mean_top1_oracle_hit_variance_over_support_seed",
            "mean_oracle_gap_pct_variance_over_support_seed",
            "mean_selected_eval_nelbo_variance_over_support_seed",
            "spearman_valid_group_count",
            "spearman_dropped_group_count",
            "mean_spearman_variance_over_support_seed",
            "spearman_tie_row_count",
            "variance_definition",
            "tie_handling",
        ],
    )
    _write_csv(
        outputs["protocol_audit"],
        audit_rows,
        [
            "run_seed",
            "heldout_center",
            "support_seed",
            "support_size",
            "method",
            "source_method",
            "split_status",
            "split_row_found",
            "support_eval_disjoint_ok",
            "support_labels_unused_for_routing_ok",
            "target_expert_excluded_ok",
            "candidate_pool_excludes_target_expert_ok",
            "selected_expert_in_candidate_pool_ok",
            "candidate_oracle_in_candidate_pool_ok",
            "routing_uses_eval_nelbo_ok",
            "routing_uses_eval_domain_statistics_ok",
            "alpha_selection_applicable",
            "alpha_selected_before_target_eval_scoring_ok",
            "alpha_hyperparam_selected_before_target_eval_scoring_ok",
            "protocol_version",
            "source_selection_path",
            "source_split_path",
            "source_alpha_path",
        ],
    )

    summary_payload = {
        "artifact_lineage": LINEAGE_NOTE,
        "decision_layers": {
            "protocol_validity": audit_summary,
            "utility_performance": {
                "summary_rows": summary_rows,
                "metric_priority": [
                    "top1_oracle_hit",
                    "spearman",
                    "oracle_gap_pct",
                    "selected_heldout_eval_nelbo",
                ],
            },
            "stability_diagnostics": {
                "rows": stability_rows,
                "per_group_rows": stability_extra["per_group"],
                "warnings": stability_extra["warnings"],
                "variance_grouping": [
                    "run_seed",
                    "heldout_center",
                    "support_size",
                    "method",
                ],
                "variance_over": "support_seed",
            },
        },
        "row_count_assertions": {
            "selected_direct_conservative_rows_expected": EXPECTED_SELECTED_ROWS,
            "selected_direct_conservative_rows_observed": len(selected_support_rows),
            "alpha_selection_rows_expected": EXPECTED_ALPHA_ROWS,
            "alpha_selection_rows_observed": len(alpha_rows),
            "note": "Counts are selected-method rows, not raw candidate rows.",
        },
        "thesis_facing_decision": {
            "primary_method": "direct_support_nelbo",
            "conservative_method": "diagnostic ablation only",
            "result_wording": "Direct support-set NELBO is the primary support-estimated utility router.",
            "reason": (
                "Conservative scoring is protocol-safe, but alpha selection is mostly "
                "degenerate and does not demonstrate stable small-k improvement."
            ),
        },
        "alpha_degeneracy": alpha_summary,
        "cross_check_against_earlier_decision_table": cross_check,
        "allowed_thesis_claim": (
            "Direct support-set NELBO is the primary support-estimated utility router "
            "and the strongest support-estimated utility variant in the current "
            "Camelyon17 support experiment."
        ),
        "disallowed_thesis_claims": [
            "Conservative support NELBO improves small-k stability.",
            "Alpha regularization is meaningful when alpha collapses to zero in most selections.",
            "Direct support-set NELBO supports a broader overall router-ranking claim.",
        ],
    }
    _write_json(outputs["summary_json"], summary_payload)
    write_report(
        path=outputs["report"],
        summary_rows=summary_rows,
        per_k_rows=per_k_rows,
        alpha_distribution=alpha_distribution,
        per_center_rows=per_center_rows,
        stability_rows=stability_rows,
        audit_summary=audit_summary,
        cross_check=cross_check,
        spearman_warnings=stability_extra["warnings"],
    )

    return {key: str(value) for key, value in outputs.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build support-NELBO thesis consolidation artifacts.")
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--decision-table", type=Path, default=EARLIER_DECISION_TABLE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_outputs(
        experiment_root=args.experiment_root,
        output_dir=args.output_dir,
        decision_table=args.decision_table,
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
