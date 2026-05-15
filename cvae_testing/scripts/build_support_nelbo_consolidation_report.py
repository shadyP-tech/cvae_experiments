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
RUN_ID_TEMPLATE = "support_utility_v2_seed{seed}"
OUTPUT_PREFIX = "support_nelbo"
DATASET_CONTEXT = "camelyon17"

DIRECT_METHOD = "support_set_nelbo_top1"
CONSERVATIVE_METHOD = "support_set_nelbo_conservative"
METADATA_METHOD = "support_metadata_routing"
STATIC_METHOD = "support_static_embedding_routing"
RANDOM_METHOD = "support_random_expert_floor"
METHOD_LABELS = {
    DIRECT_METHOD: "direct_support_nelbo",
    CONSERVATIVE_METHOD: "conservative_support_nelbo",
    METADATA_METHOD: "metadata_routing",
    STATIC_METHOD: "static_embedding_routing",
    RANDOM_METHOD: "random_expert_floor",
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
EXPECTED_RAW_SUPPORT_ROWS = (
    len(RUN_SEEDS)
    * len(HELDOUT_CENTERS)
    * len(SUPPORT_SEEDS)
    * sum(SUPPORT_SIZES)
    * max(len(HELDOUT_CENTERS) - 1, 0)
)
LINEAGE_NOTE = (
    "This report supersedes earlier selection summaries for thesis interpretation. "
    "It does not invalidate their protocol checks; it revises the claim level based "
    "on per-k support-seed stability and alpha-selection diagnostics."
)
SEED_TOP1_MARGIN = 0.0
STATIC_GAP_MATERIAL_LOSS_TOLERANCE_PCT = 1.0
RANDOM_FLOOR_TOP1_MARGIN = 0.05


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


def _dynamic_fieldnames(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            name = str(key)
            if name in seen:
                continue
            seen.add(name)
            fieldnames.append(name)
    return fieldnames


def _write_dynamic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    _write_csv(path, rows, _dynamic_fieldnames(rows))


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


def _quantile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not vals:
        return 0.0
    pos = max(0.0, min(1.0, float(q))) * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    weight = pos - lo
    return float(vals[lo] * (1.0 - weight) + vals[hi] * weight)


def _entropy_from_counts(counts: Mapping[int, int]) -> float:
    total = int(sum(int(value) for value in counts.values()))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = float(count) / float(total)
        if p > 0:
            entropy -= p * math.log2(p)
    return float(entropy)


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


def _json_score_keys(value: object) -> List[int]:
    try:
        payload = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    out: List[int] = []
    for key in payload.keys():
        try:
            out.append(int(float(key)))
        except Exception:
            continue
    return out


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
        run_dir = experiment_root / RUN_ID_TEMPLATE.format(seed=seed)
        reports_dir = run_dir / "reports"
        required = [
            run_dir / "config_resolved.yaml",
            reports_dir / "support_response_sample_selections.csv",
            reports_dir / "support_utility_selected_hyperparams.csv",
            reports_dir / "support_response_support_nelbo_rows.csv",
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


def _load_source_rows(experiment_root: Path) -> Tuple[List[dict], List[dict], List[dict], int]:
    selected_rows: List[dict] = []
    alpha_rows: List[dict] = []
    split_rows: List[dict] = []
    raw_support_rows_observed = 0
    for run_dir in _result_paths(experiment_root):
        run_seed = _run_seed_from_run_dir(run_dir)
        reports_dir = run_dir / "reports"

        sample_path = reports_dir / "support_response_sample_selections.csv"
        for row in _read_csv(sample_path):
            method = str(row.get("method", ""))
            if method not in {DIRECT_METHOD, CONSERVATIVE_METHOD, METADATA_METHOD, STATIC_METHOD, RANDOM_METHOD}:
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

        raw_path = reports_dir / "support_response_support_nelbo_rows.csv"
        raw_support_rows_observed += len(_read_csv(raw_path))

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
    if raw_support_rows_observed != EXPECTED_RAW_SUPPORT_ROWS:
        raise RuntimeError(
            "Raw support-NELBO row-count assertion failed. "
            f"Expected {EXPECTED_RAW_SUPPORT_ROWS} raw rows; found {raw_support_rows_observed}."
        )
    return selected_rows, alpha_rows, split_rows, raw_support_rows_observed


def _load_run_artifact_rows(
    experiment_root: Path,
    filename: str,
    *,
    required: bool,
) -> List[dict]:
    out: List[dict] = []
    for run_dir in _result_paths(experiment_root):
        run_seed = _run_seed_from_run_dir(run_dir)
        path = run_dir / "reports" / filename
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Missing required support-NELBO source artifact: {path}")
            continue
        for row in _read_csv(path):
            row = dict(row)
            row["run_seed"] = run_seed
            row["run_id"] = run_dir.name
            row["source_path"] = str(path)
            out.append(row)
    return out


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
        "role": (
            "primary"
            if method == DIRECT_METHOD
            else "diagnostic_ablation"
            if method == CONSERVATIVE_METHOD
            else "diagnostic_floor"
            if method == RANDOM_METHOD
            else "baseline"
        ),
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
    methods = [METADATA_METHOD, STATIC_METHOD, RANDOM_METHOD, DIRECT_METHOD, CONSERVATIVE_METHOD]
    return [
        _summarize_method(rows, method)
        for method in methods
        if any(str(row.get("method", "")) == method for row in rows)
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


def _matched_rows_by_unit(rows: Sequence[Mapping[str, Any]], method: str) -> Dict[Tuple[int, int, int, int], Mapping[str, Any]]:
    return {
        (
            _to_int(row.get("run_seed", row.get("seed", 0))),
            _to_int(row.get("query_domain", row.get("target_domain", 0))),
            _to_int(row.get("support_size_requested", row.get("support_size", 0))),
            _to_int(row.get("support_seed", 0)),
        ): row
        for row in rows
        if str(row.get("method", "")) == method
    }


def build_per_magnification_decision_rows(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    metadata = _matched_rows_by_unit(rows, METADATA_METHOD)
    static = _matched_rows_by_unit(rows, STATIC_METHOD)
    out: List[dict] = []
    for row in sorted(
        [r for r in rows if str(r.get("method", "")) == DIRECT_METHOD],
        key=lambda r: (
            _to_int(r.get("query_domain", 0)),
            _to_int(r.get("support_size_requested", 0)),
            _to_int(r.get("support_seed", 0)),
            _to_int(r.get("run_seed", r.get("seed", 0))),
        ),
    ):
        key = (
            _to_int(row.get("run_seed", row.get("seed", 0))),
            _to_int(row.get("query_domain", row.get("target_domain", 0))),
            _to_int(row.get("support_size_requested", 0)),
            _to_int(row.get("support_seed", 0)),
        )
        metadata_row = metadata.get(key, {})
        static_row = static.get(key, {})
        out.append(
            {
                "heldout_magnification": key[1],
                "support_size": key[2],
                "support_seed": key[3],
                "seed": key[0],
                "selected_expert": _to_int(row.get("selected_expert", 0)),
                "oracle_expert": _to_int(row.get("candidate_oracle_expert", row.get("oracle_expert", 0))),
                "support_nelbo_selected": _to_float(row.get("mean_support_nelbo", 0.0)),
                "eval_nelbo_selected": _to_float(row.get("selected_nelbo", 0.0)),
                "eval_nelbo_oracle": _to_float(row.get("candidate_oracle_nelbo", row.get("oracle_nelbo", 0.0))),
                "oracle_gap_pct": _to_float(row.get("mean_oracle_gap_pct", row.get("oracle_gap_pct", 0.0))),
                "metadata_selected_expert": _to_int(metadata_row.get("selected_expert", 0)),
                "static_embedding_selected_expert": _to_int(static_row.get("selected_expert", 0)),
            }
        )
    return out


def build_rank_consistency_rows(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    out: List[dict] = []
    for center in HELDOUT_CENTERS:
        center_rows = [
            row for row in rows
            if str(row.get("method", "")) == DIRECT_METHOD
            and _to_int(row.get("query_domain", row.get("target_domain", 0))) == int(center)
        ]
        spearman_vals = [_to_float(row.get("spearman", 0.0)) for row in center_rows if _spearman_valid(row)]
        spearman_sorted = sorted(spearman_vals)
        if not spearman_sorted:
            median_spearman = 0.0
        elif len(spearman_sorted) % 2 == 1:
            median_spearman = float(spearman_sorted[len(spearman_sorted) // 2])
        else:
            mid = len(spearman_sorted) // 2
            median_spearman = float((spearman_sorted[mid - 1] + spearman_sorted[mid]) / 2.0)
        by_k: Dict[int, List[Mapping[str, Any]]] = {}
        for row in center_rows:
            by_k.setdefault(_to_int(row.get("support_size_requested", 0)), []).append(row)
        gap_by_k = {
            k: _mean(_to_float(row.get("mean_oracle_gap_pct", 0.0)) for row in vals)
            for k, vals in by_k.items()
        }
        best_k = min(gap_by_k, key=gap_by_k.get) if gap_by_k else ""
        worst_k = max(gap_by_k, key=gap_by_k.get) if gap_by_k else ""
        out.append(
            {
                "heldout_magnification": int(center),
                "mean_spearman": _mean(spearman_vals),
                "median_spearman": median_spearman,
                "top1": _mean(_to_float(row.get("top1_oracle_hit", 0.0)) for row in center_rows),
                "oracle_gap_pct": _mean(_to_float(row.get("mean_oracle_gap_pct", 0.0)) for row in center_rows),
                "best_k": best_k,
                "worst_k": worst_k,
            }
        )
    return out


def build_seed_stability_rows(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    return [
        {
            "seed": seed,
            "method": method,
            "top1_oracle_hit_mean": _mean(
                _to_float(row.get("top1_oracle_hit", 0.0))
                for row in rows
                if _to_int(row.get("run_seed", row.get("seed", 0))) == int(seed)
                and str(row.get("method", "")) == method
            ),
            "spearman_mean": _mean(
                _to_float(row.get("spearman", 0.0))
                for row in rows
                if _to_int(row.get("run_seed", row.get("seed", 0))) == int(seed)
                and str(row.get("method", "")) == method
                and _spearman_valid(row)
            ),
            "oracle_gap_pct_mean": _mean(
                _to_float(row.get("mean_oracle_gap_pct", 0.0))
                for row in rows
                if _to_int(row.get("run_seed", row.get("seed", 0))) == int(seed)
                and str(row.get("method", "")) == method
            ),
        }
        for seed in RUN_SEEDS
        for method in [METADATA_METHOD, STATIC_METHOD, RANDOM_METHOD, DIRECT_METHOD, CONSERVATIVE_METHOD]
        if any(
            _to_int(row.get("run_seed", row.get("seed", 0))) == int(seed)
            and str(row.get("method", "")) == method
            for row in rows
        )
    ]


def build_support_size_monotonicity_rows(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    groups: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (
                str(row.get("method", "")),
                _to_int(row.get("support_size_requested", row.get("support_size", 0))),
            ),
            [],
        ).append(row)
    out: List[dict] = []
    for (method, support_size), vals in sorted(groups.items(), key=lambda item: item[0]):
        spearman_vals = [_to_float(row.get("spearman", 0.0)) for row in vals if _spearman_valid(row)]
        out.append(
            {
                "method": METHOD_LABELS.get(method, method),
                "source_method": method,
                "support_size": support_size,
                "n_rows": len(vals),
                "n_run_seeds": len({_to_int(row.get("run_seed", row.get("seed", 0))) for row in vals}),
                "n_support_seeds": len({_to_int(row.get("support_seed", 0)) for row in vals}),
                "top1_oracle_hit": _mean(_to_float(row.get("top1_oracle_hit", 0.0)) for row in vals),
                "spearman": _mean(spearman_vals),
                "spearman_valid_rows": len(spearman_vals),
                "mean_oracle_gap_pct": _mean(
                    _to_float(row.get("mean_oracle_gap_pct", row.get("oracle_gap_pct", 0.0))) for row in vals
                ),
                "rank_agreement_spearman": _mean(spearman_vals),
                "selection_stability_unique_experts": len(
                    {_to_int(row.get("selected_expert", -1)) for row in vals}
                ),
                "support_margin": _mean(_to_float(row.get("support_margin", 0.0)) for row in vals),
                "eval_margin": _mean(_to_float(row.get("eval_margin", 0.0)) for row in vals),
            }
        )
    return out


def build_margin_diagnostic_rows(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    eligible = _selected_rows(rows, [DIRECT_METHOD, CONSERVATIVE_METHOD])
    margins_by_method: Dict[str, List[float]] = {}
    for row in eligible:
        method = str(row.get("method", ""))
        margins_by_method.setdefault(method, []).append(_to_float(row.get("support_margin", 0.0)))
    quantiles = {
        method: (
            _quantile(values, 1.0 / 3.0),
            _quantile(values, 2.0 / 3.0),
        )
        for method, values in margins_by_method.items()
        if values
    }
    out: List[dict] = []
    for row in eligible:
        method = str(row.get("method", ""))
        margin = _to_float(row.get("support_margin", 0.0))
        lo, hi = quantiles.get(method, (0.0, 0.0))
        bucket = "low" if margin <= lo else ("mid" if margin <= hi else "high")
        out.append(
            {
                "run_seed": _to_int(row.get("run_seed", row.get("seed", 0))),
                "query_domain": _to_int(row.get("query_domain", 0)),
                "support_seed": _to_int(row.get("support_seed", 0)),
                "support_size": _to_int(row.get("support_size_requested", 0)),
                "method": METHOD_LABELS.get(method, method),
                "source_method": method,
                "selected_expert": _to_int(row.get("selected_expert", -1)),
                "candidate_oracle_expert": _to_int(row.get("candidate_oracle_expert", -1)),
                "support_margin": margin,
                "eval_margin": _to_float(row.get("eval_margin", 0.0)),
                "support_margin_quantile": bucket,
                "oracle_gap_pct": _to_float(row.get("mean_oracle_gap_pct", row.get("oracle_gap_pct", 0.0))),
                "top1_oracle_hit": _to_int(row.get("top1_oracle_hit", 0)),
            }
        )
    return out


def build_selection_entropy_rows(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    groups: Dict[Tuple[str, int, int], List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (
                str(row.get("method", "")),
                _to_int(row.get("query_domain", 0)),
                _to_int(row.get("support_size_requested", 0)),
            ),
            [],
        ).append(row)
    out: List[dict] = []
    for (method, query_domain, support_size), vals in sorted(groups.items(), key=lambda item: item[0]):
        counts: Dict[int, int] = {}
        for row in vals:
            selected = _to_int(row.get("selected_expert", -1))
            counts[selected] = counts.get(selected, 0) + 1
        entropy = _entropy_from_counts(counts)
        mean_gap = _mean(
            _to_float(row.get("mean_oracle_gap_pct", row.get("oracle_gap_pct", 0.0))) for row in vals
        )
        if entropy <= 0.5 and mean_gap <= 5.0:
            interpretation = "reliable"
        elif entropy > 0.5 and mean_gap <= 5.0:
            interpretation = "low_regret_ambiguity"
        elif entropy > 0.5 and mean_gap > 5.0:
            interpretation = "unstable_router"
        else:
            interpretation = "systematically_wrong"
        out.append(
            {
                "method": METHOD_LABELS.get(method, method),
                "source_method": method,
                "query_domain": query_domain,
                "support_size": support_size,
                "n_rows": len(vals),
                "n_run_seeds": len({_to_int(row.get("run_seed", row.get("seed", 0))) for row in vals}),
                "n_support_seeds": len({_to_int(row.get("support_seed", 0)) for row in vals}),
                "selection_entropy": entropy,
                "selected_expert_counts_json": json.dumps(
                    {str(k): int(v) for k, v in sorted(counts.items())},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "mean_oracle_gap_pct": mean_gap,
                "interpretation": interpretation,
            }
        )
    return out


def build_pass_rule_summary(
    summary_rows: Sequence[Mapping[str, Any]],
    seed_stability_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_source = {str(row.get("source_method", "")): row for row in summary_rows}
    seed_by_method = {
        (int(row["seed"]), str(row["method"])): row
        for row in seed_stability_rows
    }

    direct = by_source.get(DIRECT_METHOD, {})
    metadata = by_source.get(METADATA_METHOD, {})
    static = by_source.get(STATIC_METHOD, {})
    random_floor = by_source.get(RANDOM_METHOD, {})

    per_seed_top1 = []
    per_seed_spearman = []
    per_seed_gap = []
    for seed in RUN_SEEDS:
        direct_seed = seed_by_method.get((int(seed), DIRECT_METHOD))
        metadata_seed = seed_by_method.get((int(seed), METADATA_METHOD))
        if direct_seed is None or metadata_seed is None:
            continue
        direct_top1 = _to_float(direct_seed.get("top1_oracle_hit_mean", 0.0))
        metadata_top1 = _to_float(metadata_seed.get("top1_oracle_hit_mean", 0.0))
        direct_gap = _to_float(direct_seed.get("oracle_gap_pct_mean", 0.0))
        metadata_gap = _to_float(metadata_seed.get("oracle_gap_pct_mean", 0.0))
        direct_spearman_seed = _to_float(direct_seed.get("spearman_mean", 0.0))
        metadata_spearman_seed = _to_float(metadata_seed.get("spearman_mean", 0.0))
        per_seed_top1.append(
            {
                "seed": int(seed),
                "direct_top1": direct_top1,
                "metadata_top1": metadata_top1,
                "margin": direct_top1 - metadata_top1,
                "pass": int(direct_top1 > metadata_top1 + SEED_TOP1_MARGIN),
            }
        )
        per_seed_spearman.append(
            {
                "seed": int(seed),
                "direct_spearman": direct_spearman_seed,
                "metadata_spearman": metadata_spearman_seed,
                "margin": direct_spearman_seed - metadata_spearman_seed,
                "pass": int(direct_spearman_seed > metadata_spearman_seed),
            }
        )
        per_seed_gap.append(
            {
                "seed": int(seed),
                "direct_oracle_gap_pct": direct_gap,
                "metadata_oracle_gap_pct": metadata_gap,
                "direct_minus_metadata_gap_pct": direct_gap - metadata_gap,
                "pass": int(direct_gap <= metadata_gap + 1.0e-12),
            }
        )

    direct_top1 = _to_float(direct.get("top1_oracle_hit_mean", 0.0))
    metadata_top1 = _to_float(metadata.get("top1_oracle_hit_mean", 0.0))
    direct_spearman = _to_float(direct.get("spearman_mean", 0.0))
    metadata_spearman = _to_float(metadata.get("spearman_mean", 0.0))
    direct_gap = _to_float(direct.get("oracle_gap_pct_mean", 0.0))
    metadata_gap = _to_float(metadata.get("oracle_gap_pct_mean", 0.0))
    static_gap = _to_float(static.get("oracle_gap_pct_mean", float("inf")), default=float("inf"))
    random_top1 = _to_float(random_floor.get("top1_oracle_hit_mean", float("nan")), default=float("nan"))

    checks = {
        "top1_beats_metadata_consistent_seed_margin": {
            "status": "pass" if per_seed_top1 and all(row["pass"] for row in per_seed_top1) else "fail",
            "margin_required": SEED_TOP1_MARGIN,
            "per_seed": per_seed_top1,
        },
        "spearman_beats_metadata_and_positive": {
            "status": "pass" if direct_spearman > metadata_spearman and direct_spearman > 0.0 else "fail",
            "direct_spearman": direct_spearman,
            "metadata_spearman": metadata_spearman,
            "per_seed": per_seed_spearman,
        },
        "oracle_gap_lower_than_metadata": {
            "status": "pass" if direct_gap < metadata_gap else "fail",
            "direct_oracle_gap_pct": direct_gap,
            "metadata_oracle_gap_pct": metadata_gap,
        },
        "no_seed_oracle_gap_sign_reversal_against_metadata": {
            "status": "pass" if per_seed_gap and all(row["pass"] for row in per_seed_gap) else "fail",
            "per_seed": per_seed_gap,
        },
        "oracle_gap_not_materially_worse_than_static_embedding": {
            "status": (
                "pass"
                if math.isfinite(static_gap)
                and direct_gap <= static_gap + STATIC_GAP_MATERIAL_LOSS_TOLERANCE_PCT
                else "fail"
            ),
            "direct_oracle_gap_pct": direct_gap,
            "static_embedding_oracle_gap_pct": static_gap,
            "material_loss_tolerance_pct": STATIC_GAP_MATERIAL_LOSS_TOLERANCE_PCT,
        },
        "top1_meaningfully_above_random_floor": {
            "status": (
                "pass"
                if math.isfinite(random_top1)
                and direct_top1 >= random_top1 + RANDOM_FLOOR_TOP1_MARGIN
                else "fail"
            ),
            "direct_top1": direct_top1,
            "random_floor_top1": random_top1,
            "margin_required": RANDOM_FLOOR_TOP1_MARGIN,
            "chance_level_note": (
                "Random floor is matched to candidate count per held-out fold; aggregate top1 must exceed it."
            ),
        },
    }
    context = str(DATASET_CONTEXT).strip().lower()
    if context == "midogpp_scanner":
        gap_all = checks["no_seed_oracle_gap_sign_reversal_against_metadata"]["status"] == "pass"
        top1_improving = sum(int(row["pass"]) for row in per_seed_top1)
        spearman_improving = sum(int(row["pass"]) for row in per_seed_spearman)
        n_seed_pairs = max(len(per_seed_gap), 1)
        strong = (
            gap_all
            and top1_improving == n_seed_pairs
            and spearman_improving == n_seed_pairs
            and checks["top1_meaningfully_above_random_floor"]["status"] == "pass"
        )
        passed = (
            gap_all
            and (top1_improving >= 2 or spearman_improving >= 2)
            and checks["top1_meaningfully_above_random_floor"]["status"] == "pass"
        )
        weak = (
            direct_gap < metadata_gap
            or (direct_spearman > metadata_spearman and direct_spearman > 0.0)
        )
        if strong:
            verdict = "STRONG PASS"
        elif passed:
            verdict = "PASS"
        elif weak:
            verdict = "WEAK PASS"
        else:
            verdict = "FAIL"
        return {
            "verdict": verdict,
            "checks": checks,
            "metric_priority": ["mean_oracle_gap_pct", "spearman", "top1_oracle_hit"],
            "decision_tiers": {
                "STRONG PASS": "beats metadata on top1, Spearman, and oracle gap in all seeds",
                "PASS": "beats metadata on oracle gap in all seeds, improves top1 or Spearman in at least 2/3 seeds, and clears random floor",
                "WEAK PASS": "improves oracle gap or Spearman, but top1 or selection stability is weak",
                "DIAGNOSTIC ONLY": "protocol passes but confounding, degeneracy, or fold instability limits claims",
                "FAIL": "support-NELBO is worse than metadata on oracle gap",
                "REJECTED": "missing preflight validity, utility matrix, protocol audit, or split validity",
            },
            "interpretation": (
                "MIDOG++ scanner-indexed PASS is conditional on preflight fold validity and confounding diagnostics; "
                "it is acquisition-domain evidence, not pure scanner-shift evidence."
            ),
        }
    first_five = [
        "top1_beats_metadata_consistent_seed_margin",
        "spearman_beats_metadata_and_positive",
        "oracle_gap_lower_than_metadata",
        "no_seed_oracle_gap_sign_reversal_against_metadata",
        "oracle_gap_not_materially_worse_than_static_embedding",
    ]
    first_five_pass = all(checks[name]["status"] == "pass" for name in first_five)
    random_pass = checks["top1_meaningfully_above_random_floor"]["status"] == "pass"
    if first_five_pass and random_pass:
        verdict = "PASS"
    elif first_five_pass:
        verdict = "DOWNGRADED_RANDOM_FLOOR"
    else:
        verdict = "FAIL"
    return {
        "verdict": verdict,
        "checks": checks,
        "interpretation": (
            "PASS supports cross-domain plausibility under BreakHis magnification shift only. "
            "DOWNGRADED_RANDOM_FLOOR means the non-random criteria pass but top1 is too close to matched random."
        ),
    }


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
        routing_score_experts = set(_json_score_keys(row.get("predicted_score_by_expert_json", "{}")))
        heldout_in_routing_scores = int(target in routing_score_experts)
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
        patient_required = _to_int(split.get("patient_disjoint_required", 0))
        support_eval_patient_disjoint_ok = int(
            patient_required == 0
            or _to_int(split.get("support_eval_patient_disjoint", 0)) == 1
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
                "patient_disjoint_required": patient_required,
                "support_eval_patient_disjoint_ok": support_eval_patient_disjoint_ok,
                "missing_patient_id_count": _to_int(split.get("missing_patient_id_count", 0)),
                "support_label_counts_json": split.get("support_label_counts_json", "{}"),
                "support_labels_unused_for_routing_ok": int(_to_int(split.get("support_labels_used", 1)) == 0),
                "target_expert_excluded_ok": int(_to_int(row.get("target_expert_excluded", 0)) == 1),
                "candidate_pool_excludes_target_expert_ok": int(target not in candidates),
                "selected_expert_in_candidate_pool_ok": int(selected in candidates),
                "candidate_oracle_in_candidate_pool_ok": int(candidate_oracle in candidates),
                "routing_uses_eval_nelbo_ok": int(_to_int(row.get("routing_uses_eval_nelbo", 1)) == 0),
                "routing_uses_eval_domain_statistics_ok": int(
                    _to_int(row.get("routing_uses_eval_domain_statistics", 1)) == 0
                ),
                "routing_time_scores_exclude_heldout_expert_ok": int(not heldout_in_routing_scores),
                "heldout_expert_checkpoint_used_only_for_oracle_diagnostic": int(not heldout_in_routing_scores),
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
        "support_eval_patient_disjoint_ok",
        "support_labels_unused_for_routing_ok",
        "target_expert_excluded_ok",
        "candidate_pool_excludes_target_expert_ok",
        "selected_expert_in_candidate_pool_ok",
        "candidate_oracle_in_candidate_pool_ok",
        "routing_uses_eval_nelbo_ok",
        "routing_uses_eval_domain_statistics_ok",
        "routing_time_scores_exclude_heldout_expert_ok",
        "heldout_expert_checkpoint_used_only_for_oracle_diagnostic",
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


def outputs_protocol_audit_name() -> str:
    if OUTPUT_PREFIX == "support_nelbo":
        return "support_nelbo_protocol_audit.csv"
    return f"{OUTPUT_PREFIX}_protocol_audit.csv"


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
    pass_rule: Mapping[str, Any],
    spearman_warnings: Sequence[str],
) -> None:
    alpha_summary = _alpha_degeneracy_summary(alpha_distribution)
    overall_alpha = [
        row for row in alpha_distribution
        if str(row.get("scope", "")) == "overall" and str(row.get("alpha", "")) in {"0.0", "nonzero"}
    ]
    primary = next(row for row in summary_rows if row["method"] == PRIMARY_METHOD_LABEL)
    conservative = next(row for row in summary_rows if row["method"] == CONSERVATIVE_METHOD_LABEL)
    is_breakhis = str(DATASET_CONTEXT).strip().lower() == "breakhis"
    title = (
        "BreakHis Cross-Dataset Stress Test Of Direct Support-NELBO"
        if is_breakhis
        else "Support-NELBO Consolidation Report"
    )
    claim_boundary = (
        "direct support-set NELBO is stress-tested under BreakHis magnification-domain shift; "
        "this does not prove robustness across hospital, scanner, staining, lab, or patient-population shifts."
        if is_breakhis
        else "direct support-set NELBO is the strongest support-estimated utility variant in the current Camelyon17 support experiment; this report does not make a broader overall router-ranking claim."
    )
    allowed_claim = (
        "Direct support-set NELBO was stress-tested as a target-local utility estimator under BreakHis leave-one-magnification-out routing using unlabeled patient-disjoint target support and held-out NELBO utility evaluation."
        if is_breakhis
        else "Direct support-set NELBO is the primary support-estimated utility router and the strongest support-estimated utility variant in the current Camelyon17 support experiment."
    )
    disallowed_claim = (
        "This experiment does not prove general support-NELBO robustness across all medical domain shifts; BreakHis magnification shift is narrower than hospital, scanner, staining, lab, or patient-population shift."
        if is_breakhis
        else "Conservative support NELBO improves small-k stability, or alpha regularization is meaningful in this run."
    )
    small_k = _small_k_stability_conclusion(stability_rows)
    warnings = list(spearman_warnings)
    if not warnings:
        warnings.append("No Spearman variance groups were dropped.")

    lines = [
        f"# {title}",
        "",
        LINEAGE_NOTE,
        "",
        "## Thesis-facing decision",
        "",
        "- Primary method: `direct_support_nelbo`",
        "- Conservative method: `diagnostic ablation only`",
        "- Result wording: Direct support-set NELBO is the primary support-estimated utility router.",
        *([f"- PASS-rule verdict: `{pass_rule.get('verdict', 'unknown')}`"] if is_breakhis else []),
        f"- Claim boundary: {claim_boundary}",
        "- Reason: conservative scoring is protocol-safe, but alpha selection is mostly degenerate and does not demonstrate a stable small-k improvement.",
        "",
        "## Decision layers",
        "",
        "### Protocol validity",
        "",
        f"- Overall protocol validity: `{audit_summary.get('overall_protocol_validity', 'unknown')}`",
        f"- Support/eval disjointness, target expert exclusion, candidate-pool exclusion, eval-NELBO isolation, eval-statistics isolation, and alpha preselection are audited in `{outputs_protocol_audit_name()}`.",
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
        f"Allowed thesis claim: {allowed_claim}",
        "",
        f"Not allowed: {disallowed_claim}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_outputs(experiment_root: Path, output_dir: Path, decision_table: Path) -> Dict[str, Any]:
    all_rows, alpha_rows, split_rows, raw_support_rows_observed = _load_source_rows(experiment_root)
    selected_support_rows = _selected_rows(all_rows, [DIRECT_METHOD, CONSERVATIVE_METHOD])

    summary_rows = build_summary_rows(all_rows)
    per_k_rows = build_per_k_rows(selected_support_rows)
    alpha_distribution = _alpha_distribution_rows(alpha_rows)
    per_center_rows = build_per_center_gap_rows(selected_support_rows)
    per_magnification_rows = build_per_magnification_decision_rows(all_rows)
    rank_consistency_rows = build_rank_consistency_rows(all_rows)
    seed_stability_rows = build_seed_stability_rows(all_rows)
    support_size_monotonicity_rows = build_support_size_monotonicity_rows(all_rows)
    margin_diagnostic_rows = build_margin_diagnostic_rows(all_rows)
    selection_entropy_rows = build_selection_entropy_rows(all_rows)
    context = str(DATASET_CONTEXT).strip().lower()
    pass_rule = (
        build_pass_rule_summary(summary_rows, seed_stability_rows)
        if context in {"breakhis", "midogpp_scanner"}
        else {
            "verdict": "NOT_APPLICABLE_CAMELYON17_DEFAULT",
            "checks": {},
            "interpretation": "BreakHis PASS/downgrade rule is only applied when --dataset-context breakhis.",
        }
    )
    stability_rows, stability_extra = build_stability_rows(selected_support_rows)
    audit_rows = build_protocol_audit_rows(all_rows, alpha_rows, split_rows)
    audit_summary = _audit_summary(audit_rows)
    cross_check = _cross_check_decision_table(summary_rows, decision_table)
    alpha_summary = _alpha_degeneracy_summary(alpha_distribution)
    consolidated_sample_rows = _load_run_artifact_rows(
        experiment_root,
        "support_response_sample_selections.csv",
        required=True,
    )
    consolidated_raw_support_rows = _load_run_artifact_rows(
        experiment_root,
        "support_response_support_nelbo_rows.csv",
        required=True,
    )
    consolidated_metadata_diag_rows = _load_run_artifact_rows(
        experiment_root,
        "support_response_metadata_baseline_diagnostics.csv",
        required=False,
    )

    if OUTPUT_PREFIX == "support_nelbo":
        outputs = {
            "report": output_dir / "support_nelbo_consolidation_report.md",
            "summary_csv": output_dir / "support_nelbo_consolidation_summary.csv",
            "summary_json": output_dir / "support_nelbo_consolidation_summary.json",
            "per_k_metrics": output_dir / "support_nelbo_per_k_metrics.csv",
            "alpha_distribution": output_dir / "support_nelbo_alpha_distribution.csv",
            "per_center_gap": output_dir / "support_nelbo_per_center_gap.csv",
            "per_magnification_decisions": output_dir / "support_nelbo_per_magnification_decisions.csv",
            "rank_consistency": output_dir / "support_nelbo_rank_consistency.csv",
            "stability_by_k": output_dir / "support_nelbo_stability_by_k.csv",
            "seed_stability": output_dir / "support_nelbo_seed_stability.csv",
            "protocol_audit": output_dir / "support_nelbo_protocol_audit.csv",
            "support_response_selections": output_dir / "support_nelbo_support_response_selections.csv",
            "raw_support_nelbo_rows": output_dir / "support_nelbo_raw_support_nelbo_rows.csv",
            "metadata_baseline_diagnostics": output_dir / "support_nelbo_metadata_baseline_diagnostics.csv",
            "support_size_monotonicity": output_dir / "support_nelbo_support_size_monotonicity.csv",
            "margin_diagnostics": output_dir / "support_nelbo_margin_diagnostics.csv",
            "selection_entropy": output_dir / "support_nelbo_selection_entropy.csv",
        }
    else:
        outputs = {
            "report": output_dir / f"{OUTPUT_PREFIX}.md",
            "summary_csv": output_dir / f"{OUTPUT_PREFIX}_decision_table.csv",
            "summary_json": output_dir / f"{OUTPUT_PREFIX}_decision_summary.json",
            "per_k_metrics": output_dir / f"{OUTPUT_PREFIX}_per_k_metrics.csv",
            "alpha_distribution": output_dir / f"{OUTPUT_PREFIX}_alpha_distribution.csv",
            "per_center_gap": output_dir / f"{OUTPUT_PREFIX}_per_center_gap.csv",
            "per_magnification_decisions": output_dir / f"{OUTPUT_PREFIX}_per_magnification_decisions.csv",
            "rank_consistency": output_dir / f"{OUTPUT_PREFIX}_rank_consistency.csv",
            "stability_by_k": output_dir / f"{OUTPUT_PREFIX}_stability_by_k.csv",
            "seed_stability": output_dir / f"{OUTPUT_PREFIX}_seed_stability.csv",
            "protocol_audit": output_dir / f"{OUTPUT_PREFIX}_protocol_audit.csv",
            "support_response_selections": output_dir / f"{OUTPUT_PREFIX}_support_response_selections.csv",
            "raw_support_nelbo_rows": output_dir / f"{OUTPUT_PREFIX}_raw_support_nelbo_rows.csv",
            "metadata_baseline_diagnostics": output_dir / f"{OUTPUT_PREFIX}_metadata_baseline_diagnostics.csv",
            "support_size_monotonicity": output_dir / f"{OUTPUT_PREFIX}_support_size_monotonicity.csv",
            "margin_diagnostics": output_dir / f"{OUTPUT_PREFIX}_margin_diagnostics.csv",
            "selection_entropy": output_dir / f"{OUTPUT_PREFIX}_selection_entropy.csv",
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
        outputs["per_magnification_decisions"],
        per_magnification_rows,
        [
            "heldout_magnification",
            "support_size",
            "support_seed",
            "seed",
            "selected_expert",
            "oracle_expert",
            "support_nelbo_selected",
            "eval_nelbo_selected",
            "eval_nelbo_oracle",
            "oracle_gap_pct",
            "metadata_selected_expert",
            "static_embedding_selected_expert",
        ],
    )
    _write_csv(
        outputs["rank_consistency"],
        rank_consistency_rows,
        [
            "heldout_magnification",
            "mean_spearman",
            "median_spearman",
            "top1",
            "oracle_gap_pct",
            "best_k",
            "worst_k",
        ],
    )
    _write_csv(
        outputs["seed_stability"],
        seed_stability_rows,
        ["seed", "method", "top1_oracle_hit_mean", "spearman_mean", "oracle_gap_pct_mean"],
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
            "patient_disjoint_required",
            "support_eval_patient_disjoint_ok",
            "missing_patient_id_count",
            "support_label_counts_json",
            "support_labels_unused_for_routing_ok",
            "target_expert_excluded_ok",
            "candidate_pool_excludes_target_expert_ok",
            "selected_expert_in_candidate_pool_ok",
            "candidate_oracle_in_candidate_pool_ok",
            "routing_uses_eval_nelbo_ok",
            "routing_uses_eval_domain_statistics_ok",
            "routing_time_scores_exclude_heldout_expert_ok",
            "heldout_expert_checkpoint_used_only_for_oracle_diagnostic",
            "alpha_selection_applicable",
            "alpha_selected_before_target_eval_scoring_ok",
            "alpha_hyperparam_selected_before_target_eval_scoring_ok",
            "protocol_version",
            "source_selection_path",
            "source_split_path",
            "source_alpha_path",
        ],
    )
    _write_dynamic_csv(outputs["support_response_selections"], consolidated_sample_rows)
    _write_dynamic_csv(outputs["raw_support_nelbo_rows"], consolidated_raw_support_rows)
    _write_dynamic_csv(outputs["metadata_baseline_diagnostics"], consolidated_metadata_diag_rows)
    _write_dynamic_csv(outputs["support_size_monotonicity"], support_size_monotonicity_rows)
    _write_dynamic_csv(outputs["margin_diagnostics"], margin_diagnostic_rows)
    _write_dynamic_csv(outputs["selection_entropy"], selection_entropy_rows)

    is_breakhis = context == "breakhis"
    is_midogpp = context == "midogpp_scanner"
    if is_breakhis:
        allowed_claim = (
            "Direct support-set NELBO was stress-tested as a target-local utility estimator "
            "under BreakHis leave-one-magnification-out routing, using an unlabeled "
            "patient-disjoint target support set and held-out NELBO utility evaluation."
        )
        disallowed_claims = [
            "This experiment does not prove general support-NELBO robustness across all medical domain shifts.",
            "BreakHis magnification shift is equivalent to Camelyon17 hospital or site shift.",
            "A BreakHis failure invalidates the Camelyon17 support-NELBO result.",
        ]
    elif is_midogpp:
        allowed_claim = (
            "Direct support-set NELBO is stress-tested as a compatibility estimator under "
            "scanner-indexed MIDOG++ acquisition-domain shift, using group-disjoint "
            "unlabeled support/evaluation splits and held-out NELBO utility evaluation."
        )
        disallowed_claims = [
            "Direct support-NELBO solves pure scanner shift.",
            "Scanner identity is true compatibility rather than an acquisition-domain proxy.",
            "A result is thesis-facing without the MIDOG++ preflight scanner/confounding gates.",
            "Tumor, lab, species, or resolution confounding is absent unless the confounding table proves it.",
        ]
    else:
        allowed_claim = (
            "Direct support-set NELBO is the primary support-estimated utility router "
            "and the strongest support-estimated utility variant in the current "
            "Camelyon17 support experiment."
        )
        disallowed_claims = [
            "Conservative support NELBO improves small-k stability.",
            "Alpha regularization is meaningful when alpha collapses to zero in most selections.",
            "Direct support-set NELBO supports a broader overall router-ranking claim.",
        ]

    summary_payload = {
        "artifact_lineage": LINEAGE_NOTE,
        "decision_layers": {
            "protocol_validity": audit_summary,
            "utility_performance": {
                "summary_rows": summary_rows,
                "metric_priority": (
                    [
                        "mean_oracle_gap_pct",
                        "spearman",
                        "top1_oracle_hit",
                        "margin_and_entropy_diagnostics",
                    ]
                    if is_midogpp
                    else [
                        "top1_oracle_hit",
                        "spearman",
                        "oracle_gap_pct",
                        "selected_heldout_eval_nelbo",
                    ]
                ),
            },
            "stability_diagnostics": {
                "rows": stability_rows,
                "rank_consistency_rows": rank_consistency_rows,
                "per_magnification_decision_rows": per_magnification_rows,
                "seed_stability_rows": seed_stability_rows,
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
            "raw_support_nelbo_rows_expected": EXPECTED_RAW_SUPPORT_ROWS,
            "raw_support_nelbo_rows_observed": raw_support_rows_observed,
            "note": "Selected counts are selected-method rows; raw support rows are candidate-by-support-image rows.",
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
        "pass_rule": pass_rule,
        "cross_check_against_earlier_decision_table": cross_check,
        "allowed_thesis_claim": allowed_claim,
        "disallowed_thesis_claims": disallowed_claims,
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
        pass_rule=pass_rule,
        spearman_warnings=stability_extra["warnings"],
    )

    return {key: str(value) for key, value in outputs.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build support-NELBO thesis consolidation artifacts.")
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--decision-table", type=Path, default=EARLIER_DECISION_TABLE)
    parser.add_argument("--run-seeds", default="42,43,44")
    parser.add_argument("--heldout-domains", default="0,1,2,3,4")
    parser.add_argument("--support-seeds", default="17,23,31")
    parser.add_argument("--support-sizes", default="4,8,16,32")
    parser.add_argument("--run-id-template", default=RUN_ID_TEMPLATE)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--dataset-context", default=DATASET_CONTEXT)
    return parser.parse_args()


def _parse_int_tuple(value: str) -> Tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def main() -> None:
    args = parse_args()
    global RUN_SEEDS, HELDOUT_CENTERS, SUPPORT_SEEDS, SUPPORT_SIZES
    global RUN_ID_TEMPLATE, OUTPUT_PREFIX, DATASET_CONTEXT
    global EXPECTED_SELECTED_ROWS, EXPECTED_ALPHA_ROWS, EXPECTED_RAW_SUPPORT_ROWS
    RUN_SEEDS = _parse_int_tuple(args.run_seeds)
    HELDOUT_CENTERS = _parse_int_tuple(args.heldout_domains)
    SUPPORT_SEEDS = _parse_int_tuple(args.support_seeds)
    SUPPORT_SIZES = _parse_int_tuple(args.support_sizes)
    RUN_ID_TEMPLATE = str(args.run_id_template)
    OUTPUT_PREFIX = str(args.output_prefix)
    DATASET_CONTEXT = str(args.dataset_context)
    EXPECTED_SELECTED_ROWS = (
        len(RUN_SEEDS)
        * len(HELDOUT_CENTERS)
        * len(SUPPORT_SEEDS)
        * len(SUPPORT_SIZES)
        * 2
    )
    EXPECTED_ALPHA_ROWS = len(RUN_SEEDS) * len(HELDOUT_CENTERS) * len(SUPPORT_SIZES)
    EXPECTED_RAW_SUPPORT_ROWS = (
        len(RUN_SEEDS)
        * len(HELDOUT_CENTERS)
        * len(SUPPORT_SEEDS)
        * sum(SUPPORT_SIZES)
        * max(len(HELDOUT_CENTERS) - 1, 0)
    )
    outputs = build_outputs(
        experiment_root=args.experiment_root,
        output_dir=args.output_dir,
        decision_table=args.decision_table,
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
