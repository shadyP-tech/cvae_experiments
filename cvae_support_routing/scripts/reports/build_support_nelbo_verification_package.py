#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


SUPPORT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SUPPORT_ROOT.parent
PROJECT_ROOT = REPO_ROOT / "cvae_testing"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import spearman_corr


EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "camelyon17"
    / "camelyon17_support_estimated_utility_routing_v2"
)
OUTPUT_DIR = SUPPORT_ROOT / "artifacts" / "comparison_tables"
PROTOCOL_AUDIT = OUTPUT_DIR / "support_nelbo_protocol_audit.csv"

DIRECT_METHOD = "support_set_nelbo_top1"
CONSERVATIVE_METHOD = "support_set_nelbo_conservative"
METADATA_METHOD = "support_metadata_routing"
STATIC_METHOD = "support_static_embedding_routing"
SOURCE_GLOBAL_METHOD = "source_global_prior_routing"

METHOD_LABELS = {
    DIRECT_METHOD: "direct_support_nelbo",
    CONSERVATIVE_METHOD: "conservative_support_nelbo",
    METADATA_METHOD: "metadata_routing",
    STATIC_METHOD: "static_embedding_routing",
    SOURCE_GLOBAL_METHOD: "source_global_prior_routing",
}
THESIS_FACING_METHODS = (
    METADATA_METHOD,
    STATIC_METHOD,
    SOURCE_GLOBAL_METHOD,
    DIRECT_METHOD,
    CONSERVATIVE_METHOD,
)
BASELINE_METHODS = (METADATA_METHOD, STATIC_METHOD, SOURCE_GLOBAL_METHOD)
SUPPORT_SIZES = (4, 8, 16, 32)
BOOTSTRAP_REPS_DEFAULT = 10000
BOOTSTRAP_SEED_DEFAULT = 1337
STRONG_STATIC_HIGH_REGRET_CI_GATE = 0.433

HIGH_REGRET_THRESHOLDS = (1.0, 2.0, 5.0)
PRIMARY_HIGH_REGRET_THRESHOLD = 2.0
NEAR_MATCH_TOP1_TOL = 0.05
NEAR_MATCH_GAP_TOL = 0.25
EXPERT_DOMINANCE_SHARE = 0.70
LOW_OVERALL_HIGH_REGRET_RATE = 0.10
LIMITED_HIGH_REGRET_RATE = 0.20
SEVERE_TOP1_DEGRADATION = 0.10
SEVERE_GAP_DEGRADATION = 0.50
SEVERE_HIGH_REGRET_DEGRADATION = 0.10


DecisionKey = Tuple[int, int, int, int]
BootstrapKey = Tuple[int, int, int, int, str]


def _read_csv(path: Path, *, required: bool = True) -> List[dict]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required artifact: {path}")
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return 0.0
    mu = _mean(vals)
    return float((sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5)


def _rate(rows: Sequence[Mapping[str, Any]], predicate: Any) -> float:
    return float(sum(1 for row in rows if predicate(row)) / len(rows)) if rows else 0.0


def _quantile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def _parse_candidate_experts(value: object) -> List[int]:
    out: List[int] = []
    for part in str(value or "").split("|"):
        part = part.strip()
        if part:
            out.append(int(float(part)))
    return out


def _json_float_map(value: object) -> Dict[int, float]:
    if isinstance(value, Mapping):
        raw = value
    else:
        try:
            raw = json.loads(str(value or "{}"))
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, Mapping):
        return {}
    out: Dict[int, float] = {}
    for key, val in raw.items():
        try:
            expert = int(float(key))
            score = float(val)
        except Exception:
            continue
        if math.isfinite(score):
            out[expert] = score
    return out


def _rank_map(score_by_expert: Mapping[int, float]) -> Dict[int, int]:
    ordered = sorted(score_by_expert, key=lambda expert: (float(score_by_expert[expert]), int(expert)))
    return {int(expert): idx for idx, expert in enumerate(ordered, start=1)}


def _margin(score_by_expert: Mapping[int, float]) -> float:
    ordered = sorted((float(score), int(expert)) for expert, score in score_by_expert.items())
    if len(ordered) < 2:
        return 0.0
    return float(ordered[1][0] - ordered[0][0])


def _zscore(value: float, values: Sequence[float]) -> float:
    sd = _std(values)
    if sd <= 1.0e-12:
        return 0.0
    return float((float(value) - _mean(values)) / sd)


def _run_seed_from_run_dir(run_dir: Path) -> int:
    match = re.search(r"seed(\d+)", run_dir.name)
    if not match:
        raise ValueError(f"Cannot infer seed from run directory: {run_dir}")
    return int(match.group(1))


def _decision_key(row: Mapping[str, Any]) -> DecisionKey:
    return (
        _to_int(row.get("run_seed", row.get("seed", 0))),
        _to_int(row.get("query_domain", row.get("fold_query_domain", row.get("target_domain", 0)))),
        _to_int(row.get("support_seed", 0)),
        _to_int(row.get("support_size_requested", row.get("support_size", 0))),
    )


def _bootstrap_key_from_decision(row: Mapping[str, Any]) -> BootstrapKey:
    return (
        _to_int(row.get("run_seed", row.get("seed", 0))),
        _to_int(row.get("query_domain", row.get("fold_query_domain", row.get("target_domain", 0)))),
        _to_int(row.get("support_seed", 0)),
        _to_int(row.get("support_size_requested", row.get("support_size", 0))),
        str(row.get("support_eval_split_id", row.get("split_id", ""))),
    )


def _bootstrap_key_from_raw(row: Mapping[str, Any]) -> BootstrapKey:
    return (
        _to_int(row.get("experiment_seed", row.get("run_seed", row.get("seed", 0)))),
        _to_int(row.get("heldout_center", row.get("query_domain", row.get("target_domain", 0)))),
        _to_int(row.get("support_seed", 0)),
        _to_int(row.get("support_size", row.get("support_size_requested", 0))),
        str(row.get("split_id", row.get("support_eval_split_id", ""))),
    )


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


def _result_paths(experiment_root: Path) -> List[Path]:
    paths = sorted(path for path in experiment_root.glob("support_utility_v2_seed*") if path.is_dir())
    if not paths:
        raise FileNotFoundError(f"No support utility run directories found under {experiment_root}")
    for run_dir in paths:
        reports = run_dir / "reports"
        required = [
            reports / "support_response_sample_selections.csv",
            reports / "support_response_split_manifest.csv",
            reports / "support_response_protocol_lock.json",
            reports / "leakage_report.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required support-NELBO artifacts:\n" + "\n".join(missing))
    return paths


def load_source_artifacts(
    experiment_root: Path,
    protocol_audit: Path,
) -> Tuple[List[dict], List[dict], List[dict], List[dict], List[dict]]:
    sample_rows: List[dict] = []
    split_rows: List[dict] = []
    support_raw_rows: List[dict] = []
    protocol_locks: List[dict] = []
    for run_dir in _result_paths(experiment_root):
        run_seed = _run_seed_from_run_dir(run_dir)
        reports = run_dir / "reports"
        sample_path = reports / "support_response_sample_selections.csv"
        for row in _read_csv(sample_path):
            row = dict(row)
            row["run_seed"] = run_seed
            row["run_id"] = run_dir.name
            row["source_path"] = str(sample_path)
            sample_rows.append(row)

        split_path = reports / "support_response_split_manifest.csv"
        for row in _read_csv(split_path):
            row = dict(row)
            row["run_seed"] = run_seed
            row["run_id"] = run_dir.name
            row["source_path"] = str(split_path)
            split_rows.append(row)

        raw_path = reports / "support_response_support_nelbo_rows.csv"
        for row in _read_csv(raw_path, required=False):
            row = dict(row)
            row["run_seed"] = run_seed
            row["run_id"] = row.get("run_id") or run_dir.name
            row["source_path"] = str(raw_path)
            support_raw_rows.append(row)

        lock_path = reports / "support_response_protocol_lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock["run_seed"] = run_seed
        lock["run_id"] = run_dir.name
        lock["source_path"] = str(lock_path)
        protocol_locks.append(lock)

    audit_rows = _read_csv(protocol_audit, required=False)
    return sample_rows, split_rows, support_raw_rows, protocol_locks, audit_rows


def _method_rows(sample_rows: Sequence[Mapping[str, Any]], method: str) -> List[dict]:
    return [dict(row) for row in sample_rows if str(row.get("method", "")) == method]


def decision_diagnostic(row: Mapping[str, Any]) -> dict:
    candidates = _parse_candidate_experts(row.get("candidate_experts", ""))
    support_map = _json_float_map(row.get("support_nelbo_by_expert_json", "{}"))
    eval_map = _json_float_map(row.get("eval_nelbo_by_expert_json", "{}"))
    predicted_map = _json_float_map(row.get("predicted_score_by_expert_json", "{}"))
    if not candidates:
        candidates = sorted(set(support_map) | set(eval_map) | set(predicted_map))

    support_map = {expert: support_map[expert] for expert in candidates if expert in support_map}
    eval_map = {expert: eval_map[expert] for expert in candidates if expert in eval_map}
    predicted_map = {expert: predicted_map[expert] for expert in candidates if expert in predicted_map}
    support_ranks = _rank_map(support_map)
    eval_ranks = _rank_map(eval_map)
    selected = _to_int(row.get("selected_expert", -999999))
    oracle = _to_int(row.get("candidate_oracle_expert", row.get("oracle_expert", -999999)))
    selected_eval = eval_map.get(selected, _to_float(row.get("selected_nelbo", 0.0)))
    oracle_eval = eval_map.get(oracle, _to_float(row.get("candidate_oracle_nelbo", row.get("oracle_nelbo", 0.0))))
    selected_support = support_map.get(selected, _to_float(row.get("mean_support_nelbo", 0.0)))
    oracle_support = support_map.get(oracle, float("nan"))
    top1 = int(selected == oracle or _to_int(row.get("top1_oracle_hit", 0)) == 1)
    gap_pct = _to_float(row.get("mean_oracle_gap_pct", row.get("oracle_gap_pct", 0.0)))

    return {
        "method": METHOD_LABELS.get(str(row.get("method", "")), str(row.get("method", ""))),
        "source_method": str(row.get("method", "")),
        "run_seed": _to_int(row.get("run_seed", row.get("seed", 0))),
        "run_id": row.get("run_id", ""),
        "heldout_center": _to_int(row.get("query_domain", row.get("fold_query_domain", row.get("target_domain", 0)))),
        "support_seed": _to_int(row.get("support_seed", 0)),
        "k": _to_int(row.get("support_size_requested", row.get("support_size", 0))),
        "support_eval_split_id": row.get("support_eval_split_id", ""),
        "candidate_experts": "|".join(str(expert) for expert in candidates),
        "n_candidate_experts": len(candidates),
        "selected_expert": selected,
        "oracle_expert": oracle,
        "selected_eval_nelbo": float(selected_eval),
        "oracle_eval_nelbo": float(oracle_eval),
        "selected_support_nelbo": float(selected_support),
        "oracle_support_nelbo": float(oracle_support) if math.isfinite(float(oracle_support)) else "",
        "oracle_gap_pct": float(gap_pct),
        "top1_oracle_hit": top1,
        "spearman": _to_float(row.get("spearman", 0.0)),
        "pairwise_auc": _to_float(row.get("pairwise_auc", 0.0)),
        "selected_rank": _to_float(row.get("selected_rank", 0.0)),
        "support_margin": _margin(support_map),
        "eval_margin": float(selected_eval) - float(oracle_eval),
        "support_rank_of_eval_oracle": support_ranks.get(oracle, 0),
        "eval_rank_of_support_selected": eval_ranks.get(selected, 0),
        "support_map": support_map,
        "eval_map": eval_map,
        "predicted_map": predicted_map,
        "support_ranks": support_ranks,
        "eval_ranks": eval_ranks,
        "source_path": row.get("source_path", ""),
    }


def regret_class(*, top1_oracle_hit: int, oracle_gap_pct: float) -> str:
    if int(top1_oracle_hit) == 1:
        return "top1_success"
    if float(oracle_gap_pct) <= 1.0:
        return "near_miss"
    if float(oracle_gap_pct) <= 2.0:
        return "moderate_regret"
    if float(oracle_gap_pct) <= 5.0:
        return "high_regret"
    return "catastrophic"


def support_confidence_class(
    *,
    top1_oracle_hit: int,
    oracle_gap_pct: float,
    support_margin: float,
    q1_margin: float,
    q3_margin: float,
) -> str:
    if (
        int(top1_oracle_hit) == 0
        and float(oracle_gap_pct) > PRIMARY_HIGH_REGRET_THRESHOLD
        and float(support_margin) >= float(q3_margin)
    ):
        return "wrong_confident"
    if float(support_margin) <= float(q1_margin):
        return "ambiguous_support"
    return "normal_margin"


def annotate_decisions(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    diagnostics = [decision_diagnostic(row) for row in rows]
    margins_by_k: Dict[int, List[float]] = defaultdict(list)
    for row in diagnostics:
        margins_by_k[int(row["k"])].append(float(row["support_margin"]))
    thresholds = {
        k: {
            "q1": _quantile(vals, 0.25),
            "q2": _quantile(vals, 0.50),
            "q3": _quantile(vals, 0.75),
        }
        for k, vals in margins_by_k.items()
    }
    for row in diagnostics:
        k = int(row["k"])
        row["support_margin_q1_for_k"] = thresholds[k]["q1"]
        row["support_margin_q3_for_k"] = thresholds[k]["q3"]
        row["regret_class"] = regret_class(
            top1_oracle_hit=_to_int(row["top1_oracle_hit"]),
            oracle_gap_pct=_to_float(row["oracle_gap_pct"]),
        )
        row["support_confidence_class"] = support_confidence_class(
            top1_oracle_hit=_to_int(row["top1_oracle_hit"]),
            oracle_gap_pct=_to_float(row["oracle_gap_pct"]),
            support_margin=_to_float(row["support_margin"]),
            q1_margin=thresholds[k]["q1"],
            q3_margin=thresholds[k]["q3"],
        )
    return diagnostics


def flatten_candidate_rows(decisions: Sequence[Mapping[str, Any]]) -> List[dict]:
    out: List[dict] = []
    for row in decisions:
        support_map = dict(row.get("support_map", {}))
        eval_map = dict(row.get("eval_map", {}))
        support_vals = list(support_map.values())
        eval_vals = list(eval_map.values())
        support_ranks = dict(row.get("support_ranks", {}))
        eval_ranks = dict(row.get("eval_ranks", {}))
        selected = _to_int(row.get("selected_expert", -999999))
        oracle = _to_int(row.get("oracle_expert", -999999))
        for expert in sorted(set(support_map) | set(eval_map)):
            out.append(
                {
                    "method": row.get("method", ""),
                    "source_method": row.get("source_method", ""),
                    "run_seed": row.get("run_seed", ""),
                    "heldout_center": row.get("heldout_center", ""),
                    "support_seed": row.get("support_seed", ""),
                    "k": row.get("k", ""),
                    "support_eval_split_id": row.get("support_eval_split_id", ""),
                    "candidate_expert": expert,
                    "support_nelbo": support_map.get(expert, ""),
                    "eval_nelbo": eval_map.get(expert, ""),
                    "support_rank": support_ranks.get(expert, ""),
                    "eval_rank": eval_ranks.get(expert, ""),
                    "is_selected": int(expert == selected),
                    "is_eval_oracle": int(expert == oracle),
                    "support_margin": row.get("support_margin", ""),
                    "eval_margin": row.get("eval_margin", ""),
                    "support_rank_of_eval_oracle": row.get("support_rank_of_eval_oracle", ""),
                    "eval_rank_of_support_selected": row.get("eval_rank_of_support_selected", ""),
                    "support_z_within_decision": (
                        _zscore(float(support_map[expert]), support_vals) if expert in support_map else ""
                    ),
                    "eval_z_within_decision": (
                        _zscore(float(eval_map[expert]), eval_vals) if expert in eval_map else ""
                    ),
                }
            )
    return out


def build_expected_count_assertions(
    *,
    direct_rows: Sequence[Mapping[str, Any]],
    direct_candidate_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
    support_raw_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict:
    expected_keys = {
        _decision_key(row)
        for row in split_rows
        if str(row.get("split_role", "")) == "target"
    }
    observed_keys = [_decision_key(row) for row in direct_rows]
    observed_counter = Counter(observed_keys)
    missing = sorted(expected_keys - set(observed_keys))
    extra = sorted(set(observed_keys) - expected_keys)
    duplicates = sorted(key for key, count in observed_counter.items() if count > 1)
    candidate_mismatches = []
    for row in direct_rows:
        candidates = _parse_candidate_experts(row.get("candidate_experts", ""))
        support_map = _json_float_map(row.get("support_nelbo_by_expert_json", "{}"))
        eval_map = _json_float_map(row.get("eval_nelbo_by_expert_json", "{}"))
        if set(candidates) != set(support_map) or set(candidates) != set(eval_map):
            candidate_mismatches.append(_decision_key(row))

    expected_candidates = sum(len(_parse_candidate_experts(row.get("candidate_experts", ""))) for row in direct_rows)
    status = "pass"
    if missing or extra or duplicates or candidate_mismatches:
        status = "fail"
    if len(direct_rows) != len(expected_keys) or len(direct_candidate_rows) != expected_candidates:
        status = "fail"
    raw_rows = list(support_raw_rows) if support_raw_rows is not None else []
    expected_raw_rows = sum(
        _to_int(row.get("support_size_requested", row.get("support_size", 0)))
        * len(_parse_candidate_experts(row.get("candidate_experts", "")))
        for row in direct_rows
    )
    raw_by_key: Dict[BootstrapKey, List[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        raw_by_key[_bootstrap_key_from_raw(row)].append(row)
    raw_missing: List[BootstrapKey] = []
    raw_extra = sorted(set(raw_by_key) - {_bootstrap_key_from_decision(row) for row in direct_rows})
    raw_group_failures: List[str] = []
    for row in direct_rows:
        key = _bootstrap_key_from_decision(row)
        group = raw_by_key.get(key, [])
        candidates = _parse_candidate_experts(row.get("candidate_experts", ""))
        k = _to_int(row.get("support_size_requested", row.get("support_size", 0)))
        if not group:
            raw_missing.append(key)
            continue
        support_positions = sorted({_to_int(raw.get("support_pos_anon", -1), -1) for raw in group})
        if len(support_positions) != k:
            raw_group_failures.append(f"{key}: unique support_pos_anon {len(support_positions)} != {k}")
        if len(group) != k * len(candidates):
            raw_group_failures.append(f"{key}: raw row count {len(group)} != {k * len(candidates)}")
        expected_candidate_set = set(candidates)
        for pos in support_positions:
            pos_rows = [raw for raw in group if _to_int(raw.get("support_pos_anon", -1), -1) == pos]
            pos_candidates = {_to_int(raw.get("candidate_expert", -999999)) for raw in pos_rows}
            if len(pos_rows) != len(candidates) or pos_candidates != expected_candidate_set:
                raw_group_failures.append(f"{key}: support_pos_anon {pos} does not cover all candidates once")

    if support_raw_rows is not None and (
        len(raw_rows) != expected_raw_rows or raw_missing or raw_extra or raw_group_failures
    ):
        status = "fail"
    return {
        "status": status,
        "expected_decisions": len(expected_keys),
        "observed_decisions": len(direct_rows),
        "missing_decision_keys": ["|".join(map(str, key)) for key in missing],
        "extra_decision_keys": ["|".join(map(str, key)) for key in extra],
        "duplicate_decision_keys": ["|".join(map(str, key)) for key in duplicates],
        "expected_candidates": expected_candidates,
        "observed_candidates": len(direct_candidate_rows),
        "candidate_json_mismatch_count": len(candidate_mismatches),
        "expected_support_raw_rows": expected_raw_rows,
        "observed_support_raw_rows": len(raw_rows),
        "missing_support_raw_keys": ["|".join(map(str, key)) for key in raw_missing],
        "extra_support_raw_keys": ["|".join(map(str, key)) for key in raw_extra],
        "support_raw_group_failure_count": len(raw_group_failures),
        "support_raw_group_failures": raw_group_failures,
    }


def build_protocol_gate(
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
    protocol_locks: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
) -> dict:
    failures: List[str] = []
    required_audit_checks = [
        "split_row_found",
        "support_eval_disjoint_ok",
        "support_labels_unused_for_routing_ok",
        "target_expert_excluded_ok",
        "candidate_pool_excludes_target_expert_ok",
        "selected_expert_in_candidate_pool_ok",
        "candidate_oracle_in_candidate_pool_ok",
        "routing_uses_eval_nelbo_ok",
        "routing_uses_eval_domain_statistics_ok",
    ]
    if not audit_rows:
        failures.append("missing support_nelbo_protocol_audit.csv rows")
    for idx, row in enumerate(audit_rows):
        for check in required_audit_checks:
            if _to_int(row.get(check, 0)) != 1:
                failures.append(f"audit row {idx} failed {check}")

    split_by_key = {
        _decision_key(row): row
        for row in split_rows
        if str(row.get("split_role", "")) == "target"
    }
    for row in sample_rows:
        method = str(row.get("method", ""))
        if method not in THESIS_FACING_METHODS:
            continue
        key = _decision_key(row)
        candidates = _parse_candidate_experts(row.get("candidate_experts", ""))
        target = _to_int(row.get("fold_query_domain", row.get("target_domain", row.get("query_domain", 0))))
        selected = _to_int(row.get("selected_expert", -999999))
        oracle = _to_int(row.get("candidate_oracle_expert", -999999))
        prefix = f"{METHOD_LABELS.get(method, method)} {key}"
        if _to_int(row.get("routing_uses_eval_nelbo", 1)) != 0:
            failures.append(f"{prefix} uses eval NELBO for routing")
        if _to_int(row.get("routing_uses_eval_domain_statistics", 1)) != 0:
            failures.append(f"{prefix} uses eval domain statistics")
        if _to_int(row.get("target_expert_excluded", 0)) != 1:
            failures.append(f"{prefix} does not mark target expert excluded")
        if target in candidates:
            failures.append(f"{prefix} candidate pool includes target expert")
        if selected not in candidates:
            failures.append(f"{prefix} selected expert not in candidate pool")
        if oracle not in candidates:
            failures.append(f"{prefix} oracle expert not in candidate pool")
        split = split_by_key.get(key)
        if split is None:
            failures.append(f"{prefix} missing target split row")
        else:
            if str(split.get("split_status", "")) != "ok":
                failures.append(f"{prefix} split status is {split.get('split_status', '')}")
            if _to_int(split.get("support_eval_disjoint", 0)) != 1:
                failures.append(f"{prefix} support/eval split is not disjoint")
            if _to_int(split.get("support_labels_used", 1)) != 0:
                failures.append(f"{prefix} support labels are used")

    for lock in protocol_locks:
        if str(lock.get("protocol_version", "")) != "support_response_candidate_specific_v1":
            failures.append(f"{lock.get('run_id', '')} has unexpected protocol version")
        expected_lock_fields = {
            "support_raw_rows_exported": 1,
            "support_raw_rows_contains_eval_nelbo": 0,
            "support_raw_rows_contains_identity_fields": 0,
            "support_bootstrap_posthoc_only": 1,
        }
        for field, expected in expected_lock_fields.items():
            if _to_int(lock.get(field, 0 if expected else 1)) != int(expected):
                failures.append(f"{lock.get('run_id', '')} has invalid {field}")
        if _to_int(lock.get("bootstrap_reps", 0)) != BOOTSTRAP_REPS_DEFAULT:
            failures.append(f"{lock.get('run_id', '')} has unexpected bootstrap_reps")
        if _to_int(lock.get("bootstrap_seed", 0)) != BOOTSTRAP_SEED_DEFAULT:
            failures.append(f"{lock.get('run_id', '')} has unexpected bootstrap_seed")
        if str(lock.get("conservative_alpha_selection", "")) != "source_inner_fixed":
            failures.append(f"{lock.get('run_id', '')} has invalid conservative_alpha_selection")
        support_cfg = lock.get("support_estimated_utility", {})
        if isinstance(support_cfg, Mapping):
            if _to_int(support_cfg.get("selected_before_target_eval_scoring", 0)) != 1:
                failures.append(f"{lock.get('run_id', '')} support alpha was not preselected")
            if _to_int(support_cfg.get("support_labels_used_for_routing", 1)) != 0:
                failures.append(f"{lock.get('run_id', '')} support labels are allowed for routing")

    return {
        "status": "pass" if not failures else "fail",
        "failure_count": len(failures),
        "failures": failures,
        "audit_rows": len(audit_rows),
        "protocol_locks": len(protocol_locks),
    }


def summarize_methods(sample_rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    out: List[dict] = []
    for method in THESIS_FACING_METHODS:
        rows = _method_rows(sample_rows, method)
        if not rows:
            continue
        out.append(
            {
                "method": METHOD_LABELS[method],
                "source_method": method,
                "n_decisions": len(rows),
                "top1": _mean(_to_float(row.get("top1_oracle_hit", 0.0)) for row in rows),
                "spearman": _mean(_to_float(row.get("spearman", 0.0)) for row in rows),
                "oracle_gap_pct": _mean(_to_float(row.get("mean_oracle_gap_pct", 0.0)) for row in rows),
                "selected_eval_nelbo": _mean(_to_float(row.get("selected_nelbo", 0.0)) for row in rows),
                "oracle_eval_nelbo": _mean(_to_float(row.get("candidate_oracle_nelbo", row.get("oracle_nelbo", 0.0))) for row in rows),
                "high_regret_rate_gt2": _rate(
                    rows,
                    lambda row: _to_float(row.get("mean_oracle_gap_pct", 0.0)) > PRIMARY_HIGH_REGRET_THRESHOLD,
                ),
                "catastrophic_rate_gt5": _rate(
                    rows,
                    lambda row: _to_float(row.get("mean_oracle_gap_pct", 0.0)) > 5.0,
                ),
            }
        )
    return out


def build_per_center_per_k(decisions: Sequence[Mapping[str, Any]]) -> List[dict]:
    groups: Dict[Tuple[int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in decisions:
        groups[(_to_int(row.get("heldout_center", 0)), _to_int(row.get("k", 0)))].append(row)
    out: List[dict] = []
    for (center, k), rows in sorted(groups.items()):
        out.append(
            {
                "heldout_center": center,
                "k": k,
                "n_decisions": len(rows),
                "top1": _mean(_to_float(row.get("top1_oracle_hit", 0.0)) for row in rows),
                "spearman": _mean(_to_float(row.get("spearman", 0.0)) for row in rows),
                "oracle_gap_pct": _mean(_to_float(row.get("oracle_gap_pct", 0.0)) for row in rows),
                "high_regret_rate_gt1": _rate(rows, lambda row: _to_float(row.get("oracle_gap_pct", 0.0)) > 1.0),
                "high_regret_rate_gt2": _rate(rows, lambda row: _to_float(row.get("oracle_gap_pct", 0.0)) > 2.0),
                "high_regret_rate_gt5": _rate(rows, lambda row: _to_float(row.get("oracle_gap_pct", 0.0)) > 5.0),
                "selected_eval_nelbo": _mean(_to_float(row.get("selected_eval_nelbo", 0.0)) for row in rows),
                "oracle_eval_nelbo": _mean(_to_float(row.get("oracle_eval_nelbo", 0.0)) for row in rows),
                "p_eval_oracle_support_rank_le2": _rate(
                    rows,
                    lambda row: _to_int(row.get("support_rank_of_eval_oracle", 999)) <= 2,
                ),
                "p_selected_eval_rank_le2": _rate(
                    rows,
                    lambda row: _to_int(row.get("eval_rank_of_support_selected", 999)) <= 2,
                ),
            }
        )
    return out


def build_failure_cases(decisions: Sequence[Mapping[str, Any]]) -> List[dict]:
    out: List[dict] = []
    for row in decisions:
        if _to_int(row.get("top1_oracle_hit", 0)) == 1:
            continue
        out.append(
            {
                "run_seed": row.get("run_seed", ""),
                "heldout_center": row.get("heldout_center", ""),
                "k": row.get("k", ""),
                "support_seed": row.get("support_seed", ""),
                "selected_expert": row.get("selected_expert", ""),
                "oracle_expert": row.get("oracle_expert", ""),
                "support_nelbo_selected": row.get("selected_support_nelbo", ""),
                "support_nelbo_oracle": row.get("oracle_support_nelbo", ""),
                "eval_nelbo_selected": row.get("selected_eval_nelbo", ""),
                "eval_nelbo_oracle": row.get("oracle_eval_nelbo", ""),
                "oracle_gap_pct": row.get("oracle_gap_pct", ""),
                "support_margin": row.get("support_margin", ""),
                "eval_margin": row.get("eval_margin", ""),
                "support_rank_of_eval_oracle": row.get("support_rank_of_eval_oracle", ""),
                "eval_rank_of_support_selected": row.get("eval_rank_of_support_selected", ""),
                "regret_class": row.get("regret_class", ""),
                "support_confidence_class": row.get("support_confidence_class", ""),
                "support_margin_q1_for_k": row.get("support_margin_q1_for_k", ""),
                "support_margin_q3_for_k": row.get("support_margin_q3_for_k", ""),
                "source_path": row.get("source_path", ""),
            }
        )
    return sorted(out, key=lambda row: (-_to_float(row.get("oracle_gap_pct", 0.0)), row["heldout_center"], row["k"]))


def build_high_regret_distribution(sample_rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    rows = [dict(row) for row in sample_rows if str(row.get("method", "")) in THESIS_FACING_METHODS]
    groups: Dict[Tuple[str, str, object, object], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        method = METHOD_LABELS[str(row["method"])]
        center = _to_int(row.get("query_domain", row.get("fold_query_domain", 0)))
        k = _to_int(row.get("support_size_requested", 0))
        groups[("overall", method, "", "")].append(row)
        groups[("by_k", method, "", k)].append(row)
        groups[("by_center", method, center, "")].append(row)
        groups[("by_center_k", method, center, k)].append(row)

    out: List[dict] = []
    for (scope, method, center, k), vals in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        out.append(
            {
                "scope": scope,
                "method": method,
                "heldout_center": center,
                "k": k,
                "n_decisions": len(vals),
                "mean_oracle_gap_pct": _mean(_to_float(row.get("mean_oracle_gap_pct", 0.0)) for row in vals),
                "max_oracle_gap_pct": max((_to_float(row.get("mean_oracle_gap_pct", 0.0)) for row in vals), default=0.0),
                "high_regret_rate_gt1": _rate(vals, lambda row: _to_float(row.get("mean_oracle_gap_pct", 0.0)) > 1.0),
                "high_regret_rate_gt2": _rate(vals, lambda row: _to_float(row.get("mean_oracle_gap_pct", 0.0)) > 2.0),
                "high_regret_rate_gt5": _rate(vals, lambda row: _to_float(row.get("mean_oracle_gap_pct", 0.0)) > 5.0),
            }
        )
    return out


def _margin_bin(row: Mapping[str, Any], thresholds: Mapping[str, float]) -> str:
    margin = _to_float(row.get("support_margin", 0.0))
    if margin <= float(thresholds["q1"]):
        return "q1_low"
    if margin <= float(thresholds["q2"]):
        return "q2_mid_low"
    if margin <= float(thresholds["q3"]):
        return "q3_mid_high"
    return "q4_high"


def build_margin_reliability(decisions: Sequence[Mapping[str, Any]]) -> List[dict]:
    out: List[dict] = []
    for k in sorted({int(row["k"]) for row in decisions}):
        vals = [row for row in decisions if int(row["k"]) == k]
        thresholds = {
            "q1": _quantile([_to_float(row.get("support_margin", 0.0)) for row in vals], 0.25),
            "q2": _quantile([_to_float(row.get("support_margin", 0.0)) for row in vals], 0.50),
            "q3": _quantile([_to_float(row.get("support_margin", 0.0)) for row in vals], 0.75),
        }
        groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in vals:
            groups[_margin_bin(row, thresholds)].append(row)
        for bin_name in ["q1_low", "q2_mid_low", "q3_mid_high", "q4_high"]:
            rows = groups.get(bin_name, [])
            out.append(
                {
                    "k": k,
                    "margin_bin": bin_name,
                    "n_decisions": len(rows),
                    "support_margin_q1": thresholds["q1"],
                    "support_margin_q2": thresholds["q2"],
                    "support_margin_q3": thresholds["q3"],
                    "support_margin_min": min((_to_float(row.get("support_margin", 0.0)) for row in rows), default=0.0),
                    "support_margin_max": max((_to_float(row.get("support_margin", 0.0)) for row in rows), default=0.0),
                    "top1": _mean(_to_float(row.get("top1_oracle_hit", 0.0)) for row in rows),
                    "oracle_gap_pct": _mean(_to_float(row.get("oracle_gap_pct", 0.0)) for row in rows),
                    "high_regret_rate_gt2": _rate(rows, lambda row: _to_float(row.get("oracle_gap_pct", 0.0)) > 2.0),
                    "wrong_confident_rate": _rate(
                        rows,
                        lambda row: str(row.get("support_confidence_class", "")) == "wrong_confident",
                    ),
                }
            )
    return out


def _score_margin_lower(score_by_expert: Mapping[int, float]) -> float:
    ordered = sorted((float(score), int(expert)) for expert, score in score_by_expert.items())
    if len(ordered) < 2:
        return 0.0
    return float(ordered[1][0] - ordered[0][0])


def _support_stderr(values: Sequence[float], *, k: int) -> float:
    vals = [float(v) for v in values]
    if len(vals) <= 1:
        return 0.0
    return float(np.std(np.asarray(vals, dtype=np.float64), ddof=1) / math.sqrt(float(k)))


def _ci(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return _quantile(values, 0.025), _quantile(values, 0.975)


def _method_deterministic_scores(row: Mapping[str, Any], method: str) -> Dict[int, float]:
    support = _json_float_map(row.get("support_nelbo_by_expert_json", "{}"))
    if method == DIRECT_METHOD:
        return support
    stderr = _json_float_map(row.get("support_stderr_nelbo_by_expert_json", "{}"))
    alpha = _to_float(row.get("alpha", 0.0))
    return {
        int(expert): float(support[expert]) + float(alpha) * float(stderr.get(expert, 0.0))
        for expert in support
    }


def _select_bootstrap_expert(
    *,
    score_by_expert: Mapping[int, float],
    deterministic_score_by_expert: Mapping[int, float],
) -> int:
    return int(
        sorted(
            score_by_expert,
            key=lambda expert: (
                float(score_by_expert[int(expert)]),
                float(deterministic_score_by_expert.get(int(expert), 0.0)),
                int(expert),
            ),
        )[0]
    )


def _bootstrap_margin_bin(
    *,
    margin: float,
    thresholds: Mapping[str, float],
) -> str:
    if float(margin) <= float(thresholds["q1"]):
        return "q1_low"
    if float(margin) <= float(thresholds["q2"]):
        return "q2_mid_low"
    if float(margin) <= float(thresholds["q3"]):
        return "q3_mid_high"
    return "q4_high"


def _fixed_method_metric_by_k(
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    metric: str,
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for k in SUPPORT_SIZES:
        rows = [
            row for row in sample_rows
            if str(row.get("method", "")) == method
            and _to_int(row.get("support_size_requested", row.get("support_size", 0))) == int(k)
        ]
        if not rows:
            continue
        if metric == "top1":
            out[int(k)] = _mean(_to_float(row.get("top1_oracle_hit", 0.0)) for row in rows)
        elif metric == "gap":
            out[int(k)] = _mean(_to_float(row.get("mean_oracle_gap_pct", 0.0)) for row in rows)
        elif metric == "high_regret":
            out[int(k)] = _rate(rows, lambda row: _to_float(row.get("mean_oracle_gap_pct", 0.0)) > 2.0)
    return out


def build_support_bootstrap_artifacts(
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    support_raw_rows: Sequence[Mapping[str, Any]],
    bootstrap_reps: int = BOOTSTRAP_REPS_DEFAULT,
    bootstrap_seed: int = BOOTSTRAP_SEED_DEFAULT,
) -> Tuple[List[dict], List[dict], List[dict], dict]:
    methods = (DIRECT_METHOD, CONSERVATIVE_METHOD)
    method_rows = {
        method: {
            _bootstrap_key_from_decision(row): dict(row)
            for row in _method_rows(sample_rows, method)
        }
        for method in methods
    }
    raw_by_key: Dict[BootstrapKey, Dict[int, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in support_raw_rows:
        key = _bootstrap_key_from_raw(row)
        pos = _to_int(row.get("support_pos_anon", -1), -1)
        expert = _to_int(row.get("candidate_expert", -999999))
        raw_by_key[key][pos][expert] = _to_float(row.get("support_nelbo", 0.0))

    available_keys = sorted(set(method_rows[DIRECT_METHOD]) & set(method_rows[CONSERVATIVE_METHOD]) & set(raw_by_key))
    if not available_keys or int(bootstrap_reps) <= 0:
        return [], [], [], {
            "status": "skipped",
            "reason": "missing support raw rows or bootstrap_reps <= 0",
            "n_decision_groups": len(available_keys),
        }

    margin_thresholds: Dict[Tuple[str, int], Dict[str, float]] = {}
    deterministic_margin_by_method_key: Dict[Tuple[str, BootstrapKey], float] = {}
    for method in methods:
        by_k: Dict[int, List[float]] = defaultdict(list)
        for key, row in method_rows[method].items():
            scores = _method_deterministic_scores(row, method)
            margin = _score_margin_lower(scores)
            deterministic_margin_by_method_key[(method, key)] = margin
            by_k[int(key[3])].append(margin)
        for k, vals in by_k.items():
            margin_thresholds[(method, int(k))] = {
                "q1": _quantile(vals, 0.25),
                "q2": _quantile(vals, 0.50),
                "q3": _quantile(vals, 0.75),
            }

    rng = np.random.default_rng(int(bootstrap_seed))
    summary_acc: Dict[Tuple[str, int], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0 for _ in range(int(bootstrap_reps))])
    )
    summary_counts: Dict[Tuple[str, int], int] = defaultdict(int)
    stability_rows: List[dict] = []

    for method in methods:
        for key in available_keys:
            row = method_rows[method][key]
            raw_group = raw_by_key[key]
            positions = sorted(raw_group)
            candidates = _parse_candidate_experts(row.get("candidate_experts", ""))
            if not positions or not candidates:
                continue
            k = int(key[3])
            deterministic_scores = _method_deterministic_scores(row, method)
            deterministic_selected = _to_int(row.get("selected_expert", -999999))
            eval_map = _json_float_map(row.get("eval_nelbo_by_expert_json", "{}"))
            oracle = _to_int(row.get("candidate_oracle_expert", row.get("oracle_expert", -999999)))
            eval_ranks = _rank_map(eval_map)
            oracle_eval = float(eval_map.get(oracle, _to_float(row.get("candidate_oracle_nelbo", 0.0))))
            alpha = _to_float(row.get("alpha", 0.0)) if method == CONSERVATIVE_METHOD else 0.0
            oracle_hits: List[float] = []
            rank_le2: List[float] = []
            gap_pcts: List[float] = []
            high_regret: List[float] = []
            catastrophic: List[float] = []
            margins: List[float] = []
            changed: List[float] = []

            for rep in range(int(bootstrap_reps)):
                sampled_positions = [
                    positions[int(i)]
                    for i in rng.integers(0, len(positions), size=k).tolist()
                ]
                score_by_expert: Dict[int, float] = {}
                for expert in candidates:
                    values = [float(raw_group[pos][int(expert)]) for pos in sampled_positions]
                    mean = float(_mean(values))
                    stderr = _support_stderr(values, k=k)
                    score_by_expert[int(expert)] = mean + float(alpha) * stderr
                selected = _select_bootstrap_expert(
                    score_by_expert=score_by_expert,
                    deterministic_score_by_expert=deterministic_scores,
                )
                selected_eval = float(eval_map[selected])
                gap_pct = float(((selected_eval - oracle_eval) / max(abs(oracle_eval), 1.0e-12)) * 100.0)
                margin = _score_margin_lower(score_by_expert)
                spearman = float(
                    spearman_corr(
                        [-float(score_by_expert[expert]) for expert in candidates],
                        [-float(eval_map[expert]) for expert in candidates],
                    )
                )
                oracle_hits.append(1.0 if int(selected) == int(oracle) else 0.0)
                rank_le2.append(1.0 if _to_int(eval_ranks.get(selected, 999)) <= 2 else 0.0)
                gap_pcts.append(gap_pct)
                high_regret.append(1.0 if gap_pct > 2.0 else 0.0)
                catastrophic.append(1.0 if gap_pct > 5.0 else 0.0)
                margins.append(margin)
                changed.append(1.0 if int(selected) != int(deterministic_selected) else 0.0)

                summary_key = (method, k)
                summary_acc[summary_key]["top1"][rep] += oracle_hits[-1]
                summary_acc[summary_key]["spearman"][rep] += spearman
                summary_acc[summary_key]["oracle_gap_pct"][rep] += gap_pct
                summary_acc[summary_key]["high_regret_rate"][rep] += high_regret[-1]
                summary_acc[summary_key]["selection_stability"][rep] += 1.0 - changed[-1]
            summary_counts[(method, k)] += 1
            gap_lo, gap_hi = _ci(gap_pcts)
            thresholds = margin_thresholds[(method, k)]
            deterministic_margin = deterministic_margin_by_method_key[(method, key)]
            stability_rows.append(
                {
                    "experiment_seed": key[0],
                    "heldout_center": key[1],
                    "support_size": key[3],
                    "support_seed": key[2],
                    "method": METHOD_LABELS[method],
                    "source_method": method,
                    "deterministic_selected_expert": deterministic_selected,
                    "oracle_expert": oracle,
                    "selection_stability": float(1.0 - _mean(changed)),
                    "p_oracle_selected": _mean(oracle_hits),
                    "p_eval_rank_le_2": _mean(rank_le2),
                    "mean_bootstrap_oracle_gap_pct": _mean(gap_pcts),
                    "ci_low_oracle_gap_pct": gap_lo,
                    "ci_high_oracle_gap_pct": gap_hi,
                    "p_high_regret_gt_2": _mean(high_regret),
                    "p_catastrophic_gt_5": _mean(catastrophic),
                    "p_selection_changed": _mean(changed),
                    "mean_bootstrap_support_margin": _mean(margins),
                    "margin_bin": _bootstrap_margin_bin(
                        margin=deterministic_margin,
                        thresholds=thresholds,
                    ),
                    "split_id": key[4],
                }
            )

    metadata_top1 = _fixed_method_metric_by_k(sample_rows, method=METADATA_METHOD, metric="top1")
    metadata_gap = _fixed_method_metric_by_k(sample_rows, method=METADATA_METHOD, metric="gap")
    summary_rows: List[dict] = []
    for (method, k), metric_lists in sorted(summary_acc.items(), key=lambda item: (item[0][0], item[0][1])):
        n = int(summary_counts[(method, k)])
        if n <= 0:
            continue
        normalized = {
            name: [float(value) / float(n) for value in values]
            for name, values in metric_lists.items()
        }
        top1_lo, top1_hi = _ci(normalized["top1"])
        spearman_lo, spearman_hi = _ci(normalized["spearman"])
        gap_lo, gap_hi = _ci(normalized["oracle_gap_pct"])
        regret_lo, regret_hi = _ci(normalized["high_regret_rate"])
        stability_lo, stability_hi = _ci(normalized["selection_stability"])
        summary_rows.append(
            {
                "method": METHOD_LABELS[method],
                "source_method": method,
                "support_size": int(k),
                "n_decisions": n,
                "top1_mean": _mean(normalized["top1"]),
                "top1_ci_low": top1_lo,
                "top1_ci_high": top1_hi,
                "spearman_mean": _mean(normalized["spearman"]),
                "spearman_ci_low": spearman_lo,
                "spearman_ci_high": spearman_hi,
                "oracle_gap_pct_mean": _mean(normalized["oracle_gap_pct"]),
                "oracle_gap_pct_ci_low": gap_lo,
                "oracle_gap_pct_ci_high": gap_hi,
                "high_regret_rate_mean": _mean(normalized["high_regret_rate"]),
                "high_regret_rate_ci_low": regret_lo,
                "high_regret_rate_ci_high": regret_hi,
                "selection_stability_mean": _mean(normalized["selection_stability"]),
                "selection_stability_ci_low": stability_lo,
                "selection_stability_ci_high": stability_hi,
                "beats_metadata_gap_prob": _rate(
                    normalized["oracle_gap_pct"],
                    lambda value: float(value) < float(metadata_gap.get(int(k), float("inf"))),
                ),
                "beats_metadata_top1_prob": _rate(
                    normalized["top1"],
                    lambda value: float(value) > float(metadata_top1.get(int(k), float("-inf"))),
                ),
                "bootstrap_reps": int(bootstrap_reps),
                "bootstrap_seed": int(bootstrap_seed),
            }
        )

    margin_groups: Dict[Tuple[str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in stability_rows:
        margin_groups[(str(row["method"]), _to_int(row["support_size"], 0), str(row["margin_bin"]))].append(row)
    margin_rows: List[dict] = []
    for (method, k, margin_bin), rows in sorted(margin_groups.items(), key=lambda item: item[0]):
        margin_rows.append(
            {
                "method": method,
                "support_size": int(k),
                "margin_bin": margin_bin,
                "n_decisions": len(rows),
                "selection_stability_mean": _mean(_to_float(row.get("selection_stability", 0.0)) for row in rows),
                "p_oracle_selected_mean": _mean(_to_float(row.get("p_oracle_selected", 0.0)) for row in rows),
                "oracle_gap_pct_mean": _mean(_to_float(row.get("mean_bootstrap_oracle_gap_pct", 0.0)) for row in rows),
                "high_regret_rate_mean": _mean(_to_float(row.get("p_high_regret_gt_2", 0.0)) for row in rows),
                "wrong_confident_rate": _rate(
                    rows,
                    lambda row: _to_int(row.get("deterministic_selected_expert", -1))
                    != _to_int(row.get("oracle_expert", -2))
                    and _to_float(row.get("mean_bootstrap_oracle_gap_pct", 0.0)) > 2.0,
                ),
            }
        )

    return stability_rows, summary_rows, margin_rows, {
        "status": "pass",
        "n_decision_groups": len(available_keys),
        "bootstrap_reps": int(bootstrap_reps),
        "bootstrap_seed": int(bootstrap_seed),
    }


def classify_uncertainty_support(
    *,
    bootstrap_summary: Sequence[Mapping[str, Any]],
    sample_rows: Sequence[Mapping[str, Any]],
) -> dict:
    if not bootstrap_summary:
        return {"classification": "Unavailable", "reasons": ["support bootstrap artifacts are missing"]}
    static_top1 = _fixed_method_metric_by_k(sample_rows, method=STATIC_METHOD, metric="top1")
    static_gap = _fixed_method_metric_by_k(sample_rows, method=STATIC_METHOD, metric="gap")
    direct = [row for row in bootstrap_summary if str(row.get("source_method", "")) == DIRECT_METHOD]
    strong_rows = [
        row for row in direct
        if _to_float(row.get("top1_ci_low", 0.0)) > _to_float(static_top1.get(_to_int(row.get("support_size", 0)), 1.0))
        and _to_float(row.get("oracle_gap_pct_ci_high", 0.0)) < _to_float(static_gap.get(_to_int(row.get("support_size", 0)), 0.0))
        and _to_float(row.get("selection_stability_mean", 0.0)) >= 0.80
        and _to_float(row.get("high_regret_rate_ci_high", 1.0)) < STRONG_STATIC_HIGH_REGRET_CI_GATE
    ]
    if direct and len(strong_rows) == len(direct):
        return {"classification": "Strong uncertainty support", "reasons": ["all direct support-size rows pass uncertainty gates"]}
    if direct and any(_to_float(row.get("selection_stability_mean", 0.0)) >= 0.80 for row in direct):
        return {"classification": "Moderate uncertainty support", "reasons": ["point estimates are bootstrap-supported but at least one CI or support-size row is weaker"]}
    return {"classification": "Weak uncertainty support", "reasons": ["selection stability or bootstrap regret uncertainty remains weak"]}


def _trend_row(scope: str, group_value: object, rows: Sequence[Mapping[str, Any]]) -> dict:
    by_k = {k: [row for row in rows if _to_int(row.get("k", 0)) == k] for k in SUPPORT_SIZES}

    def metric(k: int, name: str) -> float:
        vals = by_k.get(k, [])
        if name == "top1":
            return _mean(_to_float(row.get("top1_oracle_hit", 0.0)) for row in vals)
        if name == "gap":
            return _mean(_to_float(row.get("oracle_gap_pct", 0.0)) for row in vals)
        if name == "high_regret":
            return _rate(vals, lambda row: _to_float(row.get("oracle_gap_pct", 0.0)) > 2.0)
        raise KeyError(name)

    top1_vals = [metric(k, "top1") for k in SUPPORT_SIZES]
    gap_vals = [metric(k, "gap") for k in SUPPORT_SIZES]
    high_vals = [metric(k, "high_regret") for k in SUPPORT_SIZES]
    top1_endpoint_delta = top1_vals[-1] - top1_vals[0]
    gap_endpoint_delta = gap_vals[0] - gap_vals[-1]
    severe_k16_to_k32 = int(
        (top1_vals[-1] < top1_vals[-2] - SEVERE_TOP1_DEGRADATION)
        or (gap_vals[-1] > gap_vals[-2] + SEVERE_GAP_DEGRADATION)
        or (high_vals[-1] > max(LOW_OVERALL_HIGH_REGRET_RATE, high_vals[-2] + SEVERE_HIGH_REGRET_DEGRADATION))
    )
    out = {
        "scope": scope,
        "group_value": group_value,
        "n_decisions": len(rows),
        "top1_k4": top1_vals[0],
        "top1_k8": top1_vals[1],
        "top1_k16": top1_vals[2],
        "top1_k32": top1_vals[3],
        "oracle_gap_pct_k4": gap_vals[0],
        "oracle_gap_pct_k8": gap_vals[1],
        "oracle_gap_pct_k16": gap_vals[2],
        "oracle_gap_pct_k32": gap_vals[3],
        "high_regret_rate_gt2_k4": high_vals[0],
        "high_regret_rate_gt2_k8": high_vals[1],
        "high_regret_rate_gt2_k16": high_vals[2],
        "high_regret_rate_gt2_k32": high_vals[3],
        "top1_k32_minus_k4": top1_endpoint_delta,
        "gap_k4_minus_k32": gap_endpoint_delta,
        "strict_top1_monotonic": int(all(top1_vals[idx + 1] >= top1_vals[idx] - 1.0e-12 for idx in range(3))),
        "strict_gap_monotonic": int(all(gap_vals[idx + 1] <= gap_vals[idx] + 1.0e-12 for idx in range(3))),
        "directional_support_size_evidence": int(top1_endpoint_delta > 0.0 or gap_endpoint_delta > 0.0),
        "severe_k16_to_k32_degradation": severe_k16_to_k32,
        "large_k_high_regret_low": int(max(high_vals[2:]) <= LOW_OVERALL_HIGH_REGRET_RATE),
    }
    return out


def build_support_size_trends(decisions: Sequence[Mapping[str, Any]]) -> List[dict]:
    out = [_trend_row("overall", "all", decisions)]
    for center in sorted({int(row["heldout_center"]) for row in decisions}):
        out.append(_trend_row("heldout_center", center, [row for row in decisions if int(row["heldout_center"]) == center]))
    for seed in sorted({int(row["run_seed"]) for row in decisions}):
        out.append(_trend_row("run_seed", seed, [row for row in decisions if int(row["run_seed"]) == seed]))
    for support_seed in sorted({int(row["support_seed"]) for row in decisions}):
        out.append(_trend_row("support_seed", support_seed, [row for row in decisions if int(row["support_seed"]) == support_seed]))
    return out


def build_selection_counts(sample_rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    rows = [row for row in sample_rows if str(row.get("method", "")) in THESIS_FACING_METHODS]
    out: List[dict] = []
    grouped: Dict[Tuple[str, object, object], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        method = METHOD_LABELS[str(row["method"])]
        center = _to_int(row.get("query_domain", row.get("fold_query_domain", 0)))
        k = _to_int(row.get("support_size_requested", 0))
        grouped[(method, "", "")].append(row)
        grouped[(method, center, k)].append(row)
    for (method, center, k), vals in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        counts = Counter(_to_int(row.get("selected_expert", -999999)) for row in vals)
        for expert, count in sorted(counts.items()):
            out.append(
                {
                    "scope": "overall" if center == "" else "by_center_k",
                    "method": method,
                    "heldout_center": center,
                    "k": k,
                    "selected_expert": expert,
                    "count": count,
                    "share": float(count / len(vals)) if vals else 0.0,
                }
            )
    return out


def build_scale_control(candidate_rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    groups: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        groups[_to_int(row.get("candidate_expert", -999999))].append(row)
    out: List[dict] = []
    total_selected = sum(_to_int(row.get("is_selected", 0)) for row in candidate_rows)
    for expert, rows in sorted(groups.items()):
        selected_count = sum(_to_int(row.get("is_selected", 0)) for row in rows)
        out.append(
            {
                "candidate_expert": expert,
                "n_opportunities": len(rows),
                "selected_count": selected_count,
                "selected_share_of_opportunities": float(selected_count / len(rows)) if rows else 0.0,
                "selected_share_of_all_decisions": float(selected_count / total_selected) if total_selected else 0.0,
                "oracle_count": sum(_to_int(row.get("is_eval_oracle", 0)) for row in rows),
                "support_nelbo_mean": _mean(_to_float(row.get("support_nelbo", 0.0)) for row in rows),
                "support_nelbo_std": _std(_to_float(row.get("support_nelbo", 0.0)) for row in rows),
                "eval_nelbo_mean": _mean(_to_float(row.get("eval_nelbo", 0.0)) for row in rows),
                "eval_nelbo_std": _std(_to_float(row.get("eval_nelbo", 0.0)) for row in rows),
                "support_z_within_decision_mean": _mean(_to_float(row.get("support_z_within_decision", 0.0)) for row in rows),
                "eval_z_within_decision_mean": _mean(_to_float(row.get("eval_z_within_decision", 0.0)) for row in rows),
                "dominant_selection_flag": int(float(selected_count / total_selected) > EXPERT_DOMINANCE_SHARE) if total_selected else 0,
            }
        )
    return out


def build_direct_vs_conservative(sample_rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    direct = { _decision_key(row): decision_diagnostic(row) for row in _method_rows(sample_rows, DIRECT_METHOD) }
    conservative = { _decision_key(row): decision_diagnostic(row) for row in _method_rows(sample_rows, CONSERVATIVE_METHOD) }
    out: List[dict] = []
    for key in sorted(set(direct) & set(conservative)):
        drow = direct[key]
        crow = conservative[key]
        dgap = _to_float(drow.get("oracle_gap_pct", 0.0))
        cgap = _to_float(crow.get("oracle_gap_pct", 0.0))
        if abs(dgap - cgap) <= 1.0e-12:
            winner = "tie"
        elif dgap < cgap:
            winner = "direct"
        else:
            winner = "conservative"
        out.append(
            {
                "run_seed": key[0],
                "heldout_center": key[1],
                "support_seed": key[2],
                "k": key[3],
                "direct_selected_expert": drow.get("selected_expert", ""),
                "conservative_selected_expert": crow.get("selected_expert", ""),
                "agreement_flag": int(drow.get("selected_expert") == crow.get("selected_expert")),
                "direct_oracle_gap_pct": dgap,
                "conservative_oracle_gap_pct": cgap,
                "direct_minus_conservative_oracle_gap_pct": dgap - cgap,
                "winner": winner,
                "direct_support_margin": drow.get("support_margin", ""),
                "conservative_support_margin": crow.get("support_margin", ""),
            }
        )
    return out


def _summary_by_method(summary_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {str(row["source_method"]): row for row in summary_rows}


def _direct_improves_over_baselines(summary_rows: Sequence[Mapping[str, Any]]) -> int:
    by_method = _summary_by_method(summary_rows)
    direct = by_method[DIRECT_METHOD]
    baselines = [by_method[m] for m in BASELINE_METHODS if m in by_method]
    wins = 0
    if baselines and _to_float(direct["top1"]) > max(_to_float(row["top1"]) for row in baselines):
        wins += 1
    if baselines and _to_float(direct["spearman"]) > max(_to_float(row["spearman"]) for row in baselines):
        wins += 1
    if baselines and _to_float(direct["oracle_gap_pct"]) < min(_to_float(row["oracle_gap_pct"]) for row in baselines):
        wins += 1
    return wins


def _center1_strong_blocked(
    *,
    decisions: Sequence[Mapping[str, Any]],
    sample_rows: Sequence[Mapping[str, Any]],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    center1 = [row for row in decisions if _to_int(row.get("heldout_center", -1)) == 1]
    by_k = {
        k: [row for row in center1 if _to_int(row.get("k", 0)) == k]
        for k in (16, 32)
    }
    high_k16 = _rate(by_k[16], lambda row: _to_float(row.get("oracle_gap_pct", 0.0)) > 2.0)
    high_k32 = _rate(by_k[32], lambda row: _to_float(row.get("oracle_gap_pct", 0.0)) > 2.0)
    if high_k16 > 0.0 and high_k32 > 0.0:
        reasons.append("center 1 high-regret failures persist at k>=16")
    wrong_confident = [
        row for row in center1
        if str(row.get("support_confidence_class", "")) == "wrong_confident"
    ]
    if wrong_confident:
        reasons.append("center 1 has wrong-confident high-regret failures")

    large_k_rows = [
        row for row in sample_rows
        if str(row.get("method", "")) in {DIRECT_METHOD, METADATA_METHOD, SOURCE_GLOBAL_METHOD}
        and _to_int(row.get("query_domain", row.get("fold_query_domain", 0))) == 1
        and _to_int(row.get("support_size_requested", 0)) >= 16
    ]
    gaps = {
        method: _mean(
            _to_float(row.get("mean_oracle_gap_pct", 0.0))
            for row in large_k_rows
            if str(row.get("method", "")) == method
        )
        for method in {DIRECT_METHOD, METADATA_METHOD, SOURCE_GLOBAL_METHOD}
    }
    direct_gap = gaps.get(DIRECT_METHOD, 0.0)
    for baseline in (METADATA_METHOD, SOURCE_GLOBAL_METHOD):
        if baseline in gaps and direct_gap > gaps[baseline] + 1.0e-12:
            reasons.append(f"center 1 direct gap at k>=16 is worse than {METHOD_LABELS[baseline]}")
    return bool(reasons), reasons


def classify_verification(
    *,
    protocol_gate: Mapping[str, Any],
    count_assertions: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    support_size_trends: Sequence[Mapping[str, Any]],
    selection_counts: Sequence[Mapping[str, Any]],
    sample_rows: Sequence[Mapping[str, Any]],
) -> dict:
    reasons: List[str] = []
    if protocol_gate.get("status") != "pass":
        return {
            "classification": "Blocked",
            "reasons": list(protocol_gate.get("failures", [])),
        }
    if count_assertions.get("status") != "pass":
        return {
            "classification": "Blocked",
            "reasons": ["expected decision/candidate counts do not match observed artifacts"],
        }

    by_method = _summary_by_method(summary_rows)
    direct = by_method[DIRECT_METHOD]
    direct_top1 = _to_float(direct["top1"])
    direct_spearman = _to_float(direct["spearman"])
    direct_gap = _to_float(direct["oracle_gap_pct"])
    direct_high_regret = _rate(decisions, lambda row: _to_float(row.get("oracle_gap_pct", 0.0)) > 2.0)
    direct_catastrophic = _rate(decisions, lambda row: _to_float(row.get("oracle_gap_pct", 0.0)) > 5.0)
    wrong_confident_count = sum(
        1 for row in decisions if str(row.get("support_confidence_class", "")) == "wrong_confident"
    )
    direct_beats_all_gap = all(
        direct_gap < _to_float(by_method[method]["oracle_gap_pct"])
        for method in (*BASELINE_METHODS, CONSERVATIVE_METHOD)
        if method in by_method
    )
    min_gap = min(_to_float(row["oracle_gap_pct"]) for row in summary_rows)
    max_top1 = max(_to_float(row["top1"]) for row in summary_rows)
    max_spearman = max(_to_float(row["spearman"]) for row in summary_rows)
    direct_best_or_tied = (
        direct_gap <= min_gap + 0.25
        and direct_top1 >= max_top1 - 0.05
        and direct_spearman >= max_spearman - 0.05
    )
    overall_trend = next(row for row in support_size_trends if row["scope"] == "overall")
    directional_k = _to_int(overall_trend.get("directional_support_size_evidence", 0)) == 1
    no_severe_k_degradation = _to_int(overall_trend.get("severe_k16_to_k32_degradation", 0)) == 0
    large_k_high_regret_low = _to_int(overall_trend.get("large_k_high_regret_low", 0)) == 1

    direct_overall_selection = [
        row for row in selection_counts
        if row.get("scope") == "overall" and row.get("method") == METHOD_LABELS[DIRECT_METHOD]
    ]
    dominant_share = max((_to_float(row.get("share", 0.0)) for row in direct_overall_selection), default=0.0)
    source_global = by_method.get(SOURCE_GLOBAL_METHOD)
    source_global_nearly_matches = False
    if source_global:
        source_global_nearly_matches = (
            direct_top1 - _to_float(source_global["top1"]) <= NEAR_MATCH_TOP1_TOL
            and _to_float(source_global["oracle_gap_pct"]) - direct_gap <= NEAR_MATCH_GAP_TOL
        )
    expert_dominance_downgrade = dominant_share > EXPERT_DOMINANCE_SHARE and source_global_nearly_matches
    center1_block, center1_reasons = _center1_strong_blocked(decisions=decisions, sample_rows=sample_rows)

    if (
        direct_beats_all_gap
        and direct_top1 >= 0.75
        and direct_spearman >= 0.70
        and direct_gap <= 1.0
        and directional_k
        and no_severe_k_degradation
        and direct_high_regret <= LOW_OVERALL_HIGH_REGRET_RATE
        and large_k_high_regret_low
        and wrong_confident_count == 0
        and not expert_dominance_downgrade
        and not center1_block
    ):
        return {"classification": "Strong", "reasons": ["all Strong gates passed"]}

    if center1_block:
        reasons.extend(center1_reasons)
    if expert_dominance_downgrade:
        reasons.append("expert dominance plus source-global near-match prevents Strong")
    if wrong_confident_count:
        reasons.append(f"{wrong_confident_count} wrong-confident failure(s) prevent Strong")
    if direct_high_regret > LOW_OVERALL_HIGH_REGRET_RATE:
        reasons.append("overall high-regret rate prevents Strong")

    if (
        direct_best_or_tied
        and direct_top1 >= 0.65
        and direct_spearman >= 0.60
        and direct_gap <= 2.0
        and direct_high_regret <= LIMITED_HIGH_REGRET_RATE
        and direct_catastrophic <= 0.05
        and wrong_confident_count <= 1
    ):
        return {
            "classification": "Moderate-Strong",
            "reasons": reasons or ["direct is best or tied-best with limited high-regret failures"],
        }

    if _direct_improves_over_baselines(summary_rows) >= 2 and direct_catastrophic <= 0.10:
        return {
            "classification": "Moderate",
            "reasons": reasons or ["direct improves over baselines on at least two primary metrics"],
        }
    return {
        "classification": "Weak",
        "reasons": reasons or ["direct has partial signal but does not satisfy higher gates"],
    }


def write_scatter_pdf(path: Path, candidate_rows: Sequence[Mapping[str, Any]]) -> None:
    if not os.environ.get("XDG_CACHE_HOME"):
        xdg_cache = Path(tempfile.gettempdir()) / "cvae_support_nelbo_cache"
        xdg_cache.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(xdg_cache)
    if not os.environ.get("MPLCONFIGDIR"):
        mpl_config = Path(tempfile.gettempdir()) / "cvae_support_nelbo_matplotlib"
        mpl_config.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config)
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.backends.backend_pdf import PdfPages  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)

    def add_page(pdf: Any, rows: Sequence[Mapping[str, Any]], title: str) -> None:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        xs = [_to_float(row.get("support_nelbo", 0.0)) for row in rows]
        ys = [_to_float(row.get("eval_nelbo", 0.0)) for row in rows]
        colors = [_to_int(row.get("k", 0)) for row in rows]
        ax.scatter(xs, ys, c=colors, s=16, alpha=0.65, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("support NELBO per candidate expert")
        ax.set_ylabel("held-out eval NELBO per candidate expert")
        ax.grid(True, alpha=0.25)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    with PdfPages(path) as pdf:
        add_page(pdf, candidate_rows, "Direct support-NELBO: pooled candidates")
        for center in sorted({int(row["heldout_center"]) for row in candidate_rows}):
            rows = [row for row in candidate_rows if int(row["heldout_center"]) == center]
            add_page(pdf, rows, f"Direct support-NELBO: heldout center {center}")


def write_report(
    *,
    path: Path,
    classification: Mapping[str, Any],
    protocol_gate: Mapping[str, Any],
    count_assertions: Mapping[str, Any],
    method_summary: Sequence[Mapping[str, Any]],
    per_center_per_k: Sequence[Mapping[str, Any]],
    failure_cases: Sequence[Mapping[str, Any]],
    margin_reliability: Sequence[Mapping[str, Any]],
    support_size_trends: Sequence[Mapping[str, Any]],
    selection_counts: Sequence[Mapping[str, Any]],
    disagreement_rows: Sequence[Mapping[str, Any]],
    bootstrap_summary: Sequence[Mapping[str, Any]],
    bootstrap_status: Mapping[str, Any],
    uncertainty_classification: Mapping[str, Any],
) -> None:
    direct = next(row for row in method_summary if row["source_method"] == DIRECT_METHOD)
    center1 = [
        row for row in per_center_per_k
        if _to_int(row.get("heldout_center", -1)) == 1
    ]
    direct_selection = [
        row for row in selection_counts
        if row.get("scope") == "overall" and row.get("method") == METHOD_LABELS[DIRECT_METHOD]
    ]
    rank_total = sum(_to_int(row.get("n_decisions", 0)) for row in per_center_per_k)
    p_oracle_support_rank_le2 = (
        sum(
            _to_float(row.get("p_eval_oracle_support_rank_le2", 0.0))
            * _to_int(row.get("n_decisions", 0))
            for row in per_center_per_k
        )
        / rank_total
        if rank_total
        else 0.0
    )
    p_selected_eval_rank_le2 = (
        sum(
            _to_float(row.get("p_selected_eval_rank_le2", 0.0))
            * _to_int(row.get("n_decisions", 0))
            for row in per_center_per_k
        )
        / rank_total
        if rank_total
        else 0.0
    )
    disagreement_count = sum(1 for row in disagreement_rows if _to_int(row.get("agreement_flag", 0)) == 0)
    direct_wins = sum(1 for row in disagreement_rows if row.get("winner") == "direct")
    conservative_wins = sum(1 for row in disagreement_rows if row.get("winner") == "conservative")
    ties = sum(1 for row in disagreement_rows if row.get("winner") == "tie")
    failure_counts = Counter(str(row.get("regret_class", "")) for row in failure_cases)
    confidence_counts = Counter(str(row.get("support_confidence_class", "")) for row in failure_cases)
    overall_trend = next(row for row in support_size_trends if row["scope"] == "overall")
    protocol_status = protocol_gate.get("status", "unknown")

    lines = [
        "# Support-NELBO Verification Report",
        "",
        "## Decision",
        "",
        f"- Classification: `{classification.get('classification', 'unknown')}`",
        f"- Protocol gate: `{protocol_status}`",
        f"- Count gate: `{count_assertions.get('status', 'unknown')}`",
        f"- Uncertainty support: `{uncertainty_classification.get('classification', 'unknown')}`",
        "- Allowed claim: the support-NELBO routing result passes protocol and count gates; target-support resampling supports selected-expert stability, margin-dependent reliability, and low oracle-gap regret under support-selection uncertainty when the bootstrap gates pass.",
        "- Disallowed claim: this bootstrap proves full held-out test-set uncertainty or general robustness across support regimes and domains.",
        "",
        "Reasons:",
        *[f"- {reason}" for reason in classification.get("reasons", [])],
        "",
        "## Headline Metrics",
        "",
        _markdown_table(
            method_summary,
            [
                ("Method", "method"),
                ("Rows", "n_decisions"),
                ("Top1", "top1"),
                ("Spearman", "spearman"),
                ("Oracle gap pct", "oracle_gap_pct"),
                ("High regret >2%", "high_regret_rate_gt2"),
            ],
        ),
        "",
        f"Direct headline check: top1={_format_float(direct['top1'])}, Spearman={_format_float(direct['spearman'])}, oracle gap pct={_format_float(direct['oracle_gap_pct'])}.",
        "",
        "## Protocol And Counts",
        "",
        f"- Protocol failures: {protocol_gate.get('failure_count', 0)}",
        f"- Expected direct decisions: {count_assertions.get('expected_decisions', '')}",
        f"- Observed direct decisions: {count_assertions.get('observed_decisions', '')}",
        f"- Expected direct candidate rows: {count_assertions.get('expected_candidates', '')}",
        f"- Observed direct candidate rows: {count_assertions.get('observed_candidates', '')}",
        f"- Expected raw support-NELBO rows: {count_assertions.get('expected_support_raw_rows', '')}",
        f"- Observed raw support-NELBO rows: {count_assertions.get('observed_support_raw_rows', '')}",
        "",
        "## Support-Size Evidence",
        "",
        _markdown_table(
            [overall_trend],
            [
                ("Top1 k4", "top1_k4"),
                ("Top1 k32", "top1_k32"),
                ("Gap k4", "oracle_gap_pct_k4"),
                ("Gap k32", "oracle_gap_pct_k32"),
                ("Directional", "directional_support_size_evidence"),
                ("Severe k16->k32", "severe_k16_to_k32_degradation"),
            ],
        ),
        "",
        "## Center 1 Stress Case",
        "",
        _markdown_table(
            center1,
            [
                ("k", "k"),
                ("Top1", "top1"),
                ("Spearman", "spearman"),
                ("Gap pct", "oracle_gap_pct"),
                ("High regret >2%", "high_regret_rate_gt2"),
            ],
        ),
        "",
        "## Failure Anatomy",
        "",
        f"- Top1 failure count: {len(failure_cases)}",
        f"- Regret classes: {dict(failure_counts)}",
        f"- Support-confidence classes among failures: {dict(confidence_counts)}",
        f"- P(eval oracle support-rank <= 2): {_format_float(p_oracle_support_rank_le2)}",
        f"- P(selected expert eval-rank <= 2): {_format_float(p_selected_eval_rank_le2)}",
        "",
        _markdown_table(
            failure_cases,
            [
                ("center", "heldout_center"),
                ("k", "k"),
                ("support_seed", "support_seed"),
                ("selected", "selected_expert"),
                ("oracle", "oracle_expert"),
                ("gap pct", "oracle_gap_pct"),
                ("support margin", "support_margin"),
                ("regret", "regret_class"),
                ("confidence", "support_confidence_class"),
            ],
            limit=12,
        ),
        "",
        "## Margin Reliability",
        "",
        _markdown_table(
            margin_reliability,
            [
                ("k", "k"),
                ("bin", "margin_bin"),
                ("n", "n_decisions"),
                ("top1", "top1"),
                ("gap pct", "oracle_gap_pct"),
                ("high regret >2%", "high_regret_rate_gt2"),
                ("wrong confident", "wrong_confident_rate"),
            ],
        ),
        "",
        "## Adaptivity And Conservative Check",
        "",
        _markdown_table(
            direct_selection,
            [
                ("expert", "selected_expert"),
                ("count", "count"),
                ("share", "share"),
            ],
        ),
        "",
        f"Direct/conservative disagreements: {disagreement_count}; direct wins: {direct_wins}; conservative wins: {conservative_wins}; ties: {ties}.",
        "",
        "## Bootstrap Stability",
        "",
        f"- Bootstrap status: `{bootstrap_status.get('status', 'unknown')}`",
        f"- Bootstrap reps: {bootstrap_status.get('bootstrap_reps', '')}",
        f"- Bootstrap seed: {bootstrap_status.get('bootstrap_seed', '')}",
        "",
        _markdown_table(
            bootstrap_summary,
            [
                ("Method", "method"),
                ("k", "support_size"),
                ("n", "n_decisions"),
                ("Top1", "top1_mean"),
                ("Top1 lo", "top1_ci_low"),
                ("Gap", "oracle_gap_pct_mean"),
                ("Gap hi", "oracle_gap_pct_ci_high"),
                ("High regret hi", "high_regret_rate_ci_high"),
                ("Stability", "selection_stability_mean"),
            ],
        ) if bootstrap_summary else "Skipped: support-response raw support NELBO rows are unavailable.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_outputs(
    *,
    experiment_root: Path,
    output_dir: Path,
    protocol_audit: Path,
    write_plots: bool = True,
    bootstrap_reps: int = BOOTSTRAP_REPS_DEFAULT,
    bootstrap_seed: int = BOOTSTRAP_SEED_DEFAULT,
) -> Dict[str, str]:
    sample_rows, split_rows, support_raw_rows, protocol_locks, audit_rows = load_source_artifacts(experiment_root, protocol_audit)
    direct_rows = _method_rows(sample_rows, DIRECT_METHOD)
    direct_decisions = annotate_decisions(direct_rows)
    direct_candidates = flatten_candidate_rows(direct_decisions)
    protocol_gate = build_protocol_gate(
        sample_rows=sample_rows,
        split_rows=split_rows,
        protocol_locks=protocol_locks,
        audit_rows=audit_rows,
    )
    count_assertions = build_expected_count_assertions(
        direct_rows=direct_rows,
        direct_candidate_rows=direct_candidates,
        split_rows=split_rows,
        support_raw_rows=support_raw_rows,
    )
    method_summary = summarize_methods(sample_rows)
    per_center_per_k = build_per_center_per_k(direct_decisions)
    failure_cases = build_failure_cases(direct_decisions)
    high_regret_distribution = build_high_regret_distribution(sample_rows)
    margin_reliability = build_margin_reliability(direct_decisions)
    support_size_trends = build_support_size_trends(direct_decisions)
    selection_counts = build_selection_counts(sample_rows)
    scale_control = build_scale_control(direct_candidates)
    disagreement = build_direct_vs_conservative(sample_rows)
    bootstrap_stability, bootstrap_summary, bootstrap_margin_summary, bootstrap_status = build_support_bootstrap_artifacts(
        sample_rows=sample_rows,
        support_raw_rows=support_raw_rows,
        bootstrap_reps=int(bootstrap_reps),
        bootstrap_seed=int(bootstrap_seed),
    )
    uncertainty_classification = classify_uncertainty_support(
        bootstrap_summary=bootstrap_summary,
        sample_rows=sample_rows,
    )
    classification = classify_verification(
        protocol_gate=protocol_gate,
        count_assertions=count_assertions,
        summary_rows=method_summary,
        decisions=direct_decisions,
        support_size_trends=support_size_trends,
        selection_counts=selection_counts,
        sample_rows=sample_rows,
    )

    outputs = {
        "per_center_per_k": output_dir / "support_nelbo_per_center_per_k.csv",
        "failure_cases": output_dir / "support_nelbo_failure_cases.csv",
        "high_regret_distribution": output_dir / "support_nelbo_high_regret_distribution.csv",
        "margin_reliability": output_dir / "support_nelbo_margin_reliability.csv",
        "support_size_trends": output_dir / "support_nelbo_support_size_trends.csv",
        "selection_counts": output_dir / "support_nelbo_selection_counts.csv",
        "scale_control": output_dir / "support_nelbo_scale_control.csv",
        "direct_vs_conservative": output_dir / "support_nelbo_direct_vs_conservative_disagreement.csv",
        "bootstrap_stability": output_dir / "support_nelbo_bootstrap_stability.csv",
        "bootstrap_summary": output_dir / "support_nelbo_bootstrap_summary.csv",
        "bootstrap_margin_summary": output_dir / "support_nelbo_bootstrap_margin_summary.csv",
        "scatter_pdf": output_dir / "support_vs_eval_nelbo_rank_scatter.pdf",
        "report": output_dir / "support_nelbo_verification_report.md",
    }

    _write_csv(
        outputs["per_center_per_k"],
        per_center_per_k,
        [
            "heldout_center",
            "k",
            "n_decisions",
            "top1",
            "spearman",
            "oracle_gap_pct",
            "high_regret_rate_gt1",
            "high_regret_rate_gt2",
            "high_regret_rate_gt5",
            "selected_eval_nelbo",
            "oracle_eval_nelbo",
            "p_eval_oracle_support_rank_le2",
            "p_selected_eval_rank_le2",
        ],
    )
    _write_csv(
        outputs["failure_cases"],
        failure_cases,
        [
            "run_seed",
            "heldout_center",
            "k",
            "support_seed",
            "selected_expert",
            "oracle_expert",
            "support_nelbo_selected",
            "support_nelbo_oracle",
            "eval_nelbo_selected",
            "eval_nelbo_oracle",
            "oracle_gap_pct",
            "support_margin",
            "eval_margin",
            "support_rank_of_eval_oracle",
            "eval_rank_of_support_selected",
            "regret_class",
            "support_confidence_class",
            "support_margin_q1_for_k",
            "support_margin_q3_for_k",
            "source_path",
        ],
    )
    _write_csv(
        outputs["high_regret_distribution"],
        high_regret_distribution,
        [
            "scope",
            "method",
            "heldout_center",
            "k",
            "n_decisions",
            "mean_oracle_gap_pct",
            "max_oracle_gap_pct",
            "high_regret_rate_gt1",
            "high_regret_rate_gt2",
            "high_regret_rate_gt5",
        ],
    )
    _write_csv(
        outputs["margin_reliability"],
        margin_reliability,
        [
            "k",
            "margin_bin",
            "n_decisions",
            "support_margin_q1",
            "support_margin_q2",
            "support_margin_q3",
            "support_margin_min",
            "support_margin_max",
            "top1",
            "oracle_gap_pct",
            "high_regret_rate_gt2",
            "wrong_confident_rate",
        ],
    )
    _write_csv(
        outputs["support_size_trends"],
        support_size_trends,
        [
            "scope",
            "group_value",
            "n_decisions",
            "top1_k4",
            "top1_k8",
            "top1_k16",
            "top1_k32",
            "oracle_gap_pct_k4",
            "oracle_gap_pct_k8",
            "oracle_gap_pct_k16",
            "oracle_gap_pct_k32",
            "high_regret_rate_gt2_k4",
            "high_regret_rate_gt2_k8",
            "high_regret_rate_gt2_k16",
            "high_regret_rate_gt2_k32",
            "top1_k32_minus_k4",
            "gap_k4_minus_k32",
            "strict_top1_monotonic",
            "strict_gap_monotonic",
            "directional_support_size_evidence",
            "severe_k16_to_k32_degradation",
            "large_k_high_regret_low",
        ],
    )
    _write_csv(
        outputs["selection_counts"],
        selection_counts,
        ["scope", "method", "heldout_center", "k", "selected_expert", "count", "share"],
    )
    _write_csv(
        outputs["scale_control"],
        scale_control,
        [
            "candidate_expert",
            "n_opportunities",
            "selected_count",
            "selected_share_of_opportunities",
            "selected_share_of_all_decisions",
            "oracle_count",
            "support_nelbo_mean",
            "support_nelbo_std",
            "eval_nelbo_mean",
            "eval_nelbo_std",
            "support_z_within_decision_mean",
            "eval_z_within_decision_mean",
            "dominant_selection_flag",
        ],
    )
    _write_csv(
        outputs["direct_vs_conservative"],
        disagreement,
        [
            "run_seed",
            "heldout_center",
            "support_seed",
            "k",
            "direct_selected_expert",
            "conservative_selected_expert",
            "agreement_flag",
            "direct_oracle_gap_pct",
            "conservative_oracle_gap_pct",
            "direct_minus_conservative_oracle_gap_pct",
            "winner",
            "direct_support_margin",
            "conservative_support_margin",
        ],
    )
    _write_csv(
        outputs["bootstrap_stability"],
        bootstrap_stability,
        [
            "experiment_seed",
            "heldout_center",
            "support_size",
            "support_seed",
            "method",
            "source_method",
            "deterministic_selected_expert",
            "oracle_expert",
            "selection_stability",
            "p_oracle_selected",
            "p_eval_rank_le_2",
            "mean_bootstrap_oracle_gap_pct",
            "ci_low_oracle_gap_pct",
            "ci_high_oracle_gap_pct",
            "p_high_regret_gt_2",
            "p_catastrophic_gt_5",
            "p_selection_changed",
            "mean_bootstrap_support_margin",
            "margin_bin",
            "split_id",
        ],
    )
    _write_csv(
        outputs["bootstrap_summary"],
        bootstrap_summary,
        [
            "method",
            "source_method",
            "support_size",
            "n_decisions",
            "top1_mean",
            "top1_ci_low",
            "top1_ci_high",
            "spearman_mean",
            "spearman_ci_low",
            "spearman_ci_high",
            "oracle_gap_pct_mean",
            "oracle_gap_pct_ci_low",
            "oracle_gap_pct_ci_high",
            "high_regret_rate_mean",
            "high_regret_rate_ci_low",
            "high_regret_rate_ci_high",
            "selection_stability_mean",
            "selection_stability_ci_low",
            "selection_stability_ci_high",
            "beats_metadata_gap_prob",
            "beats_metadata_top1_prob",
            "bootstrap_reps",
            "bootstrap_seed",
        ],
    )
    _write_csv(
        outputs["bootstrap_margin_summary"],
        bootstrap_margin_summary,
        [
            "method",
            "support_size",
            "margin_bin",
            "n_decisions",
            "selection_stability_mean",
            "p_oracle_selected_mean",
            "oracle_gap_pct_mean",
            "high_regret_rate_mean",
            "wrong_confident_rate",
        ],
    )
    if write_plots:
        write_scatter_pdf(outputs["scatter_pdf"], direct_candidates)
    write_report(
        path=outputs["report"],
        classification=classification,
        protocol_gate=protocol_gate,
        count_assertions=count_assertions,
        method_summary=method_summary,
        per_center_per_k=per_center_per_k,
        failure_cases=failure_cases,
        margin_reliability=margin_reliability,
        support_size_trends=support_size_trends,
        selection_counts=selection_counts,
        disagreement_rows=disagreement,
        bootstrap_summary=bootstrap_summary,
        bootstrap_status=bootstrap_status,
        uncertainty_classification=uncertainty_classification,
    )
    return {key: str(value) for key, value in outputs.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build direct support-NELBO verification artifacts.")
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--protocol-audit", type=Path, default=PROTOCOL_AUDIT)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS_DEFAULT)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED_DEFAULT)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_outputs(
        experiment_root=args.experiment_root,
        output_dir=args.output_dir,
        protocol_audit=args.protocol_audit,
        write_plots=not args.skip_plots,
        bootstrap_reps=int(args.bootstrap_reps),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
