#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from math import isfinite
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


PRIMARY_METHOD = "support_response_pairwise_static_response_indirect"
ANCHOR_METHOD = "support_metadata_routing"
PROTOCOL_VERSION = "support_response_failure_localization_v1"
DEFAULT_MARGIN_THRESHOLDS = (0.0, 0.25, 0.5, 1.0, 1.5)
DEFAULT_SUPPORT_REGRET_THRESHOLDS = (0.0, 2.5, 5.0, 10.0)
UNIT_KEY_FIELDS = (
    "source_csv",
    "seed",
    "query_domain",
    "support_seed",
    "support_size_requested",
    "sampling_policy",
)


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if isfinite(out) else float(default)
    except Exception:
        return float(default)


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return int(default)


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            key_s = str(key)
            if key_s not in seen:
                seen.add(key_s)
                fieldnames.append(key_s)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_map(value: object) -> Dict[int, float]:
    try:
        raw = json.loads(str(value))
    except Exception:
        return {}
    out: Dict[int, float] = {}
    for key, val in dict(raw).items():
        try:
            out[int(key)] = float(val)
        except Exception:
            continue
    return out


def _json_counts(counts: Mapping[int, int]) -> str:
    return json.dumps({str(int(k)): int(v) for k, v in sorted(counts.items())}, sort_keys=True)


def _rank_map(values: Mapping[int, float], *, lower_is_better: bool = True) -> Dict[int, int]:
    direction = 1.0 if lower_is_better else -1.0
    order = sorted(values, key=lambda expert: (direction * float(values[int(expert)]), int(expert)))
    return {int(expert): rank for rank, expert in enumerate(order, start=1)}


def _average_rank_desc(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(float(v) for v in values), key=lambda x: x[1], reverse=True)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    vx = x_arr - float(np.mean(x_arr))
    vy = y_arr - float(np.mean(y_arr))
    denom = float(np.sqrt(np.sum(vx * vx) * np.sum(vy * vy)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(vx * vy) / denom)


def _spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    return _pearson_corr(_average_rank_desc(x), _average_rank_desc(y))


def _argmin_expert(values: Mapping[int, float]) -> int:
    if not values:
        return 0
    return int(sorted(values, key=lambda expert: (float(values[int(expert)]), int(expert)))[0])


def _score_margin(values: Mapping[int, float]) -> float:
    ordered = sorted(float(v) for v in values.values())
    if len(ordered) < 2:
        return 0.0
    return float(ordered[1] - ordered[0])


def _pct_delta(numerator: float, denominator: float) -> float:
    return float((float(numerator) / max(abs(float(denominator)), 1e-12)) * 100.0)


def _unit_key(row: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in UNIT_KEY_FIELDS)


def _infer_run_dir(selection_csv: Path) -> Path:
    if selection_csv.parent.name == "reports":
        return selection_csv.parent.parent
    return selection_csv.parent


def read_selection_rows(paths: Sequence[Path]) -> List[dict]:
    rows: List[dict] = []
    for path in paths:
        source = Path(path)
        for row in _read_csv(source):
            row = dict(row)
            row["source_csv"] = str(source)
            row["run_id"] = _infer_run_dir(source).name
            rows.append(row)
    return rows


def _manifest_label_context(selection_csv: Path) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    manifest = _infer_run_dir(selection_csv) / "manifests" / "samples.csv"
    if not manifest.exists():
        return {}, {}
    labels_by_index: Dict[int, int] = {}
    indices_by_domain: Dict[int, List[int]] = {}
    for idx, row in enumerate(_read_csv(manifest)):
        domain = _to_int(row.get("magnification", row.get("domain", 0)))
        label = _to_int(row.get("label", 0))
        labels_by_index[int(idx)] = int(label)
        indices_by_domain.setdefault(int(domain), []).append(int(idx))
    return labels_by_index, indices_by_domain


def _random_order(indices: Sequence[int], seed: int) -> List[int]:
    rng = np.random.default_rng(int(seed))
    arr = np.asarray(sorted(int(i) for i in indices), dtype=np.int64)
    return [int(i) for i in rng.permutation(arr).tolist()]


def _balanced_order(indices: Sequence[int], labels_by_index: Mapping[int, int], seed: int) -> List[int]:
    rng = np.random.default_rng(int(seed))
    by_label: Dict[int, List[int]] = {}
    for idx in sorted(int(i) for i in indices):
        by_label.setdefault(int(labels_by_index[int(idx)]), []).append(int(idx))

    shuffled: Dict[int, List[int]] = {}
    for label, vals in by_label.items():
        arr = np.asarray(vals, dtype=np.int64)
        shuffled[int(label)] = [int(i) for i in rng.permutation(arr).tolist()]

    labels = sorted(shuffled)
    positions = {label: 0 for label in labels}
    order: List[int] = []
    while True:
        added = False
        for label in labels:
            pos = positions[label]
            if pos < len(shuffled[label]):
                order.append(int(shuffled[label][pos]))
                positions[label] = pos + 1
                added = True
        if not added:
            break
    return order


def _balanced_possible(
    indices: Sequence[int],
    labels_by_index: Mapping[int, int],
    support_size: int,
) -> bool:
    labels = sorted({int(labels_by_index[int(i)]) for i in indices})
    if len(labels) < 2:
        return False
    available = {label: 0 for label in labels}
    for idx in indices:
        available[int(labels_by_index[int(idx)])] += 1
    needed = {label: 0 for label in labels}
    for pos in range(int(support_size)):
        needed[labels[pos % len(labels)]] += 1
    return all(available[label] >= needed[label] for label in labels)


def _support_eval_indices(
    *,
    target_domain: int,
    target_indices: Sequence[int],
    labels_by_index: Mapping[int, int],
    support_size: int,
    sampling_policy: str,
    support_seed: int,
) -> Tuple[List[int], List[int]]:
    indices = sorted(int(i) for i in target_indices)
    requested = int(support_size)
    if len(indices) < requested + 1:
        return [], []
    split_seed = int(support_seed) + int(target_domain) * 1009
    policy = str(sampling_policy).strip().lower()
    if policy == "class_balanced":
        if _balanced_possible(indices, labels_by_index, requested):
            order = _balanced_order(indices, labels_by_index, split_seed)
        else:
            order = _random_order(indices, split_seed)
    elif policy == "random":
        order = _random_order(indices, split_seed)
    else:
        raise ValueError(f"Unknown sampling policy: {sampling_policy}")
    support = [int(i) for i in order[:requested]]
    support_set = set(support)
    evaluate = [int(i) for i in indices if int(i) not in support_set]
    return support, evaluate


def attach_label_summaries(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    contexts: Dict[str, Tuple[Dict[int, int], Dict[int, List[int]]]] = {}
    out: List[dict] = []
    for row_raw in rows:
        row = dict(row_raw)
        source = str(row.get("source_csv", ""))
        if source not in contexts:
            contexts[source] = _manifest_label_context(Path(source))
        labels_by_index, indices_by_domain = contexts[source]
        target_domain = _to_int(row.get("query_domain", row.get("target_domain", 0)))
        target_indices = indices_by_domain.get(int(target_domain), [])
        if labels_by_index and target_indices:
            support_indices, eval_indices = _support_eval_indices(
                target_domain=int(target_domain),
                target_indices=target_indices,
                labels_by_index=labels_by_index,
                support_size=_to_int(row.get("support_size_requested", 0)),
                sampling_policy=str(row.get("sampling_policy", "random")),
                support_seed=_to_int(row.get("support_seed", 0)),
            )
            support_counts: Dict[int, int] = {}
            eval_counts: Dict[int, int] = {}
            for idx in support_indices:
                label = int(labels_by_index[int(idx)])
                support_counts[label] = support_counts.get(label, 0) + 1
            for idx in eval_indices:
                label = int(labels_by_index[int(idx)])
                eval_counts[label] = eval_counts.get(label, 0) + 1
            support_total = max(sum(support_counts.values()), 1)
            eval_total = max(sum(eval_counts.values()), 1)
            row["support_label_counts_json"] = _json_counts(support_counts)
            row["eval_label_counts_json"] = _json_counts(eval_counts)
            row["support_positive_fraction"] = float(support_counts.get(1, 0) / support_total)
            row["eval_positive_fraction"] = float(eval_counts.get(1, 0) / eval_total)
            row["label_summary_available"] = 1
        else:
            row["support_label_counts_json"] = ""
            row["eval_label_counts_json"] = ""
            row["support_positive_fraction"] = ""
            row["eval_positive_fraction"] = ""
            row["label_summary_available"] = 0
        out.append(row)
    return out


def build_failure_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary_method: str = PRIMARY_METHOD,
    focus_query_domain: int = 3,
    focus_expert: int = 4,
) -> List[dict]:
    primary_rows = attach_label_summaries([r for r in rows if str(r.get("method", "")) == primary_method])
    out: List[dict] = []
    for row in primary_rows:
        pred = _json_map(row.get("predicted_score_by_expert_json", "{}"))
        eval_nelbo = _json_map(row.get("eval_nelbo_by_expert_json", "{}"))
        support_nelbo = _json_map(row.get("support_nelbo_by_expert_json", "{}"))
        if not pred or not eval_nelbo:
            continue
        pred_rank = _rank_map(pred, lower_is_better=True)
        eval_rank = _rank_map(eval_nelbo, lower_is_better=True)
        support_rank = _rank_map(support_nelbo, lower_is_better=True) if support_nelbo else {}
        selected = _to_int(row.get("selected_expert", _argmin_expert(pred)))
        oracle = _to_int(row.get("oracle_expert", _argmin_expert(eval_nelbo)))
        selected_eval = float(eval_nelbo.get(selected, _to_float(row.get("selected_nelbo", 0.0))))
        oracle_eval = float(eval_nelbo.get(oracle, _to_float(row.get("oracle_nelbo", 0.0))))
        selected_support = float(support_nelbo.get(selected, 0.0))
        oracle_support = float(support_nelbo.get(oracle, 0.0))
        selected_pred = float(pred.get(selected, 0.0))
        oracle_pred = float(pred.get(oracle, 0.0))
        hit = int(_to_int(row.get("top1_oracle_hit", 0)) == 1 or selected == oracle)
        selected_eval_rank = int(eval_rank.get(selected, 0))
        selected_support_rank = int(support_rank.get(selected, 0))
        if hit:
            failure_mode = "hit"
        elif selected_support_rank and selected_support_rank <= 2 and selected_eval_rank > 2:
            failure_mode = "support_eval_mismatch"
        elif int(pred_rank.get(selected, 0)) == 1 and selected_eval_rank > 1:
            failure_mode = "learned_score_mismatch"
        else:
            failure_mode = "non_oracle_selection"

        focus_present = int(focus_expert in pred and focus_expert in eval_nelbo)
        focus_eval_delta = float(eval_nelbo.get(focus_expert, 0.0) - oracle_eval) if focus_present else 0.0
        focus_pred_delta = float(pred.get(focus_expert, 0.0) - oracle_pred) if focus_present else 0.0
        focus_support_delta = (
            float(support_nelbo.get(focus_expert, 0.0) - oracle_support)
            if focus_present and support_nelbo
            else 0.0
        )
        out.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "source_csv": row.get("source_csv", ""),
                "run_id": row.get("run_id", ""),
                "method": primary_method,
                "seed": _to_int(row.get("seed", 0)),
                "query_domain": _to_int(row.get("query_domain", 0)),
                "support_seed": _to_int(row.get("support_seed", 0)),
                "support_size_requested": _to_int(row.get("support_size_requested", 0)),
                "sampling_policy": row.get("sampling_policy", ""),
                "candidate_experts": row.get("candidate_experts", ""),
                "selected_expert": selected,
                "oracle_expert": oracle,
                "selected_to_oracle_pair": f"{selected}->{oracle}",
                "top1_oracle_hit": hit,
                "oracle_gap_pct": _to_float(row.get("oracle_gap_pct", row.get("mean_oracle_gap_pct", 0.0))),
                "selected_rank": _to_float(row.get("selected_rank", selected_eval_rank)),
                "selected_pred_rank": int(pred_rank.get(selected, 0)),
                "selected_eval_rank": selected_eval_rank,
                "selected_support_rank": selected_support_rank,
                "confidence_margin": _score_margin(pred),
                "selected_pred_score": selected_pred,
                "oracle_pred_score": oracle_pred,
                "selected_pred_delta_vs_oracle": float(selected_pred - oracle_pred),
                "selected_eval_delta_vs_oracle": float(selected_eval - oracle_eval),
                "selected_support_delta_vs_oracle": float(selected_support - oracle_support),
                "failure_mode": failure_mode,
                "focus_query_domain": int(focus_query_domain),
                "focus_expert": int(focus_expert),
                "focus_query_row": int(_to_int(row.get("query_domain", 0)) == int(focus_query_domain)),
                "focus_expert_present": focus_present,
                "focus_expert_selected": int(selected == int(focus_expert)),
                "focus_pred_rank": int(pred_rank.get(int(focus_expert), 0)),
                "focus_eval_rank": int(eval_rank.get(int(focus_expert), 0)),
                "focus_support_rank": int(support_rank.get(int(focus_expert), 0)),
                "focus_pred_delta_vs_oracle": focus_pred_delta,
                "focus_eval_delta_vs_oracle": focus_eval_delta,
                "focus_support_delta_vs_oracle": focus_support_delta,
                "focus_eval_gap_pct_vs_oracle": _pct_delta(focus_eval_delta, oracle_eval) if focus_present else 0.0,
                "focus_misleading_signal": int(focus_present and focus_pred_delta < 0.0 and focus_eval_delta > 0.0),
                "support_label_counts_json": row.get("support_label_counts_json", ""),
                "eval_label_counts_json": row.get("eval_label_counts_json", ""),
                "support_positive_fraction": row.get("support_positive_fraction", ""),
                "eval_positive_fraction": row.get("eval_positive_fraction", ""),
                "label_summary_available": row.get("label_summary_available", 0),
            }
        )
    return out


def _label_bucket(value: object) -> str:
    if value == "":
        return "unavailable"
    frac = _to_float(value)
    if frac <= 0.25:
        return "low_positive"
    if frac >= 0.75:
        return "high_positive"
    return "mixed"


def build_concentration_rows(failure_rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    expanded: List[dict] = []
    for row in failure_rows:
        enriched = dict(row)
        enriched["support_positive_bucket"] = _label_bucket(row.get("support_positive_fraction", ""))
        enriched["eval_positive_bucket"] = _label_bucket(row.get("eval_positive_fraction", ""))
        expanded.append(enriched)

    group_specs = {
        "query_domain": ("query_domain",),
        "support_size": ("support_size_requested",),
        "selected_expert": ("selected_expert",),
        "selected_to_oracle_pair": ("selected_to_oracle_pair",),
        "support_positive_bucket": ("support_positive_bucket",),
        "eval_positive_bucket": ("eval_positive_bucket",),
        "query_domain_x_support_size": ("query_domain", "support_size_requested"),
    }
    out: List[dict] = []
    for group_type, fields in group_specs.items():
        grouped: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
        for row in expanded:
            key = tuple(str(row.get(field, "")) for field in fields)
            grouped.setdefault(key, []).append(row)
        for key, rows in sorted(grouped.items(), key=lambda item: item[0]):
            n = len(rows)
            failures = [r for r in rows if _to_int(r.get("top1_oracle_hit", 0)) == 0]
            out.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "group_type": group_type,
                    "group_value": "|".join(key),
                    "n_units": n,
                    "failure_count": len(failures),
                    "failure_rate": float(len(failures) / n) if n else 0.0,
                    "mean_oracle_gap_pct": _mean([_to_float(r.get("oracle_gap_pct", 0.0)) for r in rows]),
                    "mean_confidence_margin": _mean([_to_float(r.get("confidence_margin", 0.0)) for r in rows]),
                    "focus_expert_selection_rate": _mean(
                        [_to_float(r.get("focus_expert_selected", 0.0)) for r in rows]
                    ),
                    "focus_misleading_signal_rate": _mean(
                        [_to_float(r.get("focus_misleading_signal", 0.0)) for r in rows]
                    ),
                    "mean_support_positive_fraction": _mean(
                        [
                            _to_float(r.get("support_positive_fraction", 0.0))
                            for r in rows
                            if r.get("support_positive_fraction", "") != ""
                        ]
                    ),
                    "mean_eval_positive_fraction": _mean(
                        [
                            _to_float(r.get("eval_positive_fraction", 0.0))
                            for r in rows
                            if r.get("eval_positive_fraction", "") != ""
                        ]
                    ),
                }
            )
    return out


def _parse_thresholds(raw: str | None, defaults: Sequence[float]) -> Tuple[float, ...]:
    if raw is None or str(raw).strip() == "":
        return tuple(float(v) for v in defaults)
    return tuple(float(v.strip()) for v in str(raw).split(",") if v.strip())


def _policy_score_row(
    *,
    base_row: Mapping[str, Any],
    selected_expert: int,
    score_source_row: Mapping[str, Any],
) -> Dict[str, float]:
    eval_nelbo = _json_map(base_row.get("eval_nelbo_by_expert_json", "{}"))
    scores = _json_map(score_source_row.get("predicted_score_by_expert_json", "{}"))
    if not eval_nelbo:
        return {
            "selected_nelbo": 0.0,
            "oracle_nelbo": 0.0,
            "oracle_gap_pct": 0.0,
            "top1_oracle_hit": 0.0,
            "selected_rank": 0.0,
            "spearman": 0.0,
        }
    oracle_expert = _argmin_expert(eval_nelbo)
    ranks = _rank_map(eval_nelbo, lower_is_better=True)
    selected_nelbo = float(eval_nelbo.get(int(selected_expert), 0.0))
    oracle_nelbo = float(eval_nelbo.get(int(oracle_expert), 0.0))
    score_values = [float(scores.get(expert, 0.0)) for expert in sorted(eval_nelbo)]
    eval_values = [float(eval_nelbo[expert]) for expert in sorted(eval_nelbo)]
    return {
        "selected_nelbo": selected_nelbo,
        "oracle_nelbo": oracle_nelbo,
        "oracle_gap_pct": _pct_delta(selected_nelbo - oracle_nelbo, oracle_nelbo),
        "top1_oracle_hit": float(int(int(selected_expert) == int(oracle_expert))),
        "selected_rank": float(ranks.get(int(selected_expert), 0)),
        "spearman": float(_spearman_corr([-v for v in score_values], [-v for v in eval_values])),
    }


def build_risk_policy_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary_method: str = PRIMARY_METHOD,
    anchor_method: str = ANCHOR_METHOD,
    margin_thresholds: Sequence[float] = DEFAULT_MARGIN_THRESHOLDS,
    support_regret_thresholds: Sequence[float] = DEFAULT_SUPPORT_REGRET_THRESHOLDS,
    focus_query_domain: int = 3,
    focus_expert: int = 4,
) -> Tuple[List[dict], List[dict]]:
    by_key: Dict[Tuple[str, ...], Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        method = str(row.get("method", ""))
        if method not in {primary_method, anchor_method}:
            continue
        by_key.setdefault(_unit_key(row), {})[method] = row

    unit_pairs: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for methods in by_key.values():
        primary = methods.get(primary_method)
        anchor = methods.get(anchor_method)
        if primary is not None and anchor is not None:
            unit_pairs.append((primary, anchor))

    detail_rows: List[dict] = []
    for min_margin in margin_thresholds:
        for max_support_regret_pct in support_regret_thresholds:
            for primary, anchor in unit_pairs:
                pred = _json_map(primary.get("predicted_score_by_expert_json", "{}"))
                support = _json_map(primary.get("support_nelbo_by_expert_json", "{}"))
                if not pred or not support:
                    continue
                learned_selected = _to_int(primary.get("selected_expert", _argmin_expert(pred)))
                anchor_selected = _to_int(anchor.get("selected_expert", 0))
                confidence_margin = _score_margin(pred)
                support_regret_pct = _pct_delta(
                    float(support.get(learned_selected, 0.0)) - float(support.get(anchor_selected, 0.0)),
                    float(support.get(anchor_selected, 0.0)),
                )
                override_candidate = int(learned_selected != anchor_selected)
                accept_override = int(
                    override_candidate
                    and confidence_margin >= float(min_margin)
                    and support_regret_pct <= float(max_support_regret_pct)
                )
                selected_expert = learned_selected if accept_override else anchor_selected
                score_source_row = primary if accept_override else anchor
                policy_scores = _policy_score_row(
                    base_row=primary,
                    selected_expert=selected_expert,
                    score_source_row=score_source_row,
                )
                anchor_scores = _policy_score_row(
                    base_row=primary,
                    selected_expert=anchor_selected,
                    score_source_row=anchor,
                )
                learned_scores = _policy_score_row(
                    base_row=primary,
                    selected_expert=learned_selected,
                    score_source_row=primary,
                )
                selected_source = (
                    "learned_response_override"
                    if accept_override
                    else ("metadata_agreement" if not override_candidate else "metadata_fallback")
                )
                true_delta_vs_anchor = float(policy_scores["selected_nelbo"] - anchor_scores["selected_nelbo"])
                detail_rows.append(
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "policy_name": "metadata_anchor_support_regret_gate",
                        "policy_role": "diagnostic_threshold_grid",
                        "threshold_selection_policy": "grid_report_only_no_target_eval_selection",
                        "primary_method": primary_method,
                        "anchor_method": anchor_method,
                        "min_confidence_margin": float(min_margin),
                        "max_support_regret_pct_vs_anchor": float(max_support_regret_pct),
                        "source_csv": primary.get("source_csv", ""),
                        "run_id": primary.get("run_id", ""),
                        "seed": _to_int(primary.get("seed", 0)),
                        "query_domain": _to_int(primary.get("query_domain", 0)),
                        "support_seed": _to_int(primary.get("support_seed", 0)),
                        "support_size_requested": _to_int(primary.get("support_size_requested", 0)),
                        "sampling_policy": primary.get("sampling_policy", ""),
                        "anchor_selected_expert": anchor_selected,
                        "learned_selected_expert": learned_selected,
                        "selected_expert": selected_expert,
                        "selected_source": selected_source,
                        "override_candidate": override_candidate,
                        "accepted_override": accept_override,
                        "confidence_margin": confidence_margin,
                        "support_regret_pct_vs_anchor": support_regret_pct,
                        "true_nelbo_delta_vs_anchor": true_delta_vs_anchor,
                        "true_harmful_override": int(accept_override and true_delta_vs_anchor > 1e-9),
                        "true_improving_override": int(accept_override and true_delta_vs_anchor < -1e-9),
                        "top1_oracle_hit": policy_scores["top1_oracle_hit"],
                        "selected_rank": policy_scores["selected_rank"],
                        "oracle_gap_pct": policy_scores["oracle_gap_pct"],
                        "spearman": policy_scores["spearman"],
                        "anchor_top1_oracle_hit": anchor_scores["top1_oracle_hit"],
                        "anchor_oracle_gap_pct": anchor_scores["oracle_gap_pct"],
                        "anchor_spearman": anchor_scores["spearman"],
                        "learned_top1_oracle_hit": learned_scores["top1_oracle_hit"],
                        "learned_oracle_gap_pct": learned_scores["oracle_gap_pct"],
                        "learned_spearman": learned_scores["spearman"],
                        "focus_query_domain": int(focus_query_domain),
                        "focus_expert": int(focus_expert),
                        "focus_query_row": int(
                            _to_int(primary.get("query_domain", 0)) == int(focus_query_domain)
                        ),
                        "focus_expert_override_candidate": int(
                            learned_selected == int(focus_expert) and learned_selected != anchor_selected
                        ),
                        "focus_expert_override_accepted": int(
                            accept_override and learned_selected == int(focus_expert)
                        ),
                        "focus_expert_override_blocked": int(
                            (not accept_override)
                            and learned_selected == int(focus_expert)
                            and learned_selected != anchor_selected
                        ),
                    }
                )

    grid_rows: List[dict] = []
    grouped: Dict[Tuple[float, float], List[Mapping[str, Any]]] = {}
    for row in detail_rows:
        grouped.setdefault(
            (
                _to_float(row.get("min_confidence_margin", 0.0)),
                _to_float(row.get("max_support_regret_pct_vs_anchor", 0.0)),
            ),
            [],
        ).append(row)
    for (min_margin, max_support_regret_pct), group in sorted(grouped.items()):
        n = len(group)
        anchor_top1 = _mean([_to_float(r.get("anchor_top1_oracle_hit", 0.0)) for r in group])
        anchor_gap = _mean([_to_float(r.get("anchor_oracle_gap_pct", 0.0)) for r in group])
        anchor_spearman = _mean([_to_float(r.get("anchor_spearman", 0.0)) for r in group])
        learned_top1 = _mean([_to_float(r.get("learned_top1_oracle_hit", 0.0)) for r in group])
        learned_gap = _mean([_to_float(r.get("learned_oracle_gap_pct", 0.0)) for r in group])
        learned_spearman = _mean([_to_float(r.get("learned_spearman", 0.0)) for r in group])
        policy_top1 = _mean([_to_float(r.get("top1_oracle_hit", 0.0)) for r in group])
        policy_gap = _mean([_to_float(r.get("oracle_gap_pct", 0.0)) for r in group])
        policy_spearman = _mean([_to_float(r.get("spearman", 0.0)) for r in group])
        accepted = int(sum(_to_int(r.get("accepted_override", 0)) for r in group))
        harmful = int(sum(_to_int(r.get("true_harmful_override", 0)) for r in group))
        improving = int(sum(_to_int(r.get("true_improving_override", 0)) for r in group))
        focus_candidates = int(sum(_to_int(r.get("focus_expert_override_candidate", 0)) for r in group))
        focus_accepted = int(sum(_to_int(r.get("focus_expert_override_accepted", 0)) for r in group))
        focus_blocked = int(sum(_to_int(r.get("focus_expert_override_blocked", 0)) for r in group))
        grid_rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "policy_name": "metadata_anchor_support_regret_gate",
                "policy_role": "diagnostic_threshold_grid",
                "threshold_selection_policy": "grid_report_only_no_target_eval_selection",
                "min_confidence_margin": float(min_margin),
                "max_support_regret_pct_vs_anchor": float(max_support_regret_pct),
                "n_units": n,
                "accepted_override_count": accepted,
                "override_rate": float(accepted / n) if n else 0.0,
                "harmful_override_count": harmful,
                "harmful_override_rate_among_accepted": float(harmful / accepted) if accepted else 0.0,
                "improving_override_count": improving,
                "improving_override_rate_among_accepted": float(improving / accepted) if accepted else 0.0,
                "top1_oracle_hit": policy_top1,
                "spearman": policy_spearman,
                "mean_oracle_gap_pct": policy_gap,
                "mean_rank": _mean([_to_float(r.get("selected_rank", 0.0)) for r in group]),
                "top1_uplift_vs_metadata_anchor": float(policy_top1 - anchor_top1),
                "spearman_uplift_vs_metadata_anchor": float(policy_spearman - anchor_spearman),
                "oracle_gap_pct_reduction_vs_metadata_anchor": float(anchor_gap - policy_gap),
                "top1_delta_vs_learned_response": float(policy_top1 - learned_top1),
                "spearman_delta_vs_learned_response": float(policy_spearman - learned_spearman),
                "oracle_gap_pct_reduction_vs_learned_response": float(learned_gap - policy_gap),
                "metadata_anchor_top1_oracle_hit": anchor_top1,
                "metadata_anchor_spearman": anchor_spearman,
                "metadata_anchor_mean_oracle_gap_pct": anchor_gap,
                "learned_response_top1_oracle_hit": learned_top1,
                "learned_response_spearman": learned_spearman,
                "learned_response_mean_oracle_gap_pct": learned_gap,
                "focus_expert_override_candidate_count": focus_candidates,
                "focus_expert_override_accepted_count": focus_accepted,
                "focus_expert_override_blocked_count": focus_blocked,
            }
        )
    return detail_rows, grid_rows


def build_summary(
    *,
    failure_rows: Sequence[Mapping[str, Any]],
    grid_rows: Sequence[Mapping[str, Any]],
    primary_method: str,
    anchor_method: str,
    focus_query_domain: int,
    focus_expert: int,
) -> Dict[str, Any]:
    focus_rows = [
        r
        for r in failure_rows
        if _to_int(r.get("query_domain", 0)) == int(focus_query_domain)
    ]
    focus_selected = [
        r
        for r in focus_rows
        if _to_int(r.get("selected_expert", 0)) == int(focus_expert)
    ]
    best_gap = min(
        grid_rows,
        key=lambda r: (
            _to_float(r.get("mean_oracle_gap_pct", 0.0)),
            -_to_float(r.get("top1_oracle_hit", 0.0)),
        ),
        default={},
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "primary_method": primary_method,
        "anchor_method": anchor_method,
        "focus_query_domain": int(focus_query_domain),
        "focus_expert": int(focus_expert),
        "n_primary_units": int(len(failure_rows)),
        "n_focus_query_units": int(len(focus_rows)),
        "n_focus_expert_selections_on_focus_query": int(len(focus_selected)),
        "focus_query_expert_selection_counts": {
            str(expert): int(
                sum(1 for row in focus_rows if _to_int(row.get("selected_expert", 0)) == int(expert))
            )
            for expert in sorted({_to_int(row.get("selected_expert", 0)) for row in focus_rows})
        },
        "focus_query_oracle_expert_counts": {
            str(expert): int(
                sum(1 for row in focus_rows if _to_int(row.get("oracle_expert", 0)) == int(expert))
            )
            for expert in sorted({_to_int(row.get("oracle_expert", 0)) for row in focus_rows})
        },
        "focus_expert_mean_eval_gap_pct_when_selected": _mean(
            [_to_float(row.get("oracle_gap_pct", 0.0)) for row in focus_selected]
        ),
        "focus_expert_misleading_signal_rate_when_selected": _mean(
            [_to_float(row.get("focus_misleading_signal", 0.0)) for row in focus_selected]
        ),
        "diagnostic_best_grid_row_by_gap": dict(best_gap),
        "claim_boundary": (
            "Risk-policy grid rows are diagnostic only. Per-row policy decisions use learned response "
            "scores and disjoint target-support NELBO, but choosing thresholds from these held-out "
            "evaluation outcomes would be target-evaluation leakage."
        ),
        "adoption_requirement": (
            "Predeclare thresholds or select them with inner source-validation folds before applying "
            "to a held-out target center."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_selection_csvs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--primary-method", default=PRIMARY_METHOD)
    parser.add_argument("--anchor-method", default=ANCHOR_METHOD)
    parser.add_argument("--focus-query-domain", type=int, default=3)
    parser.add_argument("--focus-expert", type=int, default=4)
    parser.add_argument("--margin-thresholds", default="")
    parser.add_argument("--support-regret-thresholds", default="")
    args = parser.parse_args(argv)

    rows = read_selection_rows(args.sample_selection_csvs)
    failure_rows = build_failure_rows(
        rows,
        primary_method=str(args.primary_method),
        focus_query_domain=int(args.focus_query_domain),
        focus_expert=int(args.focus_expert),
    )
    concentration_rows = build_concentration_rows(failure_rows)
    detail_rows, grid_rows = build_risk_policy_rows(
        rows,
        primary_method=str(args.primary_method),
        anchor_method=str(args.anchor_method),
        margin_thresholds=_parse_thresholds(args.margin_thresholds, DEFAULT_MARGIN_THRESHOLDS),
        support_regret_thresholds=_parse_thresholds(
            args.support_regret_thresholds, DEFAULT_SUPPORT_REGRET_THRESHOLDS
        ),
        focus_query_domain=int(args.focus_query_domain),
        focus_expert=int(args.focus_expert),
    )
    summary = build_summary(
        failure_rows=failure_rows,
        grid_rows=grid_rows,
        primary_method=str(args.primary_method),
        anchor_method=str(args.anchor_method),
        focus_query_domain=int(args.focus_query_domain),
        focus_expert=int(args.focus_expert),
    )

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "failure_localization.csv", failure_rows)
    _write_csv(out_dir / "failure_concentration.csv", concentration_rows)
    _write_csv(out_dir / "risk_policy_detail.csv", detail_rows)
    _write_csv(out_dir / "risk_policy_grid.csv", grid_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "failure_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
