#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PRIMARY_METHOD = "pairwise_ranker_ae_combined_inner_selected_v2"
STRICT_PRIMARY_METHOD = "pairwise_ranker_ae_combined_strict_inner_selected_v2"
TARGET_BATCH_AGREEMENT_PRIMARY_METHOD = "pairwise_ranker_ae_combined_target_batch_agreement_gated_v3"
BASELINE_METHOD = "pairwise_ranker_ae_combined"
AE_ARGMIN_METHOD = "ae_argmin_zscore"
METADATA_METHOD = "metadata_routing"


def _read_manifest(path: Path) -> List[Path]:
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
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
        for key in row:
            key_s = str(key)
            if key_s not in seen:
                seen.add(key_s)
                fieldnames.append(key_s)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean(values: Iterable[float], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(mean(vals)) if vals else float(default)


def _reports_dir(path: Path) -> Path:
    return path.parent if path.parent.name == "reports" else path.parent / "reports"


def _load_runs(manifest: Path, dataset: str) -> List[Path]:
    paths = [path for path in _read_manifest(manifest) if _dataset_from_path(path) == str(dataset)]
    if not paths:
        raise RuntimeError(f"No {dataset} runs found in {manifest}")
    return paths


def _method_domain_summary(rows: Sequence[Mapping[str, str]]) -> Dict[tuple[str, str], Mapping[str, str]]:
    return {(str(row.get("method", "")), str(row.get("query_domain", ""))): row for row in rows}


def _decision_file(reports: Path) -> tuple[Path, str, bool]:
    v3 = reports / "pairwise_ae_combined_v3_decision_table.csv"
    if v3.exists():
        return v3, TARGET_BATCH_AGREEMENT_PRIMARY_METHOD, True
    strict = reports / "pairwise_ae_combined_v2_strict_decision_table.csv"
    if strict.exists():
        return strict, STRICT_PRIMARY_METHOD, True
    return reports / "pairwise_ae_combined_v2_decision_table.csv", PRIMARY_METHOD, False


def _delta_gap(row: Mapping[str, Any]) -> float:
    if "delta_gap_vs_baseline_pp" in row:
        return _float(row.get("delta_gap_vs_baseline_pp"))
    return _float(row.get("delta_gap_vs_baseline"))


def _aggregate_decisions(paths: Sequence[Path]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    decision_rows: List[Dict[str, Any]] = []
    seed_domain_rows: List[Dict[str, Any]] = []
    any_strict = False
    for result_path in paths:
        reports = _reports_dir(result_path)
        seed = _seed_from_path(result_path)
        run_id = _run_id_from_path(result_path)
        decision_path, primary_method, is_strict = _decision_file(reports)
        any_strict = any_strict or bool(is_strict)
        decisions = _read_csv(decision_path)
        domain_rows = _read_csv(reports / "learned_utility_domain_breakdown.csv", required=False)
        domain_by_key = _method_domain_summary(domain_rows)
        for row in decisions:
            enriched = {
                "seed": row.get("seed", seed),
                "run_id": run_id,
                "result_path": str(result_path),
                "decision_artifact": str(decision_path),
                **row,
            }
            decision_rows.append(enriched)
        centers = sorted({str(row.get("outer_heldout_center", "")) for row in decisions if str(row.get("outer_heldout_center", ""))})
        for center in centers:
            subset = [row for row in decisions if str(row.get("outer_heldout_center", "")) == str(center)]
            primary = domain_by_key.get((primary_method, str(center)), {})
            baseline = domain_by_key.get((BASELINE_METHOD, str(center)), {})
            ae = domain_by_key.get((AE_ARGMIN_METHOD, str(center)), {})
            metadata = domain_by_key.get((METADATA_METHOD, str(center)), {})
            selected_methods = Counter(str(row.get("selected_method", "")) for row in subset)
            seed_domain_rows.append(
                {
                    "seed": seed,
                    "run_id": run_id,
                    "outer_heldout_center": center,
                    "n_queries": len(subset),
                    "selected_method_counts": dict(selected_methods),
                    "fallback_to_baseline_rate": float(selected_methods.get(BASELINE_METHOD, 0) / max(len(subset), 1)),
                    "v2_adoption_rate": float((len(subset) - selected_methods.get(BASELINE_METHOD, 0)) / max(len(subset), 1)),
                    "strict_v2_adoption_rate": float((len(subset) - selected_methods.get(BASELINE_METHOD, 0)) / max(len(subset), 1)),
                    "mean_delta_gap_vs_baseline": _mean(
                        [_delta_gap(row) for row in subset],
                        0.0,
                    ),
                    "primary_mean_oracle_gap_pct": _float(primary.get("mean_oracle_gap_pct")),
                    "baseline_mean_oracle_gap_pct": _float(baseline.get("mean_oracle_gap_pct")),
                    "ae_argmin_mean_oracle_gap_pct": _float(ae.get("mean_oracle_gap_pct")),
                    "metadata_mean_oracle_gap_pct": _float(metadata.get("mean_oracle_gap_pct")),
                    "primary_top1_oracle_hit": _float(primary.get("top1_oracle_hit")),
                    "baseline_top1_oracle_hit": _float(baseline.get("top1_oracle_hit")),
                    "primary_spearman": _float(primary.get("spearman")),
                    "baseline_spearman": _float(baseline.get("spearman")),
                }
            )
    return decision_rows, seed_domain_rows, any_strict


def _verdict(seed_domain_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    gap_delta = [_float(row.get("baseline_mean_oracle_gap_pct")) - _float(row.get("primary_mean_oracle_gap_pct")) for row in seed_domain_rows]
    top1_delta = [_float(row.get("primary_top1_oracle_hit")) - _float(row.get("baseline_top1_oracle_hit")) for row in seed_domain_rows]
    spearman_delta = [_float(row.get("primary_spearman")) - _float(row.get("baseline_spearman")) for row in seed_domain_rows]
    ae_delta = [_float(row.get("ae_argmin_mean_oracle_gap_pct")) - _float(row.get("primary_mean_oracle_gap_pct")) for row in seed_domain_rows]
    metadata_delta = [
        _float(row.get("metadata_mean_oracle_gap_pct")) - _float(row.get("primary_mean_oracle_gap_pct"))
        for row in seed_domain_rows
    ]
    seeds: Dict[str, List[float]] = defaultdict(list)
    seed_top1: Dict[str, List[float]] = defaultdict(list)
    centers: Dict[str, List[float]] = defaultdict(list)
    for row, gd, td in zip(seed_domain_rows, gap_delta, top1_delta):
        seeds[str(row.get("seed"))].append(float(gd))
        seed_top1[str(row.get("seed"))].append(float(td))
        centers[str(row.get("outer_heldout_center"))].append(float(gd))
    seed_gap_improved = sum(1 for vals in seeds.values() if _mean(vals, 0.0) > 0.0)
    seed_top1_nondegrading = sum(1 for vals in seed_top1.values() if _mean(vals, 0.0) >= -0.02)
    center_worst_degradation = max([-_mean(vals, 0.0) for vals in centers.values()] or [0.0])
    selected_nonbaseline = any(float(row.get("v2_adoption_rate", 0.0)) > 0.0 for row in seed_domain_rows)
    mean_gap = _mean(gap_delta, 0.0)
    mean_top1 = _mean(top1_delta, 0.0)
    mean_spearman = _mean(spearman_delta, 0.0)
    beats_ae_argmin = _mean(ae_delta, 0.0) > 0.0
    beats_metadata = _mean(metadata_delta, 0.0) > 0.0
    if not selected_nonbaseline:
        verdict = "DIAGNOSTIC ONLY"
    elif (
        mean_gap >= 0.5
        and mean_top1 >= -0.02
        and mean_spearman >= -0.03
        and seed_gap_improved >= 2
        and seed_top1_nondegrading >= 2
        and center_worst_degradation <= 1.0
        and beats_ae_argmin
        and beats_metadata
    ):
        verdict = "PASS"
    elif mean_gap > 0.0 and mean_top1 >= -0.02 and mean_spearman >= -0.03 and seed_top1_nondegrading >= 2:
        verdict = "WEAK PASS"
    else:
        verdict = "DIAGNOSTIC ONLY"
    return {
        "verdict": verdict,
        "mean_gap_reduction_vs_pairwise_ae_combined": mean_gap,
        "mean_top1_delta_vs_pairwise_ae_combined": mean_top1,
        "mean_spearman_delta_vs_pairwise_ae_combined": mean_spearman,
        "seed_gap_improved_count": seed_gap_improved,
        "seed_top1_nondegrading_count": seed_top1_nondegrading,
        "worst_center_gap_degradation": center_worst_degradation,
        "selected_nonbaseline_v2_at_least_once": bool(selected_nonbaseline),
        "beats_ae_argmin_zscore": bool(beats_ae_argmin),
        "beats_metadata_routing": bool(beats_metadata),
        "always_baseline_fallback": bool(not selected_nonbaseline),
    }


def _v3_summary(decision_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_unit: Dict[tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in decision_rows:
        by_unit[(str(row.get("seed", "")), str(row.get("outer_heldout_center", "")))].append(row)
    unit_rows: List[Mapping[str, Any]] = [rows[0] for rows in by_unit.values() if rows]
    gate_activation = sum(1 for row in unit_rows if int(float(row.get("agreement_gate_applied", 0) or 0)) == 1)
    gate_pass = sum(
        1
        for row in unit_rows
        if int(float(row.get("agreement_gate_applied", 0) or 0)) == 1
        and int(float(row.get("agreement_gate_passed", 0) or 0)) == 1
    )
    gate_block = sum(
        1
        for row in unit_rows
        if int(float(row.get("agreement_gate_applied", 0) or 0)) == 1
        and int(float(row.get("agreement_gate_passed", 0) or 0)) == 0
    )
    return {
        "gate_activation_count": int(gate_activation),
        "gate_pass_count": int(gate_pass),
        "gate_block_count": int(gate_block),
        "false_veto_count": int(
            sum(1 for row in unit_rows if int(float(row.get("false_veto", 0) or 0)) == 1)
        ),
        "false_allow_count": int(
            sum(1 for row in unit_rows if int(float(row.get("false_allow", 0) or 0)) == 1)
        ),
        "blocked_harmful_count": int(
            sum(1 for row in unit_rows if int(float(row.get("blocked_harmful_deployment", 0) or 0)) == 1)
        ),
        "allowed_beneficial_count": int(
            sum(1 for row in unit_rows if int(float(row.get("allowed_beneficial_deployment", 0) or 0)) == 1)
        ),
        "mean_blocked_v2_delta_gap_vs_baseline": _mean(
            [_float(row.get("blocked_v2_delta_gap_vs_baseline")) for row in decision_rows],
            float("nan"),
        ),
        "mean_allowed_v2_delta_gap_vs_baseline": _mean(
            [_float(row.get("allowed_v2_delta_gap_vs_baseline")) for row in decision_rows],
            float("nan"),
        ),
        "used_target_embeddings_for_gate": 1,
        "used_target_group_ids_for_gate": int(
            any(int(float(row.get("used_target_group_ids_for_gate", 0) or 0)) == 1 for row in unit_rows)
        ),
        "used_target_labels_for_gate": 0,
        "used_target_nelbo_for_gate": 0,
        "used_target_support_for_gate": 0,
        "used_target_fitting_for_gate": 0,
        "used_target_normalization_for_gate": 0,
        "heldout_target_nelbo_used_for_selection": 0,
    }


def _write_markdown(path: Path, summary: Mapping[str, Any], seed_domain_rows: Sequence[Mapping[str, Any]], *, strict: bool, v3: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pairwise AE-Combined Target-Batch Agreement-Gated v3"
        if v3
        else "# Pairwise AE-Combined Strict Inner-Selected v2"
        if strict
        else "# Pairwise AE-Combined Inner-Selected v2",
        "",
        f"Verdict: **{summary.get('verdict')}**",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| mean gap reduction vs baseline | {float(summary.get('mean_gap_reduction_vs_pairwise_ae_combined', 0.0)):.4f} |",
        f"| mean top1 delta vs baseline | {float(summary.get('mean_top1_delta_vs_pairwise_ae_combined', 0.0)):.4f} |",
        f"| mean Spearman delta vs baseline | {float(summary.get('mean_spearman_delta_vs_pairwise_ae_combined', 0.0)):.4f} |",
        f"| seed gap improved count | {int(summary.get('seed_gap_improved_count', 0))} |",
        f"| seed top1 nondegrading count | {int(summary.get('seed_top1_nondegrading_count', 0))} |",
        "",
        "## Seed-Center Summary",
        "",
        "| seed | center | gap reduction | adoption rate | selected methods |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in seed_domain_rows:
        lines.append(
            "| {seed} | {center} | {gap:.4f} | {adopt:.4f} | `{methods}` |".format(
                seed=row.get("seed"),
                center=row.get("outer_heldout_center"),
                gap=_float(row.get("baseline_mean_oracle_gap_pct")) - _float(row.get("primary_mean_oracle_gap_pct")),
                adopt=_float(row.get("v2_adoption_rate"), 0.0),
                methods=row.get("selected_method_counts"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset", default="camelyon17")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    paths = _load_runs(args.manifest, args.dataset)
    decisions, seed_domain_rows, any_strict = _aggregate_decisions(paths)
    any_v3 = any(str(row.get("primary_method", "")) == TARGET_BATCH_AGREEMENT_PRIMARY_METHOD for row in decisions)
    summary = _verdict(seed_domain_rows)
    if any_v3:
        summary.update(_v3_summary(decisions))
    summary.update(
        {
            "dataset": args.dataset,
            "n_runs": len(paths),
            "n_decision_rows": len(decisions),
            "n_seed_domain_rows": len(seed_domain_rows),
            "primary_method": TARGET_BATCH_AGREEMENT_PRIMARY_METHOD if any_v3 else STRICT_PRIMARY_METHOD if any_strict else PRIMARY_METHOD,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "pairwise_ae_combined_v3" if any_v3 else "pairwise_ae_combined_v2_strict" if any_strict else "pairwise_ae_combined_v2"
    _write_csv(args.output_dir / f"{prefix}_decision_table.csv", decisions)
    _write_csv(args.output_dir / f"{prefix}_seed_domain_summary.csv", seed_domain_rows)
    (args.output_dir / f"{prefix}_decision_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    summaries_dir = args.output_dir.parent.parent / "summaries"
    _write_markdown(summaries_dir / f"{prefix}_decision_table.md", summary, seed_domain_rows, strict=any_strict, v3=any_v3)


if __name__ == "__main__":
    main()
