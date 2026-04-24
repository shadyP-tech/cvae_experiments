#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _to_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    clean = [float(v) for v in vals if math.isfinite(float(v))]
    if not clean:
        return 0.0, 0.0
    mu = sum(clean) / len(clean)
    var = sum((v - mu) ** 2 for v in clean) / len(clean)
    return float(mu), float(math.sqrt(max(var, 0.0)))


def _method_key(row: dict) -> str:
    method = str(row.get("method", ""))
    feature_set = str(row.get("feature_set", ""))
    if method == "metadata_routing":
        return "metadata_routing"
    return f"{method}__{feature_set}"


def _read_rows(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    if not rows:
        raise RuntimeError(f"Raw CSV is empty: {path}")
    return rows


def _select_rows(rows: Sequence[dict], only_feature_set_b: bool) -> List[dict]:
    out: List[dict] = []
    for r in rows:
        m = str(r.get("method", ""))
        fs = str(r.get("feature_set", ""))
        if m == "metadata_routing":
            out.append(r)
            continue
        if only_feature_set_b and fs != "B":
            continue
        out.append(r)
    return out


def _build_domain_records(rows: Sequence[dict], uplift_reference_method: str) -> List[dict]:
    grouped: Dict[Tuple[str, str, str, str, str], List[dict]] = {}
    for r in rows:
        key = (
            str(r.get("dataset_name", "")),
            str(r.get("backbone_type", "")),
            str(r.get("run_id", "")),
            str(r.get("variant", "")),
            str(r.get("heldout_query_domain", "")),
        )
        grouped.setdefault(key, []).append(r)

    out: List[dict] = []
    for key, vals in grouped.items():
        ds, bb, run_id, variant, heldout = key
        base = None
        for r in vals:
            if str(r.get("method", "")) == str(uplift_reference_method):
                base = r
                break
        if base is None:
            continue

        b_top1 = _to_float(base.get("top1_agreement_with_best_expert", 0.0))
        b_spearman = _to_float(base.get("spearman_similarity_vs_neg_nelbo", 0.0))
        b_gap = _to_float(base.get("metadata_to_oracle_gap", 0.0))

        for r in vals:
            mkey = _method_key(r)
            top1 = _to_float(r.get("top1_agreement_with_best_expert", 0.0))
            spearman = _to_float(r.get("spearman_similarity_vs_neg_nelbo", 0.0))
            gap = _to_float(r.get("metadata_to_oracle_gap", 0.0))
            out.append(
                {
                    "dataset_name": ds,
                    "backbone_type": bb,
                    "run_id": run_id,
                    "variant": variant,
                    "heldout_query_domain": heldout,
                    "method_key": mkey,
                    "top1": top1,
                    "spearman": spearman,
                    "oracle_gap": gap,
                    "top1_uplift_vs_metadata": top1 - b_top1,
                    "spearman_uplift_vs_metadata": spearman - b_spearman,
                    "oracle_gap_reduction_vs_metadata": b_gap - gap,
                }
            )
    return out


def _aggregate_domains(domain_records: Sequence[dict]) -> List[dict]:
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for r in domain_records:
        key = (str(r["method_key"]), str(r["heldout_query_domain"]))
        groups.setdefault(key, []).append(r)

    out: List[dict] = []
    for (method_key, heldout), vals in groups.items():
        top1_mu, top1_std = _mean_std([_to_float(v["top1"]) for v in vals])
        sp_mu, sp_std = _mean_std([_to_float(v["spearman"]) for v in vals])
        gap_mu, gap_std = _mean_std([_to_float(v["oracle_gap"]) for v in vals])
        up1_mu, up1_std = _mean_std([_to_float(v["top1_uplift_vs_metadata"]) for v in vals])
        usp_mu, usp_std = _mean_std([_to_float(v["spearman_uplift_vs_metadata"]) for v in vals])
        ugr_mu, ugr_std = _mean_std([_to_float(v["oracle_gap_reduction_vs_metadata"]) for v in vals])
        out.append(
            {
                "method_key": method_key,
                "heldout_query_domain": heldout,
                "n_points": int(len(vals)),
                "top1_mean": top1_mu,
                "top1_std": top1_std,
                "spearman_mean": sp_mu,
                "spearman_std": sp_std,
                "oracle_gap_mean": gap_mu,
                "oracle_gap_std": gap_std,
                "top1_uplift_vs_metadata_mean": up1_mu,
                "top1_uplift_vs_metadata_std": up1_std,
                "spearman_uplift_vs_metadata_mean": usp_mu,
                "spearman_uplift_vs_metadata_std": usp_std,
                "oracle_gap_reduction_vs_metadata_mean": ugr_mu,
                "oracle_gap_reduction_vs_metadata_std": ugr_std,
            }
        )

    out.sort(key=lambda r: (str(r["method_key"]), str(r["heldout_query_domain"])))
    return out


def _method_summary(
    domain_rows: Sequence[dict],
    *,
    top1_collapse_threshold: float,
    spearman_collapse_threshold: float,
    gap_collapse_threshold: float,
) -> List[dict]:
    groups: Dict[str, List[dict]] = {}
    for r in domain_rows:
        groups.setdefault(str(r["method_key"]), []).append(r)

    out: List[dict] = []
    for method_key, vals in groups.items():
        top1_list = [(_to_float(v["top1_uplift_vs_metadata_mean"]), str(v["heldout_query_domain"])) for v in vals]
        sp_list = [(_to_float(v["spearman_uplift_vs_metadata_mean"]), str(v["heldout_query_domain"])) for v in vals]
        gap_list = [(_to_float(v["oracle_gap_reduction_vs_metadata_mean"]), str(v["heldout_query_domain"])) for v in vals]

        worst_top1, worst_top1_domain = min(top1_list, key=lambda t: t[0])
        worst_sp, worst_sp_domain = min(sp_list, key=lambda t: t[0])
        worst_gap, worst_gap_domain = min(gap_list, key=lambda t: t[0])

        collapsed_domains: List[str] = []
        for v in vals:
            top1 = _to_float(v["top1_uplift_vs_metadata_mean"])
            sp = _to_float(v["spearman_uplift_vs_metadata_mean"])
            gap = _to_float(v["oracle_gap_reduction_vs_metadata_mean"])
            if top1 < float(top1_collapse_threshold) or sp < float(spearman_collapse_threshold) or gap < float(gap_collapse_threshold):
                collapsed_domains.append(str(v["heldout_query_domain"]))

        out.append(
            {
                "method_key": method_key,
                "n_domains": int(len(vals)),
                "mean_top1_uplift_vs_metadata": _mean_std([t[0] for t in top1_list])[0],
                "mean_spearman_uplift_vs_metadata": _mean_std([t[0] for t in sp_list])[0],
                "mean_oracle_gap_reduction_vs_metadata": _mean_std([t[0] for t in gap_list])[0],
                "worst_top1_uplift_vs_metadata": float(worst_top1),
                "worst_top1_domain": worst_top1_domain,
                "worst_spearman_uplift_vs_metadata": float(worst_sp),
                "worst_spearman_domain": worst_sp_domain,
                "worst_oracle_gap_reduction_vs_metadata": float(worst_gap),
                "worst_oracle_gap_domain": worst_gap_domain,
                "collapse_flag": int(len(collapsed_domains) > 0),
                "n_collapsed_domains": int(len(collapsed_domains)),
                "collapsed_domains": "|".join(collapsed_domains),
            }
        )

    out.sort(key=lambda r: (int(r["collapse_flag"]), str(r["method_key"])))
    return out


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _write_md(
    path: Path,
    *,
    dataset_name: str,
    method_rows: Sequence[dict],
    top1_collapse_threshold: float,
    spearman_collapse_threshold: float,
    gap_collapse_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# LOQDO Stress Test: {dataset_name}")
    lines.append("")
    lines.append("Worst-domain collapse is diagnostic-only and does not override decision-table tiers.")
    lines.append("")
    lines.append(
        f"- Collapse rule: top1_uplift_vs_metadata < {top1_collapse_threshold}, or spearman_uplift_vs_metadata < {spearman_collapse_threshold}, or oracle_gap_reduction_vs_metadata < {gap_collapse_threshold} on any heldout domain."
    )
    lines.append("- Threshold asymmetry rationale: oracle-gap-reduction uses tighter absolute bound due to smaller empirical dynamic range.")
    lines.append("")
    lines.append("| Method | Collapse | n collapsed domains | mean top1 uplift | mean spearman uplift | mean gap reduction | worst top1 (domain) | worst spearman (domain) | worst gap reduction (domain) |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|---|")
    for r in method_rows:
        lines.append(
            "| {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} ({}) | {:.4f} ({}) | {:.4f} ({}) |".format(
                r["method_key"],
                int(r["collapse_flag"]),
                int(r["n_collapsed_domains"]),
                _to_float(r["mean_top1_uplift_vs_metadata"]),
                _to_float(r["mean_spearman_uplift_vs_metadata"]),
                _to_float(r["mean_oracle_gap_reduction_vs_metadata"]),
                _to_float(r["worst_top1_uplift_vs_metadata"]),
                r["worst_top1_domain"],
                _to_float(r["worst_spearman_uplift_vs_metadata"]),
                r["worst_spearman_domain"],
                _to_float(r["worst_oracle_gap_reduction_vs_metadata"]),
                r["worst_oracle_gap_domain"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Build LOQDO per-domain stress-test artifacts with collapse diagnostics.")
    p.add_argument("--raw-csv", type=Path, required=True)
    p.add_argument("--output-domain-csv", type=Path, required=True)
    p.add_argument("--output-method-csv", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    p.add_argument("--uplift-reference-method", type=str, default="metadata_routing")
    p.add_argument("--only-feature-set-b", action="store_true")
    p.add_argument("--top1-collapse-threshold", type=float, default=-0.10)
    p.add_argument("--spearman-collapse-threshold", type=float, default=-0.10)
    p.add_argument("--gap-collapse-threshold", type=float, default=-0.005)
    args = p.parse_args()

    rows = _read_rows(args.raw_csv)
    rows = _select_rows(rows, only_feature_set_b=bool(args.only_feature_set_b))
    domain_records = _build_domain_records(rows, uplift_reference_method=str(args.uplift_reference_method))
    if not domain_records:
        raise RuntimeError("No domain records built; check raw CSV and filters.")

    domain_rows = _aggregate_domains(domain_records)
    method_rows = _method_summary(
        domain_rows,
        top1_collapse_threshold=float(args.top1_collapse_threshold),
        spearman_collapse_threshold=float(args.spearman_collapse_threshold),
        gap_collapse_threshold=float(args.gap_collapse_threshold),
    )

    _write_csv(args.output_domain_csv, domain_rows)
    _write_csv(args.output_method_csv, method_rows)
    dataset_name = str(domain_records[0].get("dataset_name", "unknown"))
    _write_md(
        args.output_md,
        dataset_name=dataset_name,
        method_rows=method_rows,
        top1_collapse_threshold=float(args.top1_collapse_threshold),
        spearman_collapse_threshold=float(args.spearman_collapse_threshold),
        gap_collapse_threshold=float(args.gap_collapse_threshold),
    )

    print(f"Wrote {args.output_domain_csv}")
    print(f"Wrote {args.output_method_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
