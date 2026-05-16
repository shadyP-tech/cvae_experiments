#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_compatibility_decision_table import (  # noqa: E402
    _aggregate,
    _load_manifest,
    _read_rows,
    _write_csv,
    _write_md,
)
from scripts.compatibility_stability import (  # noqa: E402
    LEGACY_STD_POLICY,
    SIGN_CI_POLICY,
    validate_decision_policy_version,
)


_RESIDUAL_METHODS = {
    "metadata_routing",
    "candidate_oracle_routing",
    "unconstrained_learned_reference",
    "metadata_residual_argmax",
    "metadata_residual_thresholded",
    "metadata_residual_group_robust",
    "metadata_residual_thresholded_safe_v2",
    "metadata_residual_group_robust_safe_v2",
    "metadata_residual_inner_selected",
}


def _filter_residual_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = [dict(r) for r in rows if str(r.get("method", "")) in _RESIDUAL_METHODS]
    if not any(str(r.get("method", "")) == "metadata_routing" for r in out):
        raise RuntimeError("metadata_routing baseline is required for residual decision-table aggregation")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build residual-routing decision table from learned utility run manifest."
    )
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
        default=Path("results/comparison_tables/residual_routing_decision_table.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/summaries/residual_routing_decision_table.md"),
    )
    args = parser.parse_args()

    decision_policy_version = validate_decision_policy_version(args.decision_policy_version)
    result_paths = _load_manifest(args.manifest)
    rows = _read_rows(result_paths, uplift_reference_method=str(args.uplift_reference_method))
    rows = _filter_residual_rows(rows)
    if not rows:
        raise RuntimeError("No residual routing rows could be read from result json files.")

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
