"""Run MIDOG++ real-feature classifier reference surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .classifier_grid import add_classifier_grid_arguments, classifier_specs_from_args, csv_values
from .midogpp_real_feature_classifier import (
    run_midogpp_real_feature_source_inner_classifier_tuning,
)
from .protocol import ProtocolError
from .real_feature_frame import load_midogpp_real_feature_frame
from .schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="surface", required=True)
    tune = subparsers.add_parser("tune", help="Run the historical source-inner tuning reference.")
    _add_tune_arguments(tune)
    matched = subparsers.add_parser(
        "matched-reference",
        help="Run the eligible-only predict-policy matched reference v2.",
    )
    matched.add_argument("--config", required=True)
    matched.add_argument("--artifact-root", default=None)
    fixed_risk = subparsers.add_parser(
        "fixed-c-risk-diagnostic",
        help="Run the fixed-C four-arm risk-weighting diagnostic.",
    )
    fixed_risk.add_argument("--config", required=True)
    fixed_risk.add_argument("--artifact-root", default=None)
    alignment = subparsers.add_parser(
        "conditional-logit-alignment",
        help="Run the nested conditional-logit alignment diagnostic.",
    )
    alignment.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.surface == "matched-reference":
        from .matched_reference import load_matched_reference_config, run_matched_reference

        config = load_matched_reference_config(args.config)
        output = run_matched_reference(
            config,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(output)
        return 0
    if args.surface == "fixed-c-risk-diagnostic":
        from .fixed_c_risk_diagnostic import (
            load_fixed_c_risk_config,
            run_fixed_c_risk_diagnostic,
        )

        config = load_fixed_c_risk_config(args.config)
        output = run_fixed_c_risk_diagnostic(
            config,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        print(output)
        return 0
    if args.surface == "conditional-logit-alignment":
        from .conditional_logit_alignment import (
            load_conditional_logit_alignment_config,
            run_conditional_logit_alignment,
        )

        config = load_conditional_logit_alignment_config(args.config)
        output = run_conditional_logit_alignment(config)
        print(output)
        return 0
    if args.preflight_only:
        frame = load_midogpp_real_feature_frame(
            manifest_path=Path(args.manifest),
            feature_cache_path=Path(args.feature_cache),
            expected_feature_dim=int(args.expected_feature_dim),
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "manifest_hash": frame.manifest_hash,
                    "feature_cache_hash": frame.feature_cache_hash,
                    "eligible_centers": list(frame.eligible_centers),
                    "expected_feature_dim": int(frame.expected_feature_dim),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    outputs = run_midogpp_real_feature_source_inner_classifier_tuning(
        manifest_path=Path(args.manifest),
        feature_cache_path=Path(args.feature_cache),
        out_dir=Path(args.out_dir),
        candidate_specs=classifier_specs_from_args(args),
        heldout_centers=csv_values(args.heldout_centers),
        experiment_seed=int(args.experiment_seed),
        classifier_seed=int(args.classifier_seed),
        expected_feature_dim=int(args.expected_feature_dim),
    )
    for label, path in outputs.as_dict().items():
        print(f"Wrote {label}: {path}")
    return 0


def _add_tune_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--experiment-seed", type=int, default=42)
    parser.add_argument("--classifier-seed", type=int, default=23)
    parser.add_argument("--heldout-centers", default=",".join(MIDOGPP_ELIGIBLE_CENTERS))
    parser.add_argument("--expected-feature-dim", type=int, default=2560)
    parser.add_argument("--preflight-only", action="store_true")
    add_classifier_grid_arguments(
        parser,
        default_c_grid="0.01,0.1,1.0,10.0,100.0",
        default_penalties="l2",
        default_solvers="lbfgs",
        default_class_weights="none,balanced",
        default_max_iters="2000,5000",
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProtocolError as exc:
        raise SystemExit(str(exc)) from exc
