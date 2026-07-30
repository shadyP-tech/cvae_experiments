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
    physical_multiscale = subparsers.add_parser(
        "physical-multiscale-center-pooling-pilot",
        help="Run the locked non-adoptive physical-multiscale representation pilot.",
    )
    physical_multiscale.add_argument("--config", required=True)
    annotation_local = subparsers.add_parser(
        "physical-multiscale-annotation-local-pooling-pilot",
        help="Run a locked versioned annotation-local representation pilot.",
    )
    annotation_local.add_argument("--config", required=True)
    uniform_b_replay = subparsers.add_parser(
        "uniform-b-v3-replay",
        help="Run the non-adoptive Stage-90 retrospective uniform-B replay.",
    )
    uniform_b_replay.add_argument("--config", required=True)
    uniform_b_cache = subparsers.add_parser(
        "build-uniform-b-v3-test-cache",
        help="Build the immutable B cache for prospective test confirmation.",
    )
    uniform_b_cache.add_argument("--config", required=True)
    uniform_b_confirmation = subparsers.add_parser(
        "uniform-b-v3-confirmation",
        help="Run prospective within-center confirmation of uniform B.",
    )
    uniform_b_confirmation.add_argument("--config", required=True)
    uniform_b_canonical_cache = subparsers.add_parser(
        "build-uniform-b-canonical-train-cache",
        help="Standardize the reviewed B train shards as a canonical cache.",
    )
    uniform_b_canonical_cache.add_argument("--config", required=True)
    uniform_b_canonical_reference = subparsers.add_parser(
        "uniform-b-canonical-reference",
        help="Run the separately reviewed Stage-10 Uniform-B reference.",
    )
    uniform_b_canonical_reference.add_argument("--config", required=True)
    uniform_b_nonlinear = subparsers.add_parser(
        "uniform-b-nystroem-nonlinear-probe",
        help="Run the bounded Stage-90 nonlinear-boundary probe on canonical B.",
    )
    uniform_b_nonlinear.add_argument("--config", required=True)
    uniform_b_robust = subparsers.add_parser(
        "uniform-b-robust-interaction-probe",
        help="Run the Stage-90 robust-Nyström versus bilinear B+ probe.",
    )
    uniform_b_robust.add_argument("--config", required=True)
    uniform_b_constrained = subparsers.add_parser(
        "uniform-b-sens-spec-constrained-nystroem-probe",
        help="Run the fixed-threshold constrained nonlinear-capacity B+ probe.",
    )
    uniform_b_constrained.add_argument("--config", required=True)
    uniform_b_spatial_cache = subparsers.add_parser(
        "build-uniform-b-spatial-cache",
        help="Build the immutable dual-GPU central-token and B-spatial cache.",
    )
    uniform_b_spatial_cache.add_argument("--config", required=True)
    uniform_b_spatial = subparsers.add_parser(
        "uniform-b-spatial-probe",
        help="Run the frozen-capacity Stage-90 B-spatial representation diagnostic.",
    )
    uniform_b_spatial.add_argument("--config", required=True)
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
    if args.surface in {
        "physical-multiscale-center-pooling-pilot",
        "physical-multiscale-annotation-local-pooling-pilot",
    }:
        from .physical_multiscale_center_pooling import (
            load_physical_multiscale_pilot_config,
            run_physical_multiscale_center_pooling_pilot,
        )

        config = load_physical_multiscale_pilot_config(args.config)
        output = run_physical_multiscale_center_pooling_pilot(config)
        print(output)
        return 0
    if args.surface == "uniform-b-v3-replay":
        from .uniform_b_replay import (
            load_uniform_b_replay_config,
            run_uniform_b_replay,
        )

        config = load_uniform_b_replay_config(args.config)
        output = run_uniform_b_replay(config)
        print(output)
        return 0
    if args.surface == "build-uniform-b-v3-test-cache":
        from .uniform_b_confirmation import (
            build_uniform_b_test_cache,
            load_uniform_b_test_cache_config,
        )

        output = build_uniform_b_test_cache(
            load_uniform_b_test_cache_config(args.config)
        )
        print(output)
        return 0
    if args.surface == "uniform-b-v3-confirmation":
        from .uniform_b_confirmation import (
            load_uniform_b_confirmation_config,
            run_uniform_b_confirmation,
        )

        output = run_uniform_b_confirmation(
            load_uniform_b_confirmation_config(args.config)
        )
        print(output)
        return 0
    if args.surface == "build-uniform-b-canonical-train-cache":
        from .uniform_b_reference import (
            build_uniform_b_canonical_train_cache,
            load_uniform_b_canonical_cache_config,
        )

        output = build_uniform_b_canonical_train_cache(
            load_uniform_b_canonical_cache_config(args.config)
        )
        print(output)
        return 0
    if args.surface == "uniform-b-canonical-reference":
        from .uniform_b_reference import (
            load_uniform_b_canonical_reference_config,
            run_uniform_b_canonical_reference,
        )

        output = run_uniform_b_canonical_reference(
            load_uniform_b_canonical_reference_config(args.config)
        )
        print(output)
        return 0
    if args.surface == "uniform-b-nystroem-nonlinear-probe":
        from .uniform_b_nonlinear_probe import (
            load_nonlinear_probe_config,
            run_nonlinear_probe,
        )

        output = run_nonlinear_probe(load_nonlinear_probe_config(args.config))
        print(output)
        return 0
    if args.surface == "uniform-b-robust-interaction-probe":
        from .uniform_b_robust_interaction_probe import (
            load_robust_interaction_config,
            run_robust_interaction_probe,
        )

        output = run_robust_interaction_probe(
            load_robust_interaction_config(args.config)
        )
        print(output)
        return 0
    if args.surface == "uniform-b-sens-spec-constrained-nystroem-probe":
        from .uniform_b_sens_spec_constrained_nystroem_probe import (
            load_constrained_nystroem_config,
            run_constrained_nystroem_probe,
        )

        output = run_constrained_nystroem_probe(
            load_constrained_nystroem_config(args.config)
        )
        print(output)
        return 0
    if args.surface == "build-uniform-b-spatial-cache":
        from .uniform_b_spatial_probe import (
            build_uniform_b_spatial_cache,
            load_spatial_cache_config,
        )

        output = build_uniform_b_spatial_cache(
            load_spatial_cache_config(args.config)
        )
        print(output)
        return 0
    if args.surface == "uniform-b-spatial-probe":
        from .uniform_b_spatial_probe import (
            load_spatial_probe_config,
            run_spatial_probe,
        )

        output = run_spatial_probe(load_spatial_probe_config(args.config))
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
