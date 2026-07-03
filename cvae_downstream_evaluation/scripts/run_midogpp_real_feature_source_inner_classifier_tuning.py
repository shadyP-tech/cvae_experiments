"""Run MIDOG++ real-feature source-inner classifier tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.classifier_grid import (  # noqa: E402
    add_classifier_grid_arguments,
    classifier_specs_from_args,
    csv_values,
)
from cvae_downstream_evaluation.midogpp_real_feature_classifier import (  # noqa: E402
    load_midogpp_real_feature_frame,
    run_midogpp_real_feature_source_inner_classifier_tuning,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from cvae_downstream_evaluation.schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune MIDOG++ source-only real-feature classifiers with source-inner LODO."
    )
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
        return
    specs = classifier_specs_from_args(args)
    outputs = run_midogpp_real_feature_source_inner_classifier_tuning(
        manifest_path=Path(args.manifest),
        feature_cache_path=Path(args.feature_cache),
        out_dir=Path(args.out_dir),
        candidate_specs=specs,
        heldout_centers=csv_values(args.heldout_centers),
        experiment_seed=int(args.experiment_seed),
        classifier_seed=int(args.classifier_seed),
        expected_feature_dim=int(args.expected_feature_dim),
    )
    for label, path in outputs.as_dict().items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    try:
        main()
    except ProtocolError as exc:
        raise SystemExit(str(exc)) from exc
