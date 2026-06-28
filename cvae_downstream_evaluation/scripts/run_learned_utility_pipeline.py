"""Run the learned downstream utility artifact pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.compatibility.pipeline import (  # noqa: E402
    LearnedUtilityPipelineInputs,
    run_learned_utility_pipeline,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run learned downstream utility artifact pipeline.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--source-inner-training", required=True)
    parser.add_argument("--diagnostic-matrix", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--features", required=True, help="Comma-separated deployable feature columns.")
    parser.add_argument("--support-features", default=None)
    parser.add_argument("--source-inner-features", default=None)
    parser.add_argument("--metadata-features", default=None)
    parser.add_argument("--label", default="source_inner_heldout_bacc")
    parser.add_argument("--ridge-lambda", type=float, default=1e-6)
    parser.add_argument("--method", default="learned_downstream_utility_top1")
    parser.add_argument("--generation-not-frozen", action="store_true")
    parser.add_argument("--classifier-not-frozen", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = run_learned_utility_pipeline(
        LearnedUtilityPipelineInputs(
            candidates=Path(args.candidates),
            source_inner_training=Path(args.source_inner_training),
            diagnostic_matrix=Path(args.diagnostic_matrix),
            out_dir=Path(args.out_dir),
            feature_columns=tuple(part.strip() for part in str(args.features).split(",") if part.strip()),
            support_features=Path(args.support_features) if args.support_features else None,
            source_inner_features=Path(args.source_inner_features) if args.source_inner_features else None,
            metadata_features=Path(args.metadata_features) if args.metadata_features else None,
            label=str(args.label),
            ridge_lambda=float(args.ridge_lambda),
            method=str(args.method),
            generation_frozen=not bool(args.generation_not_frozen),
            classifier_frozen=not bool(args.classifier_not_frozen),
        )
    )
    print(f"Wrote allowed features: {outputs.allowed_features}")
    print(f"Wrote predicted features: {outputs.predicted_features}")
    print(f"Wrote estimator model: {outputs.estimator_model}")
    print(f"Wrote estimator diagnostics: {outputs.estimator_diagnostics}")
    print(f"Wrote selections: {outputs.selections}")
    print(f"Wrote alignment: {outputs.alignment}")
    print(f"Wrote leakage report: {outputs.leakage_report}")
    print(f"Wrote pipeline manifest: {outputs.manifest}")


if __name__ == "__main__":
    try:
        main()
    except ProtocolError as exc:
        raise SystemExit(str(exc)) from exc
