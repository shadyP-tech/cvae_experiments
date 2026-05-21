"""Run G1 source-only discriminative CVAE objective diagnostic."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.c41_workstation import (  # noqa: E402
    c41_training_profile_from_config,
    safe_support_selection_units_from_paths,
)
from cvae_downstream_evaluation.g1_source_discriminative_cvae import (  # noqa: E402
    G1_ARTIFACTS_ROOT,
    G1_DEFAULT_C41_ROOT,
    G1_DEFAULT_C42_ROOT,
    G1_DEFAULT_C63_ROOT,
    G1_FULL_VARIANTS,
    G1_STAGE1_VARIANTS,
    G1RunLimits,
    run_g1_source_discriminative_cvae,
)
from cvae_downstream_evaluation.protocol import ProtocolError, load_locked_v1_config  # noqa: E402
from cvae_downstream_evaluation.routing import write_support_selection_units  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run G1 source-only discriminative CVAE downstream diagnostic.")
    parser.add_argument("--config", required=True, help="Path to locked downstream v1 YAML config.")
    parser.add_argument("--artifacts-root", default=G1_ARTIFACTS_ROOT, help="Output root for G1 artifacts.")
    parser.add_argument("--c41-artifacts-root", default=G1_DEFAULT_C41_ROOT, help="Input C4.1 full artifact root.")
    parser.add_argument("--c42-artifacts-root", default=G1_DEFAULT_C42_ROOT, help="Input C4.2 latent-prior artifact root for C6.3 augmentation.")
    parser.add_argument("--c63-artifacts-root", default=G1_DEFAULT_C63_ROOT, help="Input C6.3 artifact root for replay deltas.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the execution plan.")
    parser.add_argument("--smoke", action="store_true", help="Shortcut for one seed/center/support/classifier and stage-1 variants.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing G1 checkpoints and projections.")
    parser.add_argument("--training-profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--variant-stage", choices=("stage1", "full"), default="full", help="Variant set to run.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda:0"))
    parser.add_argument(
        "--allow-legacy-audit-columns",
        action="store_true",
        help="Drop forbidden oracle/eval audit columns from legacy support-selection artifacts instead of failing.",
    )
    parser.add_argument("--limit-experiment-seeds", default=None, help="Comma-separated experiment seeds.")
    parser.add_argument("--limit-heldout-centers", default=None, help="Comma-separated heldout centers.")
    parser.add_argument("--limit-support-sizes", default=None, help="Comma-separated support sizes.")
    parser.add_argument("--limit-support-seeds", default=None, help="Comma-separated support seeds.")
    parser.add_argument("--limit-generation-seeds", default=None, help="Comma-separated generation seeds.")
    parser.add_argument("--limit-classifier-seeds", default=None, help="Comma-separated classifier seeds.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path.cwd()
    config_path = Path(args.config)
    config = load_locked_v1_config(config_path)
    c41_root = repo_root / str(args.c41_artifacts_root)
    c42_root = repo_root / str(args.c42_artifacts_root)
    c63_root = repo_root / str(args.c63_artifacts_root)
    _assert_inputs_exist(c41_root, c42_root)
    support_paths = [Path(path) for path in glob.glob(str(repo_root / config.support_selection_glob))]
    if not support_paths:
        raise ProtocolError(f"No support selection artifacts matched: {config.support_selection_glob}")
    support_units = safe_support_selection_units_from_paths(
        support_paths,
        strict_forbidden_columns=not bool(args.allow_legacy_audit_columns),
    )
    artifacts_root = repo_root / str(args.artifacts_root)
    profile_name = "smoke" if args.smoke else args.training_profile
    profile = c41_training_profile_from_config(config_path, profile=profile_name)
    variants = G1_STAGE1_VARIANTS if (args.smoke or args.variant_stage == "stage1") else G1_FULL_VARIANTS
    limits = G1RunLimits(
        experiment_seeds=_parse_int_limit(args.limit_experiment_seeds) or ((config.experiment_seeds[0],) if args.smoke else None),
        heldout_centers=_parse_str_limit(args.limit_heldout_centers) or ((config.candidate_domains[0],) if args.smoke else None),
        support_sizes=_parse_int_limit(args.limit_support_sizes) or ((config.support_sizes[0],) if args.smoke else None),
        support_seeds=_parse_int_limit(args.limit_support_seeds) or ((config.support_seeds[0],) if args.smoke else None),
        generation_seeds=_parse_int_limit(args.limit_generation_seeds) or ((config.generation_seeds[0],) if args.smoke else None),
        classifier_seeds=_parse_int_limit(args.limit_classifier_seeds) or ((config.classifier_seeds[0],) if args.smoke else None),
        variants=variants,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_passed",
                    "experiment": "G1 source-only discriminative CVAE objective",
                    "artifacts_root": str(artifacts_root),
                    "c41_artifacts_root": str(c41_root),
                    "c42_artifacts_root": str(c42_root),
                    "c63_artifacts_root": str(c63_root),
                    "support_selection_files": len(support_paths),
                    "support_selection_units": len(support_units),
                    "training_profile": profile.__dict__,
                    "variants": variants,
                    "limits": {
                        "experiment_seeds": limits.experiment_seeds,
                        "heldout_centers": limits.heldout_centers,
                        "support_sizes": limits.support_sizes,
                        "support_seeds": limits.support_seeds,
                        "generation_seeds": limits.generation_seeds,
                        "classifier_seeds": limits.classifier_seeds,
                    },
                    "generation_mode": "posterior_sample_decoder_mean",
                    "checkpoint_selection_metric": "source_val_nelbo_reconstruction_kl_only_with_source_only_constraints",
                    "target_support_labels_used": 0,
                    "target_eval_labels_used_for_selection": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    support_units_path = artifacts_root / "tables" / "support_selection_units.csv"
    write_support_selection_units(support_units_path, support_units)
    print(f"Wrote G1-safe support units: {support_units_path}")
    outputs = run_g1_source_discriminative_cvae(
        config=config,
        repo_root=repo_root,
        artifacts_root=artifacts_root,
        c41_artifacts_root=c41_root,
        c42_artifacts_root=c42_root,
        c63_artifacts_root=c63_root,
        support_units=support_units,
        device=args.device,
        resume=bool(args.resume),
        training_profile=profile,
        limits=limits,
    )
    print(f"Wrote G1 reports under: {artifacts_root / 'tables'}")
    for name, path in sorted(outputs.items()):
        print(f"{name}: {path}")


def _assert_inputs_exist(c41_root: Path, c42_root: Path) -> None:
    required = (
        c41_root / "checkpoints",
        c41_root / "projections",
        c42_root / "latent_priors",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ProtocolError("G1 requires completed C4.1/C4.2 artifacts:\n" + "\n".join(missing))


def _parse_int_limit(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or not str(raw).strip():
        return None
    return tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())


def _parse_str_limit(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or not str(raw).strip():
        return None
    return tuple(str(part.strip()) for part in str(raw).split(",") if part.strip())


if __name__ == "__main__":
    main()
