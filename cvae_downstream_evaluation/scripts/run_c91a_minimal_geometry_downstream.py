"""Run C9.1a minimal source-geometry CVAE downstream experiment."""

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
from cvae_downstream_evaluation.c91a_minimal_geometry import (  # noqa: E402
    C91A_ARTIFACTS_ROOT,
    C91A_GENERATOR_FAMILY,
    C91A_PROBE_PROTO_MODE,
    C91aRunLimits,
    build_c91a_downstream,
)
from cvae_downstream_evaluation.protocol import ProtocolError, load_locked_v1_config  # noqa: E402
from cvae_downstream_evaluation.routing import write_support_selection_units  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run C9.1a minimal source-geometry class-conditioned PCA64 CVAE downstream evaluation."
    )
    parser.add_argument("--config", required=True, help="Path to locked downstream v1 YAML config.")
    parser.add_argument("--artifacts-root", default=C91A_ARTIFACTS_ROOT, help="Output root for C9.1a artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the C9.1a execution plan.")
    parser.add_argument("--smoke", action="store_true", help="Shortcut for one seed/center/support/generation/classifier condition.")
    parser.add_argument("--resume", action="store_true", help="Resume/reuse existing checkpoints and matrix rows.")
    parser.add_argument("--training-profile", choices=("smoke", "full"), default="full", help="Training profile.")
    parser.add_argument("--device", default="auto", help="Torch device for C9.1a training/generation, e.g. auto, cpu, cuda:0, cuda:1.")
    parser.add_argument(
        "--skip-probe-only",
        action="store_true",
        help="Skip the cheap probe-only diagnostic mode; ELBO-only and probe+prototype still run.",
    )
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
    support_paths = [Path(path) for path in glob.glob(str(repo_root / config.support_selection_glob))]
    if not support_paths:
        raise ProtocolError(f"No support selection artifacts matched: {config.support_selection_glob}")
    support_units = safe_support_selection_units_from_paths(
        support_paths,
        strict_forbidden_columns=not bool(args.allow_legacy_audit_columns),
    )
    profile_name = "smoke" if args.smoke else args.training_profile
    training_profile = c41_training_profile_from_config(config_path, profile=profile_name)
    artifacts_root = repo_root / str(args.artifacts_root)
    limits = C91aRunLimits(
        experiment_seeds=_parse_int_limit(args.limit_experiment_seeds) or ((config.experiment_seeds[0],) if args.smoke else None),
        heldout_centers=_parse_str_limit(args.limit_heldout_centers) or ((config.candidate_domains[0],) if args.smoke else None),
        support_sizes=_parse_int_limit(args.limit_support_sizes) or ((config.support_sizes[0],) if args.smoke else None),
        support_seeds=_parse_int_limit(args.limit_support_seeds) or ((config.support_seeds[0],) if args.smoke else None),
        generation_seeds=_parse_int_limit(args.limit_generation_seeds) or ((config.generation_seeds[0],) if args.smoke else None),
        classifier_seeds=_parse_int_limit(args.limit_classifier_seeds) or ((config.classifier_seeds[0],) if args.smoke else None),
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_passed",
                    "experiment": "C9.1a minimal source-geometry CVAE objective",
                    "artifacts_root": str(artifacts_root),
                    "support_selection_files": len(support_paths),
                    "support_selection_units": len(support_units),
                    "training_profile": training_profile.__dict__,
                    "limits": {
                        "experiment_seeds": limits.experiment_seeds,
                        "heldout_centers": limits.heldout_centers,
                        "support_sizes": limits.support_sizes,
                        "support_seeds": limits.support_seeds,
                        "generation_seeds": limits.generation_seeds,
                        "classifier_seeds": limits.classifier_seeds,
                    },
                    "generator_family": C91A_GENERATOR_FAMILY,
                    "primary_generation_mode": C91A_PROBE_PROTO_MODE,
                    "beta_effective": 1.0,
                    "source_val_probe_bacc_used_for_checkpoint": 0,
                    "target_support_labels_used": 0,
                    "target_eval_labels_used_for_selection": 0,
                    "include_probe_only": not bool(args.skip_probe_only),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    support_units_path = artifacts_root / "tables" / "support_selection_units.csv"
    write_support_selection_units(support_units_path, support_units)
    outputs = build_c91a_downstream(
        config=config,
        repo_root=repo_root,
        artifacts_root=artifacts_root,
        support_units=support_units,
        device=str(args.device),
        resume=bool(args.resume),
        training_profile=training_profile,
        limits=limits,
        include_probe_only=not bool(args.skip_probe_only),
    )
    print(f"Wrote C9.1a reports under: {artifacts_root / 'tables'}")
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True))


def _parse_int_limit(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or not str(raw).strip():
        return None
    return tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())


def _parse_str_limit(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or not str(raw).strip():
        return None
    return tuple(str(part.strip()) for part in str(raw).split(",") if part.strip())


if __name__ == "__main__":
    try:
        main()
    except ProtocolError as exc:
        raise SystemExit(f"Protocol error: {exc}") from exc

