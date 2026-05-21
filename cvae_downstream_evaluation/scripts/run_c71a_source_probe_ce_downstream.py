"""Run C7.1a source-probe CE geometry-regularized CVAE diagnostic."""

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
from cvae_downstream_evaluation.c71a_source_probe_ce import (  # noqa: E402
    C71A_ARTIFACTS_ROOT,
    C71A_DEFAULT_C41_ROOT,
    C71A_DEFAULT_C63_ROOT,
    run_c71a_source_probe_ce,
)
from cvae_downstream_evaluation.matrix import MatrixBuildLimits  # noqa: E402
from cvae_downstream_evaluation.protocol import ProtocolError, load_locked_v1_config  # noqa: E402
from cvae_downstream_evaluation.routing import write_support_selection_units  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run C7.1a source-probe CE CVAE downstream diagnostic.")
    parser.add_argument("--config", required=True, help="Path to locked downstream v1 YAML config.")
    parser.add_argument("--artifacts-root", default=C71A_ARTIFACTS_ROOT, help="Output root for C7.1a artifacts.")
    parser.add_argument("--c41-artifacts-root", default=C71A_DEFAULT_C41_ROOT, help="Input C4.1 full artifact root.")
    parser.add_argument("--c63-artifacts-root", default=C71A_DEFAULT_C63_ROOT, help="Optional C6.3 full-context artifact root.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the execution plan.")
    parser.add_argument("--smoke", action="store_true", help="Shortcut for a small protocol-compliant run.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing checkpoints and projections.")
    parser.add_argument("--training-profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda:0"))
    parser.add_argument(
        "--allow-legacy-audit-columns",
        action="store_true",
        help="Drop forbidden oracle/eval audit columns from legacy support-selection artifacts instead of failing.",
    )
    parser.add_argument("--limit-experiment-seeds", default=None, help="Comma-separated experiment seeds.")
    parser.add_argument("--limit-heldout-centers", default=None, help="Comma-separated heldout centers.")
    parser.add_argument("--limit-generation-seeds", default=None, help="Comma-separated generation seeds.")
    parser.add_argument("--limit-classifier-seeds", default=None, help="Comma-separated classifier seeds.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path.cwd()
    config_path = Path(args.config)
    config = load_locked_v1_config(config_path)
    c41_root = repo_root / str(args.c41_artifacts_root)
    c63_root = repo_root / str(args.c63_artifacts_root)
    _assert_inputs_exist(c41_root)
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
    limits = MatrixBuildLimits(
        experiment_seeds=_parse_int_limit(args.limit_experiment_seeds) or ((config.experiment_seeds[0],) if args.smoke else None),
        heldout_centers=_parse_str_limit(args.limit_heldout_centers) or ((config.candidate_domains[0],) if args.smoke else None),
        generation_seeds=_parse_int_limit(args.limit_generation_seeds) or ((config.generation_seeds[0],) if args.smoke else None),
        classifier_seeds=_parse_int_limit(args.limit_classifier_seeds) or ((config.classifier_seeds[0],) if args.smoke else None),
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_passed",
                    "experiment": "C7.1a source-probe CE geometry-regularized CVAE",
                    "artifacts_root": str(artifacts_root),
                    "c41_artifacts_root": str(c41_root),
                    "c63_artifacts_root": str(c63_root),
                    "support_selection_files": len(support_paths),
                    "support_selection_units": len(support_units),
                    "training_profile": profile.__dict__,
                    "limits": {
                        "experiment_seeds": limits.experiment_seeds,
                        "heldout_centers": limits.heldout_centers,
                        "generation_seeds": limits.generation_seeds,
                        "classifier_seeds": limits.classifier_seeds,
                    },
                    "variants": ["C7.1_base", "C7.1_source_probe_ce"],
                    "context_replay": "C6.3_original_c41_hetero_mean_only_replay",
                    "generation_mode": "posterior_sample_decoder_mean",
                    "source_probe_ce_lambda": 0.05,
                    "checkpoint_selection_metric": "source_val_nelbo_reconstruction_kl_only",
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
    print(f"Wrote C7.1a-safe support units: {support_units_path}")
    outputs = run_c71a_source_probe_ce(
        config=config,
        repo_root=repo_root,
        artifacts_root=artifacts_root,
        c41_artifacts_root=c41_root,
        c63_artifacts_root=c63_root,
        support_units=support_units,
        device=args.device,
        resume=bool(args.resume),
        training_profile=profile,
        limits=limits,
    )
    print(f"Wrote C7.1a reports under: {artifacts_root / 'tables'}")
    for name, path in sorted(outputs.items()):
        print(f"{name}: {path}")


def _assert_inputs_exist(c41_root: Path) -> None:
    required = (
        c41_root / "checkpoints",
        c41_root / "projections",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ProtocolError("C7.1a requires completed C4.1 artifacts:\n" + "\n".join(missing))


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
