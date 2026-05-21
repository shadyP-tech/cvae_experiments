"""Run C6.1 robust CVAE multi-source mixture downstream evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.c61_mixture import (  # noqa: E402
    C61_ARTIFACTS_ROOT,
    C61_DEFAULT_C41_ROOT,
    C61_DEFAULT_C42_ROOT,
    C61_DEFAULT_C52_ROOT,
    C61RunLimits,
    run_c61_mixture_downstream,
)
from cvae_downstream_evaluation.protocol import ProtocolError, load_locked_v1_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run C6.1 robust multi-source mixtures over the fixed C4.1/C4.2 CVAE bank."
    )
    parser.add_argument("--config", required=True, help="Path to locked downstream v1 YAML config.")
    parser.add_argument("--artifacts-root", default=C61_ARTIFACTS_ROOT, help="Output root for C6.1 artifacts.")
    parser.add_argument("--c41-artifacts-root", default=C61_DEFAULT_C41_ROOT, help="Input C4.1 full artifact root.")
    parser.add_argument("--c42-artifacts-root", default=C61_DEFAULT_C42_ROOT, help="Input C4.2 artifact root.")
    parser.add_argument("--c52-artifacts-root", default=C61_DEFAULT_C52_ROOT, help="Input C5.2 artifact root.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the C6.1 execution plan.")
    parser.add_argument("--smoke", action="store_true", help="Shortcut for one seed/center/support/generation/classifier diagnostic run.")
    parser.add_argument("--resume", action="store_true", help="Accepted for CLI symmetry; C6.1 rewrites report tables.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda:0"), help="Torch device for generation.")
    parser.add_argument("--limit-experiment-seeds", default=None, help="Comma-separated experiment seeds.")
    parser.add_argument("--limit-heldout-centers", default=None, help="Comma-separated heldout centers.")
    parser.add_argument("--limit-support-sizes", default=None, help="Comma-separated support sizes.")
    parser.add_argument("--limit-support-seeds", default=None, help="Comma-separated support seeds.")
    parser.add_argument("--limit-generation-seeds", default=None, help="Comma-separated generation seeds.")
    parser.add_argument("--limit-classifier-seeds", default=None, help="Comma-separated classifier seeds.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _ = args.resume
    repo_root = Path.cwd()
    config = load_locked_v1_config(Path(args.config))
    artifacts_root = repo_root / str(args.artifacts_root)
    c41_root = repo_root / str(args.c41_artifacts_root)
    c42_root = repo_root / str(args.c42_artifacts_root)
    c52_root = repo_root / str(args.c52_artifacts_root)
    _assert_inputs_exist(c41_root, c42_root, c52_root)
    limits = C61RunLimits(
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
                    "experiment": "C6.1 robust CVAE multi-source mixture downstream",
                    "artifacts_root": str(artifacts_root),
                    "c41_artifacts_root": str(c41_root),
                    "c42_artifacts_root": str(c42_root),
                    "c52_artifacts_root": str(c52_root),
                    "limits": {
                        "experiment_seeds": limits.experiment_seeds,
                        "heldout_centers": limits.heldout_centers,
                        "support_sizes": limits.support_sizes,
                        "support_seeds": limits.support_seeds,
                        "generation_seeds": limits.generation_seeds,
                        "classifier_seeds": limits.classifier_seeds,
                    },
                    "pooling_space": "dino_original",
                    "primary_modes": ["hetero_mean", "gmm_k1", "gmm_k2", "standard_prior"],
                    "target_support_labels_used": 0,
                    "target_eval_labels_used_for_selection": 0,
                    "checkpoints_retrained": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    outputs = run_c61_mixture_downstream(
        config=config,
        repo_root=repo_root,
        artifacts_root=artifacts_root,
        c41_artifacts_root=c41_root,
        c42_artifacts_root=c42_root,
        c52_artifacts_root=c52_root,
        device=args.device,
        limits=limits,
    )
    print(f"Wrote C6.1 reports under: {artifacts_root / 'tables'}")
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True))


def _assert_inputs_exist(c41_root: Path, c42_root: Path, c52_root: Path) -> None:
    required = (
        c41_root / "tables" / "support_selection_units.csv",
        c41_root / "tables" / "routing_to_downstream_alignment.csv",
        c41_root / "checkpoints",
        c41_root / "projections",
        c42_root / "latent_priors",
        c52_root / "tables" / "c52_predicted_utility_scores_pre_join.csv",
        c52_root / "tables" / "c52_selected_route_utility_join.csv",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise ProtocolError(f"Missing required C4.1/C4.2/C5.2 artifacts for C6.1: {missing}")


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
