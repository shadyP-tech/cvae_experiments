"""Run C5.2 source-LOCO utility-ranking router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.c52_utility_ranker import (  # noqa: E402
    C52_ARTIFACTS_ROOT,
    C52_DEFAULT_C41_ROOT,
    C52_DEFAULT_C42_ROOT,
    C52_DEFAULT_C51_ROOT,
    C52RunLimits,
    run_c52_utility_ranker,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run C5.2 source-LOCO utility-ranking router over the fixed C4.1/C4.2/C5.1 CVAE bank."
    )
    parser.add_argument("--artifacts-root", default=C52_ARTIFACTS_ROOT, help="Output root for C5.2 artifacts.")
    parser.add_argument("--c41-artifacts-root", default=C52_DEFAULT_C41_ROOT, help="Input C4.1 artifact root.")
    parser.add_argument("--c42-artifacts-root", default=C52_DEFAULT_C42_ROOT, help="Input C4.2 artifact root.")
    parser.add_argument("--c51-artifacts-root", default=C52_DEFAULT_C51_ROOT, help="Input C5.1 artifact root.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the C5.2 execution plan.")
    parser.add_argument("--smoke", action="store_true", help="Shortcut for a tiny diagnostic route build.")
    parser.add_argument("--resume", action="store_true", help="Accepted for CLI symmetry; C5.2 rewrites report tables.")
    parser.add_argument("--limit-experiment-seeds", default=None, help="Comma-separated experiment seeds.")
    parser.add_argument("--limit-heldout-centers", default=None, help="Comma-separated heldout centers.")
    parser.add_argument("--limit-support-sizes", default=None, help="Comma-separated support sizes.")
    parser.add_argument("--limit-support-seeds", default=None, help="Comma-separated support seeds.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _ = args.resume
    repo_root = Path.cwd()
    artifacts_root = repo_root / str(args.artifacts_root)
    c41_root = repo_root / str(args.c41_artifacts_root)
    c42_root = repo_root / str(args.c42_artifacts_root)
    c51_root = repo_root / str(args.c51_artifacts_root)
    _assert_inputs_exist(c41_root, c42_root, c51_root)
    limits = C52RunLimits(
        experiment_seeds=_parse_int_limit(args.limit_experiment_seeds) or ((42,) if args.smoke else None),
        heldout_centers=_parse_str_limit(args.limit_heldout_centers),
        support_sizes=_parse_int_limit(args.limit_support_sizes) or ((4,) if args.smoke else None),
        support_seeds=_parse_int_limit(args.limit_support_seeds) or ((17,) if args.smoke else None),
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_passed",
                    "experiment": "C5.2 source-LOCO utility-ranking router",
                    "artifacts_root": str(artifacts_root),
                    "c41_artifacts_root": str(c41_root),
                    "c42_artifacts_root": str(c42_root),
                    "c51_artifacts_root": str(c51_root),
                    "limits": {
                        "experiment_seeds": limits.experiment_seeds,
                        "heldout_centers": limits.heldout_centers,
                        "support_sizes": limits.support_sizes,
                        "support_seeds": limits.support_seeds,
                    },
                    "primary_selector": "c52_ridge_loco_utility_rank_top1",
                    "primary_candidate_bank_excludes_noise": 1,
                    "target_support_labels_used": 0,
                    "target_eval_labels_used_for_selection": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    outputs = run_c52_utility_ranker(
        artifacts_root=artifacts_root,
        c41_artifacts_root=c41_root,
        c42_artifacts_root=c42_root,
        c51_artifacts_root=c51_root,
        limits=limits,
    )
    print(f"Wrote C5.2 reports under: {artifacts_root / 'tables'}")
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True))


def _assert_inputs_exist(c41_root: Path, c42_root: Path, c51_root: Path) -> None:
    required = (
        c41_root / "tables" / "all_expert_downstream_matrix.csv",
        c41_root / "tables" / "routing_to_downstream_alignment.csv",
        c42_root / "tables" / "all_expert_downstream_matrix.csv",
        c51_root / "tables" / "c51_support_mode_scores.csv",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise ProtocolError(f"Missing required C4.1/C4.2/C5.1 artifacts for C5.2: {missing}")


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
