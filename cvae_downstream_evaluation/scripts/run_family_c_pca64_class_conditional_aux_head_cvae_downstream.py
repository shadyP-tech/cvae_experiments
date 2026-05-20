"""Entrypoint for the PCA-64 class-conditional aux-head CVAE downstream diagnostic."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.family_c_pca64 import FamilyCPca64BuildLimits  # noqa: E402
from cvae_downstream_evaluation.family_c_pca64_conditional import (  # noqa: E402
    build_family_c_pca64_cc_all_expert_downstream_matrix,
    build_family_c_pca64_cc_reports,
    load_family_c_pca64_aux_head_config,
    read_family_c_pca64_cc_support_units,
)
from cvae_downstream_evaluation.matrix import discover_support_run_artifacts  # noqa: E402
from cvae_downstream_evaluation.protocol import ArtifactSyncError, ProtocolError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Family C PCA64 class-conditional aux-head CVAE downstream diagnostic."
    )
    parser.add_argument("--config", required=True, help="Path to locked PCA64 aux-head CVAE YAML config.")
    parser.add_argument("--device", default="auto", help="Torch device, e.g. auto, cuda:0, or cpu.")
    parser.add_argument("--validate-config-only", action="store_true", help="Validate config and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Validate support artifacts and print run shape.")
    parser.add_argument("--build-matrix", action="store_true", help="Train/load aux-head PCA64 CVAEs and build matrix.")
    parser.add_argument("--build-reports", action="store_true", help="Build reports from existing matrix.")
    parser.add_argument("--resume", action="store_true", help="Skip matrix rows already present.")
    parser.add_argument("--limit-experiment-seeds", default=None, help="Comma-separated experiment seeds.")
    parser.add_argument("--limit-heldout-centers", default=None, help="Comma-separated heldout centers.")
    parser.add_argument("--limit-support-sizes", default=None, help="Comma-separated support sizes.")
    parser.add_argument("--limit-support-seeds", default=None, help="Comma-separated support seeds.")
    parser.add_argument("--limit-generation-seeds", default=None, help="Comma-separated generation seeds.")
    parser.add_argument("--limit-classifier-seeds", default=None, help="Comma-separated classifier seeds.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = Path.cwd()
    config = load_family_c_pca64_aux_head_config(Path(args.config))
    if args.validate_config_only:
        print(f"Config validation passed for {config.experiment_name}.")
        return

    support_paths = [Path(path) for path in glob.glob(str(repo_root / config.support_selection_glob))]
    if not support_paths:
        raise ProtocolError(f"No support selection artifacts matched: {config.support_selection_glob}")
    support_units = read_family_c_pca64_cc_support_units(support_paths)
    artifacts_root = repo_root / config.artifacts_root
    limits = FamilyCPca64BuildLimits(
        experiment_seeds=_parse_int_limit(args.limit_experiment_seeds),
        heldout_centers=_parse_str_limit(args.limit_heldout_centers),
        support_sizes=_parse_int_limit(args.limit_support_sizes),
        support_seeds=_parse_int_limit(args.limit_support_seeds),
        generation_seeds=_parse_int_limit(args.limit_generation_seeds),
        classifier_seeds=_parse_int_limit(args.limit_classifier_seeds),
    )
    discovered = discover_support_run_artifacts(config=config, repo_root=repo_root)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_passed",
                    "experiment": config.experiment_name,
                    "support_selection_files": len(support_paths),
                    "support_contexts": len(support_units),
                    "effective_support_contexts": _limit_support_count(support_units, limits),
                    "support_run_artifacts": len(discovered),
                    "artifacts_root": str(artifacts_root),
                    "pca_dim": config.pca_dim,
                    "condition_dim": 2,
                    "aux_weight": config.metadata_constraint_aux_weight,
                    "budget_per_class": config.budget_per_class,
                    "device": args.device,
                    "limits": {
                        "experiment_seeds": limits.experiment_seeds,
                        "heldout_centers": limits.heldout_centers,
                        "support_sizes": limits.support_sizes,
                        "support_seeds": limits.support_seeds,
                        "generation_seeds": limits.generation_seeds,
                        "classifier_seeds": limits.classifier_seeds,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.build_matrix:
        paths = build_family_c_pca64_cc_all_expert_downstream_matrix(
            config=config,
            repo_root=repo_root,
            artifacts_root=artifacts_root,
            support_units=support_units,
            device=args.device,
            resume=bool(args.resume),
            limits=limits,
        )
        print(f"Wrote Family C PCA64 aux-head matrix: {paths['matrix']}")
        print(f"Wrote Family C PCA64 aux-head protocol audit: {paths['protocol_audit']}")

    if args.build_reports:
        paths = build_family_c_pca64_cc_reports(
            artifacts_root=artifacts_root,
            candidate_domains=config.candidate_domains,
            config=config,
        )
        print(f"Wrote Family C PCA64 aux-head routing alignment: {paths['alignment']}")
        print(f"Wrote Family C PCA64 aux-head decision summary: {paths['decision_summary']}")

    if not args.build_matrix and not args.build_reports:
        print(
            "Validated Family C PCA64 aux-head support contexts. Add --build-matrix on the "
            "workstation to train/load source-only aux-head PCA-space CVAEs and score downstream classifiers."
        )


def _parse_int_limit(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or not str(raw).strip():
        return None
    return tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())


def _parse_str_limit(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or not str(raw).strip():
        return None
    return tuple(str(part.strip()) for part in str(raw).split(",") if part.strip())


def _limit_support_count(units: list[object], limits: FamilyCPca64BuildLimits) -> int:
    count = 0
    for unit in units:
        if limits.experiment_seeds is not None and int(unit.experiment_seed) not in set(limits.experiment_seeds):
            continue
        if limits.heldout_centers is not None and str(unit.heldout_center) not in set(limits.heldout_centers):
            continue
        if limits.support_sizes is not None and int(unit.support_size) not in set(limits.support_sizes):
            continue
        if limits.support_seeds is not None and int(unit.support_seed) not in set(limits.support_seeds):
            continue
        count += 1
    return count


if __name__ == "__main__":
    try:
        main()
    except (ProtocolError, ArtifactSyncError) as exc:
        raise SystemExit(str(exc)) from exc
