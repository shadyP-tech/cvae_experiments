"""Command-line entrypoints for SAIL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import RunLimits, load_config
from .features import CacheBuildRequest, build_virchow2_cache
from .pipeline import run_pipeline
from .protocol import ProtocolError


DEFAULT_CONFIG = "sail/configs/sail_virchow2.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAIL source-only aggregation pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-config", help="Validate the locked source-only config.")
    validate.add_argument("--config", default=DEFAULT_CONFIG)

    cache = sub.add_parser("build-cache", help="Build or dry-run frozen Virchow2 feature caches.")
    cache.add_argument("--config", default=DEFAULT_CONFIG)
    cache.add_argument("--samples-manifest", required=True)
    cache.add_argument("--experiment-seed", type=int, required=True)
    cache.add_argument("--output-root", default=None)
    cache.add_argument("--model-ref", default="hf-hub:paige-ai/Virchow2")
    cache.add_argument("--batch-size", type=int, default=32)
    cache.add_argument("--device", default="auto")
    cache.add_argument("--splits", default="train,val,test")
    cache.add_argument("--limit-samples-per-split", type=int, default=None)
    cache.add_argument("--overwrite", action="store_true")
    cache.add_argument("--dry-run", action="store_true")

    run = sub.add_parser("run", help="Run source-inner LODO selection and dense aggregation.")
    run.add_argument("--config", default=DEFAULT_CONFIG)
    run.add_argument("--limit-experiment-seeds", default=None)
    run.add_argument("--limit-heldout-centers", default=None)
    run.add_argument("--k-values", default=None)
    run.add_argument("--aggregation-rules", default=None)
    run.add_argument("--representations", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path.cwd()
    config = load_config(Path(args.config))
    if args.command == "validate-config":
        print("Config validation passed for SAIL source-only aggregation.")
        return
    if args.command == "build-cache":
        output_root = Path(args.output_root) if args.output_root is not None else repo_root / config.cache_root
        result = build_virchow2_cache(
            CacheBuildRequest(
                samples_manifest=Path(args.samples_manifest),
                output_root=output_root,
                experiment_seed=int(args.experiment_seed),
                model_ref=str(args.model_ref),
                batch_size=int(args.batch_size),
                device=str(args.device),
                splits=_parse_str_limit(args.splits) or ("train", "val", "test"),
                limit_samples_per_split=args.limit_samples_per_split,
                overwrite=bool(args.overwrite),
                dry_run=bool(args.dry_run),
            )
        )
        print(
            json.dumps(
                {
                    "status": result.status,
                    "split_counts": dict(result.split_counts),
                    "outputs": {key: str(value) for key, value in result.output_paths.items()},
                    "report": str(result.report_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "run":
        result = run_pipeline(
            config=config,
            repo_root=repo_root,
            limits=RunLimits(
                experiment_seeds=_parse_int_limit(args.limit_experiment_seeds),
                heldout_centers=_parse_str_limit(args.limit_heldout_centers),
                k_values=_parse_int_limit(args.k_values),
                aggregation_rules=_parse_str_limit(args.aggregation_rules),
                representations=_parse_str_limit(args.representations),
            ),
        )
        print(
            json.dumps(
                {
                    "status": "virchow2_source_selected_dense_aggregation_complete",
                    "decision_labels": result.decision_labels,
                    "outputs": {key: str(value) for key, value in result.output_paths.items()},
                },
                indent=2,
                sort_keys=True,
            )
        )


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
        raise SystemExit(str(exc)) from exc
