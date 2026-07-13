"""CLI for active MIDOG++ real-feature cache and signal-control methods."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .features import CacheBuildRequest, build_virchow2_cache
from .midogpp_multiaxis import (
    load_midogpp_multiaxis_config,
    run_midogpp_multiaxis_baseline,
)
from .midogpp_signal_controls import (
    load_midogpp_signal_controls_config,
    run_midogpp_signal_controls,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MIDOG++ Virchow2 real-feature cache and signal controls."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cache = sub.add_parser("build-cache", help="Build or dry-run a Virchow2 cache.")
    cache.add_argument("--samples-manifest", required=True)
    cache.add_argument("--experiment-seed", type=int, required=True)
    cache.add_argument("--output-root", required=True)
    cache.add_argument("--model-ref", default="hf-hub:paige-ai/Virchow2")
    cache.add_argument("--batch-size", type=int, default=32)
    cache.add_argument("--device", default="auto")
    cache.add_argument("--splits", default="train,val,test")
    cache.add_argument("--limit-samples-per-split", type=int, default=None)
    cache.add_argument("--overwrite", action="store_true")
    cache.add_argument("--dry-run", action="store_true")

    multiaxis = sub.add_parser(
        "run-midogpp-multiaxis",
        help="Run the MIDOG++ real-feature multi-axis LODO diagnostic.",
    )
    _add_config_overrides(multiaxis)

    signal = sub.add_parser(
        "run-midogpp-signal-controls",
        help="Run the MIDOG++ real-feature signal-control diagnostic.",
    )
    _add_config_overrides(signal)
    signal.add_argument("--prior-lodo-axis-summary-path", default=None)
    return parser


def _add_config_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--feature-cache-path", default=None)
    parser.add_argument("--artifacts-root", default=None)
    parser.add_argument("--allow-npz-test-cache", action="store_true")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path.cwd()
    if args.command == "build-cache":
        result = build_virchow2_cache(
            CacheBuildRequest(
                samples_manifest=Path(args.samples_manifest),
                output_root=Path(args.output_root),
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
        return 0

    if args.command == "run-midogpp-multiaxis":
        config = load_midogpp_multiaxis_config(Path(args.config))
        config = replace(
            config,
            manifest_path=str(args.manifest_path or config.manifest_path),
            feature_cache_path=str(args.feature_cache_path or config.feature_cache_path),
            artifacts_root=str(args.artifacts_root or config.artifacts_root),
            allow_npz_test_cache=bool(
                args.allow_npz_test_cache or config.allow_npz_test_cache
            ),
        )
        result = run_midogpp_multiaxis_baseline(config=config, repo_root=repo_root)
        _print_result("midogpp_virchow2_real_feature_multiaxis_complete", result)
        return 0

    config = load_midogpp_signal_controls_config(Path(args.config))
    config = replace(
        config,
        manifest_path=str(args.manifest_path or config.manifest_path),
        feature_cache_path=str(args.feature_cache_path or config.feature_cache_path),
        artifacts_root=str(args.artifacts_root or config.artifacts_root),
        prior_lodo_axis_summary_path=(
            str(args.prior_lodo_axis_summary_path)
            if args.prior_lodo_axis_summary_path is not None
            else config.prior_lodo_axis_summary_path
        ),
        allow_npz_test_cache=bool(args.allow_npz_test_cache or config.allow_npz_test_cache),
    )
    result = run_midogpp_signal_controls(config=config, repo_root=repo_root)
    _print_result("midogpp_virchow2_real_feature_signal_controls_complete", result)
    return 0


def _print_result(status: str, result: object) -> None:
    print(
        json.dumps(
            {
                "status": status,
                "decision_labels": list(getattr(result, "decision_labels")),
                "outputs": {
                    key: str(value)
                    for key, value in getattr(result, "output_paths").items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


def _parse_str_limit(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or not str(raw).strip():
        return None
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
