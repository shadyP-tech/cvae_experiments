"""Build frozen pathology embedding caches consumed by R1.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.pathology_cache_builder import (  # noqa: E402
    CacheBuildRequest,
    build_r12_pathology_embedding_cache,
    default_output_root,
    default_support_run_dir,
    parse_csv_list,
)
from cvae_downstream_evaluation.pathology_embedding_screen import load_r12_config  # noqa: E402
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build R1.2 frozen pathology embedding caches.")
    parser.add_argument(
        "--config",
        default="cvae_downstream_evaluation/configs/experiments/r12_pathology_embedding_screen.yaml",
        help="Path to R1.2 config.",
    )
    parser.add_argument("--backbone", required=True, help="R1.2 backbone name, e.g. phikon or plip.")
    parser.add_argument("--model-dir", required=True, help="Downloaded model directory.")
    parser.add_argument("--experiment-seed", type=int, required=True, help="Support-run experiment seed.")
    parser.add_argument(
        "--support-run-dir",
        default=None,
        help="Override support run directory. Defaults to the Z1.1/R1.2 configured seed directory.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override pathology embedding cache root. Defaults to config inputs.cache_root.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated split list.")
    parser.add_argument("--image-size", type=int, default=None, help="Recorded in metadata; HF processor controls resize.")
    parser.add_argument("--limit-samples-per-split", type=int, default=None, help="Smoke-test cap per split.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split caches.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths/counts without loading model or writing caches.")
    parser.add_argument(
        "--allow-remote-files",
        action="store_true",
        help="Allow transformers to fetch missing files. Default is local-files-only.",
    )
    parser.add_argument("--loader", default="hf_auto", help="Model loader. Currently supports hf_auto.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path.cwd()
    config = load_r12_config(Path(args.config))
    support_run_dir = (
        Path(args.support_run_dir)
        if args.support_run_dir is not None
        else default_support_run_dir(config, repo_root, int(args.experiment_seed))
    )
    output_root = Path(args.output_root) if args.output_root is not None else default_output_root(config, repo_root)
    request = CacheBuildRequest(
        backbone_name=str(args.backbone),
        model_dir=Path(args.model_dir),
        experiment_seed=int(args.experiment_seed),
        support_run_dir=support_run_dir,
        output_root=output_root,
        batch_size=int(args.batch_size),
        device=str(args.device),
        splits=parse_csv_list(args.splits, default=("train", "val", "test")),
        image_size=args.image_size,
        limit_samples_per_split=args.limit_samples_per_split,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        local_files_only=not bool(args.allow_remote_files),
        loader=str(args.loader),
    )
    result = build_r12_pathology_embedding_cache(request)
    print(
        json.dumps(
            {
                "status": result.status,
                "backbone_name": request.backbone_name,
                "experiment_seed": request.experiment_seed,
                "split_counts": dict(result.split_counts),
                "outputs": {split: str(path) for split, path in result.output_paths.items()},
                "report": str(result.report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except ProtocolError as exc:
        raise SystemExit(str(exc)) from exc
