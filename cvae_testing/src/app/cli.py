from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.app.bootstrap import resolve_config_path


@dataclass(frozen=True)
class CLIArgs:
    config: Path
    run_id: str | None
    seed: int | None
    resume: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run routed CVAE experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/breakhis/learned_utility_routing_v1.yaml"),
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoints in an existing run directory. Uses --run-id when provided, otherwise the latest run pointer.",
    )
    return parser


def parse_cli_args() -> CLIArgs:
    args = build_parser().parse_args()
    return CLIArgs(
        config=args.config,
        run_id=args.run_id,
        seed=args.seed,
        resume=bool(args.resume),
    )


def resolve_resume_run_id(project_root: Path, cfg: Dict[str, Any], run_id: str | None, resume: bool) -> str | None:
    if not resume:
        return run_id
    if run_id:
        return run_id

    exp_cfg = cfg.get("experiment", {})
    dataset_name = str(exp_cfg.get("dataset_name", "breakhis"))
    experiment_name = str(exp_cfg.get("name", "learned_utility_routing_v1"))
    output_cfg = cfg.get("output", {})
    output_root = resolve_config_path(project_root, str(output_cfg.get("root", "outputs")))
    latest_file = output_root / dataset_name / experiment_name / "latest.txt"

    if latest_file.exists():
        latest_run_id = latest_file.read_text(encoding="utf-8").strip()
        if latest_run_id:
            return latest_run_id

    raise RuntimeError(
        "--resume was requested but no run ID was provided and no latest run pointer was found. "
        "Pass --run-id explicitly to resume a specific run."
    )
