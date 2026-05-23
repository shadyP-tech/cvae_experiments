"""Entrypoint for R1.2c-V Virchow2 dense config aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from cvae_downstream_evaluation.r12c_dense_config_aggregation import (  # noqa: E402
    R12CRunLimits,
    load_r12c_config,
    run_r12c_dense_config_aggregation,
)


DEFAULT_CONFIG = "cvae_downstream_evaluation/configs/experiments/r12c_virchow2_dense_config_aggregation.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run R1.2c-V Virchow2 dense source-selected config aggregation.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-experiment-seeds", default=None)
    parser.add_argument("--limit-heldout-centers", default=None)
    parser.add_argument("--k-values", default=None)
    parser.add_argument("--aggregation-rules", default=None)
    parser.add_argument("--calibration-rules", default=None)
    parser.add_argument("--no-cross-backbone-audit", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path.cwd()
    config = load_r12c_config(Path(args.config))
    limits = R12CRunLimits(
        experiment_seeds=_parse_int_limit(args.limit_experiment_seeds),
        heldout_centers=_parse_str_limit(args.limit_heldout_centers),
        k_values=_parse_int_limit(args.k_values),
        aggregation_rules=_parse_str_limit(args.aggregation_rules),
        calibration_rules=_parse_str_limit(args.calibration_rules),
        include_cross_backbone=not bool(args.no_cross_backbone_audit),
    )
    if args.validate_config_only:
        print("Config validation passed for R1.2c-V Virchow2 dense config aggregation.")
        return
    if args.dry_run:
        r12b_root = repo_root / config.r12b_artifacts_root
        print(
            json.dumps(
                {
                    "status": "dry_run_passed",
                    "config": str(args.config),
                    "artifacts_root": config.artifacts_root,
                    "r12b_selection_exists": (r12b_root / "tables" / "r12b_source_inner_lodo_selection_matrix.csv").exists(),
                    "r12b_real_feature_exists": (r12b_root / "tables" / "r12b_real_feature_ceiling_matrix.csv").exists(),
                    "primary_backbone": config.primary_backbone,
                    "fixed_k_values": list(limits.k_values or config.fixed_k_values),
                    "aggregation_rules": list(limits.aggregation_rules or config.aggregation_rules),
                    "calibration_rules": list(limits.calibration_rules or config.audit_calibration_rules),
                    "include_cross_backbone": limits.include_cross_backbone,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = run_r12c_dense_config_aggregation(config=config, repo_root=repo_root, limits=limits)
    print(
        json.dumps(
            {
                "status": "r12c_virchow2_dense_config_aggregation_complete",
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
