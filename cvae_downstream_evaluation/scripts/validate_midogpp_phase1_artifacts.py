"""Validate MIDOG++ phase-1 diagnostic artifact directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.adapters.midogpp import validate_midogpp_phase1_artifacts  # noqa: E402
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from cvae_downstream_evaluation.schemas.midogpp import assert_midogpp_frozen_config_file  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate MIDOG++ phase-1 diagnostic matrix, summaries, and protocol reports."
    )
    parser.add_argument("--artifacts-root", required=True, help="MIDOG++ phase-1 artifact root.")
    parser.add_argument(
        "--expected-heldout-center",
        action="append",
        default=[],
        help="Heldout center that must be present in the diagnostic matrix. Repeat for multiple centers.",
    )
    parser.add_argument(
        "--expected-baseline-method",
        action="append",
        default=[],
        help="Baseline method that must be present as a diagnostic method-baseline row.",
    )
    parser.add_argument(
        "--require-preflight-reports",
        action="store_true",
        help="Require source-summary and, when baselines are expected, baseline preflight reports.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Validation report JSON path. Defaults to reports/phase1_validation_report.json.",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "utility_matrix" / "virchow2_midogpp_all_candidates.yaml"),
        help="Frozen MIDOG++ config to validate before artifact inspection.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts_root = Path(args.artifacts_root)
    out = Path(args.out) if args.out else artifacts_root / "reports" / "phase1_validation_report.json"
    try:
        assert_midogpp_frozen_config_file(Path(args.config))
        report = validate_midogpp_phase1_artifacts(
            artifacts_root,
            expected_heldout_centers=tuple(args.expected_heldout_center),
            expected_baseline_methods=tuple(args.expected_baseline_method),
            require_preflight_reports=bool(args.require_preflight_reports),
        )
    except ProtocolError as exc:
        _write_report(
            out,
            {
                "schema_version": "midogpp_phase1_validation_report_v1",
                "status": "FAIL",
                "artifacts_root": str(artifacts_root),
                "expected_heldout_centers": [str(center) for center in args.expected_heldout_center],
                "expected_baseline_methods": [str(method) for method in args.expected_baseline_method],
                "require_preflight_reports": bool(args.require_preflight_reports),
                "error_message": str(exc),
            },
        )
        raise
    _write_report(out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote validation report: {out}")


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except ProtocolError as exc:
        raise SystemExit(str(exc)) from exc
