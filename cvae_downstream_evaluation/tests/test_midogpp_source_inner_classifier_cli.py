from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from run_midogpp_source_summary_phase1 import build_parser, _classifier_specs_from_args  # noqa: E402


def test_midogpp_runner_builds_source_inner_classifier_grid_from_cli_args() -> None:
    args = build_parser().parse_args(
        [
            "--summary-manifest",
            "summary.csv",
            "--test-cache-path",
            "test.npz",
            "--out-dir",
            "out",
            "--source-inner-classifier-tuning",
            "--classifier-c-grid",
            "0.1,1.0",
            "--classifier-penalties",
            "l2",
            "--classifier-solvers",
            "lbfgs",
            "--classifier-class-weights",
            "none,balanced",
        ]
    )

    specs = _classifier_specs_from_args(args)
    assert len(specs) == 4
    assert {spec.C for spec in specs} == {0.1, 1.0}
    assert {spec.class_weight for spec in specs} == {None, "balanced"}


def test_midogpp_runner_rejects_invalid_source_inner_classifier_grid() -> None:
    args = build_parser().parse_args(
        [
            "--summary-manifest",
            "summary.csv",
            "--test-cache-path",
            "test.npz",
            "--out-dir",
            "out",
            "--source-inner-classifier-tuning",
            "--classifier-penalties",
            "elasticnet",
            "--classifier-solvers",
            "lbfgs",
            "--classifier-l1-ratios",
            "0.5",
        ]
    )

    try:
        _classifier_specs_from_args(args)
    except ProtocolError:
        pass
    else:
        raise AssertionError("MIDOG++ runner accepted invalid elasticnet/lbfgs classifier grid")


def test_midogpp_runner_parses_threshold_policy_variants() -> None:
    args = build_parser().parse_args(
        [
            "--summary-manifest",
            "summary.csv",
            "--test-cache-path",
            "test.npz",
            "--out-dir",
            "out",
            "--source-inner-classifier-tuning",
            "--threshold-policy",
            "both",
        ]
    )

    assert args.threshold_policy == "both"
