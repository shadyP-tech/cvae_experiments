from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from run_direct_support_nelbo_downstream import build_parser, _classifier_specs_from_args  # noqa: E402


def test_runner_builds_source_inner_classifier_grid_from_cli_args() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "dummy.yaml",
            "--source-inner-classifier-tuning",
            "--classifier-c-grid",
            "0.1,1.0",
            "--classifier-penalties",
            "l2",
            "--classifier-solvers",
            "lbfgs",
            "--classifier-class-weights",
            "none,balanced",
            "--classifier-max-iters",
            "2000",
        ]
    )

    specs = _classifier_specs_from_args(args)
    assert len(specs) == 4
    assert {spec.C for spec in specs} == {0.1, 1.0}
    assert {spec.class_weight for spec in specs} == {None, "balanced"}
    assert {spec.penalty for spec in specs} == {"l2"}
    assert {spec.solver for spec in specs} == {"lbfgs"}


def test_runner_rejects_invalid_source_inner_classifier_grid() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "dummy.yaml",
            "--source-inner-classifier-tuning",
            "--classifier-c-grid",
            "1.0",
            "--classifier-penalties",
            "elasticnet",
            "--classifier-solvers",
            "lbfgs",
            "--classifier-class-weights",
            "none",
            "--classifier-max-iters",
            "2000",
            "--classifier-l1-ratios",
            "0.5",
        ]
    )

    try:
        _classifier_specs_from_args(args)
    except ProtocolError:
        pass
    else:
        raise AssertionError("runner accepted invalid elasticnet/lbfgs classifier grid")
