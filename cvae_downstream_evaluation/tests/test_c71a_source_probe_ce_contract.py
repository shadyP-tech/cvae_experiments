from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "cvae_testing"))

from cvae_downstream_evaluation.c71a_source_probe_ce import (  # noqa: E402
    AUX_RAMP_EPOCHS,
    AUX_WARMUP_EPOCHS,
    COLLAPSE_COV_TRACE_RATIO_MIN,
    SourceProbe,
    _class_cov_trace_ratios,
    _class_effective_rank_ratios,
    assert_c71a_prejoin_rows_safe,
    c71a_aux_weight,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from src.models.cvae_expert import CVAEExpert, DECODER_LIKELIHOOD_GAUSSIAN_DIAG  # noqa: E402


def test_c71a_runner_cli_help_loads() -> None:
    script = ROOT / "scripts" / "run_c71a_source_probe_ce_downstream.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "C7.1a" in result.stdout
    assert "--c41-artifacts-root" in result.stdout


def test_c71a_aux_weight_warmup_and_ramp_are_predeclared() -> None:
    assert c71a_aux_weight(0) == 0.0
    assert c71a_aux_weight(AUX_WARMUP_EPOCHS - 1) == 0.0
    first = c71a_aux_weight(AUX_WARMUP_EPOCHS)
    final = c71a_aux_weight(AUX_WARMUP_EPOCHS + AUX_RAMP_EPOCHS + 2)

    assert 0.0 < first < final
    assert abs(final - 0.05) < 1.0e-12


def test_c71a_prejoin_guard_rejects_target_or_utility_columns() -> None:
    assert_c71a_prejoin_rows_safe([{"member_key": "safe", "source_expert": "1"}])
    for bad_key in ("target_eval_labels", "support_label_counts", "bacc", "oracle_expert", "current_heldout_utility"):
        try:
            assert_c71a_prejoin_rows_safe([{"member_key": "x", bad_key: 1}])
        except ProtocolError:
            continue
        raise AssertionError(f"C7.1a pre-join guard accepted forbidden column {bad_key}")


def test_c71a_auxiliary_ce_uses_posterior_mean_decoder_path() -> None:
    torch.manual_seed(7)
    model = CVAEExpert(
        input_dim=4,
        hidden_dim=8,
        latent_dim=3,
        class_condition_dim=2,
        decoder_likelihood=DECODER_LIKELIHOOD_GAUSSIAN_DIAG,
    )
    probe = SourceProbe(input_dim=4, num_classes=2)
    x = torch.randn(6, 4)
    y = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)

    mu_z, _logvar_z = model.encode(x, y=y)
    decoder_mean, decoder_logvar = model.decode(mu_z, y=y, return_distribution=True)
    loss = torch.nn.functional.cross_entropy(probe(decoder_mean), y)
    loss.backward()

    assert decoder_logvar is not None
    assert model.dec2.weight.grad is not None
    assert model.fc_mu.weight.grad is not None


def test_c71a_class_collapse_diagnostics_flag_shrunk_covariance() -> None:
    real = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [4.0, 4.0],
            [5.0, 4.0],
            [4.0, 5.0],
            [5.0, 5.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    collapsed = torch.tensor(
        [
            [0.45, 0.45],
            [0.46, 0.45],
            [0.45, 0.46],
            [0.46, 0.46],
            [4.45, 4.45],
            [4.46, 4.45],
            [4.45, 4.46],
            [4.46, 4.46],
        ],
        dtype=torch.float32,
    )

    ratios = _class_cov_trace_ratios(collapsed, labels, real, labels)
    rank_ratios = _class_effective_rank_ratios(collapsed, labels, real, labels)

    assert min(ratios.values()) < COLLAPSE_COV_TRACE_RATIO_MIN
    assert min(rank_ratios.values()) <= 1.0
