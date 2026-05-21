from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "cvae_testing"))

from cvae_downstream_evaluation.g1_source_discriminative_cvae import (  # noqa: E402
    AUX_RAMP_EPOCHS,
    AUX_WARMUP_EPOCHS,
    G1VariantSpec,
    LAMBDA_CE_MAX,
    LAMBDA_DISTILL_MAX,
    LAMBDA_MARGIN_MAX,
    MAX_WEIGHTED_AUX_TO_NLL_RATIO,
    SourceProbe,
    assert_g1_prejoin_rows_safe,
    centroid_margin_hinge,
    g1_aux_weights,
    source_val_composite_source_only,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from src.models.cvae_expert import CVAEExpert, DECODER_LIKELIHOOD_GAUSSIAN_DIAG  # noqa: E402


def test_g1_runner_cli_help_loads() -> None:
    script = ROOT / "scripts" / "run_g1_source_discriminative_cvae_downstream.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "G1" in result.stdout
    assert "--variant-stage" in result.stdout


def test_g1_aux_weight_warmup_and_ramp_are_predeclared() -> None:
    spec = G1VariantSpec("x", ce_max=LAMBDA_CE_MAX, distill_max=LAMBDA_DISTILL_MAX, margin_max=LAMBDA_MARGIN_MAX)
    assert g1_aux_weights(0, spec) == (0.0, 0.0, 0.0)
    assert g1_aux_weights(AUX_WARMUP_EPOCHS - 1, spec) == (0.0, 0.0, 0.0)
    first = g1_aux_weights(AUX_WARMUP_EPOCHS, spec)
    final = g1_aux_weights(AUX_WARMUP_EPOCHS + AUX_RAMP_EPOCHS + 2, spec)

    assert 0.0 < first[0] < final[0]
    assert final == (LAMBDA_CE_MAX, LAMBDA_DISTILL_MAX, LAMBDA_MARGIN_MAX)


def test_g1_source_val_composite_is_source_only_formula() -> None:
    value = source_val_composite_source_only(
        source_val_nelbo=2.0,
        source_probe_ce=3.0,
        distill_kl=5.0,
        centroid_margin=7.0,
        ce_weight=0.1,
        distill_weight=0.2,
        margin_weight=0.3,
    )

    assert abs(value - (2.0 + 0.3 + 1.0 + 2.1)) < 1.0e-12


def test_g1_prejoin_guard_rejects_target_or_utility_columns() -> None:
    assert_g1_prejoin_rows_safe([{"member_key": "safe", "source_expert": "1"}])
    for bad_key in ("target_eval_labels", "support_label_counts", "bacc", "oracle_expert", "current_heldout_utility"):
        try:
            assert_g1_prejoin_rows_safe([{"member_key": "x", bad_key: 1}])
        except ProtocolError:
            continue
        raise AssertionError(f"G1 pre-join guard accepted forbidden column {bad_key}")


def test_g1_auxiliary_losses_backpropagate_to_decoder_mean() -> None:
    torch.manual_seed(11)
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
    centroids = {
        0: x[y == 0].mean(dim=0),
        1: x[y == 1].mean(dim=0),
    }

    mu_z, _logvar_z = model.encode(x, y=y)
    decoder_mean, decoder_logvar = model.decode(mu_z, y=y, return_distribution=True)
    student = probe(decoder_mean)
    with torch.no_grad():
        teacher = probe(x)
    ce = torch.nn.functional.cross_entropy(student, y)
    distill = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student / 2.0, dim=1),
        torch.nn.functional.softmax(teacher / 2.0, dim=1),
        reduction="batchmean",
    ) * 4.0
    margin = centroid_margin_hinge(decoder_mean, y, centroids)
    loss = ce + distill + margin
    loss.backward()

    assert decoder_logvar is not None
    assert model.dec2.weight.grad is not None
    assert model.fc_mu.weight.grad is not None


def test_g1_aux_cap_constant_is_conservative() -> None:
    assert 0.0 < MAX_WEIGHTED_AUX_TO_NLL_RATIO <= 0.20
