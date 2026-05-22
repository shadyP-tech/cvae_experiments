from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "cvae_testing"))

from cvae_downstream_evaluation.c91a_minimal_geometry import (  # noqa: E402
    C91A_ELBO_ONLY_MODE,
    C91A_GENERATOR_FAMILY,
    C91A_PROBE_PROTO_MODE,
    build_c91a_alignment_rows,
    geometry_weight_for_epoch,
    normalized_prototype_centroid_loss,
    train_c91a_model,
)
from cvae_downstream_evaluation.c41_workstation import C41TrainingProfile  # noqa: E402
from cvae_downstream_evaluation.downstream import CandidateDownstreamRow  # noqa: E402
from cvae_downstream_evaluation.routing import SupportSelectionUnit  # noqa: E402
from cvae_downstream_evaluation.schemas import SUPPORT_NELBO_METHOD  # noqa: E402


def test_c91a_runner_cli_help_loads() -> None:
    script = ROOT / "scripts" / "run_c91a_minimal_geometry_downstream.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--skip-probe-only" in result.stdout
    assert "--device" in result.stdout


def test_c91a_geometry_ramp_is_deterministic() -> None:
    assert geometry_weight_for_epoch(0) == 0.0
    assert geometry_weight_for_epoch(4) == 0.0
    assert geometry_weight_for_epoch(5) == 0.1
    assert geometry_weight_for_epoch(14) == 1.0
    assert geometry_weight_for_epoch(20) == 1.0


def test_c91a_prototype_loss_is_normalized_and_finite() -> None:
    generated = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [4.0, 4.0]], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    centroids = {0: torch.tensor([0.5, 0.5]), 1: torch.tensor([3.5, 3.5])}
    traces = {0: torch.tensor(2.0), 1: torch.tensor(2.0)}

    loss = normalized_prototype_centroid_loss(generated, labels, centroids, traces)

    assert torch.isfinite(loss)
    assert float(loss.item()) == 0.0


def test_c91a_alignment_uses_c91_modes_as_oracle_eligible() -> None:
    support = SupportSelectionUnit(
        heldout_center="0",
        experiment_seed=42,
        support_size=4,
        support_seed=17,
        method=SUPPORT_NELBO_METHOD,
        selected_expert="1",
        candidate_experts=("1", "2"),
        support_nelbo_by_expert={"1": 1.0, "2": 2.0},
        target_expert_excluded=True,
        support_eval_split_id="split",
    )
    rows = [
        CandidateDownstreamRow(
            experiment_seed=42,
            heldout_center="0",
            support_size=4,
            support_seed=17,
            candidate_expert="1",
            generator_family=C91A_GENERATOR_FAMILY,
            generation_mode=C91A_PROBE_PROTO_MODE,
            budget_per_class=128,
            generation_seed=17,
            classifier_seed=17,
            bacc=0.71,
            macro_f1=0.7,
        ),
        CandidateDownstreamRow(
            experiment_seed=42,
            heldout_center="0",
            support_size=4,
            support_seed=17,
            candidate_expert="2",
            generator_family=C91A_GENERATOR_FAMILY,
            generation_mode=C91A_PROBE_PROTO_MODE,
            budget_per_class=128,
            generation_seed=17,
            classifier_seed=17,
            bacc=0.75,
            macro_f1=0.74,
        ),
    ]

    alignment = build_c91a_alignment_rows(support_units=[support], downstream_rows=rows, modes=(C91A_PROBE_PROTO_MODE,))

    assert alignment[0]["selected_bacc"] == 0.71
    assert alignment[0]["oracle_bacc"] == 0.75
    assert alignment[0]["downstream_oracle_expert"] == "2"


def test_c91a_training_freezes_source_probe_and_uses_source_val_elbo_checkpoint(tmp_path: Path) -> None:
    train_x = torch.randn(12, 4)
    train_y = torch.tensor([0, 1] * 6, dtype=torch.long)
    val_x = torch.randn(8, 4)
    val_y = torch.tensor([0, 1] * 4, dtype=torch.long)
    profile = C41TrainingProfile(
        name="smoke",
        hidden_dim=8,
        latent_dim=2,
        lr=1.0e-3,
        epochs=1,
        patience=1,
        batch_size=4,
        pca_components=4,
    )
    from cvae_downstream_evaluation.c91a_minimal_geometry import C91aVariant

    result = train_c91a_model(
        repo_root=REPO_ROOT,
        artifacts_root=tmp_path,
        experiment_seed=42,
        candidate_expert="1",
        variant=C91aVariant(C91A_ELBO_ONLY_MODE, 0.0, 0.0),
        train_x=train_x,
        val_x=val_x,
        train_y=train_y,
        val_y=val_y,
        profile=profile,
        device="cpu",
        resume=False,
    )

    assert result.checkpoint_path.exists()
    assert result.probe_diagnostics["source_probe_frozen"] == 1
    assert result.history_rows[0]["geometry_weight_current_epoch"] == 0.0
    assert "val_elbo_nll_checkpoint_metric" in result.history_rows[0]

