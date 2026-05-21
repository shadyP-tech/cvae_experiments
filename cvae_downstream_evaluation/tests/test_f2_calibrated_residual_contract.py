from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "cvae_testing"))

from cvae_downstream_evaluation.f1_source_anchored import (  # noqa: E402
    AnchorPairDataset,
    AnchoredResidualCVAE,
    build_source_anchor_index,
)
from cvae_downstream_evaluation.f2_calibrated_residual import (  # noqa: E402
    DECISION_SUPERIORITY_SUCCESS,
    F2_GENERATOR_FAMILY,
    F2_MODE_CALIBRATED_NOISE,
    F2_MODE_CALIBRATED_NOISE_NO_PENALTY,
    F2_MODE_EMPIRICAL_BOOTSTRAP,
    F2_MODE_IDENTITY_BOOTSTRAP,
    F2_MODE_TRANSFER_BOOTSTRAP,
    _f2_geometry_diagnostics,
    anchored_residual_loss_terms_f2,
    build_f2_delta_summary_rows,
    fit_residual_calibration,
    generate_f2_anchor_residual_embeddings,
)
from cvae_downstream_evaluation.schemas import SUPPORT_NELBO_METHOD  # noqa: E402


def test_f2_runner_cli_help_loads() -> None:
    script = ROOT / "scripts" / "run_f2_calibrated_residual_downstream.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--f1-artifacts-root" in result.stdout
    assert "--c61-artifacts-root" in result.stdout


def test_f2_loss_penalties_can_be_disabled() -> None:
    torch.manual_seed(7)
    delta_true = torch.randn(6, 4)
    delta_mu = torch.randn(6, 4)
    delta_logvar = torch.zeros(6, 4)
    mu = torch.randn(6, 3)
    logvar = torch.zeros(6, 3)

    penalty = anchored_residual_loss_terms_f2(
        delta_mu=delta_mu,
        delta_true=delta_true,
        delta_logvar=delta_logvar,
        mu=mu,
        logvar=logvar,
        use_moment_penalties=True,
    )
    no_penalty = anchored_residual_loss_terms_f2(
        delta_mu=delta_mu,
        delta_true=delta_true,
        delta_logvar=delta_logvar,
        mu=mu,
        logvar=logvar,
        use_moment_penalties=False,
    )

    assert penalty["loss"].ndim == 0
    assert torch.isfinite(penalty["loss"])
    assert torch.isfinite(no_penalty["loss"])
    assert no_penalty["residual_energy_penalty"].item() == 0.0
    assert no_penalty["residual_cov_trace_penalty"].item() == 0.0


def test_source_val_calibration_is_deterministic_and_falls_back_by_class() -> None:
    torch.manual_seed(11)
    model = AnchoredResidualCVAE(input_dim=2, hidden_dim=8, latent_dim=2, class_condition_dim=2)
    val_pairs = _val_pairs(n0=18, n1=3)

    first = fit_residual_calibration(
        model=model,
        val_pairs=val_pairs,
        experiment_seed=42,
        heldout_center="0",
        candidate_expert="1",
        model_variant="penalty",
        device="cpu",
    )
    second = fit_residual_calibration(
        model=model,
        val_pairs=val_pairs,
        experiment_seed=42,
        heldout_center="0",
        candidate_expert="1",
        model_variant="penalty",
        device="cpu",
    )

    assert first.class_scales == second.class_scales
    rows = {int(row["class_label"]): row for row in first.rows}
    assert rows[0]["fallback_used"] == 0
    assert rows[1]["fallback_used"] == 1
    assert all(0.5 <= float(row["scale_geomean"]) <= 2.5 for row in first.rows)
    assert {row["calibration_split"] for row in first.rows} == {"source_val"}


def test_f2_generation_is_seeded_and_uses_source_residual_reference_provenance() -> None:
    torch.manual_seed(13)
    model = AnchoredResidualCVAE(input_dim=2, hidden_dim=8, latent_dim=2, class_condition_dim=2)
    anchor_index = _anchor_index()
    calibration = _calibration()

    first = generate_f2_anchor_residual_embeddings(
        model=model,
        anchor_index=anchor_index,
        calibration=calibration,
        class_label=0,
        n_samples=4,
        seed=23,
        generation_mode=F2_MODE_CALIBRATED_NOISE,
        experiment_seed=42,
        heldout_center="0",
        candidate_expert="1",
    )
    second = generate_f2_anchor_residual_embeddings(
        model=model,
        anchor_index=anchor_index,
        calibration=calibration,
        class_label=0,
        n_samples=4,
        seed=23,
        generation_mode=F2_MODE_CALIBRATED_NOISE,
        experiment_seed=42,
        heldout_center="0",
        candidate_expert="1",
    )

    assert torch.allclose(first.embeddings, second.embeddings)
    assert len(first.provenance_rows) == 4
    assert {row["anchor_split"] for row in first.provenance_rows} == {"source_train"}
    assert {row["residual_reference_split"] for row in first.provenance_rows} == {"source_train"}
    assert {row["calibration_split"] for row in first.provenance_rows} == {"source_val"}
    assert {row["generation_conditioning"] for row in first.provenance_rows} == {
        "source_train_residual_reference_posterior"
    }
    assert not any("x_target" in key for row in first.provenance_rows for key in row)


def test_f2_calibrated_no_penalty_generation_mode_is_reproducible() -> None:
    model = AnchoredResidualCVAE(input_dim=2, hidden_dim=8, latent_dim=2, class_condition_dim=2)
    anchor_index = _anchor_index()
    calibration = _calibration()

    first = generate_f2_anchor_residual_embeddings(
        model=model,
        anchor_index=anchor_index,
        calibration=calibration,
        class_label=1,
        n_samples=5,
        seed=31,
        generation_mode=F2_MODE_CALIBRATED_NOISE_NO_PENALTY,
    )
    second = generate_f2_anchor_residual_embeddings(
        model=model,
        anchor_index=anchor_index,
        calibration=calibration,
        class_label=1,
        n_samples=5,
        seed=31,
        generation_mode=F2_MODE_CALIBRATED_NOISE_NO_PENALTY,
    )

    assert torch.allclose(first.embeddings, second.embeddings)
    assert "residual_calibration_scale_used" in first.diagnostics


def test_near_copy_threshold_flags_exact_source_reuse() -> None:
    source = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    synthetic = source.clone()
    source_val = source + 10.0
    diagnostics = _f2_geometry_diagnostics(
        synthetic_embeddings=synthetic,
        synthetic_labels=[0, 0, 1, 1],
        source_train_pca=source,
        source_train_labels=[0, 0, 1, 1],
        source_val_pca=source_val,
    )

    assert diagnostics["hard_near_copy_failure"] == 1
    assert diagnostics["median_nn_copy_ratio"] < 0.25


def test_f2_delta_summary_can_emit_superiority_success() -> None:
    f1 = [_alignment_row("family_f1_source_anchored_residual", "anchor_posterior_residual_mean", 0.70, 0.78, 0.08)]
    f2 = [
        _alignment_row(F2_GENERATOR_FAMILY, F2_MODE_CALIBRATED_NOISE, 0.83, 0.87, 0.04),
        _alignment_row(F2_GENERATOR_FAMILY, F2_MODE_IDENTITY_BOOTSTRAP, 0.71, 0.80, 0.09),
        _alignment_row(F2_GENERATOR_FAMILY, F2_MODE_EMPIRICAL_BOOTSTRAP, 0.72, 0.81, 0.09),
        _alignment_row(F2_GENERATOR_FAMILY, F2_MODE_TRANSFER_BOOTSTRAP, 0.73, 0.82, 0.09),
    ]
    residual = [
        {
            "heldout_center": "0",
            "generation_mode": F2_MODE_CALIBRATED_NOISE,
            "residual_energy_ratio": 1.0,
            "residual_cov_trace_ratio": 1.0,
        }
    ]

    summary = build_f2_delta_summary_rows(
        f2_alignment_rows=f2,
        f1_alignment_rows=f1,
        residual_rows=residual,
    )
    primary = [row for row in summary if row["generation_mode"] == F2_MODE_CALIBRATED_NOISE][0]

    assert primary["beats_identity_bootstrap"] == 1
    assert primary["beats_empirical_bootstrap"] == 1
    assert primary["beats_transfer_bootstrap"] == 1
    assert primary["median_center_seed_delta_gt_0"] == 1
    assert primary["decision_label"] == DECISION_SUPERIORITY_SUCCESS


def _anchor_index():
    embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.1, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    metadata = (
        {"center": "1", "label": 0, "sample_id": "a"},
        {"center": "1", "label": 0, "sample_id": "b"},
        {"center": "1", "label": 0, "sample_id": "c"},
        {"center": "1", "label": 1, "sample_id": "d"},
        {"center": "1", "label": 1, "sample_id": "e"},
        {"center": "1", "label": 1, "sample_id": "f"},
    )
    return build_source_anchor_index(
        source_projected_embeddings=embeddings,
        source_metadata=metadata,
        source_domain="1",
        label_values=(0, 1),
        neighbor_k=2,
    )


def _calibration():
    from cvae_downstream_evaluation.f2_calibrated_residual import ResidualCalibration

    return ResidualCalibration(class_scales={0: 1.2, 1: 1.1}, global_scale=1.15, rows=())


def _val_pairs(n0: int, n1: int) -> AnchorPairDataset:
    labels = torch.tensor([0] * n0 + [1] * n1, dtype=torch.long)
    refs = torch.randn(len(labels), 2)
    anchors = refs + 0.1
    return AnchorPairDataset(
        pair_targets=refs,
        anchors=anchors,
        labels=labels,
        pair_sample_ids=tuple(f"ref_{idx}" for idx in range(len(labels))),
        anchor_sample_ids=tuple(f"anchor_{idx}" for idx in range(len(labels))),
        anchor_neighbor_ranks=tuple(1 for _ in range(len(labels))),
        pair_split="source_val",
        anchor_split="source_train",
    )


def _alignment_row(
    generator_family: str,
    generation_mode: str,
    selected_bacc: float,
    oracle_bacc: float,
    gap: float,
) -> dict[str, object]:
    return {
        "heldout_center": "0",
        "experiment_seed": 42,
        "support_size": 16,
        "support_seed": 17,
        "generator_family": generator_family,
        "generation_mode": generation_mode,
        "generation_seed": 17,
        "classifier_seed": 17,
        "method": SUPPORT_NELBO_METHOD,
        "selected_expert": "1",
        "selected_bacc": selected_bacc,
        "selected_macro_f1": selected_bacc,
        "downstream_oracle_expert": "1",
        "oracle_bacc": oracle_bacc,
        "oracle_macro_f1": oracle_bacc,
        "downstream_oracle_gap_bacc": gap,
        "downstream_oracle_gap_macro_f1": gap,
        "relative_downstream_oracle_gap_pct": 0.0,
        "top1_downstream_hit": 1,
        "spearman_neg_nelbo_vs_bacc": 1.0,
        "metadata_bacc": 0.65,
        "delta_vs_metadata": selected_bacc - 0.65,
        "selection_depends_on_support": 1,
    }
