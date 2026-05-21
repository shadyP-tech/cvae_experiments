from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "cvae_testing"))

from cvae_downstream_evaluation.c41_heteroscedastic import GENERATION_MODE_POSTERIOR_DECODER_MEAN  # noqa: E402
from cvae_downstream_evaluation.c42_latent_gmm import (  # noqa: E402
    C42_LATENT_GMM_K1_GENERATION_MODE,
    fit_source_class_latent_gmm,
    generate_latent_gmm_decoder_mean,
    generate_standard_prior_decoder_mean,
    generated_embedding_diagnostics,
)
from cvae_downstream_evaluation.c42_workstation import (  # noqa: E402
    DECISION_CEILING_ONLY,
    DECISION_PASS,
    DECISION_PROTOCOL_FAILURE,
    DECISION_REPLAY_MISMATCH,
    build_c42_delta_summary_rows,
)
from cvae_downstream_evaluation.schemas import (  # noqa: E402
    C42_POSTERIOR_REPLAY_GENERATION_MODE,
    C42_STANDARD_PRIOR_REPLAY_GENERATION_MODE,
    LATENT_GMM_PRIOR_GENERATOR_FAMILY,
    PLAIN_CLASS_CONDITIONAL_GENERATOR_FAMILY,
    SUPPORT_NELBO_METHOD,
)
from src.models.cvae_expert import CVAEExpert  # noqa: E402


def test_c42_runner_cli_help_loads() -> None:
    script = ROOT / "scripts" / "run_c42_latent_gmm_downstream.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--c41-artifacts-root" in result.stdout
    assert "--covariance-floor" in result.stdout


def test_source_class_latent_gmm_clips_components_and_samples_seededly() -> None:
    torch.manual_seed(7)
    model = CVAEExpert(input_dim=4, hidden_dim=8, latent_dim=2, class_condition_dim=2)
    projected = torch.randn(5, 4)
    labels = torch.tensor([1, 1, 1, 0, 0], dtype=torch.long)

    prior = fit_source_class_latent_gmm(
        model=model,
        projected_embeddings=projected,
        labels=labels,
        experiment_seed=42,
        source_domain="1",
        class_label=1,
        requested_components=4,
        fit_seed=17,
    )
    sample_a = prior.sample(6, seed=23, device=torch.device("cpu"))
    sample_b = prior.sample(6, seed=23, device=torch.device("cpu"))

    assert prior.requested_components == 4
    assert prior.effective_components == 3
    assert prior.component_clipped == 1
    assert prior.diagnostics["class_count"] == 3.0
    assert prior.diagnostics["effective_components"] == 3.0
    assert prior.diagnostics["component_clipped"] == 1.0
    assert torch.allclose(sample_a, sample_b)


def test_c42_generation_modes_emit_shapes_and_diagnostics() -> None:
    torch.manual_seed(11)
    model = CVAEExpert(input_dim=4, hidden_dim=8, latent_dim=2, class_condition_dim=2)
    projected = torch.randn(8, 4)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    prior = fit_source_class_latent_gmm(
        model=model,
        projected_embeddings=projected,
        labels=labels,
        experiment_seed=42,
        source_domain="1",
        class_label=0,
        requested_components=1,
        fit_seed=17,
    )

    gmm = generate_latent_gmm_decoder_mean(
        model=model,
        prior=prior,
        class_label=0,
        n_samples=5,
        seed=31,
        generation_mode=C42_LATENT_GMM_K1_GENERATION_MODE,
    )
    standard = generate_standard_prior_decoder_mean(model=model, class_label=1, n_samples=5, seed=31)

    assert gmm.embeddings.shape == (5, 4)
    assert standard.embeddings.shape == (5, 4)
    assert gmm.labels.tolist() == [0, 0, 0, 0, 0]
    assert "gmm_vs_posterior_mu_mmd" in gmm.diagnostics
    assert standard.generation_mode == C42_STANDARD_PRIOR_REPLAY_GENERATION_MODE


def test_generated_embedding_diagnostics_include_class_counts() -> None:
    diagnostics = generated_embedding_diagnostics(
        synthetic_embeddings=torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        synthetic_labels=[0, 0, 1, 1],
        source_train_embeddings=torch.tensor([[0.0, 0.0], [0.5, 0.5], [2.0, 2.0]]),
        source_train_labels=[0, 1, 1],
    )

    assert diagnostics["synthetic_count_class_0"] == 2.0
    assert diagnostics["synthetic_count_class_1"] == 2.0
    assert diagnostics["real_source_train_count_class_0"] == 1.0
    assert diagnostics["real_source_train_count_class_1"] == 2.0
    assert diagnostics["synthetic_classifier_train_class_balance"] == 1.0


def test_c42_delta_summary_labels_pass_and_ceiling_only() -> None:
    c41 = [
        _alignment_row(
            PLAIN_CLASS_CONDITIONAL_GENERATOR_FAMILY,
            GENERATION_MODE_POSTERIOR_DECODER_MEAN,
            "1",
            selected_bacc=0.68,
            oracle_bacc=0.70,
            oracle="1",
        )
    ]
    c42 = [
        _alignment_row(LATENT_GMM_PRIOR_GENERATOR_FAMILY, C42_POSTERIOR_REPLAY_GENERATION_MODE, "1", selected_bacc=0.68, oracle_bacc=0.70, oracle="1"),
        _alignment_row(LATENT_GMM_PRIOR_GENERATOR_FAMILY, C42_STANDARD_PRIOR_REPLAY_GENERATION_MODE, "1", selected_bacc=0.67, oracle_bacc=0.69, oracle="1"),
        _alignment_row(LATENT_GMM_PRIOR_GENERATOR_FAMILY, C42_LATENT_GMM_K1_GENERATION_MODE, "1", selected_bacc=0.72, oracle_bacc=0.73, oracle="1"),
    ]

    summary = build_c42_delta_summary_rows(c42_alignment_rows=c42, c41_alignment_rows=c41)

    assert summary[0]["plain_replay_matches_c41_within_tolerance"] == 1
    assert summary[0]["oracle_bacc_delta_vs_plain_retrained"] > 0.02
    assert summary[0]["decision_label"] == DECISION_PASS

    c42[-1] = _alignment_row(
        LATENT_GMM_PRIOR_GENERATOR_FAMILY,
        C42_LATENT_GMM_K1_GENERATION_MODE,
        "1",
        selected_bacc=0.67,
        oracle_bacc=0.73,
        oracle="1",
    )
    summary = build_c42_delta_summary_rows(c42_alignment_rows=c42, c41_alignment_rows=c41)
    assert summary[0]["decision_label"] == DECISION_CEILING_ONLY


def test_c42_delta_summary_marks_replay_mismatch_and_selected_drift() -> None:
    c41 = [
        _alignment_row(
            PLAIN_CLASS_CONDITIONAL_GENERATOR_FAMILY,
            GENERATION_MODE_POSTERIOR_DECODER_MEAN,
            "1",
            selected_bacc=0.68,
            oracle_bacc=0.70,
            oracle="1",
        )
    ]
    replay_mismatch_rows = [
        _alignment_row(LATENT_GMM_PRIOR_GENERATOR_FAMILY, C42_POSTERIOR_REPLAY_GENERATION_MODE, "1", selected_bacc=0.681, oracle_bacc=0.70, oracle="1"),
        _alignment_row(LATENT_GMM_PRIOR_GENERATOR_FAMILY, C42_LATENT_GMM_K1_GENERATION_MODE, "1", selected_bacc=0.72, oracle_bacc=0.73, oracle="1"),
    ]
    selected_drift_rows = [
        _alignment_row(LATENT_GMM_PRIOR_GENERATOR_FAMILY, C42_POSTERIOR_REPLAY_GENERATION_MODE, "1", selected_bacc=0.68, oracle_bacc=0.70, oracle="1"),
        _alignment_row(LATENT_GMM_PRIOR_GENERATOR_FAMILY, C42_LATENT_GMM_K1_GENERATION_MODE, "2", selected_bacc=0.72, oracle_bacc=0.73, oracle="1"),
    ]

    replay_summary = build_c42_delta_summary_rows(c42_alignment_rows=replay_mismatch_rows, c41_alignment_rows=c41)
    drift_summary = build_c42_delta_summary_rows(c42_alignment_rows=selected_drift_rows, c41_alignment_rows=c41)

    assert replay_summary[0]["decision_label"] == DECISION_REPLAY_MISMATCH
    assert drift_summary[0]["selected_expert_changed_across_modes"] == 1
    assert drift_summary[0]["decision_label"] == DECISION_PROTOCOL_FAILURE


def _alignment_row(
    generator_family: str,
    generation_mode: str,
    selected: str,
    *,
    oracle_bacc: float,
    selected_bacc: float,
    oracle: str,
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
        "selected_expert": selected,
        "selected_bacc": selected_bacc,
        "selected_macro_f1": selected_bacc,
        "downstream_oracle_expert": oracle,
        "oracle_bacc": oracle_bacc,
        "oracle_macro_f1": oracle_bacc,
        "downstream_oracle_gap_bacc": oracle_bacc - selected_bacc,
        "downstream_oracle_gap_macro_f1": oracle_bacc - selected_bacc,
        "relative_downstream_oracle_gap_pct": 0.0,
        "top1_downstream_hit": int(selected == oracle),
        "spearman_neg_nelbo_vs_bacc": 1.0,
        "metadata_bacc": 0.65,
        "delta_vs_metadata": selected_bacc - 0.65,
        "selection_depends_on_support": 1,
    }
