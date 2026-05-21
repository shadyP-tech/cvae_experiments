from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "cvae_testing"))

from cvae_downstream_evaluation.f1_source_anchored import (  # noqa: E402
    DECISION_GENERATOR_SUCCESS,
    F1_GENERATOR_FAMILY,
    F1_MODE_EMPIRICAL_BOOTSTRAP,
    F1_MODE_IDENTITY_BOOTSTRAP,
    F1_MODE_POSTERIOR_MEAN,
    F1_MODE_TRANSFER_BOOTSTRAP,
    AnchoredResidualCVAE,
    anchored_residual_loss_terms,
    build_anchor_pair_dataset,
    build_f1_delta_summary_rows,
    build_source_anchor_index,
    generate_anchor_residual_embeddings,
    kl_latent_dim_mean,
    residual_gaussian_nll_terms,
)
from cvae_downstream_evaluation.schemas import (  # noqa: E402
    HETEROSCEDASTIC_GENERATOR_FAMILY,
    POSTERIOR_DECODER_MEAN_GENERATION_MODE,
    SUPPORT_NELBO_METHOD,
)


def test_f1_runner_cli_help_loads() -> None:
    script = ROOT / "scripts" / "run_f1_source_anchored_residual_downstream.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--c41-artifacts-root" in result.stdout
    assert "--training-profile" in result.stdout


def test_anchored_residual_cvae_shape_contracts() -> None:
    model = AnchoredResidualCVAE(input_dim=4, hidden_dim=8, latent_dim=3, class_condition_dim=2)
    x_pair_target = torch.randn(5, 4)
    x_anchor = torch.randn(5, 4)
    y = torch.tensor([0, 1, 0, 1, 1])

    mu, logvar = model.encode(x_pair_target, x_anchor, y)
    delta_mu, delta_logvar = model.decode_residual(mu, x_anchor, y)
    fwd_delta, fwd_logvar, fwd_mu, fwd_latent_logvar = model(x_pair_target, x_anchor, y)

    assert mu.shape == (5, 3)
    assert logvar.shape == (5, 3)
    assert delta_mu.shape == (5, 4)
    assert delta_logvar.shape == (5, 4)
    assert fwd_delta.shape == (5, 4)
    assert fwd_logvar.shape == (5, 4)
    assert fwd_mu.shape == (5, 3)
    assert fwd_latent_logvar.shape == (5, 3)


def test_residual_gaussian_nll_and_kl_use_dim_mean_reductions() -> None:
    delta_true = torch.tensor([[1.0, 3.0]])
    delta_mu = torch.tensor([[0.0, 1.0]])
    delta_logvar = torch.zeros_like(delta_true)
    mu = torch.tensor([[1.0, 3.0]])
    logvar = torch.zeros_like(mu)

    nll = residual_gaussian_nll_terms(delta_mu=delta_mu, delta_true=delta_true, delta_logvar=delta_logvar)
    kl = kl_latent_dim_mean(mu, logvar)
    terms = anchored_residual_loss_terms(
        delta_mu=delta_mu,
        delta_true=delta_true,
        delta_logvar=delta_logvar,
        mu=mu,
        logvar=logvar,
    )

    assert torch.allclose(nll["squared_error_scaled"], torch.tensor([1.25]))
    assert torch.allclose(kl, torch.tensor([2.5]))
    assert torch.allclose(terms["loss"], terms["recon_nll"] + terms["kl"])


def test_anchor_pair_dataset_enforces_same_class_and_self_exclusion() -> None:
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

    pairs = build_anchor_pair_dataset(
        pair_projected_embeddings=embeddings,
        pair_metadata=metadata,
        anchor_projected_embeddings=embeddings,
        anchor_metadata=metadata,
        source_domain="1",
        label_values=(0, 1),
        pairs_per_sample=2,
        neighbor_k=2,
        seed=17,
        pair_split="source_train",
        anchor_split="source_train",
    )

    assert pairs.pair_targets.shape[0] == 12
    assert all(pair_id != anchor_id for pair_id, anchor_id in zip(pairs.pair_sample_ids, pairs.anchor_sample_ids))
    assert set(int(v) for v in pairs.labels.tolist()) == {0, 1}


def test_anchor_index_rejects_singleton_class_pool() -> None:
    embeddings = torch.randn(3, 2)
    metadata = (
        {"center": "1", "label": 0, "sample_id": "a"},
        {"center": "1", "label": 0, "sample_id": "b"},
        {"center": "1", "label": 1, "sample_id": "c"},
    )

    try:
        build_source_anchor_index(
            source_projected_embeddings=embeddings,
            source_metadata=metadata,
            source_domain="1",
            label_values=(0, 1),
            neighbor_k=2,
        )
    except Exception as exc:
        assert "at least two" in str(exc)
    else:
        raise AssertionError("F1 accepted a singleton source-train class pool")


def test_f1_generation_is_seeded_and_source_train_provenance_only() -> None:
    torch.manual_seed(13)
    model = AnchoredResidualCVAE(input_dim=2, hidden_dim=8, latent_dim=2, class_condition_dim=2)
    anchor_index = _anchor_index()

    first = generate_anchor_residual_embeddings(
        model=model,
        anchor_index=anchor_index,
        class_label=0,
        n_samples=4,
        seed=23,
        generation_mode=F1_MODE_POSTERIOR_MEAN,
        experiment_seed=42,
        heldout_center="0",
        candidate_expert="1",
    )
    second = generate_anchor_residual_embeddings(
        model=model,
        anchor_index=anchor_index,
        class_label=0,
        n_samples=4,
        seed=23,
        generation_mode=F1_MODE_POSTERIOR_MEAN,
        experiment_seed=42,
        heldout_center="0",
        candidate_expert="1",
    )

    assert torch.allclose(first.embeddings, second.embeddings)
    assert len(first.provenance_rows) == 4
    assert {row["anchor_split"] for row in first.provenance_rows} == {"source_train"}
    assert {row["residual_reference_split"] for row in first.provenance_rows} == {"source_train"}
    assert {row["same_class_anchor"] for row in first.provenance_rows} == {1}


def test_empirical_transfer_bootstrap_is_reproducible() -> None:
    model = AnchoredResidualCVAE(input_dim=2, hidden_dim=8, latent_dim=2, class_condition_dim=2)
    anchor_index = _anchor_index()

    first = generate_anchor_residual_embeddings(
        model=model,
        anchor_index=anchor_index,
        class_label=1,
        n_samples=5,
        seed=31,
        generation_mode=F1_MODE_TRANSFER_BOOTSTRAP,
    )
    second = generate_anchor_residual_embeddings(
        model=model,
        anchor_index=anchor_index,
        class_label=1,
        n_samples=5,
        seed=31,
        generation_mode=F1_MODE_TRANSFER_BOOTSTRAP,
    )

    assert torch.allclose(first.embeddings, second.embeddings)
    assert "residual_energy_ratio" in first.diagnostics


def test_f1_delta_summary_requires_beating_bootstrap_for_generator_success() -> None:
    c41 = [
        _alignment_row(
            HETEROSCEDASTIC_GENERATOR_FAMILY,
            POSTERIOR_DECODER_MEAN_GENERATION_MODE,
            selected_bacc=0.70,
            oracle_bacc=0.76,
            gap=0.06,
        )
    ]
    f1 = [
        _alignment_row(F1_GENERATOR_FAMILY, F1_MODE_POSTERIOR_MEAN, selected_bacc=0.82, oracle_bacc=0.86, gap=0.04),
        _alignment_row(F1_GENERATOR_FAMILY, F1_MODE_IDENTITY_BOOTSTRAP, selected_bacc=0.71, oracle_bacc=0.80, gap=0.09),
        _alignment_row(F1_GENERATOR_FAMILY, F1_MODE_EMPIRICAL_BOOTSTRAP, selected_bacc=0.72, oracle_bacc=0.81, gap=0.09),
        _alignment_row(F1_GENERATOR_FAMILY, F1_MODE_TRANSFER_BOOTSTRAP, selected_bacc=0.73, oracle_bacc=0.82, gap=0.09),
    ]

    summary = build_f1_delta_summary_rows(f1_alignment_rows=f1, c41_alignment_rows=c41)
    primary = [row for row in summary if row["generation_mode"] == F1_MODE_POSTERIOR_MEAN][0]

    assert primary["selected_ge_080"] == 1
    assert primary["beats_identity_bootstrap"] == 1
    assert primary["beats_empirical_bootstrap"] == 1
    assert primary["beats_transfer_bootstrap"] == 1
    assert primary["decision_label"] == DECISION_GENERATOR_SUCCESS


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


def _alignment_row(
    generator_family: str,
    generation_mode: str,
    *,
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
