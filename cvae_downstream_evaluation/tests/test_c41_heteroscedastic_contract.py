from pathlib import Path
import sys
import subprocess

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "cvae_testing"))

from cvae_downstream_evaluation.c41_heteroscedastic import (  # noqa: E402
    GENERATION_MODE_POSTERIOR_DECODER_MEAN,
    GENERATION_MODE_POSTERIOR_DECODER_NOISE,
    GENERATOR_FAMILY_HETEROSCEDASTIC,
    GENERATOR_FAMILY_PLAIN_CLASS_CONDITIONAL,
    SELECTED_EXPERT_IDS_SOURCE,
    fit_source_train_pca_projection,
    build_source_train_reference_pools,
    c41_routing_provenance_fields,
    generate_posterior_sampled_embeddings,
)
from cvae_downstream_evaluation.c41_workstation import (  # noqa: E402
    DECISION_PROTOCOL_FAILURE,
    DECISION_RANK_INSTABILITY,
    DECISION_SUCCESS,
    assert_selected_expert_invariant,
    build_c41_delta_summary_rows,
    safe_support_selection_units_from_paths,
)
from cvae_downstream_evaluation.downstream import (  # noqa: E402
    CandidateDownstreamRow,
    validate_candidate_downstream_matrix,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from cvae_downstream_evaluation.schemas import SUPPORT_NELBO_METHOD  # noqa: E402
from src.models.cvae_expert import CVAEExpert  # noqa: E402


def test_source_train_pca_projection_uses_only_source_train_rows() -> None:
    train_embeddings = torch.eye(6, dtype=torch.float32)
    train_metadata = (
        {"magnification": "1", "label": 0},
        {"magnification": "1", "label": 1},
        {"magnification": "1", "label": 0},
        {"magnification": "2", "label": 0},
        {"magnification": "2", "label": 1},
        {"magnification": "2", "label": 1},
    )

    projection = fit_source_train_pca_projection(
        train_embeddings=train_embeddings,
        train_metadata=train_metadata,
        source_domain="1",
        seed=42,
        n_components=2,
    )

    assert projection.fit_split == "source_train"
    assert projection.source_domain == "1"
    assert projection.effective_components == 2
    assert projection.provenance()["fit_split"] == "source_train"


def test_c41_runner_cli_help_loads() -> None:
    script = ROOT / "scripts" / "run_c41_heteroscedastic_downstream.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--training-profile" in result.stdout
    assert "--allow-legacy-audit-columns" in result.stdout


def test_source_train_reference_pools_do_not_accept_val_metadata() -> None:
    train_projected = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    train_metadata = (
        {"magnification": "1", "label": 0, "split": "train"},
        {"magnification": "1", "label": 1, "split": "train"},
        {"magnification": "1", "label": 0, "split": "train"},
        {"magnification": "2", "label": 0, "split": "train"},
        {"magnification": "2", "label": 1, "split": "train"},
        {"magnification": "2", "label": 1, "split": "train"},
    )

    pools = build_source_train_reference_pools(
        train_projected_embeddings=train_projected,
        train_metadata=train_metadata,
        source_domain="1",
        label_values=(0, 1),
    )

    assert set(pools) == {0, 1}
    assert pools[0].shape[0] == 2
    assert pools[1].shape[0] == 1


def test_c41_generation_modes_are_seeded_and_distinct() -> None:
    model = CVAEExpert(
        input_dim=4,
        hidden_dim=8,
        latent_dim=2,
        class_condition_dim=2,
        decoder_likelihood="gaussian_diag",
    )
    refs = torch.randn(5, 4)

    mean_a = generate_posterior_sampled_embeddings(
        model=model,
        reference_pool=refs,
        class_label=1,
        n_samples=4,
        seed=17,
        generation_mode=GENERATION_MODE_POSTERIOR_DECODER_MEAN,
    )
    mean_b = generate_posterior_sampled_embeddings(
        model=model,
        reference_pool=refs,
        class_label=1,
        n_samples=4,
        seed=17,
        generation_mode=GENERATION_MODE_POSTERIOR_DECODER_MEAN,
    )
    noise = generate_posterior_sampled_embeddings(
        model=model,
        reference_pool=refs,
        class_label=1,
        n_samples=4,
        seed=17,
        generation_mode=GENERATION_MODE_POSTERIOR_DECODER_NOISE,
    )

    assert torch.allclose(mean_a.embeddings, mean_b.embeddings)
    assert not torch.allclose(mean_a.embeddings, noise.embeddings)
    assert mean_a.diagnostics["decoder_noise_energy_ratio"] == 0.0
    assert noise.diagnostics["decoder_noise_energy_ratio"] > 0.0


def test_generator_family_prevents_downstream_matrix_key_collision() -> None:
    base = {
        "experiment_seed": 42,
        "heldout_center": "0",
        "candidate_expert": "1",
        "generation_mode": GENERATION_MODE_POSTERIOR_DECODER_MEAN,
        "budget_per_class": 128,
        "generation_seed": 17,
        "classifier_seed": 17,
        "bacc": 0.7,
        "macro_f1": 0.65,
    }
    plain = CandidateDownstreamRow(**base, generator_family=GENERATOR_FAMILY_PLAIN_CLASS_CONDITIONAL)
    hetero = CandidateDownstreamRow(**base, generator_family=GENERATOR_FAMILY_HETEROSCEDASTIC)

    validate_candidate_downstream_matrix([plain, hetero])


def test_c41_selected_rows_carry_baseline_router_provenance() -> None:
    fields = c41_routing_provenance_fields()
    row = CandidateDownstreamRow(
        experiment_seed=42,
        heldout_center="0",
        candidate_expert="1",
        generator_family=GENERATOR_FAMILY_HETEROSCEDASTIC,
        generation_mode=GENERATION_MODE_POSTERIOR_DECODER_NOISE,
        budget_per_class=128,
        generation_seed=17,
        classifier_seed=17,
        bacc=0.7,
        macro_f1=0.65,
        **fields,
    )

    csv_row = row.to_csv_row()
    assert csv_row["routing_scores_recomputed_for_heteroscedastic"] == 0
    assert csv_row["selected_expert_ids_source"] == SELECTED_EXPERT_IDS_SOURCE


def test_c41_matrix_key_includes_support_condition_but_dedupes_utility_context() -> None:
    base = {
        "experiment_seed": 42,
        "heldout_center": "0",
        "candidate_expert": "1",
        "generator_family": GENERATOR_FAMILY_HETEROSCEDASTIC,
        "generation_mode": GENERATION_MODE_POSTERIOR_DECODER_MEAN,
        "budget_per_class": 128,
        "generation_seed": 17,
        "classifier_seed": 17,
        "bacc": 0.7,
        "macro_f1": 0.65,
    }
    support17 = CandidateDownstreamRow(**base, support_size=16, support_seed=17)
    support23 = CandidateDownstreamRow(**base, support_size=16, support_seed=23)

    validate_candidate_downstream_matrix([support17, support23])
    assert support17.primary_key() != support23.primary_key()
    assert support17.resolved_utility_context_key() == support23.resolved_utility_context_key()


def test_c41_duplicate_utility_context_rejects_metric_drift() -> None:
    base = {
        "experiment_seed": 42,
        "heldout_center": "0",
        "candidate_expert": "1",
        "generator_family": GENERATOR_FAMILY_HETEROSCEDASTIC,
        "generation_mode": GENERATION_MODE_POSTERIOR_DECODER_MEAN,
        "budget_per_class": 128,
        "generation_seed": 17,
        "classifier_seed": 17,
        "macro_f1": 0.65,
    }
    first = CandidateDownstreamRow(**base, support_size=16, support_seed=17, bacc=0.70)
    drifted = CandidateDownstreamRow(**base, support_size=16, support_seed=23, bacc=0.72)

    try:
        validate_candidate_downstream_matrix([first, drifted])
    except ProtocolError:
        pass
    else:
        raise AssertionError("support-replicated utility metric drift was not rejected")


def test_c41_safe_support_loader_rejects_oracle_columns(tmp_path: Path) -> None:
    path = tmp_path / "support_response_sample_selections.csv"
    path.write_text(
        "fold_query_domain,seed,support_size_requested,support_seed,method,selected_expert,"
        "candidate_experts,support_nelbo_by_expert_json,target_expert_excluded,"
        "support_eval_split_id,oracle_expert\n"
        '0,42,16,17,support_set_nelbo_top1,1,1|2,"{""1"": 1.0, ""2"": 2.0}",1,split,1\n',
        encoding="utf-8",
    )

    try:
        safe_support_selection_units_from_paths([path])
    except ProtocolError:
        pass
    else:
        raise AssertionError("C4.1 safe loader accepted oracle columns")


def test_c41_selected_expert_invariance_detects_protocol_drift() -> None:
    rows = [
        _alignment_row("plain", GENERATION_MODE_POSTERIOR_DECODER_MEAN, "1"),
        _alignment_row("hetero", GENERATION_MODE_POSTERIOR_DECODER_MEAN, "2"),
    ]

    try:
        assert_selected_expert_invariant(rows)
    except ProtocolError:
        pass
    else:
        raise AssertionError("selected expert changes across modes were not rejected")


def test_c41_delta_summary_labels_success_and_rank_instability() -> None:
    rows = [
        _alignment_row(GENERATOR_FAMILY_PLAIN_CLASS_CONDITIONAL, GENERATION_MODE_POSTERIOR_DECODER_MEAN, "1", oracle_bacc=0.70, selected_bacc=0.68, oracle="1"),
        _alignment_row(GENERATOR_FAMILY_HETEROSCEDASTIC, GENERATION_MODE_POSTERIOR_DECODER_MEAN, "1", oracle_bacc=0.73, selected_bacc=0.69, oracle="1"),
        _alignment_row(GENERATOR_FAMILY_HETEROSCEDASTIC, GENERATION_MODE_POSTERIOR_DECODER_NOISE, "1", oracle_bacc=0.70, selected_bacc=0.68, oracle="2"),
    ]

    summary = build_c41_delta_summary_rows(alignment_rows=rows)
    by_mode = {row["generation_mode"]: row for row in summary}
    assert by_mode[GENERATION_MODE_POSTERIOR_DECODER_MEAN]["decision_label"] == DECISION_SUCCESS
    assert by_mode[GENERATION_MODE_POSTERIOR_DECODER_NOISE]["oracle_expert_changed_vs_plain"] == 1
    assert by_mode[GENERATION_MODE_POSTERIOR_DECODER_NOISE]["decision_label"] == DECISION_RANK_INSTABILITY


def test_c41_delta_summary_marks_selected_expert_change_as_protocol_failure() -> None:
    rows = [
        _alignment_row(GENERATOR_FAMILY_PLAIN_CLASS_CONDITIONAL, GENERATION_MODE_POSTERIOR_DECODER_MEAN, "1", oracle_bacc=0.70, selected_bacc=0.68, oracle="1"),
        _alignment_row(GENERATOR_FAMILY_HETEROSCEDASTIC, GENERATION_MODE_POSTERIOR_DECODER_MEAN, "2", oracle_bacc=0.73, selected_bacc=0.69, oracle="1"),
    ]

    summary = build_c41_delta_summary_rows(alignment_rows=rows)
    assert summary[0]["selected_expert_changed_across_modes"] == 1
    assert summary[0]["decision_label"] == DECISION_PROTOCOL_FAILURE


def _alignment_row(
    generator_family: str,
    generation_mode: str,
    selected: str,
    *,
    oracle_bacc: float = 0.70,
    selected_bacc: float = 0.68,
    oracle: str = "1",
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
