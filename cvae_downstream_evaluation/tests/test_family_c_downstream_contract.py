from pathlib import Path
import math
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "cvae_testing"))

from cvae_downstream_evaluation.family_c import (  # noqa: E402
    FAMILY_C_ENSEMBLE_EXPERT_ID,
    FAMILY_C_NEGATIVE_CONTROL_ROW_TYPE,
    FAMILY_C_PRIMARY_GENERATION_MODE,
    FAMILY_C_SOURCE_TRANSFER_METHOD,
    FamilyCDownstreamRow,
    allocate_same_budget_ensemble,
    assert_family_c_config_text,
    build_family_c_baseline_comparison_rows,
    build_family_c_source_transfer_prior_audit_rows,
    build_family_c_source_transfer_selection_alignment_rows,
    candidate_level_spearman,
    classify_source_transfer_downstream_prior,
    classifier_cache_key,
    compute_family_c_oracles,
    eval_metric_validity,
    generate_label_conditioned_prior_embeddings,
    load_family_c_downstream_config,
    source_transfer_diversity_diagnostics,
    validate_family_c_downstream_matrix,
)
from cvae_downstream_evaluation.schemas import METHOD_BASELINE_ROW_TYPE, SINGLE_EXPERT_ROW_TYPE  # noqa: E402


def test_family_c_config_validates() -> None:
    config_path = ROOT / "configs" / "experiments" / "family_c_label_conditioned_downstream_v1.yaml"
    text = config_path.read_text(encoding="utf-8")
    assert_family_c_config_text(text)
    config = load_family_c_downstream_config(config_path)
    assert config.budget_per_class == 128
    assert config.label_values == (0, 1)


def test_label_conditioned_decoder_requires_y() -> None:
    torch = pytest.importorskip("torch")
    from src.models.cvae_expert import CVAEExpert

    model = CVAEExpert(input_dim=4, hidden_dim=8, latent_dim=2, class_condition_dim=2)
    z = torch.zeros((3, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="Class-condition tensor is required"):
        model.decode(z)
    y = torch.zeros((3, 2), dtype=torch.float32)
    y[:, 1] = 1.0
    decoded = model.decode(z, y=y)
    assert tuple(decoded.shape) == (3, 4)


def test_label_conditioned_generation_assigns_condition_labels() -> None:
    batch = generate_label_conditioned_prior_embeddings(
        _FakeLabelBackend(),
        expert_domain=2,
        generation_seed=17,
        budget_per_class=3,
        label_values=(0, 1),
    )
    assert list(batch.labels) == [0, 0, 0, 1, 1, 1]


def test_wrong_label_control_inverts_labels() -> None:
    batch = generate_label_conditioned_prior_embeddings(
        _FakeLabelBackend(),
        expert_domain=2,
        generation_seed=17,
        budget_per_class=2,
        label_values=(0, 1),
        wrong_label_control=True,
    )
    assert list(batch.labels) == [1, 1, 0, 0]


def test_classifier_cache_key_excludes_support_split_fields() -> None:
    key_a = classifier_cache_key(
        heldout_center="0",
        candidate_expert="1",
        generation_seed=17,
        classifier_seed=23,
        budget_per_class=128,
        generation_mode=FAMILY_C_PRIMARY_GENERATION_MODE,
    )
    key_b = classifier_cache_key(
        heldout_center="0",
        candidate_expert="1",
        generation_seed=17,
        classifier_seed=23,
        budget_per_class=128,
        generation_mode=FAMILY_C_PRIMARY_GENERATION_MODE,
    )
    assert key_a == key_b
    assert len(key_a) == 6


def test_downstream_matrix_includes_eval_split_and_rejects_duplicates() -> None:
    row = _row("1", bacc=0.70)
    assert row.support_eval_split_id == "target0_seed17_random_k4"
    assert row.eval_n == 196
    with pytest.raises(Exception, match="Duplicate Family C downstream row"):
        validate_family_c_downstream_matrix([row, row])


def test_metric_validity_rejects_missing_eval_class() -> None:
    valid = eval_metric_validity([0, 0, 1])
    invalid = eval_metric_validity([0, 0, 0])
    assert valid["metric_valid_bacc"] == 1
    assert invalid["metric_valid_bacc"] == 0
    assert invalid["target_eval_min_class_count"] == 0


def test_same_budget_ensemble_does_not_exceed_single_expert_budget() -> None:
    allocation = allocate_same_budget_ensemble(total_per_class=128, candidate_experts=["4", "1", "2", "3"])
    assert allocation == {"1": 32, "2": 32, "3": 32, "4": 32}
    assert sum(allocation.values()) == 128


def test_negative_control_and_ensemble_are_excluded_from_single_expert_oracle() -> None:
    rows = [
        _row("1", bacc=0.70, macro_f1=0.70, row_type=SINGLE_EXPERT_ROW_TYPE),
        _row("2", bacc=0.80, macro_f1=0.60, row_type=SINGLE_EXPERT_ROW_TYPE),
        _row("3", bacc=0.99, macro_f1=0.99, row_type=FAMILY_C_NEGATIVE_CONTROL_ROW_TYPE),
        _row(FAMILY_C_ENSEMBLE_EXPERT_ID, bacc=0.95, macro_f1=0.95, row_type=METHOD_BASELINE_ROW_TYPE),
    ]
    oracle = compute_family_c_oracles(rows)
    assert next(iter(oracle.values())).expert == "2"


def test_oracle_is_selected_by_bacc_then_macro_f1_then_expert_id() -> None:
    rows = [
        _row("1", bacc=0.80, macro_f1=0.10),
        _row("2", bacc=0.79, macro_f1=0.99),
        _row("3", bacc=0.80, macro_f1=0.10),
    ]
    oracle = compute_family_c_oracles(rows)
    assert next(iter(oracle.values())).expert == "1"


def test_candidate_level_spearman_uses_negative_support_score() -> None:
    value = candidate_level_spearman(
        {"1": 1.0, "2": 2.0, "3": 3.0},
        {"1": 0.9, "2": 0.8, "3": 0.7},
    )
    assert value > 0
    assert math.isclose(value, 1.0)


def test_source_transfer_prior_excludes_target_and_self_rows_and_aggregates_by_center() -> None:
    rows = [
        _row("2", heldout_center="0", bacc=0.10, support_seed=17),
        _row("2", heldout_center="1", bacc=0.90, support_seed=17),
        _row("2", heldout_center="1", bacc=0.70, support_seed=23),
        _row("2", heldout_center="2", bacc=0.99, support_seed=17),
        _row("2", heldout_center="3", bacc=0.60, support_seed=17),
        _row("1", heldout_center="0", bacc=0.10, support_seed=31),
        _row("1", heldout_center="2", bacc=0.50, support_seed=23),
        _row("1", heldout_center="3", bacc=0.50, support_seed=31),
    ]
    audit = build_family_c_source_transfer_prior_audit_rows(
        downstream_rows=rows,
        min_required_source_centers=2,
    )
    candidate = next(
        row
        for row in audit
        if row["heldout_center"] == "0" and row["candidate_expert"] == "2"
    )
    assert math.isclose(float(candidate["prior_score"]), 0.70)
    assert math.isclose(float(candidate["prior_score_std_across_source_centers"]), 0.10)
    assert math.isclose(float(candidate["prior_score_min_across_source_centers"]), 0.60)
    assert math.isclose(float(candidate["prior_score_max_across_source_centers"]), 0.80)
    assert candidate["source_centers_used"] == "1|3"
    assert candidate["n_rows_used"] == 3
    assert candidate["coverage_ok"] == 1
    assert candidate["selected_expert"] == "2"
    assert candidate["target_heldout_rows_used"] == 0
    assert candidate["self_expert_excluded_from_source_prior"] == 1


def test_source_transfer_prior_requires_coverage_and_tie_breaks_by_ascending_expert() -> None:
    rows = [
        _row("1", heldout_center="0", bacc=0.10, support_seed=17),
        _row("2", heldout_center="0", bacc=0.10, support_seed=23),
        _row("1", heldout_center="2", bacc=0.70, support_seed=17),
        _row("2", heldout_center="1", bacc=0.70, support_seed=17),
    ]
    audit_missing = build_family_c_source_transfer_prior_audit_rows(
        downstream_rows=rows,
        min_required_source_centers=2,
    )
    assert all(int(row["available"]) == 0 for row in audit_missing if row["heldout_center"] == "0")
    audit = build_family_c_source_transfer_prior_audit_rows(
        downstream_rows=rows,
        min_required_source_centers=1,
    )
    selected = {row["selected_expert"] for row in audit if row["heldout_center"] == "0"}
    assert selected == {"1"}


def test_source_transfer_alignment_reuses_existing_matrix_and_is_support_invariant() -> None:
    rows = [
        _row("1", heldout_center="0", bacc=0.60, support_size=4, support_seed=17),
        _row("2", heldout_center="0", bacc=0.80, support_size=4, support_seed=17),
        _row("1", heldout_center="0", bacc=0.61, support_size=8, support_seed=23),
        _row("2", heldout_center="0", bacc=0.81, support_size=8, support_seed=23),
        _row("1", heldout_center="2", bacc=0.50, support_seed=31),
        _row("2", heldout_center="1", bacc=0.90, support_seed=31),
    ]
    audit = build_family_c_source_transfer_prior_audit_rows(
        downstream_rows=rows,
        min_required_source_centers=1,
    )
    alignment = build_family_c_source_transfer_selection_alignment_rows(
        source_transfer_audit_rows=audit,
        downstream_rows=rows,
    )
    target_rows = [
        row
        for row in alignment
        if row["heldout_center"] == "0" and row["method"] == FAMILY_C_SOURCE_TRANSFER_METHOD
    ]
    assert {row["selected_expert"] for row in target_rows} == {"2"}
    assert {row["support_size"] for row in target_rows} == {4, 8}
    assert all(math.isnan(float(row["spearman_neg_support_score_vs_bacc"])) for row in target_rows)
    assert {row["selection_source"] for row in target_rows} == {"source_transfer_downstream_prior_loto"}


def test_source_transfer_baseline_and_decision_use_center_level_aggregation() -> None:
    alignment = []
    for center in ("0", "1"):
        alignment.extend(
            [
                _alignment(center, FAMILY_C_SOURCE_TRANSFER_METHOD, selected_bacc=0.75, gap=0.01),
                _alignment(center, "family_c_label_marginal", selected_bacc=0.70, gap=0.05),
                _alignment(center, "source_global_static_expert", selected_bacc=0.69, gap=0.06),
            ]
        )
    baseline = build_family_c_baseline_comparison_rows(alignment_rows=alignment, downstream_rows=[])
    methods = {row["method"] for row in baseline}
    assert FAMILY_C_SOURCE_TRANSFER_METHOD in methods
    audit = [_source_transfer_audit_row("0", "1"), _source_transfer_audit_row("1", "2")]
    summary = classify_source_transfer_downstream_prior(
        baseline,
        alignment_rows=alignment,
        source_transfer_audit_rows=audit,
        required_centers_improved=2,
    )
    assert summary["classification"] == "PASS"
    assert summary["metrics"]["center_pass_count"] == 2
    assert summary["metrics"]["num_unique_selected_experts"] == 2


def test_source_transfer_diversity_diagnostics_report_global_winner() -> None:
    diagnostics = source_transfer_diversity_diagnostics(
        [
            _source_transfer_audit_row("0", "4"),
            _source_transfer_audit_row("1", "4"),
            _source_transfer_audit_row("2", "1"),
        ]
    )
    assert diagnostics["num_unique_selected_experts"] == 2
    assert diagnostics["most_frequent_selected_expert"] == "4"
    assert float(diagnostics["selected_expert_entropy"]) > 0


class _FakeLabelBackend:
    def sample_label_conditioned_prior(self, domain: int, class_label: int, n_samples: int, seed: int):
        _ = seed
        return [[float(domain), float(class_label)]] * int(n_samples)


def _alignment(
    heldout_center: str,
    method: str,
    *,
    selected_bacc: float,
    gap: float,
) -> dict[str, object]:
    return {
        "heldout_center": heldout_center,
        "method": method,
        "selected_expert": "1",
        "generation_seed": 17,
        "classifier_seed": 17,
        "budget_per_class": 128,
        "generation_mode": FAMILY_C_PRIMARY_GENERATION_MODE,
        "support_size": 4,
        "support_seed": 17,
        "support_eval_split_id": f"target{heldout_center}_seed17_random_k4",
        "selected_bacc": selected_bacc,
        "selected_macro_f1": selected_bacc,
        "downstream_oracle_expert": "1",
        "oracle_bacc": selected_bacc + gap,
        "oracle_macro_f1": selected_bacc + gap,
        "downstream_oracle_gap_bacc": gap,
        "downstream_oracle_gap_macro_f1": gap,
        "top1_downstream_oracle_hit": int(gap == 0),
        "spearman_neg_support_score_vs_bacc": math.nan,
        "available": 1,
        "selection_source": "test",
    }


def _source_transfer_audit_row(heldout_center: str, selected_expert: str) -> dict[str, object]:
    return {
        "heldout_center": heldout_center,
        "candidate_expert": selected_expert,
        "prior_score": 0.75,
        "prior_score_std_across_source_centers": 0.0,
        "prior_score_min_across_source_centers": 0.75,
        "prior_score_max_across_source_centers": 0.75,
        "selected_expert": selected_expert,
        "n_source_centers_used": 3,
        "source_centers_used": "0|1|2",
        "n_rows_used": 9,
        "min_required_source_centers": 3,
        "coverage_ok": 1,
        "self_expert_excluded_from_source_prior": 1,
        "target_heldout_rows_used": 0,
        "target_eval_labels_used": 0,
        "uses_target_support_embeddings": 0,
        "uses_target_support_labels": 0,
        "uses_target_eval_labels_for_selection": 0,
        "uses_target_eval_downstream_scores_for_selection": 0,
        "selection_source": "source_transfer_downstream_prior_loto",
        "available": 1,
    }


def _row(
    expert: str,
    *,
    bacc: float,
    macro_f1: float | None = None,
    row_type: str = SINGLE_EXPERT_ROW_TYPE,
    heldout_center: str = "0",
    generation_seed: int = 17,
    classifier_seed: int = 17,
    support_size: int = 4,
    support_seed: int = 17,
    support_eval_split_id: str | None = None,
) -> FamilyCDownstreamRow:
    split_id = support_eval_split_id or f"target{heldout_center}_seed{support_seed}_random_k{support_size}"
    return FamilyCDownstreamRow(
        heldout_center=str(heldout_center),
        candidate_expert=str(expert),
        generation_seed=int(generation_seed),
        classifier_seed=int(classifier_seed),
        budget_per_class=128,
        generation_mode=FAMILY_C_PRIMARY_GENERATION_MODE,
        support_size=int(support_size),
        support_seed=int(support_seed),
        support_eval_split_id=split_id,
        eval_n=196,
        eval_class_counts='{"0": 100, "1": 96}',
        target_eval_n_class0=100,
        target_eval_n_class1=96,
        target_eval_min_class_count=96,
        metric_valid_bacc=1,
        metric_valid_macro_f1=1,
        bacc=float(bacc),
        macro_f1=float(macro_f1 if macro_f1 is not None else bacc),
        row_type=row_type,
    )
