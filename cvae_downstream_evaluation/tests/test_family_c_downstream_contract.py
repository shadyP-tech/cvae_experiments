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
    FamilyCDownstreamRow,
    allocate_same_budget_ensemble,
    assert_family_c_config_text,
    candidate_level_spearman,
    classifier_cache_key,
    compute_family_c_oracles,
    eval_metric_validity,
    generate_label_conditioned_prior_embeddings,
    load_family_c_downstream_config,
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


class _FakeLabelBackend:
    def sample_label_conditioned_prior(self, domain: int, class_label: int, n_samples: int, seed: int):
        _ = seed
        return [[float(domain), float(class_label)]] * int(n_samples)


def _row(
    expert: str,
    *,
    bacc: float,
    macro_f1: float | None = None,
    row_type: str = SINGLE_EXPERT_ROW_TYPE,
) -> FamilyCDownstreamRow:
    return FamilyCDownstreamRow(
        heldout_center="0",
        candidate_expert=str(expert),
        generation_seed=17,
        classifier_seed=17,
        budget_per_class=128,
        generation_mode=FAMILY_C_PRIMARY_GENERATION_MODE,
        support_size=4,
        support_seed=17,
        support_eval_split_id="target0_seed17_random_k4",
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
