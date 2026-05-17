from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "cvae_testing"))

from cvae_downstream_evaluation.family_c import FamilyCDownstreamRow  # noqa: E402
from cvae_downstream_evaluation.family_d import (  # noqa: E402
    FAMILY_D_PRIMARY_GENERATION_MODE,
    FAMILY_D_SOURCE_TRANSFER_METHOD,
    assert_family_d_downstream_config_text,
    build_family_d_source_transfer_prior_audit_rows,
    build_family_d_source_transfer_selection_alignment_rows,
    load_family_d_downstream_config,
    validate_family_d_checkpoint_provenance,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from cvae_downstream_evaluation.schemas import SINGLE_EXPERT_ROW_TYPE  # noqa: E402


def test_family_d_downstream_config_validates() -> None:
    config_path = ROOT / "configs" / "experiments" / "family_d_discriminative_downstream_v1.yaml"
    text = config_path.read_text(encoding="utf-8")
    assert_family_d_downstream_config_text(text)
    config = load_family_d_downstream_config(config_path)
    assert config.label_values == (0, 1)
    assert config.budget_per_class == 128


def test_downstream_preflight_rejects_family_c_checkpoints_when_family_d_required() -> None:
    rows = [
        {
            "expert_domain": "0",
            "expert_family": "family_c_label_conditioned_v1",
            "condition_type": "class_label_one_hot",
            "discriminative_training_enabled": "0",
            "label_values_json": "[0, 1]",
            "class_condition_dim": "2",
            "embedding_dim": "768",
            "latent_dim": "16",
            "feature_extractor_name": "dinov2_vitb14",
            "feature_extractor_checkpoint": "facebook/dinov2-base",
            "early_stopping_metric": "source_val_total_loss",
            "lambda_prior_cls": "0.50",
            "reconstruction_loss": "mse_sum",
        }
    ]
    with pytest.raises(ProtocolError, match="not a Family D checkpoint"):
        validate_family_d_checkpoint_provenance(rows)


def test_source_transfer_prior_excludes_target_and_self_rows_and_aggregates_by_center() -> None:
    rows = [
        _row("0", "1", 0.10),
        _row("1", "1", 0.99),  # self row for candidate 1, excluded
        _row("2", "1", 0.70),
        _row("2", "1", 0.90),
        _row("3", "1", 0.80),
        _row("4", "1", 0.60),
        _row("1", "2", 0.50),
        _row("3", "2", 0.50),
        _row("4", "2", 0.50),
    ]
    audit = build_family_d_source_transfer_prior_audit_rows(rows, min_required_source_centers=3)
    candidate_1_for_target_0 = [
        row for row in audit if row["heldout_center"] == "0" and row["candidate_expert"] == "1"
    ][0]
    assert candidate_1_for_target_0["source_centers_used"] == "2|3|4"
    assert candidate_1_for_target_0["n_source_centers_used"] == 3
    assert candidate_1_for_target_0["target_heldout_rows_used"] == 0
    assert candidate_1_for_target_0["available"] == 1
    assert abs(float(candidate_1_for_target_0["prior_score"]) - 0.7333333333333334) < 1e-9


def test_source_transfer_selection_tie_breaks_by_ascending_expert_id() -> None:
    rows = []
    for heldout in ["0", "1", "2", "3", "4"]:
        for candidate in ["1", "2"]:
            if heldout != candidate:
                rows.append(_row(heldout, candidate, 0.70))
    audit = build_family_d_source_transfer_prior_audit_rows(rows, min_required_source_centers=2)
    selected_for_0 = {row["selected_expert"] for row in audit if row["heldout_center"] == "0"}
    assert selected_for_0 == {"1"}


def test_source_transfer_alignment_joins_selected_expert_to_downstream_rows() -> None:
    rows = [_row("0", "1", 0.75), _row("0", "2", 0.80), _row("1", "2", 0.70), _row("2", "1", 0.72), _row("3", "1", 0.73)]
    audit = build_family_d_source_transfer_prior_audit_rows(rows, min_required_source_centers=2)
    alignment = build_family_d_source_transfer_selection_alignment_rows(
        source_transfer_audit_rows=audit,
        downstream_rows=rows,
    )
    assert alignment
    assert all(row["method"] == FAMILY_D_SOURCE_TRANSFER_METHOD for row in alignment)
    assert all(row["selection_source"] == "family_d_source_transfer_downstream_prior_loto" for row in alignment)


def _row(heldout_center: str, candidate_expert: str, bacc: float) -> FamilyCDownstreamRow:
    return FamilyCDownstreamRow(
        heldout_center=heldout_center,
        candidate_expert=candidate_expert,
        generation_seed=17,
        classifier_seed=17,
        budget_per_class=128,
        generation_mode=FAMILY_D_PRIMARY_GENERATION_MODE,
        support_size=4,
        support_seed=17,
        support_eval_split_id=f"target{heldout_center}_seed17_random_k4",
        eval_n=100,
        eval_class_counts='{"0": 50, "1": 50}',
        target_eval_n_class0=50,
        target_eval_n_class1=50,
        target_eval_min_class_count=50,
        metric_valid_bacc=1,
        metric_valid_macro_f1=1,
        bacc=bacc,
        macro_f1=bacc,
        row_type=SINGLE_EXPERT_ROW_TYPE,
    )
