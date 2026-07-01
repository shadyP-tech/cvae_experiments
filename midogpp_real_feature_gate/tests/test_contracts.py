from midogpp_real_feature_gate.contracts import (
    ELIGIBLE_CENTERS,
    QUARANTINE_CENTERS,
    REQUIRED_MATRIX_COLUMNS,
    SCHEMA_VERSION,
)
from midogpp_real_feature_gate.splits import is_eligible_center, is_quarantine_center


def test_schema_version_is_frozen() -> None:
    assert SCHEMA_VERSION == "midogpp_real_feature_transfer_ceiling_v1"


def test_center_sets_are_explicit_and_disjoint() -> None:
    assert ELIGIBLE_CENTERS == ("0", "1", "2", "3", "5", "6", "7", "8", "9")
    assert QUARANTINE_CENTERS == ("4",)
    assert not set(ELIGIBLE_CENTERS).intersection(QUARANTINE_CENTERS)
    assert is_eligible_center("0")
    assert not is_eligible_center("4")
    assert is_quarantine_center("4")


def test_required_columns_include_protocol_flags_and_metrics() -> None:
    for column in (
        "row_role",
        "adoption_eligible",
        "diagnostic_only",
        "fit_used_target_center",
        "selection_used_target_labels",
        "target_eval_labels_used_for_scoring_only",
        "predicted_positive_rate",
        "auroc",
        "pr_auc",
        "pr_auc_baseline",
    ):
        assert column in REQUIRED_MATRIX_COLUMNS

