import pytest

from midogpp_real_feature_gate.validation import ValidationError, validate_row_role_flags


def test_source_only_rows_require_target_exclusion() -> None:
    validate_row_role_flags(
        {
            "row_role": "source_only_transfer",
            "adoption_eligible": True,
            "diagnostic_only": False,
            "fit_used_target_center": False,
            "selection_used_target_labels": False,
            "target_eval_labels_used_for_scoring_only": True,
        }
    )

    with pytest.raises(ValidationError):
        validate_row_role_flags(
            {
                "row_role": "source_only_transfer",
                "adoption_eligible": True,
                "diagnostic_only": False,
                "fit_used_target_center": True,
                "selection_used_target_labels": False,
                "target_eval_labels_used_for_scoring_only": True,
            }
        )


def test_pooled_and_oracle_rows_are_diagnostic_only() -> None:
    for role in ("pooled_diagnostic_ceiling", "source_oracle_diagnostic"):
        validate_row_role_flags(
            {
                "row_role": role,
                "adoption_eligible": False,
                "diagnostic_only": True,
            }
        )

        with pytest.raises(ValidationError):
            validate_row_role_flags(
                {
                    "row_role": role,
                    "adoption_eligible": True,
                    "diagnostic_only": False,
                }
            )

