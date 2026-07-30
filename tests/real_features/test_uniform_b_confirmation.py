from __future__ import annotations

from pathlib import Path

import pytest

from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError
from midogpp_thesis.real_features.classifier_reference.uniform_b_confirmation.artifacts import (
    prospective_paired_case_bootstrap,
)
from midogpp_thesis.real_features.classifier_reference.uniform_b_confirmation.config import (
    ConfirmationRule,
    EXPECTED_TEST_ROWS,
    load_uniform_b_confirmation_config,
    load_uniform_b_test_cache_config,
)
from midogpp_thesis.real_features.classifier_reference.uniform_b_confirmation.runner import (
    _summary,
)


CACHE_CONFIG = Path(
    "datasets/midogpp/configs/uniform_b_v3_prospective_test_confirmation_v1.yaml"
)
RUN_CONFIG = Path(
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v3_prospective_test_confirmation_v1.yaml"
)


def test_prospective_configs_freeze_untouched_test_protocol() -> None:
    cache = load_uniform_b_test_cache_config(CACHE_CONFIG)
    run = load_uniform_b_confirmation_config(RUN_CONFIG)

    assert cache.expected_test_rows == EXPECTED_TEST_ROWS == 9928
    assert cache.device == "cuda"
    assert run.confirmation_rule == ConfirmationRule()
    assert run.heldout_centers == ("0", "1", "2", "3", "5", "6", "7", "8", "9")


def test_prospective_config_rejects_posthoc_rule_change(tmp_path: Path) -> None:
    text = RUN_CONFIG.read_text(encoding="utf-8").replace(
        "minimum_mean_bacc_delta: 0.02", "minimum_mean_bacc_delta: 0.00"
    )
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ProtocolError, match="confirmation rule"):
        load_uniform_b_confirmation_config(path)


def test_prospective_bootstrap_marks_within_center_scope() -> None:
    rows = []
    for center in ("0", "1"):
        for role, predictions in (
            ("canonical_a", (0, 1, 0, 1)),
            ("uniform_b", (0, 1, 1, 1)),
        ):
            for index, (label, prediction) in enumerate(
                zip((0, 1, 1, 0), predictions, strict=True)
            ):
                rows.append(
                    {
                        "heldout_center": center,
                        "role": role,
                        "case_id": f"{center}-case-{index // 2}",
                        "label": label,
                        "prediction": prediction,
                    }
                )
    result = prospective_paired_case_bootstrap(
        rows, seed=42, valid_replicates=20, max_attempts=200
    )

    assert result["valid_replicates"] == 20
    assert result["covers_new_case_uncertainty_within_centers"] is True
    assert result["covers_new_center_uncertainty"] is False


def test_confirmation_summary_applies_all_predeclared_gates() -> None:
    config = load_uniform_b_confirmation_config(RUN_CONFIG)
    comparisons = [
        {
            "canonical_a_bacc": 0.70,
            "uniform_b_bacc": 0.74,
            "delta_bacc": 0.04,
        }
        for _center in config.heldout_centers
    ]
    bootstrap = {"percentile_2_5": 0.01}

    summary = _summary(config, comparisons, bootstrap)

    assert summary["decision"] == "CONFIRMED_WITHIN_CENTER"
    assert summary["confirmation_passed"] is True
    assert summary["may_replace_canonical_reference"] is False
