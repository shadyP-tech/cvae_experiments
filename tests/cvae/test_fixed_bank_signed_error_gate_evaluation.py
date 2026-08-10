from __future__ import annotations

from dataclasses import replace

import pytest

from midogpp_thesis.cvae.diagnostics import (
    fixed_bank_signed_error_gate as signed_error_gate,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.contracts import (
    BinaryLabel,
    PredictionRow,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.scientific_constants import (  # noqa: E501
    MIDOGPP_CENTERS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate.constants import (
    METHOD_IDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate import (
    evaluation as evaluation_module,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate.evaluation import (
    PRIMARY_CONTRASTS,
    SECONDARY_CONTRASTS,
    _evaluate_terminal_predictions,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _terminal_surface() -> tuple[
    dict[str, tuple[PredictionRow, ...]], tuple[BinaryLabel, ...]
]:
    labels: list[BinaryLabel] = []
    predictions: dict[str, list[PredictionRow]] = {
        method: [] for method in METHOD_IDS
    }
    case_labels = {
        "positive-only": (1,),
        "negative-only": (0,),
        "mixed": (0, 1),
    }
    for center in MIDOGPP_CENTERS:
        for case_suffix, values in case_labels.items():
            case_id = f"case-{center}-{case_suffix}"
            for sample_index, label in enumerate(values):
                sample_id = f"{case_id}-sample-{sample_index}"
                labels.append(
                    BinaryLabel(
                        center,
                        case_id,
                        sample_id,
                        label,
                        "terminal_evaluation",
                    )
                )
                hard_by_method = {
                    "B": 0,
                    "B_cal": label,
                    "G": 1,
                    "R_raw": 1 - label,
                    "R_safe": label,
                    "P": 0,
                }
                for method in METHOD_IDS:
                    hard_prediction = hard_by_method[method]
                    predictions[method].append(
                        PredictionRow(
                            method,
                            center,
                            case_id,
                            sample_id,
                            0.8 if hard_prediction else 0.2,
                            hard_prediction,
                        )
                    )
    return (
        {method: tuple(rows) for method, rows in predictions.items()},
        tuple(labels),
    )


def test_terminal_endpoint_retains_single_class_cases_without_case_bacc() -> None:
    predictions, labels = _terminal_surface()
    result = _evaluate_terminal_predictions(
        predictions_by_method=predictions,
        labels=labels,
        bootstrap_replicates=48,
        bootstrap_seed=1729,
        bootstrap_workers=1,
        multiprocessing_start_method="spawn",
        bootstrap_threads_per_worker=1,
    )

    assert tuple(row.method_id for row in result.method_results) == METHOD_IDS
    assert all(
        len(row.case_confusions) == len(MIDOGPP_CENTERS) * 3
        for row in result.method_results
    )
    assert all(
        row.single_class_case_count == len(MIDOGPP_CENTERS) * 2
        for row in result.method_results
    )
    assert all(
        any(count.n_positive == 0 for count in row.case_confusions)
        and any(count.n_negative == 0 for count in row.case_confusions)
        for row in result.method_results
    )
    score_by_method = {
        row.method_id: row.equal_center_exact_bacc for row in result.method_results
    }
    assert score_by_method == {
        "B": 0.5,
        "B_cal": 1.0,
        "G": 0.5,
        "R_raw": 0.0,
        "R_safe": 1.0,
        "P": 0.5,
    }

    expected_pairs = PRIMARY_CONTRASTS + SECONDARY_CONTRASTS
    assert tuple(
        (
            row.equal_center.challenger_method,
            row.equal_center.reference_method,
        )
        for row in result.contrasts
    ) == expected_pairs
    assert all(row.bootstrap is not None for row in result.primary_contrasts)
    assert all(row.bootstrap is None for row in result.secondary_contrasts)
    assert [row.equal_center.mean_difference for row in result.primary_contrasts] == [
        0.0,
        0.5,
        0.5,
    ]
    assert [row.equal_center.mean_difference for row in result.secondary_contrasts] == [
        -1.0,
        0.5,
    ]

    payload = result.to_payload()
    assert payload["single_class_cases_retained"] is True
    assert payload["per_case_bacc_stored_or_used"] is False
    assert "evidence_status" not in payload
    assert "terminal_consumed_test_diagnostic_only" not in payload
    assert "policy_update_authorized" not in payload
    for method_payload in payload["methods"]:
        assert method_payload["per_case_bacc_stored_or_used"] is False
        assert all(
            set(row)
            == {
                "method_id",
                "target_center",
                "case_id",
                "n_positive",
                "true_positive",
                "n_negative",
                "true_negative",
            }
            for row in method_payload["case_confusions"]
        )


def test_bootstrap_is_deterministic_across_input_order_and_spawn_workers() -> None:
    predictions, labels = _terminal_surface()
    reversed_predictions = {
        method: tuple(reversed(predictions[method])) for method in reversed(METHOD_IDS)
    }
    serial = _evaluate_terminal_predictions(
        predictions_by_method=reversed_predictions,
        labels=tuple(reversed(labels)),
        bootstrap_replicates=64,
        bootstrap_seed=8102026,
        bootstrap_workers=1,
        multiprocessing_start_method="spawn",
        bootstrap_threads_per_worker=1,
    )
    spawned = _evaluate_terminal_predictions(
        predictions_by_method=predictions,
        labels=labels,
        bootstrap_replicates=64,
        bootstrap_seed=8102026,
        bootstrap_workers=3,
        multiprocessing_start_method="spawn",
        bootstrap_threads_per_worker=1,
    )

    assert serial.to_payload() == spawned.to_payload()
    assert serial.scientific_result_hash == spawned.scientific_result_hash
    assert all(
        row.bootstrap is not None and row.bootstrap.replicate_count == 64
        for row in spawned.primary_contrasts
    )


def test_terminal_evaluation_rejects_nonterminal_labels_and_surface_drift() -> None:
    predictions, labels = _terminal_surface()
    support_labels = (replace(labels[0], label_scope="target_support"), *labels[1:])
    with pytest.raises(ProtocolError, match="terminal-evaluation labels"):
        _evaluate_terminal_predictions(
            predictions_by_method=predictions,
            labels=support_labels,
            bootstrap_replicates=8,
        )

    missing_method = dict(predictions)
    missing_method.pop("P")
    with pytest.raises(ProtocolError, match="exactly six methods"):
        _evaluate_terminal_predictions(
            predictions_by_method=missing_method,
            labels=labels,
            bootstrap_replicates=8,
        )

    wrong_method_rows = dict(predictions)
    wrong_method_rows["G"] = predictions["B"]
    with pytest.raises(ProtocolError, match="mapping key"):
        _evaluate_terminal_predictions(
            predictions_by_method=wrong_method_rows,
            labels=labels,
            bootstrap_replicates=8,
        )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("bootstrap_workers", 0, "positive integer"),
        ("bootstrap_workers", 5, "frozen budget"),
        ("bootstrap_threads_per_worker", 0, "positive integer"),
        ("bootstrap_threads_per_worker", 4, "frozen budget"),
        ("bootstrap_replicates", 10_001, "canonical replicate budget"),
        ("multiprocessing_start_method", "fork", "requires spawn"),
    ),
)
def test_terminal_evaluation_rejects_invalid_bootstrap_runtime(
    keyword: str, value: object, message: str
) -> None:
    predictions, labels = _terminal_surface()
    arguments: dict[str, object] = {
        "predictions_by_method": predictions,
        "labels": labels,
        "bootstrap_replicates": 8,
        keyword: value,
    }
    with pytest.raises(ProtocolError, match=message):
        _evaluate_terminal_predictions(**arguments)  # type: ignore[arg-type]


def test_pure_terminal_evaluator_is_internal_to_the_sealed_adapter() -> None:
    assert hasattr(signed_error_gate, "evaluate_sealed_fold_products")
    assert not hasattr(evaluation_module, "evaluate_terminal_predictions")
    assert "_evaluate_terminal_predictions" not in evaluation_module.__all__
    assert "evaluate_terminal_predictions" not in signed_error_gate.__all__
