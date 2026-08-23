"""Exact terminal method, center, and case inventory validation."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from ....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ....protocol import ProtocolError
from ..identity import METHOD_MENU, P_METHOD_ID
from ..inventory import CANONICAL_CASE_COUNT, CANONICAL_ROW_COUNT


def validate_terminal_row_inventory(
    *,
    method_rows: Sequence[Mapping[str, object]],
    center_rows: Sequence[Mapping[str, object]],
    case_rows: Sequence[Mapping[str, object]],
    case_ids_by_center: Mapping[str, Sequence[str]],
    case_sample_counts_by_center: Mapping[str, Mapping[str, int]],
) -> dict[str, int]:
    """Require the canonical 6 x 9 x 218 terminal diagnostic rectangles."""

    methods = tuple(dict(row) for row in method_rows)
    centers = tuple(dict(row) for row in center_rows)
    cases = tuple(dict(row) for row in case_rows)
    expected_cases = {
        str(center): tuple(str(value) for value in case_ids_by_center.get(center, ()))
        for center in CENTERS
    }
    expected_case_counts = {
        center: {
            str(case_id): count
            for case_id, count in case_sample_counts_by_center.get(
                center, {}
            ).items()
        }
        for center in CENTERS
    }
    if (
        tuple(row.get("method_id") for row in methods) != METHOD_MENU
        or sum(len(values) for values in expected_cases.values())
        != CANONICAL_CASE_COUNT
        or any(
            not values
            or len(values) != len(set(values))
            or tuple(sorted(values)) != values
            for values in expected_cases.values()
        )
        or any(
            tuple(expected_case_counts[center]) != expected_cases[center]
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for value in expected_case_counts[center].values()
            )
            for center in CENTERS
        )
        or sum(
            value
            for center in CENTERS
            for value in expected_case_counts[center].values()
        )
        != CANONICAL_ROW_COUNT
    ):
        raise ProtocolError("P-DCAPS v2 terminal method or case inventory drifted.")

    expected_center_keys = tuple(
        (method, center) for method in METHOD_MENU for center in CENTERS
    )
    observed_center_keys = tuple(
        (str(row.get("method_id")), str(row.get("target_center")))
        for row in centers
    )
    if observed_center_keys != expected_center_keys:
        raise ProtocolError("P-DCAPS v2 terminal center rectangle drifted.")
    center_by_key = {
        (str(row["method_id"]), str(row["target_center"])): row
        for row in centers
    }
    case_by_key: dict[tuple[str, str], list[dict[str, object]]] = {
        (method, center): [] for method in METHOD_MENU for center in CENTERS
    }

    expected_case_keys = tuple(
        (center, method, case_id)
        for center in CENTERS
        for method in METHOD_MENU
        for case_id in expected_cases[center]
    )
    observed_case_keys = tuple(
        (
            str(row.get("target_center")),
            str(row.get("method_id")),
            str(row.get("case_id")),
        )
        for row in cases
    )
    if observed_case_keys != expected_case_keys:
        raise ProtocolError("P-DCAPS v2 terminal case rectangle drifted.")

    for row in cases:
        center = str(row["target_center"])
        method = str(row["method_id"])
        sample_count = _nonnegative_int(row, "sample_count", strictly_positive=True)
        delta = _signed_int(row, "threshold_error_delta_vs_P")
        changed = row.get("probability_changed_vs_P")
        harmed = row.get("case_harmed_vs_P")
        if (
            not isinstance(changed, bool)
            or not isinstance(harmed, bool)
            or abs(delta) > sample_count
            or sample_count != expected_case_counts[center][str(row["case_id"])]
            or harmed is not (delta > 0)
            or row.get("raw_labels_persisted") is not False
            or row.get("formal_claim_authorized") is not False
            or (
                method == P_METHOD_ID
                and (changed is not False or delta != 0 or harmed is not False)
            )
        ):
            raise ProtocolError("P-DCAPS v2 terminal case semantics drifted.")
        case_by_key[(method, center)].append(row)

    for row in centers:
        method = str(row["method_id"])
        center = str(row["target_center"])
        sample_count = _nonnegative_int(row, "sample_count", strictly_positive=True)
        case_count = _nonnegative_int(row, "case_count", strictly_positive=True)
        positive = _nonnegative_int(row, "n_positive", strictly_positive=True)
        negative = _nonnegative_int(row, "n_negative", strictly_positive=True)
        true_positive = _nonnegative_int(row, "true_positive")
        true_negative = _nonnegative_int(row, "true_negative")
        false_positive = _nonnegative_int(row, "false_positive")
        false_negative = _nonnegative_int(row, "false_negative")
        changed_count = _nonnegative_int(row, "changed_case_count")
        switch_count = _nonnegative_int(row, "threshold_switch_count")
        helpful_switches = _nonnegative_int(
            row, "helpful_threshold_switch_count"
        )
        harmful_switches = _nonnegative_int(
            row, "harmful_threshold_switch_count"
        )
        center_bacc = _finite_float(row, "center_bacc")
        center_brier = _finite_float(row, "center_brier")
        center_log_loss = _finite_float(row, "center_log_loss")
        squared_error_sum = _finite_float(row, "squared_error_sum")
        log_loss_sum = _finite_float(row, "log_loss_sum")
        grouped_cases = case_by_key[(method, center)]
        expected_bacc = 0.5 * (
            true_positive / positive + true_negative / negative
        )
        if (
            row.get("reference_method") != P_METHOD_ID
            or case_count != len(expected_cases[center])
            or len(grouped_cases) != case_count
            or sum(int(value["sample_count"]) for value in grouped_cases)
            != sample_count
            or sum(
                int(bool(value["probability_changed_vs_P"]))
                for value in grouped_cases
            )
            != changed_count
            or positive + negative != sample_count
            or true_positive + false_negative != positive
            or true_negative + false_positive != negative
            or changed_count > case_count
            or switch_count > sample_count
            or helpful_switches + harmful_switches != switch_count
            or not 0.0 <= center_brier <= 1.0
            or center_log_loss < 0.0
            or not 0.0 <= squared_error_sum <= sample_count
            or log_loss_sum < 0.0
            or not _close(center_bacc, expected_bacc)
            or not _close(center_brier, squared_error_sum / sample_count)
            or not _close(center_log_loss, log_loss_sum / sample_count)
            or row.get("formal_claim_authorized") is not False
            or (
                method == P_METHOD_ID
                and any(
                    not _close(_finite_float(row, key), 0.0)
                    for key in (
                        "center_bacc_delta_vs_P",
                        "center_brier_delta_vs_P",
                        "center_log_loss_delta_vs_P",
                    )
                )
            )
        ):
            raise ProtocolError("P-DCAPS v2 terminal center semantics drifted.")

    for method in METHOD_MENU:
        for center in CENTERS:
            row = center_by_key[(method, center)]
            reference = center_by_key[(P_METHOD_ID, center)]
            method_cases = case_by_key[(method, center)]
            reference_cases = case_by_key[(P_METHOD_ID, center)]
            if (
                any(
                    _nonnegative_int(row, key)
                    != _nonnegative_int(reference, key)
                    for key in (
                        "sample_count",
                        "case_count",
                        "n_positive",
                        "n_negative",
                    )
                )
                or tuple(
                    (value["case_id"], value["sample_count"])
                    for value in method_cases
                )
                != tuple(
                    (value["case_id"], value["sample_count"])
                    for value in reference_cases
                )
                or not _close(
                    _finite_float(row, "center_bacc_delta_vs_P"),
                    _finite_float(row, "center_bacc")
                    - _finite_float(reference, "center_bacc"),
                )
                or not _close(
                    _finite_float(row, "center_brier_delta_vs_P"),
                    _finite_float(row, "center_brier")
                    - _finite_float(reference, "center_brier"),
                )
                or not _close(
                    _finite_float(row, "center_log_loss_delta_vs_P"),
                    _finite_float(row, "center_log_loss")
                    - _finite_float(reference, "center_log_loss"),
                )
            ):
                raise ProtocolError(
                    "P-DCAPS v2 terminal reference semantics drifted."
                )

    for row in methods:
        method = str(row["method_id"])
        method_centers = [center_by_key[(method, center)] for center in CENTERS]
        method_cases = [
            value
            for center in CENTERS
            for value in case_by_key[(method, center)]
        ]
        bacc_deltas = [
            _finite_float(value, "center_bacc_delta_vs_P")
            for value in method_centers
        ]
        brier_deltas = [
            _finite_float(value, "center_brier_delta_vs_P")
            for value in method_centers
        ]
        log_deltas = [
            _finite_float(value, "center_log_loss_delta_vs_P")
            for value in method_centers
        ]
        route_count = sum(
            _nonnegative_int(value, "changed_case_count")
            for value in method_centers
        )
        harm_count = sum(
            int(bool(value["case_harmed_vs_P"])) for value in method_cases
        )
        positive_count = sum(value > 1.0e-12 for value in bacc_deltas)
        negative_count = sum(value < -1.0e-12 for value in bacc_deltas)
        zero_count = len(CENTERS) - positive_count - negative_count
        total_positive = sum(
            _nonnegative_int(value, "n_positive", strictly_positive=True)
            for value in method_centers
        )
        total_negative = sum(
            _nonnegative_int(value, "n_negative", strictly_positive=True)
            for value in method_centers
        )
        pooled_bacc = 0.5 * (
            sum(_nonnegative_int(value, "true_positive") for value in method_centers)
            / total_positive
            + sum(
                _nonnegative_int(value, "true_negative")
                for value in method_centers
            )
            / total_negative
        )
        if (
            row.get("formal_claim_authorized") is not False
            or row.get("descriptive_interval_has_no_nominal_coverage_claim")
            is not True
            or _nonnegative_int(row, "route_count") != route_count
            or _nonnegative_int(row, "case_harm_count") != harm_count
            or not _close(
                _finite_float(row, "case_harm_rate"),
                harm_count / CANONICAL_CASE_COUNT,
            )
            or _nonnegative_int(row, "positive_center_count") != positive_count
            or _nonnegative_int(row, "negative_center_count") != negative_count
            or _nonnegative_int(row, "zero_center_count") != zero_count
            or not _close(
                _finite_float(row, "equal_center_bacc"),
                sum(
                    _finite_float(value, "center_bacc")
                    for value in method_centers
                )
                / len(CENTERS),
            )
            or not _close(_finite_float(row, "sample_pooled_bacc"), pooled_bacc)
            or not _close(
                _finite_float(row, "global_brier"),
                sum(
                    _finite_float(value, "squared_error_sum")
                    for value in method_centers
                )
                / CANONICAL_ROW_COUNT,
            )
            or not _close(
                _finite_float(row, "equal_center_brier"),
                sum(
                    _finite_float(value, "center_brier")
                    for value in method_centers
                )
                / len(CENTERS),
            )
            or not _close(
                _finite_float(row, "global_log_loss"),
                sum(
                    _finite_float(value, "log_loss_sum")
                    for value in method_centers
                )
                / CANONICAL_ROW_COUNT,
            )
            or not _close(
                _finite_float(row, "equal_center_log_loss"),
                sum(
                    _finite_float(value, "center_log_loss")
                    for value in method_centers
                )
                / len(CENTERS),
            )
            or not _close(
                _finite_float(row, "mean_center_bacc_delta_vs_P"),
                sum(bacc_deltas) / len(CENTERS),
            )
            or not _close(
                _finite_float(row, "minimum_center_bacc_delta_vs_P"),
                min(bacc_deltas),
            )
            or not _close(
                _finite_float(row, "maximum_center_bacc_delta_vs_P"),
                max(bacc_deltas),
            )
            or not _close(
                _finite_float(row, "mean_center_brier_delta_vs_P"),
                sum(brier_deltas) / len(CENTERS),
            )
            or not _close(
                _finite_float(row, "mean_center_log_loss_delta_vs_P"),
                sum(log_deltas) / len(CENTERS),
            )
            or _finite_float(row, "descriptive_t8_lower")
            > _finite_float(row, "mean_center_bacc_delta_vs_P")
            or _finite_float(row, "descriptive_t8_upper")
            < _finite_float(row, "mean_center_bacc_delta_vs_P")
            or (
                method == P_METHOD_ID
                and (route_count != 0 or harm_count != 0)
            )
        ):
            raise ProtocolError("P-DCAPS v2 terminal method semantics drifted.")

    reference_sample_count = sum(
        int(center_by_key[(P_METHOD_ID, center)]["sample_count"])
        for center in CENTERS
    )
    if reference_sample_count != CANONICAL_ROW_COUNT or any(
        sum(
            int(row["sample_count"])
            for row in cases
            if row["method_id"] == method
        )
        != CANONICAL_ROW_COUNT
        for method in METHOD_MENU
    ):
        raise ProtocolError("P-DCAPS v2 terminal row coverage drifted.")
    return {
        "method_count": len(methods),
        "center_method_count": len(centers),
        "case_diagnostic_count": len(cases),
        "canonical_case_count": CANONICAL_CASE_COUNT,
        "canonical_row_count": CANONICAL_ROW_COUNT,
    }


def _nonnegative_int(
    row: Mapping[str, object], key: str, *, strictly_positive: bool = False
) -> int:
    value = row.get(key)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < (1 if strictly_positive else 0)
    ):
        raise ProtocolError("P-DCAPS v2 terminal integer semantics drifted.")
    return value


def _signed_int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProtocolError("P-DCAPS v2 terminal integer semantics drifted.")
    return value


def _finite_float(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError("P-DCAPS v2 terminal metric semantics drifted.")
    number = float(value)
    if not math.isfinite(number):
        raise ProtocolError("P-DCAPS v2 terminal metric semantics drifted.")
    return number


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-12)


__all__ = ("validate_terminal_row_inventory",)
