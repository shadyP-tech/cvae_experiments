"""Versioned process-stable payload contracts for fitted router artifacts.

Raw fitted numbers remain persisted and self-fingerprinted.  These helpers
define only the semantic hashes that deliberately exclude those named numbers
so independent fresh-process refits can be compared by the validator's narrow
field allow-list.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .hashing import canonical_hash


FITTED_NUMERIC_VALIDATION = "replay_isclose_atol_5e-12_rtol_5e-12"
DECISION_NUMERIC_FIELDS = (
    "predicted_gain",
    "action_margin",
    "epistemic_standard_error",
    "calibration_standard_error",
    "margin_standard_error",
    "margin_lcb",
)
SCORE_NUMERIC_FIELDS = (
    "expected_gain",
    "epistemic_standard_error",
    "calibration_standard_error",
)


def decision_semantic_payload(value: object) -> dict[str, object]:
    raw = value.to_payload() if hasattr(value, "to_payload") else value
    if not isinstance(raw, Mapping):
        raise TypeError("Decision semantic payload requires a mapping product.")
    payload = dict(raw)
    for field in DECISION_NUMERIC_FIELDS:
        payload.pop(field)
    return {
        **payload,
        "fitted_numeric_fields": list(DECISION_NUMERIC_FIELDS),
        "fitted_numeric_validation": FITTED_NUMERIC_VALIDATION,
    }


def score_semantic_payload(row: Mapping[str, object]) -> Mapping[str, object]:
    payload = dict(row)
    payload.pop("row_hash", None)
    for field in SCORE_NUMERIC_FIELDS:
        payload.pop(field)
    return {
        **payload,
        "fitted_numeric_fields": list(SCORE_NUMERIC_FIELDS),
        "fitted_numeric_validation": FITTED_NUMERIC_VALIDATION,
    }


def directional_calibration_semantic_payload(
    row: Mapping[str, object],
) -> Mapping[str, object]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"offset", "offset_variance", "calibration_fingerprint"}
    } | {
        "fitted_numeric_fields": ["offset", "offset_variance"],
        "fitted_numeric_validation": FITTED_NUMERIC_VALIDATION,
    }


def calibration_semantic_hash(payload: Mapping[str, object]) -> str:
    semantic = dict(payload)
    families = semantic.pop("family_calibrations")
    if not isinstance(families, Mapping):
        raise TypeError("Calibration family payload must be a mapping.")
    semantic["family_calibration_contracts"] = {
        str(family): {
            str(direction): directional_calibration_semantic_payload(row)
            for direction, row in sorted(by_direction.items())
        }
        for family, by_direction in sorted(families.items())
    }
    single = semantic.pop("single_challenger_calibration")
    if not isinstance(single, Mapping):
        raise TypeError("Single-challenger calibration must be a mapping.")
    semantic["single_challenger_calibration_contract"] = {
        key: value
        for key, value in single.items()
        if key not in {"gamma_0to1", "gamma_1to0", "calibration_hash"}
    }
    semantic["single_challenger_calibration_contract"][
        "fitted_numeric_fields"
    ] = ["gamma_0to1", "gamma_1to0"]
    semantic["single_challenger_calibration_contract"][
        "fitted_numeric_validation"
    ] = FITTED_NUMERIC_VALIDATION
    return canonical_hash(semantic)


def calibration_observation_semantic_payload(value: object) -> Mapping[str, object]:
    return {
        "case_id": str(getattr(value, "case_id")),
        "action_id": str(getattr(value, "action_id")),
        "direction": str(getattr(value, "direction")),
        "success_count": int(getattr(value, "success_count")),
        "trial_count": int(getattr(value, "trial_count")),
        "fitted_numeric_fields": ["base_probability"],
        "fitted_numeric_validation": FITTED_NUMERIC_VALIDATION,
    }


def router_metric_semantic_payload(
    row: Mapping[str, object],
) -> Mapping[str, object]:
    payload = dict(row)
    payload.pop("row_hash", None)
    payload.pop("spearman")
    payload["spearman_validation"] = FITTED_NUMERIC_VALIDATION
    return payload


def terminal_table_semantic_hash(
    name: str, rows: Sequence[Mapping[str, object]]
) -> str:
    if name == "router_identification_metrics":
        return canonical_hash(
            [router_metric_semantic_payload(row) for row in rows]
        )
    return canonical_hash(list(rows))


__all__ = (
    "DECISION_NUMERIC_FIELDS",
    "FITTED_NUMERIC_VALIDATION",
    "SCORE_NUMERIC_FIELDS",
    "calibration_observation_semantic_payload",
    "calibration_semantic_hash",
    "decision_semantic_payload",
    "directional_calibration_semantic_payload",
    "router_metric_semantic_payload",
    "score_semantic_payload",
    "terminal_table_semantic_hash",
)
