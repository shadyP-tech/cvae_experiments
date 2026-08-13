"""Terminal truth admission, confusion reduction, metrics, and oracles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    B_ACTION_ID,
    CENTERS,
    DIRECTION_IDS,
    METHOD_IDS,
    PRE_TERMINAL_METHOD_IDS,
    a1_action_id,
    candidate_sources,
)
from .ensemble import DESCRIPTIVE_METHOD_IDS
from .probability_surfaces import ProbabilityIndex, hard_prediction
from .products import (
    BinaryLabel,
    CaseMethodConfusion,
    EqualCenterContrast,
    MethodPrediction,
    PooledBacc,
)
from .scoring import pooled_bacc, score_case_action_confusions


TERMINAL_REPORTED_METHOD_IDS = (*METHOD_IDS, *DESCRIPTIVE_METHOD_IDS)


def method_role(method_id: str) -> str:
    if method_id == "DCSE_LOO":
        return "sealed_preterminal_primary"
    if method_id in PRE_TERMINAL_METHOD_IDS:
        return "sealed_preterminal_control"
    if method_id in ("O_directional_static", "O_case_directional"):
        return "terminal_label_oracle_descriptive"
    if method_id in DESCRIPTIVE_METHOD_IDS:
        return "sealed_preterminal_descriptive_control"
    raise ProtocolError("DCSE terminal method role is unknown.")


def label_value(row: object) -> int:
    if hasattr(row, "value"):
        value = getattr(row, "value")
    elif hasattr(row, "label"):
        value = getattr(row, "label")
    else:
        raise ProtocolError("DCSE terminal label row has no binary value.")
    if isinstance(value, bool) or int(value) not in (0, 1):
        raise ProtocolError("DCSE terminal label is not a binary integer.")
    return int(value)


def terminal_truth(
    labels: Sequence[BinaryLabel] | Sequence[object],
) -> dict[tuple[str, str, str], int]:
    result: dict[tuple[str, str, str], int] = {}
    for row in labels:
        key = (str(row.target_center), str(row.case_id), str(row.sample_id))
        if key in result or key[0] not in CENTERS or not key[1] or not key[2]:
            raise ProtocolError(
                "DCSE terminal label identities are malformed or duplicated."
            )
        result[key] = label_value(row)
    if not result:
        raise ProtocolError("DCSE terminal labels are empty.")
    return result


def score_terminal_predictions(
    predictions: Sequence[MethodPrediction] | Sequence[object],
    terminal_labels: Sequence[BinaryLabel] | Sequence[object],
    *,
    require_methods: Sequence[str] | None = None,
) -> tuple[CaseMethodConfusion, ...]:
    """Reduce sealed sample predictions to int64 whole-case confusions."""

    truth = terminal_truth(terminal_labels)
    rows = tuple(predictions)
    if not rows:
        raise ProtocolError("DCSE terminal prediction surface is empty.")
    by_method: dict[str, dict[tuple[str, str, str], object]] = {}
    method_order: list[str] = []
    for row in rows:
        method = str(row.method_id)
        if method not in by_method:
            by_method[method] = {}
            method_order.append(method)
        key = (str(row.target_center), str(row.case_id), str(row.sample_id))
        if key in by_method[method]:
            raise ProtocolError(
                "DCSE terminal predictions duplicate a method/sample cell."
            )
        by_method[method][key] = row
    if require_methods is not None and tuple(method_order) != tuple(require_methods):
        raise ProtocolError("DCSE terminal method order/coverage drifted.")
    if any(set(cells) != set(truth) for cells in by_method.values()):
        raise ProtocolError(
            "Every DCSE terminal method must cover the exact label universe."
        )

    cases: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for key in sorted(
        truth, key=lambda value: (CENTERS.index(value[0]), value[1], value[2])
    ):
        cases.setdefault(key[:2], []).append(key)
    output: list[CaseMethodConfusion] = []
    for method in method_order:
        for (target, case_id), keys in cases.items():
            labels = np.asarray([truth[key] for key in keys], dtype=np.int8)
            predicted = np.asarray(
                [
                    int(getattr(by_method[method][key], "hard_prediction"))
                    for key in keys
                ],
                dtype=np.int8,
            )
            positive = labels == 1
            negative = ~positive
            output.append(
                CaseMethodConfusion(
                    target,
                    case_id,
                    method,
                    int(np.sum(positive & (predicted == 1), dtype=np.int64)),
                    int(np.sum(negative & (predicted == 0), dtype=np.int64)),
                    int(np.sum(negative & (predicted == 1), dtype=np.int64)),
                    int(np.sum(positive & (predicted == 0), dtype=np.int64)),
                )
            )
    return tuple(output)


def center_metrics(
    confusions: Sequence[CaseMethodConfusion], method_order: Sequence[str]
) -> tuple[PooledBacc, ...]:
    rows = tuple(confusions)
    output: list[PooledBacc] = []
    for method in method_order:
        for center in CENTERS:
            selected = tuple(
                row
                for row in rows
                if row.method_id == method and row.target_center == center
            )
            if not selected:
                raise ProtocolError(
                    "DCSE terminal center/method confusion surface is incomplete."
                )
            output.append(
                pooled_bacc(
                    selected, scope_id=f"center={center}", method_id=method
                )
            )
    return tuple(output)


def center_metric_index(
    metrics: Sequence[PooledBacc],
) -> dict[tuple[str, str], PooledBacc]:
    result = {
        (row.method_id, row.scope_id.removeprefix("center=")): row
        for row in metrics
    }
    if len(result) != len(metrics):
        raise ProtocolError("DCSE terminal center metrics are duplicated.")
    return result


def method_metric_rows(
    metrics: Sequence[PooledBacc], method_order: Sequence[str]
) -> tuple[dict[str, object], ...]:
    index = center_metric_index(metrics)
    output = []
    for method in method_order:
        values = tuple(index[(method, center)].exact for center in CENTERS)
        exact = sum(values, Fraction(0)) / len(values)
        output.append(
            {
                "schema_version": "fixed_bank_dcse_equal_center_method_metric_v1",
                "method_id": method,
                "center_bacc_fractions": [
                    [center, value.numerator, value.denominator]
                    for center, value in zip(CENTERS, values, strict=True)
                ],
                "exact_fraction": [exact.numerator, exact.denominator],
                "equal_center_bacc": float(exact),
                "center_count": len(CENTERS),
                "center_weighting": "equal",
                "descriptive_only": True,
                "method_role": method_role(method),
                "success_gate_eligible": method
                in {"B", "U", "DCSE_LOO", "G_directional_matched"},
            }
        )
    return tuple(output)


def equal_center_contrast(
    metrics: Sequence[PooledBacc], *, method_id: str, reference_id: str
) -> EqualCenterContrast:
    index = center_metric_index(metrics)
    differences = tuple(
        (
            center,
            (
                index[(method_id, center)].exact
                - index[(reference_id, center)].exact
            ).numerator,
            (
                index[(method_id, center)].exact
                - index[(reference_id, center)].exact
            ).denominator,
        )
        for center in CENTERS
    )
    return EqualCenterContrast(
        f"{method_id}-{reference_id}", method_id, reference_id, differences
    )


def _choose_oracle(values: Mapping[str | None, Fraction]) -> str | None:
    maximum = max(values.values())
    tolerance = Fraction(1, 10**12)
    return min(
        (
            source
            for source, value in values.items()
            if maximum - value <= tolerance
        ),
        key=lambda source: -1 if source is None else int(source),
    )


def terminal_oracle_predictions(
    probability_surface: object,
    terminal_labels: Sequence[object],
) -> tuple[MethodPrediction, ...]:
    """Build the two declared terminal-only directional oracle controls."""

    index = ProbabilityIndex(
        tuple(getattr(probability_surface, "rows", probability_surface))
    )
    counts = score_case_action_confusions(probability_surface, terminal_labels)
    truth = terminal_truth(terminal_labels)
    static: dict[tuple[str, str], str | None] = {}
    casewise: dict[tuple[str, str, str], str | None] = {}
    for target in CENTERS:
        target_rows = tuple(row for row in counts if row.target_center == target)
        case_ids = tuple(sorted({row.case_id for row in target_rows}))
        n_positive = sum(
            row.n_positive
            for row in target_rows
            if row.action_id == B_ACTION_ID
        )
        n_negative = sum(
            row.n_negative
            for row in target_rows
            if row.action_id == B_ACTION_ID
        )
        if n_positive <= 0 or n_negative <= 0:
            raise ProtocolError(
                "DCSE terminal oracle requires both classes per center."
            )
        for direction in DIRECTION_IDS:
            static_values: dict[str | None, Fraction] = {None: Fraction(0)}
            for source in candidate_sources(target):
                selected = tuple(
                    row
                    for row in target_rows
                    if row.action_id == a1_action_id(source)
                )
                if direction == "zero_to_one":
                    favorable = sum(row.flip_0to1_positive for row in selected)
                    adverse = sum(row.flip_0to1_negative for row in selected)
                    value = Fraction(favorable, 2 * n_positive) - Fraction(
                        adverse, 2 * n_negative
                    )
                else:
                    favorable = sum(row.flip_1to0_negative for row in selected)
                    adverse = sum(row.flip_1to0_positive for row in selected)
                    value = Fraction(favorable, 2 * n_negative) - Fraction(
                        adverse, 2 * n_positive
                    )
                static_values[source] = value
            static[(target, direction)] = _choose_oracle(static_values)
            for case_id in case_ids:
                values: dict[str | None, Fraction] = {None: Fraction(0)}
                for source in candidate_sources(target):
                    row = next(
                        value
                        for value in target_rows
                        if value.case_id == case_id
                        and value.action_id == a1_action_id(source)
                    )
                    if direction == "zero_to_one":
                        value = Fraction(
                            row.flip_0to1_positive, 2 * n_positive
                        ) - Fraction(row.flip_0to1_negative, 2 * n_negative)
                    else:
                        value = Fraction(
                            row.flip_1to0_negative, 2 * n_negative
                        ) - Fraction(row.flip_1to0_positive, 2 * n_positive)
                    values[source] = value
                casewise[(target, case_id, direction)] = _choose_oracle(values)

    output: list[MethodPrediction] = []
    for method_id in ("O_directional_static", "O_case_directional"):
        for key in sorted(
            truth,
            key=lambda value: (CENTERS.index(value[0]), value[1], value[2]),
        ):
            target, case_id, sample_id = key
            baseline = float(
                index[(target, case_id, sample_id, B_ACTION_ID)].probability_mean
            )
            branch = hard_prediction(baseline)
            direction = DIRECTION_IDS[branch]
            source = (
                static[(target, direction)]
                if method_id == "O_directional_static"
                else casewise[(target, case_id, direction)]
            )
            probability = (
                baseline
                if source is None
                else float(
                    index[
                        (target, case_id, sample_id, a1_action_id(source))
                    ].probability_mean
                )
            )
            output.append(
                MethodPrediction(
                    target,
                    case_id,
                    sample_id,
                    method_id,
                    probability,
                    branch,
                    (source,),
                )
            )
    return tuple(output)


__all__ = (
    "TERMINAL_REPORTED_METHOD_IDS",
    "center_metrics",
    "equal_center_contrast",
    "method_metric_rows",
    "method_role",
    "score_terminal_predictions",
    "terminal_oracle_predictions",
    "terminal_truth",
)
