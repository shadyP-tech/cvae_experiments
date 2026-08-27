"""Aggregate-only terminal scoring of already sealed method probabilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path

import numpy as np

from ..artifacts.io import atomic_json, member_path, read_json_object
from ..artifacts.hashing import canonical_hash, require_sha256
from ..protocol import GovernanceError
from .contracts import (
    CenterMetrics,
    TerminalAggregate,
    TerminalComparison,
    TerminalMetrics,
)


def sealed_probability_hash(values: object) -> str:
    """Match the v2 physical layer's dtype/shape/byte probability identity."""

    array = np.ascontiguousarray(values)
    if (
        array.ndim != 1
        or len(array) == 0
        or array.dtype.kind != "f"
        or not np.isfinite(array).all()
        or np.any((array < 0.0) | (array > 1.0))
    ):
        raise GovernanceError("SCALE-BP v2 sealed probability vector drifted.")
    header = f"{array.dtype.str}|{array.shape}".encode("ascii")
    return hashlib.sha256(header + memoryview(array).cast("B")).hexdigest()


def score_sealed_method_probabilities(
    method_probabilities: Mapping[str, object],
    *,
    expected_probability_hashes: Mapping[str, str],
    labels: Sequence[int] | np.ndarray,
    centers: Sequence[str] | np.ndarray,
    protected_method_id: str,
    decision_seal_hash: str,
    epsilon: float = 1.0e-7,
) -> TerminalAggregate:
    """Score in memory and return only aggregate metrics and hash bindings.

    Raw labels are neither returned nor accepted by any persistence function.
    The exact probability vectors must already have been sealed preterminally.
    """

    decision = require_sha256(decision_seal_hash, "terminal decision seal")
    truth = np.ascontiguousarray(labels, dtype=np.int8)
    center_array = np.ascontiguousarray([str(value) for value in centers])
    methods = {
        str(key): np.ascontiguousarray(value)
        for key, value in method_probabilities.items()
    }
    expected = {
        str(key): require_sha256(value, f"prediction {key}")
        for key, value in expected_probability_hashes.items()
    }
    protected = str(protected_method_id)
    if (
        truth.ndim != 1
        or len(truth) == 0
        or not np.isin(truth, (0, 1)).all()
        or center_array.shape != truth.shape
        or not methods
        or set(methods) != set(expected)
        or protected not in methods
        or not 0.0 < float(epsilon) < 0.5
    ):
        raise GovernanceError("SCALE-BP v2 terminal scoring inputs drifted.")
    method_order = (protected, *sorted(set(methods) - {protected}))
    metric_rows: list[TerminalMetrics] = []
    for method_id in method_order:
        probabilities = methods[method_id]
        observed_hash = sealed_probability_hash(probabilities)
        if observed_hash != expected[method_id] or probabilities.shape != truth.shape:
            raise GovernanceError(
                "SCALE-BP v2 terminal probability/preterminal seal binding drifted."
            )
        metric_rows.append(
            _score_method(
                method_id,
                probabilities.astype(np.float64, copy=False),
                truth,
                center_array,
                prediction_hash=observed_hash,
                decision_seal_hash=decision,
                epsilon=float(epsilon),
            )
        )
    protected_metrics = metric_rows[0]
    comparisons = tuple(
        TerminalComparison(
            method_id=row.method_id,
            protected_method_id=protected,
            pooled_delta_bacc=row.pooled_bacc - protected_metrics.pooled_bacc,
            pooled_delta_brier=row.pooled_brier - protected_metrics.pooled_brier,
            pooled_delta_log_loss=(
                row.pooled_log_loss - protected_metrics.pooled_log_loss
            ),
            equal_center_delta_bacc=(
                row.equal_center_bacc - protected_metrics.equal_center_bacc
            ),
            equal_center_delta_brier=(
                row.equal_center_brier - protected_metrics.equal_center_brier
            ),
            equal_center_delta_log_loss=(
                row.equal_center_log_loss - protected_metrics.equal_center_log_loss
            ),
        )
        for row in metric_rows[1:]
    )
    return TerminalAggregate(decision, protected, tuple(metric_rows), comparisons)


def persist_terminal_aggregate(
    root: str | Path, aggregate: TerminalAggregate
) -> dict[str, object]:
    if not isinstance(aggregate, TerminalAggregate):
        raise GovernanceError("SCALE-BP v2 terminal persistence received a foreign DTO.")
    payload = aggregate.to_payload()
    if _contains_label_key(payload):
        raise GovernanceError("SCALE-BP v2 terminal artifact contains raw labels.")
    atomic_json(member_path(root, "reports/terminal_metrics.json"), payload)
    return payload


def validate_persisted_terminal_aggregate(
    root: str | Path,
    *,
    expected_decision_seal_hash: str | None = None,
) -> dict[str, object]:
    payload = read_json_object(member_path(root, "reports/terminal_metrics.json"))
    body = {key: value for key, value in payload.items() if key != "terminal_seal_hash"}
    methods = payload.get("methods")
    comparisons = payload.get("comparisons_to_protected_p")
    if (
        payload.get("schema_version") != "scale_bp_v2_terminal_aggregate_v1"
        or not isinstance(methods, list)
        or not methods
        or payload.get("method_count") != len(methods)
        or not isinstance(comparisons, list)
        or len(comparisons) != len(methods) - 1
        or payload.get("raw_labels_persisted") is not False
        or payload.get("row_level_labels_persisted") is not False
        or payload.get("terminal_scoring_only") is not True
        or payload.get("terminal_seal_hash") != canonical_hash(body)
        or payload.get("terminal_metrics_hash")
        != canonical_hash([row.get("metrics_hash") for row in methods if isinstance(row, Mapping)])
        or _contains_label_key(payload)
        or (
            expected_decision_seal_hash is not None
            and payload.get("decision_seal_hash")
            != require_sha256(expected_decision_seal_hash, "expected decision seal")
        )
    ):
        raise GovernanceError("SCALE-BP v2 persisted terminal aggregate drifted.")
    validated_methods = [_validate_method_payload(row) for row in methods]
    protected = validated_methods[0]
    if protected.get("method_id") != payload.get("protected_method_id"):
        raise GovernanceError("SCALE-BP v2 protected terminal method drifted.")
    for row, method in zip(comparisons, validated_methods[1:], strict=True):
        _validate_comparison_payload(
            row,
            method=method,
            protected=protected,
            protected_method_id=str(payload["protected_method_id"]),
        )
    require_sha256(payload.get("decision_seal_hash"), "terminal decision seal")
    require_sha256(payload.get("terminal_seal_hash"), "terminal seal")
    require_sha256(payload.get("terminal_metrics_hash"), "terminal metrics hash")
    return payload


def _validate_method_payload(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise GovernanceError("SCALE-BP v2 terminal method row is malformed.")
    row = dict(value)
    centers = row.get("centers")
    pooled = row.get("pooled")
    equal_center = row.get("equal_center")
    body = {key: item for key, item in row.items() if key != "metrics_hash"}
    if (
        row.get("schema_version") != "scale_bp_v2_terminal_metrics_v1"
        or not isinstance(centers, list)
        or not centers
        or not isinstance(pooled, Mapping)
        or not isinstance(equal_center, Mapping)
        or row.get("raw_labels_persisted") is not False
        or row.get("metrics_hash") != canonical_hash(body)
        or row.get("row_count")
        != sum(
            int(center.get("row_count", -1))
            for center in centers
            if isinstance(center, Mapping)
        )
    ):
        raise GovernanceError("SCALE-BP v2 terminal method metric hash drifted.")
    center_rows: list[dict[str, object]] = []
    for value in centers:
        if not isinstance(value, Mapping):
            raise GovernanceError("SCALE-BP v2 terminal center row is malformed.")
        center = dict(value)
        center_body = {
            key: item for key, item in center.items() if key != "metrics_hash"
        }
        if (
            center.get("schema_version")
            != "scale_bp_v2_center_terminal_metrics_v1"
            or center.get("metrics_hash") != canonical_hash(center_body)
            or type(center.get("row_count")) is not int
            or int(center["row_count"]) <= 0
        ):
            raise GovernanceError("SCALE-BP v2 terminal center metric hash drifted.")
        center_rows.append(center)
    for metric in ("bacc", "brier", "log_loss"):
        values = np.asarray([center[metric] for center in center_rows], dtype=np.float64)
        observed = float(equal_center.get(metric, np.nan))
        if not np.isfinite(values).all() or observed != float(
            np.mean(values, dtype=np.float64)
        ):
            raise GovernanceError("SCALE-BP v2 equal-center metric drifted.")
    require_sha256(row.get("prediction_hash"), "terminal prediction hash")
    require_sha256(row.get("decision_seal_hash"), "terminal decision seal")
    return row


def _validate_comparison_payload(
    value: object,
    *,
    method: Mapping[str, object],
    protected: Mapping[str, object],
    protected_method_id: str,
) -> None:
    if not isinstance(value, Mapping):
        raise GovernanceError("SCALE-BP v2 terminal comparison row is malformed.")
    row = dict(value)
    body = {key: item for key, item in row.items() if key != "comparison_hash"}
    pooled = row.get("pooled_delta")
    equal_center = row.get("equal_center_delta")
    method_pooled = method.get("pooled")
    protected_pooled = protected.get("pooled")
    method_equal = method.get("equal_center")
    protected_equal = protected.get("equal_center")
    if (
        row.get("schema_version") != "scale_bp_v2_terminal_comparison_v1"
        or row.get("method_id") != method.get("method_id")
        or row.get("protected_method_id") != protected_method_id
        or row.get("comparison_hash") != canonical_hash(body)
        or not all(
            isinstance(item, Mapping)
            for item in (
                pooled,
                equal_center,
                method_pooled,
                protected_pooled,
                method_equal,
                protected_equal,
            )
        )
    ):
        raise GovernanceError("SCALE-BP v2 terminal comparison hash drifted.")
    for metric in ("bacc", "brier", "log_loss"):
        pooled_expected = float(method_pooled[metric]) - float(  # type: ignore[index]
            protected_pooled[metric]  # type: ignore[index]
        )
        equal_expected = float(method_equal[metric]) - float(  # type: ignore[index]
            protected_equal[metric]  # type: ignore[index]
        )
        if (
            float(pooled[metric]) != pooled_expected  # type: ignore[index]
            or float(equal_center[metric]) != equal_expected  # type: ignore[index]
        ):
            raise GovernanceError("SCALE-BP v2 terminal comparison value drifted.")


def _score_method(
    method_id: str,
    probabilities: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    *,
    prediction_hash: str,
    decision_seal_hash: str,
    epsilon: float,
) -> TerminalMetrics:
    center_ids = tuple(sorted(set(str(value) for value in centers)))
    rows = tuple(
        _metric_row(
            str(center),
            probabilities[centers == center],
            labels[centers == center],
            epsilon=epsilon,
        )
        for center in center_ids
    )
    pooled = _metric_row("POOLED", probabilities, labels, epsilon=epsilon)
    return TerminalMetrics(
        method_id=method_id,
        row_count=len(labels),
        prediction_hash=prediction_hash,
        decision_seal_hash=decision_seal_hash,
        pooled_bacc=pooled.bacc,
        pooled_brier=pooled.brier,
        pooled_log_loss=pooled.log_loss,
        equal_center_bacc=float(np.mean([row.bacc for row in rows], dtype=np.float64)),
        equal_center_brier=float(np.mean([row.brier for row in rows], dtype=np.float64)),
        equal_center_log_loss=float(
            np.mean([row.log_loss for row in rows], dtype=np.float64)
        ),
        center_metrics=rows,
    )


def _metric_row(
    center: str,
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    epsilon: float,
) -> CenterMetrics:
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        raise GovernanceError("SCALE-BP v2 BACC requires both classes per center.")
    hard = probabilities >= 0.5
    sensitivity = float(np.mean(hard[positive], dtype=np.float64))
    specificity = float(np.mean(~hard[negative], dtype=np.float64))
    bacc = 0.5 * (sensitivity + specificity)
    brier = float(np.mean((probabilities - labels) ** 2, dtype=np.float64))
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    log_loss = float(
        np.mean(
            -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)),
            dtype=np.float64,
        )
    )
    return CenterMetrics(center, len(labels), bacc, brier, log_loss)


def _contains_label_key(value: object) -> bool:
    forbidden = {
        "labels",
        "raw_labels",
        "sample_labels",
        "target_labels",
        "label_values",
        "truth",
        "ground_truth",
        "y_true",
    }
    if isinstance(value, Mapping):
        return bool({str(key).casefold() for key in value} & forbidden) or any(
            _contains_label_key(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_label_key(item) for item in value)
    return False


__all__ = (
    "persist_terminal_aggregate",
    "score_sealed_method_probabilities",
    "sealed_probability_hash",
    "validate_persisted_terminal_aggregate",
)
