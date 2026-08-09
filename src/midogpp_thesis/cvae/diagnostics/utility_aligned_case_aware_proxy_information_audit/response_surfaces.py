"""Post-seal exact and smooth response construction.

The exact response thresholds the exact-nine mean probability and remains the
only primary response.  Soft BACC is computed from the same sealed means after
label access and is diagnostic-only.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import (
    array_bundle_sha256,
    array_sha256,
    canonical_sha256,
)
from .contracts import (
    EXACT_SEED_PAIR_COUNT,
    EXPECTED_FEATURE_ROW_COUNT,
    RESPONSE_ROW_SCHEMA,
    CaseAwareFeatureSurface,
    CaseAwareProxyFeatureRow,
    CaseAwareResponseRow,
    CaseAwareResponseSurface,
    expected_row_keys,
)


_HASH = re.compile(r"(?:[0-9a-f]{16}|[0-9a-f]{64})")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RESPONSE_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "outer_target_id",
        "query_id",
        "candidate_source",
        "support_partition_hash",
        "feature_row_hash",
        "feature_surface_seal_hash",
        "evaluation_partition_hash",
        "evaluation_case_hashes",
        "evaluation_row_hash",
        "evaluation_label_sha256",
        "response_prediction_hash",
        "exact_base_bacc",
        "exact_tail_bacc",
        "exact_bacc_delta",
        "smooth_base_bacc",
        "smooth_tail_bacc",
        "smooth_bacc_delta",
        "support_eval_disjoint",
        "features_sealed_before_label_access",
        "exact_response_is_primary",
        "smooth_response_is_diagnostic_only",
        "policy_update_authorized",
        "response_unit",
        "technical_seed_rows_are_independent_observations",
        "response_row_hash",
    }
)


@dataclass(frozen=True, eq=False)
class ExactNineEvaluationVectors:
    """Evaluation predictions opened for scoring only after feature sealing."""

    evaluation_partition_hash: str
    evaluation_case_hashes: tuple[str, ...]
    evaluation_row_hash: str
    prediction_provenance_hash: str
    base_probabilities: np.ndarray
    tail_probabilities: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        partition = _hash_token(
            self.evaluation_partition_hash, "evaluation_partition_hash"
        )
        row_hash = _hash_token(self.evaluation_row_hash, "evaluation_row_hash")
        provenance = _hash_token(
            self.prediction_provenance_hash, "prediction_provenance_hash"
        )
        case_hashes = tuple(
            _hash_token(value, "evaluation_case_hashes")
            for value in self.evaluation_case_hashes
        )
        if not case_hashes or len(set(case_hashes)) != len(case_hashes):
            raise ProtocolError("Evaluation requires distinct remaining whole cases.")
        base = _probability_matrix(self.base_probabilities, "base_probabilities")
        tail = _probability_matrix(self.tail_probabilities, "tail_probabilities")
        if base.shape != tail.shape:
            raise ProtocolError("Evaluation base/tail exact-nine geometry drifted.")
        labels = _binary_labels(self.labels, expected_length=base.shape[1])
        object.__setattr__(self, "evaluation_partition_hash", partition)
        object.__setattr__(self, "evaluation_case_hashes", case_hashes)
        object.__setattr__(self, "evaluation_row_hash", row_hash)
        object.__setattr__(self, "prediction_provenance_hash", provenance)
        object.__setattr__(self, "base_probabilities", base)
        object.__setattr__(self, "tail_probabilities", tail)
        object.__setattr__(self, "labels", labels)

    @property
    def exact_nine_prediction_hash(self) -> str:
        return array_bundle_sha256(self.base_probabilities, self.tail_probabilities)

    @property
    def evaluation_label_sha256(self) -> str:
        return array_sha256(self.labels)


def mean_exact_nine_probabilities(
    vectors: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Average nine seed-pair probability vectors before downstream logic."""

    matrix = _probability_matrix(vectors, "exact_nine_probability_vectors")
    result = np.mean(matrix, axis=0, dtype=np.float64)
    result.setflags(write=False)
    return result


# Verbose alias documents what is averaged for adapters and tests.
mean_exact_nine_positive_class_probabilities = mean_exact_nine_probabilities


def balanced_accuracy(
    labels: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
) -> float:
    """Balanced accuracy for binary predictions with both classes required."""

    truth = _binary_labels(labels)
    predicted = _binary_labels(predictions, expected_length=len(truth))
    positive = truth == 1
    negative = truth == 0
    return 0.5 * (
        float(np.mean(predicted[positive] == 1, dtype=np.float64))
        + float(np.mean(predicted[negative] == 0, dtype=np.float64))
    )


def soft_balanced_accuracy(
    labels: Sequence[int] | np.ndarray,
    positive_class_probabilities: Sequence[float] | np.ndarray,
) -> float:
    """Compute ``.5 * (mean_y1(p) + mean_y0(1-p))`` exactly."""

    truth = _binary_labels(labels)
    probability = np.asarray(positive_class_probabilities, dtype=np.float64)
    if (
        probability.shape != truth.shape
        or not np.isfinite(probability).all()
        or np.any(probability < 0.0)
        or np.any(probability > 1.0)
    ):
        raise ProtocolError("Soft BACC requires one finite [0,1] probability per row.")
    return 0.5 * (
        float(np.mean(probability[truth == 1], dtype=np.float64))
        + float(np.mean(1.0 - probability[truth == 0], dtype=np.float64))
    )


soft_bacc = soft_balanced_accuracy


def exact_nine_response_values(
    base_vectors: Sequence[Sequence[float]] | np.ndarray,
    tail_vectors: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Return exact-base/tail/delta then smooth-base/tail/delta."""

    base = mean_exact_nine_probabilities(base_vectors)
    tail = mean_exact_nine_probabilities(tail_vectors)
    if base.shape != tail.shape:
        raise ProtocolError("Response base/tail row geometry drifted.")
    truth = _binary_labels(labels, expected_length=len(base))
    exact_base = balanced_accuracy(truth, (base >= 0.5).astype(np.int64))
    exact_tail = balanced_accuracy(truth, (tail >= 0.5).astype(np.int64))
    smooth_base = soft_balanced_accuracy(truth, base)
    smooth_tail = soft_balanced_accuracy(truth, tail)
    return (
        exact_base,
        exact_tail,
        exact_tail - exact_base,
        smooth_base,
        smooth_tail,
        smooth_tail - smooth_base,
    )


def build_response_row(
    *,
    feature_row: CaseAwareProxyFeatureRow,
    feature_surface_seal_hash: str,
    evaluation: ExactNineEvaluationVectors,
) -> CaseAwareResponseRow:
    """Score one already-sealed feature row on disjoint remaining cases."""

    if not isinstance(feature_row, CaseAwareProxyFeatureRow):
        raise ProtocolError("Response construction requires a typed feature row.")
    if not isinstance(evaluation, ExactNineEvaluationVectors):
        raise ProtocolError("Response construction requires typed evaluation vectors.")
    seal = _sha256(feature_surface_seal_hash, "feature_surface_seal_hash")
    if set(feature_row.support_case_hashes).intersection(
        evaluation.evaluation_case_hashes
    ):
        raise ProtocolError("Support and remaining evaluation cases overlap.")
    values = exact_nine_response_values(
        evaluation.base_probabilities,
        evaluation.tail_probabilities,
        evaluation.labels,
    )
    # Bind the supplied execution provenance and the exact probability content.
    response_prediction_hash = canonical_sha256(
        {
            "prediction_provenance_hash": evaluation.prediction_provenance_hash,
            "exact_nine_prediction_hash": evaluation.exact_nine_prediction_hash,
            "feature_surface_seal_hash": seal,
            "row_key": list(feature_row.row_key),
        }
    )
    return CaseAwareResponseRow(
        outer_target_id=feature_row.outer_target_id,
        query_id=feature_row.query_id,
        candidate_source=feature_row.candidate_source,
        support_partition_hash=feature_row.support_partition_hash,
        feature_row_hash=feature_row.feature_row_hash,
        feature_surface_seal_hash=seal,
        evaluation_partition_hash=evaluation.evaluation_partition_hash,
        evaluation_case_hashes=evaluation.evaluation_case_hashes,
        evaluation_row_hash=evaluation.evaluation_row_hash,
        evaluation_label_sha256=evaluation.evaluation_label_sha256,
        response_prediction_hash=response_prediction_hash,
        exact_base_bacc=values[0],
        exact_tail_bacc=values[1],
        exact_bacc_delta=values[2],
        smooth_base_bacc=values[3],
        smooth_tail_bacc=values[4],
        smooth_bacc_delta=values[5],
    )


build_case_aware_response_row = build_response_row


def response_row_from_payload(payload: Mapping[str, object]) -> CaseAwareResponseRow:
    if not isinstance(payload, Mapping) or set(payload) != _RESPONSE_PAYLOAD_KEYS:
        raise ProtocolError("Case-aware response payload does not match the exact schema.")
    if (
        payload.get("schema_version") != RESPONSE_ROW_SCHEMA
        or payload.get("response_unit")
        != "candidate_H_q_e_after_exact_nine_probability_mean"
        or payload.get("technical_seed_rows_are_independent_observations") is not False
    ):
        raise ProtocolError("Case-aware response payload semantics drifted.")
    supplied_hash = payload.get("response_row_hash")
    unhashed = {
        key: payload[key] for key in payload if key != "response_row_hash"
    }
    if supplied_hash != canonical_sha256(unhashed):
        raise ProtocolError("Case-aware response payload hash drifted.")
    ignored = {
        "schema_version",
        "response_unit",
        "technical_seed_rows_are_independent_observations",
        "response_row_hash",
    }
    row = CaseAwareResponseRow(
        **{key: payload[key] for key in payload if key not in ignored}  # type: ignore[arg-type]
    )
    if row.response_row_hash != supplied_hash:
        raise ProtocolError("Case-aware response reconstruction hash drifted.")
    return row


def build_response_surface(
    feature_surface: CaseAwareFeatureSurface,
    rows: Sequence[CaseAwareResponseRow | Mapping[str, object]],
) -> CaseAwareResponseSurface:
    """Bind all 504 post-seal responses to the label-free feature surface."""

    if not isinstance(feature_surface, CaseAwareFeatureSurface):
        raise ProtocolError("Response surface requires a typed feature surface.")
    typed = tuple(
        value if isinstance(value, CaseAwareResponseRow) else response_row_from_payload(value)
        for value in rows
    )
    if len(typed) != EXPECTED_FEATURE_ROW_COUNT:
        raise ProtocolError("Response surface requires complete H/q/e coverage.")
    keyed = {row.row_key: row for row in typed}
    expected = expected_row_keys()
    if len(keyed) != len(typed) or set(keyed) != set(expected):
        raise ProtocolError("Response H/q/e geometry drifted.")
    ordered = tuple(keyed[key] for key in expected)
    features = {row.row_key: row for row in feature_surface.rows}
    feature_seals = {row.feature_surface_seal_hash for row in ordered}
    if len(feature_seals) != 1:
        raise ProtocolError(
            "Response rows must share one persisted pre-label feature seal."
        )
    feature_seal_hash = next(iter(feature_seals))
    for response in ordered:
        feature = features[response.row_key]
        if (
            response.feature_row_hash != feature.feature_row_hash
            or response.support_partition_hash != feature.support_partition_hash
            or set(feature.support_case_hashes).intersection(
                response.evaluation_case_hashes
            )
        ):
            raise ProtocolError("Response crossed its sealed support/evaluation boundary.")
    for query in {key[1] for key in expected}:
        label_identities = {
            (
                row.evaluation_partition_hash,
                row.evaluation_case_hashes,
                row.evaluation_row_hash,
                row.evaluation_label_sha256,
            )
            for row in ordered
            if row.query_id == query
        }
        if len(label_identities) != 1:
            raise ProtocolError(
                "Evaluation case/row/label identity drifted across candidate responses."
            )
    unhashed = {
        "schema_version": "midogpp_stage90_case_aware_response_surface_v1",
        "feature_surface_hash": feature_surface.surface_hash,
        "feature_surface_seal_hash": feature_seal_hash,
        "ordered_response_row_hashes": [row.response_row_hash for row in ordered],
        "ordered_exact_bacc_delta": [row.exact_bacc_delta for row in ordered],
        "ordered_smooth_bacc_delta": [row.smooth_bacc_delta for row in ordered],
        "row_count": len(ordered),
        "exact_bacc_delta_is_primary": True,
        "smooth_bacc_delta_is_diagnostic_only": True,
        "features_sealed_before_label_access": True,
        "support_eval_disjoint": True,
    }
    return CaseAwareResponseSurface(
        rows=ordered,
        row_keys=expected,
        feature_surface_hash=feature_surface.surface_hash,
        feature_surface_seal_hash=feature_seal_hash,
        surface_hash=canonical_sha256(unhashed),
    )


build_case_aware_response_surface = build_response_surface


def _probability_matrix(value: object, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64).copy()
    if (
        matrix.ndim != 2
        or matrix.shape[0] != EXACT_SEED_PAIR_COUNT
        or matrix.shape[1] < 1
        or not np.isfinite(matrix).all()
        or np.any(matrix < 0.0)
        or np.any(matrix > 1.0)
    ):
        raise ProtocolError(f"{name} must have finite [0,1] shape (9, n_rows).")
    matrix.setflags(write=False)
    return matrix


def _binary_labels(value: object, expected_length: int | None = None) -> np.ndarray:
    labels = np.asarray(value)
    if labels.ndim != 1 or not len(labels):
        raise ProtocolError("Binary labels must be a nonempty one-dimensional vector.")
    if expected_length is not None and len(labels) != expected_length:
        raise ProtocolError("Binary label row geometry drifted.")
    if not np.all(np.isin(labels, np.asarray([0, 1]))):
        raise ProtocolError("Balanced accuracy requires binary labels in {0,1}.")
    result = np.asarray(labels, dtype=np.uint8)
    if set(result.tolist()) != {0, 1}:
        raise ProtocolError("Balanced accuracy requires both label classes.")
    result.setflags(write=False)
    return result


def _hash_token(value: object, name: str) -> str:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise ProtocolError(f"{name} must be a lowercase 16- or 64-hex hash.")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ProtocolError(f"{name} must be a canonical lowercase SHA-256 hash.")
    return value


__all__ = (
    "ExactNineEvaluationVectors",
    "balanced_accuracy",
    "build_case_aware_response_row",
    "build_case_aware_response_surface",
    "build_response_row",
    "build_response_surface",
    "exact_nine_response_values",
    "mean_exact_nine_positive_class_probabilities",
    "mean_exact_nine_probabilities",
    "response_row_from_payload",
    "soft_bacc",
    "soft_balanced_accuracy",
)
