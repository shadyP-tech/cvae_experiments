"""Pure exact-nine probability ensemble and candidate-utility helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from ...metrics import balanced_accuracy
from ...protocol import ProtocolError
from ..residual_topup.hashing import array_sha256, canonical_sha256
from .ensemble_endpoint_contracts import (
    ENSEMBLE_ENDPOINT_SEMANTICS,
    ENSEMBLE_SEED_KEYS,
    ENSEMBLE_SEED_PAIR_COUNT,
    ENSEMBLE_THRESHOLD,
    ENSEMBLE_UTILITY_SEMANTICS,
    SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
    SUPPORT_ACTION_PROBABILITY_SHIFT_SCHEMA,
    SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
    SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS,
    ProbabilityEnsembleEndpoint,
    SeedProbabilityVector,
    SupportActionProbabilityShift,
)
from .ensemble_utility_contracts import (
    EnsembleUtilityResponse,
    EnsembleUtilitySurface,
    ScoredEnsembleUtilityResponse,
)
from .row_contracts import INNER_CANDIDATE_COUNT, TARGET_CANDIDATE_COUNT


def score_nine_seed_probability_ensemble(
    vectors: Sequence[SeedProbabilityVector],
    labels: Sequence[int] | np.ndarray,
) -> ProbabilityEnsembleEndpoint:
    """Average exact-nine probabilities, threshold once, then score BACC.

    The input sequence itself must use the canonical training-major 3x3 order.
    This makes accidental order drift or duplicated seed cells fail closed even
    though arithmetic averaging would otherwise hide the problem.
    """

    cells = _validated_probability_vectors(vectors, name="ensemble probabilities")
    truth = _validated_binary_labels(labels, row_count=len(cells[0].positive_class_probabilities))
    mean_probability = mean_exact_nine_positive_class_probabilities(cells)
    prediction = (mean_probability >= ENSEMBLE_THRESHOLD).astype(np.uint8)
    for value in (mean_probability, prediction):
        value.setflags(write=False)
    label_hash = array_sha256(truth)
    bacc = float(balanced_accuracy(truth.tolist(), prediction.tolist()))
    unhashed = {
        "schema_version": "midogpp_utility_aligned_probability_ensemble_endpoint_v1",
        "row_identity_hash": cells[0].row_identity_hash,
        "label_sha256": label_hash,
        "seed_pair_count": ENSEMBLE_SEED_PAIR_COUNT,
        "seed_keys": [list(key) for key in ENSEMBLE_SEED_KEYS],
        "component_vector_hashes": [cell.vector_hash for cell in cells],
        "row_count": len(truth),
        "mean_probability_sha256": array_sha256(mean_probability),
        "prediction_sha256": array_sha256(prediction),
        "balanced_accuracy": bacc,
        "threshold": ENSEMBLE_THRESHOLD,
        "endpoint_semantics": ENSEMBLE_ENDPOINT_SEMANTICS,
    }
    return ProbabilityEnsembleEndpoint(
        row_identity_hash=cells[0].row_identity_hash,
        label_hash=label_hash,
        seed_keys=ENSEMBLE_SEED_KEYS,
        component_vector_hashes=tuple(cell.vector_hash for cell in cells),
        mean_positive_probabilities=mean_probability,
        predictions=prediction,
        balanced_accuracy=bacc,
        endpoint_hash=canonical_sha256(unhashed),
    )


def mean_exact_nine_positive_class_probabilities(
    vectors: Sequence[SeedProbabilityVector],
) -> np.ndarray:
    """Return only the immutable float64 exact-nine arithmetic mean.

    This helper is label-free and performs no thresholding or scoring, so a
    frozen Stage-70 policy can compose predictions without importing an
    evaluation-label capability.
    """

    cells = _validated_probability_vectors(vectors, name="ensemble probabilities")
    probabilities = np.stack(
        [cell.positive_class_probabilities for cell in cells], axis=0
    )
    mean_probability = np.mean(probabilities, axis=0, dtype=np.float64)
    if mean_probability.dtype != np.float64 or not np.isfinite(mean_probability).all():
        raise ProtocolError("Exact-nine probability mean is non-finite or not float64.")
    mean_probability.setflags(write=False)
    return mean_probability


def support_action_probability_shift(
    base_vectors: Sequence[SeedProbabilityVector],
    tail_vectors: Sequence[SeedProbabilityVector],
) -> SupportActionProbabilityShift:
    """Compute a fixed label-free one-dimensional support action shift.

    The scalar first averages the exact canonical nine probability vectors,
    then takes the absolute tail-minus-base difference row by row, and only
    then averages over support rows.  Per-seed absolute shifts are retained as
    descriptive technical-seed spread and may never feed the model.
    """

    base = _validated_probability_vectors(base_vectors, name="base support probabilities")
    tail = _validated_probability_vectors(tail_vectors, name="tail support probabilities")
    _validate_paired_vector_geometry(base, tail)
    per_seed = np.asarray(
        [
            float(
                np.mean(
                    np.abs(
                        tail_cell.positive_class_probabilities
                        - base_cell.positive_class_probabilities
                    ),
                    dtype=np.float64,
                )
            )
            for base_cell, tail_cell in zip(base, tail)
        ],
        dtype=np.float64,
    )
    base_ensemble = mean_exact_nine_positive_class_probabilities(base)
    tail_ensemble = mean_exact_nine_positive_class_probabilities(tail)
    absolute_difference = np.abs(tail_ensemble - base_ensemble)
    value = float(np.mean(absolute_difference, dtype=np.float64))
    standard_deviation = float(np.std(per_seed, ddof=0, dtype=np.float64))
    minimum = float(np.min(per_seed))
    maximum = float(np.max(per_seed))
    unhashed = {
        "schema_version": SUPPORT_ACTION_PROBABILITY_SHIFT_SCHEMA,
        "row_identity_hash": base[0].row_identity_hash,
        "seed_pair_count": ENSEMBLE_SEED_PAIR_COUNT,
        "seed_keys": [list(key) for key in ENSEMBLE_SEED_KEYS],
        "base_component_vector_hashes": [cell.vector_hash for cell in base],
        "tail_component_vector_hashes": [cell.vector_hash for cell in tail],
        "per_seed_mean_absolute_shifts": per_seed.tolist(),
        "technical_seed_spread_semantics": (
            SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS
        ),
        "technical_seed_values_may_feed_model": False,
        "base_ensemble_probability_sha256": array_sha256(base_ensemble),
        "tail_ensemble_probability_sha256": array_sha256(tail_ensemble),
        "ensemble_absolute_difference_sha256": array_sha256(
            absolute_difference
        ),
        "value": value,
        "seed_standard_deviation": standard_deviation,
        "seed_minimum": minimum,
        "seed_maximum": maximum,
        "seed_range": maximum - minimum,
        "scalar_name": SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
        "scalar_semantics": SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
        "labels_used": False,
    }
    return SupportActionProbabilityShift(
        row_identity_hash=base[0].row_identity_hash,
        seed_keys=ENSEMBLE_SEED_KEYS,
        base_component_vector_hashes=tuple(cell.vector_hash for cell in base),
        tail_component_vector_hashes=tuple(cell.vector_hash for cell in tail),
        per_seed_mean_absolute_shifts=tuple(float(value) for value in per_seed),
        base_ensemble_probability_hash=array_sha256(base_ensemble),
        tail_ensemble_probability_hash=array_sha256(tail_ensemble),
        ensemble_absolute_difference_hash=array_sha256(absolute_difference),
        value=value,
        seed_standard_deviation=standard_deviation,
        seed_minimum=minimum,
        seed_maximum=maximum,
        seed_range=maximum - minimum,
        shift_hash=canonical_sha256(unhashed),
    )


def build_ensemble_utility_response(
    *,
    outer_target_id: str,
    query_id: str,
    candidate_source: str,
    base_vectors: Sequence[SeedProbabilityVector],
    tail_vectors: Sequence[SeedProbabilityVector],
    labels: Sequence[int] | np.ndarray,
    support_partition_hash: str,
    evaluation_partition_hash: str,
    prediction_seal_hash: str,
    support_eval_disjoint: bool,
    predictions_sealed_before_labels: bool,
    source_expert_frozen: bool,
    target_labels_used_for_routing: bool = False,
) -> EnsembleUtilityResponse:
    """Build the only label-bearing candidate-level utility response."""

    base = _validated_probability_vectors(base_vectors, name="base probabilities")
    tail = _validated_probability_vectors(tail_vectors, name="tail probabilities")
    _validate_paired_vector_geometry(base, tail)
    base_endpoint = score_nine_seed_probability_ensemble(base, labels)
    tail_endpoint = score_nine_seed_probability_ensemble(tail, labels)
    return EnsembleUtilityResponse(
        outer_target_id=outer_target_id,
        query_id=query_id,
        candidate_source=candidate_source,
        candidate_source_count=INNER_CANDIDATE_COUNT,
        support_partition_hash=support_partition_hash,
        evaluation_partition_hash=evaluation_partition_hash,
        prediction_seal_hash=prediction_seal_hash,
        base_endpoint=base_endpoint,
        tail_endpoint=tail_endpoint,
        support_eval_disjoint=support_eval_disjoint,
        predictions_sealed_before_labels=predictions_sealed_before_labels,
        source_expert_frozen=source_expert_frozen,
        target_labels_used_for_routing=target_labels_used_for_routing,
        utility_semantics=ENSEMBLE_UTILITY_SEMANTICS,
    )


def validate_ensemble_utility_responses(
    rows: Sequence[
        EnsembleUtilityResponse | ScoredEnsembleUtilityResponse | Mapping[str, object]
    ],
) -> EnsembleUtilitySurface:
    """Validate complete candidate-level source-inner H/q/e geometry."""

    if not rows:
        raise ProtocolError("Ensemble utility requires nonempty responses.")
    coerced = tuple(coerce_scored_ensemble_utility_response(row) for row in rows)
    ordered = tuple(sorted(coerced, key=lambda row: row.row_key))
    row_keys = tuple(row.row_key for row in ordered)
    if len(set(row_keys)) != len(row_keys):
        raise ProtocolError("Ensemble utility contains duplicate H/q/e responses.")
    by_outer: dict[str, list[ScoredEnsembleUtilityResponse]] = defaultdict(list)
    for row in ordered:
        by_outer[row.outer_target_id].append(row)
    universe: tuple[str, ...] | None = None
    for outer, outer_rows in sorted(by_outer.items()):
        queries = tuple(sorted({row.query_id for row in outer_rows}))
        if len(queries) != TARGET_CANDIDATE_COUNT or outer in queries:
            raise ProtocolError("Each outer target requires eight non-target pseudoqueries.")
        observed_universe = tuple(sorted((outer, *queries)))
        if universe is None:
            universe = observed_universe
        elif observed_universe != universe:
            raise ProtocolError("Ensemble outer-target domain universes drifted.")
        for query in queries:
            query_rows = tuple(row for row in outer_rows if row.query_id == query)
            expected_sources = tuple(source for source in queries if source != query)
            if (
                len(query_rows) != INNER_CANDIDATE_COUNT
                or {row.candidate_source for row in query_rows} != set(expected_sources)
            ):
                raise ProtocolError("Ensemble candidate response list is incomplete.")
            if len({row.support_partition_hash for row in query_rows}) != 1:
                raise ProtocolError("Ensemble support partition drifted within query.")
            if len({row.evaluation_partition_hash for row in query_rows}) != 1:
                raise ProtocolError("Ensemble evaluation partition drifted within query.")
            if len({row.prediction_seal_hash for row in query_rows}) != 1:
                raise ProtocolError("Ensemble prediction seal drifted within query.")
            if len({row.base_endpoint_hash for row in query_rows}) != 1:
                raise ProtocolError("Paired candidates must share one base ensemble endpoint.")
    payload = {
        "schema_version": "midogpp_utility_aligned_ensemble_utility_surface_v1",
        "outer_target_ids": sorted(by_outer),
        "row_count": len(ordered),
        "row_hashes": [row.row_hash for row in ordered],
        "response_unit": "candidate_H_q_e_after_exact_nine_probability_ensemble",
        "seed_rows_are_independent_observations": False,
        "target_labels_used_for_routing": False,
    }
    return EnsembleUtilitySurface(
        rows=ordered,
        outer_target_ids=tuple(sorted(by_outer)),
        row_keys=row_keys,
        surface_hash=canonical_sha256(payload),
    )


def scored_ensemble_utility_response_from_payload(
    payload: Mapping[str, object],
) -> ScoredEnsembleUtilityResponse:
    """Parse either the neutral scored DTO or the Stage-60 endpoint-row schema."""

    if not isinstance(payload, Mapping):
        raise ProtocolError("Scored ensemble response payload must be a mapping.")
    schema = str(payload.get("schema_version", ""))
    if schema == "midogpp_utility_aligned_scored_ensemble_utility_response_v1":
        response = ScoredEnsembleUtilityResponse(
            outer_target_id=str(payload["outer_target_id"]),
            query_id=str(payload["query_id"]),
            candidate_source=str(payload["candidate_source"]),
            candidate_source_count=int(payload["candidate_source_count"]),
            support_partition_hash=str(payload["support_partition_hash"]),
            evaluation_partition_hash=str(payload["evaluation_partition_hash"]),
            prediction_seal_hash=str(payload["prediction_seal_hash"]),
            evaluation_row_identity_hash=str(payload["evaluation_row_identity_hash"]),
            evaluation_label_hash=(
                None
                if payload.get("evaluation_label_hash") is None
                else str(payload["evaluation_label_hash"])
            ),
            base_endpoint_hash=str(payload["base_endpoint_hash"]),
            tail_endpoint_hash=str(payload["tail_endpoint_hash"]),
            base_probability_cell_hashes_hash=str(
                payload["base_probability_cell_hashes_hash"]
            ),
            tail_probability_cell_hashes_hash=str(
                payload["tail_probability_cell_hashes_hash"]
            ),
            base_ensemble_probability_hash=str(
                payload["base_ensemble_probability_hash"]
            ),
            tail_ensemble_probability_hash=str(
                payload["tail_ensemble_probability_hash"]
            ),
            base_ensemble_prediction_hash=str(
                payload["base_ensemble_prediction_hash"]
            ),
            tail_ensemble_prediction_hash=str(
                payload["tail_ensemble_prediction_hash"]
            ),
            source_response_hash=(
                None
                if payload.get("source_response_hash") is None
                else str(payload["source_response_hash"])
            ),
            source_endpoint_row_hash=(
                None
                if payload.get("source_endpoint_row_hash") is None
                else str(payload["source_endpoint_row_hash"])
            ),
            base_component_vector_hashes=tuple(
                str(value)
                for value in payload.get("base_component_vector_hashes", ())
            ),
            tail_component_vector_hashes=tuple(
                str(value)
                for value in payload.get("tail_component_vector_hashes", ())
            ),
            base_bacc=float(payload["base_bacc"]),
            tail_bacc=float(payload["tail_bacc"]),
            support_eval_disjoint=payload["support_eval_disjoint"] is True,
            predictions_sealed_before_labels=(
                payload["predictions_sealed_before_labels"] is True
            ),
            source_expert_frozen=payload["source_expert_frozen"] is True,
            target_labels_used_for_routing=(
                payload.get("target_labels_used_for_routing", False) is True
            ),
            utility_semantics=str(payload["utility_semantics"]),
        )
        if "utility_delta" in payload and not np.isclose(
            float(payload["utility_delta"]), response.utility_delta, rtol=0.0, atol=1e-12
        ):
            raise ProtocolError("Persisted scored utility delta drifted.")
        return response
    if schema != "midogpp_exact_tail_ensemble_endpoint_row_v1":
        raise ProtocolError("Scored ensemble response schema is unsupported.")
    required_true = (
        "development_labels_used_for_scoring_only",
        "technical_seed_repeats_are_not_independent_units",
    )
    required_false = (
        "target_support_labels_used",
        "target_evaluation_labels_used",
        "seed_selection_performed",
    )
    if any(payload.get(name) is not True for name in required_true) or any(
        payload.get(name) is not False for name in required_false
    ):
        raise ProtocolError("Persisted endpoint row violates label/seed-unit boundaries.")
    if (
        int(payload.get("seed_pair_count", -1)) != ENSEMBLE_SEED_PAIR_COUNT
        or not np.isclose(
            float(payload.get("threshold", np.nan)),
            ENSEMBLE_THRESHOLD,
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise ProtocolError("Persisted endpoint row seed count/threshold drifted.")
    base_bacc = float(payload["base_bacc"])
    tail_bacc = float(payload["tail_bacc"])
    if not np.isclose(
        float(payload["delta_bacc"]), tail_bacc - base_bacc, rtol=0.0, atol=1e-12
    ):
        raise ProtocolError("Persisted endpoint row delta does not match BACC fields.")
    evaluation_row_count = int(payload["evaluation_row_count"])
    evaluation_case_count = int(payload["evaluation_case_count"])
    if evaluation_row_count <= 0 or not 0 < evaluation_case_count <= evaluation_row_count:
        raise ProtocolError("Persisted endpoint evaluation geometry is invalid.")
    evaluation_row_hash = str(payload["evaluation_row_hash"])
    support_hash = str(payload["support_partition_hash"])
    response = ScoredEnsembleUtilityResponse(
        outer_target_id=str(payload["outer_target"]),
        query_id=str(payload["pseudo_query"]),
        candidate_source=str(payload["candidate_source"]),
        candidate_source_count=INNER_CANDIDATE_COUNT,
        support_partition_hash=support_hash,
        evaluation_partition_hash=evaluation_row_hash,
        prediction_seal_hash=str(payload["prediction_seal_hash"]),
        evaluation_row_identity_hash=evaluation_row_hash,
        evaluation_label_hash=str(payload["evaluation_label_sha256"]),
        base_endpoint_hash=str(payload["base_endpoint_hash"]),
        tail_endpoint_hash=str(payload["tail_endpoint_hash"]),
        base_probability_cell_hashes_hash=str(
            payload["base_probability_cell_hashes_hash"]
        ),
        tail_probability_cell_hashes_hash=str(
            payload["tail_probability_cell_hashes_hash"]
        ),
        base_ensemble_probability_hash=str(
            payload["base_ensemble_probability_sha256"]
        ),
        tail_ensemble_probability_hash=str(
            payload["tail_ensemble_probability_sha256"]
        ),
        base_ensemble_prediction_hash=str(
            payload["base_ensemble_prediction_sha256"]
        ),
        tail_ensemble_prediction_hash=str(
            payload["tail_ensemble_prediction_sha256"]
        ),
        source_response_hash=str(payload["ensemble_utility_response_hash"]),
        source_endpoint_row_hash=str(payload["endpoint_row_hash"]),
        base_component_vector_hashes=(),
        tail_component_vector_hashes=(),
        base_bacc=base_bacc,
        tail_bacc=tail_bacc,
        support_eval_disjoint=support_hash != evaluation_row_hash,
        predictions_sealed_before_labels=True,
        source_expert_frozen=True,
        target_labels_used_for_routing=False,
        utility_semantics=ENSEMBLE_UTILITY_SEMANTICS,
    )
    return response


def coerce_scored_ensemble_utility_response(
    value: EnsembleUtilityResponse | ScoredEnsembleUtilityResponse | Mapping[str, object],
) -> ScoredEnsembleUtilityResponse:
    """Normalize scoring-boundary objects or persisted rows to the model DTO."""

    if isinstance(value, ScoredEnsembleUtilityResponse):
        return value
    if isinstance(value, EnsembleUtilityResponse):
        return value.to_scored_response()
    if isinstance(value, Mapping):
        return scored_ensemble_utility_response_from_payload(value)
    raise ProtocolError("Ensemble utility response cannot be coerced to scored DTO.")


def _validated_probability_vectors(
    vectors: Sequence[SeedProbabilityVector], *, name: str
) -> tuple[SeedProbabilityVector, ...]:
    cells = tuple(vectors)
    if (
        len(cells) != ENSEMBLE_SEED_PAIR_COUNT
        or any(not isinstance(cell, SeedProbabilityVector) for cell in cells)
    ):
        raise ProtocolError(f"{name} require exactly nine typed seed vectors.")
    keys = tuple(cell.seed_key for cell in cells)
    if len(set(keys)) != len(keys):
        raise ProtocolError(f"{name} contain duplicate seed keys.")
    if keys != ENSEMBLE_SEED_KEYS:
        raise ProtocolError(f"{name} must use canonical training-major seed order.")
    shapes = {cell.positive_class_probabilities.shape for cell in cells}
    row_hashes = {cell.row_identity_hash for cell in cells}
    if len(shapes) != 1 or len(row_hashes) != 1:
        raise ProtocolError(f"{name} have inconsistent row geometry.")
    return cells


def _validate_paired_vector_geometry(
    base: tuple[SeedProbabilityVector, ...],
    tail: tuple[SeedProbabilityVector, ...],
) -> None:
    for base_cell, tail_cell in zip(base, tail):
        if (
            base_cell.seed_key != tail_cell.seed_key
            or base_cell.row_identity_hash != tail_cell.row_identity_hash
            or base_cell.positive_class_probabilities.shape
            != tail_cell.positive_class_probabilities.shape
        ):
            raise ProtocolError("Base/tail probability vector geometry is not paired.")


def _validated_binary_labels(
    labels: Sequence[int] | np.ndarray, *, row_count: int
) -> np.ndarray:
    raw = np.asarray(labels)
    if raw.ndim != 1 or raw.shape != (row_count,):
        raise ProtocolError("Ensemble labels do not match probability row geometry.")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Ensemble labels must be binary integers.") from exc
    if (
        not np.isfinite(numeric).all()
        or not np.equal(numeric, np.floor(numeric)).all()
        or set(int(value) for value in numeric) != {0, 1}
    ):
        raise ProtocolError("Ensemble BACC requires both binary classes.")
    truth = numeric.astype(np.uint8)
    truth.setflags(write=False)
    return truth


__all__ = (
    "build_ensemble_utility_response",
    "coerce_scored_ensemble_utility_response",
    "mean_exact_nine_positive_class_probabilities",
    "scored_ensemble_utility_response_from_payload",
    "score_nine_seed_probability_ensemble",
    "support_action_probability_shift",
    "validate_ensemble_utility_responses",
)
