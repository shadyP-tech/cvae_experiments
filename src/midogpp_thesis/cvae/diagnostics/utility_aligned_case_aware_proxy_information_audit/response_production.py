"""Post-seal response and descriptive seed-row production."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import array_sha256, canonical_sha256
from ..utility_aligned_ensemble_endpoint_router.contracts import (
    BASE_ACTION_ID,
    h_x_e_action_id,
)
from .contracts import (
    CENTERS,
    EXPECTED_FEATURE_ROW_COUNT,
    EXPECTED_STRICT_CROSSFIT_TRAINING_ROW_COUNT,
    FAMILY_IDS,
    RESPONSE_NAMES,
    SEED_PAIRS,
    CaseAwareCrossfitResult,
    CaseAwareFeatureSurface,
    CaseAwareResponseSurface,
)
from .feature_production import case_identity_hash
from .input_contracts import row_identity_hash
from .response_surfaces import (
    ExactNineEvaluationVectors,
    balanced_accuracy,
    build_response_row,
    build_response_surface,
    soft_balanced_accuracy,
)


_HASH_TOKEN = re.compile(r"(?:[0-9a-f]{16}|[0-9a-f]{64})")


@dataclass(frozen=True)
class CaseAwareResponseProduction:
    """Complete post-seal scientific responses and descriptive seed cells."""

    surface: CaseAwareResponseSurface
    descriptive_seed_rows: tuple[Mapping[str, object], ...]

    @property
    def rows(self) -> tuple[object, ...]:
        return self.surface.rows


def produce_case_aware_responses(
    feature_surface: CaseAwareFeatureSurface,
    feature_lock: Mapping[str, object],
    development: object,
    labels: object,
    partitions: object,
) -> CaseAwareResponseProduction:
    """Score exact-nine predictions only after the durable seal opens labels."""

    if not isinstance(feature_surface, CaseAwareFeatureSurface):
        raise ProtocolError("Response production requires the sealed feature surface.")
    store, seal_hash = _validated_scoring_capability(
        development, labels=labels, partitions=partitions
    )
    feature_lock_hash = _validated_feature_lock(
        feature_surface,
        feature_lock,
        expected_prediction_seal_hash=seal_hash,
        expected_partition_lock_hash=str(getattr(partitions, "lock_hash", "")),
    )
    evaluation_by_center = getattr(partitions, "evaluation_rows_by_center", None)
    if not isinstance(evaluation_by_center, Mapping) or tuple(evaluation_by_center) != CENTERS:
        raise ProtocolError("Case-aware evaluation partitions are absent.")

    vector_cache: dict[tuple[str, str], tuple[object, ...]] = {}
    response_rows: list[object] = []
    seed_rows: list[Mapping[str, object]] = []
    for feature in feature_surface.rows:
        outer, query, source = feature.row_key
        scope = f"{outer}::{query}"
        evaluation_rows = tuple(evaluation_by_center[query])
        evaluation_row_hash = row_identity_hash(evaluation_rows)
        truth = np.asarray(labels.labels_by_center[query], dtype=np.int64)
        if (
            labels.evaluation_row_hash_by_center[query] != evaluation_row_hash
            or truth.shape != (len(evaluation_rows),)
        ):
            raise ProtocolError("Opened test labels drifted from sealed evaluation rows.")
        base_vectors = _evaluation_vectors(
            store,
            scope=scope,
            action_id=BASE_ACTION_ID,
            expected_row_hash=evaluation_row_hash,
            cache=vector_cache,
        )
        tail_vectors = _evaluation_vectors(
            store,
            scope=scope,
            action_id=h_x_e_action_id(source),
            expected_row_hash=evaluation_row_hash,
            cache=vector_cache,
        )
        base_matrix = _probability_matrix(base_vectors)
        tail_matrix = _probability_matrix(tail_vectors)
        case_hashes = tuple(
            case_identity_hash(query, case_id)
            for case_id in sorted({str(row.case_id) for row in evaluation_rows})
        )
        prediction_provenance_hash = canonical_sha256(
            {
                "schema_version": (
                    "midogpp_stage90_case_aware_evaluation_prediction_provenance_v1"
                ),
                "row_key": list(feature.row_key),
                "evaluation_row_hash": evaluation_row_hash,
                "prediction_seal_hash": seal_hash,
                "feature_surface_hash": feature_surface.surface_hash,
                "case_aware_feature_lock_hash": feature_lock_hash,
                "base_vector_hashes": [value.vector_hash for value in base_vectors],
                "tail_vector_hashes": [value.vector_hash for value in tail_vectors],
                "features_sealed_before_label_access": True,
            }
        )
        response_rows.append(
            build_response_row(
                feature_row=feature,
                feature_surface_seal_hash=feature_lock_hash,
                evaluation=ExactNineEvaluationVectors(
                    evaluation_partition_hash=evaluation_row_hash,
                    evaluation_case_hashes=case_hashes,
                    evaluation_row_hash=evaluation_row_hash,
                    prediction_provenance_hash=prediction_provenance_hash,
                    base_probabilities=base_matrix,
                    tail_probabilities=tail_matrix,
                    labels=truth,
                ),
            )
        )
        seed_rows.extend(
            _descriptive_seed_rows(
                feature.row_key,
                base_vectors=base_vectors,
                tail_vectors=tail_vectors,
                labels=truth,
                evaluation_row_hash=evaluation_row_hash,
                feature_surface_hash=feature_surface.surface_hash,
                feature_lock_hash=feature_lock_hash,
                prediction_seal_hash=seal_hash,
            )
        )
    if (
        len(response_rows) != EXPECTED_FEATURE_ROW_COUNT
        or len(seed_rows) != EXPECTED_FEATURE_ROW_COUNT * len(SEED_PAIRS)
    ):
        raise ProtocolError("Case-aware response execution coverage drifted.")
    return CaseAwareResponseProduction(
        surface=build_response_surface(feature_surface, response_rows),
        descriptive_seed_rows=tuple(seed_rows),
    )


def build_crossfit_fold_lock(
    crossfit: CaseAwareCrossfitResult,
) -> dict[str, object]:
    """Compactly bind every exact and smooth strict H/q/e fold."""

    if not isinstance(crossfit, CaseAwareCrossfitResult):
        raise ProtocolError("Crossfit fold locking requires a typed result.")
    expected_count = len(FAMILY_IDS) * len(RESPONSE_NAMES) * EXPECTED_FEATURE_ROW_COUNT
    if (
        crossfit.family_ids != FAMILY_IDS
        or crossfit.response_names != RESPONSE_NAMES
        or len(crossfit.fold_audits) != expected_count
        or any(
            fold.training_row_count != EXPECTED_STRICT_CROSSFIT_TRAINING_ROW_COUNT
            or set(fold.predicted_row_key) != set(fold.excluded_domain_ids)
            for fold in crossfit.fold_audits
        )
    ):
        raise ProtocolError("Case-aware strict crossfit fold coverage drifted.")
    return crossfit.fold_lock_payload()


def _validated_scoring_capability(
    development: object, *, labels: object, partitions: object
) -> tuple[object, str]:
    store = getattr(development, "store", None)
    seal = getattr(development, "seal", None)
    prediction_seal_hash = str(getattr(seal, "prediction_seal_hash", ""))
    if (
        store is None
        or seal is None
        or getattr(store, "role", None) != "development"
        or str(getattr(store, "partition_lock_hash", ""))
        != str(getattr(partitions, "lock_hash", ""))
        or str(getattr(labels, "prediction_seal_hash", ""))
        != prediction_seal_hash
        or _HASH_TOKEN.fullmatch(prediction_seal_hash) is None
    ):
        raise ProtocolError("Case-aware response scoring lacks its sealed capability.")
    return store, prediction_seal_hash


def _validated_feature_lock(
    feature_surface: CaseAwareFeatureSurface,
    feature_lock: Mapping[str, object],
    *,
    expected_prediction_seal_hash: str,
    expected_partition_lock_hash: str,
) -> str:
    if not isinstance(feature_lock, Mapping):
        raise ProtocolError("Response scoring requires the persisted pre-label feature lock.")
    observed = dict(feature_lock)
    supplied_hash = str(observed.get("case_aware_feature_lock_hash", ""))
    unhashed = {
        key: value
        for key, value in observed.items()
        if key != "case_aware_feature_lock_hash"
    }
    if (
        observed.get("schema_version")
        != "midogpp_stage90_case_aware_feature_lock_v1"
        or observed.get("status") != "SEALED_BEFORE_TEST_LABEL_ACCESS"
        or observed.get("feature_surface_hash") != feature_surface.surface_hash
        or observed.get("ordered_feature_row_hashes")
        != [row.feature_row_hash for row in feature_surface.rows]
        or observed.get("feature_row_count") != len(feature_surface.rows)
        or observed.get("development_prediction_seal_hash")
        != expected_prediction_seal_hash
        or observed.get("support_partition_lock_hash")
        != expected_partition_lock_hash
        or observed.get("test_labels_opened") is not False
        or observed.get("support_labels_used") is not False
        or observed.get("evaluation_probabilities_used_as_features") is not False
        or supplied_hash != canonical_sha256(unhashed)
    ):
        raise ProtocolError("Case-aware persisted pre-label feature lock drifted.")
    return supplied_hash


def _evaluation_vectors(
    store: object,
    *,
    scope: str,
    action_id: str,
    expected_row_hash: str,
    cache: dict[tuple[str, str], tuple[object, ...]],
) -> tuple[object, ...]:
    key = (scope, action_id)
    if key not in cache:
        values = tuple(
            sorted(
                store.vectors(scope, action_id, "evaluation"),
                key=lambda value: value.seed_key,
            )
        )
        if (
            tuple(value.seed_key for value in values) != SEED_PAIRS
            or any(value.row_identity_hash != expected_row_hash for value in values)
        ):
            raise ProtocolError("Case-aware exact-nine evaluation vector binding drifted.")
        cache[key] = values
    return cache[key]


def _probability_matrix(vectors: Sequence[object]) -> np.ndarray:
    return np.ascontiguousarray(
        np.stack(
            [
                np.asarray(value.positive_class_probabilities, dtype=np.float64)
                for value in vectors
            ]
        ),
        dtype=np.float64,
    )


def _descriptive_seed_rows(
    row_key: tuple[str, str, str],
    *,
    base_vectors: Sequence[object],
    tail_vectors: Sequence[object],
    labels: np.ndarray,
    evaluation_row_hash: str,
    feature_surface_hash: str,
    feature_lock_hash: str,
    prediction_seal_hash: str,
) -> tuple[Mapping[str, object], ...]:
    output: list[Mapping[str, object]] = []
    label_sha256 = array_sha256(
        np.ascontiguousarray(labels, dtype=np.uint8)
    )
    for base, tail in zip(base_vectors, tail_vectors, strict=True):
        base_probability = np.asarray(base.positive_class_probabilities, dtype=np.float64)
        tail_probability = np.asarray(tail.positive_class_probabilities, dtype=np.float64)
        exact_base = balanced_accuracy(labels, (base_probability >= 0.5).astype(np.int64))
        exact_tail = balanced_accuracy(labels, (tail_probability >= 0.5).astype(np.int64))
        smooth_base = soft_balanced_accuracy(labels, base_probability)
        smooth_tail = soft_balanced_accuracy(labels, tail_probability)
        unhashed = {
            "schema_version": "midogpp_stage90_case_aware_seed_utility_diagnostic_v1",
            "outer_target_id": row_key[0],
            "query_id": row_key[1],
            "candidate_source": row_key[2],
            "training_seed": int(base.training_seed),
            "generation_seed": int(base.generation_seed),
            "evaluation_row_hash": evaluation_row_hash,
            "evaluation_label_sha256": label_sha256,
            "feature_surface_hash": feature_surface_hash,
            "case_aware_feature_lock_hash": feature_lock_hash,
            "prediction_seal_hash": prediction_seal_hash,
            "base_vector_hash": str(base.vector_hash),
            "tail_vector_hash": str(tail.vector_hash),
            "exact_base_bacc": exact_base,
            "exact_tail_bacc": exact_tail,
            "exact_bacc_delta": exact_tail - exact_base,
            "smooth_base_bacc": smooth_base,
            "smooth_tail_bacc": smooth_tail,
            "smooth_bacc_delta": smooth_tail - smooth_base,
            "descriptive_only": True,
            "may_feed_crossfit_or_primary_gate": False,
            "technical_seed_row_is_independent_observation": False,
        }
        output.append(
            {**unhashed, "row_hash": canonical_sha256(unhashed)}
        )
    return tuple(output)


produce_response_surface = produce_case_aware_responses


__all__ = (
    "CaseAwareResponseProduction",
    "build_crossfit_fold_lock",
    "produce_case_aware_responses",
    "produce_response_surface",
)
