"""Case-balanced, label-free support feature construction."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .contracts import (
    CENTERS,
    EXPECTED_FEATURE_ROW_COUNT,
    FEATURE_ROW_SCHEMA,
    CaseAwareFeatureSurface,
    CaseAwareProxyFeatureRow,
    SupportCaseVectors,
    candidate_sources,
    expected_row_keys,
)


_FEATURE_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "outer_target_id",
        "query_id",
        "candidate_source",
        "candidate_source_count",
        "support_partition_hash",
        "prediction_seal_hash",
        "support_case_count",
        "support_row_count",
        "support_case_hashes",
        "support_row_hashes",
        "support_provenance_hashes",
        "base_vector_hashes_by_case",
        "tail_vector_hashes_by_case",
        "metadata_similarity",
        "pooled_row_weighted_abs_shift",
        "equal_case_abs_shift",
        "case_abs_shift_sd",
        "equal_case_signed_margin",
        "case_balanced_flip_rate",
        "case_balanced_entropy_change",
        "case_balanced_reconstruction",
        "case_balanced_kl",
        "case_balanced_log_mmd",
        "probability_role_used",
        "labels_used",
        "evaluation_probabilities_used_as_features",
        "technical_seed_rows_are_independent_observations",
        "feature_row_hash",
    }
)


def build_case_aware_feature_row(
    *,
    outer_target_id: str,
    query_id: str,
    candidate_source: str,
    support_partition_hash: str,
    prediction_seal_hash: str,
    metadata_similarity: float,
    cases: Sequence[SupportCaseVectors],
) -> CaseAwareProxyFeatureRow:
    """Collapse exact-nine support vectors into equal-case primitives.

    Seed probabilities are averaged first, within each row.  Row summaries are
    then computed within each case, and all case-aware primitives give every
    whole case equal weight.  Only the explicitly named pooled control weights
    rows directly.
    """

    ordered = tuple(sorted(cases, key=lambda value: (value.case_id, value.case_hash)))
    if not ordered or any(not isinstance(value, SupportCaseVectors) for value in ordered):
        raise ProtocolError("Case-aware feature construction requires typed cases.")
    if len({value.case_id for value in ordered}) != len(ordered):
        raise ProtocolError("Support cases must have unique case IDs.")
    if len({value.case_hash for value in ordered}) != len(ordered):
        raise ProtocolError("Support cases must have unique case hashes.")

    case_abs_shift: list[float] = []
    case_signed_margin: list[float] = []
    case_flip_rate: list[float] = []
    case_entropy_change: list[float] = []
    pooled_abs_sum = 0.0
    pooled_row_count = 0
    for case in ordered:
        # The exact-nine probability mean is deliberately upstream of every
        # threshold, absolute value, entropy, and case collapse.
        base = np.mean(case.base_probabilities, axis=0, dtype=np.float64)
        tail = np.mean(case.tail_probabilities, axis=0, dtype=np.float64)
        delta = tail - base
        absolute = np.abs(delta)
        pseudo_sign = np.where(base >= 0.5, 1.0, -1.0)
        case_abs_shift.append(float(np.mean(absolute, dtype=np.float64)))
        case_signed_margin.append(
            float(np.mean(pseudo_sign * delta, dtype=np.float64))
        )
        case_flip_rate.append(
            float(
                np.mean(
                    (base - 0.5) * (tail - 0.5) < 0.0,
                    dtype=np.float64,
                )
            )
        )
        case_entropy_change.append(
            float(
                np.mean(
                    _binary_entropy(tail) - _binary_entropy(base),
                    dtype=np.float64,
                )
            )
        )
        pooled_abs_sum += float(np.sum(absolute, dtype=np.float64))
        pooled_row_count += case.row_count

    abs_array = np.asarray(case_abs_shift, dtype=np.float64)
    return CaseAwareProxyFeatureRow(
        outer_target_id=outer_target_id,
        query_id=query_id,
        candidate_source=candidate_source,
        candidate_source_count=len(candidate_sources(outer_target_id, query_id)),
        support_partition_hash=support_partition_hash,
        prediction_seal_hash=prediction_seal_hash,
        support_case_count=len(ordered),
        support_row_count=pooled_row_count,
        support_case_hashes=tuple(value.case_hash for value in ordered),
        support_row_hashes=tuple(value.row_hash for value in ordered),
        support_provenance_hashes=tuple(value.provenance_hash for value in ordered),
        base_vector_hashes_by_case=tuple(
            value.base_vector_hashes for value in ordered
        ),
        tail_vector_hashes_by_case=tuple(
            value.tail_vector_hashes for value in ordered
        ),
        metadata_similarity=metadata_similarity,
        pooled_row_weighted_abs_shift=pooled_abs_sum / float(pooled_row_count),
        equal_case_abs_shift=float(np.mean(abs_array, dtype=np.float64)),
        # Population SD describes the complete fixed support-case set; these
        # case values are a predictor, not an inferential variance estimator.
        case_abs_shift_sd=float(np.std(abs_array, ddof=0)),
        equal_case_signed_margin=_mean(case_signed_margin),
        case_balanced_flip_rate=_mean(case_flip_rate),
        case_balanced_entropy_change=_mean(case_entropy_change),
        case_balanced_reconstruction=_mean(
            value.reconstruction_summary for value in ordered
        ),
        case_balanced_kl=_mean(value.kl_summary for value in ordered),
        case_balanced_log_mmd=_mean(
            value.log_mmd_summary for value in ordered
        ),
    )


# Longer name retained as a discoverable, explicit alias for adapters.
build_case_aware_proxy_feature_row = build_case_aware_feature_row


def feature_row_from_payload(payload: Mapping[str, object]) -> CaseAwareProxyFeatureRow:
    """Fail-closed parser for the sealed label-free row schema."""

    if not isinstance(payload, Mapping) or set(payload) != _FEATURE_PAYLOAD_KEYS:
        raise ProtocolError("Case-aware feature payload does not match the exact schema.")
    if payload.get("schema_version") != FEATURE_ROW_SCHEMA:
        raise ProtocolError("Case-aware feature payload schema drifted.")
    supplied_hash = payload.get("feature_row_hash")
    unhashed = {key: payload[key] for key in payload if key != "feature_row_hash"}
    if supplied_hash != canonical_sha256(unhashed):
        raise ProtocolError("Case-aware feature payload hash drifted.")
    kwargs = {
        key: payload[key]
        for key in payload
        if key not in {"schema_version", "feature_row_hash"}
    }
    row = CaseAwareProxyFeatureRow(**kwargs)  # type: ignore[arg-type]
    if row.feature_row_hash != supplied_hash:
        raise ProtocolError("Case-aware feature reconstruction hash drifted.")
    return row


def build_case_aware_feature_surface(
    rows: Sequence[CaseAwareProxyFeatureRow | Mapping[str, object]],
) -> CaseAwareFeatureSurface:
    """Validate and canonically order the complete 504-row H/q/e surface."""

    typed = tuple(
        value if isinstance(value, CaseAwareProxyFeatureRow) else feature_row_from_payload(value)
        for value in rows
    )
    if len(typed) != EXPECTED_FEATURE_ROW_COUNT:
        raise ProtocolError("Case-aware feature surface requires complete H/q/e coverage.")
    keyed = {row.row_key: row for row in typed}
    expected = expected_row_keys()
    if len(keyed) != len(typed) or set(keyed) != set(expected):
        raise ProtocolError("Case-aware feature H/q/e geometry drifted.")
    ordered = tuple(keyed[key] for key in expected)
    if len({row.prediction_seal_hash for row in ordered}) != 1:
        raise ProtocolError("Feature rows must share one pre-label prediction seal.")
    for outer in CENTERS:
        for query in (center for center in CENTERS if center != outer):
            group = tuple(
                row
                for row in ordered
                if row.outer_target_id == outer and row.query_id == query
            )
            if tuple(row.candidate_source for row in group) != candidate_sources(
                outer, query
            ):
                raise ProtocolError("Feature candidate ordering drifted.")
            # Whole support partitions and base probabilities are shared by
            # all candidate tails for one H/q query.  Candidate provenance is
            # deliberately excluded: it binds the source-specific tail,
            # generated streams, components, and proxy summaries.
            shared_support_identity = {
                (
                    row.support_partition_hash,
                    row.support_case_count,
                    row.support_row_count,
                    row.support_case_hashes,
                    row.support_row_hashes,
                    row.base_vector_hashes_by_case,
                )
                for row in group
            }
            if len(shared_support_identity) != 1:
                raise ProtocolError("Within-query shared support identity drifted.")
            candidate_lineage = {
                (
                    row.support_provenance_hashes,
                    row.tail_vector_hashes_by_case,
                )
                for row in group
            }
            if len(candidate_lineage) != len(group):
                raise ProtocolError(
                    "Within-query candidate support lineage collapsed."
                )
    for query in CENTERS:
        partition_identities = {
            (
                row.support_partition_hash,
                row.support_case_count,
                row.support_row_count,
                row.support_case_hashes,
                row.support_row_hashes,
            )
            for row in ordered
            if row.query_id == query
        }
        if len(partition_identities) != 1:
            raise ProtocolError(
                "Support whole-case partition identity drifted across outer folds."
            )
        for source in (center for center in CENTERS if center != query):
            group = tuple(
                row
                for row in ordered
                if row.query_id == query and row.candidate_source == source
            )
            expected_outers = tuple(
                center for center in CENTERS if center not in {query, source}
            )
            if tuple(row.outer_target_id for row in group) != expected_outers:
                raise ProtocolError("Query/source outer-fold coverage drifted.")
            candidate_static_identity = {
                (
                    row.metadata_similarity,
                    row.case_balanced_reconstruction,
                    row.case_balanced_kl,
                    row.case_balanced_log_mmd,
                )
                for row in group
            }
            if len(candidate_static_identity) != 1:
                raise ProtocolError("Query/source proxy identity drifted.")
    unhashed = {
        "schema_version": "midogpp_stage90_case_aware_feature_surface_v1",
        "ordered_feature_row_hashes": [row.feature_row_hash for row in ordered],
        "row_keys": [list(key) for key in expected],
        "row_count": len(ordered),
        "minimum_support_case_count": min(
            row.support_case_count for row in ordered
        ),
        "labels_used": False,
        "evaluation_probabilities_used_as_features": False,
        "exact_nine_mean_before_case_aggregation": True,
        "technical_seed_rows_are_independent_observations": False,
    }
    return CaseAwareFeatureSurface(
        rows=ordered,
        row_keys=expected,
        surface_hash=canonical_sha256(unhashed),
    )


build_feature_surface = build_case_aware_feature_surface


def _binary_entropy(probability: np.ndarray) -> np.ndarray:
    epsilon = np.finfo(np.float64).eps
    values = np.clip(np.asarray(probability, dtype=np.float64), epsilon, 1.0 - epsilon)
    return -(
        values * np.log(values) + (1.0 - values) * np.log(1.0 - values)
    )


def _mean(values: object) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)  # type: ignore[arg-type]
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ProtocolError("Case-balanced mean requires finite nonempty values.")
    return float(np.mean(array, dtype=np.float64))


__all__ = (
    "build_case_aware_feature_row",
    "build_case_aware_feature_surface",
    "build_case_aware_proxy_feature_row",
    "build_feature_surface",
    "feature_row_from_payload",
)
