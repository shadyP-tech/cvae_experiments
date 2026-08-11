"""Strict source-OOF exact-BACC additive response construction."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from ...protocol import ProtocolError
from .contracts import (
    CaseActionResponseRow,
    DisagreementFeatureSurface,
    ExactRegretSurface,
    SourceOOFLabelRow,
)
from .hashing import canonical_sha256
from .provenance import DevelopmentContext, DevelopmentScope, assert_development_context
from .probability_contracts import LABEL_FREE_INFERENCE_SURFACE_ROLE


def build_exact_regret_surface(
    features: DisagreementFeatureSurface,
    labels: Sequence[SourceOOFLabelRow],
    *,
    context: DevelopmentContext,
) -> ExactRegretSurface:
    """Create exact additive BACC gains from synthetic/source-OOF labels only.

    The declared outer target must have no label row.  A labeled donor query is
    accepted only as a complete case/sample surface, which prevents a target
    support subset from masquerading as source-OOF training evidence.
    """

    assert_development_context(context)
    if not isinstance(features, DisagreementFeatureSurface):
        raise ProtocolError("Exact regret construction requires a typed feature surface.")
    if features.surface_role == LABEL_FREE_INFERENCE_SURFACE_ROLE:
        raise ProtocolError("Inference surfaces cannot enter exact-regret construction.")
    label_rows = tuple(labels)
    if not label_rows or any(not isinstance(row, SourceOOFLabelRow) for row in label_rows):
        raise ProtocolError("Exact regret construction requires typed source-OOF labels.")
    label_by_key = {row.row_key: row for row in label_rows}
    if len(label_by_key) != len(label_rows):
        raise ProtocolError("Source-OOF labels contain duplicate sample identities.")
    if any(row.query_id == context.outer_target_id for row in label_rows):
        raise ProtocolError("Outer-target labels cannot enter disagreement-regret fitting.")
    context_hash = canonical_sha256(context.to_payload())
    if (
        features.development_context_hash != context_hash
        or features.dataset_family != context.dataset_family
        or features.outer_target_id != context.outer_target_id
        or features.family != "R"
    ):
        raise ProtocolError("Feature surface drifted from its development context.")
    feature_queries = set(features.query_ids)
    label_queries = {row.query_id for row in label_rows}
    if not label_queries.issubset(feature_queries):
        raise ProtocolError("Source-OOF labels contain a query absent from features.")
    if (
        context.scope
        in (
            DevelopmentScope.AUTHORIZED_SOURCE_OOF,
            DevelopmentScope.AUTHORIZED_POSTHOC_SOURCE_OOF,
        )
        and label_queries != set(context.authorized_query_ids)
    ):
        raise ProtocolError("Authorized source-OOF donor-query allowlist drifted.")
    expected_label_keys = {
        key for key in features.sample_keys if key[0] in label_queries
    }
    if set(label_by_key) != expected_label_keys:
        raise ProtocolError(
            "Source-OOF labels must exactly match complete case sample identities."
        )
    if context.scope in (
        DevelopmentScope.AUTHORIZED_SOURCE_OOF,
        DevelopmentScope.AUTHORIZED_POSTHOC_SOURCE_OOF,
    ):
        observed_sample_hash = canonical_sha256(
            {"sample_keys": [list(key) for key in sorted(label_by_key)]}
        )
        if observed_sample_hash != context.authorized_sample_keys_hash:
            raise ProtocolError("Authorized source-OOF sample allowlist drifted.")

    label_keys_by_case: dict[tuple[str, str], set[tuple[str, str, str]]] = defaultdict(set)
    for row in label_rows:
        label_keys_by_case[(row.query_id, row.case_id)].add(row.row_key)
    feature_rows_by_case: dict[tuple[str, str], list[object]] = defaultdict(list)
    for row in features.rows:
        if row.query_id in label_queries:
            feature_rows_by_case[(row.query_id, row.case_id)].append(row)
    if set(label_keys_by_case) != set(feature_rows_by_case):
        raise ProtocolError("Labeled source queries must cover every complete feature case.")
    for case_key, case_features in feature_rows_by_case.items():
        counts = {row.sample_count for row in case_features}
        if len(counts) != 1 or next(iter(counts)) != len(label_keys_by_case[case_key]):
            raise ProtocolError("Source-OOF label rows do not cover a complete case.")

    class_counts: dict[str, tuple[int, int]] = {}
    for query in sorted(label_queries):
        query_labels = [row.label for row in label_rows if row.query_id == query]
        negative = sum(value == 0 for value in query_labels)
        positive = sum(value == 1 for value in query_labels)
        if negative <= 0 or positive <= 0:
            raise ProtocolError("Exact BACC response queries must retain both classes.")
        class_counts[query] = (negative, positive)

    disagreement_by_case_action: dict[
        tuple[str, str, str], list[object]
    ] = defaultdict(list)
    for row in features.disagreements:
        if row.query_id in label_queries:
            if (row.query_id, row.case_id, row.sample_id) not in label_by_key:
                raise ProtocolError("A disagreement row lacks its source-OOF label.")
            disagreement_by_case_action[(row.query_id, row.case_id, row.action_id)].append(row)

    provisional: list[tuple[object, float]] = []
    for feature in features.rows:
        if feature.query_id not in label_queries:
            continue
        negative, positive = class_counts[feature.query_id]
        disagreements = disagreement_by_case_action[
            (feature.query_id, feature.case_id, feature.action_id)
        ]
        if len(disagreements) != feature.disagreement_count:
            raise ProtocolError("Sparse disagreement counts drifted from case features.")
        gain = 0.0
        for disagreement in disagreements:
            label = label_by_key[
                (disagreement.query_id, disagreement.case_id, disagreement.sample_id)
            ].label
            signed_weight = label / positive - (1 - label) / negative
            gain += 0.5 * disagreement.flip_direction * signed_weight
        provisional.append((feature, gain))

    best_by_case: dict[tuple[str, str], float] = defaultdict(float)
    for feature, gain in provisional:
        key = (feature.query_id, feature.case_id)
        best_by_case[key] = max(best_by_case[key], gain, 0.0)
    response_rows = tuple(
        sorted(
            (
                CaseActionResponseRow(
                    query_id=feature.query_id,
                    case_id=feature.case_id,
                    action_id=feature.action_id,
                    source_id=feature.source_id,
                    exact_bacc_gain_vs_control=gain,
                    exact_regret_from_case_best=(
                        best_by_case[(feature.query_id, feature.case_id)] - gain
                    ),
                    disagreement_count=feature.disagreement_count,
                    positive_class_count=class_counts[feature.query_id][1],
                    negative_class_count=class_counts[feature.query_id][0],
                )
                for feature, gain in provisional
            ),
            key=lambda row: row.row_key,
        )
    )
    label_surface_hash = canonical_sha256(
        {
            "schema_version": "midogpp_disagreement_regret_source_oof_labels_v1",
            "context": context.to_payload(),
            "labels": [
                {
                    "query_id": row.query_id,
                    "case_id": row.case_id,
                    "sample_id": row.sample_id,
                    "label": row.label,
                    "role": row.role,
                }
                for row in sorted(label_rows, key=lambda row: row.row_key)
            ],
        }
    )
    return ExactRegretSurface(
        rows=response_rows,
        feature_surface_hash=features.surface_hash,
        label_surface_hash=label_surface_hash,
        prediction_seal_hash=features.prediction_seal_hash,
        development_context_hash=context_hash,
    )


__all__ = ("build_exact_regret_surface",)
