"""Strict parsing and predeclared compact proxy-family designs."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import array_sha256, canonical_sha256
from .contracts import (
    ABSOLUTE_SHIFT_CONTROL,
    CENTERS,
    CONTROL_FAMILY_IDS,
    CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL,
    CYCLIC_PERMUTATION_SEED,
    CYCLIC_PERMUTATION_SHIFT,
    DIRECTIONAL_ACTION_COMPACT,
    EQUAL_UNION_NULL,
    EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT,
    EXPECTED_PROXY_FEATURE_ROW_COUNT,
    FAMILY_IDS,
    HYBRID_COMPACT,
    METADATA_ONLY_CONTROL,
    PROXY_FEATURE_SCHEMA,
    PROXY_UTILITY_SCHEMA,
    RICH_DISTRIBUTIONAL_COMPACT,
    SCREENING_FAMILY_IDS,
    ProxyFamilyDesign,
    ProxyFamilySpec,
    ProxyFeatureRow,
    ProxyFeatureSurface,
    ProxyUtilityRow,
    ProxyUtilitySurface,
    candidate_sources,
    family_specs_payload,
)


_FEATURE_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "outer_target_id",
        "query_id",
        "candidate_source",
        "candidate_source_count",
        "support_partition_hash",
        "support_case_count",
        "support_row_count",
        "seed_pair_count",
        "seed_feature_row_hashes",
        "base_support_vector_hashes",
        "tail_support_vector_hashes",
        "metadata_similarity",
        "absolute_ensemble_shift",
        "reconstruction_mean_within_query_z",
        "kl_mean_within_query_z",
        "log_distribution_mmd_within_query_z",
        "signed_margin_projection",
        "threshold_flip_rate",
        "mean_entropy_change",
        "development_prediction_seal_hash",
        "probability_role_used",
        "labels_used",
        "evaluation_probabilities_used_as_features",
        "technical_seed_rows_are_independent_observations",
        "proxy_feature_row_hash",
    }
)

_UTILITY_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "outer_target_id",
        "query_id",
        "candidate_source",
        "candidate_source_count",
        "support_partition_hash",
        "utility_delta",
        "response_hash",
        "response_unit",
        "technical_seed_rows_are_independent_observations",
        "support_eval_disjoint",
        "predictions_sealed_before_labels",
        "source_expert_frozen",
        "target_labels_used_for_routing",
    }
)


PROXY_FAMILY_SPECS: Mapping[str, ProxyFamilySpec] = MappingProxyType(
    {
        EQUAL_UNION_NULL: ProxyFamilySpec(
            family_id=EQUAL_UNION_NULL,
            predictor_names=(),
            family_role="negative_or_baseline_control",
        ),
        METADATA_ONLY_CONTROL: ProxyFamilySpec(
            family_id=METADATA_ONLY_CONTROL,
            predictor_names=("metadata_similarity",),
            family_role="negative_or_baseline_control",
        ),
        ABSOLUTE_SHIFT_CONTROL: ProxyFamilySpec(
            family_id=ABSOLUTE_SHIFT_CONTROL,
            predictor_names=("absolute_ensemble_shift",),
            family_role="negative_or_baseline_control",
        ),
        RICH_DISTRIBUTIONAL_COMPACT: ProxyFamilySpec(
            family_id=RICH_DISTRIBUTIONAL_COMPACT,
            predictor_names=(
                "reconstruction_mean_within_query_z",
                "kl_mean_within_query_z",
                "log_distribution_mmd_within_query_z",
            ),
            family_role="screening_candidate",
        ),
        DIRECTIONAL_ACTION_COMPACT: ProxyFamilySpec(
            family_id=DIRECTIONAL_ACTION_COMPACT,
            predictor_names=(
                "signed_margin_projection",
                "threshold_flip_rate",
                "mean_entropy_change",
            ),
            family_role="screening_candidate",
        ),
        HYBRID_COMPACT: ProxyFamilySpec(
            family_id=HYBRID_COMPACT,
            predictor_names=(
                "metadata_similarity",
                "log_distribution_mmd_within_query_z",
                "signed_margin_projection",
            ),
            family_role="screening_candidate",
        ),
        CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL: ProxyFamilySpec(
            family_id=CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL,
            predictor_names=(
                "cyclic_signed_margin_projection",
                "cyclic_threshold_flip_rate",
                "cyclic_mean_entropy_change",
            ),
            family_role="negative_or_baseline_control",
            cyclic_shift=CYCLIC_PERMUTATION_SHIFT,
        ),
    }
)


def proxy_feature_row_from_payload(payload: Mapping[str, object]) -> ProxyFeatureRow:
    """Parse the exact sealed label-free feature schema and verify its hash.

    Unknown fields fail closed.  In particular, raw labels, BACC, utility,
    oracle values, evaluation probabilities, and seed-level numeric spreads
    cannot be smuggled through this constructor.
    """

    if not isinstance(payload, Mapping) or set(payload) != _FEATURE_PAYLOAD_KEYS:
        raise ProtocolError("Proxy feature payload does not match the exact schema.")
    if payload.get("schema_version") != PROXY_FEATURE_SCHEMA:
        raise ProtocolError("Proxy feature payload schema drifted.")
    supplied_hash = payload.get("proxy_feature_row_hash")
    unhashed = {key: payload[key] for key in payload if key != "proxy_feature_row_hash"}
    if supplied_hash != canonical_sha256(unhashed):
        raise ProtocolError("Proxy feature payload hash drifted.")
    row = ProxyFeatureRow(
        outer_target_id=payload["outer_target_id"],  # type: ignore[arg-type]
        query_id=payload["query_id"],  # type: ignore[arg-type]
        candidate_source=payload["candidate_source"],  # type: ignore[arg-type]
        candidate_source_count=payload["candidate_source_count"],  # type: ignore[arg-type]
        support_partition_hash=payload["support_partition_hash"],  # type: ignore[arg-type]
        support_case_count=payload["support_case_count"],  # type: ignore[arg-type]
        support_row_count=payload["support_row_count"],  # type: ignore[arg-type]
        seed_pair_count=payload["seed_pair_count"],  # type: ignore[arg-type]
        seed_feature_row_hashes=payload["seed_feature_row_hashes"],  # type: ignore[arg-type]
        base_support_vector_hashes=payload["base_support_vector_hashes"],  # type: ignore[arg-type]
        tail_support_vector_hashes=payload["tail_support_vector_hashes"],  # type: ignore[arg-type]
        metadata_similarity=payload["metadata_similarity"],  # type: ignore[arg-type]
        absolute_ensemble_shift=payload["absolute_ensemble_shift"],  # type: ignore[arg-type]
        reconstruction_mean_within_query_z=payload[
            "reconstruction_mean_within_query_z"
        ],  # type: ignore[arg-type]
        kl_mean_within_query_z=payload["kl_mean_within_query_z"],  # type: ignore[arg-type]
        log_distribution_mmd_within_query_z=payload[
            "log_distribution_mmd_within_query_z"
        ],  # type: ignore[arg-type]
        signed_margin_projection=payload["signed_margin_projection"],  # type: ignore[arg-type]
        threshold_flip_rate=payload["threshold_flip_rate"],  # type: ignore[arg-type]
        mean_entropy_change=payload["mean_entropy_change"],  # type: ignore[arg-type]
        development_prediction_seal_hash=payload[
            "development_prediction_seal_hash"
        ],  # type: ignore[arg-type]
        probability_role_used=payload["probability_role_used"],  # type: ignore[arg-type]
        labels_used=payload["labels_used"],  # type: ignore[arg-type]
        evaluation_probabilities_used_as_features=payload[
            "evaluation_probabilities_used_as_features"
        ],  # type: ignore[arg-type]
        technical_seed_rows_are_independent_observations=payload[
            "technical_seed_rows_are_independent_observations"
        ],  # type: ignore[arg-type]
    )
    if row.proxy_feature_row_hash != supplied_hash:
        raise ProtocolError("Proxy feature row reconstruction hash drifted.")
    return row


def build_proxy_feature_surface(
    rows: Sequence[ProxyFeatureRow | Mapping[str, object]],
) -> ProxyFeatureSurface:
    """Validate the complete 504-row support-only candidate surface."""

    typed = tuple(
        row if isinstance(row, ProxyFeatureRow) else proxy_feature_row_from_payload(row)
        for row in rows
    )
    if len(typed) != EXPECTED_PROXY_FEATURE_ROW_COUNT:
        raise ProtocolError("Proxy feature surface requires exactly 504 candidate rows.")
    by_key = {row.row_key: row for row in typed}
    expected_keys = _expected_row_keys()
    if len(by_key) != len(typed) or set(by_key) != set(expected_keys):
        raise ProtocolError("Proxy feature H/q/e coverage drifted.")
    ordered = tuple(by_key[key] for key in expected_keys)
    seals = {row.development_prediction_seal_hash for row in ordered}
    if len(seals) != 1:
        raise ProtocolError("Proxy feature rows do not share one development seal.")
    for outer in CENTERS:
        for query in (value for value in CENTERS if value != outer):
            group = tuple(
                row
                for row in ordered
                if row.outer_target_id == outer and row.query_id == query
            )
            if (
                tuple(row.candidate_source for row in group)
                != candidate_sources(outer, query)
                or len({row.support_partition_hash for row in group}) != 1
                or len({row.support_case_count for row in group}) != 1
                or len({row.support_row_count for row in group}) != 1
                or len({row.base_support_vector_hashes for row in group}) != 1
            ):
                raise ProtocolError("Proxy feature within-query provenance drifted.")
    payload = {
        "schema_version": "midogpp_stage90_proxy_information_feature_surface_v1",
        "ordered_row_hashes": [row.proxy_feature_row_hash for row in ordered],
        "row_count": len(ordered),
        "response_unit": "candidate_H_q_e_after_exact_nine_support_collapse",
        "labels_used": False,
        "evaluation_probabilities_used_as_features": False,
        "technical_seed_rows_are_independent_observations": False,
    }
    return ProxyFeatureSurface(
        rows=ordered,
        row_keys=expected_keys,
        surface_hash=canonical_sha256(payload),
    )


def proxy_utility_row_from_object(value: object) -> ProxyUtilityRow:
    """Narrow a typed neutral ensemble response to the audit utility DTO."""

    if isinstance(value, ProxyUtilityRow):
        return value
    if isinstance(value, Mapping):
        if set(value) != _UTILITY_PAYLOAD_KEYS:
            raise ProtocolError("Proxy utility mapping does not match its exact schema.")
        if (
            value.get("schema_version") != PROXY_UTILITY_SCHEMA
            or value.get("response_unit")
            != "candidate_H_q_e_exact_nine_probability_ensemble"
            or value.get("technical_seed_rows_are_independent_observations") is not False
        ):
            raise ProtocolError("Proxy utility mapping semantics drifted.")
        return ProxyUtilityRow(
            outer_target_id=value["outer_target_id"],  # type: ignore[arg-type]
            query_id=value["query_id"],  # type: ignore[arg-type]
            candidate_source=value["candidate_source"],  # type: ignore[arg-type]
            candidate_source_count=value["candidate_source_count"],  # type: ignore[arg-type]
            support_partition_hash=value["support_partition_hash"],  # type: ignore[arg-type]
            utility_delta=value["utility_delta"],  # type: ignore[arg-type]
            response_hash=value["response_hash"],  # type: ignore[arg-type]
            support_eval_disjoint=value["support_eval_disjoint"],  # type: ignore[arg-type]
            predictions_sealed_before_labels=value[
                "predictions_sealed_before_labels"
            ],  # type: ignore[arg-type]
            source_expert_frozen=value["source_expert_frozen"],  # type: ignore[arg-type]
            target_labels_used_for_routing=value[
                "target_labels_used_for_routing"
            ],  # type: ignore[arg-type]
        )
    required_attributes = (
        "outer_target_id",
        "query_id",
        "candidate_source",
        "candidate_source_count",
        "support_partition_hash",
        "utility_delta",
        "row_hash",
        "support_eval_disjoint",
        "predictions_sealed_before_labels",
        "source_expert_frozen",
        "target_labels_used_for_routing",
    )
    if any(not hasattr(value, name) for name in required_attributes):
        raise ProtocolError("Proxy utility input is not a typed ensemble response.")
    return ProxyUtilityRow(
        outer_target_id=getattr(value, "outer_target_id"),
        query_id=getattr(value, "query_id"),
        candidate_source=getattr(value, "candidate_source"),
        candidate_source_count=getattr(value, "candidate_source_count"),
        support_partition_hash=getattr(value, "support_partition_hash"),
        utility_delta=getattr(value, "utility_delta"),
        response_hash=getattr(value, "row_hash"),
        support_eval_disjoint=getattr(value, "support_eval_disjoint"),
        predictions_sealed_before_labels=getattr(
            value, "predictions_sealed_before_labels"
        ),
        source_expert_frozen=getattr(value, "source_expert_frozen"),
        target_labels_used_for_routing=getattr(
            value, "target_labels_used_for_routing"
        ),
    )


def build_proxy_utility_surface(rows: Sequence[object]) -> ProxyUtilitySurface:
    typed = tuple(proxy_utility_row_from_object(row) for row in rows)
    if len(typed) != EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT:
        raise ProtocolError("Proxy utility surface requires exactly 504 responses.")
    by_key = {row.row_key: row for row in typed}
    expected_keys = _expected_row_keys()
    if len(by_key) != len(typed) or set(by_key) != set(expected_keys):
        raise ProtocolError("Proxy utility H/q/e coverage drifted.")
    ordered = tuple(by_key[key] for key in expected_keys)
    payload = {
        "schema_version": "midogpp_stage90_proxy_information_utility_surface_v1",
        "ordered_response_hashes": [row.response_hash for row in ordered],
        "ordered_utility_delta": [row.utility_delta for row in ordered],
        "response_count": len(ordered),
        "response_unit": "candidate_H_q_e_exact_nine_probability_ensemble",
        "technical_seed_rows_are_independent_observations": False,
    }
    return ProxyUtilitySurface(
        rows=ordered,
        row_keys=expected_keys,
        surface_hash=canonical_sha256(payload),
    )


def build_proxy_family_designs(
    surface: ProxyFeatureSurface,
) -> Mapping[str, ProxyFamilyDesign]:
    """Materialize all seven predeclared designs, each with at most 3 columns."""

    if not isinstance(surface, ProxyFeatureSurface):
        raise ProtocolError("Proxy family designs require a typed feature surface.")
    row_by_key = {row.row_key: row for row in surface.rows}
    designs: dict[str, ProxyFamilyDesign] = {}
    for family_id in FAMILY_IDS:
        spec = PROXY_FAMILY_SPECS[family_id]
        values: list[tuple[float, ...]] = []
        provenance: list[str] = []
        for key in surface.row_keys:
            row = row_by_key[key]
            source_row = row
            if family_id == CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL:
                sources = candidate_sources(row.outer_target_id, row.query_id)
                source_index = sources.index(row.candidate_source)
                donor = sources[(source_index + CYCLIC_PERMUTATION_SHIFT) % len(sources)]
                source_row = row_by_key[(row.outer_target_id, row.query_id, donor)]
                vector = (
                    source_row.signed_margin_projection,
                    source_row.threshold_flip_rate,
                    source_row.mean_entropy_change,
                )
            else:
                vector = tuple(float(getattr(row, name)) for name in spec.predictor_names)
            values.append(vector)
            provenance.append(source_row.proxy_feature_row_hash)
        matrix = np.asarray(values, dtype=np.float64).reshape(
            len(surface.rows), spec.predictor_count
        )
        payload = {
            "schema_version": "midogpp_stage90_proxy_information_family_design_v1",
            "family_spec": spec.to_payload(),
            "feature_surface_hash": surface.surface_hash,
            "row_keys": [list(key) for key in surface.row_keys],
            "source_row_hashes": provenance,
            "values_sha256": array_sha256(matrix),
            "cyclic_permutation_seed": (
                CYCLIC_PERMUTATION_SEED
                if family_id == CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL
                else None
            ),
            "label_free_candidate_list_transforms_may_be_transductive": True,
            "utility_or_evaluation_labels_used": False,
        }
        designs[family_id] = ProxyFamilyDesign(
            spec=spec,
            row_keys=surface.row_keys,
            values=matrix,
            source_row_hashes=tuple(provenance),
            design_hash=canonical_sha256(payload),
        )
    if tuple(designs) != FAMILY_IDS:
        raise ProtocolError("Proxy family design coverage drifted.")
    # Constructing this payload is itself a capacity/order assertion.
    family_specs_payload(PROXY_FAMILY_SPECS)
    return MappingProxyType(designs)


def _expected_row_keys() -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (outer, query, source)
        for outer in CENTERS
        for query in CENTERS
        if query != outer
        for source in candidate_sources(outer, query)
    )


__all__ = (
    "PROXY_FAMILY_SPECS",
    "build_proxy_family_designs",
    "build_proxy_feature_surface",
    "build_proxy_utility_surface",
    "proxy_feature_row_from_payload",
    "proxy_utility_row_from_object",
)
