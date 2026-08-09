"""Public ensemble-core facade and spawn-safe worker boundary.

This module is the one policy-layer import boundary for candidate-level
ensemble fitting.  It also strips immutable mapping proxies from the large
endpoint object before multiprocessing and reconstructs the typed contracts in
the worker.  No legacy per-seed utility DTO is representable here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256
from ..utility_aligned import (
    ENSEMBLE_SEED_KEYS,
    SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
    SUPPORT_ACTION_PROBABILITY_SHIFT_SCHEMA,
    SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
    SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS,
    EnsembleUtilitySurface,
    SupportActionProbabilityShift,
    aggregate_candidate_seed_features,
    build_ensemble_feature_surface,
    build_ensemble_utility_policy,
    build_target_ensemble_feature_surfaces,
    cyclically_permute_target_scalar,
    derive_label_free_global_source_control,
    evaluate_ensemble_cardinality_transfer,
    fit_ensemble_utility_model,
    validate_ensemble_utility_responses,
)


@dataclass(frozen=True)
class EnsembleEndpointWorkerPayload:
    """Only standard-pickle-safe endpoint and source-inner shift dictionaries."""

    utility_rows: tuple[dict[str, object], ...]
    support_shift_rows: tuple[dict[str, object], ...]


def make_endpoint_worker_payload(
    endpoint: object,
    *,
    outer_target_id: str,
) -> EnsembleEndpointWorkerPayload:
    try:
        utility_rows = tuple(
            dict(row.to_payload()) for row in endpoint.utility_surface.rows
        )
        shifts = endpoint.support_shifts_by_outer[outer_target_id]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ProtocolError("Ensemble endpoint cannot cross the worker boundary.") from exc
    support_shift_rows = tuple(
        {
            "row_key": list(key),
            "shift": dict(shifts[key].to_payload()),
        }
        for key in sorted(shifts)
    )
    return EnsembleEndpointWorkerPayload(
        utility_rows=utility_rows,
        support_shift_rows=support_shift_rows,
    )


def restore_endpoint_worker_payload(
    payload: EnsembleEndpointWorkerPayload,
    *,
    outer_target_id: str,
) -> tuple[
    EnsembleUtilitySurface,
    Mapping[tuple[str, str, str], SupportActionProbabilityShift],
]:
    if not isinstance(payload, EnsembleEndpointWorkerPayload):
        raise ProtocolError("Ensemble worker endpoint payload is untyped.")
    utility = validate_ensemble_utility_responses(payload.utility_rows)
    shifts: dict[tuple[str, str, str], SupportActionProbabilityShift] = {}
    for wrapped in payload.support_shift_rows:
        if set(wrapped) != {"row_key", "shift"}:
            raise ProtocolError("Ensemble worker support-shift wrapper drifted.")
        raw_key = wrapped["row_key"]
        raw_shift = wrapped["shift"]
        if (
            not isinstance(raw_key, Sequence)
            or isinstance(raw_key, (str, bytes))
            or len(raw_key) != 3
            or not isinstance(raw_shift, Mapping)
        ):
            raise ProtocolError("Ensemble worker support-shift wrapper is malformed.")
        key = tuple(str(value) for value in raw_key)
        if key[0] != outer_target_id or key in shifts:
            raise ProtocolError("Ensemble worker support-shift key drifted.")
        shifts[key] = _support_shift_from_payload(raw_shift)
    expected = {
        row.row_key
        for row in utility.rows_for_outer_target(outer_target_id)
    }
    if set(shifts) != expected:
        raise ProtocolError("Ensemble worker support-shift coverage drifted.")
    return utility, MappingProxyType(shifts)


def _support_shift_from_payload(
    raw: Mapping[str, object],
) -> SupportActionProbabilityShift:
    expected = {
        "schema_version", "row_identity_hash", "seed_pair_count", "seed_keys",
        "base_component_vector_hashes", "tail_component_vector_hashes",
        "per_seed_mean_absolute_shifts", "technical_seed_spread_semantics",
        "technical_seed_values_may_feed_model",
        "base_ensemble_probability_sha256",
        "tail_ensemble_probability_sha256",
        "ensemble_absolute_difference_sha256", "value", "seed_standard_deviation",
        "seed_minimum", "seed_maximum", "seed_range", "scalar_name",
        "scalar_semantics", "labels_used", "shift_hash",
    }
    if set(raw) != expected:
        raise ProtocolError("Ensemble worker support-shift schema drifted.")
    try:
        seed_keys = tuple(tuple(int(value) for value in key) for key in raw["seed_keys"])
        values = tuple(float(value) for value in raw["per_seed_mean_absolute_shifts"])
        base_hashes = tuple(str(value) for value in raw["base_component_vector_hashes"])
        tail_hashes = tuple(str(value) for value in raw["tail_component_vector_hashes"])
        unhashed = {key: raw[key] for key in raw if key != "shift_hash"}
        shift = SupportActionProbabilityShift(
            row_identity_hash=str(raw["row_identity_hash"]),
            seed_keys=seed_keys,
            base_component_vector_hashes=base_hashes,
            tail_component_vector_hashes=tail_hashes,
            per_seed_mean_absolute_shifts=values,
            base_ensemble_probability_hash=str(
                raw["base_ensemble_probability_sha256"]
            ),
            tail_ensemble_probability_hash=str(
                raw["tail_ensemble_probability_sha256"]
            ),
            ensemble_absolute_difference_hash=str(
                raw["ensemble_absolute_difference_sha256"]
            ),
            value=float(raw["value"]),
            seed_standard_deviation=float(raw["seed_standard_deviation"]),
            seed_minimum=float(raw["seed_minimum"]),
            seed_maximum=float(raw["seed_maximum"]),
            seed_range=float(raw["seed_range"]),
            shift_hash=str(raw["shift_hash"]),
            scalar_name=str(raw["scalar_name"]),
            scalar_semantics=str(raw["scalar_semantics"]),
        )
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        raise ProtocolError("Ensemble worker support-shift values are malformed.") from exc
    if (
        raw["schema_version"]
        != SUPPORT_ACTION_PROBABILITY_SHIFT_SCHEMA
        or raw["seed_pair_count"] != 9
        or seed_keys != ENSEMBLE_SEED_KEYS
        or len(values) != 9
        or len(base_hashes) != 9
        or len(tail_hashes) != 9
        or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values)
        or raw["scalar_name"] != SUPPORT_ACTION_PROBABILITY_SHIFT_NAME
        or raw["scalar_semantics"] != SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS
        or raw["technical_seed_spread_semantics"]
        != SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS
        or raw["technical_seed_values_may_feed_model"] is not False
        or raw["labels_used"] is not False
        or canonical_sha256(unhashed) != raw["shift_hash"]
        or shift.to_payload() != dict(raw)
    ):
        raise ProtocolError("Ensemble worker support-shift contract drifted.")
    return shift


__all__ = (
    "EnsembleEndpointWorkerPayload",
    "aggregate_candidate_seed_features",
    "build_ensemble_feature_surface",
    "build_ensemble_utility_policy",
    "build_target_ensemble_feature_surfaces",
    "cyclically_permute_target_scalar",
    "derive_label_free_global_source_control",
    "evaluate_ensemble_cardinality_transfer",
    "fit_ensemble_utility_model",
    "make_endpoint_worker_payload",
    "restore_endpoint_worker_payload",
)
