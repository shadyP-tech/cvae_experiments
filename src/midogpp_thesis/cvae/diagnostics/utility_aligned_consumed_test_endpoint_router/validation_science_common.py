"""Strict, lossless persistence decoders shared by science validators."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .artifact_io import sha256_file


def decode_hashed_row(
    raw: Mapping[str, str], fields: set[str], hash_field: str, role: str
) -> dict[str, object]:
    require_fields(raw, fields, role)
    payload: dict[str, object] = {}
    json_fields = {"model_hashes_by_role", "geometry", "topup_counts_by_source"}
    bool_fields = {
        "strict_H_q_e_exclusion",
        "same_outer_H_evaluation_labels_used_for_fit", "support_labels_used_for_fit",
        "target_features_used_for_fit", "seed_rows_are_independent_observations",
        "exact_B_fallback", "source_inner_transfer_authorized", "target_static",
        "case_router_used", "support_labels_used",
        "same_outer_H_evaluation_labels_used", "target_utility_used",
        "may_update_from_terminal_scores", "diagnostic_only", "diagnostic_control",
        "labels_used_to_build", "terminal_scores_used_to_build",
        "same_outer_H_evaluation_labels_opened_after_plan_and_global_seal",
        "terminal_scores_may_update_plan", "technical_seed_cells_are_independent_units",
        "consumed_test_diagnostic_only",
    }
    int_fields = {
        "training_response_count", "realized_total_per_class",
        "evaluation_case_count", "evaluation_row_count",
        "observed_class_0_row_count", "observed_class_1_row_count",
    }
    float_fields = {"balanced_accuracy"}
    nullable_fields = {
        "executed_routed_source", "fallback_reason", "selected_source",
        "core_action_hash", "policy_hash",
    }
    for name, value in raw.items():
        if name in json_fields:
            payload[name] = json_value(value, name, (dict, list))
        elif name in bool_fields:
            payload[name] = boolean(value, name)
        elif name in int_fields:
            payload[name] = integer(value, name)
        elif name in float_fields:
            payload[name] = floating(value, name)
        elif name in nullable_fields:
            payload[name] = nullable_text(value)
        else:
            payload[name] = value
    hash_payload = payload
    if role == "policy":
        if payload.get("selected_action_role") != payload.get("selected_action_id"):
            raise ProtocolError("Policy selected-action compatibility alias drifted.")
        hash_payload = {
            key: value for key, value in payload.items()
            if key != "selected_action_role"
        }
    require_payload_hash(hash_payload, hash_field, role)
    return payload


def validate_prediction_seal(
    root: Path,
    payload: Mapping[str, object],
    *,
    hash_field: str,
    expected_arrays_member: str,
    expected_index_member: str,
    store: object,
) -> None:
    require_payload_hash(payload, hash_field, "prediction seal")
    if (
        payload.get("arrays_member") != expected_arrays_member
        or payload.get("index_member") != expected_index_member
        or payload.get("arrays_sha256") != sha256_file(root / expected_arrays_member)
        or payload.get("index_sha256") != sha256_file(root / expected_index_member)
        or payload.get("prediction_store_hash") != getattr(store, "store_hash", None)
        or payload.get("source_stream_lock_hash")
        != getattr(store, "source_stream_lock_hash", None)
        or payload.get("partition_lock_hash")
        != getattr(store, "partition_lock_hash", None)
        or payload.get("cache_binding_hash")
        != getattr(store, "cache_binding_hash", None)
        or payload.get("action_library_hash")
        != getattr(store, "action_library_hash", None)
        or payload.get("labels_stored") is not False
        or payload.get("storage_dtype") != "float32"
        or payload.get("scientific_reductions_dtype") != "float64"
    ):
        raise ProtocolError("Prediction seal/store reconstruction drifted.")


def decode_inference_row(
    raw: Mapping[str, str], *, int_fields: set[str], float_fields: set[str],
    bool_fields: set[str], json_fields: set[str] | None = None,
    nullable_fields: set[str] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for name, value in raw.items():
        if name in int_fields:
            payload[name] = integer(value, name)
        elif name in float_fields:
            payload[name] = floating(value, name)
        elif name in bool_fields:
            payload[name] = boolean(value, name)
        elif name in (json_fields or set()):
            payload[name] = json_value(value, name, list)
        elif name in (nullable_fields or set()):
            payload[name] = None if value == "" else (
                floating(value, name)
                if name == "predicted_gain_hxe_bacc_spearman" else value
            )
        else:
            payload[name] = value
    return payload


def require_payload_hash(
    payload: Mapping[str, object], field: str, role: str
) -> None:
    if field not in payload:
        raise ProtocolError(f"{role} lacks {field}.")
    unhashed = {key: value for key, value in payload.items() if key != field}
    if payload[field] != canonical_sha256(unhashed):
        raise ProtocolError(f"{role} {field} drifted.")


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise ProtocolError(f"Cannot read scientific CSV: {path}.") from exc


def require_fields(raw: Mapping[str, object], expected: set[str], role: str) -> None:
    if set(raw) != expected:
        raise ProtocolError(
            f"{role} CSV schema drifted: missing={sorted(expected - set(raw))}, "
            f"extra={sorted(set(raw) - expected)}."
        )


def integer(value: object, name: str) -> int:
    try:
        rendered = str(value)
        parsed = int(rendered)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(f"{name} is not an integer.") from exc
    if rendered != str(parsed):
        raise ProtocolError(f"{name} is not canonically encoded.")
    return parsed


def floating(value: object, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(f"{name} is not numeric.") from exc
    if not math.isfinite(parsed):
        raise ProtocolError(f"{name} is not finite.")
    return parsed


def boolean(value: object, name: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ProtocolError(f"{name} is not a canonical CSV boolean.")


def json_value(
    value: str, name: str, expected_type: type | tuple[type, ...]
) -> object:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"{name} is not canonical JSON.") from exc
    if not isinstance(parsed, expected_type):
        raise ProtocolError(f"{name} JSON type drifted.")
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    if value != canonical:
        raise ProtocolError(f"{name} JSON encoding drifted.")
    return parsed


def nullable_text(value: object) -> str | None:
    return None if value in {None, ""} else str(value)


def mapping(value: object, role: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{role} is not a mapping.")
    return value


def mapping_field(
    payload: Mapping[str, object], field: str
) -> Mapping[str, object]:
    return mapping(payload.get(field), field)


def nested_float_mapping(
    value: Mapping[str, object], role: str
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for outer, raw in value.items():
        nested = mapping(raw, role)
        output[str(outer)] = {
            str(key): float(item) for key, item in nested.items()
        }
    return output


__all__ = (
    "boolean", "decode_hashed_row", "decode_inference_row", "floating",
    "integer", "json_value", "mapping", "mapping_field",
    "nested_float_mapping", "nullable_text", "read_csv", "require_fields",
    "require_payload_hash", "validate_prediction_seal",
)
