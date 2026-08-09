"""Per-case action-shift aggregation and its closed provenance lock."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ..exact_tail_utility_surface.config import CLASSIFIER
from ..residual_topup.hashing import canonical_sha256
from ..utility_aligned.ensemble_contracts import (
    ENSEMBLE_SEED_KEYS,
    SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
    SupportActionProbabilityShift,
)
from .action_shift_aggregation import (
    build_action_shift_rows,
    build_task_action_shift_rows,
)
from .action_probe_contracts import (
    ACTION_SHIFT_AGGREGATE_SCALAR_SEMANTICS,
    ACTION_SHIFT_LOCK_SCHEMA,
    ACTION_SHIFT_ROW_SCALAR_SEMANTICS,
    ACTION_SHIFT_ROW_SCHEMA,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    ActionProbeRuntime,
    TargetSupportActionShiftRow,
    TargetSupportActionShiftSurface,
    action_geometry_payload,
    action_probe_topology_payload,
    workstation_action_probe_runtime,
)


ACTION_SHIFT_LOCK_KEYS = frozenset(
    {
        "schema_version",
        "scalar_name",
        "row_scalar_semantics",
        "aggregate_scalar_semantics",
        "seed_pair_count",
        "support_reservation_hash",
        "target_support_cache_binding_hash",
        "source_generation_lock_hash",
        "generated_cache_hash",
        "classifier_hash",
        "action_geometry_hash",
        "table_sha256",
        "ordered_row_hashes_hash",
        "row_count",
        "case_ensemble_group_count",
        "row_key_grid_hash",
        "descriptive_seed_values_may_feed_model",
        "labels_used",
        "target_evaluation_rows_opened",
        "seeds_selected_by_support",
        "shift_lock_hash",
    }
)


def build_action_shift_lock(
    *,
    rows: Sequence[TargetSupportActionShiftRow],
    table_path: Path,
    support_reservation_hash: str,
    target_support_cache_binding_hash: str,
    source_generation_lock_hash: str,
    generated_cache_hash: str,
    runtime: ActionProbeRuntime,
) -> Mapping[str, object]:
    """Bind the canonical table and every label-free upstream dependency."""

    ordered = tuple(rows)
    _validate_canonical_rows(ordered)
    row_grid = [_row_grid_key(row) for row in ordered]
    unhashed: dict[str, object] = {
        "schema_version": ACTION_SHIFT_LOCK_SCHEMA,
        "scalar_name": SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
        "row_scalar_semantics": ACTION_SHIFT_ROW_SCALAR_SEMANTICS,
        "aggregate_scalar_semantics": ACTION_SHIFT_AGGREGATE_SCALAR_SEMANTICS,
        "seed_pair_count": len(TRAINING_SEEDS) * len(GENERATION_SEEDS),
        "support_reservation_hash": str(support_reservation_hash),
        "target_support_cache_binding_hash": str(
            target_support_cache_binding_hash
        ),
        "source_generation_lock_hash": str(source_generation_lock_hash),
        "generated_cache_hash": str(generated_cache_hash),
        "classifier_hash": canonical_sha256(CLASSIFIER.to_payload()),
        "action_geometry_hash": _action_geometry_hash(runtime),
        "table_sha256": _sha256_file(table_path),
        "ordered_row_hashes_hash": canonical_sha256(
            [row.row_hash for row in ordered]
        ),
        "row_count": len(ordered),
        "case_ensemble_group_count": len(ordered) // len(ENSEMBLE_SEED_KEYS),
        "row_key_grid_hash": canonical_sha256(row_grid),
        "descriptive_seed_values_may_feed_model": False,
        "labels_used": False,
        "target_evaluation_rows_opened": False,
        "seeds_selected_by_support": False,
    }
    return {**unhashed, "shift_lock_hash": canonical_sha256(unhashed)}


def validate_action_shift_surface(
    *,
    rows: Sequence[TargetSupportActionShiftRow],
    lock: Mapping[str, object],
    table_path: Path,
) -> TargetSupportActionShiftSurface:
    """Reconstruct the row and lock hashes without trusting producer state."""

    ordered = tuple(rows)
    _validate_canonical_rows(ordered)
    unhashed = {key: value for key, value in lock.items() if key != "shift_lock_hash"}
    if (
        set(lock) != ACTION_SHIFT_LOCK_KEYS
        or lock.get("schema_version") != ACTION_SHIFT_LOCK_SCHEMA
        or lock.get("scalar_name") != SUPPORT_ACTION_PROBABILITY_SHIFT_NAME
        or lock.get("row_scalar_semantics")
        != ACTION_SHIFT_ROW_SCALAR_SEMANTICS
        or lock.get("aggregate_scalar_semantics")
        != ACTION_SHIFT_AGGREGATE_SCALAR_SEMANTICS
        or lock.get("seed_pair_count") != 9
        or lock.get("action_geometry_hash")
        != _action_geometry_hash(workstation_action_probe_runtime())
        or lock.get("row_count") != len(ordered)
        or lock.get("case_ensemble_group_count")
        != len(ordered) // len(ENSEMBLE_SEED_KEYS)
        or lock.get("table_sha256") != _sha256_file(table_path)
        or lock.get("ordered_row_hashes_hash")
        != canonical_sha256([row.row_hash for row in ordered])
        or lock.get("row_key_grid_hash")
        != canonical_sha256([_row_grid_key(row) for row in ordered])
        or lock.get("labels_used") is not False
        or lock.get("target_evaluation_rows_opened") is not False
        or lock.get("seeds_selected_by_support") is not False
        or lock.get("descriptive_seed_values_may_feed_model") is not False
        or lock.get("shift_lock_hash") != canonical_sha256(unhashed)
        or any(
            not _is_sha256(lock.get(key))
            for key in (
                "support_reservation_hash",
                "target_support_cache_binding_hash",
                "source_generation_lock_hash",
                "generated_cache_hash",
                "classifier_hash",
                "action_geometry_hash",
                "table_sha256",
                "ordered_row_hashes_hash",
                "row_key_grid_hash",
                "shift_lock_hash",
            )
        )
    ):
        raise ProtocolError("Target-support action-shift lock drifted.")
    return TargetSupportActionShiftSurface(rows=ordered, lock_payload=lock)


def action_shift_row_from_payload(
    raw: Mapping[str, object],
) -> TargetSupportActionShiftRow:
    expected = {
        "schema_version",
        "outer_target_id",
        "query_id",
        "candidate_source",
        "training_seed",
        "generation_seed",
        "case_id",
        "support_partition_hash",
        "case_row_identity_hash",
        "support_row_count",
        "base_probability_sha256",
        "tail_probability_sha256",
        "base_component_vector_hash",
        "tail_component_vector_hash",
        "descriptive_seed_mean_absolute_positive_probability_shift",
        "case_ensemble_mean_absolute_positive_probability_shift",
        "case_base_ensemble_probability_sha256",
        "case_tail_ensemble_probability_sha256",
        "case_ensemble_absolute_difference_sha256",
        "case_ensemble_shift_hash",
        "scalar_name",
        "scalar_semantics",
        "descriptive_seed_value_may_feed_model",
        "labels_used",
        "row_hash",
    }
    if (
        set(raw) != expected
        or raw.get("schema_version") != ACTION_SHIFT_ROW_SCHEMA
        or raw.get("descriptive_seed_value_may_feed_model") not in (False, "False")
    ):
        raise ProtocolError("Target-support action-shift row schema drifted.")
    try:
        labels_used = _canonical_bool(raw["labels_used"])
        return TargetSupportActionShiftRow(
            outer_target_id=str(raw["outer_target_id"]),
            query_id=str(raw["query_id"]),
            candidate_source=str(raw["candidate_source"]),
            training_seed=int(raw["training_seed"]),
            generation_seed=int(raw["generation_seed"]),
            case_id=str(raw["case_id"]),
            support_partition_hash=str(raw["support_partition_hash"]),
            case_row_identity_hash=str(raw["case_row_identity_hash"]),
            support_row_count=int(raw["support_row_count"]),
            base_probability_sha256=str(raw["base_probability_sha256"]),
            tail_probability_sha256=str(raw["tail_probability_sha256"]),
            base_component_vector_hash=str(raw["base_component_vector_hash"]),
            tail_component_vector_hash=str(raw["tail_component_vector_hash"]),
            descriptive_seed_mean_absolute_positive_probability_shift=float(
                raw[
                    "descriptive_seed_mean_absolute_positive_probability_shift"
                ]
            ),
            case_ensemble_mean_absolute_positive_probability_shift=float(
                raw[
                    "case_ensemble_mean_absolute_positive_probability_shift"
                ]
            ),
            case_base_ensemble_probability_sha256=str(
                raw["case_base_ensemble_probability_sha256"]
            ),
            case_tail_ensemble_probability_sha256=str(
                raw["case_tail_ensemble_probability_sha256"]
            ),
            case_ensemble_absolute_difference_sha256=str(
                raw["case_ensemble_absolute_difference_sha256"]
            ),
            case_ensemble_shift_hash=str(raw["case_ensemble_shift_hash"]),
            scalar_name=str(raw["scalar_name"]),
            scalar_semantics=str(raw["scalar_semantics"]),
            labels_used=labels_used,
            row_hash=str(raw["row_hash"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Target-support action-shift row is malformed.") from exc


def _validate_canonical_rows(
    rows: tuple[TargetSupportActionShiftRow, ...],
) -> None:
    if (
        not rows
        or rows != tuple(sorted(rows, key=lambda row: row.row_key))
        or len({row.row_key for row in rows}) != len(rows)
        or any(row.labels_used is not False for row in rows)
    ):
        raise ProtocolError("Target-support action-shift rows are not canonical.")
    grouped: dict[
        tuple[str, str, str], list[TargetSupportActionShiftRow]
    ] = {}
    for row in rows:
        grouped.setdefault(
            (row.outer_target_id, row.candidate_source, row.case_id), []
        ).append(row)
    for group in grouped.values():
        ordered = tuple(group)
        if tuple(
            (row.training_seed, row.generation_seed) for row in ordered
        ) != ENSEMBLE_SEED_KEYS:
            raise ProtocolError(
                "Target-support case aggregate requires canonical exact-nine rows."
            )
        bindings = {
            (
                row.support_partition_hash,
                row.case_row_identity_hash,
                row.support_row_count,
                row.case_ensemble_mean_absolute_positive_probability_shift,
                row.case_base_ensemble_probability_sha256,
                row.case_tail_ensemble_probability_sha256,
                row.case_ensemble_absolute_difference_sha256,
                row.case_ensemble_shift_hash,
            )
            for row in ordered
        }
        if len(bindings) != 1:
            raise ProtocolError(
                "Target-support seed rows disagree on their case ensemble aggregate."
            )
        ensemble_value = (
            ordered[0].case_ensemble_mean_absolute_positive_probability_shift
        )
        descriptive_mean = float(
            np.mean(
                np.asarray(
                    [
                        row.descriptive_seed_mean_absolute_positive_probability_shift
                        for row in ordered
                    ],
                    dtype=np.float64,
                ),
                dtype=np.float64,
            )
        )
        if ensemble_value > descriptive_mean + 1.0e-12:
            raise ProtocolError(
                "Target-support ensemble-first shift exceeds its technical-seed bound."
            )
        descriptive = np.asarray(
            [
                row.descriptive_seed_mean_absolute_positive_probability_shift
                for row in ordered
            ],
            dtype=np.float64,
        )
        SupportActionProbabilityShift(
            row_identity_hash=ordered[0].case_row_identity_hash,
            seed_keys=ENSEMBLE_SEED_KEYS,
            base_component_vector_hashes=tuple(
                row.base_component_vector_hash for row in ordered
            ),
            tail_component_vector_hashes=tuple(
                row.tail_component_vector_hash for row in ordered
            ),
            per_seed_mean_absolute_shifts=tuple(
                float(value) for value in descriptive
            ),
            base_ensemble_probability_hash=(
                ordered[0].case_base_ensemble_probability_sha256
            ),
            tail_ensemble_probability_hash=(
                ordered[0].case_tail_ensemble_probability_sha256
            ),
            ensemble_absolute_difference_hash=(
                ordered[0].case_ensemble_absolute_difference_sha256
            ),
            value=ensemble_value,
            seed_standard_deviation=float(
                np.std(descriptive, ddof=0, dtype=np.float64)
            ),
            seed_minimum=float(np.min(descriptive)),
            seed_maximum=float(np.max(descriptive)),
            seed_range=float(np.max(descriptive) - np.min(descriptive)),
            shift_hash=ordered[0].case_ensemble_shift_hash,
        )


def _row_grid_key(row: TargetSupportActionShiftRow) -> list[object]:
    return [
        row.outer_target_id,
        row.query_id,
        row.candidate_source,
        row.training_seed,
        row.generation_seed,
        row.case_id,
    ]


def _canonical_bool(value: object) -> bool:
    if value is False or value == "False":
        return False
    if value is True or value == "True":
        return True
    raise ValueError("Expected canonical boolean.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    rendered = str(value or "")
    return len(rendered) == 64 and all(
        character in "0123456789abcdef" for character in rendered
    )


def _centers() -> tuple[str, ...]:
    from ..utility_aligned_identities import CENTERS

    return CENTERS


def _action_geometry_hash(runtime: ActionProbeRuntime) -> str:
    return canonical_sha256(
        {
            "schema_version": "midogpp_target_support_action_geometry_v1",
            "geometry_by_target": {
                target: action_geometry_payload(
                    tuple(source for source in _centers() if source != target)
                )
                for target in _centers()
            },
            **action_probe_topology_payload(runtime),
        }
    )


__all__ = (
    "ACTION_SHIFT_LOCK_KEYS",
    "action_shift_row_from_payload",
    "build_action_shift_lock",
    "build_action_shift_rows",
    "build_task_action_shift_rows",
    "validate_action_shift_surface",
)
