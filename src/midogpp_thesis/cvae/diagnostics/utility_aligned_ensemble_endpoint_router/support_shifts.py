"""Durable ensemble-first support-action shift surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...routing.utility_aligned.ensemble_endpoint import support_action_probability_shift
from ...routing.utility_aligned.ensemble_endpoint_contracts import SupportActionProbabilityShift
from .artifact_io import atomic_json, persist_or_validate_csv, read_json, sha256_file
from .contracts import BASE_ACTION_ID, CENTERS, candidate_sources, h_x_e_action_id, inner_candidate_sources
from .development_seal import DevelopmentPredictionCapability, validate_global_development_seal
from .prediction_contracts import CombinedPredictionStore
from .target_prediction_execution import validate_target_probe_seal


SOURCE_INNER_SHIFT_TABLE_MEMBER = "tables/source_inner_support_action_shifts.csv"
SOURCE_INNER_SHIFT_LOCK_MEMBER = "manifests/source_inner_support_action_shift_lock.json"
TARGET_SHIFT_TABLE_MEMBER = "tables/target_support_action_shifts.csv"
TARGET_SHIFT_LOCK_MEMBER = "manifests/target_support_action_shift_lock.json"

SHIFT_COLUMNS = (
    "schema_version", "role", "outer_target_id", "query_id", "candidate_source",
    "row_identity_hash", "seed_keys_json", "base_component_vector_hashes_json",
    "tail_component_vector_hashes_json", "per_seed_mean_absolute_shifts_json",
    "base_ensemble_probability_sha256", "tail_ensemble_probability_sha256",
    "ensemble_absolute_difference_sha256", "value", "seed_standard_deviation",
    "seed_minimum", "seed_maximum", "seed_range", "scalar_name", "scalar_semantics",
    "technical_seed_values_may_feed_model", "labels_used", "evaluation_embeddings_used",
    "shift_hash",
)


@dataclass(frozen=True)
class SupportShiftSurface:
    role: str
    by_candidate: Mapping[tuple[str, str, str], SupportActionProbabilityShift]
    parent_prediction_seal_hash: str
    lock_hash: str

    def __post_init__(self) -> None:
        values = {tuple(map(str, key)): value for key, value in self.by_candidate.items()}
        expected_count = 504 if self.role == "source_inner" else 72 if self.role == "target" else -1
        if (
            len(values) != expected_count
            or len(set(values)) != len(values)
            or any(not isinstance(value, SupportActionProbabilityShift) for value in values.values())
        ):
            raise ProtocolError("Support-shift surface coverage drifted.")
        object.__setattr__(self, "by_candidate", MappingProxyType(values))


def build_and_persist_source_inner_support_shifts(
    root: Path, capability: DevelopmentPredictionCapability
) -> SupportShiftSurface:
    seal = validate_global_development_seal(capability)
    values: dict[tuple[str, str, str], SupportActionProbabilityShift] = {}
    for outer in CENTERS:
        for query in candidate_sources(outer):
            scope = f"{outer}::{query}"
            base = capability.store.vectors(scope, BASE_ACTION_ID, "support")
            for source in inner_candidate_sources(outer, query):
                values[(outer, query, source)] = support_action_probability_shift(
                    base, capability.store.vectors(scope, h_x_e_action_id(source), "support")
                )
    return _persist(
        root, role="source_inner", values=values,
        parent_prediction_seal_hash=str(seal["prediction_seal_hash"]),
        table_member=SOURCE_INNER_SHIFT_TABLE_MEMBER,
        lock_member=SOURCE_INNER_SHIFT_LOCK_MEMBER,
    )


def build_and_persist_target_support_shifts(
    root: Path, probe: CombinedPredictionStore, partitions: object
) -> SupportShiftSurface:
    probe_seal = validate_target_probe_seal(root, probe, partitions)
    values: dict[tuple[str, str, str], SupportActionProbabilityShift] = {}
    for target in CENTERS:
        base = probe.vectors(target, BASE_ACTION_ID, "support")
        for source in candidate_sources(target):
            values[(target, target, source)] = support_action_probability_shift(
                base, probe.vectors(target, h_x_e_action_id(source), "support")
            )
    return _persist(
        root, role="target", values=values,
        parent_prediction_seal_hash=str(probe_seal["probe_seal_hash"]),
        table_member=TARGET_SHIFT_TABLE_MEMBER,
        lock_member=TARGET_SHIFT_LOCK_MEMBER,
    )


def _persist(
    root: Path, *, role: str,
    values: Mapping[tuple[str, str, str], SupportActionProbabilityShift],
    parent_prediction_seal_hash: str, table_member: str, lock_member: str,
) -> SupportShiftSurface:
    ordered = tuple(sorted(values.items()))
    rows = [_row(role, key, value) for key, value in ordered]
    table_path = root / table_member
    persist_or_validate_csv(table_path, rows, SHIFT_COLUMNS)
    unhashed = {
        "schema_version": "midogpp_stage90_ensemble_endpoint_support_shift_lock_v1",
        "status": "SEALED_LABEL_FREE_ENSEMBLE_FIRST_SUPPORT_SHIFTS",
        "role": role, "parent_prediction_seal_hash": parent_prediction_seal_hash,
        "table_member": table_member, "table_sha256": sha256_file(table_path),
        "row_count": len(rows), "ordered_shift_hashes": [value.shift_hash for _, value in ordered],
        "scalar_name": ordered[0][1].scalar_name,
        "scalar_semantics": ordered[0][1].scalar_semantics,
        "exact_nine_ensemble_first": True,
        "per_seed_shifts_descriptive_only": True,
        "technical_seed_values_may_feed_model": False,
        "labels_used": False, "evaluation_embeddings_used": False,
    }
    payload = {**unhashed, "support_shift_lock_hash": stable_hash(unhashed)}
    path = root / lock_member
    if path.is_file() and read_json(path) != payload:
        raise ProtocolError("Persisted support-shift lock drifted.")
    if not path.is_file(): atomic_json(path, payload)
    return SupportShiftSurface(
        role=role, by_candidate=values,
        parent_prediction_seal_hash=parent_prediction_seal_hash,
        lock_hash=str(payload["support_shift_lock_hash"]),
    )


def _row(
    role: str, key: tuple[str, str, str], value: SupportActionProbabilityShift
) -> dict[str, object]:
    payload = value.to_payload()
    return {
        "schema_version": "midogpp_stage90_ensemble_endpoint_support_shift_row_v1",
        "role": role, "outer_target_id": key[0], "query_id": key[1],
        "candidate_source": key[2], "row_identity_hash": value.row_identity_hash,
        "seed_keys_json": json.dumps(payload["seed_keys"], separators=(",", ":")),
        "base_component_vector_hashes_json": json.dumps(payload["base_component_vector_hashes"], separators=(",", ":")),
        "tail_component_vector_hashes_json": json.dumps(payload["tail_component_vector_hashes"], separators=(",", ":")),
        "per_seed_mean_absolute_shifts_json": json.dumps(payload["per_seed_mean_absolute_shifts"], separators=(",", ":")),
        "base_ensemble_probability_sha256": value.base_ensemble_probability_hash,
        "tail_ensemble_probability_sha256": value.tail_ensemble_probability_hash,
        "ensemble_absolute_difference_sha256": value.ensemble_absolute_difference_hash,
        "value": value.value, "seed_standard_deviation": value.seed_standard_deviation,
        "seed_minimum": value.seed_minimum, "seed_maximum": value.seed_maximum,
        "seed_range": value.seed_range, "scalar_name": value.scalar_name,
        "scalar_semantics": value.scalar_semantics,
        "technical_seed_values_may_feed_model": False, "labels_used": False,
        "evaluation_embeddings_used": False, "shift_hash": value.shift_hash,
    }


__all__ = (
    "SHIFT_COLUMNS", "SOURCE_INNER_SHIFT_LOCK_MEMBER", "SOURCE_INNER_SHIFT_TABLE_MEMBER",
    "SupportShiftSurface", "TARGET_SHIFT_LOCK_MEMBER", "TARGET_SHIFT_TABLE_MEMBER",
    "build_and_persist_source_inner_support_shifts", "build_and_persist_target_support_shifts",
)
