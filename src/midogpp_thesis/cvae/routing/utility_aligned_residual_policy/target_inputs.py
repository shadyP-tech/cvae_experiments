"""Fresh target reservation and typed feature-surface admission."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256
from ..utility_aligned import (
    ENSEMBLE_SEED_KEYS,
    SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
    CaseBootstrapPlan,
    FeatureSurface,
    SupportActionProbabilityShift,
    TargetSupportActionShiftCase,
)
from ..utility_aligned_target_support_surface.action_probe_contracts import (
    ACTION_SHIFT_AGGREGATE_SCALAR_SEMANTICS,
    ACTION_SHIFT_LOCK_SCHEMA,
    ACTION_SHIFT_ROW_SCALAR_SEMANTICS,
    ACTION_SHIFT_ROW_SCHEMA,
    TargetSupportActionShiftRow,
)
from ..utility_aligned_target_support_surface.action_probe_surface import (
    action_shift_row_from_payload,
)
from ..utility_aligned.target_features import target_feature_production_from_payload
from ..utility_aligned_identities import (
    CENTERS,
    METADATA_PROFILE_SHA256,
    STAGE70_EXPERIMENT_ID,
    TARGET_SUPPORT_PRODUCER_EXPERIMENT_ID,
)
from .contracts import (
    EXPERIMENT_ID, MINIMUM_SUPPORT_CASE_COUNT, TARGET_RESERVATION_ARTIFACT_ID,
    TARGET_SUPPORT_PARENT_RESERVATION_ARTIFACT_ID, TARGET_SUPPORT_SCHEMA,
)
from .input_io import read_csv, read_json


TARGET_SUPPORT_LOCK_MEMBER = "manifests/target_support_surface_lock.json"
TARGET_ACTION_SHIFT_TABLE_MEMBER = "tables/target_support_action_shifts.csv"
TARGET_ACTION_SHIFT_LOCK_MEMBER = "manifests/target_support_action_shifts_lock.json"
TARGET_RESERVATION_MEMBER = "manifests/reservation.json"
TARGET_SUPPORT_RESERVATION_MEMBER = "manifests/reservation.json"


@dataclass(frozen=True)
class TargetFeatureSet:
    target_id: str
    plan: CaseBootstrapPlan
    point_surface: FeatureSurface
    bootstrap_surfaces: tuple[FeatureSurface, ...]


@dataclass(frozen=True)
class LoadedTargetInputs:
    surface_hash: str
    parent_artifact_id: str
    parent_hash: str
    reservation_hash: str
    evaluation_binding_hash: str
    support_case_ids_by_target: Mapping[str, tuple[str, ...]]
    evaluation_case_ids_by_target: Mapping[str, tuple[str, ...]]
    feature_sets: Mapping[str, TargetFeatureSet]
    action_shift_bindings: Mapping[str, object]
    action_shift_cases_by_target: Mapping[str, tuple[TargetSupportActionShiftCase, ...]]


def load_target_inputs(*, support_surface_root: Path, parent_reservation_root: Path, target_reservation_root: Path) -> LoadedTargetInputs:
    parent = _load_parent(parent_reservation_root)
    target = _load_target_reservation(target_reservation_root)
    parent_support = case_mapping(parent["support_case_ids_by_center"], "support")
    support = case_mapping(target["support_case_ids_by_center"], "support")
    evaluation = case_mapping(target["evaluation_case_ids_by_center"], "evaluation")
    if parent_support != support:
        raise ProtocolError("Target-support parent cases differ from Stage-70 support cases.")
    surface = _load_surface(support_surface_root, parent=parent, target=target)
    shifts, shift_cases = _load_action_shift_lock(
        support_surface_root,
        support_case_ids_by_target=support,
        feature_sets=surface["target_features"],
    )
    for surface_key, shift_key in (
        ("target_local_scalar_name", "target_local_scalar_name"),
        ("target_support_action_shift_lock_hash", "target_support_action_shift_lock_hash"),
        ("target_support_action_shift_table_sha256", "target_support_action_shift_table_sha256"),
        ("target_support_action_shift_row_hashes_hash", "target_support_action_shift_row_hashes_hash"),
        ("target_support_action_shift_row_count", "target_support_action_shift_row_count"),
    ):
        if surface[surface_key] != shifts[shift_key]:
            raise ProtocolError("Target-support surface/action-shift lock binding drifted.")
    return LoadedTargetInputs(
        surface_hash=str(surface["surface_hash"]), parent_artifact_id=str(parent["artifact_id"]),
        parent_hash=str(parent["reservation_hash"]), reservation_hash=str(target["reservation_hash"]),
        evaluation_binding_hash=str(target["target_evaluation_binding_hash"]),
        support_case_ids_by_target=support, evaluation_case_ids_by_target=evaluation,
        feature_sets=MappingProxyType(surface["target_features"]),
        action_shift_bindings=shifts,
        action_shift_cases_by_target=shift_cases,
    )


def _load_target_reservation(root: Path) -> Mapping[str, object]:
    raw = read_json(root / TARGET_RESERVATION_MEMBER)
    required = {"schema_version", "artifact_id", "status", "authorized_consumer_experiment_ids", "dataset_family", "fresh_unconsumed_surface", "support_evaluation_case_disjoint", "labels_opened", "consumed_test_used", "consumed_validation_used", "consumed_stage70_used", "consumed_stage90_used", "scoring_manifest_artifact_id", "scoring_manifest_sha256", "reservation_id", "target_evaluation_binding_hash", "support_case_ids_by_center", "evaluation_case_ids_by_center", "reservation_hash"}
    unhashed = {key: value for key, value in raw.items() if key != "reservation_hash"}
    if set(raw) != required or raw.get("schema_version") != "midogpp_utility_aligned_fresh_target_reservation_v1" or raw.get("artifact_id") != TARGET_RESERVATION_ARTIFACT_ID or raw.get("status") != "ACTIVE" or raw.get("authorized_consumer_experiment_ids") != [str(EXPERIMENT_ID), STAGE70_EXPERIMENT_ID] or raw.get("dataset_family") != "MIDOG++" or raw.get("fresh_unconsumed_surface") is not True or raw.get("support_evaluation_case_disjoint") is not True or raw.get("labels_opened") is not False or any(raw.get(key) is not False for key in ("consumed_test_used", "consumed_validation_used", "consumed_stage70_used", "consumed_stage90_used")) or raw.get("reservation_hash") != stable_hash(unhashed):
        raise ProtocolError("Fresh target reservation failed closed.")
    support = case_mapping(raw["support_case_ids_by_center"], "support")
    evaluation = case_mapping(raw["evaluation_case_ids_by_center"], "evaluation")
    if {value for values in support.values() for value in values} & {value for values in evaluation.values() for value in values}:
        raise ProtocolError("Fresh target support/evaluation cases overlap.")
    return MappingProxyType(dict(raw))


def _load_parent(root: Path) -> Mapping[str, object]:
    raw = read_json(root / TARGET_SUPPORT_RESERVATION_MEMBER)
    required = {"schema_version", "artifact_id", "status", "authorized_consumer_experiment_ids", "dataset_family", "fresh_unconsumed_surface", "labels_present", "target_evaluation_rows_present", "support_case_ids_by_center", "support_rows_by_center", "reservation_id", "reservation_hash"}
    unhashed = {key: value for key, value in raw.items() if key != "reservation_hash"}
    if set(raw) != required or raw.get("schema_version") != "midogpp_utility_aligned_target_support_reservation_v1" or raw.get("artifact_id") != TARGET_SUPPORT_PARENT_RESERVATION_ARTIFACT_ID or raw.get("status") != "ACTIVE" or raw.get("authorized_consumer_experiment_ids") != [TARGET_SUPPORT_PRODUCER_EXPERIMENT_ID, str(EXPERIMENT_ID)] or raw.get("dataset_family") != "MIDOG++" or raw.get("fresh_unconsumed_surface") is not True or raw.get("labels_present") is not False or raw.get("target_evaluation_rows_present") is not False or raw.get("reservation_hash") != canonical_sha256(unhashed):
        raise ProtocolError("Target-support parent reservation failed closed.")
    cases = case_mapping(raw["support_case_ids_by_center"], "support")
    row_map = raw.get("support_rows_by_center")
    if not isinstance(row_map, Mapping) or {str(key) for key in row_map} != set(CENTERS):
        raise ProtocolError("Target-support parent row coverage drifted.")
    seen_samples: set[str] = set(); seen_cache: set[tuple[str, int]] = set()
    for center in CENTERS:
        values = row_map[center]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
            raise ProtocolError("Target-support parent rows are absent.")
        observed_cases = set()
        for ordinal, value in enumerate(values):
            if not isinstance(value, Mapping) or set(value) != {"row_ordinal", "sample_id", "case_id", "center", "cache_shard_path", "cache_row_index"}:
                raise ProtocolError("Target-support parent row schema drifted.")
            sample = str(value["sample_id"]); case = str(value["case_id"]); shard = str(value["cache_shard_path"]); index = int(value["cache_row_index"])
            if int(value["row_ordinal"]) != ordinal or value.get("center") != center or not sample or not case or not shard or index < 0 or sample in seen_samples or (shard, index) in seen_cache:
                raise ProtocolError("Target-support parent row identity drifted.")
            seen_samples.add(sample); seen_cache.add((shard, index)); observed_cases.add(case)
        if observed_cases != set(cases[center]):
            raise ProtocolError("Target-support parent rows/cases drifted.")
    return MappingProxyType(dict(raw))


def _load_surface(root: Path, *, parent: Mapping[str, object], target: Mapping[str, object]) -> dict[str, object]:
    from ..utility_aligned_target_support_surface.production import validate_target_support_surface_bundle
    validate_target_support_surface_bundle(root)
    raw = read_json(root / TARGET_SUPPORT_LOCK_MEMBER)
    required = {
        "schema_version", "artifact_id", "claim_scope", "status",
        "target_support_parent_reservation_artifact_id",
        "target_support_parent_reservation_hash", "metadata_profile_sha256",
        "config_contract_hash", "input_artifact_ids",
        "target_support_cache_binding_hash", "target_support_cache_lock_hash",
        "expert_bank_lock_hash", "generation_lock_hash",
        "source_generation_lock_hash", "generated_cache_hash",
        "target_local_scalar_name", "target_support_action_shift_lock_hash",
        "target_support_action_shift_table_sha256",
        "target_support_action_shift_row_hashes_hash",
        "target_support_action_shift_row_count",
        "feature_reference_rows_per_class",
        "final_action_source_prefix_rows_per_class",
        "final_action_geometry_executed_by_this_artifact",
        "support_case_ids_by_target", "target_features", "labels_persisted",
        "target_evaluation_rows_opened", "surface_hash",
    }
    unhashed = {key: value for key, value in raw.items() if key != "surface_hash"}
    if set(raw) != required or raw.get("schema_version") != TARGET_SUPPORT_SCHEMA or raw.get("artifact_id") != "midogpp_utility_aligned_target_support_surface_v1" or raw.get("claim_scope") != "routing_compatibility_only" or raw.get("status") != "COMPLETE" or raw.get("target_support_parent_reservation_artifact_id") != TARGET_SUPPORT_PARENT_RESERVATION_ARTIFACT_ID or raw.get("target_support_parent_reservation_hash") != parent["reservation_hash"] or raw.get("support_case_ids_by_target") != parent["support_case_ids_by_center"] or raw.get("support_case_ids_by_target") != target["support_case_ids_by_center"] or raw.get("metadata_profile_sha256") != METADATA_PROFILE_SHA256 or raw.get("feature_reference_rows_per_class") != 270 or raw.get("final_action_source_prefix_rows_per_class") != 256 or raw.get("final_action_geometry_executed_by_this_artifact") is not True or raw.get("target_local_scalar_name") != SUPPORT_ACTION_PROBABILITY_SHIFT_NAME or not all(_sha256_like(raw.get(key)) for key in ("target_support_action_shift_lock_hash", "target_support_action_shift_table_sha256", "target_support_action_shift_row_hashes_hash")) or not isinstance(raw.get("target_support_action_shift_row_count"), int) or raw.get("target_support_action_shift_row_count", 0) <= 0 or raw.get("labels_persisted") is not False or raw.get("target_evaluation_rows_opened") is not False or raw.get("surface_hash") != canonical_sha256(unhashed):
        raise ProtocolError("Target-support surface binding drifted.")
    values = raw.get("target_features")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ProtocolError("Target-support feature cells are absent.")
    by_target = {}
    for value in values:
        feature_set = parse_target_feature_set(value)
        if feature_set.target_id in by_target:
            raise ProtocolError("Target-support feature target is duplicated.")
        by_target[feature_set.target_id] = feature_set
    if tuple(by_target) != CENTERS:
        raise ProtocolError("Target-support feature coverage drifted.")
    return {
        "surface_hash": raw["surface_hash"],
        "target_features": by_target,
        "target_local_scalar_name": raw["target_local_scalar_name"],
        "target_support_action_shift_lock_hash": raw[
            "target_support_action_shift_lock_hash"
        ],
        "target_support_action_shift_table_sha256": raw[
            "target_support_action_shift_table_sha256"
        ],
        "target_support_action_shift_row_hashes_hash": raw[
            "target_support_action_shift_row_hashes_hash"
        ],
        "target_support_action_shift_row_count": raw[
            "target_support_action_shift_row_count"
        ],
    }


def _load_action_shift_lock(
    root: Path,
    *,
    support_case_ids_by_target: Mapping[str, tuple[str, ...]],
    feature_sets: Mapping[str, TargetFeatureSet],
) -> tuple[
    Mapping[str, object], Mapping[str, tuple[TargetSupportActionShiftCase, ...]]
]:
    """Admit only the label-free per-case action-shift provenance contract."""

    table = root / TARGET_ACTION_SHIFT_TABLE_MEMBER
    lock = read_json(root / TARGET_ACTION_SHIFT_LOCK_MEMBER)
    if not table.is_file():
        raise ProtocolError("Target-support action-shift table is absent.")
    required = {
        "schema_version", "scalar_name", "row_scalar_semantics",
        "aggregate_scalar_semantics", "seed_pair_count",
        "support_reservation_hash", "target_support_cache_binding_hash",
        "source_generation_lock_hash", "generated_cache_hash", "classifier_hash",
        "action_geometry_hash", "table_sha256", "ordered_row_hashes_hash",
        "row_count", "case_ensemble_group_count", "row_key_grid_hash",
        "descriptive_seed_values_may_feed_model", "labels_used",
        "target_evaluation_rows_opened", "seeds_selected_by_support", "shift_lock_hash",
    }
    unhashed = {key: value for key, value in lock.items() if key != "shift_lock_hash"}
    if (
        set(lock) != required
        or lock.get("schema_version")
        != ACTION_SHIFT_LOCK_SCHEMA
        or lock.get("scalar_name")
        != SUPPORT_ACTION_PROBABILITY_SHIFT_NAME
        or not isinstance(lock.get("row_scalar_semantics"), str)
        or not str(lock["row_scalar_semantics"])
        or lock.get("row_scalar_semantics") != ACTION_SHIFT_ROW_SCALAR_SEMANTICS
        or lock.get("aggregate_scalar_semantics")
        != ACTION_SHIFT_AGGREGATE_SCALAR_SEMANTICS
        or lock.get("seed_pair_count") != 9
        or not isinstance(lock.get("row_count"), int)
        or int(lock["row_count"]) <= 0
        or not isinstance(lock.get("case_ensemble_group_count"), int)
        or int(lock["case_ensemble_group_count"]) <= 0
        or int(lock["row_count"])
        != int(lock["case_ensemble_group_count"]) * len(ENSEMBLE_SEED_KEYS)
        or lock.get("descriptive_seed_values_may_feed_model") is not False
        or lock.get("labels_used") is not False
        or lock.get("target_evaluation_rows_opened") is not False
        or lock.get("seeds_selected_by_support") is not False
        or lock.get("shift_lock_hash") != canonical_sha256(unhashed)
        or any(
            not _sha256_like(lock.get(key))
            for key in (
                "support_reservation_hash", "target_support_cache_binding_hash",
                "source_generation_lock_hash", "generated_cache_hash",
                "classifier_hash", "action_geometry_hash", "table_sha256",
                "row_key_grid_hash", "ordered_row_hashes_hash", "shift_lock_hash",
            )
        )
    ):
        raise ProtocolError("Target-support action-shift lock drifted.")
    import hashlib

    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    if digest != lock["table_sha256"]:
        raise ProtocolError("Target-support action-shift table bytes drifted.")
    rows = read_csv(table)
    expected_columns = {
        "schema_version", "outer_target_id", "query_id", "candidate_source",
        "training_seed", "generation_seed", "case_id", "support_partition_hash",
        "case_row_identity_hash", "support_row_count",
        "base_probability_sha256", "tail_probability_sha256",
        "base_component_vector_hash", "tail_component_vector_hash",
        "descriptive_seed_mean_absolute_positive_probability_shift",
        "case_ensemble_mean_absolute_positive_probability_shift",
        "case_base_ensemble_probability_sha256",
        "case_tail_ensemble_probability_sha256",
        "case_ensemble_absolute_difference_sha256", "case_ensemble_shift_hash",
        "scalar_name", "scalar_semantics",
        "descriptive_seed_value_may_feed_model", "labels_used", "row_hash",
    }
    if len(rows) != lock["row_count"] or any(set(row) != expected_columns for row in rows):
        raise ProtocolError("Target-support action-shift row coverage/schema drifted.")
    try:
        typed_rows = tuple(action_shift_row_from_payload(row) for row in rows)
        keys = [
            (
                row.outer_target_id, row.query_id, row.candidate_source,
                row.training_seed, row.generation_seed, row.case_id,
            )
            for row in typed_rows
        ]
        row_hashes = [row.row_hash for row in typed_rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Target-support action-shift rows are malformed.") from exc
    if (
        len(set(keys)) != len(keys)
        or canonical_sha256([list(key) for key in keys]) != lock["row_key_grid_hash"]
        or canonical_sha256(row_hashes) != lock["ordered_row_hashes_hash"]
        or any(
            row.to_payload()["schema_version"] != ACTION_SHIFT_ROW_SCHEMA
            or row.scalar_name != lock["scalar_name"]
            or row.scalar_semantics != lock["row_scalar_semantics"]
            or row.labels_used is not False
            for row in typed_rows
        )
    ):
        raise ProtocolError("Target-support action-shift content drifted.")
    cases = _group_target_shift_cases(
        typed_rows,
        support_case_ids_by_target=support_case_ids_by_target,
        feature_sets=feature_sets,
    )
    expected_group_count = sum(
        (len(CENTERS) - 1) * len(support_case_ids_by_target[target])
        for target in CENTERS
    )
    if lock["case_ensemble_group_count"] != expected_group_count:
        raise ProtocolError("Target-support action-shift case group count drifted.")
    return MappingProxyType(
        {
            "target_local_scalar_name": lock["scalar_name"],
            "target_local_scalar_semantics": lock["aggregate_scalar_semantics"],
            "target_local_scalar_row_semantics": lock["row_scalar_semantics"],
            "target_support_action_shift_lock_hash": lock["shift_lock_hash"],
            "target_support_action_shift_table_sha256": lock["table_sha256"],
            "target_support_action_shift_row_hashes_hash": lock[
                "ordered_row_hashes_hash"
            ],
            "target_support_action_shift_row_count": lock["row_count"],
            "target_support_action_shift_case_ensemble_group_count": lock[
                "case_ensemble_group_count"
            ],
            "target_support_action_shift_descriptive_seed_values_may_feed_model": (
                lock["descriptive_seed_values_may_feed_model"]
            ),
        }
    ), cases


def _sha256_like(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _group_target_shift_cases(
    rows: Sequence[TargetSupportActionShiftRow],
    *,
    support_case_ids_by_target: Mapping[str, tuple[str, ...]],
    feature_sets: Mapping[str, TargetFeatureSet],
) -> Mapping[str, tuple[TargetSupportActionShiftCase, ...]]:
    ordered_rows = tuple(rows)
    if ordered_rows != tuple(sorted(ordered_rows, key=lambda row: row.row_key)):
        raise ProtocolError("Target-support action-shift table order is noncanonical.")
    grouped: dict[tuple[str, str, str], list[TargetSupportActionShiftRow]] = {}
    for row in rows:
        grouped.setdefault(
            (row.outer_target_id, row.candidate_source, row.case_id), []
        ).append(row)
    expected_groups = {
        (target, source, case_id)
        for target in CENTERS
        for source in CENTERS
        if source != target
        for case_id in support_case_ids_by_target[target]
    }
    if set(grouped) != expected_groups:
        raise ProtocolError("Target-support action-shift target/source/case grid drifted.")
    by_target: dict[str, list[TargetSupportActionShiftCase]] = {target: [] for target in CENTERS}
    for (target, source, case_id), values in grouped.items():
        ordered = tuple(
            sorted(values, key=lambda row: (row.training_seed, row.generation_seed))
        )
        feature_set = feature_sets[target]
        if (
            tuple((row.training_seed, row.generation_seed) for row in ordered)
            != ENSEMBLE_SEED_KEYS
            or len({row.case_row_identity_hash for row in ordered}) != 1
            or len({row.support_row_count for row in ordered}) != 1
            or len({row.support_partition_hash for row in ordered}) != 1
            or len(
                {
                    row.case_ensemble_mean_absolute_positive_probability_shift
                    for row in ordered
                }
            )
            != 1
            or len(
                {row.case_base_ensemble_probability_sha256 for row in ordered}
            )
            != 1
            or len(
                {row.case_tail_ensemble_probability_sha256 for row in ordered}
            )
            != 1
            or len(
                {row.case_ensemble_absolute_difference_sha256 for row in ordered}
            )
            != 1
            or len({row.case_ensemble_shift_hash for row in ordered}) != 1
            or ordered[0].support_partition_hash
            != feature_set.plan.support_partition_hash
            or case_id not in feature_set.plan.support_case_ids
        ):
            raise ProtocolError("Target-support action-shift case lacks an exact-nine seed grid.")
        try:
            diagnostic_values = tuple(
                row.descriptive_seed_mean_absolute_positive_probability_shift
                for row in ordered
            )
            descriptive = np.asarray(diagnostic_values, dtype=np.float64)
            aggregate = SupportActionProbabilityShift(
                row_identity_hash=ordered[0].case_row_identity_hash,
                seed_keys=ENSEMBLE_SEED_KEYS,
                base_component_vector_hashes=tuple(
                    row.base_component_vector_hash for row in ordered
                ),
                tail_component_vector_hashes=tuple(
                    row.tail_component_vector_hash for row in ordered
                ),
                per_seed_mean_absolute_shifts=diagnostic_values,
                base_ensemble_probability_hash=(
                    ordered[0].case_base_ensemble_probability_sha256
                ),
                tail_ensemble_probability_hash=(
                    ordered[0].case_tail_ensemble_probability_sha256
                ),
                ensemble_absolute_difference_hash=(
                    ordered[0].case_ensemble_absolute_difference_sha256
                ),
                value=(
                    ordered[0].case_ensemble_mean_absolute_positive_probability_shift
                ),
                seed_standard_deviation=float(
                    np.std(descriptive, ddof=0, dtype=np.float64)
                ),
                seed_minimum=float(np.min(descriptive)),
                seed_maximum=float(np.max(descriptive)),
                seed_range=float(np.max(descriptive) - np.min(descriptive)),
                shift_hash=ordered[0].case_ensemble_shift_hash,
            )
            if any(
                row.base_component_vector_hash == row.base_probability_sha256
                or row.tail_component_vector_hash == row.tail_probability_sha256
                for row in ordered
            ):
                raise ProtocolError(
                    "Target-support action shift substituted a raw probability SHA."
                )
            typed = TargetSupportActionShiftCase(
                target_id=target, candidate_source=source, case_id=case_id,
                support_row_identity_hash=ordered[0].case_row_identity_hash,
                support_row_count=ordered[0].support_row_count,
                seed_keys=ENSEMBLE_SEED_KEYS,
                per_seed_mean_absolute_shifts=diagnostic_values,
                base_component_vector_hashes=tuple(
                    row.base_component_vector_hash for row in ordered
                ),
                tail_component_vector_hashes=tuple(
                    row.tail_component_vector_hash for row in ordered
                ),
                ensemble_mean_absolute_shift=aggregate.value,
                base_ensemble_probability_hash=(
                    aggregate.base_ensemble_probability_hash
                ),
                tail_ensemble_probability_hash=(
                    aggregate.tail_ensemble_probability_hash
                ),
                ensemble_absolute_difference_hash=(
                    aggregate.ensemble_absolute_difference_hash
                ),
            )
        except (ProtocolError, TypeError, ValueError) as exc:
            raise ProtocolError("Target-support action-shift case is malformed.") from exc
        by_target[target].append(typed)
    result = {
        target: tuple(sorted(values, key=lambda row: (row.candidate_source, row.case_id)))
        for target, values in by_target.items()
    }
    if any(
        len(result[target])
        != (len(CENTERS) - 1) * len(support_case_ids_by_target[target])
        for target in CENTERS
    ):
        raise ProtocolError("Target-support action-shift case coverage drifted.")
    return MappingProxyType(result)


def parse_target_feature_set(raw: object) -> TargetFeatureSet:
    production = target_feature_production_from_payload(raw)
    return TargetFeatureSet(
        target_id=production.target_id,
        plan=production.bootstrap_plan,
        point_surface=production.point_surface,
        bootstrap_surfaces=production.bootstrap_surfaces,
    )


def case_mapping(raw: object, role: str) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(raw, Mapping) or {str(key) for key in raw} != set(CENTERS): raise ProtocolError(f"Target reservation {role} case mapping drifted.")
    result = {}; seen: set[str] = set()
    for center in CENTERS:
        values = raw[center]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)): raise ProtocolError(f"Target reservation {role} cases are malformed.")
        cases = tuple(str(value) for value in values); minimum = MINIMUM_SUPPORT_CASE_COUNT if role == "support" else 1
        if len(cases) < minimum or cases != tuple(sorted(cases)) or len(set(cases)) != len(cases) or any(not value for value in cases) or seen.intersection(cases): raise ProtocolError(f"Target reservation {role} cases drifted.")
        seen.update(cases)
        result[center] = cases
    return MappingProxyType(result)


__all__ = ("LoadedTargetInputs", "TargetFeatureSet", "case_mapping", "load_target_inputs", "parse_target_feature_set")
