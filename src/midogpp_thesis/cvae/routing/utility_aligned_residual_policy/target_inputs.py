"""Fresh target reservation and typed feature-surface admission."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256
from ..utility_aligned import CaseBootstrapPlan, FeatureSurface
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
from .input_io import read_json


TARGET_SUPPORT_LOCK_MEMBER = "manifests/target_support_surface_lock.json"
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


def load_target_inputs(*, support_surface_root: Path, parent_reservation_root: Path, target_reservation_root: Path) -> LoadedTargetInputs:
    parent = _load_parent(parent_reservation_root)
    target = _load_target_reservation(target_reservation_root)
    parent_support = case_mapping(parent["support_case_ids_by_center"], "support")
    support = case_mapping(target["support_case_ids_by_center"], "support")
    evaluation = case_mapping(target["evaluation_case_ids_by_center"], "evaluation")
    if parent_support != support:
        raise ProtocolError("Target-support parent cases differ from Stage-70 support cases.")
    surface = _load_surface(support_surface_root, parent=parent, target=target)
    return LoadedTargetInputs(
        surface_hash=str(surface["surface_hash"]), parent_artifact_id=str(parent["artifact_id"]),
        parent_hash=str(parent["reservation_hash"]), reservation_hash=str(target["reservation_hash"]),
        evaluation_binding_hash=str(target["target_evaluation_binding_hash"]),
        support_case_ids_by_target=support, evaluation_case_ids_by_target=evaluation,
        feature_sets=MappingProxyType(surface["target_features"]),
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
        "feature_reference_rows_per_class",
        "final_action_source_prefix_rows_per_class",
        "final_action_geometry_executed_by_this_artifact",
        "support_case_ids_by_target", "target_features", "labels_persisted",
        "target_evaluation_rows_opened", "surface_hash",
    }
    unhashed = {key: value for key, value in raw.items() if key != "surface_hash"}
    if set(raw) != required or raw.get("schema_version") != TARGET_SUPPORT_SCHEMA or raw.get("artifact_id") != "midogpp_utility_aligned_target_support_surface_v1" or raw.get("claim_scope") != "routing_compatibility_only" or raw.get("status") != "COMPLETE" or raw.get("target_support_parent_reservation_artifact_id") != TARGET_SUPPORT_PARENT_RESERVATION_ARTIFACT_ID or raw.get("target_support_parent_reservation_hash") != parent["reservation_hash"] or raw.get("support_case_ids_by_target") != parent["support_case_ids_by_center"] or raw.get("support_case_ids_by_target") != target["support_case_ids_by_center"] or raw.get("metadata_profile_sha256") != METADATA_PROFILE_SHA256 or raw.get("feature_reference_rows_per_class") != 270 or raw.get("final_action_source_prefix_rows_per_class") != 256 or raw.get("final_action_geometry_executed_by_this_artifact") is not False or raw.get("labels_persisted") is not False or raw.get("target_evaluation_rows_opened") is not False or raw.get("surface_hash") != canonical_sha256(unhashed):
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
    return {"surface_hash": raw["surface_hash"], "target_features": by_target}


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
