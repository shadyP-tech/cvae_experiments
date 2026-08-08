"""Atomic closed-world serialization for target-support feature products."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ..exact_tail_utility_surface.source_generation import GeneratedDevelopmentCache
from ..residual_topup.hashing import canonical_sha256
from ..utility_aligned.target_features import TargetFeatureProduction
from ..utility_aligned_identities import CENTERS, METADATA_PROFILE_SHA256
from .config import TargetSupportSurfaceConfig
from .contracts import (
    CACHE_ARTIFACT_ID,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    REQUIRED_FILES,
)
from .inputs import TargetSupportInputs


def persist_target_support_artifact(
    config: TargetSupportSurfaceConfig,
    inputs: TargetSupportInputs,
    generated: GeneratedDevelopmentCache,
    productions: Sequence[TargetFeatureProduction],
) -> Path:
    root = config.artifact_root; root.mkdir(parents=True, exist_ok=True)
    for member in ("config.resolved.yaml", "provenance/input_artifacts.json"):
        if not (root / member).is_file(): raise ProtocolError(f"Target-support workspace prelude is absent: {member}.")
    write_json(root / "manifests/target_reservation.json", inputs.reservation_payload)
    cache_lock = hashed({
        "schema_version": "midogpp_utility_aligned_target_support_cache_lock_v1",
        "cache_artifact_id": CACHE_ARTIFACT_ID,
        "reservation_hash": inputs.reservation_hash, "cache_binding_hash": inputs.cache_binding_hash,
        "labels_stored": False, "target_evaluation_rows_present": False,
    }, "target_support_cache_lock_hash")
    write_json(root / "manifests/target_support_cache_lock.json", cache_lock)
    generation = hashed({
        "schema_version": "midogpp_utility_aligned_target_feature_generation_lock_v1",
        "expert_bank_artifact_id": EXPERT_BANK_ARTIFACT_ID,
        "generation_lock_artifact_id": GENERATION_LOCK_ARTIFACT_ID,
        "generation_lock_hash": generated.generation_lock_hash, "bank_lock_hash": generated.bank_lock_hash,
        "generated_cache_hash": generated.cache_hash, "source_stream_count": len(generated.source_records),
        "feature_component_count": len(generated.component_records),
        "feature_reference_rows_per_class": 270,
        "final_action_source_prefix_rows_per_class": 256,
        "final_action_geometry_executed_by_this_artifact": False,
        "generation_devices": ["cuda:0", "cuda:1"], "tf32_enabled": False,
        "amp_enabled": False, "support_labels_used": False,
    }, "source_generation_lock_hash")
    write_json(root / "manifests/source_generation_lock.json", generation)
    plans = hashed({
        "schema_version": "midogpp_utility_aligned_target_case_bootstrap_plans_v1",
        "plans": [value.bootstrap_plan.to_payload() for value in productions],
        "case_level_resampling": True, "minimum_independent_cases": 8,
        "replicate_count_per_target": 32,
    }, "case_bootstrap_plans_hash")
    write_json(root / "manifests/case_bootstrap_plans.json", plans)
    features = [feature_payload(value) for value in productions]
    surface = hashed({
        "schema_version": "midogpp_utility_aligned_target_support_surface_v1",
        "artifact_id": OUTPUT_ARTIFACT_ID, "claim_scope": "routing_compatibility_only",
        "status": "COMPLETE",
        "target_support_parent_reservation_artifact_id": "midogpp_utility_aligned_target_support_reservation_v1",
        "target_support_parent_reservation_hash": inputs.reservation_hash,
        "metadata_profile_sha256": METADATA_PROFILE_SHA256,
        "config_contract_hash": config.contract_hash,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "target_support_cache_binding_hash": inputs.cache_binding_hash,
        "target_support_cache_lock_hash": cache_lock["target_support_cache_lock_hash"],
        "expert_bank_lock_hash": generated.bank_lock_hash,
        "generation_lock_hash": generated.generation_lock_hash,
        "source_generation_lock_hash": generation["source_generation_lock_hash"],
        "generated_cache_hash": generated.cache_hash,
        "feature_reference_rows_per_class": 270,
        "final_action_source_prefix_rows_per_class": 256,
        "final_action_geometry_executed_by_this_artifact": False,
        "support_case_ids_by_target": {target: sorted(set(inputs.case_ids_by_target[target])) for target in CENTERS},
        "target_features": features, "labels_persisted": False,
        "target_evaluation_rows_opened": False,
    }, "surface_hash")
    write_json(root / "manifests/target_support_surface_lock.json", surface)
    point = []; bootstraps = []
    for production in productions:
        point.extend({**row.to_payload(), "row_hash": row.row_hash} for row in production.point_rows)
        for replicate, feature_surface in zip(production.bootstrap_plan.replicates, production.bootstrap_surfaces, strict=True):
            bootstraps.extend({"bootstrap_replicate_index": replicate.replicate_index, "bootstrap_replicate_hash": replicate.replicate_hash, **row.to_payload(), "row_hash": row.row_hash} for row in feature_surface.rows)
    write_csv(root / "tables/target_candidate_features.csv", point)
    write_csv(root / "tables/target_candidate_feature_bootstraps.csv", bootstraps)
    write_json(root / "reports/leakage_report.json", {
        "schema_version": "midogpp_utility_aligned_target_support_leakage_report_v1",
        "status": "PASS", "surface_hash": surface["surface_hash"], "support_labels_used": False,
        "target_evaluation_rows_opened": False, "target_expert_excluded": True,
        "case_level_resampling": True,
    })
    write_json(root / "reports/run_state.json", {
        "schema_version": "midogpp_utility_aligned_target_support_run_state_v1",
        "status": "COMPLETE", "surface_hash": surface["surface_hash"],
        "point_feature_row_count": len(point), "bootstrap_feature_row_count": len(bootstraps),
        "target_count": len(productions),
    })
    write_json(root / "reports/validation_report.json", {
        "schema_version": "midogpp_utility_aligned_target_support_validation_report_v1",
        "status": "PASS", "surface_hash": surface["surface_hash"],
        "closed_world_cache_validated": True, "typed_bootstrap_plans_reconstructed": True,
        "label_free_target_features_reconstructed": True,
    })
    member_sha = {member: sha256_file(root / member) for member in REQUIRED_FILES if member != "manifests/content_index.json"}
    write_json(root / "manifests/content_index.json", hashed({
        "schema_version": "midogpp_utility_aligned_target_support_content_index_v1",
        "artifact_id": OUTPUT_ARTIFACT_ID,
        "member_sha256": member_sha, "surface_hash": surface["surface_hash"],
    }, "content_index_hash"))
    return root


def feature_payload(production: TargetFeatureProduction) -> Mapping[str, object]:
    payload = {
        "target_id": production.target_id, "case_bootstrap_plan": production.bootstrap_plan.to_payload(),
        "point_rows": [{**row.to_payload(), "row_hash": row.row_hash} for row in production.point_rows],
        "point_surface_hash": production.point_surface.surface_hash,
        "bootstrap_surfaces": [
            {"replicate_index": replicate.replicate_index, "replicate_hash": replicate.replicate_hash,
             "rows": [{**row.to_payload(), "row_hash": row.row_hash} for row in surface.rows],
             "surface_hash": surface.surface_hash}
            for replicate, surface in zip(production.bootstrap_plan.replicates, production.bootstrap_surfaces, strict=True)
        ],
    }
    return {**payload, "target_feature_hash": canonical_sha256(payload)}


def hashed(payload: Mapping[str, object], key: str) -> dict[str, object]:
    result = dict(payload); result[key] = canonical_sha256(result); return result


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    values = tuple(dict(value) for value in rows)
    if not values: raise ProtocolError("Target-support refuses an empty feature table.")
    columns = tuple(values[0])
    if any(tuple(value) != columns for value in values): raise ProtocolError("Target-support CSV schema drifted.")
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n"); writer.writeheader(); writer.writerows(values)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


__all__ = ("feature_payload", "persist_target_support_artifact", "sha256_file")
