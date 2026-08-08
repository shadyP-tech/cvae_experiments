"""Independent typed reconstruction of target-support artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256
from ..utility_aligned.target_features import target_feature_production_from_payload
from ..utility_aligned_identities import (
    CENTERS,
    METADATA_PROFILE_SHA256,
    POLICY_EXPERIMENT_ID,
)
from .artifact_writer import sha256_file
from .config import load_utility_aligned_target_support_surface_config
from .contracts import (
    CACHE_ARTIFACT_ID,
    EXPERT_BANK_ARTIFACT_ID,
    EXPERIMENT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    REQUIRED_FILES,
    RESERVATION_ARTIFACT_ID,
)


SURFACE_KEYS = {
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


def validate_target_support_surface_bundle(root: str | Path) -> Mapping[str, object]:
    path = Path(root); discovered = tuple(path.rglob("*")) if path.exists() else ()
    if any(value.is_symlink() for value in discovered): raise ProtocolError("Target-support artifact forbids symbolic links.")
    actual = {str(value.relative_to(path)) for value in discovered if value.is_file()}
    if actual != set(REQUIRED_FILES): raise ProtocolError("Target-support artifact is not closed-world complete.")
    content = _json(path / "manifests/content_index.json")
    unhashed = {key: value for key, value in content.items() if key != "content_index_hash"}
    member_sha = content.get("member_sha256")
    if set(content) != {"schema_version", "artifact_id", "member_sha256", "surface_hash", "content_index_hash"} or content.get("schema_version") != "midogpp_utility_aligned_target_support_content_index_v1" or content.get("artifact_id") != OUTPUT_ARTIFACT_ID or not isinstance(member_sha, Mapping) or content.get("content_index_hash") != canonical_sha256(unhashed) or set(member_sha) != set(REQUIRED_FILES) - {"manifests/content_index.json"} or any(sha256_file(path / str(member)) != digest for member, digest in member_sha.items()):
        raise ProtocolError("Target-support content index drifted.")
    config = load_utility_aligned_target_support_surface_config(path / "config.resolved.yaml")
    _validate_provenance(path)
    reservation = _validate_reservation(path)
    cache = _validate_cache_lock(path, reservation)
    generation = _validate_generation_lock(path)
    surface = _json(path / "manifests/target_support_surface_lock.json")
    surface_unhashed = {key: value for key, value in surface.items() if key != "surface_hash"}
    if set(surface) != SURFACE_KEYS or surface.get("schema_version") != "midogpp_utility_aligned_target_support_surface_v1" or surface.get("artifact_id") != OUTPUT_ARTIFACT_ID or surface.get("surface_hash") != canonical_sha256(surface_unhashed) or surface.get("status") != "COMPLETE" or surface.get("claim_scope") != "routing_compatibility_only" or surface.get("target_support_parent_reservation_artifact_id") != RESERVATION_ARTIFACT_ID or surface.get("target_support_parent_reservation_hash") != reservation["reservation_hash"] or surface.get("metadata_profile_sha256") != METADATA_PROFILE_SHA256 or surface.get("config_contract_hash") != config.contract_hash or surface.get("input_artifact_ids") != list(INPUT_ARTIFACT_IDS) or surface.get("target_support_cache_binding_hash") != cache["cache_binding_hash"] or surface.get("target_support_cache_lock_hash") != cache["target_support_cache_lock_hash"] or surface.get("expert_bank_lock_hash") != generation["bank_lock_hash"] or surface.get("generation_lock_hash") != generation["generation_lock_hash"] or surface.get("source_generation_lock_hash") != generation["source_generation_lock_hash"] or surface.get("generated_cache_hash") != generation["generated_cache_hash"] or surface.get("feature_reference_rows_per_class") != 270 or surface.get("final_action_source_prefix_rows_per_class") != 256 or surface.get("final_action_geometry_executed_by_this_artifact") is not False or surface.get("labels_persisted") is not False or surface.get("target_evaluation_rows_opened") is not False:
        raise ProtocolError("Target-support surface lock drifted.")
    raw_features = surface.get("target_features")
    if not isinstance(raw_features, Sequence) or isinstance(raw_features, (str, bytes)) or len(raw_features) != len(CENTERS):
        raise ProtocolError("Target-support target feature coverage drifted.")
    productions = tuple(target_feature_production_from_payload(value) for value in raw_features)
    if tuple(value.target_id for value in productions) != CENTERS:
        raise ProtocolError("Target-support target order drifted.")
    _validate_tables(path, productions)
    plans = _json(path / "manifests/case_bootstrap_plans.json")
    if set(plans) != {"schema_version", "plans", "case_level_resampling", "minimum_independent_cases", "replicate_count_per_target", "case_bootstrap_plans_hash"} or plans.get("schema_version") != "midogpp_utility_aligned_target_case_bootstrap_plans_v1" or plans.get("case_level_resampling") is not True or plans.get("minimum_independent_cases") != 8 or plans.get("replicate_count_per_target") != 32 or plans.get("plans") != [value.bootstrap_plan.to_payload() for value in productions] or plans.get("case_bootstrap_plans_hash") != canonical_sha256({key: value for key, value in plans.items() if key != "case_bootstrap_plans_hash"}):
        raise ProtocolError("Target-support bootstrap-plan lock drifted.")
    if surface.get("support_case_ids_by_target") != reservation["support_case_ids_by_center"]:
        raise ProtocolError("Target-support surface case identities drifted.")
    state = _json(path / "reports/run_state.json")
    validation = _json(path / "reports/validation_report.json")
    leakage = _json(path / "reports/leakage_report.json")
    if set(leakage) != {"schema_version", "status", "surface_hash", "support_labels_used", "target_evaluation_rows_opened", "target_expert_excluded", "case_level_resampling"} or leakage.get("schema_version") != "midogpp_utility_aligned_target_support_leakage_report_v1" or leakage.get("status") != "PASS" or leakage.get("surface_hash") != surface["surface_hash"] or leakage.get("support_labels_used") is not False or leakage.get("target_evaluation_rows_opened") is not False or leakage.get("target_expert_excluded") is not True or leakage.get("case_level_resampling") is not True:
        raise ProtocolError("Target-support leakage report drifted.")
    if set(state) != {"schema_version", "status", "surface_hash", "point_feature_row_count", "bootstrap_feature_row_count", "target_count"} or state.get("schema_version") != "midogpp_utility_aligned_target_support_run_state_v1" or state.get("status") != "COMPLETE" or state.get("surface_hash") != surface["surface_hash"] or state.get("point_feature_row_count") != 648 or state.get("bootstrap_feature_row_count") != 20_736 or state.get("target_count") != 9 or set(validation) != {"schema_version", "status", "surface_hash", "closed_world_cache_validated", "typed_bootstrap_plans_reconstructed", "label_free_target_features_reconstructed"} or validation.get("schema_version") != "midogpp_utility_aligned_target_support_validation_report_v1" or validation.get("status") != "PASS" or validation.get("surface_hash") != surface["surface_hash"] or any(validation.get(key) is not True for key in ("closed_world_cache_validated", "typed_bootstrap_plans_reconstructed", "label_free_target_features_reconstructed")):
        raise ProtocolError("Target-support completion reports drifted.")
    if content.get("surface_hash") != surface["surface_hash"]:
        raise ProtocolError("Target-support content index escaped its surface.")
    return surface


def _validate_provenance(path: Path) -> None:
    raw = _json(path / "provenance/input_artifacts.json")
    required = {"schema_version", "dataset_id", "experiment_id", "stage", "claim_scope", "selection_used_target_eval_artifacts", "input_artifacts", "repository_revision", "repository_dirty", "repository_status_hash"}
    rows = raw.get("input_artifacts")
    if set(raw) != required or raw.get("schema_version") != "midogpp_input_artifacts_v2" or raw.get("dataset_id") != "midogpp" or raw.get("experiment_id") != EXPERIMENT_ID or raw.get("stage") != "60_routing_and_composition" or raw.get("claim_scope") != "routing_compatibility_only" or raw.get("selection_used_target_eval_artifacts") is not False or not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or tuple(str(row.get("artifact_id")) for row in rows if isinstance(row, Mapping)) != INPUT_ARTIFACT_IDS:
        raise ProtocolError("Target-support workspace provenance drifted.")


def _validate_reservation(path: Path) -> Mapping[str, object]:
    raw = _json(path / "manifests/target_reservation.json")
    required = {"schema_version", "artifact_id", "status", "authorized_consumer_experiment_ids", "dataset_family", "fresh_unconsumed_surface", "labels_present", "target_evaluation_rows_present", "support_case_ids_by_center", "support_rows_by_center", "reservation_id", "reservation_hash"}
    unhashed = {key: value for key, value in raw.items() if key != "reservation_hash"}
    if set(raw) != required or raw.get("schema_version") != "midogpp_utility_aligned_target_support_reservation_v1" or raw.get("artifact_id") != RESERVATION_ARTIFACT_ID or raw.get("status") != "ACTIVE" or raw.get("authorized_consumer_experiment_ids") != [EXPERIMENT_ID, POLICY_EXPERIMENT_ID] or raw.get("dataset_family") != "MIDOG++" or raw.get("fresh_unconsumed_surface") is not True or raw.get("labels_present") is not False or raw.get("target_evaluation_rows_present") is not False or raw.get("reservation_hash") != canonical_sha256(unhashed):
        raise ProtocolError("Target-support persisted reservation drifted.")
    cases = raw.get("support_case_ids_by_center")
    rows = raw.get("support_rows_by_center")
    if not isinstance(cases, Mapping) or not isinstance(rows, Mapping) or tuple(cases) != CENTERS or tuple(rows) != CENTERS:
        raise ProtocolError("Target-support persisted reservation coverage drifted.")
    seen_cases: set[str] = set(); seen_samples: set[str] = set(); seen_cache: set[tuple[str, int]] = set()
    for center in CENTERS:
        values = cases[center]; raw_rows = rows[center]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            raise ProtocolError("Target-support persisted reservation rows are malformed.")
        rendered = tuple(str(value) for value in values)
        if len(rendered) < 8 or len(set(rendered)) != len(rendered) or any(not value for value in rendered) or seen_cases.intersection(rendered):
            raise ProtocolError("Target-support persisted cases are not globally unique.")
        seen_cases.update(rendered); observed: set[str] = set()
        for ordinal, value in enumerate(raw_rows):
            if not isinstance(value, Mapping) or set(value) != {"row_ordinal", "sample_id", "case_id", "center", "cache_shard_path", "cache_row_index"}:
                raise ProtocolError("Target-support persisted row schema drifted.")
            try: row_ordinal = int(value["row_ordinal"]); cache_index = int(value["cache_row_index"])
            except (TypeError, ValueError, OverflowError) as exc: raise ProtocolError("Target-support persisted row numeric field drifted.") from exc
            sample = str(value["sample_id"]); case = str(value["case_id"]); shard = str(value["cache_shard_path"])
            if row_ordinal != ordinal or value.get("center") != center or not sample or not case or not shard or cache_index < 0 or sample in seen_samples or (shard, cache_index) in seen_cache:
                raise ProtocolError("Target-support persisted row identity drifted.")
            seen_samples.add(sample); seen_cache.add((shard, cache_index)); observed.add(case)
        if observed != set(rendered): raise ProtocolError("Target-support persisted row/case coverage drifted.")
    return raw


def _validate_cache_lock(path: Path, reservation: Mapping[str, object]) -> Mapping[str, object]:
    raw = _json(path / "manifests/target_support_cache_lock.json")
    required = {"schema_version", "cache_artifact_id", "reservation_hash", "cache_binding_hash", "labels_stored", "target_evaluation_rows_present", "target_support_cache_lock_hash"}
    if set(raw) != required or raw.get("schema_version") != "midogpp_utility_aligned_target_support_cache_lock_v1" or raw.get("cache_artifact_id") != CACHE_ARTIFACT_ID or raw.get("reservation_hash") != reservation["reservation_hash"] or not _sha(raw.get("cache_binding_hash")) or raw.get("labels_stored") is not False or raw.get("target_evaluation_rows_present") is not False or raw.get("target_support_cache_lock_hash") != canonical_sha256({key: value for key, value in raw.items() if key != "target_support_cache_lock_hash"}):
        raise ProtocolError("Target-support persisted cache lock drifted.")
    return raw


def _validate_generation_lock(path: Path) -> Mapping[str, object]:
    raw = _json(path / "manifests/source_generation_lock.json")
    required = {"schema_version", "expert_bank_artifact_id", "generation_lock_artifact_id", "generation_lock_hash", "bank_lock_hash", "generated_cache_hash", "source_stream_count", "feature_component_count", "feature_reference_rows_per_class", "final_action_source_prefix_rows_per_class", "final_action_geometry_executed_by_this_artifact", "generation_devices", "tf32_enabled", "amp_enabled", "support_labels_used", "source_generation_lock_hash"}
    if set(raw) != required or raw.get("schema_version") != "midogpp_utility_aligned_target_feature_generation_lock_v1" or raw.get("expert_bank_artifact_id") != EXPERT_BANK_ARTIFACT_ID or raw.get("generation_lock_artifact_id") != GENERATION_LOCK_ARTIFACT_ID or any(not _digest(raw.get(key)) for key in ("generation_lock_hash", "bank_lock_hash", "generated_cache_hash")) or raw.get("source_stream_count") != 81 or raw.get("feature_component_count") != 216 or raw.get("feature_reference_rows_per_class") != 270 or raw.get("final_action_source_prefix_rows_per_class") != 256 or raw.get("final_action_geometry_executed_by_this_artifact") is not False or raw.get("generation_devices") != ["cuda:0", "cuda:1"] or raw.get("tf32_enabled") is not False or raw.get("amp_enabled") is not False or raw.get("support_labels_used") is not False or raw.get("source_generation_lock_hash") != canonical_sha256({key: value for key, value in raw.items() if key != "source_generation_lock_hash"}):
        raise ProtocolError("Target-support persisted generation lock drifted.")
    return raw


def _sha(value: object) -> bool:
    rendered = str(value or "")
    return len(rendered) == 64 and all(character in "0123456789abcdef" for character in rendered)


def _digest(value: object) -> bool:
    rendered = str(value or "")
    return len(rendered) in {16, 64} and all(character in "0123456789abcdef" for character in rendered)


def _validate_tables(path: Path, productions: Sequence[object]) -> None:
    point = _csv(path / "tables/target_candidate_features.csv")
    bootstrap = _csv(path / "tables/target_candidate_feature_bootstraps.csv")
    expected_point = [{**row.to_payload(), "row_hash": row.row_hash} for production in productions for row in production.point_rows]
    expected_bootstrap = [
        {"bootstrap_replicate_index": replicate.replicate_index, "bootstrap_replicate_hash": replicate.replicate_hash, **row.to_payload(), "row_hash": row.row_hash}
        for production in productions
        for replicate, surface in zip(production.bootstrap_plan.replicates, production.bootstrap_surfaces, strict=True)
        for row in surface.rows
    ]
    if len(point) != 648 or len(bootstrap) != 20_736 or not _rows_equal(point, expected_point) or not _rows_equal(bootstrap, expected_bootstrap):
        raise ProtocolError("Target-support feature tables drifted from typed reconstruction.")


def _rows_equal(observed: Sequence[Mapping[str, str]], expected: Sequence[Mapping[str, object]]) -> bool:
    if len(observed) != len(expected): return False
    return all(tuple(left) == tuple(right) and all(str(left[key]) == str(value) for key, value in right.items()) for left, right in zip(observed, expected, strict=True))


def _csv(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return tuple(dict(value) for value in reader)
    except OSError as exc: raise ProtocolError(f"Cannot read target-support CSV: {path}.") from exc


def _json(path: Path) -> dict[str, object]:
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise ProtocolError(f"Cannot read target-support JSON: {path}.") from exc
    if not isinstance(value, dict): raise ProtocolError("Target-support JSON must be an object.")
    return value


__all__ = ("validate_target_support_surface_bundle",)
