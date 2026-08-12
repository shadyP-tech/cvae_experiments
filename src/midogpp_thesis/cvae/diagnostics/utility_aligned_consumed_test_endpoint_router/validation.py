"""Content-first reconstructive validation for the endpoint-router bundle."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...routing.metadata_compatibility.contracts import DOMAIN_MAPPING_SHA256
from .artifact_io import read_json, sha256_file
from .bundle import assert_closed_world, validate_content_index
from .config import (
    ConsumedTestEndpointRouterConfig,
    load_utility_aligned_consumed_test_endpoint_router_config,
)
from .experiment_contracts import (
    CENTERS,
    DEVELOPMENT_RESPONSE_COUNT,
    EXPECTED_EVALUATION_CASE_COUNT,
    EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER,
    EXPECTED_EVALUATION_ROW_COUNT,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SUPPORT_CASE_COUNT,
    EXPECTED_SUPPORT_ROW_COUNT,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    PRIMARY_CONTRASTS,
)
from .persistence import persist_validation_report
from .protocol import assert_consumed_test_diagnostic_only, canonical_consumed_test_protocol
from ...routing.residual_topup.hashing import canonical_sha256
from .inputs import (
    validate_active_diagnostic_workspace_binding,
    validate_workspace_provenance,
)
from .validation_science import validate_scientific_surfaces


def validate_utility_aligned_consumed_test_endpoint_router_bundle(
    root: str | Path,
    *,
    config: ConsumedTestEndpointRouterConfig | None = None,
    allow_pending: bool = False,
) -> dict[str, object]:
    """Validate bytes, identities, phase seals, and reconstructive row counts."""

    path = Path(root).resolve()
    validation_exists = (path / "reports/validation_report.json").is_file()
    assert_closed_world(
        path,
        allow_incomplete=False,
        allow_pending_validation=allow_pending or not validation_exists,
    )
    resolved = load_utility_aligned_consumed_test_endpoint_router_config(
        path / "config.resolved.yaml"
    )
    if (
        resolved.source_path.resolve() != (path / "config.resolved.yaml").resolve()
        or resolved.artifact_root.resolve() != path
        or any(
            not value.is_absolute()
            for value in (
                resolved.expert_bank_root,
                resolved.generation_lock_root,
                resolved.test_cache_root,
                resolved.test_manifest_path,
                resolved.domain_mapping_path,
                resolved.test_consumption_ledger_path,
                resolved.ledger_amendment_path,
            )
        )
    ):
        raise ProtocolError("Endpoint-router bundle/config path binding drifted.")
    if config is not None and (
        resolved.contract_hash != config.contract_hash
        or resolved.input_artifact_ids != config.input_artifact_ids
        or resolved.artifact_root.resolve() != config.artifact_root.resolve()
    ):
        raise ProtocolError("Supplied/resolved endpoint-router config drifted.")
    protocol = canonical_consumed_test_protocol()
    assert_consumed_test_diagnostic_only(protocol)
    validate_active_diagnostic_workspace_binding(resolved)

    # The content index is verified before scientific JSON or CSV is trusted.
    content = validate_content_index(
        path,
        config_contract_hash=resolved.contract_hash,
        protocol_contract_hash=protocol.contract_hash,
    )
    provenance = _validate_provenance(path, resolved)
    _validate_ledger_chain(resolved)
    support = _validate_support_partition(path)
    protocol_manifest = _validate_protocol_manifest(
        path, resolved, protocol, provenance=provenance
    )
    development = _validate_development(path)
    plans = _validate_plans_and_seals(path)
    terminal = _validate_terminal(path)
    scientific = validate_scientific_surfaces(path)
    runtime = _validate_runtime(path, runtime_contract=resolved.runtime)
    _validate_claim_reports(path)

    checks = {
        "schema_version": (
            "midogpp_utility_aligned_consumed_test_endpoint_router_validation_v1"
        ),
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "config_contract_hash": resolved.contract_hash,
        "protocol_contract_hash": protocol.contract_hash,
        "content_hash": content["content_hash"],
        "input_artifact_count": len(provenance),
        "support_case_count": support["support_case_count"],
        "support_row_count": support["support_row_count"],
        "evaluation_case_count": support["evaluation_case_count"],
        "evaluation_row_count": support["evaluation_row_count"],
        "development_response_count": development["response_count"],
        "target_plan_count": plans["target_plan_count"],
        "terminal_contrast_ids": terminal["contrast_ids"],
        "scientific": dict(scientific),
        "runtime_generation_devices": runtime["generation_devices"],
        "manifest_admission_hash": protocol_manifest["manifest_admission_hash"],
        "support_labels_used": False,
        "same_outer_H_evaluation_labels_used_for_plan_H": False,
        "previous_stage90_outputs_or_amendments_used": False,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
    }
    observed_path = path / "reports/validation_report.json"
    if observed_path.is_file() and read_json(observed_path) != checks:
        raise ProtocolError("Persisted endpoint-router validation report drifted.")
    if not allow_pending:
        state = read_json(path / "reports/run_state.json")
        if state.get("status") != "COMPLETE" or state.get("phase") != "COMPLETE":
            raise ProtocolError("Completed endpoint-router bundle lacks COMPLETE state.")
        persist_validation_report(path, checks)
    return checks


def _validate_provenance(
    root: Path, config: ConsumedTestEndpointRouterConfig
) -> Mapping[str, object]:
    rows = validate_workspace_provenance(root, config)
    if "midogpp_routing_metadata_profiles_v1" in rows:
        raise ProtocolError("Endpoint-router provenance admitted a seventh metadata input.")
    return rows


def _validate_ledger_chain(config: ConsumedTestEndpointRouterConfig) -> None:
    parent = config.test_consumption_ledger_path
    amendment = config.ledger_amendment_path
    if sha256_file(parent) != EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256:
        raise ProtocolError("Endpoint-router original ledger bytes drifted.")
    if sha256_file(amendment) != EXPECTED_LEDGER_AMENDMENT_SHA256:
        raise ProtocolError("Endpoint-router amendment bytes drifted.")
    payload = read_json(amendment)
    if (
        payload.get("parent_artifact_id")
        != "midogpp_uniform_b_test_consumption_ledger_v1"
        or payload.get("parent_sha256") != EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        or payload.get("authorized_consumer_experiment_ids") != [EXPERIMENT_ID]
        or payload.get("previous_stage90_outputs_used") is not False
        or payload.get("previous_stage90_amendments_used") is not False
        or payload.get("support_labels_used") is not False
        or payload.get("target_support_labels_used") is not False
        or payload.get(
            "cross_center_evaluation_labels_used_as_development_q_labels_after_development_seal"
        ) is not True
        or payload.get("same_outer_H_evaluation_labels_used_for_plan_H") is not False
        or payload.get(
            "same_outer_H_evaluation_labels_open_only_after_plan_H_and_global_target_prediction_seal"
        ) is not True
        or payload.get("generic_consumer_authorized") is not False
        or payload.get("may_feed_another_experiment") is not False
    ):
        raise ProtocolError("Endpoint-router amendment is not direct and single-consumer.")


def _validate_support_partition(root: Path) -> dict[str, int]:
    lock = read_json(root / "manifests/support_partition_lock.json")
    rows = _read_csv(root / "tables/support_partitions.csv")
    by_center: dict[str, dict[str, set[str]]] = {
        center: {"support": set(), "evaluation": set()} for center in CENTERS
    }
    row_counts = {"support": 0, "evaluation": 0}
    seen_ordinals: set[int] = set()
    seen_manifest_indices: set[int] = set()
    seen_evaluation_ids: set[str] = set()
    by_center_rows: dict[str, dict[str, list[dict[str, object]]]] = {
        center: {"support": [], "evaluation": []} for center in CENTERS
    }
    for expected_ordinal, row in enumerate(rows):
        center = row.get("center", "")
        role = row.get("partition_role", "")
        case_id = row.get("case_id", "")
        try:
            ordinal = int(row.get("row_ordinal", ""))
            manifest_index = int(row.get("manifest_row_index", ""))
        except ValueError as exc:
            raise ProtocolError("Endpoint-router partition ordinal is malformed.") from exc
        evaluation_id = row.get("evaluation_row_id", "")
        if (
            center not in by_center
            or role not in row_counts
            or not case_id
            or ordinal != expected_ordinal
            or ordinal in seen_ordinals
            or manifest_index in seen_manifest_indices
            or not evaluation_id
            or evaluation_id in seen_evaluation_ids
            or row.get("split") != "test"
            or row.get("support_partition_namespace")
            != lock.get("support_partition_namespace")
            or row.get("membership_seed") not in {"", "None"}
            or row.get("label_present") != "False"
        ):
            raise ProtocolError("Endpoint-router support-partition row is malformed.")
        seen_ordinals.add(ordinal)
        seen_manifest_indices.add(manifest_index)
        seen_evaluation_ids.add(evaluation_id)
        by_center[center][role].add(case_id)
        by_center_rows[center][role].append(
            {
                "row_ordinal": ordinal,
                "manifest_row_index": manifest_index,
                "evaluation_row_id": evaluation_id,
                "case_id": case_id,
                "center": center,
                "split": "test",
                "partition_role": role,
            }
        )
        row_counts[role] += 1
    for center in CENTERS:
        support = by_center[center]["support"]
        evaluation = by_center[center]["evaluation"]
        all_cases = tuple(sorted(support | evaluation))
        if (
            len(support) != 8
            or len(evaluation) != EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER[center]
            or support.intersection(evaluation)
            or tuple(sorted(support)) != all_cases[:8]
            or tuple(sorted(evaluation)) != all_cases[8:]
        ):
            raise ProtocolError("Endpoint-router whole-case partition drifted.")
        center_payload = {
            "schema_version": "midogpp_consumed_test_center_partition_v1",
            "center": center,
            "namespace": lock.get("support_partition_namespace"),
            "membership_rule": "canonical_case_id_sort_then_first_eight",
            "seed_used": False,
            "support_case_ids": sorted(support),
            "evaluation_case_ids": sorted(evaluation),
            "support_row_identity_hash": canonical_sha256(by_center_rows[center]["support"]),
            "evaluation_row_identity_hash": canonical_sha256(by_center_rows[center]["evaluation"]),
        }
        partition_hashes = lock.get("partition_hashes_by_center")
        if (
            not isinstance(partition_hashes, Mapping)
            or partition_hashes.get(center) != canonical_sha256(center_payload)
            or any(
                row.get("center_partition_hash") != partition_hashes.get(center)
                for row in rows
                if row.get("center") == center
            )
        ):
            raise ProtocolError("Endpoint-router center partition hash drifted.")
    if (
        lock.get("membership_seed") is not None
        or "seed_used" in lock
        or lock.get("membership_rule")
        != "canonical_case_id_sort_then_first_eight"
        or lock.get("labels_used") is not False
    ):
        raise ProtocolError("Endpoint-router support partition became seed-dependent.")
    support_cases = sum(len(value["support"]) for value in by_center.values())
    evaluation_cases = sum(len(value["evaluation"]) for value in by_center.values())
    support_rows = int(lock.get("support_row_count_total", -1))
    evaluation_rows = int(lock.get("evaluation_row_count_total", -1))
    if (
        support_cases != EXPECTED_SUPPORT_CASE_COUNT
        or evaluation_cases != EXPECTED_EVALUATION_CASE_COUNT
        or support_rows != EXPECTED_SUPPORT_ROW_COUNT
        or evaluation_rows != EXPECTED_EVALUATION_ROW_COUNT
        or row_counts["support"] != EXPECTED_SUPPORT_ROW_COUNT
        or row_counts["evaluation"] != EXPECTED_EVALUATION_ROW_COUNT
        or int(lock.get("support_case_count_total", -1))
        != EXPECTED_SUPPORT_CASE_COUNT
        or int(lock.get("evaluation_case_count_total", -1))
        != EXPECTED_EVALUATION_CASE_COUNT
    ):
        raise ProtocolError("Endpoint-router support/evaluation counts drifted.")
    lock_unhashed = {
        key: value for key, value in lock.items() if key != "support_partition_lock_hash"
    }
    if lock.get("support_partition_lock_hash") != canonical_sha256(lock_unhashed):
        raise ProtocolError("Endpoint-router support-partition lock hash drifted.")
    return {
        "support_case_count": support_cases,
        "support_row_count": support_rows,
        "evaluation_case_count": evaluation_cases,
        "evaluation_row_count": evaluation_rows,
    }


def _validate_protocol_manifest(
    root: Path,
    config: ConsumedTestEndpointRouterConfig,
    protocol: object,
    *,
    provenance: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    payload = read_json(root / "manifests/protocol_manifest.json")
    unhashed = {
        key: value for key, value in payload.items() if key != "protocol_manifest_hash"
    }
    admission_unhashed = {
        "schema_version": "midogpp_endpoint_router_manifest_admission_v1",
        "status": "PASS",
        "manifest_sha256": config.expected_manifest_sha256,
        "manifest_parsed": False,
        "labels_opened": False,
        "domain_mapping_may_now_be_parsed": True,
    }
    firewall = payload.get("pre_gpu_firewall")
    if (
        payload.get("schema_version")
        != "midogpp_utility_aligned_consumed_test_endpoint_router_protocol_manifest_v1"
        or payload.get("protocol_manifest_hash") != canonical_sha256(unhashed)
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("config_contract_hash") != config.contract_hash
        or payload.get("signed_protocol") != protocol.to_payload()
        or payload.get("input_artifact_count") != 6
        or payload.get("global_source_control_provenance")
        != "experiment_manifest_only"
        or payload.get("exact_nelbo_computed") is not False
        or payload.get("reconstruction_kl_or_mmd_enter_learned_router")
        is not False
        or payload.get("learned_router_predictors")
        != [
            "experiment_manifest_metadata_global_source_control",
            "unsigned_ensemble_first_support_action_probability_shift",
        ]
        or payload.get("support_probability_shift_is_generative_compatibility")
        is not False
        or payload.get("Hxe_utility_action_semantics")
        != "equal_union_B_plus_single_source_tail"
        or payload.get("domain_mapping_member_shares_test_manifest_artifact_id")
        is not True
        or payload.get("support_labels_used") is not False
        or payload.get("same_outer_H_evaluation_labels_used_for_plan_H") is not False
        or payload.get("input_artifact_hashes")
        != {
            artifact_id: canonical_sha256(provenance[artifact_id])
            for artifact_id in config.input_artifact_ids
        }
        or payload.get("manifest_admission_hash")
        != canonical_sha256(admission_unhashed)
        or not isinstance(firewall, Mapping)
        or firewall.get("schema_version")
        != "midogpp_endpoint_router_pre_gpu_firewall_v1"
        or firewall.get("firewall_hash")
        != canonical_sha256(
            {key: value for key, value in firewall.items() if key != "firewall_hash"}
        )
        or firewall.get("status") != "PASS"
        or firewall.get("test_cache_binding_hash")
        != payload.get("test_cache_binding_hash")
        or firewall.get("support_labels_opened") is not False
        or firewall.get("evaluation_labels_opened") is not False
    ):
        raise ProtocolError("Endpoint-router protocol manifest drifted.")
    if sha256_file(config.test_manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ProtocolError("Endpoint-router manifest bytes drifted.")
    if sha256_file(config.domain_mapping_path) != DOMAIN_MAPPING_SHA256:
        raise ProtocolError("Endpoint-router domain mapping bytes drifted.")
    return payload


def _validate_development(root: Path) -> dict[str, int]:
    rows = _read_csv(root / "tables/development_endpoint_responses.csv")
    seal = read_json(root / "manifests/development_endpoint_response_seal.json")
    label_report = read_json(root / "reports/development_label_access_report.json")
    if (
        len(rows) != DEVELOPMENT_RESPONSE_COUNT
        or int(seal.get("response_count", -1)) != DEVELOPMENT_RESPONSE_COUNT
        or label_report.get("cross_center_evaluation_labels_used_as_development_q_labels")
        is not True
        or label_report.get("same_outer_H_labels_excluded_from_each_H_model") is not True
        or label_report.get("support_labels_opened") is not False
        or label_report.get("strict_H_q_e_exclusion") is not True
    ):
        raise ProtocolError("Endpoint-router development response boundary drifted.")
    return {"response_count": len(rows)}


def _validate_plans_and_seals(root: Path) -> dict[str, int]:
    plans = read_json(root / "manifests/target_policy_plans.json")
    frozen = read_json(root / "manifests/frozen_actions.json")
    prelabel = read_json(root / "manifests/global_prelabel_seal.json")
    plan_rows = _read_csv(root / "tables/target_policy_plans.csv")
    if (
        len(plan_rows) != len(CENTERS)
        or int(plans.get("target_plan_count", len(plan_rows))) != len(CENTERS)
        or int(frozen.get("target_count", -1)) != len(CENTERS)
        or int(frozen.get("reported_action_count", -1)) != 117
        or int(frozen.get("physical_action_count", -1)) != 90
        or prelabel.get("target_plan_count") != len(CENTERS)
        or prelabel.get("support_labels_used") is not False
        or prelabel.get("same_outer_H_evaluation_labels_used_for_plan_H") is not False
        or prelabel.get("global_target_prediction_seal_hash") in (None, "")
    ):
        raise ProtocolError("Endpoint-router target plan/global seal drifted.")
    for row in plan_rows:
        selected = row.get("selected_action_role", row.get("selected_action", ""))
        if selected not in {"B", "R"}:
            raise ProtocolError("Endpoint-router deployable diagnostic plan is not R-or-B.")
    return {"target_plan_count": len(plan_rows)}


def _validate_terminal(root: Path) -> dict[str, object]:
    contrasts = _read_csv(root / "tables/center_contrasts.csv")
    aggregate = _read_csv(root / "tables/aggregate_contrasts.csv")
    oracle = _read_csv(root / "tables/oracle_rank_diagnostics.csv")
    sealed = read_json(root / "manifests/sealed_terminal_evaluation.json")
    contrast_ids = sorted({row.get("contrast_id", "") for row in contrasts})
    aggregate_ids = sorted(row.get("contrast_id", "") for row in aggregate)
    if (
        contrast_ids != sorted(PRIMARY_CONTRASTS)
        or aggregate_ids != sorted(PRIMARY_CONTRASTS)
        or any(
            row.get("center_count") != "9"
            or row.get("degrees_of_freedom") != "8"
            or row.get("inference_unit") != "target_center"
            or row.get("terminal_scores_may_update_plan") != "False"
            for row in aggregate
        )
        or len(oracle) != len(CENTERS)
        or sealed.get("same_outer_H_evaluation_labels_used_for_plan_H") is not False
        or sealed.get("support_labels_used") is not False
        or sealed.get("terminal_only_no_plan_or_policy_update") is not True
    ):
        raise ProtocolError("Endpoint-router terminal scoring surface drifted.")
    return {"contrast_ids": contrast_ids}


def _validate_runtime(
    root: Path, *, runtime_contract: Mapping[str, object]
) -> Mapping[str, object]:
    payload = read_json(root / "reports/runtime_summary.json")
    preflight = payload.get("workstation_preflight")
    counts = payload.get("counts")
    expected_counts = {
        "source_stream_count": 81,
        "development_prediction_cell_count": 5_184,
        "development_response_count": 504,
        "target_prediction_cell_count": 810,
        "target_reported_action_count": 117,
        "terminal_score_count": 117,
    }
    gpu_rows = preflight.get("gpus") if isinstance(preflight, Mapping) else None
    if (
        payload.get("generation_devices") != ["cuda:0", "cuda:1"]
        or payload.get("generation_workers_per_device") != 1
        or payload.get("classifier_workers") != 4
        or payload.get("classifier_threads_per_worker") != 3
        or payload.get("array_storage_dtype") != "float32"
        or payload.get("scientific_reduction_dtype") != "float64"
        or payload.get("phase_disjoint_gpu_and_cpu_pools") is not True
        or payload.get("parent_cuda_context_forbidden_during_cpu_phase") is not True
        or payload.get("hash_validated_resume") is not True
        or not isinstance(preflight, Mapping)
        or preflight.get("status") != "PASS"
        or preflight.get("available_cpu_affinity_count", 0)
        < int(runtime_contract["minimum_logical_cpu_count"])
        or preflight.get("physical_ram_bytes", 0)
        < int(runtime_contract["minimum_physical_ram_bytes"])
        or preflight.get("disk_free_bytes_at_launch", 0)
        < int(runtime_contract["minimum_artifact_disk_free_bytes"])
        or not isinstance(gpu_rows, list)
        or len(gpu_rows) != 2
        or not all(isinstance(row, Mapping) for row in gpu_rows)
        or sorted(int(row.get("index", -1)) for row in gpu_rows) != [0, 1]
        or any(
            "RTX A5000" not in str(row.get("name", ""))
            or int(row.get("memory_free_mib", -1))
            < int(runtime_contract["minimum_gpu_free_mib_per_device"])
            for row in gpu_rows
        )
        or not isinstance(counts, Mapping)
        or dict(counts) != expected_counts
        or preflight.get("development_prediction_cell_count") != 5_184
        or preflight.get("target_prediction_cell_count") != 810
        or preflight.get("maximum_total_classifier_fit_count") != 5_994
        or preflight.get("preflight_reprobed_before_each_compute_session") is not True
    ):
        raise ProtocolError("Endpoint-router runtime summary drifted.")
    return payload


def _validate_claim_reports(root: Path) -> None:
    leakage = read_json(root / "reports/leakage_report.json")
    publication = read_json(root / "reports/publication_decision.json")
    capability = read_json(root / "reports/label_capability_report.json")
    if (
        leakage.get("status") != "PASS"
        or leakage.get("support_labels_used") is not False
        or leakage.get("same_outer_H_evaluation_labels_used_for_plan_H") is not False
        or leakage.get("exact_nelbo_computed_or_claimed") is not False
        or leakage.get("reconstruction_kl_or_mmd_enter_learned_router") is not False
        or leakage.get("support_probability_shift_is_unsigned_classifier_sensitivity")
        is not True
        or leakage.get("Hxe_is_hybrid_B_plus_single_source_tail") is not True
        or publication.get("decision") != "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
        or publication.get("actual_learned_predictors")
        != [
            "experiment_manifest_metadata_global_source_control",
            "unsigned_ensemble_first_support_action_probability_shift",
        ]
        or publication.get("descriptive_cvae_diagnostics_are_not_nelbo_or_utility")
        is not True
        or publication.get("support_probability_shift_is_not_generative_compatibility")
        is not True
        or publication.get("Hxe_and_R_are_B_plus_tail_actions_not_standalone_expert_utility")
        is not True
        or any(
            publication.get(key) is not False
            for key in (
                "fresh_evidence",
                "routing_success_claimed",
                "routing_quality_claimed",
                "action_selection_authorized",
                "policy_update_authorized",
                "model_update_authorized",
                "expert_update_authorized",
                "promotion_eligible",
                "may_feed_stage50",
                "may_feed_stage60",
                "may_feed_stage70",
                "may_feed_another_stage90",
                "may_feed_another_experiment",
                "generic_consumer_authorized",
            )
        )
        or capability.get("support_labels_opened") is not False
        or capability.get("same_outer_H_evaluation_labels_used_for_plan_H") is not False
    ):
        raise ProtocolError("Endpoint-router claim or capability report drifted.")


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ProtocolError(f"Cannot read endpoint-router CSV: {path}.") from exc
    if not rows:
        raise ProtocolError(f"Endpoint-router CSV is empty: {path}.")
    return rows


__all__ = ("validate_utility_aligned_consumed_test_endpoint_router_bundle",)
