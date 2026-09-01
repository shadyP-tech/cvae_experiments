"""Protocol-ordered, workstation-optimized HARP v3 production runner."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import atomic_json, read_json, sha256_file
from ...runtime.harp_v3_execution.contracts import (
    ArtifactValue,
    HarpV3Pipeline,
    PrelabelRouteSet,
)
from ...runtime.harp_v3_execution.durability import durable_barrier
from ...runtime.harp_v3_execution.journal import LabelFreeProgressJournal
from ...runtime.harp_v3_execution.phases import PHASE_ORDER, PhaseLedger
from ...runtime.harp_v3_execution.prelabel_diagnostics import (
    build_prelabel_diagnostics,
)
from ...runtime.harp_v3_execution.production import HarpV3ProductionPipeline
from ...runtime.harp_v3_execution.stores import (
    read_label_free_outer_menu,
    read_prelabel_routes,
    write_artifact_value,
    write_label_free_outer_menu,
    write_prelabel_routes,
)
from ...runtime.harp_v3_execution.validation import run_two_fresh_validations
from ....workspace.preparation_authority import HARP_V3_RUN_CONFIRMATION_TOKEN
from .authorization import (
    HarpV3Authorization,
    HarpV3AuthorizationLease,
    claim_authorization,
    finalize_authorization,
    load_authorization,
)
from .config import HarpStage90V3Config
from .identity import (
    EXECUTION_REVISION,
    EXPERIMENT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from .input_surfaces import (
    DEVELOPMENT_ROLE,
    EVALUATION_ROLE,
    HarpConsumedCacheIndex,
    load_cache_index,
    load_development_labels,
    load_evaluation_truth,
)


IMPLEMENTED_COMPONENTS = (
    "separate_single_use_v3_authority_before_all_output_or_input_mutation",
    "exact_launch_confirmation_before_authority_input_path_or_output_access",
    "catalog_bound_crash_recoverable_v3_input_preparation_before_activation",
    "mutation_free_activation_plan_with_resumable_amendment_and_registry_last_commit",
    "v3_owned_label_blind_cache_and_physical_generation_lineage",
    "two_persistent_gpu_source_workers_closed_before_cpu_classifier_pool",
    "four_spawned_cuda_blind_cpu_workers_with_enforced_three_thread_blas_limits",
    "process_local_verified_source_frame_memmaps_and_source_block_hash_cache",
    "grouped_once_per_shard_consumed_frame_staging",
    "physical_lambda_one_B_U_Hxe_action_slate_without_probability_blends",
    "strict_outer_H_query_candidate_exclusion",
    "float32_probability_transport_and_float64_scientific_reductions",
    "deterministic_compact_npz_stores_with_chunk_hashes",
    "case_level_source_only_joint_multioutput_model",
    "per_outer_exact_identity_ridge_memo_with_byte_identical_fit_payload",
    "source_lodo_statistic_matched_ensemble_max_geometry_not_raw_leverage",
    "hierarchical_B_then_U_then_Hxe_joint_endpoint_policy",
    "byte_identical_exact_B_case_fallback",
    "label_free_preterminal_rejection_diagnostics",
    "two_distinct_spawn_process_route_reconstructions",
    "external_config_center_and_all_candidate_cross_binding",
    "disk_reconstructed_frozen_routes_are_the_only_terminal_input",
    "evaluation_labels_opened_only_after_frozen_route_seal",
    "single_content_index_and_atomic_fsync_completion_commit",
    "mutation_free_live_cpu_ram_gpu_scratch_and_dependency_preflight",
)


@dataclass(frozen=True, slots=True)
class HarpV3RunnerServices:
    config_type: type
    authorization_type: type
    lease_type: type
    load_authorization: Callable[[Any], Any]
    claim_authorization: Callable[..., Any]
    finalize_authorization: Callable[..., Any]
    load_cache_index: Callable[[Any], Any]
    load_development_labels: Callable[[Any, Any], object]
    load_evaluation_truth: Callable[[Any, Any], object]


V3_RUNNER_SERVICES = HarpV3RunnerServices(
    config_type=HarpStage90V3Config,
    authorization_type=HarpV3Authorization,
    lease_type=HarpV3AuthorizationLease,
    load_authorization=load_authorization,
    claim_authorization=claim_authorization,
    finalize_authorization=finalize_authorization,
    load_cache_index=load_cache_index,
    load_development_labels=load_development_labels,
    load_evaluation_truth=load_evaluation_truth,
)


def _pipeline_or_production(pipeline: HarpV3Pipeline | None) -> HarpV3Pipeline:
    return (
        HarpV3ProductionPipeline(
            development_role=DEVELOPMENT_ROLE, evaluation_role=EVALUATION_ROLE
        )
        if pipeline is None
        else pipeline
    )


def inspect_harp_stage90_v3(config: HarpStage90V3Config) -> Mapping[str, object]:
    """Path-free, mutation-free inspection of planned or activated identity."""

    if type(config) is not HarpStage90V3Config:
        raise ProtocolError("HARP v3 inspection requires a typed configuration.")
    body = {
        "schema_version": "midogpp_harp_v3_implementation_inspection_v1",
        "status": (
            "EXECUTABLE_AUTHORIZED_UNPROBED"
            if config.execution_authorized
            else "PLANNED_NEEDS_SEPARATE_EXECUTION_AMENDMENT"
        ),
        "experiment_id": EXPERIMENT_ID,
        "execution_revision": EXECUTION_REVISION,
        "config_hash": config.config_hash,
        "implemented_components": list(IMPLEMENTED_COMPONENTS),
        "phase_order": list(PHASE_ORDER),
        "execution_authorized": config.execution_authorized,
        "authorization_probed": False,
        "paths_resolved": False,
        "filesystem_mutations": 0,
        "development_labels_opened": False,
        "evaluation_labels_opened": False,
        "old_aggregate_utility_surface_used": False,
        "predecessor_policy_cache_output_or_authority_used": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "may_feed_another_experiment": False,
    }
    return {**body, "inspection_hash": canonical_hash(body)}


def dry_run_harp_stage90_v3(
    config: HarpStage90V3Config,
    *,
    artifact_root: str | Path,
    pipeline: HarpV3Pipeline | None = None,
) -> Mapping[str, object]:
    """Mutation-free readiness check; planned mode resolves no path."""

    if type(config) is not HarpStage90V3Config:
        raise ProtocolError("HARP v3 dry run requires a typed configuration.")
    if not config.execution_authorized:
        body = {
            "schema_version": "midogpp_harp_v3_mutation_free_dry_run_v1",
            "status": "NEEDS_SEPARATE_EXECUTION_AMENDMENT",
            "experiment_id": EXPERIMENT_ID,
            "execution_revision": EXECUTION_REVISION,
            "config_hash": config.config_hash,
            "execution_authorized": False,
            "authorization_probed": False,
            "paths_resolved": False,
            "artifact_root_argument_recorded": False,
            "filesystem_mutations": 0,
            "development_labels_opened": False,
            "evaluation_labels_opened": False,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "fresh_evidence": False,
        }
        return {**body, "dry_run_hash": canonical_hash(body)}
    authority = V3_RUNNER_SERVICES.load_authorization(config)
    root = _exact_output_root(config, artifact_root)
    _assert_pristine_output(root)
    parent_hash = _validate_parent_ledger(config)
    cache = V3_RUNNER_SERVICES.load_cache_index(config)
    preflight = dict(_pipeline_or_production(pipeline).preflight(config, cache))
    _validate_preflight(preflight)
    body = {
        "schema_version": "midogpp_harp_v3_mutation_free_dry_run_v1",
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "execution_revision": EXECUTION_REVISION,
        "config_hash": config.config_hash,
        "execution_authorized": True,
        **_authorization_provenance(authority),
        "parent_ledger_sha256": parent_hash,
        "cache_hash": cache.cache_hash,
        "preflight_hash": canonical_hash(preflight),
        "paths_resolved": True,
        "filesystem_mutations": 0,
        "development_labels_opened": False,
        "evaluation_labels_opened": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
    }
    return {**body, "dry_run_hash": canonical_hash(body)}


def run_harp_stage90_v3(
    config: HarpStage90V3Config,
    *,
    artifact_root: str | Path,
    pipeline: HarpV3Pipeline | None = None,
    confirmation_token: str | None = None,
) -> str:
    """Run one separately authorized terminal consumed-test diagnostic."""

    # Keep this check before the typed-config guard and, critically, before any
    # authority probe, path resolution, cache/input access, scratch creation,
    # lease claim, label access, or output mutation.
    if confirmation_token != HARP_V3_RUN_CONFIRMATION_TOKEN:
        raise ProtocolError(
            "HARP v3 execution requires the exact confirmation token "
            f"{HARP_V3_RUN_CONFIRMATION_TOKEN}."
        )
    if type(config) is not HarpStage90V3Config:
        raise ProtocolError("HARP v3 execution requires a typed configuration.")
    # This is deliberately first.  A planned config cannot resolve an output,
    # open a scientific input, create scratch, or mutate any path.
    authority = V3_RUNNER_SERVICES.load_authorization(config)
    root = _exact_output_root(config, artifact_root)
    _assert_pristine_output(root)
    parent_hash = _validate_parent_ledger(config)
    cache = V3_RUNNER_SERVICES.load_cache_index(config)
    active_pipeline = _pipeline_or_production(pipeline)
    preflight = dict(active_pipeline.preflight(config, cache))
    _validate_preflight(preflight)
    admission = {
        "schema_version": "midogpp_harp_v3_execution_admission_v1",
        "experiment_id": EXPERIMENT_ID,
        "execution_revision": EXECUTION_REVISION,
        "config_hash": config.config_hash,
        **_authorization_provenance(authority),
        "parent_ledger_sha256": parent_hash,
        "cache_hash": cache.cache_hash,
        "preflight_hash": canonical_hash(preflight),
        "all_gates_read_only": True,
        "filesystem_mutations_before_lease": 0,
        "launch_confirmation_validated": True,
        "launch_confirmation_token_persisted": False,
        "development_labels_opened": False,
        "evaluation_labels_opened": False,
    }
    admission_hash = canonical_hash(admission)
    lease: Any | None = None
    ledger = PhaseLedger()
    try:
        lease = V3_RUNNER_SERVICES.claim_authorization(
            authority, admission_hash=admission_hash
        )
        ledger.advance("AUTHORITY_ADMISSION")
        _announce("AUTHORITY_ADMISSION")
        admission_path = root / "manifests/admission.json"
        protocol_path = root / "manifests/protocol_manifest.json"
        atomic_json(admission_path, {**admission, "admission_hash": admission_hash})
        atomic_json(
            protocol_path,
            {
                "schema_version": "midogpp_harp_v3_protocol_manifest_v1",
                "experiment_id": EXPERIMENT_ID,
                "execution_revision": EXECUTION_REVISION,
                "config_hash": config.config_hash,
                "protocol": dict(config.protocol),
                "model": dict(config.model),
                "runtime": dict(config.runtime),
                "claim_boundary": dict(config.claim_boundary),
                "utility_kind": "downstream_classifier_utility_not_NELBO",
                "routing_stage_compatibility_estimated": False,
                "generative_expert_compatibility_claimed": False,
                "authorization_lease_hash": lease.lease_hash,
            },
        )

        scratch = _dedicated_scratch(
            config,
            admission_hash=admission_hash,
            authorization_lease_hash=lease.lease_hash,
            root=root,
        )
        journal = LabelFreeProgressJournal(
            root / "manifests/label_free_progress_journal.json", admission_hash
        )
        journal.initialize()
        ledger.advance("LABEL_FREE_PHYSICAL_MENU")
        _announce("LABEL_FREE_PHYSICAL_MENU")
        centers = tuple(str(value) for value in config.protocol["centers"])
        produced = tuple(
            active_pipeline.materialize_label_free_outer_menus(
                config,
                cache,
                outer_targets=centers,
                scratch_root=scratch,
            )
        )
        if tuple(menu.outer_target_id for menu in produced) != centers:
            raise ProtocolError("HARP v3 materializer returned incomplete outer-H coverage.")
        _validate_complete_physical_menus(produced, centers=centers)
        menu_roots: dict[str, Path] = {}
        menu_receipts = []
        for menu in produced:
            store_root = root / "stores/physical_menu" / f"outer_{menu.outer_target_id}"
            receipt = write_label_free_outer_menu(store_root, menu)
            reconstructed = read_label_free_outer_menu(store_root)
            if reconstructed.menu_hash != menu.menu_hash:
                raise ProtocolError("HARP v3 compact physical menu changed identity.")
            journal.record(
                outer_target_id=menu.outer_target_id,
                menu_hash=menu.menu_hash,
                manifest_path=receipt.manifest_path,
                npz_path=receipt.npz_path,
            )
            menu_roots[menu.outer_target_id] = store_root
            menu_receipts.append(receipt)
        development_seal = {
            "schema_version": "midogpp_harp_v3_development_surface_seal_v1",
            "status": "DURABLE_COMPLETE_LABEL_FREE_B_U_HXE_MENU",
            "outer_menu_hashes": {
                menu.outer_target_id: menu.menu_hash for menu in produced
            },
            "outer_menu_manifest_sha256": {
                receipt.root.name.removeprefix("outer_"): receipt.manifest_sha256
                for receipt in menu_receipts
            },
            "strict_outer_center_exclusion": True,
            "physical_expert_weight": 1.0,
            "probability_transport_dtype": "float32",
            "all_action_cells_present_before_label_access": True,
            "labels_consumed": False,
        }
        development_seal = {
            **development_seal,
            "seal_hash": canonical_hash(development_seal),
        }
        development_seal_path = root / "manifests/development_surface_seal.json"
        atomic_json(development_seal_path, development_seal)
        durable_barrier(
            [
                development_seal_path,
                *(receipt.manifest_path for receipt in menu_receipts),
                *(receipt.npz_path for receipt in menu_receipts),
            ]
        )
        ledger.advance("DEVELOPMENT_SURFACE_SEALED")
        _announce("DEVELOPMENT_SURFACE_SEALED")

        ledger.advance("DEVELOPMENT_LABELS_OPENED")
        _announce("DEVELOPMENT_LABELS_OPENED")
        development_labels = V3_RUNNER_SERVICES.load_development_labels(config, cache)
        development = active_pipeline.build_development_case_surface(
            produced, development_labels, config=config
        )
        development = _bind_development_artifact(
            development, config_hash=config.config_hash, centers=centers
        )
        development_receipt = write_artifact_value(
            root / "stores/development_case_surface",
            development,
            role="source_development_case_surface",
        )
        atomic_json(
            root / "reports/development_label_access.json",
            {
                "schema_version": "midogpp_harp_v3_development_label_access_v1",
                "opened_after_development_surface_seal": True,
                "development_surface_seal_hash": development_seal["seal_hash"],
                "evaluation_labels_opened": False,
                "access_count": 1,
            },
        )

        ledger.advance("SOURCE_ONLY_MODEL_FIT")
        _announce("SOURCE_ONLY_MODEL_FIT")
        fitted = active_pipeline.fit_source_only_model(development, config=config)
        fitted = _bind_model_artifact(
            fitted,
            development_surface_hash=str(development.manifest["surface_hash"]),
            config_hash=config.config_hash,
            centers=centers,
        )
        model_receipt = write_artifact_value(
            root / "stores/source_only_model", fitted, role="source_only_model"
        )
        model_lock = {
            "schema_version": "midogpp_harp_v3_model_lock_v1",
            "config_hash": config.config_hash,
            "expected_center_ids": list(centers),
            "development_surface_hash": development.manifest["surface_hash"],
            "model_hash": fitted.manifest["model_hash"],
            "model_store_manifest_sha256": model_receipt.manifest_sha256,
            "model_store_npz_sha256": model_receipt.npz_sha256,
            "scientific_manifest": dict(fitted.manifest),
            "source_development_store_manifest_sha256": development_receipt.manifest_sha256,
            "evaluation_labels_used": False,
        }
        model_lock = {**model_lock, "model_lock_hash": canonical_hash(model_lock)}
        model_lock_path = root / "manifests/model_lock.json"
        atomic_json(model_lock_path, model_lock)

        ledger.advance("TARGET_ACTIONS_COMPLETE")
        _announce("TARGET_ACTIONS_COMPLETE")
        target_actions = active_pipeline.build_complete_target_case_actions(
            produced, fitted, config=config
        )
        target_actions = _bind_target_action_artifact(
            target_actions,
            model_hash=str(fitted.manifest["model_hash"]),
            menu_hashes={menu.outer_target_id: menu.menu_hash for menu in produced},
            config_hash=config.config_hash,
            centers=centers,
        )
        target_receipt = write_artifact_value(
            root / "stores/target_case_actions",
            target_actions,
            role="complete_target_case_actions",
        )
        target_seal = {
            "schema_version": "midogpp_harp_v3_target_action_seal_v1",
            "status": "COMPLETE_B_U_HXE_BEFORE_EVALUATION_LABELS",
            "config_hash": config.config_hash,
            "expected_center_ids": list(centers),
            "model_hash": fitted.manifest["model_hash"],
            "outer_menu_hashes": target_actions.manifest["outer_menu_hashes"],
            "target_store_manifest_sha256": target_receipt.manifest_sha256,
            "target_store_npz_sha256": target_receipt.npz_sha256,
            "target_action_hash": target_actions.manifest["target_action_hash"],
            "physical_expert_weight": 1.0,
            "evaluation_labels_opened": False,
        }
        target_seal = {**target_seal, "seal_hash": canonical_hash(target_seal)}
        target_seal_path = root / "manifests/target_action_seal.json"
        atomic_json(target_seal_path, target_seal)

        routes = active_pipeline.route_case_actions(
            produced, target_actions, fitted, config=config
        )
        _validate_in_memory_route_bindings(
            routes,
            model_hash=str(fitted.manifest["model_hash"]),
            target_action_hash=str(target_actions.manifest["target_action_hash"]),
            centers=centers,
        )
        route_root = root / "stores/prelabel_routes"
        route_receipt = write_prelabel_routes(route_root, routes)
        reconstructed_routes = read_prelabel_routes(route_root)
        if reconstructed_routes.route_hash != routes.route_hash:
            raise ProtocolError("HARP v3 prelabel route store changed identity.")
        route_summary = _prelabel_route_summary(routes)
        rejection_diagnostics = build_prelabel_diagnostics(routes)
        rejection_diagnostics_path = (
            root / "reports/prelabel_rejection_diagnostics.json"
        )
        atomic_json(rejection_diagnostics_path, rejection_diagnostics)
        prelabel = {
            "schema_version": "midogpp_harp_v3_prelabel_route_bundle_v1",
            "status": "DURABLE_CASE_ROUTES_BEFORE_EVALUATION_LABELS",
            "development_surface_seal_hash": development_seal["seal_hash"],
            "config_hash": config.config_hash,
            "expected_center_ids": list(centers),
            "model_lock_hash": model_lock["model_lock_hash"],
            "target_action_seal_hash": target_seal["seal_hash"],
            "route_hash": routes.route_hash,
            "policy_hash": routes.policy_hash,
            "model_hash": routes.model_hash,
            "target_action_hash": routes.target_action_hash,
            "route_store_manifest_sha256": route_receipt.manifest_sha256,
            "route_store_npz_sha256": route_receipt.npz_sha256,
            "route_summary": route_summary,
            "prelabel_rejection_diagnostic_hash": rejection_diagnostics[
                "diagnostic_hash"
            ],
            "case_consistent": True,
            "exact_b_fallback_byte_identity": route_summary[
                "exact_b_fallback_byte_identity"
            ],
            "evaluation_labels_opened": False,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "fresh_evidence": False,
        }
        prelabel = {**prelabel, "bundle_hash": canonical_hash(prelabel)}
        prelabel_path = root / "manifests/prelabel_route_bundle.json"
        atomic_json(prelabel_path, prelabel)
        durable_barrier(
            [
                model_lock_path,
                target_seal_path,
                route_receipt.manifest_path,
                route_receipt.npz_path,
                rejection_diagnostics_path,
                prelabel_path,
            ]
        )
        ledger.advance("PRELABEL_ROUTES_DURABLE")
        _announce("PRELABEL_ROUTES_DURABLE")

        validations = run_two_fresh_validations(
            route_root,
            menu_roots,
            root / "stores/development_case_surface",
            root / "stores/source_only_model",
            root / "stores/target_case_actions",
            expected_center_ids=centers,
            expected_config_hash=config.config_hash,
        )
        validation_bundle = {
            "schema_version": "midogpp_harp_v3_fresh_validation_bundle_v1",
            "config_hash": config.config_hash,
            "expected_center_ids": list(centers),
            "validations": list(validations),
            "distinct_process_ids": len(
                {value["process_id"] for value in validations}
            )
            == 2,
            "evaluation_labels_opened": False,
        }
        validation_bundle = {
            **validation_bundle,
            "bundle_hash": canonical_hash(validation_bundle),
        }
        validation_bundle_path = root / "manifests/fresh_validations.json"
        atomic_json(validation_bundle_path, validation_bundle)
        ledger.advance("FRESH_RECONSTRUCTIONS_COMPLETE")
        _announce("FRESH_RECONSTRUCTIONS_COMPLETE")

        frozen = {
            "schema_version": "midogpp_harp_v3_frozen_route_seal_v1",
            "status": "FROZEN_AFTER_TWO_FRESH_RECONSTRUCTIONS",
            "prelabel_bundle_hash": prelabel["bundle_hash"],
            "config_hash": config.config_hash,
            "expected_center_ids": list(centers),
            "route_hash": routes.route_hash,
            "policy_hash": routes.policy_hash,
            "model_hash": routes.model_hash,
            "target_action_hash": routes.target_action_hash,
            "validation_bundle_hash": validation_bundle["bundle_hash"],
            "independent_validation_hashes": [
                value["validation_hash"] for value in validations
            ],
            "case_count": len(routes.cases),
            "exact_b_fallback_byte_identity": route_summary[
                "exact_b_fallback_byte_identity"
            ],
            "evaluation_labels_opened": False,
        }
        frozen = {**frozen, "seal_hash": canonical_hash(frozen)}
        frozen_path = root / "manifests/frozen_route_seal.json"
        atomic_json(frozen_path, frozen)
        durable_barrier([validation_bundle_path, frozen_path])
        if read_json(frozen_path) != frozen:
            raise ProtocolError("HARP v3 frozen route seal failed a fresh read.")
        ledger.advance("FROZEN_ROUTE_SEAL")
        _announce("FROZEN_ROUTE_SEAL")

        sealed_routes = _reconstruct_frozen_routes_for_evaluation(
            route_root,
            frozen=frozen,
            model_hash=str(fitted.manifest["model_hash"]),
            target_action_hash=str(target_actions.manifest["target_action_hash"]),
            centers=centers,
            config_hash=config.config_hash,
        )
        ledger.advance("EVALUATION_LABELS_OPENED")
        _announce("EVALUATION_LABELS_OPENED")
        evaluation_truth = V3_RUNNER_SERVICES.load_evaluation_truth(config, cache)
        terminal = active_pipeline.evaluate_terminal(
            sealed_routes, evaluation_truth, config=config
        )
        terminal_metrics = dict(terminal.metrics)
        terminal_metrics.pop("result_hash", None)
        terminal_metrics.update(
            {
                "utility_kind": "downstream_classifier_utility_not_NELBO",
                "routing_stage_compatibility_estimated": False,
                "generative_expert_compatibility_claimed": False,
            }
        )
        terminal_metrics["result_hash"] = canonical_hash(terminal_metrics)
        terminal_result_path = root / "reports/terminal_result.json"
        action_oracle_path = root / "reports/action_oracle_diagnostics.json"
        route_reasons_path = root / "reports/route_and_fallback_reasons.json"
        evaluation_access_path = root / "reports/evaluation_label_access.json"
        leakage_path = root / "reports/leakage_report.json"
        validation_report_path = root / "reports/validation_report.json"
        atomic_json(terminal_result_path, terminal_metrics)
        atomic_json(
            action_oracle_path,
            dict(terminal.oracle_diagnostic),
        )
        atomic_json(
            route_reasons_path,
            dict(terminal.route_reasons),
        )
        atomic_json(
            evaluation_access_path,
            {
                "schema_version": "midogpp_harp_v3_evaluation_label_access_v1",
                "opened_after_frozen_route_seal": True,
                "frozen_route_seal_hash": frozen["seal_hash"],
                "reconstructed_route_hash": sealed_routes.route_hash,
                "route_store_reopened_after_frozen_seal": True,
                "access_count": 1,
            },
        )
        atomic_json(leakage_path, _leakage_report())
        validation_report = {
            "schema_version": "midogpp_harp_v3_validation_report_v1",
            "status": "PASS",
            "development_surface_seal_hash": development_seal["seal_hash"],
            "model_lock_hash": model_lock["model_lock_hash"],
            "target_action_seal_hash": target_seal["seal_hash"],
            "frozen_route_seal_hash": frozen["seal_hash"],
            "independent_validation_hashes": [
                value["validation_hash"] for value in validations
            ],
            "terminal_result_hash": terminal_metrics["result_hash"],
            "action_oracle_diagnostic_hash": terminal.oracle_diagnostic[
                "diagnostic_hash"
            ],
            "exact_b_fallback_byte_identity": route_summary[
                "exact_b_fallback_byte_identity"
            ],
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "fresh_evidence": False,
        }
        atomic_json(validation_report_path, validation_report)
        terminal_paths = (
            terminal_result_path,
            action_oracle_path,
            route_reasons_path,
            evaluation_access_path,
            leakage_path,
            validation_report_path,
        )
        durable_barrier(terminal_paths)
        ledger.advance("TERMINAL_DIAGNOSTIC_COMPLETE")
        _announce("TERMINAL_DIAGNOSTIC_COMPLETE")

        lease = V3_RUNNER_SERVICES.finalize_authorization(
            lease, status="COMPLETE_EXHAUSTED"
        )
        finalization = read_json(lease.root / "lease.json")
        if finalization.get("status") != "COMPLETE_EXHAUSTED":
            raise ProtocolError("HARP v3 authorization did not finalize.")
        finalization_path = root / "manifests/authorization_finalization.json"
        atomic_json(finalization_path, finalization)
        durable_barrier((finalization_path,))
        content_index_path = _write_content_index(root)
        _validate_content_index(root, content_index_path)
        run_state_path = root / "reports/run_state.json"
        run_state = {
            "schema_version": "midogpp_harp_v3_run_state_v1",
            "status": "COMPLETE_EXHAUSTED",
            "phase": "TERMINAL_DIAGNOSTIC_COMPLETE",
            "phase_order": list(ledger.observed),
            "authorization_lease_hash": lease.lease_hash,
            "authorization_finalization_sha256": sha256_file(finalization_path),
            "content_index_sha256": sha256_file(content_index_path),
            "terminal_member_sha256": {
                path.relative_to(root).as_posix(): sha256_file(path)
                for path in terminal_paths
            },
            "frozen_route_seal_hash": frozen["seal_hash"],
            "evaluated_reconstructed_route_hash": sealed_routes.route_hash,
            "terminal_result_hash": terminal_metrics["result_hash"],
            "scratch_root_used": str(scratch),
            "completion_commit_protocol": "fsync_files_then_atomic_marker_then_fsync_directories",
            "final_commit": True,
        }
        _commit_completion_state(
            root,
            run_state_path,
            run_state,
            durable_members=(
                *terminal_paths,
                finalization_path,
                content_index_path,
            ),
        )
        return str(root)
    except BaseException as exc:
        if lease is not None:
            try:
                atomic_json(
                    root / "reports/failure_report.json",
                    {
                        "schema_version": "midogpp_harp_v3_failure_report_v1",
                        "status": "FAILED_EXHAUSTED",
                        "phase_order": list(ledger.observed),
                        "error_class": exc.__class__.__name__,
                        "error": str(exc)[:2000],
                        "publication_status": PUBLICATION_STATUS,
                        "terminal_decision": TERMINAL_DECISION,
                    },
                )
                V3_RUNNER_SERVICES.finalize_authorization(
                    lease, status="FAILED_EXHAUSTED", error=str(exc)
                )
            except BaseException:
                pass
        raise


def _bind_development_artifact(
    value: ArtifactValue,
    *,
    config_hash: str,
    centers: tuple[str, ...],
) -> ArtifactValue:
    if not isinstance(value, ArtifactValue):
        raise ProtocolError("HARP v3 development surface is not a typed artifact.")
    body = dict(value.manifest)
    body.pop("surface_hash", None)
    if tuple(body.get("outer_targets", ())) != centers:
        raise ProtocolError("HARP v3 development surface target universe drifted.")
    for key, expected in (
        ("config_hash", config_hash),
        ("expected_center_ids", list(centers)),
    ):
        if key in body and body[key] != expected:
            raise ProtocolError("HARP v3 development identity binding conflicted.")
        body[key] = expected
    return ArtifactValue(
        state=value.state,
        manifest={**body, "surface_hash": canonical_hash(body)},
        arrays=value.arrays,
    )


def _bind_model_artifact(
    value: ArtifactValue,
    *,
    development_surface_hash: str,
    config_hash: str,
    centers: tuple[str, ...],
) -> ArtifactValue:
    if not isinstance(value, ArtifactValue):
        raise ProtocolError("HARP v3 source-only model is not a typed artifact.")
    body = dict(value.manifest)
    body.pop("model_hash", None)
    if body.get("development_surface_hash") != development_surface_hash:
        raise ProtocolError("HARP v3 fitted model escaped its development surface.")
    for key, expected in (
        ("config_hash", config_hash),
        ("expected_center_ids", list(centers)),
    ):
        if key in body and body[key] != expected:
            raise ProtocolError("HARP v3 model identity binding conflicted.")
        body[key] = expected
    return ArtifactValue(
        state=value.state,
        manifest={**body, "model_hash": canonical_hash(body)},
        arrays=value.arrays,
    )


def _bind_target_action_artifact(
    value: ArtifactValue,
    *,
    model_hash: str,
    menu_hashes: Mapping[str, str],
    config_hash: str,
    centers: tuple[str, ...],
) -> ArtifactValue:
    if not isinstance(value, ArtifactValue):
        raise ProtocolError("HARP v3 target action surface is not a typed artifact.")
    body = dict(value.manifest)
    body.pop("target_action_hash", None)
    bindings: tuple[tuple[str, object], ...] = (
        ("config_hash", config_hash),
        ("expected_center_ids", list(centers)),
        ("model_hash", model_hash),
        ("outer_menu_hashes", dict(sorted(menu_hashes.items()))),
    )
    for key, expected in bindings:
        if key in body and body[key] != expected:
            raise ProtocolError("HARP v3 target action identity binding conflicted.")
        body[key] = expected
    return ArtifactValue(
        state=value.state,
        manifest={**body, "target_action_hash": canonical_hash(body)},
        arrays=value.arrays,
    )


def _validate_in_memory_route_bindings(
    routes: PrelabelRouteSet,
    *,
    model_hash: str,
    target_action_hash: str,
    centers: tuple[str, ...],
) -> None:
    if not isinstance(routes, PrelabelRouteSet):
        raise ProtocolError("HARP v3 routing returned an untyped route set.")
    if (
        routes.model_hash != model_hash
        or routes.target_action_hash != target_action_hash
        or tuple(sorted({case.outer_target_id for case in routes.cases})) != centers
    ):
        raise ProtocolError("HARP v3 in-memory route cross-binding drifted.")


def _reconstruct_frozen_routes_for_evaluation(
    route_root: Path,
    *,
    frozen: Mapping[str, object],
    model_hash: str,
    target_action_hash: str,
    centers: tuple[str, ...],
    config_hash: str,
) -> PrelabelRouteSet:
    if (
        frozen.get("status") != "FROZEN_AFTER_TWO_FRESH_RECONSTRUCTIONS"
        or frozen.get("config_hash") != config_hash
        or tuple(frozen.get("expected_center_ids", ())) != centers
        or frozen.get("model_hash") != model_hash
        or frozen.get("target_action_hash") != target_action_hash
    ):
        raise ProtocolError("HARP v3 frozen route identity binding drifted.")
    routes = read_prelabel_routes(route_root)
    if (
        routes.route_hash != frozen.get("route_hash")
        or routes.policy_hash != frozen.get("policy_hash")
        or routes.model_hash != model_hash
        or routes.target_action_hash != target_action_hash
        or len(routes.cases) != frozen.get("case_count")
        or tuple(sorted({case.outer_target_id for case in routes.cases})) != centers
    ):
        raise ProtocolError("HARP v3 disk-reconstructed frozen routes drifted.")
    return routes


def _validate_content_index(root: Path, path: Path) -> None:
    payload = read_json(path)
    stored_hash = payload.get("content_index_hash")
    body = {key: value for key, value in payload.items() if key != "content_index_hash"}
    raw_members = payload.get("members")
    if (
        payload.get("schema_version") != "midogpp_harp_v3_content_index_v1"
        or stored_hash != canonical_hash(body)
        or not isinstance(raw_members, list)
    ):
        raise ProtocolError("HARP v3 content index self-binding drifted.")
    observed: list[dict[str, str]] = []
    excluded = {"manifests/content_index.json", "reports/run_state.json"}
    for member in sorted(root.rglob("*")):
        if member.is_symlink():
            raise ProtocolError("HARP v3 content index encountered a symlink.")
        relative = member.relative_to(root).as_posix()
        if member.is_file() and relative not in excluded:
            observed.append({"path": relative, "sha256": sha256_file(member)})
    if raw_members != observed:
        raise ProtocolError("HARP v3 content index member coverage drifted.")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ProtocolError("HARP v3 completion directory fsync failed.") from exc


def _commit_completion_state(
    root: Path,
    run_state_path: Path,
    payload: Mapping[str, object],
    *,
    durable_members: tuple[Path, ...],
) -> None:
    """Commit COMPLETE only after every indexed terminal member is durable."""

    if run_state_path.exists() or run_state_path.is_symlink():
        raise ProtocolError("HARP v3 completion marker already exists.")
    pending = run_state_path.with_name(f".run_state.{os.getpid()}.pending.json")
    if pending.exists() or pending.is_symlink():
        raise ProtocolError("HARP v3 completion transaction has stale pending state.")
    atomic_json(pending, payload)
    try:
        if read_json(pending) != dict(payload):
            raise ProtocolError("HARP v3 pending completion state changed bytes.")
        durable_barrier((*durable_members, pending))
        os.replace(pending, run_state_path)
        # The first barrier makes the inode and all dependencies durable; the
        # atomic rename is the commit point.  The second barrier and directory
        # fsyncs make that commit marker itself durable.
        durable_barrier((*durable_members, run_state_path))
        _fsync_directory(run_state_path.parent)
        _fsync_directory(root / "manifests")
        _fsync_directory(root)
    except BaseException:
        pending.unlink(missing_ok=True)
        if run_state_path.exists() and not run_state_path.is_symlink():
            run_state_path.unlink(missing_ok=True)
            _fsync_directory(run_state_path.parent)
        raise


def _validate_preflight(value: Mapping[str, object]) -> None:
    expected = {
        "status": "PASS",
        "persistent_gpu_workers": 2,
        "gpu_devices": ["cuda:0", "cuda:1"],
        "probability_transport_dtype": "float32",
        "scientific_reduction_dtype": "float64",
        "physical_expert_weight": 1.0,
        "tf32_enabled": False,
        "amp_enabled": False,
        "parent_cuda_context_created": False,
        "shared_validated_menu_index": True,
        "labels_consumed": False,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ProtocolError("HARP v3 workstation preflight contract drifted.")


def _exact_output_root(config: HarpStage90V3Config, value: str | Path) -> Path:
    text = str(value)
    if "://" in text:
        raise ProtocolError("HARP v3 runner requires a workspace-resolved output path.")
    root = Path(text).resolve()
    if "://" not in config.artifact_root and Path(config.artifact_root).resolve() != root:
        raise ProtocolError("HARP v3 CLI/config output roots differ.")
    return root


def _assert_pristine_output(root: Path) -> None:
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ProtocolError("HARP v3 prepared output root is absent or unsafe.")
    allowed = {"config.resolved.yaml", "provenance/input_artifacts.json"}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ProtocolError("HARP v3 prepared output contains a symlink.")
        if path.is_file() and path.relative_to(root).as_posix() not in allowed:
            raise ProtocolError("HARP v3 output contains prior scientific state.")


def _validate_parent_ledger(config: HarpStage90V3Config) -> str:
    expected = config.expected_hashes["parent_ledger_sha256"]
    if type(expected) is not str:
        raise ProtocolError("HARP v3 parent ledger hash is absent.")
    path = config.resolved_path("parent_ledger_path")
    if sha256_file(path) != expected:
        raise ProtocolError("HARP v3 parent ledger bytes drifted.")
    read_json(path)
    return expected


def _dedicated_scratch(
    config: HarpStage90V3Config,
    *,
    admission_hash: str,
    authorization_lease_hash: str,
    root: Path,
) -> Path:
    configured = Path(str(config.runtime["scratch_root"]))
    if not configured.is_absolute() or configured == root or configured.is_symlink():
        raise ProtocolError("HARP v3 dedicated scratch root is unsafe.")
    scratch = configured / admission_hash[:20]
    receipt = scratch / "scratch_binding.json"
    binding = {
        "schema_version": "midogpp_harp_v3_scratch_binding_v1",
        "admission_hash": admission_hash,
        "authorization_lease_hash": authorization_lease_hash,
        "process_id": os.getpid(),
        "config_hash": config.config_hash,
        "label_free_resumption_only": True,
        "development_or_evaluation_labels_stored": False,
    }
    if scratch.exists() or scratch.is_symlink():
        if (
            scratch.is_symlink()
            or not scratch.is_dir()
            or not receipt.is_file()
            or receipt.is_symlink()
            or read_json(receipt) != binding
        ):
            raise ProtocolError(
                "HARP v3 pre-existing scratch is not bound to this live lease."
            )
    else:
        configured.mkdir(parents=True, exist_ok=True)
        if configured.is_symlink() or not configured.is_dir():
            raise ProtocolError("HARP v3 dedicated scratch parent is unsafe.")
        scratch.mkdir(mode=0o700, parents=False, exist_ok=False)
        atomic_json(receipt, binding)
    return scratch


def _authorization_provenance(value: object) -> dict[str, object]:
    required = (
        "amendment_sha256",
        "amendment_hash",
        "input_binding_hash",
        "scientific_contract_hash",
        "workspace_registration_execution_contract_hash",
        "source_snapshot_schema",
        "source_snapshot_manifest_sha256",
        "source_snapshot_tree_sha256",
        "source_snapshot_member_count",
    )
    if any(not hasattr(value, name) for name in required):
        raise ProtocolError("HARP v3 authorization provenance is untyped.")
    return {
        "execution_amendment_sha256": value.amendment_sha256,
        "execution_amendment_hash": value.amendment_hash,
        "authorized_input_binding_hash": value.input_binding_hash,
        "scientific_contract_hash": value.scientific_contract_hash,
        "workspace_registration_execution_contract_hash": (
            value.workspace_registration_execution_contract_hash
        ),
        "source_snapshot_schema": value.source_snapshot_schema,
        "source_snapshot_manifest_sha256": value.source_snapshot_manifest_sha256,
        "source_snapshot_tree_sha256": value.source_snapshot_tree_sha256,
        "source_snapshot_member_count": value.source_snapshot_member_count,
    }


def _prelabel_route_summary(routes: object) -> dict[str, object]:
    cases = tuple(getattr(routes, "cases"))
    action_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    exact = True
    for case in cases:
        action_counts[case.selected_kind.value] = action_counts.get(case.selected_kind.value, 0) + 1
        reason_counts[case.reason] = reason_counts.get(case.reason, 0) + 1
        if case.selected_kind.value == "B":
            exact &= case.routed_probabilities.tobytes(order="C") == case.baseline_probabilities.tobytes(order="C")
    return {
        "case_count": len(cases),
        "row_count": sum(len(case.sample_ids) for case in cases),
        "selected_action_counts": dict(sorted(action_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "case_consistent": True,
        "exact_b_fallback_byte_identity": exact,
    }


def _leakage_report() -> dict[str, object]:
    return {
        "schema_version": "midogpp_harp_v3_leakage_report_v1",
        "strict_outer_center_exclusion": True,
        "development_evaluation_case_disjoint": True,
        "development_labels_opened_after_label_free_menu_seal": True,
        "evaluation_labels_opened_after_two_fresh_reconstructions_and_frozen_seal": True,
        "source_lodo_calibration_only": True,
        "old_aggregate_utility_surface_used": False,
        "predecessor_policy_rank_output_cache_or_authority_used": False,
        "probability_blends_primary": False,
        "physical_expert_weight": 1.0,
        "terminal_oracle_may_feed_policy_or_thresholds": False,
        "utility_kind": "downstream_classifier_utility_not_NELBO",
        "routing_stage_compatibility_estimated": False,
        "generative_expert_compatibility_claimed": False,
        "status": "PASS",
    }


def _validate_complete_physical_menus(
    menus: tuple[object, ...], *, centers: tuple[str, ...]
) -> None:
    """Reject a partial B/U/Hxe surface before development labels can open."""

    for menu in menus:
        outer = str(menu.outer_target_id)
        by_context: dict[tuple[str, str], list[object]] = {}
        for block in menu.blocks:
            by_context.setdefault((block.surface_role, block.query_center_id), []).append(block)
        expected_contexts = {
            *(('development', query) for query in centers if query != outer),
            ('target', outer),
        }
        if set(by_context) != expected_contexts:
            raise ProtocolError("HARP v3 physical menu lacks a sealed query context.")
        for (role, query), blocks in by_context.items():
            baseline = [row for row in blocks if row.action_kind.value == "B"]
            uniform = [row for row in blocks if row.action_kind.value == "U"]
            experts = sorted(
                (
                    row for row in blocks if row.action_kind.value == "Hxe"
                ),
                key=lambda row: row.selected_source_id or "",
            )
            expected_sources = tuple(
                center
                for center in centers
                if center != outer and (role == "target" or center != query)
            )
            if (
                len(baseline) != 1
                or len(uniform) != 1
                or tuple(row.selected_source_id for row in experts) != expected_sources
                or any(
                    row.sample_ids != baseline[0].sample_ids
                    or row.case_ids != baseline[0].case_ids
                    for row in (uniform[0], *experts)
                )
            ):
                raise ProtocolError(
                    "HARP v3 physical context lacks exact B/U/all legal Hxe actions."
                )


def _write_content_index(root: Path) -> Path:
    members = []
    excluded = {"manifests/content_index.json", "reports/run_state.json"}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.relative_to(root).as_posix() not in excluded:
            members.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
    body = {
        "schema_version": "midogpp_harp_v3_content_index_v1",
        "members": members,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "may_feed_another_experiment": False,
        "run_state_excluded_as_final_commit": True,
    }
    path = root / "manifests/content_index.json"
    atomic_json(path, {**body, "content_index_hash": canonical_hash(body)})
    return path


def _announce(phase: str) -> None:
    print(f"[harp-v3] phase={phase}", file=sys.stderr, flush=True)


__all__ = (
    "IMPLEMENTED_COMPONENTS",
    "HarpV3RunnerServices",
    "HARP_V3_RUN_CONFIRMATION_TOKEN",
    "V3_RUNNER_SERVICES",
    "dry_run_harp_stage90_v3",
    "inspect_harp_stage90_v3",
    "run_harp_stage90_v3",
)
