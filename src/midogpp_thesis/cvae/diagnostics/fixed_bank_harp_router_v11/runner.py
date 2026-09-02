"""Protocol-ordered, workstation-optimized HARP v11 production runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import atomic_json, read_json
from ...runtime.harp_v11_execution.contracts import HarpV11Pipeline
from ...runtime.harp_v11_execution.action_capacity import (
    build_action_capacity_certificate,
    validate_action_capacity_certificate,
)
from ...runtime.harp_v11_execution.durability import durable_barrier
from ...runtime.harp_v11_execution.effective_menu_artifacts import (
    build_effective_menu_artifact,
)
from ...runtime.harp_v11_execution.hash_contracts import runtime_hash_contract_payload
from ...runtime.harp_v11_execution.journal import LabelFreeProgressJournal
from ...runtime.harp_v11_execution.phases import PHASE_ORDER, PhaseLedger
from ...runtime.harp_v11_execution.prelabel_diagnostics import (
    build_prelabel_diagnostics,
)
from ...runtime.harp_v11_execution.stores import (
    read_prelabel_routes,
    write_artifact_value,
    write_label_free_outer_menu,
    write_prelabel_routes,
)
from ...runtime.harp_v11_execution.validation import run_two_fresh_validations
from ....workspace.preparation_authority import HARP_V11_RUN_CONFIRMATION_TOKEN
from .authorization import (
    HarpV11Authorization,
    HarpV11AuthorizationLease,
    claim_authorization,
    finalize_authorization,
    load_authorization,
)
from .config import HarpStage90V11Config
from .execution import (
    authorization_provenance as _authorization_provenance,
    bind_admission_artifact as _bind_admission_artifact,
    bind_development_artifact as _bind_development_artifact,
    bind_model_artifact as _bind_model_artifact,
    bind_target_action_artifact as _bind_target_action_artifact,
    commit_completion_state as _commit_completion_state,
    dedicated_scratch as _dedicated_scratch,
    exact_output_root as _exact_output_root,
    prelabel_route_summary as _prelabel_route_summary,
    reconstruct_frozen_routes_for_evaluation as _reconstruct_frozen_routes_for_evaluation,
    validate_content_index as _validate_content_index,
    validate_in_memory_route_bindings as _validate_in_memory_route_bindings,
    validate_parent_ledger as _validate_parent_ledger,
    validate_preflight as _validate_preflight,
    validate_pristine_or_label_free_recovery as _validate_output_state,
    write_content_index as _write_content_index,
    write_terminal_reports,
)
from .identity import (
    EXECUTION_REVISION,
    EXPERIMENT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from .input_surfaces import (
    DEVELOPMENT_ROLE,
    EVALUATION_ROLE,
    load_cache_index,
    load_development_labels,
    load_evaluation_truth,
)
from .runner_payloads import (
    build_development_surface_seal,
    build_frozen_route_seal,
    build_policy_admission_seal,
    build_model_lock,
    build_prelabel_bundle,
    build_run_state,
    build_target_action_seal,
    build_validation_bundle,
)
from .runner_recovery import (
    recover_or_materialize_compatibility,
    recover_or_materialize_label_free_menus,
)
from .source_crossfit_orchestration import (
    assemble_source_crossfit_model,
    build_source_crossfit_effective_artifact,
    build_source_prelabel_prediction_artifact,
    fit_and_seal_prelabel_source_folds,
    materialize_label_free_source_crossfit,
    source_fold_capability_seal_payload,
    source_prelabel_q_prediction_seal_payload,
)


IMPLEMENTED_COMPONENTS = (
    "separate_single_use_v11_authority_before_all_output_or_input_mutation",
    "exact_launch_confirmation_before_authority_input_path_or_output_access",
    "catalog_bound_crash_recoverable_v11_input_preparation_before_activation",
    "mutation_free_activation_plan_with_resumable_amendment_and_registry_last_commit",
    "v11_owned_label_blind_cache_and_physical_generation_lineage",
    "two_persistent_gpu_source_workers_closed_before_cpu_classifier_pool",
    "globally_single_threaded_parent_with_two_persistent_gpu_workers",
    "four_explicit_single_process_classifier_executors_with_two_inflight_per_worker",
    "phase_disjoint_four_by_three_classifier_and_four_by_one_science_pools",
    "cache_row_representation_protocol_bound_source_frame_memmaps",
    "same_shape_foreign_or_unreceipted_source_frames_rejected",
    "typed_semantic16_and_sha256_digest_roles_at_all_runtime_boundaries",
    "strict_recursive_plain_json_projection_for_immutable_runtime_mappings",
    "runtime_hash_contract_bound_into_inspection_dry_run_admission_and_protocol_manifest",
    "grouped_once_per_shard_consumed_frame_staging",
    "complete_physical_lambda_one_B_U_Hxe_probability_menu",
    "strict_outer_H_query_candidate_exclusion",
    "float32_probability_transport_and_float64_scientific_reductions",
    "deterministic_compact_npz_stores_with_chunk_hashes",
    "shared_source_and_target_label_free_effective_menu_before_any_labels",
    "all_margins_structural_noops_and_byte_duplicate_aliases_removed",
    "empty_active_source_cases_retained_as_exact_B_controls",
    "budget_U_minus_B_and_allocation_Hxe_minus_U_residual_heads",
    "strict_source_center_lodo_and_candidate_identity_exclusion",
    "center_case_balanced_tie_aware_pairwise_ranker_with_virtual_exact_B",
    "cross_fitted_ranker_selected_action_acceptor",
    "numeric_oof_action_scores_acceptance_and_policy_replays_persisted",
    "rank_all_active_actions_before_selected_action_acceptance",
    "per_action_worst_center_certificate_removed",
    "per_outer_rank_and_policy_admission_without_global_kill_switch",
    "nested_whole_policy_source_oof_risk_coverage_calibration",
    "policy_accepted_exact_top1_physical_action_or_exact_B_without_probability_mixtures",
    "byte_identical_exact_B_case_fallback",
    "label_free_preterminal_rejection_diagnostics",
    "two_distinct_spawn_process_route_reconstructions",
    "external_config_center_and_all_candidate_cross_binding",
    "disk_reconstructed_frozen_routes_are_the_only_terminal_input",
    "label_free_evaluation_release_descriptor_is_the_seventh_direct_input",
    "canonical_evaluation_truth_reopened_only_by_typed_frozen_route_receipt",
    "single_content_index_and_atomic_fsync_completion_commit",
    "mutation_free_live_cpu_ram_gpu_scratch_and_dependency_preflight",
    "complete_prelease_294_row_action_capacity_certificate",
)


@dataclass(frozen=True, slots=True)
class HarpV11RunnerServices:
    config_type: type
    authorization_type: type
    lease_type: type
    load_authorization: Callable[[Any], Any]
    claim_authorization: Callable[..., Any]
    finalize_authorization: Callable[..., Any]
    load_cache_index: Callable[[Any], Any]
    load_development_labels: Callable[[Any, Any], object]
    load_evaluation_truth: Callable[[Any, Any, Any], object]


V11_RUNNER_SERVICES = HarpV11RunnerServices(
    config_type=HarpStage90V11Config,
    authorization_type=HarpV11Authorization,
    lease_type=HarpV11AuthorizationLease,
    load_authorization=load_authorization,
    claim_authorization=claim_authorization,
    finalize_authorization=finalize_authorization,
    load_cache_index=load_cache_index,
    load_development_labels=load_development_labels,
    load_evaluation_truth=load_evaluation_truth,
)


def _pipeline_or_production(pipeline: HarpV11Pipeline | None) -> HarpV11Pipeline:
    if pipeline is not None:
        return pipeline
    # Keep planned inspection and mutation-free lifecycle validation independent
    # from the optional production numerics.  This import is reached only after
    # an activated run needs a concrete pipeline.
    from ...runtime.harp_v11_execution.production import HarpV11ProductionPipeline

    return HarpV11ProductionPipeline(
        development_role=DEVELOPMENT_ROLE, evaluation_role=EVALUATION_ROLE
    )


def _capacity_certificate(config: HarpStage90V11Config) -> dict[str, object]:
    """Reconstruct and cross-bind the path-free v11 action-capacity proof."""

    centers = tuple(str(value) for value in config.protocol["centers"])
    certificate = dict(build_action_capacity_certificate(centers=centers))
    validate_action_capacity_certificate(certificate, centers=centers)
    maxima = certificate.get("required_rows_per_class_by_surface")
    expected = {
        "target": config.protocol[
            "target_max_required_rows_per_source_per_class"
        ],
        "source_prediction_seven_source": config.protocol[
            "seven_source_max_required_rows_per_source_per_class"
        ],
        "source_calibration_six_source": config.protocol[
            "six_source_max_required_rows_per_source_per_class"
        ],
    }
    if (
        certificate.get("stream_rows_per_class")
        != config.protocol["resident_source_rows_per_class"]
        or maxima != expected
        or certificate.get("global_maximum_required_rows_per_class")
        != config.protocol["global_max_required_rows_per_source_per_class"]
        or config.protocol["prelease_action_capacity_certificate_required"]
        is not True
    ):
        raise ProtocolError("HARP v11 config/capacity certificate binding drifted.")
    return certificate


def inspect_harp_stage90_v11(config: HarpStage90V11Config) -> Mapping[str, object]:
    """Path-free, mutation-free inspection of planned or activated identity."""

    if type(config) is not HarpStage90V11Config:
        raise ProtocolError("HARP v11 inspection requires a typed configuration.")
    runtime_hash_contract = dict(runtime_hash_contract_payload())
    capacity = _capacity_certificate(config)
    body = {
        "schema_version": "midogpp_harp_v11_implementation_inspection_v1",
        "status": (
            "EXECUTABLE_AUTHORIZED_UNPROBED"
            if config.execution_authorized
            else "PLANNED_NEEDS_SEPARATE_EXECUTION_AMENDMENT"
        ),
        "experiment_id": EXPERIMENT_ID,
        "execution_revision": EXECUTION_REVISION,
        "config_hash": config.config_hash,
        "runtime_hash_contract_hash": runtime_hash_contract[
            "runtime_hash_contract_hash"
        ],
        "action_capacity_certificate_hash": capacity[
            "capacity_certificate_hash"
        ],
        "resident_source_rows_per_class": capacity["stream_rows_per_class"],
        "global_maximum_required_rows_per_class": capacity[
            "global_maximum_required_rows_per_class"
        ],
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


def dry_run_harp_stage90_v11(
    config: HarpStage90V11Config,
    *,
    artifact_root: str | Path,
    pipeline: HarpV11Pipeline | None = None,
) -> Mapping[str, object]:
    """Mutation-free readiness check; planned mode resolves no path."""

    if type(config) is not HarpStage90V11Config:
        raise ProtocolError("HARP v11 dry run requires a typed configuration.")
    runtime_hash_contract = dict(runtime_hash_contract_payload())
    capacity = _capacity_certificate(config)
    if not config.execution_authorized:
        body = {
            "schema_version": "midogpp_harp_v11_mutation_free_dry_run_v1",
            "status": "NEEDS_SEPARATE_EXECUTION_AMENDMENT",
            "experiment_id": EXPERIMENT_ID,
            "execution_revision": EXECUTION_REVISION,
            "config_hash": config.config_hash,
            "runtime_hash_contract_hash": runtime_hash_contract[
                "runtime_hash_contract_hash"
            ],
            "action_capacity_certificate_hash": capacity[
                "capacity_certificate_hash"
            ],
            "resident_source_rows_per_class": capacity["stream_rows_per_class"],
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
    authority = V11_RUNNER_SERVICES.load_authorization(config)
    root = _exact_output_root(config, artifact_root)
    parent_hash = _validate_parent_ledger(config)
    cache = V11_RUNNER_SERVICES.load_cache_index(config)
    preflight = dict(_pipeline_or_production(pipeline).preflight(config, cache))
    _validate_preflight(preflight)
    if preflight.get("action_capacity_certificate") != capacity:
        raise ProtocolError("HARP v11 preflight capacity certificate drifted.")
    body = {
        "schema_version": "midogpp_harp_v11_mutation_free_dry_run_v1",
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "execution_revision": EXECUTION_REVISION,
        "config_hash": config.config_hash,
        "runtime_hash_contract_hash": runtime_hash_contract[
            "runtime_hash_contract_hash"
        ],
        "execution_authorized": True,
        **_authorization_provenance(authority),
        "parent_ledger_sha256": parent_hash,
        "cache_hash": cache.cache_hash,
        "preflight_hash": _stable_preflight_hash(preflight),
        "action_capacity_certificate_hash": capacity[
            "capacity_certificate_hash"
        ],
        "paths_resolved": True,
        "filesystem_mutations": 0,
        "development_labels_opened": False,
        "evaluation_labels_opened": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
    }
    return {**body, "dry_run_hash": canonical_hash(body)}


def run_harp_stage90_v11(
    config: HarpStage90V11Config,
    *,
    artifact_root: str | Path,
    pipeline: HarpV11Pipeline | None = None,
    confirmation_token: str | None = None,
) -> str:
    """Run one separately authorized terminal consumed-test diagnostic."""

    # Keep this check before the typed-config guard and, critically, before any
    # authority probe, path resolution, cache/input access, scratch creation,
    # lease claim, label access, or output mutation.
    if confirmation_token != HARP_V11_RUN_CONFIRMATION_TOKEN:
        raise ProtocolError(
            "HARP v11 execution requires the exact confirmation token "
            f"{HARP_V11_RUN_CONFIRMATION_TOKEN}."
        )
    if type(config) is not HarpStage90V11Config:
        raise ProtocolError("HARP v11 execution requires a typed configuration.")
    runtime_hash_contract = dict(runtime_hash_contract_payload())
    capacity = _capacity_certificate(config)
    # This is deliberately first.  A planned config cannot resolve an output,
    # open a scientific input, create scratch, or mutate any path.
    authority = V11_RUNNER_SERVICES.load_authorization(config)
    root = _exact_output_root(config, artifact_root)
    parent_hash = _validate_parent_ledger(config)
    cache = V11_RUNNER_SERVICES.load_cache_index(config)
    active_pipeline = _pipeline_or_production(pipeline)
    preflight = dict(active_pipeline.preflight(config, cache))
    _validate_preflight(preflight)
    if preflight.get("action_capacity_certificate") != capacity:
        raise ProtocolError("HARP v11 preflight capacity certificate drifted.")
    admission = {
        "schema_version": "midogpp_harp_v11_execution_admission_v1",
        "experiment_id": EXPERIMENT_ID,
        "execution_revision": EXECUTION_REVISION,
        "config_hash": config.config_hash,
        "runtime_hash_contract_hash": runtime_hash_contract[
            "runtime_hash_contract_hash"
        ],
        **_authorization_provenance(authority),
        "parent_ledger_sha256": parent_hash,
        "cache_hash": cache.cache_hash,
        "preflight_hash": _stable_preflight_hash(preflight),
        "action_capacity_certificate_hash": capacity[
            "capacity_certificate_hash"
        ],
        "all_gates_read_only": True,
        "filesystem_mutations_before_lease": 0,
        "launch_confirmation_validated": True,
        "launch_confirmation_token_persisted": False,
        "development_labels_opened": False,
        "evaluation_labels_opened": False,
    }
    admission_hash = canonical_hash(admission)
    output_state = _validate_output_state(root, admission_hash=admission_hash)
    lease: Any | None = None
    ledger = PhaseLedger()
    try:
        lease = V11_RUNNER_SERVICES.claim_authorization(
            authority, admission_hash=admission_hash
        )
        ledger.advance("AUTHORITY_ADMISSION")
        _announce("AUTHORITY_ADMISSION")
        admission_path = root / "manifests/admission.json"
        capacity_path = root / "manifests/action_capacity_certificate.json"
        protocol_path = root / "manifests/protocol_manifest.json"
        atomic_json(admission_path, {**admission, "admission_hash": admission_hash})
        atomic_json(capacity_path, capacity)
        atomic_json(
            protocol_path,
            {
                "schema_version": "midogpp_harp_v11_protocol_manifest_v1",
                "experiment_id": EXPERIMENT_ID,
                "execution_revision": EXECUTION_REVISION,
                "config_hash": config.config_hash,
                "runtime_hash_contract": runtime_hash_contract,
                "runtime_hash_contract_hash": runtime_hash_contract[
                    "runtime_hash_contract_hash"
                ],
                "action_capacity_certificate": capacity,
                "action_capacity_certificate_hash": capacity[
                    "capacity_certificate_hash"
                ],
                "protocol": dict(config.protocol),
                "model": dict(config.model),
                "runtime": dict(config.runtime),
                "claim_boundary": dict(config.claim_boundary),
                "utility_kind": "downstream_classifier_utility_not_NELBO",
                "routing_stage_compatibility_estimated": True,
                "compatibility_proxy_is_exact_nelbo": False,
                "compatibility_proxy_is_true_utility": False,
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
        centers = tuple(str(value) for value in config.protocol["centers"])
        ledger.advance("LABEL_FREE_SOURCE_CROSSFIT_SURFACE")
        _announce("LABEL_FREE_SOURCE_CROSSFIT_SURFACE")
        source_crossfit = materialize_label_free_source_crossfit(
            pipeline=active_pipeline,
            config=config,
            cache=cache,
            centers=centers,
            scratch_root=scratch,
            durable_root=root / "stores/source_crossfit_physical_surface",
        )
        source_effective_artifact = build_source_crossfit_effective_artifact(
            source_crossfit
        )
        source_effective_receipt = write_artifact_value(
            root / "stores/source_crossfit_effective_menu",
            source_effective_artifact,
            role="source_crossfit_effective_menu",
        )
        source_physical_seal = {
            "schema_version": "midogpp_harp_v11_source_crossfit_physical_surface_seal_v1",
            "source_surface_hash": source_crossfit.physical_surface.surface_hash,
            "source_surface_receipt_hash": source_crossfit.surface_receipt.receipt_hash,
            "inventory_hash": source_crossfit.surface_receipt.inventory_hash,
            "manifest_sha256": source_crossfit.surface_receipt.manifest_sha256,
            "probabilities_sha256": source_crossfit.surface_receipt.probabilities_sha256,
            "dispersion_sha256": source_crossfit.surface_receipt.dispersion_sha256,
            "compatibility_sha256": source_crossfit.surface_receipt.compatibility_sha256,
            "labels_consumed": False,
        }
        source_physical_seal = {
            **source_physical_seal,
            "seal_hash": canonical_hash(source_physical_seal),
        }
        source_physical_seal_path = (
            root / "manifests/source_crossfit_physical_surface_seal.json"
        )
        atomic_json(source_physical_seal_path, source_physical_seal)
        source_effective_seal = {
            "schema_version": "midogpp_harp_v11_source_crossfit_effective_menu_seal_v1",
            "source_surface_receipt_hash": source_crossfit.surface_receipt.receipt_hash,
            "source_surface_hash": source_crossfit.physical_surface.surface_hash,
            "effective_adapter_hash": source_crossfit.effective_surface.adapter_hash,
            "effective_menu_store_hash": source_effective_artifact.manifest[
                "effective_menu_hash"
            ],
            "store_manifest_sha256": source_effective_receipt.manifest_sha256,
            "store_npz_sha256": source_effective_receipt.npz_sha256,
            "labels_consumed": False,
        }
        source_effective_seal = {
            **source_effective_seal,
            "seal_hash": canonical_hash(source_effective_seal),
        }
        source_effective_seal_path = (
            root / "manifests/source_crossfit_effective_menu_seal.json"
        )
        atomic_json(source_effective_seal_path, source_effective_seal)

        ledger.advance("LABEL_FREE_TARGET_PHYSICAL_MENU")
        _announce("LABEL_FREE_TARGET_PHYSICAL_MENU")
        target_materializer = getattr(
            active_pipeline, "materialize_label_free_target_menus", None
        )
        bind_predictions = getattr(
            active_pipeline, "bind_source_crossfit_predictions", None
        )
        if callable(target_materializer) and callable(bind_predictions):
            target_only = target_materializer(
                config,
                cache,
                outer_targets=centers,
                scratch_root=scratch,
            )
            produced = tuple(
                bind_predictions(target_only, source_crossfit.physical_surface)
            )
            menu_roots = tuple(
                root / "stores/physical_menu" / f"outer_{outer}"
                for outer in centers
            )
            menu_receipts = tuple(
                write_label_free_outer_menu(store_root, menu)
                for store_root, menu in zip(menu_roots, produced, strict=True)
            )
        else:
            # Synthetic test doubles may retain the historical all-context
            # seam; bind their target blocks to the crossfit r=q predictions.
            menu_bundle = recover_or_materialize_label_free_menus(
                root=root,
                centers=centers,
                journal=journal,
                pipeline=active_pipeline,
                config=config,
                cache=cache,
                scratch=scratch,
            )
            if not callable(bind_predictions):
                raise ProtocolError(
                    "HARP v11 pipeline cannot bind target menus to source crossfit."
                )
            produced = tuple(
                bind_predictions(menu_bundle.menus, source_crossfit.physical_surface)
            )
            menu_roots = tuple(
                root / "stores/physical_menu" / f"outer_{outer}"
                for outer in centers
            )
            menu_receipts = tuple(
                write_label_free_outer_menu(store_root, menu)
                for store_root, menu in zip(menu_roots, produced, strict=True)
            )

        ledger.advance("LABEL_FREE_TARGET_COMPATIBILITY")
        _announce("LABEL_FREE_TARGET_COMPATIBILITY")
        compatibility_bundle = recover_or_materialize_compatibility(
            root=root,
            output_state=output_state,
            menus=produced,
            pipeline=active_pipeline,
            cache=cache,
            config=config,
            scratch=scratch,
        )
        compatibility = compatibility_bundle.state
        compatibility_receipt = compatibility_bundle.receipt
        compatibility_hash = compatibility.manifest.get("compatibility_hash")
        if type(compatibility_hash) is not str or len(compatibility_hash) != 64:
            raise ProtocolError(
                "HARP v11 label-free compatibility surface lacks a SHA-256 identity."
            )
        effective_menu = build_effective_menu_artifact(compatibility)
        effective_menu_receipt = write_artifact_value(
            root / "stores/effective_menu",
            effective_menu,
            role="label_free_effective_menu",
        )
        development_seal = build_development_surface_seal(
            config=config,
            menus=produced,
            menu_receipts=menu_receipts,
            compatibility_hash=compatibility_hash,
            compatibility_receipt=compatibility_receipt,
            effective_menu=effective_menu,
            effective_menu_receipt=effective_menu_receipt,
            source_crossfit=source_crossfit,
        )
        development_seal_path = root / "manifests/development_surface_seal.json"
        atomic_json(development_seal_path, development_seal)
        durable_barrier(
            [
                development_seal_path,
                *(receipt.manifest_path for receipt in menu_receipts),
                *(receipt.npz_path for receipt in menu_receipts),
                compatibility_receipt.manifest_path,
                compatibility_receipt.npz_path,
                effective_menu_receipt.manifest_path,
                effective_menu_receipt.npz_path,
                source_crossfit.surface_receipt.manifest_path,
                source_crossfit.surface_receipt.probabilities_path,
                source_crossfit.surface_receipt.dispersion_path,
                source_crossfit.surface_receipt.compatibility_path,
                source_physical_seal_path,
                source_effective_receipt.manifest_path,
                source_effective_receipt.npz_path,
                source_effective_seal_path,
            ]
        )
        ledger.advance("SOURCE_CROSSFIT_SURFACE_SEALED")
        _announce("SOURCE_CROSSFIT_SURFACE_SEALED")

        ledger.advance("SOURCE_FOLD_LABEL_CAPABILITIES_OPENED")
        _announce("SOURCE_FOLD_LABEL_CAPABILITIES_OPENED")
        fold_seal_set = fit_and_seal_prelabel_source_folds(
            bundle=source_crossfit,
            config=config,
            cache=cache,
            source_label_loader=V11_RUNNER_SERVICES.load_development_labels,
            fold_store_root=root / "stores/source_crossfit_folds",
            fold_set_path=root / "manifests/source_prelabel_fold_set_internal.json",
            workers=int(config.runtime["science_workers"]),
        )

        capability_seal_path = (
            root / "manifests/source_fold_label_capability_seals.json"
        )
        atomic_json(
            capability_seal_path,
            source_fold_capability_seal_payload(fold_seal_set),
        )
        source_predictions = build_source_prelabel_prediction_artifact(fold_seal_set)
        source_prediction_receipt = write_artifact_value(
            root / "stores/source_prelabel_q_predictions",
            source_predictions,
            role="source_prelabel_q_predictions",
        )
        source_prediction_seal_path = (
            root / "manifests/source_prelabel_q_prediction_seal.json"
        )
        atomic_json(
            source_prediction_seal_path,
            source_prelabel_q_prediction_seal_payload(
                fold_seal_set,
                source_predictions,
                store_manifest_sha256=source_prediction_receipt.manifest_sha256,
                store_npz_sha256=source_prediction_receipt.npz_sha256,
            ),
        )
        durable_barrier(
            (
                fold_seal_set.path,
                capability_seal_path,
                source_prediction_receipt.manifest_path,
                source_prediction_receipt.npz_path,
                source_prediction_seal_path,
            )
        )

        ledger.advance("SOURCE_PRELABEL_Q_FOLDS_SEALED")
        _announce("SOURCE_PRELABEL_Q_FOLDS_SEALED")

        ledger.advance("FULL_SOURCE_LABELS_OPENED")
        _announce("FULL_SOURCE_LABELS_OPENED")

        ledger.advance("SOURCE_ONLY_NESTED_LODO_MODEL_FIT")
        _announce("SOURCE_ONLY_NESTED_LODO_MODEL_FIT")
        crossfit_fit = assemble_source_crossfit_model(
            bundle=source_crossfit,
            fold_seal_set=fold_seal_set,
            config=config,
            cache=cache,
            source_label_loader=V11_RUNNER_SERVICES.load_development_labels,
            target_compatibility_hash=compatibility_hash,
        )
        development = crossfit_fit.development
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
                "schema_version": "midogpp_harp_v11_development_label_access_v1",
                "fold_local_labels_opened_only_in_spawned_H_q_workers": True,
                "full_source_labels_opened_after_all_q_prediction_seals": True,
                "development_surface_seal_hash": development_seal["seal_hash"],
                "source_prelabel_fold_seal_set_hash": fold_seal_set.seal_set_hash,
                "aggregate_source_label_capability_hash": (
                    crossfit_fit.aggregate_capability.capability_hash
                ),
                "evaluation_labels_opened": False,
                "fold_scoped_access_count": len(fold_seal_set.fold_seals),
                "aggregate_access_count": 1,
            },
        )

        fitted = crossfit_fit.fitted
        fitted = _bind_model_artifact(
            fitted,
            development_surface_hash=str(development.manifest["surface_hash"]),
            compatibility_hash=str(compatibility_hash),
            config_hash=config.config_hash,
            centers=centers,
        )
        model_receipt = write_artifact_value(
            root / "stores/source_only_model", fitted, role="source_only_router"
        )
        model_lock = build_model_lock(
            config_hash=config.config_hash,
            centers=centers,
            development=development,
            fitted=fitted,
            model_receipt=model_receipt,
            development_receipt=development_receipt,
            compatibility_hash=compatibility_hash,
            compatibility_receipt=compatibility_receipt,
            source_crossfit=source_crossfit,
            fold_seal_set=fold_seal_set,
        )
        model_lock_path = root / "manifests/model_lock.json"
        atomic_json(model_lock_path, model_lock)

        ledger.advance("SOURCE_ONLY_POLICY_RISK_COVERAGE_ADMISSION")
        _announce("SOURCE_ONLY_POLICY_RISK_COVERAGE_ADMISSION")
        policy = active_pipeline.admit_source_only_router(
            fitted, development, config=config
        )
        policy = _bind_admission_artifact(
            policy,
            model_hash=str(fitted.manifest["model_hash"]),
            development_surface_hash=str(development.manifest["surface_hash"]),
            config_hash=config.config_hash,
            centers=centers,
        )
        admission_receipt = write_artifact_value(
            root / "stores/source_only_policy_oof_replay",
            policy,
            role="source_only_policy_oof_replay",
        )
        policy_admission_seal = build_policy_admission_seal(
            config_hash=config.config_hash,
            fitted=fitted,
            policy=policy,
            admission_receipt=admission_receipt,
        )
        policy_admission_seal_path = root / "manifests/source_policy_admission_seal.json"
        atomic_json(policy_admission_seal_path, policy_admission_seal)

        ledger.advance("TARGET_ACTIONS_COMPLETE")
        _announce("TARGET_ACTIONS_COMPLETE")
        target_actions = active_pipeline.build_complete_target_case_actions(
            produced, compatibility, fitted, policy, config=config
        )
        target_actions = _bind_target_action_artifact(
            target_actions,
            model_hash=str(fitted.manifest["model_hash"]),
            compatibility_hash=str(compatibility_hash),
            admission_hash=str(policy.manifest["admission_hash"]),
            menu_hashes={menu.outer_target_id: menu.menu_hash for menu in produced},
            config_hash=config.config_hash,
            centers=centers,
        )
        target_receipt = write_artifact_value(
            root / "stores/target_case_actions",
            target_actions,
            role="complete_target_case_actions",
        )
        target_seal = build_target_action_seal(
            config=config,
            centers=centers,
            fitted=fitted,
            compatibility_hash=compatibility_hash,
            policy=policy,
            target_actions=target_actions,
            target_receipt=target_receipt,
        )
        target_seal_path = root / "manifests/target_action_seal.json"
        atomic_json(target_seal_path, target_seal)

        routes = active_pipeline.route_case_actions(
            produced, target_actions, fitted, policy, config=config
        )
        _validate_in_memory_route_bindings(
            routes,
            model_hash=str(fitted.manifest["model_hash"]),
            target_action_hash=str(target_actions.manifest["target_action_hash"]),
            centers=centers,
        )
        if (
            len(routes.cases) != target_actions.manifest.get("target_case_count")
            or routes.ordered_case_identity_hash
            != target_actions.manifest.get("ordered_case_identity_hash")
            or routes.ordered_sample_identity_hash
            != target_actions.manifest.get("ordered_sample_identity_hash")
        ):
            raise ProtocolError(
                "HARP v11 routed inventory differs from the sealed all-test action surface."
            )
        route_root = root / "stores/prelabel_routes"
        route_receipt = write_prelabel_routes(route_root, routes)
        reconstructed_routes = read_prelabel_routes(route_root)
        if reconstructed_routes.route_hash != routes.route_hash:
            raise ProtocolError("HARP v11 prelabel route store changed identity.")
        route_summary = _prelabel_route_summary(routes)
        rejection_diagnostics = build_prelabel_diagnostics(routes)
        rejection_diagnostics_path = (
            root / "reports/prelabel_rejection_diagnostics.json"
        )
        atomic_json(rejection_diagnostics_path, rejection_diagnostics)
        prelabel = build_prelabel_bundle(
            config_hash=config.config_hash,
            centers=centers,
            development_seal=development_seal,
            model_lock=model_lock,
            policy_admission_seal=policy_admission_seal,
            target_seal=target_seal,
            policy=policy,
            routes=routes,
            route_receipt=route_receipt,
            route_summary=route_summary,
            rejection_diagnostics=rejection_diagnostics,
        )
        prelabel_path = root / "manifests/prelabel_route_bundle.json"
        atomic_json(prelabel_path, prelabel)
        durable_barrier(
            [
                model_lock_path,
                policy_admission_seal_path,
                admission_receipt.manifest_path,
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
            compatibility_root=root / "stores/label_free_support_compatibility",
            effective_menu_root=root / "stores/effective_menu",
            admission_root=root / "stores/source_only_policy_oof_replay",
        )
        validation_bundle = build_validation_bundle(
            config_hash=config.config_hash,
            centers=centers,
            validations=validations,
        )
        validation_bundle_path = root / "manifests/fresh_validations.json"
        atomic_json(validation_bundle_path, validation_bundle)
        ledger.advance("FRESH_RECONSTRUCTIONS_COMPLETE")
        _announce("FRESH_RECONSTRUCTIONS_COMPLETE")

        frozen = build_frozen_route_seal(
            config_hash=config.config_hash,
            centers=centers,
            prelabel=prelabel,
            routes=routes,
            policy=policy,
            validation_bundle=validation_bundle,
            validations=validations,
            route_summary=route_summary,
        )
        frozen_path = root / "manifests/frozen_route_seal.json"
        atomic_json(frozen_path, frozen)
        durable_barrier([validation_bundle_path, frozen_path])
        if read_json(frozen_path) != frozen:
            raise ProtocolError("HARP v11 frozen route seal failed a fresh read.")
        ledger.advance("FROZEN_ROUTE_SEAL")
        _announce("FROZEN_ROUTE_SEAL")

        sealed_routes, frozen_receipt = _reconstruct_frozen_routes_for_evaluation(
            route_root,
            frozen=frozen,
            model_hash=str(fitted.manifest["model_hash"]),
            target_action_hash=str(target_actions.manifest["target_action_hash"]),
            centers=centers,
            config_hash=config.config_hash,
        )
        ledger.advance("EVALUATION_LABELS_OPENED")
        _announce("EVALUATION_LABELS_OPENED")
        evaluation_truth = V11_RUNNER_SERVICES.load_evaluation_truth(
            config,
            cache,
            frozen_receipt,
        )
        terminal = active_pipeline.evaluate_terminal(
            sealed_routes,
            evaluation_truth,
            frozen_receipt=frozen_receipt,
            artifact_root=root,
            config=config,
        )
        terminal_reports = write_terminal_reports(
            root,
            terminal=terminal,
            sealed_routes=sealed_routes,
            frozen=frozen,
            development_surface_seal_hash=development_seal["seal_hash"],
            model_lock_hash=model_lock["model_lock_hash"],
            target_action_seal_hash=target_seal["seal_hash"],
            validations=validations,
            route_summary=route_summary,
        )
        terminal_metrics = terminal_reports.metrics
        terminal_paths = terminal_reports.paths
        durable_barrier(terminal_paths)
        ledger.advance("TERMINAL_DIAGNOSTIC_COMPLETE")
        _announce("TERMINAL_DIAGNOSTIC_COMPLETE")

        lease = V11_RUNNER_SERVICES.finalize_authorization(
            lease, status="COMPLETE_EXHAUSTED"
        )
        finalization = read_json(lease.root / "lease.json")
        if finalization.get("status") != "COMPLETE_EXHAUSTED":
            raise ProtocolError("HARP v11 authorization did not finalize.")
        finalization_path = root / "manifests/authorization_finalization.json"
        atomic_json(finalization_path, finalization)
        durable_barrier((finalization_path,))
        content_index_path = _write_content_index(root)
        _validate_content_index(root, content_index_path)
        run_state_path = root / "reports/run_state.json"
        run_state = build_run_state(
            root=root,
            ledger=ledger,
            lease=lease,
            finalization_path=finalization_path,
            content_index_path=content_index_path,
            terminal_paths=terminal_paths,
            frozen=frozen,
            sealed_routes=sealed_routes,
            terminal_metrics=terminal_metrics,
            scratch=scratch,
        )
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
        if lease is not None and ledger.development_labels_opened:
            try:
                atomic_json(
                    root / "reports/failure_report.json",
                    {
                        "schema_version": "midogpp_harp_v11_failure_report_v1",
                        "status": "FAILED_EXHAUSTED",
                        "phase_order": list(ledger.observed),
                        "error_class": exc.__class__.__name__,
                        "error": str(exc)[:2000],
                        "publication_status": PUBLICATION_STATUS,
                        "terminal_decision": TERMINAL_DECISION,
                    },
                )
                V11_RUNNER_SERVICES.finalize_authorization(
                    lease, status="FAILED_EXHAUSTED", error=str(exc)
                )
            except BaseException:
                pass
        elif lease is not None:
            # Before the development-label capability opens, keep the exact
            # active lease and label-free journal recoverable. A later process
            # must authenticate both identities; no label-bearing artifact is
            # written on this path.
            _announce("LABEL_FREE_RECOVERY_RETAINED")
        raise


def _stable_preflight_hash(preflight: Mapping[str, object]) -> str:
    """Bind science/topology while excluding volatile capacity observations."""

    stable = dict(preflight)
    stable.pop("scratch_free_bytes", None)
    stable.pop("scratch_probe_path", None)
    raw_gpus = stable.get("gpus")
    if isinstance(raw_gpus, list):
        stable["gpus"] = [
            {
                key: value
                for key, value in row.items()
                if key != "memory_free_mib"
            }
            if isinstance(row, Mapping)
            else row
            for row in raw_gpus
        ]
    return canonical_hash(stable)


def _announce(phase: str) -> None:
    print(f"[harp-v11] phase={phase}", file=sys.stderr, flush=True)


__all__ = (
    "IMPLEMENTED_COMPONENTS",
    "HarpV11RunnerServices",
    "HARP_V11_RUN_CONFIRMATION_TOKEN",
    "V11_RUNNER_SERVICES",
    "dry_run_harp_stage90_v11",
    "inspect_harp_stage90_v11",
    "run_harp_stage90_v11",
)
