"""Protocol-ordered, workstation-optimized HARP v16 production runner.

V16 changes the estimand from zero-shot source-center transfer to known-center
support adaptation.  Every physical action for Train-H support and Test-H
evaluation is materialized and sealed before the center-scoped Train-H label
capability opens.  Test labels remain unavailable until the final route store
has been reconstructed by two fresh processes and sealed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_bytes, canonical_hash
from ...runtime.artifact_io import atomic_json, read_json
from ...runtime.harp_v16_execution.action_capacity import (
    build_action_capacity_certificate,
    validate_action_capacity_certificate,
)
from ...runtime.harp_v16_execution.contracts import (
    HarpV16Pipeline,
)
from ...runtime.harp_v16_execution.durability import durable_barrier
from ...runtime.harp_v16_execution.hash_contracts import (
    runtime_hash_contract_payload,
)
from ...runtime.harp_v16_execution.journal import LabelFreeProgressJournal
from ...runtime.harp_v16_execution.menu_root_binding import CenterMenuRootBinding
from ...runtime.harp_v16_execution.phases import PHASE_ORDER, PhaseLedger
from ...runtime.harp_v16_execution.prelabel_diagnostics import (
    build_prelabel_diagnostics,
)
from ...runtime.harp_v16_execution.stores import (
    read_prelabel_routes,
    write_artifact_value,
    write_prelabel_routes,
)
from ...runtime.harp_v16_execution.support_surface_seals import (
    SupportTargetSurfaceSealSet,
    write_support_target_surface_seals,
)
from ...runtime.harp_v16_execution.support_validation import (
    MODEL_ARTIFACT_ROLE,
    TARGET_ACTION_ARTIFACT_ROLE,
    run_two_fresh_support_validations,
)
from ....workspace.preparation_authority import HARP_V16_RUN_CONFIRMATION_TOKEN
from .authorization import (
    HarpV16Authorization,
    HarpV16AuthorizationLease,
    claim_authorization,
    finalize_authorization,
    load_authorization,
    repository_root as authorization_repository_root,
)
from .config import HarpStage90V16Config
from .execution import (
    authorization_provenance as _authorization_provenance,
    commit_completion_state as _commit_completion_state,
    dedicated_scratch as _dedicated_scratch,
    exact_output_root as _exact_output_root,
    prelabel_route_summary as _prelabel_route_summary,
    reconstruct_frozen_routes_for_evaluation as _reconstruct_frozen_routes,
    validate_complete_physical_menus,
    validate_content_index as _validate_content_index,
    validate_in_memory_route_bindings as _validate_route_bindings,
    validate_parent_ledger as _validate_parent_ledger,
    validate_preflight as _validate_preflight,
    validate_pristine_or_label_free_recovery as _validate_output_state,
    write_content_index as _write_content_index,
    write_terminal_reports,
)
from .execution.menu_roots import (
    build_center_menu_roots,
    validate_center_menu_root_bijection,
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
    load_evaluation_truth,
    load_support_labels,
)
from .runner_payloads import (
    build_frozen_route_seal,
    build_model_lock,
    build_policy_admission_seal,
    build_prelabel_bundle,
    build_run_state,
    build_surface_seal_indexes,
    build_target_action_seal,
    build_validation_bundle,
)
from .runner_recovery import (
    announce as _announce,
    persist_or_validate_json as _persist_or_validate_json,
    recover_or_materialize_menus as _recover_or_materialize_menus,
    stable_preflight_hash as _stable_preflight_hash,
    write_or_validate_artifact as _write_or_validate_artifact,
)
from .run_failure import handle_run_failure
from .source_label_capability import issue_target_support_label_capability
from .support_label_access_fence import (
    begin_support_label_access,
)


IMPLEMENTED_COMPONENTS = (
    "separate_single_use_v16_authority_before_any_scientific_mutation",
    "exact_launch_confirmation_before_authority_or_path_access",
    "v16_owned_target_train_support_and_full_test_cache",
    "same_center_train_H_support_and_test_H_evaluation",
    "per_H_fixed_bank_independence_attestation_before_support_labels",
    "candidate_pool_C_minus_H_for_support_and_target",
    "joint_support_target_prediction_with_authenticated_role_offset",
    "eighty_one_classifier_process_tasks_and_eight_hundred_ten_fits",
    "two_persistent_gpu_workers_then_four_disjoint_classifier_workers",
    "float32_probability_transport_and_float64_scientific_reductions",
    "label_free_case_local_own_source_robust_compatibility",
    "shared_support_target_effective_action_identity_before_labels",
    "durable_fsynced_support_label_access_begun_fence_before_capability_issuance",
    "same_center_leave_one_support_case_out_endpoint_models",
    "fold_local_normalization_and_predeclared_ridge_regularization",
    "case_level_max_one_sided_finite_sample_residual_certificates",
    "hierarchical_route_direction_family_expert_selection",
    "per_outer_whole_policy_support_OOF_risk_coverage_admission",
    "no_global_kill_switch_and_no_pairwise_ranker",
    "exact_top1_physical_action_or_byte_identical_exact_B",
    "all_test_cases_routed_label_blind_before_truth_release",
    "two_distinct_spawn_process_route_reconstructions",
    "center_keyed_durable_physical_menu_receipt_bijection",
    "disk_reconstructed_frozen_routes_are_terminal_only_input",
    "single_content_index_and_atomic_fsync_completion_commit",
)


@dataclass(frozen=True, slots=True)
class HarpV16RunnerServices:
    config_type: type
    authorization_type: type
    lease_type: type
    load_authorization: Callable[[Any], Any]
    claim_authorization: Callable[..., Any]
    finalize_authorization: Callable[..., Any]
    load_cache_index: Callable[[Any], Any]
    load_support_labels: Callable[..., object]
    load_evaluation_truth: Callable[[Any, Any, Any], object]


V16_RUNNER_SERVICES = HarpV16RunnerServices(
    config_type=HarpStage90V16Config,
    authorization_type=HarpV16Authorization,
    lease_type=HarpV16AuthorizationLease,
    load_authorization=load_authorization,
    claim_authorization=claim_authorization,
    finalize_authorization=finalize_authorization,
    load_cache_index=load_cache_index,
    load_support_labels=load_support_labels,
    load_evaluation_truth=load_evaluation_truth,
)


def _pipeline_or_production(pipeline: HarpV16Pipeline | None) -> HarpV16Pipeline:
    if pipeline is not None:
        return pipeline
    from ...runtime.harp_v16_execution.production import HarpV16ProductionPipeline

    return HarpV16ProductionPipeline(
        development_role=DEVELOPMENT_ROLE,
        evaluation_role=EVALUATION_ROLE,
    )


def _capacity_certificate(config: HarpStage90V16Config) -> dict[str, object]:
    centers = tuple(str(value) for value in config.protocol["centers"])
    certificate = dict(build_action_capacity_certificate(centers=centers))
    validate_action_capacity_certificate(certificate, centers=centers)
    expected_maxima = {
        "support": config.protocol[
            "support_max_required_rows_per_source_per_class"
        ],
        "target": config.protocol[
            "target_max_required_rows_per_source_per_class"
        ],
    }
    if (
        certificate["stream_rows_per_class"]
        != config.protocol["resident_source_rows_per_class"]
        or certificate["required_rows_per_class_by_surface"] != expected_maxima
        or certificate["global_maximum_required_rows_per_class"]
        != config.protocol["global_max_required_rows_per_source_per_class"]
        or certificate["support_action_count"] != 90
        or certificate["target_action_count"] != 90
        or config.protocol["prelease_action_capacity_certificate_required"]
        is not True
    ):
        raise ProtocolError("HARP v16 config/capacity certificate binding drifted.")
    return certificate


def inspect_harp_stage90_v16(
    config: HarpStage90V16Config,
) -> Mapping[str, object]:
    """Inspect architecture identity without resolving a scientific path."""

    if type(config) is not HarpStage90V16Config:
        raise ProtocolError("HARP v16 inspection requires a typed configuration.")
    runtime_hash_contract = dict(runtime_hash_contract_payload())
    capacity = _capacity_certificate(config)
    body = {
        "schema_version": "midogpp_harp_v16_implementation_inspection_v1",
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
        "implemented_components": list(IMPLEMENTED_COMPONENTS),
        "phase_order": list(PHASE_ORDER),
        "joint_support_target_classifier_task_count": 81,
        "total_classifier_fit_count": 810,
        "execution_authorized": config.execution_authorized,
        "authorization_probed": False,
        "paths_resolved": False,
        "filesystem_mutations": 0,
        "support_labels_opened": False,
        "evaluation_labels_opened": False,
        "old_aggregate_utility_surface_used": False,
        "predecessor_policy_cache_output_or_authority_used": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "may_feed_another_experiment": False,
    }
    return {**body, "inspection_hash": canonical_hash(body)}


def dry_run_harp_stage90_v16(
    config: HarpStage90V16Config,
    *,
    artifact_root: str | Path,
    pipeline: HarpV16Pipeline | None = None,
) -> Mapping[str, object]:
    """Mutation-free readiness check; planned mode opens no referenced path."""

    if type(config) is not HarpStage90V16Config:
        raise ProtocolError("HARP v16 dry run requires a typed configuration.")
    runtime_hash_contract = dict(runtime_hash_contract_payload())
    capacity = _capacity_certificate(config)
    if not config.execution_authorized:
        body = {
            "schema_version": "midogpp_harp_v16_mutation_free_dry_run_v1",
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
            "execution_authorized": False,
            "authorization_probed": False,
            "paths_resolved": False,
            "artifact_root_argument_recorded": False,
            "filesystem_mutations": 0,
            "support_labels_opened": False,
            "evaluation_labels_opened": False,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "fresh_evidence": False,
        }
        return {**body, "dry_run_hash": canonical_hash(body)}

    authority = V16_RUNNER_SERVICES.load_authorization(config)
    _exact_output_root(config, artifact_root)
    parent_hash = _validate_parent_ledger(config)
    cache = V16_RUNNER_SERVICES.load_cache_index(config)
    preflight = dict(_pipeline_or_production(pipeline).preflight(config, cache))
    _validate_preflight(preflight)
    if preflight.get("action_capacity_certificate") != capacity:
        raise ProtocolError("HARP v16 preflight capacity certificate drifted.")
    body = {
        "schema_version": "midogpp_harp_v16_mutation_free_dry_run_v1",
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
        "support_labels_opened": False,
        "evaluation_labels_opened": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
    }
    return {**body, "dry_run_hash": canonical_hash(body)}


def run_harp_stage90_v16(
    config: HarpStage90V16Config,
    *,
    artifact_root: str | Path,
    pipeline: HarpV16Pipeline | None = None,
    confirmation_token: str | None = None,
) -> str:
    """Execute one separately authorized terminal consumed-test diagnostic."""

    # This must precede config typing, authority probing, path resolution, input
    # access, scratch allocation, lease claim, and every filesystem mutation.
    if confirmation_token != HARP_V16_RUN_CONFIRMATION_TOKEN:
        raise ProtocolError(
            "HARP v16 execution requires the exact confirmation token "
            f"{HARP_V16_RUN_CONFIRMATION_TOKEN}."
        )
    if type(config) is not HarpStage90V16Config:
        raise ProtocolError("HARP v16 execution requires a typed configuration.")

    runtime_hash_contract = dict(runtime_hash_contract_payload())
    capacity = _capacity_certificate(config)
    authority = V16_RUNNER_SERVICES.load_authorization(config)
    root = _exact_output_root(config, artifact_root)
    parent_hash = _validate_parent_ledger(config)
    cache = V16_RUNNER_SERVICES.load_cache_index(config)
    active_pipeline = _pipeline_or_production(pipeline)
    preflight = dict(active_pipeline.preflight(config, cache))
    _validate_preflight(preflight)
    if preflight.get("action_capacity_certificate") != capacity:
        raise ProtocolError("HARP v16 preflight capacity certificate drifted.")

    admission = {
        "schema_version": "midogpp_harp_v16_execution_admission_v1",
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
        "support_labels_opened": False,
        "evaluation_labels_opened": False,
    }
    admission_hash = canonical_hash(admission)
    output_state = _validate_output_state(root, admission_hash=admission_hash)
    lease: Any | None = None
    ledger = PhaseLedger()
    try:
        lease = V16_RUNNER_SERVICES.claim_authorization(
            authority,
            admission_hash=admission_hash,
            repo_root=authorization_repository_root(),
        )
        ledger.advance("AUTHORITY_ADMISSION")
        _announce("AUTHORITY_ADMISSION")

        admission_path = root / "manifests/admission.json"
        capacity_path = root / "manifests/action_capacity_certificate.json"
        protocol_path = root / "manifests/protocol_manifest.json"
        _persist_or_validate_json(
            admission_path, {**admission, "admission_hash": admission_hash}
        )
        _persist_or_validate_json(capacity_path, capacity)
        _persist_or_validate_json(
            protocol_path,
            {
                "schema_version": "midogpp_harp_v16_protocol_manifest_v1",
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
                "known_center_support_adaptation_estimand": True,
                "utility_kind": "downstream_classifier_utility_not_NELBO",
                "evaluation_labels_used_for_fit": False,
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

        ledger.advance("LABEL_FREE_SUPPORT_TARGET_PHYSICAL_MENUS")
        _announce("LABEL_FREE_SUPPORT_TARGET_PHYSICAL_MENUS")
        menus, menu_receipts = _recover_or_materialize_menus(
            root=root,
            centers=centers,
            journal=journal,
            pipeline=active_pipeline,
            config=config,
            cache=cache,
            scratch=scratch,
        )
        validate_complete_physical_menus(menus, centers=centers)
        physical_parent = root / "stores/physical_menu"
        menu_roots = build_center_menu_roots(
            physical_parent, centers=centers, menus=menus
        )
        menu_roots = validate_center_menu_root_bijection(
            menu_roots,
            physical_menu_parent=physical_parent,
            centers=centers,
            menus=menus,
            receipts=menu_receipts,
        )
        menu_binding = CenterMenuRootBinding.create(
            common_parent=physical_parent,
            centers=centers,
            menu_roots=menu_roots,
            menus=menus,
            receipts=menu_receipts,
        )
        _persist_or_validate_json(
            root / "manifests/center_menu_root_binding.json",
            menu_binding.to_payload(),
        )

        bundles, compatibility = (
            active_pipeline.compile_label_free_support_target_menus(
                menus, scratch_root=scratch
            )
        )
        bundles = tuple(bundles)
        if tuple(row.outer_target_id for row in bundles) != centers:
            raise ProtocolError("HARP v16 compiled support/target inventory drifted.")
        compatibility_receipt = _write_or_validate_artifact(
            root / "stores/label_free_compatibility",
            compatibility,
            role="target_support_label_free_compatibility",
        )

        attestation = active_pipeline.physical_input_receipt.support_independence
        seal_sets = tuple(
            write_support_target_surface_seals(
                root / "manifests/support_target_role_seals",
                bundle=bundle,
                physical_store_receipt=menu_receipts[index],
                fixed_bank_independence=attestation,
            )
            for index, bundle in enumerate(bundles)
        )
        support_index, target_index, attestation_index = build_surface_seal_indexes(
            seal_sets
        )
        support_index_path = root / "manifests/target_support_menu_seals.json"
        target_index_path = root / "manifests/target_evaluation_menu_seals.json"
        attestation_index_path = (
            root / "manifests/target_bank_independence_attestations.json"
        )
        _persist_or_validate_json(support_index_path, support_index)
        _persist_or_validate_json(target_index_path, target_index)
        _persist_or_validate_json(attestation_index_path, attestation_index)
        durable_barrier(
            (
                *(receipt.manifest_path for receipt in menu_receipts),
                *(receipt.npz_path for receipt in menu_receipts),
                compatibility_receipt.manifest_path,
                compatibility_receipt.npz_path,
                *(row.support_menu_seal_path for row in seal_sets),
                *(row.target_menu_seal_path for row in seal_sets),
                *(row.bank_independence_attestation_path for row in seal_sets),
                support_index_path,
                target_index_path,
                attestation_index_path,
                root / "manifests/center_menu_root_binding.json",
            )
        )
        ledger.advance("SUPPORT_TARGET_MENUS_SEALED")
        _announce("SUPPORT_TARGET_MENUS_SEALED")
        ledger.advance("FIXED_BANK_SUPPORT_INDEPENDENCE_ATTESTED")
        _announce("FIXED_BANK_SUPPORT_INDEPENDENCE_ATTESTED")

        support_labels: dict[str, Sequence[object]] = {}
        capability_hashes: dict[str, str] = {}
        label_index_path = config.resolved_path("development_manifest_path")
        label_index_sha256 = str(
            config.expected_hashes["development_manifest_sha256"]
        )
        support_label_access_fence = begin_support_label_access(
            root,
            config_hash=config.config_hash,
            admission_hash=admission_hash,
            authorization_lease_hash=lease.lease_hash,
            ordered_center_ids=centers,
            support_surface_seal_index=support_index,
            support_surface_seal_index_path=support_index_path,
            target_surface_seal_index=target_index,
            target_surface_seal_index_path=target_index_path,
            bank_independence_index=attestation_index,
            bank_independence_index_path=attestation_index_path,
            label_index_sha256=label_index_sha256,
        )
        ledger.advance("SUPPORT_LABEL_CAPABILITIES_OPENED")
        _announce("SUPPORT_LABEL_CAPABILITIES_OPENED")
        for seal_set in seal_sets:
            capability = issue_target_support_label_capability(
                outer_target_id=seal_set.outer_target_id,
                support_menu_seal_path=seal_set.support_menu_seal_path,
                support_menu_seal_sha256=seal_set.support_menu_seal_sha256,
                target_menu_seal_path=seal_set.target_menu_seal_path,
                target_menu_seal_sha256=seal_set.target_menu_seal_sha256,
                bank_independence_attestation_path=(
                    seal_set.bank_independence_attestation_path
                ),
                bank_independence_attestation_sha256=(
                    seal_set.bank_independence_attestation_sha256
                ),
                label_index_path=label_index_path,
                label_index_sha256=label_index_sha256,
                support_label_access_fence=support_label_access_fence,
            )
            support_labels[seal_set.outer_target_id] = (
                V16_RUNNER_SERVICES.load_support_labels(
                    config,
                    cache,
                    allowed_centers=(seal_set.outer_target_id,),
                    source_label_capability=capability,
                )
            )
            capability_hashes[seal_set.outer_target_id] = capability.capability_hash
        support_access_path = root / "reports/support_label_access.json"
        atomic_json(
            support_access_path,
            {
                "schema_version": "midogpp_harp_v16_support_label_access_v1",
                "support_surface_seal_index_hash": support_index["index_hash"],
                "target_surface_seal_index_hash": target_index["index_hash"],
                "bank_independence_index_hash": attestation_index["index_hash"],
                "center_capability_hashes": capability_hashes,
                "center_scoped_access_count": len(capability_hashes),
                "support_label_access_fence_hash": (
                    support_label_access_fence.fence_hash
                ),
                "support_label_access_fence_sha256": (
                    support_label_access_fence.sha256
                ),
                "support_labels_opened": True,
                "support_labels_may_update": "H_LOCAL_ROUTER_ONLY",
                "evaluation_labels_opened": False,
            },
        )

        support_surface = active_pipeline.build_support_case_surface(
            bundles, support_labels
        )
        support_receipt = write_artifact_value(
            root / "stores/support_case_surface",
            support_surface,
            role="target_support_case_surface",
        )

        ledger.advance("TARGET_LOCAL_SUPPORT_MODELS_FIT")
        _announce("TARGET_LOCAL_SUPPORT_MODELS_FIT")
        fitted = active_pipeline.fit_target_local_support_routers(
            support_surface, config=config
        )
        model_receipt = write_artifact_value(
            root / "stores/target_local_models",
            fitted,
            role=MODEL_ARTIFACT_ROLE,
        )
        model_lock = build_model_lock(
            config_hash=config.config_hash,
            centers=centers,
            support_surface=support_surface,
            support_receipt=support_receipt,
            fitted=fitted,
            model_receipt=model_receipt,
            compatibility=compatibility,
            compatibility_receipt=compatibility_receipt,
        )
        model_lock_path = root / "manifests/model_lock.json"
        atomic_json(model_lock_path, model_lock)

        ledger.advance("SUPPORT_ONLY_POLICY_RISK_COVERAGE_ADMISSION")
        _announce("SUPPORT_ONLY_POLICY_RISK_COVERAGE_ADMISSION")
        policy_admission = build_policy_admission_seal(
            config_hash=config.config_hash, fitted=fitted
        )
        policy_path = root / "manifests/support_policy_admission_seal.json"
        atomic_json(policy_path, policy_admission)

        ledger.advance("TARGET_ACTIONS_COMPLETE")
        _announce("TARGET_ACTIONS_COMPLETE")
        target_actions = active_pipeline.build_complete_target_case_actions(
            bundles, fitted, config=config
        )
        target_receipt = write_artifact_value(
            root / "stores/target_case_actions",
            target_actions,
            role=TARGET_ACTION_ARTIFACT_ROLE,
        )
        target_seal = build_target_action_seal(
            config_hash=config.config_hash,
            centers=centers,
            target_actions=target_actions,
            target_receipt=target_receipt,
        )
        target_seal_path = root / "manifests/target_action_seal.json"
        atomic_json(target_seal_path, target_seal)

        routes = active_pipeline.route_case_actions(
            bundles, fitted, target_actions
        )
        _validate_route_bindings(
            routes,
            model_hash=str(fitted.manifest["model_hash"]),
            target_action_hash=str(target_actions.manifest["target_action_hash"]),
            centers=centers,
        )
        if len(routes.cases) != target_actions.manifest.get("target_case_count"):
            raise ProtocolError(
                "HARP v16 routed inventory differs from the sealed target surface."
            )
        route_root = root / "stores/prelabel_routes"
        route_receipt = write_prelabel_routes(route_root, routes)
        reconstructed = read_prelabel_routes(route_root)
        if reconstructed.route_hash != routes.route_hash:
            raise ProtocolError("HARP v16 prelabel route store changed identity.")
        route_summary = _prelabel_route_summary(routes)
        rejection_diagnostics = dict(build_prelabel_diagnostics(routes))
        rejection_path = root / "reports/prelabel_rejection_diagnostics.json"
        atomic_json(rejection_path, rejection_diagnostics)
        prelabel = build_prelabel_bundle(
            config_hash=config.config_hash,
            centers=centers,
            support_surface_seal_hash=str(support_index["index_hash"]),
            model_lock=model_lock,
            policy_admission_seal=policy_admission,
            target_seal=target_seal,
            routes=routes,
            route_receipt=route_receipt,
            route_summary=route_summary,
            rejection_diagnostics=rejection_diagnostics,
        )
        prelabel_path = root / "manifests/prelabel_route_bundle.json"
        atomic_json(prelabel_path, prelabel)
        durable_barrier(
            (
                support_access_path,
                support_receipt.manifest_path,
                support_receipt.npz_path,
                model_receipt.manifest_path,
                model_receipt.npz_path,
                model_lock_path,
                policy_path,
                target_receipt.manifest_path,
                target_receipt.npz_path,
                target_seal_path,
                route_receipt.manifest_path,
                route_receipt.npz_path,
                rejection_path,
                prelabel_path,
            )
        )
        ledger.advance("PRELABEL_ROUTES_DURABLE")
        _announce("PRELABEL_ROUTES_DURABLE")

        validations = run_two_fresh_support_validations(
            route_root=route_root,
            menu_binding=menu_binding,
            model_root=root / "stores/target_local_models",
            target_action_root=root / "stores/target_case_actions",
            expected_center_ids=centers,
            expected_config_hash=config.config_hash,
        )
        validation_bundle = build_validation_bundle(
            config_hash=config.config_hash,
            centers=centers,
            validations=validations,
        )
        validation_path = root / "manifests/fresh_validations.json"
        atomic_json(validation_path, validation_bundle)
        ledger.advance("FRESH_RECONSTRUCTIONS_COMPLETE")
        _announce("FRESH_RECONSTRUCTIONS_COMPLETE")

        frozen = build_frozen_route_seal(
            config_hash=config.config_hash,
            centers=centers,
            prelabel=prelabel,
            routes=routes,
            validation_bundle=validation_bundle,
            validations=validations,
            route_summary=route_summary,
        )
        frozen_path = root / "manifests/frozen_route_seal.json"
        atomic_json(frozen_path, frozen)
        durable_barrier((validation_path, frozen_path))
        if canonical_bytes(read_json(frozen_path)) != canonical_bytes(frozen):
            raise ProtocolError("HARP v16 frozen route seal failed readback.")
        ledger.advance("FROZEN_ROUTE_SEAL")
        _announce("FROZEN_ROUTE_SEAL")

        sealed_routes, frozen_receipt = _reconstruct_frozen_routes(
            route_root,
            frozen=frozen,
            model_hash=str(fitted.manifest["model_hash"]),
            target_action_hash=str(target_actions.manifest["target_action_hash"]),
            centers=centers,
            config_hash=config.config_hash,
        )
        ledger.advance("EVALUATION_LABELS_OPENED")
        _announce("EVALUATION_LABELS_OPENED")
        evaluation_truth = V16_RUNNER_SERVICES.load_evaluation_truth(
            config, cache, frozen_receipt
        )
        terminal = active_pipeline.evaluate_terminal(
            sealed_routes,
            evaluation_truth,
            frozen_receipt=frozen_receipt,
            artifact_root=root,
            config=config,
            menus=menus,
        )
        terminal_reports = write_terminal_reports(
            root,
            terminal=terminal,
            sealed_routes=sealed_routes,
            frozen=frozen,
            development_surface_seal_hash=support_index["index_hash"],
            model_lock_hash=model_lock["model_lock_hash"],
            target_action_seal_hash=target_seal["seal_hash"],
            validations=validations,
            route_summary=route_summary,
        )
        durable_barrier(terminal_reports.paths)
        ledger.advance("TERMINAL_DIAGNOSTIC_COMPLETE")
        _announce("TERMINAL_DIAGNOSTIC_COMPLETE")

        lease = V16_RUNNER_SERVICES.finalize_authorization(
            lease, status="COMPLETE_EXHAUSTED"
        )
        finalization = read_json(lease.root / "lease.json")
        if finalization.get("status") != "COMPLETE_EXHAUSTED":
            raise ProtocolError("HARP v16 authorization did not finalize.")
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
            terminal_paths=terminal_reports.paths,
            frozen=frozen,
            sealed_routes=sealed_routes,
            terminal_metrics=terminal_reports.metrics,
            scratch=scratch,
        )
        _commit_completion_state(
            root,
            run_state_path,
            run_state,
            durable_members=(
                *terminal_reports.paths,
                finalization_path,
                content_index_path,
            ),
        )
        return str(root)
    except BaseException as exc:
        try:
            handle_run_failure(
                root=root,
                lease=lease,
                ledger=ledger,
                error=exc,
                finalize_authorization=V16_RUNNER_SERVICES.finalize_authorization,
                announce=_announce,
            )
        except BaseException:
            pass
        raise


__all__ = (
    "HARP_V16_RUN_CONFIRMATION_TOKEN",
    "HarpV16RunnerServices",
    "IMPLEMENTED_COMPONENTS",
    "V16_RUNNER_SERVICES",
    "dry_run_harp_stage90_v16",
    "inspect_harp_stage90_v16",
    "run_harp_stage90_v16",
)
