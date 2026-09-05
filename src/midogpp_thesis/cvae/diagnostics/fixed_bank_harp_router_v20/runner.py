"""Protocol-ordered, workstation-optimized HARP v20 production runner.

V20 fits one pooled selected-policy router from all known-center source-train
q cases, where each q can use only C-minus-q experts. It then applies that
frozen policy to all full-test H cases using C-minus-H. All 18 label-free menu
surfaces and their bank proofs are sealed before any source truth opens; test
truth remains sealed until two fresh route reconstructions agree.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_bytes, canonical_hash
from ...runtime.artifact_io import atomic_json, read_json
from ...runtime.harp_v20_execution.action_capacity import (
    build_action_capacity_certificate,
    validate_action_capacity_certificate,
)
from ...runtime.harp_v20_execution.contracts import (
    HarpV20Pipeline,
)
from ...runtime.harp_v20_execution.durability import durable_barrier
from ...runtime.harp_v20_execution.hash_contracts import (
    runtime_hash_contract_payload,
)
from ...runtime.harp_v20_execution.journal import LabelFreeProgressJournal
from ...runtime.harp_v20_execution.menu_root_binding import CenterMenuRootBinding
from ...runtime.harp_v20_execution.phases import PHASE_ORDER, PhaseLedger
from ...runtime.harp_v20_execution.prelabel_diagnostics import (
    build_prelabel_diagnostics,
)
from ...runtime.harp_v20_execution.stores import (
    read_prelabel_routes,
    write_artifact_value,
    write_prelabel_routes,
)
from ...runtime.harp_v20_execution.support_surface_seals import (
    write_source_target_surface_seals,
)
from ...runtime.harp_v20_execution.support_validation import (
    POOLED_POLICY_ARTIFACT_ROLE,
    TARGET_EVALUATION_ACTION_ARTIFACT_ROLE,
    run_two_fresh_pooled_policy_validations,
)
from ....workspace.preparation_authority import HARP_V20_RUN_CONFIRMATION_TOKEN
from .execution.source_diagnostics import (
    enforce_admitted_target_coverage,
    write_source_diagnostics,
)
from .authorization import (
    HarpV20Authorization,
    HarpV20AuthorizationLease,
    claim_authorization,
    finalize_authorization,
    load_authorization,
    repository_root as authorization_repository_root,
)
from .config import HarpStage90V20Config
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
    SOURCE_TRAIN_ROLE,
    TARGET_EVALUATION_ROLE,
    load_cache_index,
    load_evaluation_truth,
    load_source_train_labels,
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
from .source_label_capability import issue_source_train_label_capabilities
from .source_train_label_access_fence import (
    begin_source_train_label_access,
)


IMPLEMENTED_COMPONENTS = (
    "separate_single_use_v20_authority_before_any_scientific_mutation",
    "exact_launch_confirmation_before_authority_or_path_access",
    "v20_owned_source_train_and_full_test_label_blind_cache",
    "known_center_source_train_q_to_full_test_H_estimand",
    "all_nine_source_and_all_nine_target_bank_proofs_before_source_truth",
    "candidate_pool_C_minus_q_for_source_and_C_minus_H_for_target",
    "separate_source_train_and_target_prediction_surfaces",
    "eighty_one_classifier_process_tasks_and_eight_hundred_ten_fits",
    "two_persistent_gpu_workers_then_four_disjoint_classifier_workers",
    "float32_probability_transport_and_float64_scientific_reductions",
    "label_free_case_local_own_source_robust_compatibility",
    "shared_label_free_effective_action_schema_before_labels",
    "durable_fsynced_source_train_label_access_fence_before_capability_issuance",
    "exactly_one_label_capability_per_q_with_exact_center_coverage",
    "nested_five_by_four_center_stratified_source_crossfit",
    "fully_nested_ranker_actual_candidate_outcome_then_held_winner_harm_gate",
    "unique_prediction_changing_candidates_train_coherent_safe_harm_other_model",
    "aligned_gain_minus_risk_excess_selects_unthresholded_winner",
    "negative_and_harmful_held_winners_retained_for_gate_learning",
    "gate_veto_returns_exact_B_without_trying_the_runnerup",
    "exact_B_U_D01_ONLY_D10_ONLY_BOTH_case_conditional_action_selection",
    "aligned_equal_center_equal_class_equal_supporting_case_source_and_terminal_gain",
    "candidate_frontier_and_actual_menu_oracle_persisted_before_admission",
    "candidate_prediction_outcome_and_winner_gate_joins_persisted_before_admission",
    "whole_policy_source_OOF_risk_coverage_admission",
    "approximate_center_stratified_studentized_max_stat_bounds",
    "soft_topK_lambda_selected_recipe_or_byte_identical_exact_B",
    "all_test_cases_routed_label_blind_before_truth_release",
    "two_distinct_spawn_process_route_reconstructions",
    "fresh_gate_coefficient_replay_on_sealed_features_and_iff_decision_rule",
    "center_keyed_durable_physical_menu_receipt_bijection",
    "disk_reconstructed_frozen_routes_are_terminal_only_input",
    "single_content_index_and_atomic_fsync_completion_commit",
)


@dataclass(frozen=True, slots=True)
class HarpV20RunnerServices:
    config_type: type
    authorization_type: type
    lease_type: type
    load_authorization: Callable[[Any], Any]
    claim_authorization: Callable[..., Any]
    finalize_authorization: Callable[..., Any]
    load_cache_index: Callable[[Any], Any]
    load_source_train_labels: Callable[..., object]
    load_evaluation_truth: Callable[[Any, Any, Any], object]


V20_RUNNER_SERVICES = HarpV20RunnerServices(
    config_type=HarpStage90V20Config,
    authorization_type=HarpV20Authorization,
    lease_type=HarpV20AuthorizationLease,
    load_authorization=load_authorization,
    claim_authorization=claim_authorization,
    finalize_authorization=finalize_authorization,
    load_cache_index=load_cache_index,
    load_source_train_labels=load_source_train_labels,
    load_evaluation_truth=load_evaluation_truth,
)


def _enforce_source_policy_admission(
    *,
    config: HarpStage90V20Config,
    root: Path,
    policy_path: Path,
    policy_admission: Mapping[str, object],
) -> None:
    """Apply the frozen non-admission contract before target construction."""

    raw_admission = policy_admission.get("source_only_admission")
    if not isinstance(raw_admission, Mapping):
        raise ProtocolError("HARP v20 source policy admission is untyped.")
    status = raw_admission.get("status")
    admitted = raw_admission.get("admitted")
    allowed = {
        "ADMITTED": True,
        "NO_NONZERO_SAFE_OOF_COVERAGE": False,
        "INSUFFICIENT_ROUTED_OOF": False,
        "APPROXIMATE_BOUNDS_FAILED": False,
    }
    if (
        type(status) is not str
        or status not in allowed
        or admitted is not allowed[status]
        or config.model.get("admission_fallback")
        != "exact_B_for_nonadmission_except_no_nonzero_safe_oof_coverage_aborts_before_target_actions"
        or config.model.get(
            "no_nonzero_safe_oof_coverage_aborts_before_target_actions"
        )
        is not True
    ):
        raise ProtocolError("HARP v20 source policy admission contract drifted.")
    if status != "NO_NONZERO_SAFE_OOF_COVERAGE":
        return
    nonadmission_path = root / "reports/source_policy_nonadmission.json"
    atomic_json(
        nonadmission_path,
        {
            "schema_version": "midogpp_harp_v20_source_policy_nonadmission_v1",
            "config_hash": config.config_hash,
            "status": status,
            "source_policy_admission_seal_hash": policy_admission["seal_hash"],
            "target_actions_constructed": False,
            "target_evaluation_labels_opened": False,
            "terminal_fallback_allowed": False,
            "lease_must_be_exhausted": True,
        },
    )
    durable_barrier((policy_path, nonadmission_path))
    raise ProtocolError(
        "HARP v20 source policy has no nonzero safe OOF coverage; "
        "target action construction is forbidden."
    )


def _pipeline_or_production(pipeline: HarpV20Pipeline | None) -> HarpV20Pipeline:
    if pipeline is not None:
        return pipeline
    from ...runtime.harp_v20_execution.production import HarpV20ProductionPipeline

    return HarpV20ProductionPipeline(
        development_role=SOURCE_TRAIN_ROLE,
        evaluation_role=TARGET_EVALUATION_ROLE,
    )


def _capacity_certificate(config: HarpStage90V20Config) -> dict[str, object]:
    centers = tuple(str(value) for value in config.protocol["centers"])
    certificate = dict(build_action_capacity_certificate(centers=centers))
    validate_action_capacity_certificate(certificate, centers=centers)
    expected_maxima = {
        "source_train": config.protocol[
            "source_train_max_required_rows_per_source_per_class"
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
        or certificate["source_train_action_count"] != 90
        or certificate["target_action_count"] != 90
        or config.protocol["prelease_action_capacity_certificate_required"]
        is not True
    ):
        raise ProtocolError("HARP v20 config/capacity certificate binding drifted.")
    return certificate


def inspect_harp_stage90_v20(
    config: HarpStage90V20Config,
) -> Mapping[str, object]:
    """Inspect architecture identity without resolving a scientific path."""

    if type(config) is not HarpStage90V20Config:
        raise ProtocolError("HARP v20 inspection requires a typed configuration.")
    runtime_hash_contract = dict(runtime_hash_contract_payload())
    capacity = _capacity_certificate(config)
    body = {
        "schema_version": "midogpp_harp_v20_implementation_inspection_v1",
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
        "source_train_target_classifier_task_count": 81,
        "total_classifier_fit_count": 810,
        "H_q_r_seven_expert_folds_used": False,
        "execution_authorized": config.execution_authorized,
        "authorization_probed": False,
        "paths_resolved": False,
        "filesystem_mutations": 0,
        "source_train_labels_opened": False,
        "evaluation_labels_opened": False,
        "old_aggregate_utility_surface_used": False,
        "predecessor_policy_cache_output_or_authority_used": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "may_feed_another_experiment": False,
    }
    return {**body, "inspection_hash": canonical_hash(body)}


def dry_run_harp_stage90_v20(
    config: HarpStage90V20Config,
    *,
    artifact_root: str | Path,
    pipeline: HarpV20Pipeline | None = None,
) -> Mapping[str, object]:
    """Mutation-free readiness check; planned mode opens no referenced path."""

    if type(config) is not HarpStage90V20Config:
        raise ProtocolError("HARP v20 dry run requires a typed configuration.")
    runtime_hash_contract = dict(runtime_hash_contract_payload())
    capacity = _capacity_certificate(config)
    if not config.execution_authorized:
        body = {
            "schema_version": "midogpp_harp_v20_mutation_free_dry_run_v1",
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
            "source_train_labels_opened": False,
            "evaluation_labels_opened": False,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "fresh_evidence": False,
        }
        return {**body, "dry_run_hash": canonical_hash(body)}

    authority = V20_RUNNER_SERVICES.load_authorization(config)
    _exact_output_root(config, artifact_root)
    parent_hash = _validate_parent_ledger(config)
    cache = V20_RUNNER_SERVICES.load_cache_index(config)
    preflight = dict(_pipeline_or_production(pipeline).preflight(config, cache))
    _validate_preflight(preflight)
    if preflight.get("action_capacity_certificate") != capacity:
        raise ProtocolError("HARP v20 preflight capacity certificate drifted.")
    body = {
        "schema_version": "midogpp_harp_v20_mutation_free_dry_run_v1",
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
        "source_train_labels_opened": False,
        "evaluation_labels_opened": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
    }
    return {**body, "dry_run_hash": canonical_hash(body)}


def run_harp_stage90_v20(
    config: HarpStage90V20Config,
    *,
    artifact_root: str | Path,
    pipeline: HarpV20Pipeline | None = None,
    confirmation_token: str | None = None,
) -> str:
    """Execute one separately authorized terminal consumed-test diagnostic."""

    # This must precede config typing, authority probing, path resolution, input
    # access, scratch allocation, lease claim, and every filesystem mutation.
    if confirmation_token != HARP_V20_RUN_CONFIRMATION_TOKEN:
        raise ProtocolError(
            "HARP v20 execution requires the exact confirmation token "
            f"{HARP_V20_RUN_CONFIRMATION_TOKEN}."
        )
    if type(config) is not HarpStage90V20Config:
        raise ProtocolError("HARP v20 execution requires a typed configuration.")

    runtime_hash_contract = dict(runtime_hash_contract_payload())
    capacity = _capacity_certificate(config)
    authority = V20_RUNNER_SERVICES.load_authorization(config)
    root = _exact_output_root(config, artifact_root)
    parent_hash = _validate_parent_ledger(config)
    cache = V20_RUNNER_SERVICES.load_cache_index(config)
    active_pipeline = _pipeline_or_production(pipeline)
    preflight = dict(active_pipeline.preflight(config, cache))
    _validate_preflight(preflight)
    if preflight.get("action_capacity_certificate") != capacity:
        raise ProtocolError("HARP v20 preflight capacity certificate drifted.")

    admission = {
        "schema_version": "midogpp_harp_v20_execution_admission_v1",
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
        "source_train_labels_opened": False,
        "evaluation_labels_opened": False,
    }
    admission_hash = canonical_hash(admission)
    output_state = _validate_output_state(root, admission_hash=admission_hash)
    lease: Any | None = None
    ledger = PhaseLedger()
    try:
        lease = V20_RUNNER_SERVICES.claim_authorization(
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
                "schema_version": "midogpp_harp_v20_protocol_manifest_v1",
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
                "known_center_source_train_to_full_test_estimand": True,
                "source_train_candidate_pool": "C_minus_q",
                "target_evaluation_candidate_pool": "C_minus_H",
                "pooled_source_policy_count": 1,
                "H_q_r_seven_expert_folds_used": False,
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

        ledger.advance("LABEL_FREE_SOURCE_TARGET_PHYSICAL_MENUS")
        _announce("LABEL_FREE_SOURCE_TARGET_PHYSICAL_MENUS")
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
            active_pipeline.compile_label_free_source_target_menus(
                menus, scratch_root=scratch
            )
        )
        bundles = tuple(bundles)
        if tuple(row.center_id for row in bundles) != centers:
            raise ProtocolError("HARP v20 compiled source/target inventory drifted.")
        compatibility_receipt = _write_or_validate_artifact(
            root / "stores/label_free_compatibility",
            compatibility,
            role="source_target_label_free_compatibility",
        )

        attestation = active_pipeline.physical_input_receipt.bank_independence
        seal_sets = tuple(
            write_source_target_surface_seals(
                root / "manifests/source_target_role_seals",
                bundle=bundle,
                physical_store_receipt=menu_receipts[index],
                fixed_bank_independence=attestation,
            )
            for index, bundle in enumerate(bundles)
        )
        source_train_index, target_index, attestation_index = build_surface_seal_indexes(
            seal_sets
        )
        source_train_index_path = root / "manifests/source_train_menu_seals.json"
        target_index_path = root / "manifests/target_evaluation_menu_seals.json"
        attestation_index_path = root / "manifests/bank_independence_attestations.json"
        _persist_or_validate_json(source_train_index_path, source_train_index)
        _persist_or_validate_json(target_index_path, target_index)
        _persist_or_validate_json(attestation_index_path, attestation_index)
        durable_barrier(
            (
                *(receipt.manifest_path for receipt in menu_receipts),
                *(receipt.npz_path for receipt in menu_receipts),
                compatibility_receipt.manifest_path,
                compatibility_receipt.npz_path,
                *(row.source_train_menu_seal_path for row in seal_sets),
                *(row.target_evaluation_menu_seal_path for row in seal_sets),
                *(row.bank_independence_attestation_path for row in seal_sets),
                source_train_index_path,
                target_index_path,
                attestation_index_path,
                root / "manifests/center_menu_root_binding.json",
            )
        )
        ledger.advance("SOURCE_TARGET_MENUS_SEALED")
        _announce("SOURCE_TARGET_MENUS_SEALED")
        ledger.advance("FIXED_BANK_INDEPENDENCE_ATTESTED")
        _announce("FIXED_BANK_INDEPENDENCE_ATTESTED")

        source_train_labels: dict[str, Sequence[object]] = {}
        capability_hashes: dict[str, str] = {}
        label_index_path = config.resolved_path("development_manifest_path")
        label_index_sha256 = str(
            config.expected_hashes["development_manifest_sha256"]
        )
        source_train_label_access_fence = begin_source_train_label_access(
            root,
            config_hash=config.config_hash,
            admission_hash=admission_hash,
            authorization_lease_hash=lease.lease_hash,
            ordered_center_ids=centers,
            source_train_surface_seal_index=source_train_index,
            source_train_surface_seal_index_path=source_train_index_path,
            target_surface_seal_index=target_index,
            target_surface_seal_index_path=target_index_path,
            bank_independence_index=attestation_index,
            bank_independence_index_path=attestation_index_path,
            label_index_sha256=label_index_sha256,
        )
        capability_set = issue_source_train_label_capabilities(
            seal_sets=seal_sets,
            label_index_path=label_index_path,
            label_index_sha256=label_index_sha256,
            source_train_label_access_fence=source_train_label_access_fence,
        )
        ledger.advance("SOURCE_TRAIN_LABEL_CAPABILITIES_OPENED")
        _announce("SOURCE_TRAIN_LABEL_CAPABILITIES_OPENED")
        for center in centers:
            capability = capability_set.for_center(center)
            source_train_labels[center] = (
                V20_RUNNER_SERVICES.load_source_train_labels(
                    config,
                    cache,
                    allowed_centers=(center,),
                    source_label_capability=capability,
                )
            )
            capability_hashes[center] = capability.capability_hash
        source_access_path = root / "reports/source_train_label_access.json"
        atomic_json(
            source_access_path,
            {
                "schema_version": "midogpp_harp_v20_source_train_label_access_v1",
                "source_train_surface_seal_index_hash": source_train_index["index_hash"],
                "target_surface_seal_index_hash": target_index["index_hash"],
                "bank_independence_index_hash": attestation_index["index_hash"],
                "source_center_capability_hashes": capability_hashes,
                "source_center_scoped_access_count": len(capability_hashes),
                "capability_set_hash": capability_set.capability_set_hash,
                "exactly_one_capability_per_source_center": True,
                "exact_source_center_coverage": tuple(capability_hashes) == centers,
                "source_train_label_access_fence_hash": (
                    source_train_label_access_fence.fence_hash
                ),
                "source_train_label_access_fence_sha256": (
                    source_train_label_access_fence.sha256
                ),
                "source_train_labels_opened": True,
                "source_train_labels_may_update": "POOLED_ROUTER_ONLY",
                "evaluation_labels_opened": False,
            },
        )

        source_train_surface = active_pipeline.build_source_train_case_surface(
            bundles, source_train_labels
        )
        source_train_labels.clear()
        del capability_set
        source_train_receipt = write_artifact_value(
            root / "stores/source_train_case_surface",
            source_train_surface,
            role="source_train_case_surface",
        )

        ledger.advance("POOLED_SOURCE_ROUTER_FIT")
        _announce("POOLED_SOURCE_ROUTER_FIT")
        fitted = active_pipeline.fit_pooled_source_router(
            source_train_surface, config=config
        )
        model_receipt = write_artifact_value(
            root / "stores/pooled_router_policy",
            fitted,
            role=POOLED_POLICY_ARTIFACT_ROLE,
        )
        model_lock = build_model_lock(
            config_hash=config.config_hash,
            centers=centers,
            source_train_surface=source_train_surface,
            source_train_receipt=source_train_receipt,
            fitted=fitted,
            model_receipt=model_receipt,
            compatibility=compatibility,
            compatibility_receipt=compatibility_receipt,
        )
        model_lock_path = root / "manifests/model_lock.json"
        atomic_json(model_lock_path, model_lock)
        write_source_diagnostics(
            root, fitted=fitted, source_surface=source_train_surface,
            config_hash=config.config_hash,
        )
        del source_train_surface

        ledger.advance("SOURCE_ONLY_POLICY_RISK_COVERAGE_ADMISSION")
        _announce("SOURCE_ONLY_POLICY_RISK_COVERAGE_ADMISSION")
        policy_admission = build_policy_admission_seal(
            config_hash=config.config_hash, fitted=fitted
        )
        policy_path = root / "manifests/source_policy_admission_seal.json"
        atomic_json(policy_path, policy_admission)
        durable_barrier((model_lock_path, policy_path))
        _enforce_source_policy_admission(
            config=config,
            root=root,
            policy_path=policy_path,
            policy_admission=policy_admission,
        )

        ledger.advance("TARGET_ACTIONS_COMPLETE")
        _announce("TARGET_ACTIONS_COMPLETE")
        target_actions = active_pipeline.build_complete_target_case_actions(
            bundles, fitted, config=config
        )
        target_receipt = write_artifact_value(
            root / "stores/target_case_actions",
            target_actions,
            role=TARGET_EVALUATION_ACTION_ARTIFACT_ROLE,
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
        enforce_admitted_target_coverage(
            root, routes=routes, policy_admission=policy_admission
        )
        _validate_route_bindings(
            routes,
            model_hash=str(fitted.manifest["model_hash"]),
            target_action_hash=str(target_actions.manifest["target_action_hash"]),
            centers=centers,
        )
        if len(routes.cases) != target_actions.manifest.get("target_case_count"):
            raise ProtocolError(
                "HARP v20 routed inventory differs from the sealed target surface."
            )
        route_root = root / "stores/prelabel_routes"
        route_receipt = write_prelabel_routes(route_root, routes)
        reconstructed = read_prelabel_routes(route_root)
        if reconstructed.route_hash != routes.route_hash:
            raise ProtocolError("HARP v20 prelabel route store changed identity.")
        route_summary = _prelabel_route_summary(routes)
        rejection_diagnostics = dict(build_prelabel_diagnostics(routes))
        rejection_path = root / "reports/prelabel_rejection_diagnostics.json"
        atomic_json(rejection_path, rejection_diagnostics)
        prelabel = build_prelabel_bundle(
            config_hash=config.config_hash,
            centers=centers,
            source_train_surface_seal_hash=str(source_train_index["index_hash"]),
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
                source_access_path,
                source_train_receipt.manifest_path,
                source_train_receipt.npz_path,
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

        validations = run_two_fresh_pooled_policy_validations(
            route_root=route_root,
            menu_binding=menu_binding,
            model_root=root / "stores/pooled_router_policy",
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
            raise ProtocolError("HARP v20 frozen route seal failed readback.")
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
        evaluation_truth = V20_RUNNER_SERVICES.load_evaluation_truth(
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
            development_surface_seal_hash=source_train_index["index_hash"],
            model_lock_hash=model_lock["model_lock_hash"],
            target_action_seal_hash=target_seal["seal_hash"],
            validations=validations,
            route_summary=route_summary,
        )
        durable_barrier(terminal_reports.paths)
        ledger.advance("TERMINAL_DIAGNOSTIC_COMPLETE")
        _announce("TERMINAL_DIAGNOSTIC_COMPLETE")

        lease = V20_RUNNER_SERVICES.finalize_authorization(
            lease, status="COMPLETE_EXHAUSTED"
        )
        finalization = read_json(lease.root / "lease.json")
        if finalization.get("status") != "COMPLETE_EXHAUSTED":
            raise ProtocolError("HARP v20 authorization did not finalize.")
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
                finalize_authorization=V20_RUNNER_SERVICES.finalize_authorization,
                announce=_announce,
            )
        except BaseException:
            pass
        raise


__all__ = (
    "HARP_V20_RUN_CONFIRMATION_TOKEN",
    "HarpV20RunnerServices",
    "IMPLEMENTED_COMPONENTS",
    "V20_RUNNER_SERVICES",
    "dry_run_harp_stage90_v20",
    "inspect_harp_stage90_v20",
    "run_harp_stage90_v20",
)
