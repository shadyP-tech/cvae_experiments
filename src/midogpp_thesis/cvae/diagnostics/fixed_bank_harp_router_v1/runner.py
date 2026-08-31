"""Production orchestrator for the terminal consumed-test HARP sensitivity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_action_model import (
    model_bank_collection_payload,
    training_observation_surface_payload,
    training_observations_from_surfaces,
)
from ...routing.harp_action_surface import build_directional_response_surface
from ...routing.harp_protocol import (
    HarpSourceLabelCapability,
    build_durable_prediction_seal,
    canonical_hash,
)
from ...routing.harp_replay import (
    evaluate_harp_replay,
    freeze_harp_predictions,
    issue_harp_replay_capability,
)
from ...runtime.artifact_io import atomic_json, read_json, sha256_file
from .authorization import (
    HarpAuthorization,
    HarpAuthorizationLease,
    claim_authorization,
    finalize_authorization,
    load_authorization,
)
from .config import HarpStage90Config
from .identity import (
    EXPERIMENT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from .input_surfaces import (
    HarpConsumedCacheIndex,
    load_cache_index,
    load_development_labels,
    load_evaluation_truth,
)
from .modeling import (
    fit_outer_model_banks,
    policy_hash,
    select_and_route_pair,
)
from .physical_menu import (
    build_physical_plan,
    materialize_physical_harp_menu,
    validate_physical_inputs,
)
from .reporting import (
    decision_payload,
    replay_payload,
    route_reason_summary,
    validate_prelabel_bundle,
)
from .surfaces import build_development_feature_surface, build_target_actions
from .terminal_diagnostics import build_terminal_action_diagnostics


IMPLEMENTED_COMPONENTS = (
    "new_harp_specific_single_use_authorization",
    "physical_probability_menu_from_bank_generation_lock_and_label_blind_cache",
    "strict_outer_H_query_candidate_and_delete_donor_exclusion",
    "exact_nine_per_sample_float64_reduction",
    "phase_local_prevalidated_O1_target_probability_menu_index",
    "source_standardized_correctness_and_separate_proper_loss_responses",
    "matched_budget_uniform_U_reference_for_all_candidate_utility_deltas",
    "predictive_probability_ensembling_with_required_lambda_one_physical_ablation",
    "prelabel_sealed_lambda_one_reference_preserving_U_ablation_vector",
    "nested_center_lodo_partial_pool_ridge",
    "singleton_byte_equivalent_batched_delete_donor_scoring",
    "shared_scoring_for_predictive_and_lambda_one_projections_per_reconstruction",
    "conservative_gain_loss_coverage_and_leverage_gates",
    "label_free_support_envelope_as_post_selection_veto_only",
    "globally_durable_route_vectors_before_evaluation_labels",
    "byte_identical_exact_B_fallback",
    "case_equal_within_center_and_center_equal_terminal_scoring",
)


class HarpStage90RunnerConfig(Protocol):
    artifact_root: str
    config_hash: str
    execution_authorized: bool
    input_artifact_ids: tuple[str, ...]
    expected_hashes: Mapping[str, str | None]
    protocol: Mapping[str, object]
    claim_boundary: Mapping[str, object]
    alpha_grid: tuple[float, ...]
    runtime: Mapping[str, object]
    policy: object

    def resolved_path(self, role: str) -> Path: ...


@dataclass(frozen=True, slots=True)
class HarpStage90RunnerServices:
    """Execution identity and effect edges around the shared numerical core."""

    config_type: type
    authorization_type: type
    lease_type: type
    experiment_id: str
    publication_status: str
    terminal_decision: str
    execution_revision: str
    phase_prefix: str
    load_authorization: Callable[[Any], Any]
    claim_authorization: Callable[..., Any]
    finalize_authorization: Callable[..., Any]
    load_cache_index: Callable[[Any], HarpConsumedCacheIndex]
    load_development_labels: Callable[[Any, HarpConsumedCacheIndex], Any]
    load_evaluation_truth: Callable[[Any, HarpConsumedCacheIndex], Any]


V1_RUNNER_SERVICES = HarpStage90RunnerServices(
    config_type=HarpStage90Config,
    authorization_type=HarpAuthorization,
    lease_type=HarpAuthorizationLease,
    experiment_id=EXPERIMENT_ID,
    publication_status=PUBLICATION_STATUS,
    terminal_decision=TERMINAL_DECISION,
    execution_revision="v1_original_execution",
    phase_prefix="harp-stage90-v1",
    load_authorization=load_authorization,
    claim_authorization=claim_authorization,
    finalize_authorization=finalize_authorization,
    load_cache_index=load_cache_index,
    load_development_labels=load_development_labels,
    load_evaluation_truth=load_evaluation_truth,
)


def _services_or_v1(
    services: HarpStage90RunnerServices | None,
) -> HarpStage90RunnerServices:
    return V1_RUNNER_SERVICES if services is None else services


def inspect_harp_stage90(
    config: HarpStage90RunnerConfig,
    *,
    services: HarpStage90RunnerServices | None = None,
) -> Mapping[str, object]:
    """Inspect the planned or authorized identity without resolving inputs."""

    active = _services_or_v1(services)
    if not isinstance(config, active.config_type):
        raise ProtocolError("HARP Stage-90 inspection requires a typed config.")
    body = {
        "schema_version": "midogpp_harp_stage90_implementation_inspection_v1",
        "status": (
            "EXECUTABLE_AUTHORIZED_UNPROBED"
            if config.execution_authorized
            else "PLANNED_NEEDS_NEW_EXECUTION_AMENDMENT"
        ),
        "experiment_id": active.experiment_id,
        "execution_revision": active.execution_revision,
        "config_hash": config.config_hash,
        "implemented_components": list(IMPLEMENTED_COMPONENTS),
        "physical_plan": dict(build_physical_plan()),
        "input_artifact_ids": list(config.input_artifact_ids),
        "execution_authorized": config.execution_authorized,
        "authorization_probed": False,
        "paths_resolved": False,
        "filesystem_mutations": 0,
        "development_labels_opened": False,
        "evaluation_labels_opened": False,
        "old_aggregate_utility_surface_used": False,
        "predecessor_policy_or_authority_used": False,
        "publication_status": active.publication_status,
        "terminal_decision": active.terminal_decision,
        "fresh_evidence": False,
        "may_feed_another_experiment": False,
    }
    return {**body, "inspection_hash": canonical_hash(body)}


def dry_run_harp_stage90(
    config: HarpStage90RunnerConfig,
    *,
    artifact_root: str | Path,
    services: HarpStage90RunnerServices | None = None,
) -> Mapping[str, object]:
    """Mutation-free readiness check; planned configs remain inspectable."""

    active = _services_or_v1(services)
    if not isinstance(config, active.config_type):
        raise ProtocolError("HARP Stage-90 dry run requires a typed config.")
    if not config.execution_authorized:
        body = {
            "schema_version": "midogpp_harp_stage90_mutation_free_dry_run_v1",
            "status": "NEEDS_EXECUTION_AMENDMENT",
            "experiment_id": active.experiment_id,
            "execution_revision": active.execution_revision,
            "config_hash": config.config_hash,
            "artifact_root_argument": str(artifact_root),
            "execution_authorized": False,
            "authorization_probed": False,
            "input_paths_resolved": False,
            "filesystem_mutations": 0,
            "development_labels_opened": False,
            "evaluation_labels_opened": False,
            "publication_status": active.publication_status,
            "terminal_decision": active.terminal_decision,
            "fresh_evidence": False,
        }
        return {**body, "dry_run_hash": canonical_hash(body)}
    authorization = active.load_authorization(config)
    root = _exact_output_root(config, artifact_root)
    _assert_pristine_output(root)
    _validate_parent_ledger(config)
    cache = active.load_cache_index(config)
    physical_inputs = validate_physical_inputs(config, cache)
    plan = build_physical_plan()
    body = {
        "schema_version": "midogpp_harp_stage90_mutation_free_dry_run_v1",
        "status": "PASS",
        "experiment_id": active.experiment_id,
        "execution_revision": active.execution_revision,
        "config_hash": config.config_hash,
        "artifact_root": str(root),
        "execution_authorized": True,
        **_authorization_provenance(authorization),
        "cache_hash": cache.cache_hash,
        "physical_input_receipt_hash": physical_inputs.receipt_hash,
        "physical_plan_hash": plan["plan_hash"],
        "physical_menu_materialized": False,
        "global_action_count": plan["action_count"],
        "global_exact_nine_cell_count": plan["exact_nine_cell_count"],
        "filesystem_mutations": 0,
        "development_labels_opened": False,
        "evaluation_labels_opened": False,
        "publication_status": active.publication_status,
        "terminal_decision": active.terminal_decision,
        "fresh_evidence": False,
    }
    return {**body, "dry_run_hash": canonical_hash(body)}


def run_harp_stage90(
    config: HarpStage90RunnerConfig,
    *,
    artifact_root: str | Path,
    services: HarpStage90RunnerServices | None = None,
) -> str:
    """Execute one authorized, terminal consumed-test HARP sensitivity run."""

    # Authority is deliberately checked before output paths or scientific inputs.
    active = _services_or_v1(services)
    if not isinstance(config, active.config_type):
        raise ProtocolError("HARP Stage-90 execution requires a typed config.")
    authorization = active.load_authorization(config)
    root = _exact_output_root(config, artifact_root)
    _assert_pristine_output(root)
    ledger_hash = _validate_parent_ledger(config)
    cache = active.load_cache_index(config)
    physical_inputs = validate_physical_inputs(config, cache)
    physical_plan = build_physical_plan()
    admission = {
        "schema_version": "midogpp_harp_stage90_execution_admission_v1",
        "experiment_id": active.experiment_id,
        "execution_revision": active.execution_revision,
        "config_hash": config.config_hash,
        "artifact_root": str(root),
        **_authorization_provenance(authorization),
        "parent_ledger_sha256": ledger_hash,
        "cache_hash": cache.cache_hash,
        "physical_input_receipt_hash": physical_inputs.receipt_hash,
        "physical_plan_hash": physical_plan["plan_hash"],
        "planned_action_count": physical_plan["action_count"],
        "planned_exact_nine_cell_count": physical_plan["exact_nine_cell_count"],
        "all_gates_read_only": True,
        "filesystem_mutations": 0,
        "evaluation_labels_opened": False,
    }
    admission_hash = canonical_hash(admission)
    lease: Any | None = None
    try:
        lease = active.claim_authorization(
            authorization, admission_hash=admission_hash
        )
        _announce("BEGIN", prefix=active.phase_prefix)
        atomic_json(root / "manifests/admission.json", {**admission, "admission_hash": admission_hash})
        atomic_json(
            root / "manifests/protocol_manifest.json",
            {
                "schema_version": "midogpp_harp_stage90_protocol_manifest_v1",
                "experiment_id": active.experiment_id,
                "execution_revision": active.execution_revision,
                "config_hash": config.config_hash,
                "protocol": dict(config.protocol),
                "claim_boundary": dict(config.claim_boundary),
                "authorization_identity": _authorization_provenance(
                    authorization
                ),
                "authorization_lease_hash": lease.lease_hash,
            },
        )

        _announce(
            "FRESH_PHYSICAL_SOURCE_STREAMS_AND_PROBABILITY_MENU",
            prefix=active.phase_prefix,
        )
        menu = materialize_physical_harp_menu(
            config,
            cache,
            root=root,
            expected_input_receipt_hash=physical_inputs.receipt_hash,
        )
        _announce(
            "GLOBAL_LABEL_FREE_PROBABILITY_MENU_DURABLY_SEALED",
            prefix=active.phase_prefix,
        )
        features = build_development_feature_surface(menu)
        feature_payload = _feature_surface_payload(features)
        feature_path = root / "surfaces/development_action_features.json"
        atomic_json(feature_path, feature_payload)
        source_seal = build_durable_prediction_seal(
            probability_surface_hash=features.surface_hash,
            upstream_prediction_seal_hash=menu.seal_hash,
            prediction_artifact_sha256=sha256_file(feature_path),
            prediction_row_count=len(features.rows),
        )
        source_seal_path = root / "manifests/development_prediction_seal.json"
        atomic_json(source_seal_path, source_seal.to_payload())
        _fsync_tree(root)

        _announce(
            "DEVELOPMENT_LABEL_CAPABILITY_OPENED_AFTER_SEAL",
            prefix=active.phase_prefix,
        )
        development_capability = HarpSourceLabelCapability(
            centers=CENTERS,
            seal=source_seal,
            seal_path=source_seal_path,
            prediction_artifact_path=feature_path,
            label_loader=lambda: active.load_development_labels(config, cache),
        )
        opened_development = development_capability.open()
        responses = build_directional_response_surface(features, opened_development)
        observations = training_observations_from_surfaces(features, responses)
        observation_payload = training_observation_surface_payload(
            observations,
            feature_surface_hash=features.surface_hash,
            response_surface_hash=responses.surface_hash,
        )
        atomic_json(root / "surfaces/training_observations.json", observation_payload)
        atomic_json(root / "reports/development_label_access.json", dict(development_capability.access_report()))

        _announce(
            "NESTED_CENTER_LODO_MODEL_BANKS_FIT",
            prefix=active.phase_prefix,
        )
        banks = fit_outer_model_banks(
            observations,
            alphas=config.alpha_grid,
            workers=int(config.runtime["cpu_model_workers"]),
        )
        fitted_policy_hash = policy_hash(
            banks, config.policy, menu_seal_hash=menu.seal_hash
        )
        model_payload = {
            "schema_version": "midogpp_harp_stage90_model_lock_v2",
            "model_bank_collection": model_bank_collection_payload(banks),
            "policy": asdict(config.policy),
            "policy_hash": fitted_policy_hash,
            "strict_outer_center_exclusion": True,
            "nested_center_lodo": True,
            "delete_donor_ensemble": True,
            "evaluation_labels_used": False,
        }
        atomic_json(root / "manifests/model_lock.json", model_payload)

        _announce(
            "GLOBAL_TARGET_ACTIONS_AND_ROUTES_BUILDING",
            prefix=active.phase_prefix,
        )
        target_actions = build_target_actions(menu)
        target_action_payload = _target_action_surface_payload(target_actions, menu.seal_hash)
        atomic_json(root / "surfaces/target_actions.json", target_action_payload)
        (
            decisions,
            vectors,
            physical_decisions,
            physical_vectors,
        ) = select_and_route_pair(
            menu,
            banks,
            target_actions,
            policy=config.policy,
            fitted_policy_hash=fitted_policy_hash,
        )
        routed_vector_payload = _routed_vector_surface_payload(
            vectors, vector_role="predictive_primary"
        )
        routed_vector_path = root / "surfaces/routed_vectors.json"
        atomic_json(routed_vector_path, routed_vector_payload)
        physical_vector_payload = _routed_vector_surface_payload(
            physical_vectors, vector_role="physical_lambda_one_ablation"
        )
        physical_vector_path = root / "surfaces/physical_ablation_vectors.json"
        atomic_json(physical_vector_path, physical_vector_payload)
        physical_reference_payload = (
            _physical_reference_preserving_surface_payload(physical_vectors)
        )
        physical_reference_path = (
            root / "surfaces/physical_ablation_reference_preserving_vectors.json"
        )
        atomic_json(physical_reference_path, physical_reference_payload)
        prelabel = {
            "schema_version": "midogpp_harp_stage90_prelabel_bundle_v2",
            "status": "DURABLE_ALL_ROUTES_SEALED_BEFORE_EVALUATION_LABELS",
            "experiment_id": active.experiment_id,
            "execution_revision": active.execution_revision,
            "config_hash": config.config_hash,
            "prediction_menu_seal_hash": menu.seal_hash,
            "development_feature_surface_hash": features.surface_hash,
            "response_surface_hash": responses.surface_hash,
            "training_surface_hash": observation_payload["training_surface_hash"],
            "model_policy_hash": fitted_policy_hash,
            "target_action_surface_hash": target_action_payload["surface_hash"],
            "routed_vector_seal_hashes": [vector.routed_vector_seal_hash for vector in vectors],
            "routed_vector_artifact_sha256": sha256_file(routed_vector_path),
            "physical_ablation_routed_vector_seal_hashes": [
                vector.routed_vector_seal_hash for vector in physical_vectors
            ],
            "physical_ablation_routed_vector_artifact_sha256": sha256_file(
                physical_vector_path
            ),
            "physical_ablation_reference_preserving_surface_hash": (
                physical_reference_payload["surface_hash"]
            ),
            "physical_ablation_reference_preserving_artifact_sha256": sha256_file(
                physical_reference_path
            ),
            "physical_ablation_action_universe": "Hxe_lambda_one_only",
            "physical_ablation_decisions": decision_payload(physical_decisions),
            "physical_ablation_selection_labels_used": False,
            "decisions": decision_payload(decisions),
            "route_reason_summary": route_reason_summary(decisions),
            "publication_status": active.publication_status,
            "terminal_decision": active.terminal_decision,
            "fresh_evidence": False,
            "evaluation_labels_opened": False,
            "exact_b_fallback_byte_identity": True,
            "may_feed_another_experiment": False,
        }
        prelabel = {**prelabel, "bundle_hash": canonical_hash(prelabel)}
        prelabel_path = root / "manifests/prelabel_route_bundle.json"
        atomic_json(prelabel_path, prelabel)
        # The two validators operate only after the complete route bundle and
        # exact routed probability vectors have crossed a durability barrier.
        _fsync_tree(root)
        validations = (
            _validate_route_reconstruction(
                prelabel_path,
                routed_vector_path,
                physical_vector_path,
                physical_reference_path,
                menu=menu,
                banks=banks,
                target_actions=target_actions,
                policy=config.policy,
                fitted_policy_hash=fitted_policy_hash,
                validator_id="fresh_route_reconstruction_A",
            ),
            _validate_route_reconstruction(
                prelabel_path,
                routed_vector_path,
                physical_vector_path,
                physical_reference_path,
                menu=menu,
                banks=banks,
                target_actions=target_actions,
                policy=config.policy,
                fitted_policy_hash=fitted_policy_hash,
                validator_id="fresh_route_reconstruction_B",
            ),
        )
        _announce(
            "GLOBAL_TARGET_ACTIONS_AND_ROUTES_DURABLY_SEALED",
            prefix=active.phase_prefix,
        )
        frozen = freeze_harp_predictions(
            decisions,
            prediction_surface_hash=menu.prediction_store_hash,
            policy_hash=fitted_policy_hash,
            durable_bundle_hash=sha256_file(prelabel_path),
            independent_validation_hashes=validations,
        )
        frozen_payload = _frozen_seal_payload(frozen, validations)
        frozen_path = root / "manifests/frozen_prediction_seal.json"
        atomic_json(frozen_path, frozen_payload)
        _fsync_tree(root)
        # A fresh read of the persisted seal is the final pre-evaluation barrier.
        if read_json(frozen_path) != frozen_payload:
            raise ProtocolError("HARP Stage-90 frozen prediction seal did not round-trip.")

        _announce(
            "ONE_SHOT_EVALUATION_LABEL_CAPABILITY_OPENED",
            prefix=active.phase_prefix,
        )
        evaluation_truth = active.load_evaluation_truth(config, cache)
        evaluation_capability = issue_harp_replay_capability(
            frozen,
            target_truth=evaluation_truth,
            authorization_hash=authorization.amendment_sha256,
        )
        result = evaluate_harp_replay(frozen, evaluation_capability)
        terminal = replay_payload(result)
        action_diagnostics = build_terminal_action_diagnostics(
            menu,
            vectors,
            physical_vectors,
            evaluation_truth,
            prelabel_bundle_hash=str(prelabel["bundle_hash"]),
            physical_reference_preserving_surface_hash=str(
                physical_reference_payload["surface_hash"]
            ),
        )
        reasons = route_reason_summary(decisions)
        atomic_json(root / "reports/terminal_result.json", terminal)
        atomic_json(
            root / "reports/action_oracle_diagnostics.json", action_diagnostics
        )
        atomic_json(root / "reports/route_and_fallback_reasons.json", reasons)
        atomic_json(
            root / "reports/leakage_report.json",
            {
                "schema_version": "midogpp_harp_stage90_leakage_report_v1",
                "strict_outer_center_exclusion": True,
                "development_evaluation_case_disjoint": True,
                "evaluation_labels_opened_after_global_route_seal": True,
                "full_action_matrix_scored_after_route_seal_only": True,
                "physical_lambda_one_ablation_sealed_before_evaluation_labels": True,
                "physical_ablation_selection_labels_used": False,
                "action_oracle_diagnostics_may_feed_policy_or_thresholds": False,
                "old_aggregate_utility_surface_used": False,
                "predecessor_policy_rank_output_or_authority_used": False,
                "seed_cells_are_inference_units": False,
                "status": "PASS",
            },
        )
        validation = {
            "schema_version": "midogpp_harp_stage90_validation_report_v1",
            "status": "PASS",
            "prediction_menu_seal_hash": menu.seal_hash,
            "frozen_prediction_seal_hash": frozen.seal_hash,
            "independent_prelabel_validation_hashes": list(validations),
            "terminal_result_hash": terminal["result_hash"],
            "action_oracle_diagnostic_hash": action_diagnostics["diagnostic_hash"],
            "action_oracle_diagnostic_only": True,
            "action_oracle_feedback_to_policy": False,
            "physical_lambda_one_ablation_sealed_before_labels": True,
            "physical_reference_preserving_surface_hash": (
                physical_reference_payload["surface_hash"]
            ),
            "exact_b_fallback_byte_identity": reasons["exact_b_fallback_byte_identity"],
            "publication_status": active.publication_status,
            "terminal_decision": active.terminal_decision,
            "fresh_evidence": False,
            "authorization_identity": _authorization_provenance(authorization),
        }
        atomic_json(root / "reports/validation_report.json", validation)
        # The global lease is finalized and durably mirrored before the output
        # COMPLETE marker.  A failure in any preceding step therefore cannot
        # leave a scientifically complete-looking artifact.
        lease = active.finalize_authorization(
            lease, status="COMPLETE_EXHAUSTED"
        )
        authorization_finalization = read_json(lease.root / "lease.json")
        if authorization_finalization.get("status") != "COMPLETE_EXHAUSTED":
            raise ProtocolError("HARP Stage-90 authorization did not finalize.")
        atomic_json(
            root / "manifests/authorization_finalization.json",
            authorization_finalization,
        )
        _write_content_index(
            root,
            publication_status=active.publication_status,
            terminal_decision=active.terminal_decision,
        )
        _fsync_tree(root)
        content_index_path = root / "manifests/content_index.json"
        atomic_json(
            root / "reports/run_state.json",
            {
                "schema_version": "midogpp_harp_stage90_run_state_v1",
                "status": "COMPLETE_EXHAUSTED",
                "phase": "TERMINAL_DIAGNOSTIC_COMPLETE",
                "authorization_lease_hash": lease.lease_hash,
                "execution_amendment_sha256": authorization.amendment_sha256,
                "execution_amendment_hash": authorization.amendment_hash,
                "scientific_contract_hash": authorization.scientific_contract_hash,
                "workspace_registration_execution_contract_hash": (
                    authorization.workspace_registration_execution_contract_hash
                ),
                "source_snapshot_tree_sha256": (
                    authorization.source_snapshot_tree_sha256
                ),
                "authorization_finalization_sha256": sha256_file(
                    root / "manifests/authorization_finalization.json"
                ),
                "content_index_sha256": sha256_file(content_index_path),
                "frozen_prediction_seal_hash": frozen.seal_hash,
                "terminal_result_hash": terminal["result_hash"],
                "final_commit": True,
            },
        )
        _fsync_file(root / "reports/run_state.json")
        return str(root)
    except BaseException as exc:
        if lease is not None:
            try:
                atomic_json(
                    root / "reports/failure_report.json",
                    {
                        "schema_version": "midogpp_harp_stage90_failure_report_v1",
                        "status": "FAILED_EXHAUSTED",
                        "error_class": exc.__class__.__name__,
                        "error": str(exc)[:2000],
                        "publication_status": active.publication_status,
                        "terminal_decision": active.terminal_decision,
                        "authorization_identity": _authorization_provenance(
                            authorization
                        ),
                    },
                )
                lease = active.finalize_authorization(
                    lease, status="FAILED_EXHAUSTED", error=str(exc)
                )
            except BaseException:
                pass
        raise


def _authorization_provenance(authorization: object) -> dict[str, object]:
    """Return the exact amendment, science, and source identity for artifacts."""

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
    if any(not hasattr(authorization, role) for role in required):
        raise ProtocolError("HARP Stage-90 authorization provenance is untyped.")
    return {
        "execution_amendment_sha256": authorization.amendment_sha256,
        "execution_amendment_hash": authorization.amendment_hash,
        "authorized_input_binding_hash": authorization.input_binding_hash,
        "scientific_contract_hash": authorization.scientific_contract_hash,
        "workspace_registration_execution_contract_hash": (
            authorization.workspace_registration_execution_contract_hash
        ),
        "source_snapshot_schema": authorization.source_snapshot_schema,
        "source_snapshot_manifest_sha256": (
            authorization.source_snapshot_manifest_sha256
        ),
        "source_snapshot_tree_sha256": authorization.source_snapshot_tree_sha256,
        "source_snapshot_member_count": authorization.source_snapshot_member_count,
    }


def _exact_output_root(config: HarpStage90RunnerConfig, value: str | Path) -> Path:
    text = str(value)
    if "://" in text:
        raise ProtocolError("HARP Stage-90 runner requires a resolved output path.")
    root = Path(text).resolve()
    configured = config.artifact_root
    if "://" not in configured and Path(configured).resolve() != root:
        raise ProtocolError("HARP Stage-90 CLI/config output roots differ.")
    return root


def _assert_pristine_output(root: Path) -> None:
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ProtocolError("HARP Stage-90 prepared output root is absent or unsafe.")
    allowed = {"config.resolved.yaml", "provenance/input_artifacts.json"}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ProtocolError("HARP Stage-90 prepared output contains a symlink.")
        if path.is_file() and path.relative_to(root).as_posix() not in allowed:
            raise ProtocolError("HARP Stage-90 output contains prior scientific state.")


def _validate_parent_ledger(config: HarpStage90RunnerConfig) -> str:
    expected = config.expected_hashes["parent_ledger_sha256"]
    if expected is None:
        raise ProtocolError("HARP Stage-90 parent-ledger hash is absent.")
    path = config.resolved_path("parent_ledger_path")
    observed = sha256_file(path)
    if observed != expected:
        raise ProtocolError("HARP Stage-90 parent-ledger bytes drifted.")
    read_json(path)
    return observed


def _feature_surface_payload(surface: object) -> dict[str, object]:
    rows = getattr(surface, "rows")
    payload = {
        "schema_version": "midogpp_harp_stage90_development_feature_surface_v2",
        "surface_hash": getattr(surface, "surface_hash"),
        "prediction_seal_hash": getattr(surface, "prediction_seal_hash"),
        "row_count": len(rows),
        "feature_hashes": [row.feature_hash for row in rows],
        "outer_targets": sorted({row.outer_target for row in rows}),
        "exact_nine_per_sample": True,
        "utility_reference_action": "U",
        "operational_fallback_action": "B",
        "seed_cells_are_model_rows": False,
        "labels_used": False,
    }
    return {**payload, "artifact_hash": canonical_hash(payload)}


def _target_action_surface_payload(actions: Sequence[object], menu_hash: str) -> dict[str, object]:
    rows = [
        {
            "action_key": list(row.action_key),
            "direction": row.direction,
            "feature_names": list(row.feature_names),
            "feature_values": list(row.feature_values),
            "baseline_probability_hex": row.baseline_probability_bytes.hex(),
            "predictive_reference_action_id": "U",
            "predictive_reference_probability_hex": row.baseline_probability_bytes.hex(),
            "operational_fallback_action_id": "B",
            "operational_fallback_probability_hex": (
                row.operational_fallback_probability_bytes.hex()
            ),
            "expert_probability": row.expert_probability,
            "ensemble_receipt_hash": row.ensemble_receipt_hash,
        }
        for row in actions
    ]
    payload = {
        "schema_version": "midogpp_harp_stage90_target_action_surface_v2",
        "prediction_menu_seal_hash": menu_hash,
        "row_count": len(rows),
        "rows": rows,
        "complete_candidate_lambda_grid": True,
        "utility_reference_action": "U",
        "operational_fallback_action": "B",
        "lambda_semantics": (
            "post_classifier_predictive_probability_ensemble_"
            "not_generated_distribution"
        ),
        "evaluation_labels_used": False,
    }
    return {**payload, "surface_hash": canonical_hash(payload)}


def _routed_vector_surface_payload(
    vectors: Sequence[object], *, vector_role: str = "predictive_primary"
) -> dict[str, object]:
    if vector_role not in {"predictive_primary", "physical_lambda_one_ablation"}:
        raise ProtocolError("HARP Stage-90 routed-vector role is unknown.")
    rows = []
    for vector in vectors:
        rows.append(
            {
                "outer_target_id": vector.decisions[0].outer_target_id,
                "decision_hashes": [row.decision_hash for row in vector.decisions],
                "baseline_probability_hex": vector.baseline_probabilities.tobytes(order="C").hex(),
                "reference_probability_hex": vector.reference_probabilities.tobytes(order="C").hex(),
                "selected_action_probability_hex": vector.selected_action_probabilities.tobytes(order="C").hex(),
                "routed_probability_hex": vector.routed_probabilities.tobytes(order="C").hex(),
                "baseline_bytes_sha256": vector.baseline_bytes_sha256,
                "reference_bytes_sha256": vector.reference_bytes_sha256,
                "selected_action_bytes_sha256": vector.selected_action_bytes_sha256,
                "routed_bytes_sha256": vector.routed_bytes_sha256,
                "decision_set_hash": vector.decision_set_hash,
                "routed_vector_seal_hash": vector.routed_vector_seal_hash,
                "fallback_byte_identity": vector.fallback_byte_identity,
            }
        )
    payload = {
        "schema_version": "midogpp_harp_stage90_routed_vector_surface_v2",
        "status": "DURABLE_BEFORE_EVALUATION_LABELS",
        "vector_role": vector_role,
        "vectors": rows,
        "vector_count": len(rows),
        "evaluation_labels_opened": False,
        "exact_b_fallback_byte_identity": all(
            bool(row["fallback_byte_identity"]) for row in rows
        ),
        "eligible_blends_use_matched_budget_U_reference": True,
        "eligible_lambda_one_only": vector_role == "physical_lambda_one_ablation",
    }
    return {**payload, "surface_hash": canonical_hash(payload)}


def _physical_reference_preserving_surface_payload(
    vectors: Sequence[object],
) -> dict[str, object]:
    rows = []
    for vector in vectors:
        vector.assert_valid()
        eligible = np.asarray(
            [row.eligible for row in vector.decisions], dtype=bool
        )
        if any(
            row.eligible and row.lambda_value != 1.0 for row in vector.decisions
        ):
            raise ProtocolError(
                "HARP Stage-90 physical reference vector escaped lambda=1."
            )
        values = np.ascontiguousarray(
            np.where(
                eligible,
                vector.routed_probabilities,
                vector.reference_probabilities,
            ),
            dtype=np.float64,
        )
        rows.append(
            {
                "outer_target_id": vector.decisions[0].outer_target_id,
                "decision_set_hash": vector.decision_set_hash,
                "probability_hex": values.tobytes(order="C").hex(),
                "probability_bytes_sha256": hashlib.sha256(
                    values.tobytes(order="C")
                ).hexdigest(),
                "eligible_count": int(np.sum(eligible)),
                "fallback_to_U_count": int(np.sum(~eligible)),
            }
        )
    payload = {
        "schema_version": (
            "midogpp_harp_stage90_physical_reference_preserving_surface_v1"
        ),
        "status": "DURABLE_BEFORE_EVALUATION_LABELS",
        "vector_role": "physical_lambda_one_reference_preserving_estimand",
        "vectors": rows,
        "vector_count": len(rows),
        "eligible_action": "Hxe_lambda_one",
        "ineligible_reference_action": "U",
        "operational_exact_B_fallback_used_for_this_estimand": False,
        "selection_labels_used": False,
        "evaluation_labels_opened": False,
    }
    return {**payload, "surface_hash": canonical_hash(payload)}


def _validate_route_reconstruction(
    prelabel_path: Path,
    routed_vector_path: Path,
    physical_vector_path: Path,
    physical_reference_path: Path,
    *,
    menu: object,
    banks: Sequence[object],
    target_actions: Sequence[object],
    policy: object,
    fitted_policy_hash: str,
    validator_id: str,
) -> str:
    """Freshly reconstruct every route and vector against durable artifacts."""

    bundle_validation_hash = validate_prelabel_bundle(
        prelabel_path, validator_id=validator_id
    )
    durable_bundle = read_json(prelabel_path)
    durable_vectors = read_json(routed_vector_path)
    durable_physical_vectors = read_json(physical_vector_path)
    durable_physical_reference = read_json(physical_reference_path)
    if durable_bundle.get("routed_vector_artifact_sha256") != sha256_file(
        routed_vector_path
    ):
        raise ProtocolError("HARP Stage-90 routed-vector artifact binding drifted.")
    if durable_bundle.get(
        "physical_ablation_routed_vector_artifact_sha256"
    ) != sha256_file(physical_vector_path):
        raise ProtocolError(
            "HARP Stage-90 physical-ablation artifact binding drifted."
        )
    if durable_bundle.get(
        "physical_ablation_reference_preserving_artifact_sha256"
    ) != sha256_file(physical_reference_path):
        raise ProtocolError(
            "HARP Stage-90 physical reference artifact binding drifted."
        )
    (
        decisions,
        vectors,
        physical_decisions,
        physical_vectors,
    ) = select_and_route_pair(
        menu,
        banks,
        target_actions,
        policy=policy,
        fitted_policy_hash=fitted_policy_hash,
    )
    reconstructed = _routed_vector_surface_payload(
        vectors, vector_role="predictive_primary"
    )
    reconstructed_physical = _routed_vector_surface_payload(
        physical_vectors, vector_role="physical_lambda_one_ablation"
    )
    reconstructed_physical_reference = (
        _physical_reference_preserving_surface_payload(physical_vectors)
    )
    if (
        decision_payload(decisions) != durable_bundle.get("decisions")
        or reconstructed != durable_vectors
        or [vector.routed_vector_seal_hash for vector in vectors]
        != durable_bundle.get("routed_vector_seal_hashes")
        or decision_payload(physical_decisions)
        != durable_bundle.get("physical_ablation_decisions")
        or reconstructed_physical != durable_physical_vectors
        or [vector.routed_vector_seal_hash for vector in physical_vectors]
        != durable_bundle.get("physical_ablation_routed_vector_seal_hashes")
        or reconstructed_physical_reference != durable_physical_reference
        or reconstructed_physical_reference.get("surface_hash")
        != durable_bundle.get(
            "physical_ablation_reference_preserving_surface_hash"
        )
    ):
        raise ProtocolError("HARP Stage-90 durable route reconstruction drifted.")
    return canonical_hash(
        {
            "schema_version": "midogpp_harp_stage90_route_reconstruction_v1",
            "validator_id": validator_id,
            "bundle_validation_hash": bundle_validation_hash,
            "routed_vector_surface_hash": reconstructed["surface_hash"],
            "physical_ablation_routed_vector_surface_hash": reconstructed_physical[
                "surface_hash"
            ],
            "physical_ablation_reference_preserving_surface_hash": (
                reconstructed_physical_reference["surface_hash"]
            ),
            "route_count": len(decisions),
            "exact_b_fallback_byte_identity": reconstructed[
                "exact_b_fallback_byte_identity"
            ],
            "evaluation_labels_opened": False,
        }
    )


def _frozen_seal_payload(frozen: object, validations: Sequence[str]) -> dict[str, object]:
    payload = {
        "schema_version": "midogpp_harp_stage90_frozen_prediction_seal_v1",
        "status": "DURABLE_ALL_ROUTES_SEALED_BEFORE_EVALUATION_LABELS",
        "seal_hash": frozen.seal_hash,
        "prediction_surface_hash": frozen.prediction_surface_hash,
        "policy_hash": frozen.policy_hash,
        "durable_bundle_hash": frozen.durable_bundle_hash,
        "independent_validation_hashes": list(validations),
        "row_count": len(frozen.decisions),
        "evaluation_labels_opened": False,
        "exact_b_fallback_byte_identity": True,
    }
    return {**payload, "persistence_hash": canonical_hash(payload)}


def _write_content_index(
    root: Path,
    *,
    publication_status: str = PUBLICATION_STATUS,
    terminal_decision: str = TERMINAL_DECISION,
) -> None:
    members = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.relative_to(root).as_posix() not in {
            "manifests/content_index.json",
            "reports/run_state.json",
        }:
            members.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
    payload = {
        "schema_version": "midogpp_harp_stage90_content_index_v1",
        "members": members,
        "publication_status": publication_status,
        "terminal_decision": terminal_decision,
        "may_feed_another_experiment": False,
        "run_state_excluded_as_final_commit": True,
    }
    atomic_json(root / "manifests/content_index.json", {**payload, "content_index_hash": canonical_hash(payload)})


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            _fsync_file(path)
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in (*directories, root):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _announce(phase: str, *, prefix: str = "harp-stage90-v1") -> None:
    print(f"[{prefix}] phase={phase}", file=sys.stderr, flush=True)


__all__ = (
    "HarpStage90RunnerConfig",
    "HarpStage90RunnerServices",
    "V1_RUNNER_SERVICES",
    "dry_run_harp_stage90",
    "inspect_harp_stage90",
    "run_harp_stage90",
)
