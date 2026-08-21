"""Content-first, nonrepairing reconstruction of the PCSI-PARC diagnostic."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from ...runtime.fixed_bank_a1_action_predictions import load_global_prediction_seal
from ...runtime.fixed_bank_a1_prediction_contracts import validate_action_library
from ...runtime.frozen_source_streams import load_frozen_source_streams
from .actions import action_library_by_target
from .bundle import (
    CONTENT_INDEX_MEMBERS,
    assert_closed_world,
    validate_content_index,
)
from .constants import (
    COMPOSED_POLICY_IDS,
    EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT,
    EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
    EXPECTED_OUTER_PLAN_COUNT,
    EXPECTED_POLICY_REPLAY_COUNT,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
    EXPECTED_UTILITY_MODEL_FIT_COUNT,
    PRIMARY_METHOD_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    UNPROJECTED_PARC_METHOD_ID,
)
from .controls import CONTROL_SPECS
from .donor_runtime import EXPECTED_PSEUDO_DONOR_SCOPE_COUNT, PARC_GEOMETRIES
from .engine import build_preterminal_result
from .evaluation import evaluate_terminal
from .hashing import canonical_hash
from .inputs import (
    assert_input_fence,
    load_label_free_test_frame,
    load_validated_locks,
    validate_active_workspace_binding,
    validate_pre_gpu_firewall,
    validate_workspace_provenance,
)
from .manifest_labels import read_scoped_manifest_labels
from .persistence import (
    persist_admission,
    persist_physical_surface,
    persist_preterminal,
    persist_terminal,
)
from .physical_runtime import physical_partition_hash, probability_index_rows
from .preflight import load_validated_workstation_preflight
from .probability_surface import build_physical_probability_surface
from .protocol import build_frozen_protocol
from .reports import (
    assert_transport_authorization_lineage_valid,
    leakage_report_payload,
    publication_decision_payload,
)
from .telemetry import (
    REQUIRED_PHASE_WORKLOAD_COUNTS,
    validate_phase_telemetry_payload,
)


VALIDATION_SCHEMA = "fixed_bank_pcsi_parc_validation_v1"
RECONSTRUCTIVE_MEMBERS = (
    "manifests/protocol_manifest.json",
    "manifests/action_library.json",
    "manifests/physical_surface_seal.json",
    "manifests/outer_plan_seal.json",
    "manifests/donor_runtime.json",
    "manifests/policy_replay_runtime.json",
    "manifests/policy_menu.json",
    "manifests/decision_barrier.json",
    "manifests/preterminal_aggregate_seal.json",
    "manifests/terminal_evaluation_seal.json",
    "tables/exact_nine_probability_index.json",
    "tables/outer_plans.json",
    "tables/double_exclusion_plans.json",
    "tables/physical_fingerprints.json",
    "tables/target_local_posterior_models.json",
    "tables/target_local_posterior_predictions.json",
    "tables/action_equivalence_classes.json",
    "tables/projected_utility_descriptors.json",
    "tables/projected_donor_utility_rows.json",
    "tables/double_excluded_prior_provenance.json",
    "tables/double_excluded_endpoint_scopes.json",
    "tables/pseudo_donor_utility_rows.json",
    "tables/projected_utility_models.json",
    "tables/projected_utility_predictions.json",
    "tables/fresh_legacy_utility_descriptors.json",
    "tables/fresh_legacy_donor_utility_rows.json",
    "tables/fresh_legacy_utility_models.json",
    "tables/fresh_legacy_utility_predictions.json",
    "tables/sample_influence_predictions.json",
    "tables/transport_descriptors.json",
    "tables/transport_screens.json",
    "tables/target_candidate_policies.json",
    "tables/pseudo_candidate_policies.json",
    "tables/policy_regret_replays.json",
    "tables/policy_authorizations.json",
    "tables/final_policy_predictions.json",
    "tables/route_decisions.json",
    "tables/terminal_method_metrics.json",
    "tables/terminal_center_contrasts.json",
    "tables/terminal_case_oracles.json",
    "tables/terminal_projected_action_diagnostics.json",
    "tables/terminal_policy_regret_diagnostics.json",
    "tables/terminal_transport_diagnostics.json",
    "tables/terminal_selected_case_diagnostics.json",
    "tables/terminal_policy_regret_centers.json",
    "tables/terminal_action_frequencies.json",
    "tables/terminal_diagnostic.json",
    "tables/selection_control.json",
    "reports/diagnostic_summary.json",
    "reports/label_capability_report.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
)

if not set(RECONSTRUCTIVE_MEMBERS) <= set(CONTENT_INDEX_MEMBERS):
    raise RuntimeError("PCSI-PARC reconstructive inventory escaped the content index.")


def validate_p_anchored_boundary_projected_pcsi_policy_regret_router_bundle(
    root: str | Path,
    *,
    config: object,
    allow_pending_validation: bool = False,
) -> Mapping[str, object]:
    path = Path(root).resolve()
    protocol = build_frozen_protocol()
    assert_closed_world(path, allow_pending_validation=allow_pending_validation)
    content = validate_content_index(
        path,
        config_contract_hash=str(getattr(config, "contract_hash")),
        protocol_contract_hash=protocol.protocol_hash,
    )
    telemetry = validate_phase_telemetry_payload(
        read_json(path / "reports/phase_telemetry.json"),
        required_counts=REQUIRED_PHASE_WORKLOAD_COUNTS,
    )
    assert_input_fence(config)
    workspace = validate_active_workspace_binding(config)
    provenance = validate_workspace_provenance(path, config)
    locks = load_validated_locks(config)
    frame = load_label_free_test_frame(config)
    pre_gpu = validate_pre_gpu_firewall(config, frame, locks)
    preflight = load_validated_workstation_preflight(
        path, runtime=getattr(config, "runtime")
    )
    source = load_frozen_source_streams(
        path,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=str(locks.generation.generation_lock_hash),
    )
    _payload, action_hash = validate_action_library(action_library_by_target())
    prediction = load_global_prediction_seal(
        path,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_partition_hash=physical_partition_hash(frame),
        expected_source_lock_hash=source.lock_hash,
        expected_action_library_hash=action_hash,
        expected_target_cache_binding_hash=frame.cache_binding_hash,
    )
    surface = build_physical_probability_surface(prediction.store)
    preterminal = build_preterminal_result(
        surface,
        lambda allowed, role: read_scoped_manifest_labels(
            config, frame, allowed_keys=allowed, role=role
        ),
        use_processes=True,
    )
    assert_transport_authorization_lineage_valid(preterminal)
    terminal = evaluate_terminal(preterminal)
    leakage = leakage_report_payload(
        probability_surface_hash=surface.surface_hash,
        preterminal=preterminal,
        capability_report=terminal.capability_report,
    )
    publication = publication_decision_payload(terminal)
    runtime_summary = _validate_runtime_summary(
        path,
        preflight=preflight,
        runtime=getattr(config, "runtime"),
        source=source,
        prediction=prediction,
    )
    _validate_policy_menu(path, preterminal=preterminal)
    physical = SimpleNamespace(canonical_source_cache=source, prediction=prediction)
    with tempfile.TemporaryDirectory(prefix="pcsi-reconstruction-") as temporary:
        reconstructed = Path(temporary)
        for directory in ("manifests", "reports", "tables"):
            (reconstructed / directory).mkdir(parents=True, exist_ok=True)
        persist_admission(
            reconstructed,
            config=config,
            protocol=protocol,
            provenance=provenance,
            frame=frame,
            pre_gpu_firewall=pre_gpu,
        )
        persist_physical_surface(
            reconstructed,
            physical=physical,
            surface=surface,
            probability_index=probability_index_rows(prediction),
        )
        persist_preterminal(reconstructed, preterminal)
        persist_terminal(
            reconstructed,
            terminal=terminal,
            leakage_report=leakage,
            publication_decision=publication,
            runtime_summary=runtime_summary,
        )
        for member in RECONSTRUCTIVE_MEMBERS:
            expected = reconstructed / member
            observed = path / member
            if (
                not observed.is_file()
                or observed.is_symlink()
                or observed.read_bytes() != expected.read_bytes()
            ):
                raise ProtocolError(
                    f"PCSI-PARC reconstruction disagrees: {member}."
                )

    checks = build_validation_checks_payload(
        content=content,
        config=config,
        protocol=protocol,
        workspace_binding=workspace,
        provenance=provenance,
        pre_gpu=pre_gpu,
        preflight=preflight,
        telemetry=telemetry,
        prediction=prediction,
        preterminal=preterminal,
        terminal=terminal,
    )
    if allow_pending_validation:
        return checks
    from .fresh_process_validation import verify_attested_validation

    return verify_attested_validation(
        read_json(path / "reports/validation_report.json"),
        expected_checks=checks,
        persisted_attestation=read_json(
            path / "reports/fresh_process_attestation.json"
        ),
    )


def build_validation_checks_payload(
    *,
    content: Mapping[str, object],
    config: object,
    protocol: object,
    workspace_binding: Mapping[str, object],
    provenance: Mapping[str, object],
    pre_gpu: Mapping[str, object],
    preflight: Mapping[str, object],
    telemetry: Mapping[str, object],
    prediction: object,
    preterminal: object,
    terminal: object,
) -> Mapping[str, object]:
    """Build deterministic checks from an already-computed in-memory result."""

    endpoint_fit_count = sum(
        row.endpoint_model_fit_count for row in preterminal.endpoint_products
    )
    utility_model_count = preterminal.donor_runtime.model_fit_count
    target_posterior_model_count = sum(
        len(rows)
        for rows in preterminal.target_posterior_models_by_control.values()
    )
    final_prediction_count = sum(
        len(rows)
        for rows in preterminal.policy_runtime.final_predictions_by_policy.values()
    )
    pseudo_donor_row_count = sum(
        len(runtime.pseudo_donor_rows_by_pair[(outer, pseudo)])
        for runtime in preterminal.donor_runtime.geometry_results.values()
        for outer in preterminal.surface.centers
        for pseudo in preterminal.surface.centers
        if pseudo != outer
    )
    double_excluded_scopes_valid = _double_excluded_scopes_valid(preterminal)
    if (
        len(preterminal.plans.outer_plans) != EXPECTED_OUTER_PLAN_COUNT
        or len(preterminal.plans.double_exclusion_plans)
        != EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT
        or endpoint_fit_count != EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT
        or utility_model_count != EXPECTED_UTILITY_MODEL_FIT_COUNT
        or not double_excluded_scopes_valid
        or target_posterior_model_count
        != EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT
        or len(preterminal.policy_runtime.replays) != EXPECTED_POLICY_REPLAY_COUNT
        or len(preterminal.policy_runtime.pseudo_candidate_policies)
        != EXPECTED_POLICY_REPLAY_COUNT
        or len(preterminal.policy_runtime.authorizations)
        != 2 * len(preterminal.surface.centers)
        or set(preterminal.policy_runtime.final_predictions_by_policy)
        != set(COMPOSED_POLICY_IDS)
        or any(
            len(preterminal.policy_runtime.final_predictions_by_policy[policy])
            != EXPECTED_OUTER_PLAN_COUNT
            for policy in COMPOSED_POLICY_IDS
        )
        or terminal.capability_report.get("status") != "PASS"
    ):
        raise ProtocolError("PCSI-PARC reconstructed topology drifted.")
    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": "PASS",
        "content_hash": content["content_hash"],
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol_contract_hash": protocol.protocol_hash,
        "workspace_binding": dict(workspace_binding),
        "input_artifact_count": len(provenance),
        "pre_gpu_firewall_status": pre_gpu["status"],
        "workstation_preflight_status": preflight["status"],
        "phase_telemetry_hash": telemetry["telemetry_hash"],
        "physical_probability_cell_count": len(prediction.store.cells),
        "outer_route_count": len(preterminal.plans.outer_plans),
        "double_exclusion_pair_count": len(preterminal.plans.double_exclusion_plans),
        "double_exclusion_state_count": len(
            preterminal.policy_runtime.pseudo_candidate_policies
        ),
        "policy_replay_count": len(preterminal.policy_runtime.replays),
        "outer_endpoint_model_fit_count": endpoint_fit_count,
        "utility_model_fit_count": utility_model_count,
        "double_excluded_prior_scope_count": len(
            preterminal.donor_runtime.pseudo_prior_provenance
        ),
        "double_excluded_endpoint_scope_count": len(
            preterminal.donor_runtime.pseudo_donor_endpoint_products
        ),
        "pseudo_donor_utility_row_count": pseudo_donor_row_count,
        "target_posterior_model_fit_count": target_posterior_model_count,
        "projected_raw_authorization_count": len(
            preterminal.policy_runtime.authorizations
        ),
        "final_policy_count": len(
            preterminal.policy_runtime.final_predictions_by_policy
        ),
        "final_case_prediction_count": final_prediction_count,
        "donor_runtime_hash": preterminal.donor_runtime.runtime_hash,
        "policy_replay_runtime_hash": preterminal.policy_runtime.runtime_hash,
        "terminal_evaluation_hash": terminal.evaluation_hash,
        "all_science_reconstructed_exactly": True,
        "content_index_validated_before_scientific_members": True,
        "nonrepairing_validation": True,
        "closed_world": True,
        "raw_labels_persisted": False,
        "terminal_consumed_test_diagnostic_only": True,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "terminal_success_gate_defined": False,
        "fresh_evidence": False,
        "routing_success_claimed": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
    }


def _double_excluded_scopes_valid(preterminal: object) -> bool:
    donor_runtime = preterminal.donor_runtime
    expected = {
        (outer, pseudo, donor)
        for outer in preterminal.surface.centers
        for pseudo in preterminal.surface.centers
        for donor in preterminal.surface.centers
        if len({outer, pseudo, donor}) == 3
    }
    if (
        len(expected) != EXPECTED_PSEUDO_DONOR_SCOPE_COUNT
        or set(donor_runtime.pseudo_prior_provenance) != expected
        or set(donor_runtime.pseudo_donor_endpoint_products) != expected
    ):
        return False
    for outer, pseudo, donor in sorted(expected):
        provenance = donor_runtime.pseudo_prior_provenance[(outer, pseudo, donor)]
        product = donor_runtime.pseudo_donor_endpoint_products[
            (outer, pseudo, donor)
        ]
        if (
            any(
                outer in centers
                or pseudo in centers
                or donor in centers
                or source in centers
                for source, centers in provenance.query_centers_by_source
            )
            or product.target_center != donor
            or product.endpoint_model_fit_count != 0
            or any(
                dict(state.donor_priors) != dict(provenance.prior_values)
                for _case, state in product.states
            )
        ):
            return False
    for geometry in PARC_GEOMETRIES:
        runtime = donor_runtime.geometry_results[geometry]
        for outer in preterminal.surface.centers:
            for pseudo in preterminal.surface.centers:
                if pseudo == outer:
                    continue
                rows = runtime.pseudo_donor_rows_by_pair[(outer, pseudo)]
                legal = set(preterminal.surface.centers).difference({outer, pseudo})
                if (
                    {row.donor_center for row in rows} != legal
                    or any(row.outer_target_center != outer for row in rows)
                ):
                    return False
    return True


def verify_completed_attested_bundle(
    root: str | Path,
    *,
    config: object,
    expected_checks: Mapping[str, object],
) -> Mapping[str, object]:
    """Verify final hashes and attestation without an additional scientific replay."""

    path = Path(root).resolve()
    protocol = build_frozen_protocol()
    assert_closed_world(path, allow_pending_validation=False)
    content = validate_content_index(
        path,
        config_contract_hash=str(getattr(config, "contract_hash")),
        protocol_contract_hash=protocol.protocol_hash,
    )
    if content.get("content_hash") != expected_checks.get("content_hash"):
        raise ProtocolError("PCSI-PARC final content hash drifted after reconstruction.")
    validate_phase_telemetry_payload(
        read_json(path / "reports/phase_telemetry.json"),
        required_counts=REQUIRED_PHASE_WORKLOAD_COUNTS,
    )
    from .fresh_process_validation import verify_attested_validation

    return verify_attested_validation(
        read_json(path / "reports/validation_report.json"),
        expected_checks=expected_checks,
        persisted_attestation=read_json(
            path / "reports/fresh_process_attestation.json"
        ),
    )


def _validate_policy_menu(root: Path, *, preterminal: object) -> None:
    payload = read_json(root / "manifests/policy_menu.json")
    controls = [row.to_payload() for row in CONTROL_SPECS]
    expected = {
        "schema_version": "fixed_bank_pcsi_parc_policy_menu_manifest_v1",
        "policy_menu_seal": dict(preterminal.policy_runtime.policy_menu_seal),
        "control_specs": controls,
        "control_count": 5,
        "control_spec_hash": canonical_hash(controls),
        "primary_policy_id": PRIMARY_METHOD_ID,
        "terminal_labels_used_to_define_policy": False,
        "terminal_diagnostics_may_change_policy": False,
    }
    if (
        len(CONTROL_SPECS) != 5
        or {row.policy_id for row in CONTROL_SPECS} != set(COMPOSED_POLICY_IDS)
        or {row.policy_id for row in CONTROL_SPECS if row.uses_policy_regret}
        != {PRIMARY_METHOD_ID, UNPROJECTED_PARC_METHOD_ID}
        or payload != expected
    ):
        raise ProtocolError("PCSI-PARC persisted control menu drifted.")


def _validate_runtime_summary(
    root: Path,
    *,
    preflight: Mapping[str, object],
    runtime: Mapping[str, object],
    source: object,
    prediction: object,
) -> Mapping[str, object]:
    payload = read_json(root / "reports/runtime_summary.json")
    expected = {
        "schema_version": "fixed_bank_pcsi_parc_runtime_summary_v1",
        "status": "PASS",
        "source_stream_lock_hash": getattr(source, "lock_hash"),
        "global_prediction_seal_hash": getattr(prediction, "seal_hash"),
        "source_stream_count": 81,
        "classifier_cell_count": 810,
        "unique_classifier_fit_count": 810,
        "workstation_preflight_hash": canonical_hash(preflight),
        "persistent_generation_worker_count": 2,
        "classifier_workers": int(runtime["classifier_workers"]),
        "route_workers": int(runtime["route_model_workers"]),
        "route_worker_blas_threads": int(runtime["classifier_threads_per_worker"]),
        "outer_process_blas_threads": int(runtime["classifier_threads_per_worker"]),
        "target_posterior_process_blas_threads": int(
            runtime["target_posterior_threads_per_worker"]
        ),
        "utility_process_blas_threads": int(runtime["utility_threads_per_worker"]),
        "policy_replay_process_blas_threads": int(
            runtime["policy_replay_threads_per_worker"]
        ),
        "posterior_utility_replay_workers": int(
            runtime["posterior_utility_replay_workers"]
        ),
        "gpu_generation_completed_before_cpu_phase": True,
        "cuda_visible_devices_during_route_phase": "",
        "double_exclusion_pair_count": EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT,
        "double_exclusion_state_count": EXPECTED_POLICY_REPLAY_COUNT,
        "policy_replay_count": EXPECTED_POLICY_REPLAY_COUNT,
        "projected_policy_replay_count": EXPECTED_POLICY_REPLAY_COUNT // 2,
        "raw_full_action_policy_replay_count": EXPECTED_POLICY_REPLAY_COUNT // 2,
        "unused_nested_endpoint_fits_eliminated": True,
        "outer_endpoint_model_fit_count": EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
        "utility_model_fit_count": EXPECTED_UTILITY_MODEL_FIT_COUNT,
        "target_posterior_model_fit_count": EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
        "prior_rebinding_additional_endpoint_model_fit_count": 0,
        "source_storage_dtype": "float32",
        "probability_storage_dtype": "float32",
        "scientific_reductions_dtype": "float64",
        "scratch_root_id": preflight.get("scratch_root_id"),
        "scratch_role": preflight.get("scratch_role"),
        "local_and_canonical_source_lock_identical": True,
        "previous_stage90_scratch_reused": False,
        "recomputed_from_original_six_inputs": True,
    }
    if payload != expected:
        raise ProtocolError("PCSI-PARC runtime summary drifted.")
    return payload


validate_pcsi_bundle = (
    validate_p_anchored_boundary_projected_pcsi_policy_regret_router_bundle
)


__all__ = (
    "RECONSTRUCTIVE_MEMBERS",
    "VALIDATION_SCHEMA",
    "build_validation_checks_payload",
    "validate_p_anchored_boundary_projected_pcsi_policy_regret_router_bundle",
    "validate_pcsi_bundle",
    "verify_completed_attested_bundle",
)
