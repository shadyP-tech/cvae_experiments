"""Content-first, nonrepairing reconstruction of the complete diagnostic."""

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
from .bundle import assert_closed_world, validate_content_index
from .constants import (
    EXPECTED_ENDPOINT_MODEL_FIT_COUNT,
    EXPECTED_ORDERED_VOTER_COUNT,
    EXPECTED_OUTER_PLAN_COUNT,
    EXPECTED_UNORDERED_PAIR_COUNT,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
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
from .physical_runtime import (
    physical_partition_hash,
    probability_index_rows,
)
from .preflight import load_validated_workstation_preflight
from .probability_surface import build_physical_probability_surface
from .protocol import build_frozen_protocol
from .reports import leakage_report_payload, publication_decision_payload


VALIDATION_SCHEMA = "fixed_bank_nested_regret_validation_v1"
RECONSTRUCTIVE_MEMBERS = (
    "manifests/protocol_manifest.json",
    "manifests/action_library.json",
    "manifests/physical_surface_seal.json",
    "manifests/nested_plan_seal.json",
    "manifests/policy_menu.json",
    "manifests/decision_barrier.json",
    "manifests/preterminal_aggregate_seal.json",
    "manifests/terminal_evaluation_seal.json",
    "tables/exact_nine_probability_index.json",
    "tables/outer_plans.json",
    "tables/unordered_pair_plans.json",
    "tables/candidate_descriptors.json",
    "tables/donor_regret_rows.json",
    "tables/regret_models.json",
    "tables/route_decisions.json",
    "tables/center_block_feasibility.json",
    "tables/terminal_method_metrics.json",
    "tables/terminal_center_contrasts.json",
    "tables/terminal_case_oracles.json",
    "tables/selection_control.json",
    "reports/diagnostic_summary.json",
    "reports/label_capability_report.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
)


def validate_nested_donor_endpoint_regret_bundle(
    root: str | Path,
    *,
    config: object,
    allow_pending_validation: bool = False,
) -> Mapping[str, object]:
    path = Path(root).resolve()
    protocol = build_frozen_protocol()
    assert_closed_world(
        path,
        allow_pending_validation=allow_pending_validation,
    )
    content = validate_content_index(
        path,
        config_contract_hash=str(getattr(config, "contract_hash")),
        protocol_contract_hash=protocol.protocol_hash,
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
    _action_payload, action_hash = validate_action_library(
        action_library_by_target()
    )
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
    physical = SimpleNamespace(
        canonical_source_cache=source,
        prediction=prediction,
    )
    with tempfile.TemporaryDirectory(prefix="nested-regret-reconstruction-") as tmp:
        reconstructed = Path(tmp)
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
            if not observed.is_file() or observed.is_symlink() or observed.read_bytes() != expected.read_bytes():
                raise ProtocolError(
                    f"Nested-regret reconstruction disagrees: {member}."
                )

    if (
        len(preterminal.plans.outer_plans) != EXPECTED_OUTER_PLAN_COUNT
        or len(preterminal.plans.unordered_pair_plans)
        != EXPECTED_UNORDERED_PAIR_COUNT
        or sum(row.ordered_voter_count for row in preterminal.endpoint_products)
        != EXPECTED_ORDERED_VOTER_COUNT
        or sum(row.endpoint_model_fit_count for row in preterminal.endpoint_products)
        != EXPECTED_ENDPOINT_MODEL_FIT_COUNT
        or any(row.selected_policy_id is not None for row in preterminal.ltt_authorizations)
        or terminal.capability_report.get("status") != "PASS"
    ):
        raise ProtocolError(
            "Nested-regret topology or center-block feasibility boundary drifted."
        )
    checks = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "PASS",
        "content_hash": content["content_hash"],
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol_contract_hash": protocol.protocol_hash,
        "workspace_binding": dict(workspace),
        "input_artifact_count": len(provenance),
        "pre_gpu_firewall_status": pre_gpu["status"],
        "workstation_preflight_status": preflight["status"],
        "physical_probability_cell_count": len(prediction.store.cells),
        "outer_route_count": len(preterminal.plans.outer_plans),
        "unordered_pair_state_count": len(preterminal.plans.unordered_pair_plans),
        "ordered_voter_count": sum(
            row.ordered_voter_count for row in preterminal.endpoint_products
        ),
        "endpoint_model_fit_count": sum(
            row.endpoint_model_fit_count for row in preterminal.endpoint_products
        ),
        "donor_regret_row_count": sum(
            len(rows) for rows in preterminal.donor_rows_by_outer_target.values()
        ),
        "route_decision_count": sum(
            len(rows) for rows in preterminal.decisions_by_policy.values()
        ),
        "ltt_authorized_target_center_count": 0,
        "terminal_evaluation_hash": terminal.evaluation_hash,
        "all_science_reconstructed_exactly": True,
        "content_index_validated_before_scientific_members": True,
        "nonrepairing_validation": True,
        "closed_world": True,
        "raw_labels_persisted": False,
        "terminal_consumed_test_diagnostic_only": True,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "routing_success_claimed": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
    }
    if allow_pending_validation:
        return checks
    from .fresh_process_validation import verify_attested_validation

    report = read_json(path / "reports/validation_report.json")
    attestation = read_json(path / "reports/fresh_process_attestation.json")
    return verify_attested_validation(
        report,
        expected_checks=checks,
        persisted_attestation=attestation,
    )


def _validate_runtime_summary(
    root: Path,
    *,
    preflight: Mapping[str, object],
    runtime: Mapping[str, object],
    source: object,
    prediction: object,
) -> Mapping[str, object]:
    payload = read_json(root / "reports/runtime_summary.json")
    exact_keys = {
        "schema_version",
        "status",
        "source_stream_lock_hash",
        "global_prediction_seal_hash",
        "source_stream_count",
        "classifier_cell_count",
        "unique_classifier_fit_count",
        "workstation_preflight_hash",
        "persistent_generation_worker_count",
        "classifier_workers",
        "route_workers",
        "route_worker_blas_threads",
        "outer_process_blas_threads",
        "gpu_generation_completed_before_cpu_phase",
        "cuda_visible_devices_during_route_phase",
        "unordered_pair_state_reused_for_both_voters",
        "endpoint_model_fit_count",
        "prior_rebinding_additional_endpoint_model_fit_count",
        "source_storage_dtype",
        "probability_storage_dtype",
        "scientific_reductions_dtype",
        "scratch_root_id",
        "scratch_role",
        "local_and_canonical_source_lock_identical",
        "previous_stage90_scratch_reused",
        "recomputed_from_original_six_inputs",
    }
    if (
        set(payload) != exact_keys
        or payload.get("schema_version")
        != "fixed_bank_nested_regret_runtime_summary_v1"
        or payload.get("status") != "PASS"
        or payload.get("source_stream_lock_hash") != getattr(source, "lock_hash")
        or payload.get("global_prediction_seal_hash")
        != getattr(prediction, "seal_hash")
        or payload.get("source_stream_count") != 81
        or payload.get("classifier_cell_count") != 810
        or payload.get("unique_classifier_fit_count") != 810
        or payload.get("workstation_preflight_hash") != canonical_hash(preflight)
        or payload.get("persistent_generation_worker_count")
        != runtime.get("persistent_generation_worker_count")
        or payload.get("classifier_workers") != runtime.get("classifier_workers")
        or payload.get("route_workers") != runtime.get("route_model_workers")
        or payload.get("route_worker_blas_threads")
        != runtime.get("classifier_threads_per_worker")
        or payload.get("outer_process_blas_threads") != 1
        or payload.get("gpu_generation_completed_before_cpu_phase") is not True
        or payload.get("cuda_visible_devices_during_route_phase") != ""
        or payload.get("unordered_pair_state_reused_for_both_voters") is not True
        or payload.get("endpoint_model_fit_count") != EXPECTED_ENDPOINT_MODEL_FIT_COUNT
        or payload.get("prior_rebinding_additional_endpoint_model_fit_count") != 0
        or payload.get("source_storage_dtype") != "float32"
        or payload.get("probability_storage_dtype") != "float32"
        or payload.get("scientific_reductions_dtype") != "float64"
        or payload.get("scratch_root_id") != preflight.get("scratch_root_id")
        or payload.get("scratch_role") != preflight.get("scratch_role")
        or payload.get("local_and_canonical_source_lock_identical") is not True
        or payload.get("previous_stage90_scratch_reused") is not False
        or payload.get("recomputed_from_original_six_inputs") is not True
    ):
        raise ProtocolError("Nested-regret runtime summary drifted.")
    return payload


__all__ = ("VALIDATION_SCHEMA", "validate_nested_donor_endpoint_regret_bundle")
