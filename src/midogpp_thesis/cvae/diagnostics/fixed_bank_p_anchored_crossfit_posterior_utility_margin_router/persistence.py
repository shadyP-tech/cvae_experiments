"""Phase-local persistence for reconstructive, raw-label-free PUMR artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ...runtime.artifact_io import atomic_json
from .actions import action_library_by_target
from .artifact_io import persist_json, persist_rows
from .constants import (
    BLOCKED_FINGERPRINT_CONTROL_ID,
    CENTERS,
    COMPOSED_POLICY_IDS,
    MODEL_BASED_METHOD_ID,
    PRIMARY_FINGERPRINT_CONTROL_ID,
)
from .hashing import canonical_hash, json_native
from .reports import run_state_payload


def persist_admission(
    root: Path,
    *,
    config: object,
    protocol: object,
    provenance: Mapping[str, Mapping[str, object]],
    frame: object,
    pre_gpu_firewall: Mapping[str, object],
) -> None:
    from .reports import protocol_manifest_payload

    actions = action_library_by_target()
    action_payload = {
        "schema_version": "fixed_bank_pumr_action_library_manifest_v1",
        "actions": [
            action.to_payload() for center in CENTERS for action in actions[center]
        ],
        "action_count": sum(len(actions[center]) for center in CENTERS),
        "target_expert_excluded": True,
        "labels_used": False,
    }
    action_payload["action_library_hash"] = canonical_hash(action_payload)
    persist_json(root / "manifests/action_library.json", action_payload)
    persist_json(
        root / "manifests/protocol_manifest.json",
        protocol_manifest_payload(
            config,
            protocol=protocol,
            provenance=provenance,
            cache_binding_hash=str(getattr(frame, "cache_binding_hash")),
            pre_gpu_firewall=pre_gpu_firewall,
        ),
    )


def persist_physical_surface(
    root: Path,
    *,
    physical: object,
    surface: object,
    probability_index: object,
) -> None:
    rows = tuple(probability_index)
    prediction = getattr(physical, "prediction")
    payload = {
        "schema_version": "fixed_bank_pumr_physical_surface_seal_v1",
        "source_stream_lock_hash": str(getattr(physical, "canonical_source_cache").lock_hash),
        "global_prediction_seal_hash": str(prediction.seal_hash),
        "probability_store_hash": str(prediction.store.store_hash),
        "physical_probability_surface_hash": str(getattr(surface, "surface_hash")),
        "probability_index_hash": canonical_hash([row.to_payload() for row in rows]),
        "target_action_identity_count": len(rows),
        "physical_probability_cell_count": len(prediction.store.cells),
        "all_probabilities_sealed_before_label_access": True,
        "labels_used": False,
    }
    payload["physical_surface_seal_hash"] = canonical_hash(payload)
    persist_json(root / "manifests/physical_surface_seal.json", payload)
    persist_rows(
        root / "tables/exact_nine_probability_index.json",
        rows,
        schema_version="fixed_bank_pumr_probability_index_table_v1",
    )


def persist_preterminal(root: Path, preterminal: object) -> None:
    plans = getattr(preterminal, "plans")
    outer_rows = [
        {
            "target_center": row.target_center,
            "case_id": row.case_id,
            "group_id": row.group_id,
            "support_case_ids": list(row.support_case_ids),
            "evaluation_sample_count": len(row.evaluation_sample_ids),
            "evaluation_identity_hash": canonical_hash(list(row.evaluation_sample_ids)),
            "probability_surface_hash": row.probability_surface_hash,
            "plan_hash": row.plan_hash,
            "held_case_and_group_excluded": True,
            "labels_used": False,
        }
        for row in plans.outer_plans
    ]
    plan_manifest = {
        "schema_version": "fixed_bank_pumr_outer_plan_manifest_v1",
        "probability_surface_hash": plans.probability_surface_hash,
        "outer_plan_count": len(outer_rows),
        "double_exclusion_state_count": 0,
        "outer_plan_hash": canonical_hash(outer_rows),
        "plan_seal_hash": plans.seal_hash,
        "sealed_before_any_label_access": True,
    }
    persist_json(root / "manifests/outer_plan_seal.json", plan_manifest)
    persist_rows(
        root / "tables/outer_plans.json",
        outer_rows,
        schema_version="fixed_bank_pumr_outer_plan_table_v1",
    )
    descriptors = [
        row.to_payload()
        for center in CENTERS
        for row in preterminal.utility_descriptors_by_center[center]
    ]
    donor_rows = [
        row.to_payload()
        for center in CENTERS
        for row in preterminal.donor_utility_rows_by_target[center]
    ]
    fingerprint_rows = [
        preterminal.primary_fingerprints_by_center[center].summary_payload()
        for center in CENTERS
    ] + [
        preterminal.blocked_fingerprints_by_center[center].summary_payload()
        for center in CENTERS
    ]
    posterior_models = [
        {"control_id": control, **row.to_payload()}
        for control in (
            PRIMARY_FINGERPRINT_CONTROL_ID,
            BLOCKED_FINGERPRINT_CONTROL_ID,
        )
        for row in preterminal.target_posterior_models_by_control[control]
    ]
    support_fold_plans = [
        {
            "control_id": control,
            "target_center": row.target_center,
            "held_case_id": row.held_case_id,
            "fold_id": row.fold_id,
            "training_case_ids": list(row.training_case_ids),
            "validation_case_ids": list(row.validation_case_ids),
            "fingerprint_hash": row.fingerprint_hash,
            "fold_plan_hash": row.fold_plan_hash,
            "whole_case_grouped": True,
            "held_case_labels_used": False,
        }
        for control in (
            PRIMARY_FINGERPRINT_CONTROL_ID,
            BLOCKED_FINGERPRINT_CONTROL_ID,
        )
        for row in preterminal.target_posterior_models_by_control[control]
    ]
    posterior_predictions = [
        {"control_id": control, **row.to_payload()}
        for control in (
            PRIMARY_FINGERPRINT_CONTROL_ID,
            BLOCKED_FINGERPRINT_CONTROL_ID,
        )
        for row in preterminal.target_posterior_predictions_by_control[control]
    ]
    posterior_ensembles = [
        {"control_id": control, **row.to_payload()}
        for control in (
            PRIMARY_FINGERPRINT_CONTROL_ID,
            BLOCKED_FINGERPRINT_CONTROL_ID,
        )
        for row in preterminal.route_posterior_ensembles_by_control[control]
    ]
    posterior_utilities = [
        {"utility_scope": "target_route", **row.to_payload()}
        for control in (
            PRIMARY_FINGERPRINT_CONTROL_ID,
            BLOCKED_FINGERPRINT_CONTROL_ID,
        )
        for row in preterminal.posterior_utility_predictions_by_control[control]
    ]
    donor_posterior_utilities = [
        {
            "utility_scope": "donor_margin_calibration",
            "outer_target_center": outer,
            **row.to_payload(),
        }
        for outer in CENTERS
        for control in (
            PRIMARY_FINGERPRINT_CONTROL_ID,
            BLOCKED_FINGERPRINT_CONTROL_ID,
        )
        for row in preterminal.donor_posterior_utilities_by_target_control[
            (outer, control)
        ]
    ]
    margin_calibrations = [
        preterminal.margin_calibrations[(outer, control)].to_payload()
        for outer in CENTERS
        for control in (
            PRIMARY_FINGERPRINT_CONTROL_ID,
            BLOCKED_FINGERPRINT_CONTROL_ID,
        )
    ]
    compositions = [
        row.to_payload()
        for policy in COMPOSED_POLICY_IDS
        for row in preterminal.composed_predictions_by_policy[policy]
    ]
    route_decisions = [
        decision.to_payload()
        for policy in COMPOSED_POLICY_IDS
        for composition in preterminal.composed_predictions_by_policy[policy]
        for decision in composition.decisions
    ]
    persist_rows(
        root / "tables/physical_fingerprints.json",
        fingerprint_rows,
        schema_version="fixed_bank_pumr_physical_fingerprint_table_v1",
    )
    persist_rows(
        root / "tables/support_fold_plans.json",
        support_fold_plans,
        schema_version="fixed_bank_pumr_support_fold_plan_table_v1",
    )
    persist_rows(
        root / "tables/target_local_posterior_models.json",
        posterior_models,
        schema_version="fixed_bank_pumr_target_local_posterior_model_table_v1",
    )
    persist_rows(
        root / "tables/target_local_posterior_predictions.json",
        posterior_predictions,
        schema_version="fixed_bank_pumr_target_local_posterior_prediction_table_v1",
    )
    persist_rows(
        root / "tables/route_posterior_ensembles.json",
        posterior_ensembles,
        schema_version="fixed_bank_pumr_route_posterior_ensemble_table_v1",
    )
    persist_rows(
        root / "tables/utility_descriptors.json",
        descriptors,
        schema_version="fixed_bank_pumr_utility_descriptor_table_v1",
        allow_empty=True,
    )
    persist_rows(
        root / "tables/donor_utility_rows.json",
        donor_rows,
        schema_version="fixed_bank_pumr_donor_utility_row_table_v1",
        allow_empty=True,
    )
    persist_rows(
        root / "tables/posterior_utility_predictions.json",
        posterior_utilities,
        schema_version="fixed_bank_pumr_posterior_utility_prediction_table_v1",
        allow_empty=True,
    )
    persist_rows(
        root / "tables/donor_posterior_utility_predictions.json",
        donor_posterior_utilities,
        schema_version="fixed_bank_pumr_donor_posterior_utility_prediction_table_v1",
        allow_empty=True,
    )
    persist_rows(
        root / "tables/margin_calibrations.json",
        margin_calibrations,
        schema_version="fixed_bank_pumr_margin_calibration_table_v1",
    )
    persist_rows(
        root / "tables/composed_predictions.json",
        compositions,
        schema_version="fixed_bank_pumr_composed_prediction_table_v1",
    )
    persist_rows(
        root / "tables/route_decisions.json",
        route_decisions,
        schema_version="fixed_bank_pumr_route_decision_table_v1",
    )
    persist_json(
        root / "manifests/policy_menu.json",
        {
            "schema_version": "fixed_bank_pumr_policy_menu_manifest_v1",
            "policy_ids": list(COMPOSED_POLICY_IDS),
            "policy_count": len(COMPOSED_POLICY_IDS),
            "primary_policy_id": MODEL_BASED_METHOD_ID,
            "terminal_labels_used_to_define_policy": False,
            "target_posterior_is_not_final_prediction": True,
            "donor_response_regression_fit_count": 0,
            "margin_is_inner_leave_one_donor_audited": True,
            "information_gate_is_terminal_only": True,
        },
    )
    persist_json(root / "manifests/decision_barrier.json", dict(preterminal.decision_barrier))
    persist_json(root / "manifests/preterminal_aggregate_seal.json", dict(preterminal.aggregate_seal))


def persist_terminal(
    root: Path,
    *,
    terminal: object,
    leakage_report: Mapping[str, object],
    publication_decision: Mapping[str, object],
    runtime_summary: Mapping[str, object],
) -> None:
    tree = {
        "terminal_evaluation_seal": dict(terminal.terminal_seal),
        "terminal_method_metrics": terminal.method_metrics,
        "terminal_center_contrasts": terminal.center_contrasts,
        "terminal_case_oracles": terminal.case_oracle_rows,
        "utility_information_rows": terminal.utility_information_rows,
        "utility_information_center_rows": terminal.utility_information_center_rows,
        "information_gate": dict(terminal.information_gate),
        "selection_control": dict(terminal.selection_control),
        "diagnostic_summary": dict(terminal.diagnostic_summary),
        "label_capability_report": dict(terminal.capability_report),
        "leakage_report": leakage_report,
        "publication_decision": publication_decision,
        "runtime_summary": runtime_summary,
    }
    json_native(tree)
    persist_json(root / "manifests/terminal_evaluation_seal.json", dict(terminal.terminal_seal))
    persist_rows(
        root / "tables/terminal_method_metrics.json",
        terminal.method_metrics,
        schema_version="fixed_bank_pumr_method_metric_table_v1",
    )
    persist_rows(
        root / "tables/terminal_center_contrasts.json",
        terminal.center_contrasts,
        schema_version="fixed_bank_pumr_center_contrast_table_v1",
    )
    persist_rows(
        root / "tables/terminal_case_oracles.json",
        terminal.case_oracle_rows,
        schema_version="fixed_bank_pumr_case_oracle_table_v1",
    )
    persist_rows(
        root / "tables/utility_information_rows.json",
        terminal.utility_information_rows,
        schema_version="fixed_bank_pumr_utility_information_table_v1",
        allow_empty=True,
    )
    persist_rows(
        root / "tables/utility_information_centers.json",
        terminal.utility_information_center_rows,
        schema_version="fixed_bank_pumr_utility_information_center_table_v1",
    )
    persist_json(root / "tables/information_gate.json", dict(terminal.information_gate))
    persist_json(root / "tables/selection_control.json", dict(terminal.selection_control))
    persist_json(
        root / "reports/diagnostic_summary.json",
        {**dict(terminal.diagnostic_summary), "evaluation_hash": terminal.evaluation_hash},
    )
    persist_json(root / "reports/label_capability_report.json", dict(terminal.capability_report))
    persist_json(root / "reports/leakage_report.json", leakage_report)
    persist_json(root / "reports/publication_decision.json", publication_decision)
    persist_json(root / "reports/runtime_summary.json", runtime_summary)


def persist_validation_report(root: Path, payload: Mapping[str, object]) -> None:
    persist_json(root / "reports/validation_report.json", payload)


def write_run_state(
    root: Path,
    *,
    status: str,
    phase: str,
    error: str | None = None,
    error_class: str | None = None,
) -> None:
    atomic_json(
        root / "reports/run_state.json",
        run_state_payload(status=status, phase=phase, error=error, error_class=error_class),
    )


__all__ = (
    "persist_admission",
    "persist_physical_surface",
    "persist_preterminal",
    "persist_terminal",
    "persist_validation_report",
    "write_run_state",
)
