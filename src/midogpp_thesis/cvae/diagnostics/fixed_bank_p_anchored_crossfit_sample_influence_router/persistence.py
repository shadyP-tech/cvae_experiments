"""Phase-local persistence for reconstructive, raw-label-free PCSI artifacts."""

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
        "schema_version": "fixed_bank_pcsi_action_library_manifest_v1",
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
        "schema_version": "fixed_bank_pcsi_physical_surface_seal_v1",
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
        schema_version="fixed_bank_pcsi_probability_index_table_v1",
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
        "schema_version": "fixed_bank_pcsi_outer_plan_manifest_v1",
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
        schema_version="fixed_bank_pcsi_outer_plan_table_v1",
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
    posterior_predictions = [
        {"control_id": control, **row.to_payload()}
        for control in (
            PRIMARY_FINGERPRINT_CONTROL_ID,
            BLOCKED_FINGERPRINT_CONTROL_ID,
        )
        for row in preterminal.target_posterior_predictions_by_control[control]
    ]
    sample_influences = [
        {"control_id": control, **row.to_payload()}
        for control in (
            PRIMARY_FINGERPRINT_CONTROL_ID,
            BLOCKED_FINGERPRINT_CONTROL_ID,
        )
        for row in preterminal.sample_influence_predictions_by_control[control]
    ]
    donor_veto_predictions = [
        row.to_payload() for row in preterminal.donor_veto_predictions
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
        schema_version="fixed_bank_pcsi_physical_fingerprint_table_v1",
    )
    persist_rows(
        root / "tables/target_local_posterior_models.json",
        posterior_models,
        schema_version="fixed_bank_pcsi_target_local_posterior_model_table_v1",
    )
    persist_rows(
        root / "tables/target_local_posterior_predictions.json",
        posterior_predictions,
        schema_version="fixed_bank_pcsi_target_local_posterior_prediction_table_v1",
    )
    persist_rows(
        root / "tables/sample_influence_predictions.json",
        sample_influences,
        schema_version="fixed_bank_pcsi_sample_influence_prediction_table_v1",
        allow_empty=True,
    )
    persist_rows(
        root / "tables/utility_descriptors.json",
        descriptors,
        schema_version="fixed_bank_pcsi_utility_descriptor_table_v1",
        allow_empty=True,
    )
    persist_rows(
        root / "tables/donor_utility_rows.json",
        donor_rows,
        schema_version="fixed_bank_pcsi_donor_utility_row_table_v1",
        allow_empty=True,
    )
    persist_rows(
        root / "tables/donor_veto_models.json",
        _model_rows(preterminal),
        schema_version="fixed_bank_pcsi_donor_veto_model_table_v1",
    )
    persist_rows(
        root / "tables/donor_veto_predictions.json",
        donor_veto_predictions,
        schema_version="fixed_bank_pcsi_donor_veto_prediction_table_v1",
        allow_empty=True,
    )
    persist_rows(
        root / "tables/composed_predictions.json",
        compositions,
        schema_version="fixed_bank_pcsi_composed_prediction_table_v1",
    )
    persist_rows(
        root / "tables/route_decisions.json",
        route_decisions,
        schema_version="fixed_bank_pcsi_route_decision_table_v1",
    )
    persist_json(
        root / "manifests/policy_menu.json",
        {
            "schema_version": "fixed_bank_pcsi_policy_menu_manifest_v1",
            "policy_ids": list(COMPOSED_POLICY_IDS),
            "policy_count": len(COMPOSED_POLICY_IDS),
            "primary_policy_id": MODEL_BASED_METHOD_ID,
            "terminal_labels_used_to_define_policy": False,
            "target_posterior_is_not_final_prediction": True,
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
        schema_version="fixed_bank_pcsi_method_metric_table_v1",
    )
    persist_rows(
        root / "tables/terminal_center_contrasts.json",
        terminal.center_contrasts,
        schema_version="fixed_bank_pcsi_center_contrast_table_v1",
    )
    persist_rows(
        root / "tables/terminal_case_oracles.json",
        terminal.case_oracle_rows,
        schema_version="fixed_bank_pcsi_case_oracle_table_v1",
    )
    persist_rows(
        root / "tables/utility_information_rows.json",
        terminal.utility_information_rows,
        schema_version="fixed_bank_pcsi_utility_information_table_v1",
        allow_empty=True,
    )
    persist_rows(
        root / "tables/utility_information_centers.json",
        terminal.utility_information_center_rows,
        schema_version="fixed_bank_pcsi_utility_information_center_table_v1",
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


def _model_rows(preterminal: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for outer in CENTERS:
        rows.append(
            {
                "model_kind": "donor_pareto_veto",
                "fit_scope": "full",
                "deleted_donor_center": None,
                **preterminal.full_models_by_target[outer].to_payload(),
            }
        )
        rows.extend(
            {
                "model_kind": "donor_pareto_veto",
                "fit_scope": "delete_one_donor",
                "deleted_donor_center": deleted,
                **model.to_payload(),
            }
            for deleted, model in preterminal.delete_models_by_target[outer].items()
        )
    return rows


__all__ = (
    "persist_admission",
    "persist_physical_surface",
    "persist_preterminal",
    "persist_terminal",
    "persist_validation_report",
    "write_run_state",
)
