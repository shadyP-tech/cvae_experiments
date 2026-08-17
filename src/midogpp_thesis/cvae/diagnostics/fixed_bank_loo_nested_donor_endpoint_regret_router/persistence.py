"""Phase-local persistence for reconstructive, raw-label-free audit products."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ...runtime.artifact_io import atomic_json
from .actions import action_library_by_target
from .artifact_io import persist_json, persist_rows
from .constants import CENTERS
from .hashing import canonical_hash
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
        "schema_version": "fixed_bank_nested_regret_action_library_manifest_v1",
        "actions": [
            action.to_payload()
            for center in CENTERS
            for action in actions[center]
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
        "schema_version": "fixed_bank_nested_regret_physical_surface_seal_v1",
        "source_stream_lock_hash": str(
            getattr(physical, "canonical_source_cache").lock_hash
        ),
        "global_prediction_seal_hash": str(prediction.seal_hash),
        "probability_store_hash": str(prediction.store.store_hash),
        "physical_probability_surface_hash": str(getattr(surface, "surface_hash")),
        "probability_index_hash": canonical_hash(
            [row.to_payload() for row in rows]
        ),
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
        schema_version="fixed_bank_nested_regret_probability_index_table_v1",
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
    pair_rows = [
        {
            "target_center": row.target_center,
            "first_case_id": row.first_case_id,
            "second_case_id": row.second_case_id,
            "support_case_ids": list(row.support_case_ids),
            "probability_surface_hash": row.probability_surface_hash,
            "plan_hash": row.plan_hash,
            "one_fit_state_reused_for_two_ordered_voters": True,
            "labels_used": False,
        }
        for row in plans.unordered_pair_plans
    ]
    plan_manifest = {
        "schema_version": "fixed_bank_nested_regret_plan_manifest_v1",
        "probability_surface_hash": plans.probability_surface_hash,
        "outer_plan_count": len(outer_rows),
        "unordered_pair_plan_count": len(pair_rows),
        "ordered_voter_count": 2 * len(pair_rows),
        "outer_plan_hash": canonical_hash(outer_rows),
        "unordered_pair_plan_hash": canonical_hash(pair_rows),
        "plan_seal_hash": plans.seal_hash,
        "sealed_before_any_label_access": True,
    }
    persist_json(root / "manifests/nested_plan_seal.json", plan_manifest)
    persist_rows(
        root / "tables/outer_plans.json",
        outer_rows,
        schema_version="fixed_bank_nested_regret_outer_plan_table_v1",
    )
    persist_rows(
        root / "tables/unordered_pair_plans.json",
        pair_rows,
        schema_version="fixed_bank_nested_regret_pair_plan_table_v1",
    )

    descriptors = [
        descriptor.to_payload()
        for center in CENTERS
        for descriptor in getattr(preterminal, "descriptors_by_center")[center]
    ]
    donor_rows = [
        {
            "outer_target_center": outer,
            "donor_center": row.donor_center,
            "case_id": row.case_id,
            "alternative": row.alternative,
            "feature_values": list(row.feature_values),
            "bacc_regret": row.bacc_regret,
            "log_loss_delta": row.log_loss_delta,
            "center_case_count": row.center_case_count,
            "descriptor_hash": row.descriptor_hash,
        }
        for outer in CENTERS
        for row in getattr(preterminal, "donor_rows_by_outer_target")[outer]
    ]
    model_rows = _model_rows(preterminal)
    decisions = [
        row.to_payload()
        for policy_id in getattr(preterminal, "decisions_by_policy")
        for row in getattr(preterminal, "decisions_by_policy")[policy_id]
    ]
    ltt = [row.to_payload() for row in getattr(preterminal, "ltt_authorizations")]
    persist_rows(
        root / "tables/candidate_descriptors.json",
        descriptors,
        schema_version="fixed_bank_nested_regret_descriptor_table_v1",
    )
    persist_rows(
        root / "tables/donor_regret_rows.json",
        donor_rows,
        schema_version="fixed_bank_nested_regret_donor_row_table_v1",
    )
    persist_rows(
        root / "tables/regret_models.json",
        model_rows,
        schema_version="fixed_bank_nested_regret_model_table_v1",
    )
    persist_rows(
        root / "tables/route_decisions.json",
        decisions,
        schema_version="fixed_bank_nested_regret_route_decision_table_v1",
    )
    persist_rows(
        root / "tables/center_block_feasibility.json",
        ltt,
        schema_version="fixed_bank_nested_regret_center_block_feasibility_table_v1",
    )
    persist_json(
        root / "manifests/policy_menu.json",
        {
            "schema_version": "fixed_bank_nested_regret_policy_menu_manifest_v1",
            "policies": [row.to_payload() for row in preterminal.policy_menu],
            "policy_count": len(preterminal.policy_menu),
            "center_block_feasibility_method_id": "NDR_CENTER_BLOCK_FEASIBILITY",
            "terminal_utility_used_to_define_policy": False,
        },
    )
    persist_json(
        root / "manifests/decision_barrier.json",
        dict(preterminal.decision_barrier),
    )
    persist_json(
        root / "manifests/preterminal_aggregate_seal.json",
        dict(preterminal.aggregate_seal),
    )


def persist_terminal(
    root: Path,
    *,
    terminal: object,
    leakage_report: Mapping[str, object],
    publication_decision: Mapping[str, object],
    runtime_summary: Mapping[str, object],
) -> None:
    persist_json(
        root / "manifests/terminal_evaluation_seal.json",
        dict(terminal.terminal_seal),
    )
    persist_rows(
        root / "tables/terminal_method_metrics.json",
        terminal.method_metrics,
        schema_version="fixed_bank_nested_regret_method_metric_table_v1",
    )
    persist_rows(
        root / "tables/terminal_center_contrasts.json",
        terminal.center_contrasts,
        schema_version="fixed_bank_nested_regret_center_contrast_table_v1",
    )
    persist_rows(
        root / "tables/terminal_case_oracles.json",
        terminal.case_oracle_rows,
        schema_version="fixed_bank_nested_regret_case_oracle_table_v1",
    )
    persist_json(
        root / "tables/selection_control.json", dict(terminal.selection_control)
    )
    persist_json(
        root / "reports/diagnostic_summary.json",
        {
            "schema_version": "fixed_bank_nested_regret_diagnostic_summary_v1",
            **dict(terminal.diagnostic_summary),
            "evaluation_hash": terminal.evaluation_hash,
        },
    )
    persist_json(
        root / "reports/label_capability_report.json",
        dict(terminal.capability_report),
    )
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
        run_state_payload(
            status=status,
            phase=phase,
            error=error,
            error_class=error_class,
        ),
    )


def _model_rows(preterminal: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for outer in CENTERS:
        scopes = [("full", None, preterminal.full_models_by_target[outer])]
        scopes.extend(
            ("delete_one_donor", deleted, models)
            for deleted, models in preterminal.delete_models_by_target[outer].items()
        )
        for fit_scope, deleted_center, models in scopes:
            for response_name, model in models.items():
                rows.append(
                    {
                        "outer_target_center": outer,
                        "fit_scope": fit_scope,
                        "deleted_donor_center": deleted_center,
                        "response_name": response_name,
                        "training_centers": list(model.training_centers),
                        "feature_names": list(model.feature_names),
                        "feature_mean": list(model.feature_mean),
                        "feature_scale": list(model.feature_scale),
                        "coefficients": list(model.coefficients),
                        "ridge_alpha": model.ridge_alpha,
                        "center_effect_alpha": model.center_effect_alpha,
                        "training_row_count_by_center": dict(
                            model.training_row_count_by_center
                        ),
                        "equal_total_weight_per_center": True,
                        "unseen_target_center_effect": 0.0,
                        "model_hash": model.model_hash,
                    }
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
