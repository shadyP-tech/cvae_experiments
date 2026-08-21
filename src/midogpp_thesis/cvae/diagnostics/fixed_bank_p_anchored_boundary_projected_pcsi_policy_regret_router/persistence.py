"""Phase-local persistence for the raw-label-free PCSI-PARC bundle."""

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
    EXPECTED_POLICY_REPLAY_COUNT,
    PRIMARY_FINGERPRINT_CONTROL_ID,
    PRIMARY_METHOD_ID,
    UNPROJECTED_PARC_METHOD_ID,
)
from .controls import CONTROL_SPECS
from .donor_runtime import PARC_GEOMETRIES
from .hashing import canonical_hash, json_native
from .reports import (
    assert_transport_authorization_lineage_valid,
    run_state_payload,
)


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
        "schema_version": "fixed_bank_pcsi_parc_action_library_manifest_v1",
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
        "schema_version": "fixed_bank_pcsi_parc_physical_surface_seal_v1",
        "source_stream_lock_hash": str(
            getattr(physical, "canonical_source_cache").lock_hash
        ),
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
        schema_version="fixed_bank_pcsi_parc_probability_index_table_v1",
    )


def persist_preterminal(root: Path, preterminal: object) -> None:
    assert_transport_authorization_lineage_valid(preterminal)
    plans = getattr(preterminal, "plans")
    donor_runtime = getattr(preterminal, "donor_runtime")
    policy_runtime = getattr(preterminal, "policy_runtime")
    outer_rows = [
        {
            "target_center": row.target_center,
            "case_id": row.case_id,
            "group_id": row.group_id,
            "support_case_ids": list(row.support_case_ids),
            "evaluation_sample_count": len(row.evaluation_sample_ids),
            "evaluation_identity_hash": canonical_hash(
                list(row.evaluation_sample_ids)
            ),
            "probability_surface_hash": row.probability_surface_hash,
            "plan_hash": row.plan_hash,
            "held_case_and_group_excluded": True,
            "support_scope": "H_minus_c_only",
            "labels_used": False,
        }
        for row in plans.outer_plans
    ]
    plan_manifest = {
        "schema_version": "fixed_bank_pcsi_parc_outer_plan_manifest_v1",
        "probability_surface_hash": plans.probability_surface_hash,
        "outer_plan_count": len(outer_rows),
        "double_exclusion_pair_count": len(plans.double_exclusion_plans),
        "double_exclusion_state_count": EXPECTED_POLICY_REPLAY_COUNT,
        "double_exclusion_plan_hash": canonical_hash(
            [row.to_payload() for row in plans.double_exclusion_plans]
        ),
        "outer_plan_hash": canonical_hash(outer_rows),
        "plan_seal_hash": plans.seal_hash,
        "sealed_before_any_label_access": True,
    }
    persist_json(root / "manifests/outer_plan_seal.json", plan_manifest)
    persist_rows(
        root / "tables/outer_plans.json",
        outer_rows,
        schema_version="fixed_bank_pcsi_parc_outer_plan_table_v1",
    )
    persist_rows(
        root / "tables/double_exclusion_plans.json",
        [row.to_payload() for row in plans.double_exclusion_plans],
        schema_version="fixed_bank_pcsi_parc_double_exclusion_plan_table_v1",
    )

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
    persist_rows(
        root / "tables/physical_fingerprints.json",
        fingerprint_rows,
        schema_version="fixed_bank_pcsi_parc_physical_fingerprint_table_v1",
    )
    persist_rows(
        root / "tables/target_local_posterior_models.json",
        posterior_models,
        schema_version="fixed_bank_pcsi_parc_target_local_posterior_model_table_v1",
    )
    persist_rows(
        root / "tables/target_local_posterior_predictions.json",
        posterior_predictions,
        schema_version=(
            "fixed_bank_pcsi_parc_target_local_posterior_prediction_table_v1"
        ),
    )

    persist_rows(
        root / "tables/action_equivalence_classes.json",
        _projected_action_rows(donor_runtime),
        schema_version="fixed_bank_pcsi_parc_action_equivalence_table_v1",
    )
    persist_rows(
        root / "tables/projected_utility_descriptors.json",
        _projected_descriptor_rows(donor_runtime),
        schema_version="fixed_bank_pcsi_parc_projected_descriptor_table_v1",
    )
    persist_rows(
        root / "tables/projected_donor_utility_rows.json",
        _projected_donor_rows(donor_runtime),
        schema_version="fixed_bank_pcsi_parc_projected_donor_row_table_v1",
    )
    persist_rows(
        root / "tables/double_excluded_prior_provenance.json",
        [
            donor_runtime.pseudo_prior_provenance[key].to_payload()
            for key in sorted(donor_runtime.pseudo_prior_provenance)
        ],
        schema_version="fixed_bank_pcsi_parc_double_excluded_prior_table_v1",
    )
    persist_rows(
        root / "tables/double_excluded_endpoint_scopes.json",
        _double_excluded_endpoint_scope_rows(donor_runtime),
        schema_version="fixed_bank_pcsi_parc_double_excluded_endpoint_table_v1",
    )
    persist_rows(
        root / "tables/pseudo_donor_utility_rows.json",
        [
            {
                "pseudo_target_center": pseudo,
                **row.to_payload(),
            }
            for geometry in PARC_GEOMETRIES
            for outer in CENTERS
            for pseudo in CENTERS
            if pseudo != outer
            for row in donor_runtime.geometry_results[
                geometry
            ].pseudo_donor_rows_by_pair[(outer, pseudo)]
        ],
        schema_version="fixed_bank_pcsi_parc_pseudo_donor_row_table_v1",
    )
    persist_rows(
        root / "tables/projected_utility_models.json",
        _projected_model_rows(donor_runtime),
        schema_version="fixed_bank_pcsi_parc_projected_model_table_v1",
    )
    persist_rows(
        root / "tables/projected_utility_predictions.json",
        _projected_prediction_rows(donor_runtime),
        schema_version="fixed_bank_pcsi_parc_projected_prediction_table_v1",
    )

    legacy = donor_runtime.legacy
    persist_rows(
        root / "tables/fresh_legacy_utility_descriptors.json",
        [
            {"fit_scope": "target", **row.to_payload()}
            for center in CENTERS
            for row in legacy.descriptors_by_center[center]
        ],
        schema_version="fixed_bank_pcsi_parc_fresh_legacy_descriptor_table_v1",
    )
    persist_rows(
        root / "tables/fresh_legacy_donor_utility_rows.json",
        [
            row.to_payload()
            for outer in CENTERS
            for row in legacy.donor_rows_by_outer[outer]
        ],
        schema_version="fixed_bank_pcsi_parc_fresh_legacy_donor_row_table_v1",
    )
    persist_rows(
        root / "tables/fresh_legacy_utility_models.json",
        _legacy_model_rows(legacy),
        schema_version="fixed_bank_pcsi_parc_fresh_legacy_model_table_v1",
    )
    persist_rows(
        root / "tables/fresh_legacy_utility_predictions.json",
        [
            {"target_center": center, **row.to_payload()}
            for center in CENTERS
            for row in legacy.predictions_by_center[center]
        ],
        schema_version="fixed_bank_pcsi_parc_fresh_legacy_prediction_table_v1",
    )

    persist_rows(
        root / "tables/sample_influence_predictions.json",
        [
            {"policy_id": policy, **row.to_payload()}
            for policy in COMPOSED_POLICY_IDS
            for center in CENTERS
            for row in policy_runtime.target_influences_by_policy_center[
                (policy, center)
            ]
        ],
        schema_version="fixed_bank_pcsi_parc_sample_influence_prediction_table_v1",
    )
    persist_rows(
        root / "tables/transport_descriptors.json",
        [
            {
                "outer_target_center": outer,
                "candidate_center": candidate,
                **policy_runtime.transport_descriptors_by_outer_candidate[
                    (outer, candidate)
                ].to_payload(),
            }
            for outer in CENTERS
            for candidate in CENTERS
        ],
        schema_version="fixed_bank_pcsi_parc_transport_descriptor_table_v1",
    )
    persist_rows(
        root / "tables/transport_screens.json",
        [
            {
                "outer_target_center": outer,
                "screen_role": "target" if candidate is None else "pseudo_target",
                "pseudo_target_center": candidate,
                **policy_runtime.transport_screens[(outer, candidate)].to_payload(),
            }
            for outer in CENTERS
            for candidate in (
                None,
                *(center for center in CENTERS if center != outer),
            )
        ],
        schema_version="fixed_bank_pcsi_parc_transport_screen_table_v1",
    )
    persist_rows(
        root / "tables/target_candidate_policies.json",
        [
            policy_runtime.target_candidate_policies[(policy, center)].to_payload()
            for policy in COMPOSED_POLICY_IDS
            for center in CENTERS
        ],
        schema_version="fixed_bank_pcsi_parc_target_candidate_policy_table_v1",
    )
    persist_rows(
        root / "tables/pseudo_candidate_policies.json",
        [
            {
                "outer_target_center": outer,
                **policy_runtime.pseudo_candidate_policies[
                    (geometry, outer, pseudo)
                ].to_payload(),
            }
            for geometry in PARC_GEOMETRIES
            for outer in CENTERS
            for pseudo in CENTERS
            if pseudo != outer
        ],
        schema_version="fixed_bank_pcsi_parc_pseudo_candidate_policy_table_v1",
    )
    persist_rows(
        root / "tables/policy_regret_replays.json",
        [
            policy_runtime.replays[(geometry, outer, pseudo)].to_payload()
            for geometry in PARC_GEOMETRIES
            for outer in CENTERS
            for pseudo in CENTERS
            if pseudo != outer
        ],
        schema_version="fixed_bank_pcsi_parc_policy_regret_replay_table_v1",
    )
    persist_rows(
        root / "tables/policy_authorizations.json",
        [
            policy_runtime.authorizations[(policy, center)].to_payload()
            for policy in (
                PRIMARY_METHOD_ID,
                UNPROJECTED_PARC_METHOD_ID,
            )
            for center in CENTERS
        ],
        schema_version="fixed_bank_pcsi_parc_policy_authorization_table_v1",
    )
    persist_rows(
        root / "tables/final_policy_predictions.json",
        [
            row.to_payload()
            for policy in COMPOSED_POLICY_IDS
            for row in policy_runtime.final_predictions_by_policy[policy]
        ],
        schema_version="fixed_bank_pcsi_parc_final_policy_prediction_table_v1",
    )
    persist_rows(
        root / "tables/route_decisions.json",
        [
            decision.to_payload()
            for policy in COMPOSED_POLICY_IDS
            for center in CENTERS
            for case in policy_runtime.target_candidate_policies[(policy, center)].cases
            for decision in case.decisions
        ],
        schema_version="fixed_bank_pcsi_parc_route_decision_table_v1",
    )

    persist_json(root / "manifests/donor_runtime.json", donor_runtime.summary_payload())
    persist_json(
        root / "manifests/policy_replay_runtime.json",
        policy_runtime.summary_payload(),
    )
    control_rows = [row.to_payload() for row in CONTROL_SPECS]
    control_hash = canonical_hash(control_rows)
    persist_json(
        root / "manifests/policy_menu.json",
        {
            "schema_version": "fixed_bank_pcsi_parc_policy_menu_manifest_v1",
            "policy_menu_seal": dict(policy_runtime.policy_menu_seal),
            "control_specs": control_rows,
            "control_count": len(control_rows),
            "control_spec_hash": control_hash,
            "primary_policy_id": PRIMARY_METHOD_ID,
            "terminal_labels_used_to_define_policy": False,
            "terminal_diagnostics_may_change_policy": False,
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
    tree = {
        "terminal_evaluation_seal": dict(terminal.terminal_seal),
        "terminal_method_metrics": terminal.method_metrics,
        "terminal_center_contrasts": terminal.center_contrasts,
        "terminal_case_oracles": terminal.case_oracle_rows,
        "terminal_projected_action_rows": terminal.projected_action_rows,
        "terminal_policy_regret_rows": terminal.policy_regret_rows,
        "terminal_transport_diagnostic_rows": terminal.transport_diagnostic_rows,
        "terminal_selected_case_rows": terminal.selected_case_rows,
        "terminal_policy_regret_centers": terminal.policy_regret_center_rows,
        "terminal_action_frequencies": terminal.action_frequency_rows,
        "terminal_diagnostic": dict(terminal.terminal_diagnostic),
        "selection_control": dict(terminal.selection_control),
        "diagnostic_summary": dict(terminal.diagnostic_summary),
        "label_capability_report": dict(terminal.capability_report),
        "leakage_report": leakage_report,
        "publication_decision": publication_decision,
        "runtime_summary": runtime_summary,
    }
    json_native(tree)
    persist_json(
        root / "manifests/terminal_evaluation_seal.json",
        dict(terminal.terminal_seal),
    )
    persist_rows(
        root / "tables/terminal_method_metrics.json",
        terminal.method_metrics,
        schema_version="fixed_bank_pcsi_parc_method_metric_table_v1",
    )
    persist_rows(
        root / "tables/terminal_center_contrasts.json",
        terminal.center_contrasts,
        schema_version="fixed_bank_pcsi_parc_center_contrast_table_v1",
    )
    persist_rows(
        root / "tables/terminal_case_oracles.json",
        terminal.case_oracle_rows,
        schema_version="fixed_bank_pcsi_parc_case_oracle_table_v1",
    )
    persist_rows(
        root / "tables/terminal_projected_action_diagnostics.json",
        terminal.projected_action_rows,
        schema_version="fixed_bank_pcsi_parc_terminal_projected_action_table_v1",
    )
    persist_rows(
        root / "tables/terminal_policy_regret_diagnostics.json",
        terminal.policy_regret_rows,
        schema_version="fixed_bank_pcsi_parc_terminal_policy_regret_table_v1",
    )
    persist_rows(
        root / "tables/terminal_transport_diagnostics.json",
        terminal.transport_diagnostic_rows,
        schema_version="fixed_bank_pcsi_parc_terminal_transport_table_v1",
    )
    persist_rows(
        root / "tables/terminal_selected_case_diagnostics.json",
        terminal.selected_case_rows,
        schema_version="fixed_bank_pcsi_parc_terminal_selected_case_table_v1",
    )
    persist_rows(
        root / "tables/terminal_policy_regret_centers.json",
        terminal.policy_regret_center_rows,
        schema_version="fixed_bank_pcsi_parc_terminal_policy_regret_center_table_v1",
    )
    persist_rows(
        root / "tables/terminal_action_frequencies.json",
        terminal.action_frequency_rows,
        schema_version="fixed_bank_pcsi_parc_terminal_action_frequency_table_v1",
    )
    persist_json(
        root / "tables/terminal_diagnostic.json",
        dict(terminal.terminal_diagnostic),
    )
    persist_json(
        root / "tables/selection_control.json",
        dict(terminal.selection_control),
    )
    persist_json(
        root / "reports/diagnostic_summary.json",
        {**dict(terminal.diagnostic_summary), "evaluation_hash": terminal.evaluation_hash},
    )
    persist_json(
        root / "reports/label_capability_report.json",
        dict(terminal.capability_report),
    )
    persist_json(root / "reports/leakage_report.json", leakage_report)
    persist_json(root / "reports/publication_decision.json", publication_decision)
    persist_json(root / "reports/runtime_summary.json", runtime_summary)


def persist_phase_telemetry(root: Path, payload: Mapping[str, object]) -> None:
    persist_json(root / "reports/phase_telemetry.json", payload)


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


def _projected_action_rows(donor_runtime: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for geometry in PARC_GEOMETRIES:
        runtime = donor_runtime.geometry_results[geometry]
        for center in CENTERS:
            rows.extend(
                {
                    "surface_scope": "target",
                    "outer_target_center": center,
                    "candidate_center": center,
                    **row.to_payload(),
                }
                for row in runtime.target_actions_by_center[center]
            )
        for outer in CENTERS:
            for pseudo in CENTERS:
                if pseudo != outer:
                    rows.extend(
                        {
                            "surface_scope": "H_J_double_excluded_pseudo_target",
                            "outer_target_center": outer,
                            "candidate_center": pseudo,
                            **row.to_payload(),
                        }
                        for row in runtime.pseudo_actions_by_pair[(outer, pseudo)]
                    )
    return rows


def _projected_descriptor_rows(donor_runtime: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for geometry in PARC_GEOMETRIES:
        runtime = donor_runtime.geometry_results[geometry]
        for center in CENTERS:
            rows.extend(
                {
                    "surface_scope": "target",
                    "outer_target_center": center,
                    "candidate_center": center,
                    **row.to_payload(),
                }
                for row in runtime.target_descriptors_by_center[center]
            )
        for outer in CENTERS:
            for pseudo in CENTERS:
                if pseudo != outer:
                    rows.extend(
                        {
                            "surface_scope": "H_J_double_excluded_pseudo_target",
                            "outer_target_center": outer,
                            "candidate_center": pseudo,
                            **row.to_payload(),
                        }
                        for row in runtime.pseudo_descriptors_by_pair[(outer, pseudo)]
                    )
    return rows


def _projected_donor_rows(donor_runtime: object) -> list[dict[str, object]]:
    return [
        row.to_payload()
        for geometry in PARC_GEOMETRIES
        for outer in CENTERS
        for row in donor_runtime.geometry_results[geometry].donor_rows_by_outer[outer]
    ]


def _double_excluded_endpoint_scope_rows(
    donor_runtime: object,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in sorted(donor_runtime.pseudo_donor_endpoint_products):
        outer, pseudo, donor = key
        provenance = donor_runtime.pseudo_prior_provenance[key]
        product = donor_runtime.pseudo_donor_endpoint_products[key]
        payload = {
            "outer_target_center": outer,
            "pseudo_target_center": pseudo,
            "donor_center": donor,
            "prior_hash": provenance.prior_hash,
            "state_hashes": [list(row) for row in product.state_hashes],
            "prediction_hashes": [
                row.prediction_hash for row in product.predictions
            ],
            "endpoint_model_fit_count": product.endpoint_model_fit_count,
            "outer_H_excluded_from_prior_queries": True,
            "pseudo_J_excluded_from_prior_queries": True,
            "endpoint_target_K_excluded_from_prior_queries": True,
            "raw_labels_persisted": False,
        }
        rows.append({**payload, "scope_hash": canonical_hash(payload)})
    return rows


def _projected_model_rows(donor_runtime: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for geometry in PARC_GEOMETRIES:
        runtime = donor_runtime.geometry_results[geometry]
        for outer in CENTERS:
            rows.append(
                {
                    "model_scope": "target_full",
                    "candidate_center": outer,
                    "deleted_donor_center": None,
                    **runtime.target_full_models[outer].to_payload(),
                }
            )
            rows.extend(
                {
                    "model_scope": "target_delete_one_donor",
                    "candidate_center": outer,
                    "deleted_donor_center": deleted,
                    **model.to_payload(),
                }
                for deleted, model in runtime.target_delete_models[outer].items()
            )
            for pseudo in CENTERS:
                if pseudo == outer:
                    continue
                pair = outer, pseudo
                rows.append(
                    {
                        "model_scope": "H_J_double_excluded_full",
                        "candidate_center": pseudo,
                        "deleted_donor_center": None,
                        **runtime.pseudo_full_models[pair].to_payload(),
                    }
                )
                rows.extend(
                    {
                        "model_scope": "H_J_double_excluded_delete_one_donor",
                        "candidate_center": pseudo,
                        "deleted_donor_center": deleted,
                        **model.to_payload(),
                    }
                    for deleted, model in runtime.pseudo_delete_models[pair].items()
                )
    return rows


def _projected_prediction_rows(donor_runtime: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for geometry in PARC_GEOMETRIES:
        runtime = donor_runtime.geometry_results[geometry]
        for center in CENTERS:
            rows.extend(
                {
                    "prediction_scope": "target",
                    "outer_target_center": center,
                    "candidate_center": center,
                    **row.to_payload(),
                }
                for row in runtime.target_predictions_by_center[center]
            )
        for outer in CENTERS:
            for pseudo in CENTERS:
                if pseudo != outer:
                    rows.extend(
                        {
                            "prediction_scope": "H_J_double_excluded_pseudo_target",
                            "outer_target_center": outer,
                            "candidate_center": pseudo,
                            **row.to_payload(),
                        }
                        for row in runtime.pseudo_predictions_by_pair[(outer, pseudo)]
                    )
    return rows


def _legacy_model_rows(legacy: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for outer in CENTERS:
        rows.append(
            {
                "model_scope": "target_full",
                "deleted_donor_center": None,
                **legacy.full_models_by_outer[outer].to_payload(),
            }
        )
        rows.extend(
            {
                "model_scope": "target_delete_one_donor",
                "deleted_donor_center": deleted,
                **model.to_payload(),
            }
            for deleted, model in legacy.delete_models_by_outer[outer].items()
        )
    return rows


__all__ = (
    "persist_admission",
    "persist_phase_telemetry",
    "persist_physical_surface",
    "persist_preterminal",
    "persist_terminal",
    "persist_validation_report",
    "write_run_state",
)
