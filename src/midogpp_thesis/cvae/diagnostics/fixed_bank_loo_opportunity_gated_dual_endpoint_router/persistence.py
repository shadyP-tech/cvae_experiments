"""Phase-specific persistence and lineage seals for the OGDE bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json
from .artifact_rows import row_payload
from .artifact_topology import (
    ARM_IDENTITY_COUNT,
    CASE_ACTION_SUFFICIENT_STAT_COUNT,
    CORRECTNESS_OBSERVATION_COUNT,
    DIRECTIONAL_SUPPORT_GAIN_COUNT,
    DONOR_PRIOR_COUNT,
    IDENTIFICATION_DECISION_COUNT,
    MODEL_FIT_COUNT,
    MODEL_FITS_PER_FEATURE_SURFACE,
    METHOD_PREDICTION_COUNT,
    ROBUST_ARM_DECISION_COUNT,
    ROBUST_METHOD_IDS,
    ROUTE_COUNT,
)
from .artifact_writers import persist_json, persist_rows, read_rows
from .hashing import canonical_hash, json_native
from .reports import protocol_manifest_payload, run_state_payload, seal_payload


def persist_initial_surfaces(
    root: Path,
    *,
    config: object,
    protocol: object,
    provenance: Mapping[str, Mapping[str, object]],
    frame: object,
    firewall: Mapping[str, object],
    actions: Sequence[object],
) -> Mapping[str, object]:
    input_ids = tuple(getattr(config, "input_artifact_ids"))
    if len(input_ids) != 6 or set(provenance) != set(input_ids):
        raise ProtocolError("Dual-endpoint provenance must cover exactly six inputs.")
    manifest = protocol_manifest_payload(
        config,
        protocol=protocol,
        input_artifact_hashes={
            artifact_id: canonical_hash(provenance[artifact_id])
            for artifact_id in input_ids
        },
        cache_binding_hash=str(getattr(frame, "cache_binding_hash")),
        firewall=firewall,
    )
    persist_json(root / "manifests/protocol_manifest.json", manifest)
    rows = persist_rows(root / "tables/action_library.csv", actions)
    seal = seal_payload(
        "fixed_bank_dual_endpoint_action_library_seal_v1",
        bindings={"actions_hash": canonical_hash(rows)},
        action_count=len(rows),
        actions_per_target=10,
        exact_nine_required=True,
        labels_used=False,
        target_expert_used=False,
    )
    persist_json(root / "manifests/action_library.json", seal)
    return seal


def persist_physical_prelabel(
    root: Path,
    *,
    prediction: object,
    probability_index: Sequence[object],
    probability_surface_hash: str,
) -> Mapping[str, object]:
    rows = persist_rows(
        root / "tables/exact_nine_probability_index.csv", probability_index
    )
    seal = seal_payload(
        "fixed_bank_dual_endpoint_physical_prelabel_seal_v1",
        bindings={
            "global_prediction_seal_hash": str(getattr(prediction, "seal_hash")),
            "prediction_store_hash": str(getattr(prediction, "store").store_hash),
            "probability_surface_hash": probability_surface_hash,
            "probability_index_hash": canonical_hash(rows),
        },
        physical_cell_count=len(getattr(prediction, "store").cells),
        target_action_index_count=len(rows),
        stored_probability_dtype="float32",
        exact_nine_reduction_dtype="float64",
        labels_used=False,
        sealed_before_any_label_capability=True,
    )
    persist_json(root / "manifests/physical_prelabel_seal.json", seal)
    return seal


def persist_label_free_products(
    root: Path,
    *,
    plans: Sequence[object],
    plan_seal: object,
    features: Sequence[object],
    physical_prelabel_seal_hash: str,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    plan_rows = persist_rows(root / "tables/whole_case_loo_plans.csv", plans)
    feature_rows = persist_rows(
        root / "tables/label_free_candidate_features.csv", features
    )
    science_plan_hash = str(row_payload(plan_seal)["plan_seal_hash"])
    persisted_plan = seal_payload(
        "fixed_bank_dual_endpoint_loo_plan_seal_v1",
        bindings={
            "physical_prelabel_seal_hash": physical_prelabel_seal_hash,
            "plans_hash": canonical_hash(plan_rows),
            "science_plan_seal_hash": science_plan_hash,
        },
        plan_count=len(plan_rows),
        held_case_and_group_excluded=True,
        labels_used=False,
    )
    persist_json(root / "manifests/loo_plan_seal.json", persisted_plan)
    feature_seal = seal_payload(
        "fixed_bank_dual_endpoint_label_free_feature_seal_v1",
        bindings={
            "physical_prelabel_seal_hash": physical_prelabel_seal_hash,
            "loo_plan_seal_hash": str(persisted_plan["seal_hash"]),
            "features_hash": canonical_hash(feature_rows),
        },
        feature_count=len(feature_rows),
        labels_used=False,
        feature_blocks_sealed_before_support_labels=True,
    )
    persist_json(root / "manifests/label_free_feature_seal.json", feature_seal)
    return persisted_plan, feature_seal


def persist_route_products(
    root: Path,
    *,
    case_action_confusions: Sequence[object],
    correctness_observations: Sequence[object],
    model_fits: Sequence[object],
    directional_support_gains: Sequence[object],
    donor_priors: Sequence[object],
    endpoint_arms: Sequence[object],
    identification_decisions: Sequence[object],
    robust_arm_decisions: Sequence[object],
    method_predictions: Sequence[object],
    loo_plan_seal_hash: str,
    feature_seal_hash: str,
    route_barrier: Mapping[str, object],
) -> Mapping[str, Mapping[str, object]]:
    observed_counts = {
        "case_action_confusions": len(case_action_confusions),
        "correctness_observations": len(correctness_observations),
        "model_fits": len(model_fits),
        "directional_support_gains": len(directional_support_gains),
        "donor_priors": len(donor_priors),
        "identification_decisions": len(identification_decisions),
        "robust_arm_decisions": len(robust_arm_decisions),
        "method_predictions": len(method_predictions),
    }
    expected_counts = {
        "case_action_confusions": CASE_ACTION_SUFFICIENT_STAT_COUNT,
        "correctness_observations": CORRECTNESS_OBSERVATION_COUNT,
        "model_fits": MODEL_FIT_COUNT,
        "directional_support_gains": DIRECTIONAL_SUPPORT_GAIN_COUNT,
        "donor_priors": DONOR_PRIOR_COUNT,
        "identification_decisions": IDENTIFICATION_DECISION_COUNT,
        "robust_arm_decisions": ROBUST_ARM_DECISION_COUNT,
        "method_predictions": METHOD_PREDICTION_COUNT,
    }
    if observed_counts != expected_counts or len(endpoint_arms) != ARM_IDENTITY_COUNT:
        raise ProtocolError(
            f"Dual-endpoint persisted route topology drifted: {observed_counts}."
        )
    tables = {
        "case_action_confusions": persist_rows(
            root / "tables/case_action_confusions.csv", case_action_confusions
        ),
        "correctness_observations": persist_rows(
            root / "tables/route_correctness_observations.csv",
            correctness_observations,
        ),
        "model_fits": persist_rows(root / "tables/route_model_fits.csv", model_fits),
        "directional_support_gains": persist_rows(
            root / "tables/directional_support_gains.csv",
            directional_support_gains,
        ),
        "donor_priors": persist_rows(root / "tables/donor_priors.csv", donor_priors),
        "endpoint_arms": persist_rows(root / "tables/endpoint_arms.csv", endpoint_arms),
        "identification_decisions": persist_rows(
            root / "tables/identification_decisions.csv",
            identification_decisions,
        ),
        "robust_arm_decisions": persist_rows(
            root / "tables/robust_arm_decisions.csv", robust_arm_decisions
        ),
        "method_predictions": persist_rows(
            root / "tables/method_predictions.csv", method_predictions
        ),
    }
    donor_seal = seal_payload(
        "fixed_bank_dual_endpoint_donor_prior_seal_v1",
        bindings={
            "loo_plan_seal_hash": loo_plan_seal_hash,
            "case_action_confusions_hash": canonical_hash(
                tables["case_action_confusions"]
            ),
            "directional_support_gains_hash": canonical_hash(
                tables["directional_support_gains"]
            ),
            "donor_priors_hash": canonical_hash(tables["donor_priors"]),
        },
        donor_scope="q_not_in_H_or_e",
        support_scope="H_minus_c_complete_case_block",
        donor_prior_count=DONOR_PRIOR_COUNT,
        raw_labels_persisted=False,
        support_derived_sufficient_stats_persisted=True,
    )
    identification_seal = seal_payload(
        "fixed_bank_dual_endpoint_identification_endpoint_seal_v1",
        bindings={
            "label_free_feature_seal_hash": feature_seal_hash,
            "donor_prior_seal_hash": str(donor_seal["seal_hash"]),
            "correctness_observations_hash": canonical_hash(
                tables["correctness_observations"]
            ),
            "route_model_fits_hash": canonical_hash(tables["model_fits"]),
            "identification_decisions_hash": canonical_hash(
                tables["identification_decisions"]
            ),
        },
        strict_positive_opportunity_and_case_proxy_gate=True,
        case_weight="4/5",
        donor_weight="1/5",
        invalid_route_fails_to_off=True,
        route_count=ROUTE_COUNT,
        identification_method_family_count=2,
        paired_identification_decision_count=IDENTIFICATION_DECISION_COUNT,
        model_fits_per_feature_surface=MODEL_FITS_PER_FEATURE_SURFACE,
        total_model_fit_count=MODEL_FIT_COUNT,
    )
    robust_seal = seal_payload(
        "fixed_bank_dual_endpoint_robust_endpoint_seal_v1",
        bindings={
            "donor_prior_seal_hash": str(donor_seal["seal_hash"]),
            "endpoint_arms_hash": canonical_hash(tables["endpoint_arms"]),
            "robust_arm_decisions_hash": canonical_hash(
                tables["robust_arm_decisions"]
            ),
        },
        arm_identity_count=ARM_IDENTITY_COUNT,
        method_family_count=len(ROBUST_METHOD_IDS),
        method_ids=list(ROBUST_METHOD_IDS),
        route_arm_decision_count=ROBUST_ARM_DECISION_COUNT,
        k_grid=[4, 5, 6],
        weight_grid=["1/2", "3/5", "7/10"],
        duplicate_arm_votes_preserved=True,
    )
    portfolio_seal = seal_payload(
        "fixed_bank_dual_endpoint_portfolio_prediction_seal_v1",
        bindings={
            "identification_endpoint_seal_hash": str(
                identification_seal["seal_hash"]
            ),
            "robust_endpoint_seal_hash": str(robust_seal["seal_hash"]),
            "method_predictions_hash": canonical_hash(tables["method_predictions"]),
            "route_decision_barrier_hash": str(
                route_barrier["decision_barrier_hash"]
            ),
        },
        identification_weight="3/5",
        robust_weight="2/5",
        probability_threshold=0.5,
        prediction_level_score_ensemble=True,
        terminal_labels_used=False,
        preterminal_method_count=11,
        method_prediction_count=METHOD_PREDICTION_COUNT,
    )
    aggregate_seal = seal_payload(
        "fixed_bank_dual_endpoint_aggregate_plan_decision_seal_v1",
        bindings={
            "loo_plan_seal_hash": loo_plan_seal_hash,
            "feature_seal_hash": feature_seal_hash,
            "donor_prior_seal_hash": str(donor_seal["seal_hash"]),
            "identification_endpoint_seal_hash": str(
                identification_seal["seal_hash"]
            ),
            "robust_endpoint_seal_hash": str(robust_seal["seal_hash"]),
            "portfolio_prediction_seal_hash": str(portfolio_seal["seal_hash"]),
            "route_decision_barrier_hash": str(
                route_barrier["decision_barrier_hash"]
            ),
        },
        route_count=ROUTE_COUNT,
        route_decision_seal_count=ROUTE_COUNT,
        directional_support_gain_count=DIRECTIONAL_SUPPORT_GAIN_COUNT,
        all_route_decisions_and_endpoint_probabilities_sealed=True,
        terminal_labels_used=False,
    )
    seals = {
        "donor": donor_seal,
        "identification": identification_seal,
        "robust": robust_seal,
        "portfolio": portfolio_seal,
        "aggregate": aggregate_seal,
    }
    members = {
        "donor": "donor_prior_seal.json",
        "identification": "identification_endpoint_seal.json",
        "robust": "robust_endpoint_seal.json",
        "portfolio": "portfolio_prediction_seal.json",
        "aggregate": "aggregate_plan_decision_seal.json",
    }
    for key, payload in seals.items():
        persist_json(root / "manifests" / members[key], payload)
    return seals


TERMINAL_TABLES = {
    "case_confusions": "terminal_case_confusions.csv",
    "method_metrics": "terminal_method_metrics.csv",
    "center_metrics": "terminal_center_metrics.csv",
    "contrasts": "terminal_contrasts.csv",
    "identification_metrics": "router_identification_metrics.csv",
    "calibration_metrics": "calibration_metrics.csv",
    "delete_one_center": "whole_pipeline_delete_one_center.csv",
    "attribution_controls": "attribution_controls.csv",
}


def persist_terminal(
    root: Path,
    *,
    result: Mapping[str, object],
    capability_report: Mapping[str, object],
    leakage_report: Mapping[str, object],
    publication_decision: Mapping[str, object],
    runtime_summary: Mapping[str, object],
) -> None:
    for key, filename in TERMINAL_TABLES.items():
        rows = result.get(key)
        if not isinstance(rows, (tuple, list)) or not rows:
            raise ProtocolError(f"Dual-endpoint terminal surface absent: {key}.")
        persist_rows(root / "tables" / filename, rows)
    terminal_seal = result.get("terminal_seal")
    if not isinstance(terminal_seal, Mapping):
        raise ProtocolError("Dual-endpoint terminal seal absent.")
    persist_json(root / "manifests/terminal_evaluation_seal.json", terminal_seal)
    persist_json(root / "reports/label_capability_report.json", capability_report)
    persist_json(root / "reports/leakage_report.json", leakage_report)
    persist_json(root / "reports/publication_decision.json", publication_decision)
    persist_json(root / "reports/runtime_summary.json", runtime_summary)


def persist_validation_report(root: Path, checks: Mapping[str, object]) -> None:
    from ...runtime.artifact_io import read_json
    from .fresh_process_validation import (
        ATTESTATION_KEY,
        verify_attested_validation_checks,
    )

    converted = json_native(checks)
    if not isinstance(converted, dict):
        raise ProtocolError("Dual-endpoint validation report malformed.")
    reconstructed = {
        key: value for key, value in converted.items() if key != ATTESTATION_KEY
    }
    converted = dict(
        verify_attested_validation_checks(
            converted,
            expected_reconstructed_checks=reconstructed,
            persisted_attestation=read_json(
                root / "reports/fresh_process_attestation.json"
            ),
        )
    )
    payload = {
        "schema_version": "fixed_bank_dual_endpoint_validation_report_v1",
        **converted,
    }
    persist_json(root / "reports/validation_report.json", payload)


def write_run_state(
    root: Path,
    *,
    status: str,
    phase: str,
    error: str | None = None,
    error_class: str | None = None,
) -> None:
    path = root / "reports/run_state.json"
    if path.is_symlink():
        raise ProtocolError("Dual-endpoint run state is a symlink.")
    atomic_json(
        path,
        run_state_payload(
            status=status,
            phase=phase,
            error=error,
            error_class=error_class,
        ),
    )


__all__ = (
    "TERMINAL_TABLES",
    "persist_initial_surfaces",
    "persist_json",
    "persist_label_free_products",
    "persist_physical_prelabel",
    "persist_route_products",
    "persist_rows",
    "persist_terminal",
    "persist_validation_report",
    "read_rows",
    "write_run_state",
)
