"""Reconstruct both endpoints, controls, prediction ledger, and lineage seals."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .artifact_rows import row_payload
from .artifact_topology import (
    ARM_IDENTITY_COUNT,
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
from .artifact_writers import read_rows
from .composition import (
    compose_calibration_only_predictions,
    compose_portfolio_predictions,
)
from .constants import PRE_TERMINAL_METHOD_IDS
from .hashing import canonical_hash
from .predictions import (
    compose_identification_predictions,
    compose_physical_action_predictions,
    compose_robust_predictions,
)
from .reports import seal_payload


def reconstruct_endpoint_products(
    root: Path,
    *,
    surface: object,
    route_products: Mapping[str, object],
    label_firewall: object,
) -> Mapping[str, object]:
    plans = tuple(route_products["plans"])
    identification = tuple(route_products["identification_decisions"])
    robust = tuple(route_products["robust_arm_decisions"])
    predictions = _compose_predictions(surface, identification, robust)
    observed = read_rows(root / "tables/method_predictions.csv")
    expected = tuple(row.to_payload() for row in predictions)
    if observed != expected or len(expected) != METHOD_PREDICTION_COUNT:
        raise ProtocolError("Dual-endpoint method prediction ledger drifted.")
    for plan in plans:
        key = (plan.target_center, plan.case_id)
        label_firewall.record_route_decision_seal(
            *key,
            canonical_hash(
                {
                    "identification_decisions": [
                        row.to_payload()
                        for row in identification
                        if (row.target_center, row.case_id) == key
                    ],
                    "robust_arm_decisions": [
                        row.to_payload()
                        for row in robust
                        if (row.target_center, row.case_id) == key
                    ],
                    "preterminal_predictions": [
                        row.to_payload()
                        for row in predictions
                        if (row.target_center, row.case_id) == key
                    ],
                }
            ),
        )
    barrier = label_firewall.decision_barrier_payload()
    seals = _expected_seals(root, route_products=route_products, barrier=barrier)
    filenames = {
        "donor": "donor_prior_seal.json",
        "identification": "identification_endpoint_seal.json",
        "robust": "robust_endpoint_seal.json",
        "portfolio": "portfolio_prediction_seal.json",
        "aggregate": "aggregate_plan_decision_seal.json",
    }
    for key, filename in filenames.items():
        if read_json(root / "manifests" / filename) != seals[key]:
            raise ProtocolError(f"Dual-endpoint seal is not reconstructive: {filename}.")
    label_firewall.record_aggregate_plan_decision_seal(
        str(seals["aggregate"]["seal_hash"]),
        plan_seal_hash=str(label_firewall.plan_seal_hash),
        decision_barrier_hash=str(barrier["decision_barrier_hash"]),
    )
    return {"method_predictions": predictions, "barrier": barrier, "seals": seals}


def _compose_predictions(
    surface: object,
    identification: Sequence[object],
    robust: Sequence[object],
) -> tuple[object, ...]:
    i_by_method = {
        method: tuple(row for row in identification if row.method_id == method)
        for method in ("I_OPPORTUNITY_GATED", "I_FEATURE_BLOCK_PERMUTED")
    }
    r_by_method = {
        method: tuple(row for row in robust if row.method_id == method)
        for method in ROBUST_METHOD_IDS
    }
    b = compose_physical_action_predictions(surface, action_id="B")
    u = compose_physical_action_predictions(surface, action_id="U")
    i = compose_identification_predictions(surface, i_by_method["I_OPPORTUNITY_GATED"])
    r = compose_robust_predictions(surface, r_by_method["R_NINE_ARM_ROBUST"])
    portfolio = compose_portfolio_predictions(i, r)
    calibration = compose_calibration_only_predictions(b, r)
    permuted_i = compose_identification_predictions(
        surface, i_by_method["I_FEATURE_BLOCK_PERMUTED"]
    )
    permuted_portfolio = compose_portfolio_predictions(
        permuted_i, r, method_id="OGDE_FEATURE_BLOCK_PERMUTED"
    )
    gate = compose_identification_predictions(
        surface, i_by_method["I_OPPORTUNITY_GATED"], control="gate_only"
    )
    source = compose_identification_predictions(
        surface, i_by_method["I_OPPORTUNITY_GATED"], control="source_only"
    )
    matched = compose_robust_predictions(surface, r_by_method["G_DIRECTIONAL_MATCHED"])
    by_method = {
        rows[0].method_id: rows
        for rows in (
            b,
            u,
            i,
            r,
            portfolio,
            calibration,
            permuted_i,
            permuted_portfolio,
            gate,
            source,
            matched,
        )
    }
    if tuple(by_method) != PRE_TERMINAL_METHOD_IDS or any(
        len(rows) != 9_928 for rows in by_method.values()
    ):
        raise ProtocolError("Dual-endpoint reconstructed method topology drifted.")
    return tuple(row for method in PRE_TERMINAL_METHOD_IDS for row in by_method[method])


def _expected_seals(
    root: Path,
    *,
    route_products: Mapping[str, object],
    barrier: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    tables = {
        name: read_rows(root / "tables" / filename)
        for name, filename in {
            "counts": "case_action_confusions.csv",
            "observations": "route_correctness_observations.csv",
            "models": "route_model_fits.csv",
            "gains": "directional_support_gains.csv",
            "priors": "donor_priors.csv",
            "arms": "endpoint_arms.csv",
            "identification": "identification_decisions.csv",
            "robust": "robust_arm_decisions.csv",
            "predictions": "method_predictions.csv",
        }.items()
    }
    plan_hash = str(route_products["persisted_plan_seal"]["seal_hash"])
    feature_hash = str(route_products["feature_seal"]["seal_hash"])
    donor = seal_payload(
        "fixed_bank_dual_endpoint_donor_prior_seal_v1",
        bindings={
            "loo_plan_seal_hash": plan_hash,
            "case_action_confusions_hash": canonical_hash(tables["counts"]),
            "directional_support_gains_hash": canonical_hash(tables["gains"]),
            "donor_priors_hash": canonical_hash(tables["priors"]),
        },
        donor_scope="q_not_in_H_or_e",
        support_scope="H_minus_c_complete_case_block",
        donor_prior_count=DONOR_PRIOR_COUNT,
        raw_labels_persisted=False,
        support_derived_sufficient_stats_persisted=True,
    )
    identification = seal_payload(
        "fixed_bank_dual_endpoint_identification_endpoint_seal_v1",
        bindings={
            "label_free_feature_seal_hash": feature_hash,
            "donor_prior_seal_hash": donor["seal_hash"],
            "correctness_observations_hash": canonical_hash(tables["observations"]),
            "route_model_fits_hash": canonical_hash(tables["models"]),
            "identification_decisions_hash": canonical_hash(tables["identification"]),
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
    robust = seal_payload(
        "fixed_bank_dual_endpoint_robust_endpoint_seal_v1",
        bindings={
            "donor_prior_seal_hash": donor["seal_hash"],
            "endpoint_arms_hash": canonical_hash(tables["arms"]),
            "robust_arm_decisions_hash": canonical_hash(tables["robust"]),
        },
        arm_identity_count=ARM_IDENTITY_COUNT,
        method_family_count=len(ROBUST_METHOD_IDS),
        method_ids=list(ROBUST_METHOD_IDS),
        route_arm_decision_count=ROBUST_ARM_DECISION_COUNT,
        k_grid=[4, 5, 6],
        weight_grid=["1/2", "3/5", "7/10"],
        duplicate_arm_votes_preserved=True,
    )
    portfolio = seal_payload(
        "fixed_bank_dual_endpoint_portfolio_prediction_seal_v1",
        bindings={
            "identification_endpoint_seal_hash": identification["seal_hash"],
            "robust_endpoint_seal_hash": robust["seal_hash"],
            "method_predictions_hash": canonical_hash(tables["predictions"]),
            "route_decision_barrier_hash": barrier["decision_barrier_hash"],
        },
        identification_weight="3/5",
        robust_weight="2/5",
        probability_threshold=0.5,
        prediction_level_score_ensemble=True,
        terminal_labels_used=False,
        preterminal_method_count=11,
        method_prediction_count=METHOD_PREDICTION_COUNT,
    )
    aggregate = seal_payload(
        "fixed_bank_dual_endpoint_aggregate_plan_decision_seal_v1",
        bindings={
            "loo_plan_seal_hash": plan_hash,
            "feature_seal_hash": feature_hash,
            "donor_prior_seal_hash": donor["seal_hash"],
            "identification_endpoint_seal_hash": identification["seal_hash"],
            "robust_endpoint_seal_hash": robust["seal_hash"],
            "portfolio_prediction_seal_hash": portfolio["seal_hash"],
            "route_decision_barrier_hash": barrier["decision_barrier_hash"],
        },
        route_count=ROUTE_COUNT,
        route_decision_seal_count=ROUTE_COUNT,
        directional_support_gain_count=DIRECTIONAL_SUPPORT_GAIN_COUNT,
        all_route_decisions_and_endpoint_probabilities_sealed=True,
        terminal_labels_used=False,
    )
    return {
        "donor": donor,
        "identification": identification,
        "robust": robust,
        "portfolio": portfolio,
        "aggregate": aggregate,
    }


__all__ = ("reconstruct_endpoint_products",)
