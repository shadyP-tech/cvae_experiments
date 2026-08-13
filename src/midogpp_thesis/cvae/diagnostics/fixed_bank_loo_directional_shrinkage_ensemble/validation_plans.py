"""Exact validation of persisted LOO plans, priors, endpoints, and decisions."""

from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .hashing import canonical_hash
from .persistence import read_rows
from .reports import seal_payload
from .nulls import (
    CandidateIdentityNullPlan,
    validate_candidate_identity_null_plan_contract,
)


def validate_plan_and_decision_products(
    root: Path,
    *,
    plans: Sequence[object],
    case_action_confusions: Sequence[object],
    directional_gains: Sequence[object],
    donor_priors: Sequence[object],
    endpoint_arms: Sequence[object],
    arm_decisions: Sequence[object],
    control_decisions: Sequence[object],
    method_predictions: Sequence[object],
    descriptive_control_predictions: Sequence[object],
    physical_prelabel_seal_hash: str,
    global_plan_seal_hash: str,
    route_decision_barrier: Mapping[str, object],
    null_plan: CandidateIdentityNullPlan,
    null_contract: Mapping[str, object],
) -> Mapping[str, object]:
    null_contract_check = validate_candidate_identity_null_plan_contract(
        null_plan, null_contract
    )
    expected = {
        "plans": _payloads(plans),
        "case_action_confusions": _payloads(case_action_confusions),
        "directional_gains": _payloads(directional_gains),
        "donor_priors": _payloads(donor_priors),
        "endpoint_arms": _payloads(endpoint_arms),
        "arm_decisions": _payloads(arm_decisions),
        "control_decisions": _payloads(control_decisions),
        "method_predictions": _payloads(method_predictions),
        "descriptive_control_predictions": _payloads(
            descriptive_control_predictions
        ),
    }
    if (
        len(expected["plans"]) != 218
        or len(expected["case_action_confusions"]) != 218 * 10
        or len(expected["directional_gains"]) != 218 * 16
        or len(expected["donor_priors"]) != 9 * 16
        or len(expected["endpoint_arms"]) != 9
        or len(expected["arm_decisions"]) != 218 * 18
        or len(expected["control_decisions"]) != 218 * 2
        or len(expected["method_predictions"]) != 9_928 * 6
        or len(expected["descriptive_control_predictions"]) != 9_928 * 5
        or {str(row["method_id"]) for row in expected["method_predictions"]}
        != {
            "B",
            "U",
            "DCSE_LOO",
            "G_directional_matched",
            "DLOO_raw",
            "LOO_frequency_committee",
        }
        or {
            str(row["method_id"])
            for row in expected["descriptive_control_predictions"]
        }
        != {
            "DCSE_hard_vote_descriptive",
            "DCSE_unique_mean_descriptive",
            "uniform_A1_mean_descriptive",
            "DCSE_zero_to_one_only_descriptive",
            "DCSE_one_to_zero_only_descriptive",
        }
    ):
        raise ProtocolError(
            "Directional-shrinkage reconstructed preterminal topology drifted."
        )
    members = {
        "plans": "tables/loo_plans.csv",
        "case_action_confusions": "tables/case_action_confusions.csv",
        "directional_gains": "tables/directional_gains.csv",
        "donor_priors": "tables/donor_priors.csv",
        "endpoint_arms": "tables/endpoint_arms.csv",
        "arm_decisions": "tables/arm_decisions.csv",
        "control_decisions": "tables/control_decisions.csv",
        "method_predictions": "tables/method_predictions.csv",
        "descriptive_control_predictions": (
            "tables/descriptive_control_predictions.csv"
        ),
    }
    for key, member in members.items():
        if read_rows(root / member) != expected[key]:
            raise ProtocolError(
                f"Directional-shrinkage persisted {key} are not reconstructive."
            )
    plan_seal = seal_payload(
        "fixed_bank_dcse_loo_plan_seal_v1",
        bindings={
            "physical_prelabel_seal_hash": physical_prelabel_seal_hash,
            "plans_hash": canonical_hash(expected["plans"]),
            "case_action_confusions_hash": canonical_hash(
                expected["case_action_confusions"]
            ),
            "directional_gains_hash": canonical_hash(expected["directional_gains"]),
        },
        plan_count=len(expected["plans"]),
        held_case_count=218,
        each_plan_excludes_held_whole_case=True,
        terminal_labels_used=False,
        raw_labels_persisted=False,
    )
    prior_seal = seal_payload(
        "fixed_bank_dcse_donor_prior_seal_v1",
        bindings={
            "loo_plan_seal_hash": plan_seal["seal_hash"],
            "donor_priors_hash": canonical_hash(expected["donor_priors"]),
        },
        donor_prior_count=len(expected["donor_priors"]),
        strict_target_and_source_exclusion=True,
        equal_query_center_aggregation=True,
        terminal_labels_used=False,
    )
    endpoint_seal = seal_payload(
        "fixed_bank_dcse_endpoint_library_seal_v1",
        bindings={
            "donor_prior_seal_hash": prior_seal["seal_hash"],
            "endpoint_library_hash": canonical_hash(expected["endpoint_arms"]),
        },
        arm_count=len(expected["endpoint_arms"]),
        all_nine_arm_identities_retained=True,
        terminal_labels_used=False,
    )
    decision_seal = seal_payload(
        "fixed_bank_dcse_arm_decisions_seal_v1",
        bindings={
            "loo_plan_seal_hash": plan_seal["seal_hash"],
            "global_plan_seal_hash": global_plan_seal_hash,
            "donor_prior_seal_hash": prior_seal["seal_hash"],
            "endpoint_library_seal_hash": endpoint_seal["seal_hash"],
            "arm_decisions_hash": canonical_hash(expected["arm_decisions"]),
            "control_decisions_hash": canonical_hash(
                expected["control_decisions"]
            ),
            "method_predictions_hash": canonical_hash(expected["method_predictions"]),
            "descriptive_control_predictions_hash": canonical_hash(
                expected["descriptive_control_predictions"]
            ),
        },
        arm_decision_count=len(expected["arm_decisions"]),
        control_decision_count=len(expected["control_decisions"]),
        method_prediction_count=len(expected["method_predictions"]),
        descriptive_control_prediction_count=len(
            expected["descriptive_control_predictions"]
        ),
        preterminal_method_ids=sorted(
            {str(row["method_id"]) for row in expected["method_predictions"]}
        ),
        descriptive_control_method_ids=sorted(
            {
                str(row["method_id"])
                for row in expected["descriptive_control_predictions"]
            }
        ),
        terminal_labels_used=False,
    )
    aggregate = seal_payload(
        "fixed_bank_dcse_aggregate_plan_decision_seal_v1",
        bindings={
            "loo_plan_seal_hash": plan_seal["seal_hash"],
            "global_plan_seal_hash": global_plan_seal_hash,
            "donor_prior_seal_hash": prior_seal["seal_hash"],
            "endpoint_library_seal_hash": endpoint_seal["seal_hash"],
            "arm_decisions_seal_hash": decision_seal["seal_hash"],
            "control_decisions_hash": canonical_hash(
                expected["control_decisions"]
            ),
            "method_predictions_hash": canonical_hash(
                expected["method_predictions"]
            ),
            "descriptive_control_predictions_hash": canonical_hash(
                expected["descriptive_control_predictions"]
            ),
            "route_decision_barrier_hash": route_decision_barrier[
                "decision_barrier_hash"
            ],
            "ordered_route_decision_seals_hash": canonical_hash(
                route_decision_barrier["decision_seals"]
            ),
            "candidate_identity_null_plan_hash": null_plan.plan_hash,
            "candidate_identity_null_permutation_sha256": _validate_null_digest(
                null_plan
            ),
        },
        all_218_loo_plans_complete=True,
        all_nine_arm_decisions_complete_per_case=True,
        all_two_control_decisions_complete_per_case=True,
        all_preterminal_method_predictions_complete=True,
        all_descriptive_control_predictions_complete=True,
        control_decision_count=len(expected["control_decisions"]),
        method_prediction_count=len(expected["method_predictions"]),
        descriptive_control_prediction_count=len(
            expected["descriptive_control_predictions"]
        ),
        global_barrier_complete=True,
        candidate_identity_null_plan=null_plan.to_payload(),
        null_plan_sealed_before_terminal_labels=True,
        null_plan_can_change_canonical_decisions=False,
        terminal_labels_used=False,
    )
    seals = {
        "manifests/loo_plan_seal.json": plan_seal,
        "manifests/donor_prior_seal.json": prior_seal,
        "manifests/endpoint_library_seal.json": endpoint_seal,
        "manifests/arm_decisions_seal.json": decision_seal,
        "manifests/aggregate_plan_decision_seal.json": aggregate,
    }
    for member, expected_seal in seals.items():
        if read_json(root / member) != expected_seal:
            raise ProtocolError(
                f"Directional-shrinkage decision barrier drifted: {member}."
            )
    return {
        "loo_plan_seal_hash": plan_seal["seal_hash"],
        "donor_prior_seal_hash": prior_seal["seal_hash"],
        "endpoint_library_seal_hash": endpoint_seal["seal_hash"],
        "arm_decisions_seal_hash": decision_seal["seal_hash"],
        "aggregate_plan_decision_seal_hash": aggregate["seal_hash"],
        "loo_plan_count": len(expected["plans"]),
        "arm_decision_count": len(expected["arm_decisions"]),
        "control_decision_count": len(expected["control_decisions"]),
        "method_prediction_count": len(expected["method_predictions"]),
        "descriptive_control_prediction_count": len(
            expected["descriptive_control_predictions"]
        ),
        **null_contract_check,
    }


def _payloads(values: Sequence[object]) -> tuple[dict[str, object], ...]:
    from .persistence import object_payload

    rows = tuple(object_payload(value) for value in values)
    if not rows:
        raise ProtocolError("Directional-shrinkage reconstructed product is empty.")
    return rows


def _validate_null_digest(null_plan: CandidateIdentityNullPlan) -> str:
    """Regenerate all 10k paired-direction blocks before trusting the seal."""

    regenerated = null_plan.materialize()
    digest = hashlib.sha256(regenerated.tobytes(order="C")).hexdigest()
    if digest != null_plan.permutation_sha256:
        raise ProtocolError("Directional-shrinkage null permutation digest drifted.")
    return digest


__all__ = (
    "CandidateIdentityNullPlan",
    "validate_plan_and_decision_products",
)
