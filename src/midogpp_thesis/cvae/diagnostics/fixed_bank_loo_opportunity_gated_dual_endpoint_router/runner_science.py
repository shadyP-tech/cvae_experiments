"""Route-scoped label use, endpoint construction, and preterminal composition."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .artifact_rows import row_payload
from .composition import (
    compose_calibration_only_predictions,
    compose_portfolio_predictions,
)
from .constants import CENTERS, PRE_TERMINAL_METHOD_IDS, candidate_sources
from .donor_prior import compute_donor_priors
from .hashing import canonical_hash
from .persistence import persist_route_products
from .predictions import (
    compose_identification_predictions,
    compose_physical_action_predictions,
    compose_robust_predictions,
)
from .response_products import deduplicate_sufficient_stats
from .response_scoring import score_case_action_confusions
from .robust import build_endpoint_arms
from .runner_runtime import execute_route_jobs


def execute_and_compose_route_products(
    *,
    root: Path,
    config: object,
    surface: object,
    plans: Sequence[object],
    features: Sequence[object],
    label_firewall: object,
    persisted_plan_seal: Mapping[str, object],
    feature_seal: Mapping[str, object],
) -> Mapping[str, object]:
    """Complete every donor grant before opening any route-support grant."""

    donor_counts: list[object] = []
    donor_priors: list[object] = []
    priors_by_target: dict[str, tuple[object, ...]] = {}
    for target in CENTERS:
        counts_by_source: dict[str, tuple[object, ...]] = {}
        for source in candidate_sources(target):
            labels = label_firewall.open_donor_labels(target, source)
            counts = tuple(score_case_action_confusions(surface, labels))
            counts_by_source[source] = counts
            donor_counts.extend(counts)
        priors = tuple(compute_donor_priors(counts_by_source, heldout_center=target))
        priors_by_target[target] = priors
        donor_priors.extend(priors)

    jobs: list[dict[str, object]] = []
    for plan in plans:
        labels = label_firewall.open_route_support_labels(
            plan.target_center, plan.case_id, plan_hash=plan.plan_hash
        )
        route_cases = {*plan.support_case_ids, plan.case_id}
        jobs.append(
            {
                "plan": plan,
                "support_labels": tuple(labels),
                "donor_priors": priors_by_target[plan.target_center],
                "route_features": tuple(
                    row
                    for row in features
                    if row.target_center == plan.target_center
                    and row.case_id in route_cases
                ),
            }
        )
    runtime = getattr(config, "runtime")
    route_results = execute_route_jobs(
        surface,
        jobs,
        workers=runtime["route_model_workers"],  # type: ignore[arg-type]
        threads_per_worker=runtime["classifier_threads_per_worker"],  # type: ignore[arg-type]
    )
    if len(route_results) != 218:
        raise ProtocolError("Dual-endpoint route result topology drifted.")

    route_counts = tuple(
        row for result in route_results for row in result.case_action_confusions
    )
    # Capability events retain the donor/route scopes.  The persisted table is
    # the unique scope-independent numeric sufficient-stat surface; conflicting
    # repeated views fail closed in the science helper.
    all_counts = deduplicate_sufficient_stats((*donor_counts, *route_counts))
    gains = tuple(
        row for result in route_results for row in result.directional_support_gains
    )
    identification = tuple(
        row for result in route_results for row in result.identification_decisions
    )
    robust = tuple(
        row for result in route_results for row in result.robust_arm_decisions
    )
    if len(identification) != 436 or len(robust) != 3_924:
        raise ProtocolError(
            "Dual-endpoint decision topology must be 2 I and 18 robust arms per route."
        )
    observations = tuple(
        _tag(row, "canonical")
        for result in route_results
        for row in result.correctness_observations_primary
    ) + tuple(
        _tag(row, "candidate_feature_block_permuted")
        for result in route_results
        for row in result.correctness_observations_permuted
    )
    models = tuple(
        _tag(row, "canonical")
        for result in route_results
        for row in result.model_fits_primary
    ) + tuple(
        _tag(row, "candidate_feature_block_permuted")
        for result in route_results
        for row in result.model_fits_permuted
    )

    decisions_by_method = {
        method: tuple(row for row in identification if row.method_id == method)
        for method in ("I_OPPORTUNITY_GATED", "I_FEATURE_BLOCK_PERMUTED")
    }
    robust_by_method = {
        method: tuple(row for row in robust if row.method_id == method)
        for method in ("R_NINE_ARM_ROBUST", "G_DIRECTIONAL_MATCHED")
    }
    b = compose_physical_action_predictions(surface, action_id="B")
    u = compose_physical_action_predictions(surface, action_id="U")
    i = compose_identification_predictions(
        surface, decisions_by_method["I_OPPORTUNITY_GATED"]
    )
    i_permuted = compose_identification_predictions(
        surface, decisions_by_method["I_FEATURE_BLOCK_PERMUTED"]
    )
    gate_only = compose_identification_predictions(
        surface, decisions_by_method["I_OPPORTUNITY_GATED"], control="gate_only"
    )
    source_only = compose_identification_predictions(
        surface, decisions_by_method["I_OPPORTUNITY_GATED"], control="source_only"
    )
    robust_predictions = compose_robust_predictions(
        surface, robust_by_method["R_NINE_ARM_ROBUST"]
    )
    matched = compose_robust_predictions(
        surface, robust_by_method["G_DIRECTIONAL_MATCHED"]
    )
    portfolio = compose_portfolio_predictions(i, robust_predictions)
    permuted_portfolio = compose_portfolio_predictions(
        i_permuted,
        robust_predictions,
        method_id="OGDE_FEATURE_BLOCK_PERMUTED",
    )
    calibration = compose_calibration_only_predictions(b, robust_predictions)
    by_method = {
        row[0].method_id: row
        for row in (
            b,
            u,
            i,
            robust_predictions,
            portfolio,
            calibration,
            i_permuted,
            permuted_portfolio,
            gate_only,
            source_only,
            matched,
        )
    }
    if tuple(by_method) != PRE_TERMINAL_METHOD_IDS or any(
        len(rows) != 9_928 for rows in by_method.values()
    ):
        raise ProtocolError("Dual-endpoint preterminal method topology drifted.")
    predictions = tuple(
        row for method in PRE_TERMINAL_METHOD_IDS for row in by_method[method]
    )

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
    seals = persist_route_products(
        root,
        case_action_confusions=all_counts,
        correctness_observations=observations,
        model_fits=models,
        directional_support_gains=gains,
        donor_priors=tuple(donor_priors),
        endpoint_arms=build_endpoint_arms(),
        identification_decisions=identification,
        robust_arm_decisions=robust,
        method_predictions=predictions,
        loo_plan_seal_hash=str(persisted_plan_seal["seal_hash"]),
        feature_seal_hash=str(feature_seal["seal_hash"]),
        route_barrier=barrier,
    )
    aggregate = seals["aggregate"]
    label_firewall.record_aggregate_plan_decision_seal(
        str(aggregate["seal_hash"]),
        plan_seal_hash=str(label_firewall.plan_seal_hash),
        decision_barrier_hash=str(barrier["decision_barrier_hash"]),
    )
    return {
        "case_action_confusions": all_counts,
        "directional_support_gains": gains,
        "donor_priors": tuple(donor_priors),
        "identification_decisions": identification,
        "robust_arm_decisions": robust,
        "method_predictions": predictions,
        "route_barrier": barrier,
        "seals": seals,
    }


def _tag(row: object, feature_surface_id: str) -> dict[str, object]:
    return {"feature_surface_id": feature_surface_id, **row_payload(row)}


__all__ = ("execute_and_compose_route_products",)
