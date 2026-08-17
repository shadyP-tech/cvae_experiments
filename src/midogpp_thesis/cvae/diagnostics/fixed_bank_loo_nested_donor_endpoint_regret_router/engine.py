"""Phase-separated scientific engine for the nested donor-regret router."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Callable, Mapping, Sequence

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    EXPECTED_ENDPOINT_MODEL_FIT_COUNT,
    EXPECTED_ORDERED_VOTER_COUNT,
    LTT_METHOD_ID,
    MODEL_BASED_METHOD_ID,
    candidate_sources,
)
from .contracts import (
    BinaryLabel,
    CandidateDescriptor,
    CenterBalancedRidgeModel,
    DonorRegretRow,
    EndpointCasePrediction,
    PhysicalProbabilitySurface,
    RouteDecision,
)
from .controls import (
    PRIMARY_POLICY,
    RoutePolicySpec,
    predeclared_policy_menu,
    select_route_for_policy,
)
from .donor_regret_model import fit_full_and_delete_donor_models
from .endpoint_reconstruction import (
    PreparedCenter,
    build_center_case_outcomes,
    compute_donor_priors,
    prepare_center,
)
from .hashing import canonical_hash
from .label_capabilities import NestedRegretLabelFirewall
from .ltt_execution import TargetLTTAuthorization, build_ltt_authorizations
from .nested_endpoint_regret import build_donor_regret_row
from .protocol import FrozenProtocol, build_frozen_protocol
from .route_worker_runtime import (
    CenterEndpointJob,
    CenterEndpointProducts,
    compute_center_endpoint_products,
    execute_center_endpoint_jobs,
    recompose_center_endpoint_products,
)
from .split_plans import NestedPlanSeal, build_nested_plans


LabelLoader = Callable[
    [frozenset[tuple[str, str, str]], str], Sequence[object]
]


@dataclass(frozen=True)
class PreterminalResult:
    protocol: FrozenProtocol
    surface: PhysicalProbabilitySurface
    plans: NestedPlanSeal
    endpoint_products: tuple[CenterEndpointProducts, ...]
    donor_endpoint_products: Mapping[tuple[str, str], CenterEndpointProducts]
    predictions_by_center: Mapping[str, tuple[EndpointCasePrediction, ...]]
    descriptors_by_center: Mapping[str, tuple[CandidateDescriptor, ...]]
    donor_rows_by_outer_target: Mapping[str, tuple[DonorRegretRow, ...]]
    full_models_by_target: Mapping[
        str, Mapping[str, CenterBalancedRidgeModel]
    ]
    delete_models_by_target: Mapping[
        str, Mapping[str, Mapping[str, CenterBalancedRidgeModel]]
    ]
    policy_menu: tuple[RoutePolicySpec, ...]
    decisions_by_policy: Mapping[str, tuple[RouteDecision, ...]]
    ltt_authorizations: tuple[TargetLTTAuthorization, ...]
    decision_barrier: Mapping[str, object]
    aggregate_seal: Mapping[str, object]
    label_firewall: NestedRegretLabelFirewall


def build_preterminal_result(
    surface: PhysicalProbabilitySurface,
    label_loader: LabelLoader,
    *,
    use_processes: bool = True,
) -> PreterminalResult:
    """Seal every route decision before terminal evaluation labels can open."""

    protocol = build_frozen_protocol()
    prepared = MappingProxyType(
        {center: prepare_center(surface.centers[center]) for center in CENTERS}
    )
    plans = build_nested_plans(
        _surface_identities(surface),
        probability_surface_hash=surface.surface_hash,
        strict_canonical_topology=surface.strict_canonical_topology,
    )
    firewall = NestedRegretLabelFirewall(plans, label_loader)

    source_prior_labels: dict[tuple[str, str], tuple[BinaryLabel, ...]] = {}
    donor_priors: dict[str, Mapping[tuple[str, str], float]] = {}
    for endpoint_target in CENTERS:
        labels_by_source: dict[str, Mapping[str, tuple[BinaryLabel, ...]]] = {}
        for source in candidate_sources(endpoint_target):
            labels = firewall.open_source_prior_labels(endpoint_target, source)
            source_prior_labels[(endpoint_target, source)] = labels
            legal_queries = tuple(
                center
                for center in CENTERS
                if center not in {endpoint_target, source}
            )
            labels_by_source[source] = MappingProxyType(
                {
                    center: tuple(row for row in labels if row.center == center)
                    for center in legal_queries
                }
            )
        donor_priors[endpoint_target] = compute_donor_priors(
            prepared, labels_by_source, heldout_center=endpoint_target
        )

    crossfit_donor_priors: dict[
        tuple[str, str], Mapping[tuple[str, str], float]
    ] = {}
    for outer in CENTERS:
        for donor in CENTERS:
            if donor == outer:
                continue
            labels_by_source = {}
            for source in candidate_sources(donor):
                grant = source_prior_labels[(donor, source)]
                legal_queries = tuple(
                    center
                    for center in CENTERS
                    if center not in {outer, donor, source}
                )
                labels_by_source[source] = MappingProxyType(
                    {
                        center: tuple(
                            row for row in grant if row.center == center
                        )
                        for center in legal_queries
                    }
                )
            crossfit_donor_priors[(outer, donor)] = compute_donor_priors(
                prepared,
                labels_by_source,
                heldout_center=donor,
                excluded_query_centers=(outer,),
            )

    regret_labels: dict[tuple[str, str], tuple[BinaryLabel, ...]] = {}
    for outer in CENTERS:
        for donor in CENTERS:
            if donor != outer:
                regret_labels[(outer, donor)] = firewall.open_regret_donor_labels(
                    outer, donor
                )

    jobs = _build_center_jobs(prepared, plans, donor_priors, firewall)
    if surface.strict_canonical_topology:
        endpoint_products = execute_center_endpoint_jobs(
            jobs, use_processes=use_processes
        )
    else:
        endpoint_products = tuple(
            compute_center_endpoint_products(job) for job in jobs
        )
    _record_endpoint_state_seals(firewall, endpoint_products)
    if surface.strict_canonical_topology and (
        sum(row.endpoint_model_fit_count for row in endpoint_products)
        != EXPECTED_ENDPOINT_MODEL_FIT_COUNT
        or sum(row.ordered_voter_count for row in endpoint_products)
        != EXPECTED_ORDERED_VOTER_COUNT
    ):
        raise ProtocolError("Canonical endpoint reconstruction workload drifted.")

    predictions_by_center = MappingProxyType(
        {row.target_center: row.outer_predictions for row in endpoint_products}
    )
    descriptors_by_center = MappingProxyType(
        {row.target_center: row.descriptors for row in endpoint_products}
    )
    job_by_center = {job.target_center: job for job in jobs}
    products_by_center = {
        row.target_center: row for row in endpoint_products
    }
    donor_endpoint_products = MappingProxyType(
        {
            (outer, donor): recompose_center_endpoint_products(
                job_by_center[donor],
                products_by_center[donor],
                donor_priors=crossfit_donor_priors[(outer, donor)],
            )
            for outer in CENTERS
            for donor in CENTERS
            if donor != outer
        }
    )
    donor_rows_by_target, full_models, delete_models = _fit_outer_donor_models(
        donor_endpoint_products,
        regret_labels,
    )

    policies = predeclared_policy_menu()
    decisions_by_policy_mutable: dict[
        str, dict[tuple[str, str], RouteDecision]
    ] = {policy.policy_id: {} for policy in policies}
    for outer in CENTERS:
        for descriptor in descriptors_by_center[outer]:
            for policy in policies:
                decision = select_route_for_policy(
                    descriptor,
                    policy,
                    full_models=(
                        full_models[outer] if policy.require_model else None
                    ),
                    delete_donor_models=(
                        delete_models[outer] if policy.require_model else None
                    ),
                )
                decisions_by_policy_mutable[policy.policy_id][
                    (outer, descriptor.case_id)
                ] = decision

    ltt_policies = tuple(
        policy
        for policy in policies
        if policy.policy_id == MODEL_BASED_METHOD_ID
        or policy.policy_id.startswith("SENSITIVITY_DISPERSION_")
    )
    ltt = build_ltt_authorizations(
        descriptors_by_center=descriptors_by_center,
        donor_endpoint_products=donor_endpoint_products,
        donor_rows_by_outer_target=donor_rows_by_target,
        donor_labels_by_outer_target=regret_labels,
        target_decisions_by_policy=decisions_by_policy_mutable,
        ltt_policies=ltt_policies,
    )
    ltt_by_key = {
        (row.target_center, decision.case_id): decision
        for row in ltt
        for decision in row.decisions
    }
    decisions_by_policy_mutable[LTT_METHOD_ID] = ltt_by_key

    policy_payloads = [policy.to_payload() for policy in policies]
    for outer in CENTERS:
        for descriptor in descriptors_by_center[outer]:
            key = (outer, descriptor.case_id)
            firewall.record_route_decision_seal(
                *key,
                canonical_hash(
                    {
                        "schema_version": "fixed_bank_nested_regret_case_route_seal_v1",
                        "descriptor_hash": descriptor.descriptor_hash,
                        "policy_menu": policy_payloads,
                        "policy_decisions": [
                            decisions_by_policy_mutable[policy.policy_id][key].to_payload()
                            for policy in policies
                        ],
                        "ltt_decision": ltt_by_key[key].to_payload(),
                        "terminal_labels_used": False,
                    }
                ),
            )
    barrier = firewall.decision_barrier_payload()
    aggregate_payload = {
        "schema_version": "fixed_bank_nested_regret_preterminal_aggregate_v1",
        "protocol_hash": protocol.protocol_hash,
        "probability_surface_hash": surface.surface_hash,
        "plan_seal_hash": plans.seal_hash,
        "decision_barrier_hash": barrier["decision_barrier_hash"],
        "descriptor_hash": canonical_hash(
            [
                row.descriptor_hash
                for center in CENTERS
                for row in descriptors_by_center[center]
            ]
        ),
        "outer_excluded_donor_descriptor_hash": canonical_hash(
            [
                row.descriptor_hash
                for outer in CENTERS
                for donor in CENTERS
                if donor != outer
                for row in donor_endpoint_products[(outer, donor)].descriptors
            ]
        ),
        "donor_model_hash": canonical_hash(
            [
                model.model_hash
                for center in CENTERS
                for models in (
                    full_models[center],
                    *delete_models[center].values(),
                )
                for model in models.values()
            ]
        ),
        "policy_menu_hash": canonical_hash(policy_payloads),
        "ltt_authorization_hash": canonical_hash(
            [row.authorization_hash for row in ltt]
        ),
        "terminal_labels_used": False,
    }
    aggregate = {
        **aggregate_payload,
        "aggregate_seal_hash": canonical_hash(aggregate_payload),
    }
    firewall.record_aggregate_seal(aggregate)
    decisions_by_policy = MappingProxyType(
        {
            policy_id: tuple(
                rows[(center, descriptor.case_id)]
                for center in CENTERS
                for descriptor in descriptors_by_center[center]
            )
            for policy_id, rows in decisions_by_policy_mutable.items()
        }
    )
    return PreterminalResult(
        protocol,
        surface,
        plans,
        endpoint_products,
        donor_endpoint_products,
        predictions_by_center,
        descriptors_by_center,
        donor_rows_by_target,
        full_models,
        delete_models,
        policies,
        decisions_by_policy,
        ltt,
        MappingProxyType(dict(barrier)),
        MappingProxyType(aggregate),
        firewall,
    )


def _surface_identities(
    surface: PhysicalProbabilitySurface,
) -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(
            center=center,
            case_id=case_id,
            sample_id=sample_id,
            group_id=case_id,
        )
        for center in CENTERS
        for sample_id, case_id in zip(
            surface.centers[center].sample_ids,
            surface.centers[center].case_ids,
            strict=True,
        )
    )


def _build_center_jobs(
    prepared: Mapping[str, PreparedCenter],
    plans: NestedPlanSeal,
    donor_priors: Mapping[str, Mapping[tuple[str, str], float]],
    firewall: NestedRegretLabelFirewall,
) -> tuple[CenterEndpointJob, ...]:
    jobs: list[CenterEndpointJob] = []
    for center in CENTERS:
        outer_plans = tuple(
            row for row in plans.outer_plans if row.target_center == center
        )
        pair_plans = tuple(
            row for row in plans.unordered_pair_plans if row.target_center == center
        )
        observed: dict[tuple[str, str, str], int] = {}
        for plan in outer_plans:
            labels = firewall.open_outer_support_labels(
                center, plan.case_id, plan_hash=plan.plan_hash
            )
            _merge_consistent_labels(observed, labels)
        for plan in pair_plans:
            labels = firewall.open_pair_support_labels(
                center,
                plan.first_case_id,
                plan.second_case_id,
                plan_hash=plan.plan_hash,
            )
            _merge_consistent_labels(observed, labels)
        expected = {
            (center, case_id, sample_id)
            for sample_id, case_id in zip(
                prepared[center].surface.sample_ids,
                prepared[center].surface.case_ids,
                strict=True,
            )
        }
        if set(observed) != expected:
            raise ProtocolError("Support capability union does not cover one center.")
        scope = f"derived_support_sufficient_stats::H={center}"
        case_order = tuple(plan.case_id for plan in outer_plans)
        labels_by_case = tuple(
            (
                case,
                tuple(
                    BinaryLabel(center, case, sample_id, observed[(center, case, sample_id)], scope)
                    for sample_id, observed_case in zip(
                        prepared[center].surface.sample_ids,
                        prepared[center].surface.case_ids,
                        strict=True,
                    )
                    if observed_case == case
                ),
            )
            for case in case_order
        )
        all_labels = tuple(
            label for _case, labels in labels_by_case for label in labels
        )
        outcomes = build_center_case_outcomes(prepared[center], all_labels)
        jobs.append(
            CenterEndpointJob(
                center,
                prepared[center],
                outcomes,
                labels_by_case,
                outer_plans,
                pair_plans,
                tuple(donor_priors[center].items()),
            )
        )
    return tuple(jobs)


def _merge_consistent_labels(
    observed: dict[tuple[str, str, str], int], labels: Sequence[BinaryLabel]
) -> None:
    for row in labels:
        previous = observed.setdefault(row.key, row.value)
        if previous != row.value:
            raise ProtocolError("Repeated support capabilities disagree on a label.")


def _record_endpoint_state_seals(
    firewall: NestedRegretLabelFirewall,
    products: Sequence[CenterEndpointProducts],
) -> None:
    for center in products:
        for case, digest in center.outer_state_hashes:
            firewall.record_outer_state_seal(center.target_center, case, digest)
        for first, second, digest in center.pair_state_hashes:
            firewall.record_pair_state_seal(
                center.target_center, first, second, digest
            )


def _fit_outer_donor_models(
    donor_endpoint_products: Mapping[
        tuple[str, str], CenterEndpointProducts
    ],
    regret_labels: Mapping[tuple[str, str], Sequence[BinaryLabel]],
) -> tuple[
    Mapping[str, tuple[DonorRegretRow, ...]],
    Mapping[str, Mapping[str, CenterBalancedRidgeModel]],
    Mapping[str, Mapping[str, Mapping[str, CenterBalancedRidgeModel]]],
]:
    donor_rows: dict[str, tuple[DonorRegretRow, ...]] = {}
    full_models: dict[str, Mapping[str, CenterBalancedRidgeModel]] = {}
    delete_models: dict[
        str, Mapping[str, Mapping[str, CenterBalancedRidgeModel]]
    ] = {}
    for outer in CENTERS:
        rows: list[DonorRegretRow] = []
        for donor in CENTERS:
            if donor == outer:
                continue
            labels = tuple(regret_labels[(outer, donor)])
            by_case = {
                case: tuple(row for row in labels if row.case_id == case)
                for case in dict.fromkeys(row.case_id for row in labels)
            }
            n_positive = sum(row.value == 1 for row in labels)
            n_negative = sum(row.value == 0 for row in labels)
            products = donor_endpoint_products[(outer, donor)]
            prediction_map = {
                row.case_id: row for row in products.outer_predictions
            }
            descriptor_map = {
                row.case_id: row for row in products.descriptors
            }
            if set(by_case) != set(prediction_map) or set(by_case) != set(descriptor_map):
                raise ProtocolError("Donor response cases do not align with descriptors.")
            for case in prediction_map:
                rows.append(
                    build_donor_regret_row(
                        descriptor_map[case],
                        prediction_map[case],
                        by_case[case],
                        center_case_count=len(by_case),
                        center_n_positive=n_positive,
                        center_n_negative=n_negative,
                        center_sample_count=len(labels),
                    )
                )
        donor_rows[outer] = tuple(rows)
        full, deleted = fit_full_and_delete_donor_models(
            rows, outer_target_center=outer
        )
        full_models[outer] = full
        delete_models[outer] = deleted
    return (
        MappingProxyType(donor_rows),
        MappingProxyType(full_models),
        MappingProxyType(delete_models),
    )


__all__ = (
    "LabelLoader",
    "PreterminalResult",
    "build_preterminal_result",
)
