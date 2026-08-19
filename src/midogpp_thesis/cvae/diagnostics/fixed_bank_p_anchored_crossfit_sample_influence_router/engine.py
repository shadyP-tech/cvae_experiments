"""Phase-separated preterminal engine for cross-fit sample influence routing."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Callable, Mapping, Sequence

from ...protocol import ProtocolError
from .bacc_influence import score_sample_influences
from .composition import compose_case_probabilities
from .constants import (
    BACC_ONLY_METHOD_ID,
    BLOCKED_FINGERPRINT_CONTROL_ID,
    CENTERS,
    COMPOSED_POLICY_IDS,
    EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
    EXPECTED_UTILITY_MODEL_FIT_COUNT,
    FULL_ONLY_METHOD_ID,
    MODEL_BASED_METHOD_ID,
    PERMUTATION_METHOD_ID,
    PRIMARY_FINGERPRINT_CONTROL_ID,
    candidate_sources,
)
from .contracts import BinaryLabel, EndpointCasePrediction, PhysicalProbabilitySurface
from .endpoint_reconstruction import (
    PreparedCenter,
    build_center_case_outcomes,
    compute_donor_priors,
    prepare_center,
)
from .hashing import canonical_hash
from .label_capabilities import PCSILabelFirewall
from .outer_endpoint_runtime import (
    OuterEndpointJob,
    OuterEndpointProducts,
    compute_outer_endpoint_products,
    execute_outer_endpoint_jobs,
    recompose_outer_endpoint_products,
)
from .outer_plans import OuterPlanSeal, build_outer_plans
from .physical_fingerprint import (
    blocked_within_case_fingerprint,
    build_physical_fingerprint_surface,
)
from .protocol import FrozenProtocol, build_frozen_protocol
from .sample_influence_contracts import (
    InfluencePrediction,
    PhysicalFingerprintSurface,
    TargetLocalPosteriorModel,
    TargetLocalPosteriorPrediction,
)
from .target_local_runtime import (
    TargetCenterPosteriorJob,
    TargetCenterPosteriorProducts,
    execute_target_center_posterior_jobs,
)
from .uncertainty import predict_utility_surface
from .utility_contracts import (
    ComposedCasePrediction,
    DonorUtilityRow,
    SignedUtilityModel,
    UtilityDescriptor,
    UtilityPrediction,
)
from .utility_features import build_utility_descriptor_surface
from .utility_model import fit_response_model_family
from .utility_responses import build_donor_utility_rows


LabelLoader = Callable[[frozenset[tuple[str, str, str]], str], Sequence[object]]


@dataclass(frozen=True)
class PreterminalResult:
    protocol: FrozenProtocol
    surface: PhysicalProbabilitySurface
    plans: OuterPlanSeal
    endpoint_products: tuple[OuterEndpointProducts, ...]
    donor_endpoint_products: Mapping[tuple[str, str], OuterEndpointProducts]
    predictions_by_center: Mapping[str, tuple[EndpointCasePrediction, ...]]
    utility_descriptors_by_center: Mapping[str, tuple[UtilityDescriptor, ...]]
    primary_fingerprints_by_center: Mapping[str, PhysicalFingerprintSurface]
    blocked_fingerprints_by_center: Mapping[str, PhysicalFingerprintSurface]
    target_posterior_models_by_control: Mapping[
        str, tuple[TargetLocalPosteriorModel, ...]
    ]
    target_posterior_predictions_by_control: Mapping[
        str, tuple[TargetLocalPosteriorPrediction, ...]
    ]
    sample_influence_predictions_by_control: Mapping[
        str, tuple[InfluencePrediction, ...]
    ]
    donor_utility_rows_by_target: Mapping[str, tuple[DonorUtilityRow, ...]]
    full_models_by_target: Mapping[str, SignedUtilityModel]
    delete_models_by_target: Mapping[str, Mapping[str, SignedUtilityModel]]
    donor_veto_predictions: tuple[UtilityPrediction, ...]
    composed_predictions_by_policy: Mapping[str, tuple[ComposedCasePrediction, ...]]
    decision_barrier: Mapping[str, object]
    aggregate_seal: Mapping[str, object]
    label_firewall: PCSILabelFirewall


def build_preterminal_result(
    surface: PhysicalProbabilitySurface,
    label_loader: LabelLoader,
    *,
    use_processes: bool = True,
) -> PreterminalResult:
    """Freeze all physical, posterior, veto, and route objects before evaluation."""

    protocol = build_frozen_protocol()
    prepared = MappingProxyType(
        {center: prepare_center(surface.centers[center]) for center in CENTERS}
    )
    plans = build_outer_plans(
        _surface_identities(surface),
        probability_surface_hash=surface.surface_hash,
        strict_canonical_topology=surface.strict_canonical_topology,
    )
    firewall = PCSILabelFirewall(plans, label_loader)

    # These surfaces are fully sealed before the first role-scoped label opens.
    primary_fingerprints = MappingProxyType(
        {
            center: build_physical_fingerprint_surface(surface.centers[center])
            for center in CENTERS
        }
    )
    blocked_fingerprints = MappingProxyType(
        {
            center: blocked_within_case_fingerprint(primary_fingerprints[center])
            for center in CENTERS
        }
    )

    source_prior_labels: dict[tuple[str, str], tuple[BinaryLabel, ...]] = {}
    donor_priors: dict[str, Mapping[tuple[str, str], float]] = {}
    for endpoint_target in CENTERS:
        labels_by_source: dict[str, Mapping[str, tuple[BinaryLabel, ...]]] = {}
        for source in candidate_sources(endpoint_target):
            labels = firewall.open_source_prior_labels(endpoint_target, source)
            source_prior_labels[(endpoint_target, source)] = labels
            legal_queries = tuple(
                center for center in CENTERS if center not in {endpoint_target, source}
            )
            labels_by_source[source] = MappingProxyType(
                {
                    center: tuple(row for row in labels if row.center == center)
                    for center in legal_queries
                }
            )
        donor_priors[endpoint_target] = compute_donor_priors(
            prepared,
            labels_by_source,
            heldout_center=endpoint_target,
        )

    crossfit_donor_priors: dict[tuple[str, str], Mapping[tuple[str, str], float]] = {}
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
                        center: tuple(row for row in grant if row.center == center)
                        for center in legal_queries
                    }
                )
            crossfit_donor_priors[(outer, donor)] = compute_donor_priors(
                prepared,
                labels_by_source,
                heldout_center=donor,
                excluded_query_centers=(outer,),
            )

    donor_labels = {
        (outer, donor): firewall.open_crossing_donor_labels(outer, donor)
        for outer in CENTERS
        for donor in CENTERS
        if donor != outer
    }

    jobs, route_support_labels = _build_outer_jobs(
        prepared, plans, donor_priors, firewall
    )
    use_parallel = surface.strict_canonical_topology and use_processes
    endpoint_products = (
        execute_outer_endpoint_jobs(jobs, use_processes=True)
        if use_parallel
        else tuple(compute_outer_endpoint_products(job) for job in jobs)
    )
    endpoint_fit_count = sum(row.endpoint_model_fit_count for row in endpoint_products)
    if (
        surface.strict_canonical_topology
        and endpoint_fit_count != EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT
    ):
        raise ProtocolError("PCSI canonical endpoint workload drifted.")
    for product in endpoint_products:
        for case, digest in product.state_hashes:
            firewall.record_outer_state_seal(product.target_center, case, digest)

    predictions_by_center = MappingProxyType(
        {row.target_center: row.predictions for row in endpoint_products}
    )
    utility_descriptors_by_center = MappingProxyType(
        {
            center: build_utility_descriptor_surface(predictions_by_center[center])
            for center in CENTERS
        }
    )

    posterior_products = execute_target_center_posterior_jobs(
        tuple(
            TargetCenterPosteriorJob(
                center,
                primary_fingerprints[center],
                blocked_fingerprints[center],
                tuple(
                    (
                        case_id,
                        route_support_labels[(center, case_id)],
                    )
                    for case_id in primary_fingerprints[center].cases
                ),
            )
            for center in CENTERS
        ),
        use_processes=use_parallel,
    )
    posterior_fit_count = sum(row.model_fit_count for row in posterior_products)
    if (
        surface.strict_canonical_topology
        and posterior_fit_count != EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT
    ):
        raise ProtocolError("PCSI canonical posterior workload drifted.")
    posterior_models = MappingProxyType(
        {
            PRIMARY_FINGERPRINT_CONTROL_ID: tuple(
                model for row in posterior_products for model in row.primary_models
            ),
            BLOCKED_FINGERPRINT_CONTROL_ID: tuple(
                model for row in posterior_products for model in row.blocked_models
            ),
        }
    )
    posterior_predictions = MappingProxyType(
        {
            PRIMARY_FINGERPRINT_CONTROL_ID: tuple(
                prediction
                for row in posterior_products
                for prediction in row.primary_predictions
            ),
            BLOCKED_FINGERPRINT_CONTROL_ID: tuple(
                prediction
                for row in posterior_products
                for prediction in row.blocked_predictions
            ),
        }
    )

    job_by_center = {job.target_center: job for job in jobs}
    product_by_center = {row.target_center: row for row in endpoint_products}
    donor_endpoint_products = MappingProxyType(
        {
            (outer, donor): recompose_outer_endpoint_products(
                job_by_center[donor],
                product_by_center[donor],
                donor_priors=crossfit_donor_priors[(outer, donor)],
            )
            for outer in CENTERS
            for donor in CENTERS
            if donor != outer
        }
    )
    donor_rows, full_models, delete_models = _fit_outer_utility_models(
        donor_endpoint_products, donor_labels
    )
    donor_veto_predictions = tuple(
        prediction
        for outer in CENTERS
        for prediction in predict_utility_surface(
            utility_descriptors_by_center[outer],
            donor_rows=donor_rows[outer],
            full_model=full_models[outer],
            delete_models=delete_models[outer],
        )
    )

    influence_predictions: dict[str, tuple[InfluencePrediction, ...]] = {}
    for control_id in (
        PRIMARY_FINGERPRINT_CONTROL_ID,
        BLOCKED_FINGERPRINT_CONTROL_ID,
    ):
        models_by_route = {
            (row.target_center, row.held_case_id): row
            for row in posterior_models[control_id]
        }
        predictions_by_route = {
            (row.target_center, row.case_id): row
            for row in posterior_predictions[control_id]
        }
        influence_predictions[control_id] = tuple(
            influence
            for center in CENTERS
            for case_id, descriptors in _group_descriptors_by_case(
                utility_descriptors_by_center[center]
            ).items()
            for influence in score_sample_influences(
                descriptors,
                posterior=predictions_by_route[(center, case_id)],
                model=models_by_route[(center, case_id)],
            )
        )
    influence_predictions_proxy = MappingProxyType(influence_predictions)

    donor_by_hash = {row.descriptor_hash: row for row in donor_veto_predictions}
    influence_by_control = {
        control: {row.descriptor_hash: row for row in rows}
        for control, rows in influence_predictions.items()
    }
    composed_predictions: dict[str, tuple[ComposedCasePrediction, ...]] = {}
    for policy_id in COMPOSED_POLICY_IDS:
        control_id = (
            BLOCKED_FINGERPRINT_CONTROL_ID
            if policy_id == PERMUTATION_METHOD_ID
            else PRIMARY_FINGERPRINT_CONTROL_ID
        )
        compositions: list[ComposedCasePrediction] = []
        for center in CENTERS:
            descriptors_by_case = _group_descriptors_by_case(
                utility_descriptors_by_center[center]
            )
            for endpoint in predictions_by_center[center]:
                descriptors = descriptors_by_case[endpoint.case_id]
                compositions.append(
                    compose_case_probabilities(
                        endpoint,
                        descriptors,
                        tuple(
                            influence_by_control[control_id][row.descriptor_hash]
                            for row in descriptors
                        ),
                        tuple(donor_by_hash[row.descriptor_hash] for row in descriptors),
                        policy_id=policy_id,
                    )
                )
        composed_predictions[policy_id] = tuple(compositions)

    policy_payload = {
        "schema_version": "fixed_bank_pcsi_policy_menu_v1",
        "policy_ids": list(COMPOSED_POLICY_IDS),
        "primary_policy_id": MODEL_BASED_METHOD_ID,
        "score_only_control_id": BACC_ONLY_METHOD_ID,
        "proper_only_control_id": FULL_ONLY_METHOD_ID,
        "blocked_fingerprint_control_id": PERMUTATION_METHOD_ID,
        "protected_fallback": "P_PROTECTED",
        "target_score": "crossfit_target_local_balanced_accuracy_influence",
        "donor_veto": "robust_BACC_positive_and_Brier_LogLoss_nonpositive",
        "selected_from_terminal_labels": False,
    }
    composition_by_key = {
        policy: {(row.target_center, row.case_id): row for row in rows}
        for policy, rows in composed_predictions.items()
    }
    descriptor_hash_by_case = {
        (center, case): canonical_hash(
            [row.descriptor_hash for row in descriptors]
        )
        for center in CENTERS
        for case, descriptors in _group_descriptors_by_case(
            utility_descriptors_by_center[center]
        ).items()
    }
    posterior_model_by_control_route = {
        control: {
            (row.target_center, row.held_case_id): row
            for row in posterior_models[control]
        }
        for control in posterior_models
    }
    influence_hash_by_control_case = {
        control: {
            (center, case): canonical_hash(
                [
                    row.influence_hash
                    for row in rows
                    if row.target_center == center and row.case_id == case
                ]
            )
            for center in CENTERS
            for case in primary_fingerprints[center].cases
        }
        for control, rows in influence_predictions.items()
    }
    for center in CENTERS:
        for endpoint in predictions_by_center[center]:
            key = (center, endpoint.case_id)
            firewall.record_route_decision_seal(
                *key,
                canonical_hash(
                    {
                        "schema_version": "fixed_bank_pcsi_case_route_seal_v1",
                        "endpoint_prediction_hash": endpoint.prediction_hash,
                        "utility_descriptor_hash": descriptor_hash_by_case[key],
                        "target_posterior_model_hashes": {
                            control: posterior_model_by_control_route[control][key].model_hash
                            for control in posterior_models
                        },
                        "sample_influence_hashes": {
                            control: influence_hash_by_control_case[control][key]
                            for control in influence_predictions
                        },
                        "policy_menu": policy_payload,
                        "composition_hashes": {
                            policy: composition_by_key[policy][key].prediction_hash
                            for policy in COMPOSED_POLICY_IDS
                        },
                        "terminal_labels_used": False,
                    }
                ),
            )
    barrier = firewall.decision_barrier_payload()
    aggregate_payload = {
        "schema_version": "fixed_bank_pcsi_preterminal_aggregate_v1",
        "protocol_hash": protocol.protocol_hash,
        "probability_surface_hash": surface.surface_hash,
        "plan_seal_hash": plans.seal_hash,
        "decision_barrier_hash": barrier["decision_barrier_hash"],
        "utility_descriptor_hash": canonical_hash(
            [
                row.descriptor_hash
                for center in CENTERS
                for row in utility_descriptors_by_center[center]
            ]
        ),
        "fingerprint_hash": canonical_hash(
            [
                primary_fingerprints[center].fingerprint_hash
                for center in CENTERS
            ]
            + [
                blocked_fingerprints[center].fingerprint_hash
                for center in CENTERS
            ]
        ),
        "target_posterior_model_hash": canonical_hash(
            [
                row.model_hash
                for control in posterior_models
                for row in posterior_models[control]
            ]
        ),
        "target_posterior_prediction_hash": canonical_hash(
            [
                row.prediction_hash
                for control in posterior_predictions
                for row in posterior_predictions[control]
            ]
        ),
        "sample_influence_prediction_hash": canonical_hash(
            [
                row.influence_hash
                for control in influence_predictions
                for row in influence_predictions[control]
            ]
        ),
        "donor_utility_row_hash": canonical_hash(
            [row.to_payload() for center in CENTERS for row in donor_rows[center]]
        ),
        "donor_utility_model_hash": canonical_hash(
            [
                model.model_hash
                for center in CENTERS
                for model in (full_models[center], *delete_models[center].values())
            ]
        ),
        "donor_veto_prediction_hash": canonical_hash(
            [row.prediction_hash for row in donor_veto_predictions]
        ),
        "composition_hash": canonical_hash(
            [
                row.prediction_hash
                for policy in COMPOSED_POLICY_IDS
                for row in composed_predictions[policy]
            ]
        ),
        "policy_menu_hash": canonical_hash(policy_payload),
        "terminal_labels_used": False,
    }
    aggregate = {
        **aggregate_payload,
        "aggregate_seal_hash": canonical_hash(aggregate_payload),
    }
    firewall.record_aggregate_seal(aggregate)
    if _utility_model_fit_count(full_models, delete_models) != EXPECTED_UTILITY_MODEL_FIT_COUNT:
        raise ProtocolError("PCSI donor veto model workload drifted.")
    return PreterminalResult(
        protocol=protocol,
        surface=surface,
        plans=plans,
        endpoint_products=endpoint_products,
        donor_endpoint_products=donor_endpoint_products,
        predictions_by_center=predictions_by_center,
        utility_descriptors_by_center=utility_descriptors_by_center,
        primary_fingerprints_by_center=primary_fingerprints,
        blocked_fingerprints_by_center=blocked_fingerprints,
        target_posterior_models_by_control=posterior_models,
        target_posterior_predictions_by_control=posterior_predictions,
        sample_influence_predictions_by_control=influence_predictions_proxy,
        donor_utility_rows_by_target=donor_rows,
        full_models_by_target=full_models,
        delete_models_by_target=delete_models,
        donor_veto_predictions=donor_veto_predictions,
        composed_predictions_by_policy=MappingProxyType(composed_predictions),
        decision_barrier=MappingProxyType(dict(barrier)),
        aggregate_seal=MappingProxyType(aggregate),
        label_firewall=firewall,
    )


def _surface_identities(surface: PhysicalProbabilitySurface) -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(center=center, case_id=case_id, sample_id=sample_id, group_id=case_id)
        for center in CENTERS
        for sample_id, case_id in zip(
            surface.centers[center].sample_ids,
            surface.centers[center].case_ids,
            strict=True,
        )
    )


def _build_outer_jobs(
    prepared: Mapping[str, PreparedCenter],
    plans: OuterPlanSeal,
    donor_priors: Mapping[str, Mapping[tuple[str, str], float]],
    firewall: PCSILabelFirewall,
) -> tuple[
    tuple[OuterEndpointJob, ...],
    Mapping[tuple[str, str], tuple[BinaryLabel, ...]],
]:
    jobs: list[OuterEndpointJob] = []
    support_by_route: dict[tuple[str, str], tuple[BinaryLabel, ...]] = {}
    for center in CENTERS:
        outer_plans = tuple(row for row in plans.outer_plans if row.target_center == center)
        observed: dict[tuple[str, str, str], int] = {}
        for plan in outer_plans:
            labels = firewall.open_outer_support_labels(
                center, plan.case_id, plan_hash=plan.plan_hash
            )
            support_by_route[(center, plan.case_id)] = labels
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
            raise ProtocolError("PCSI support capability union does not cover one center.")
        scope = f"derived_support_sufficient_stats::H={center}"
        sufficient_stat_labels = tuple(
            BinaryLabel(
                center,
                case_id,
                sample_id,
                observed[(center, case_id, sample_id)],
                scope,
            )
            for sample_id, case_id in zip(
                prepared[center].surface.sample_ids,
                prepared[center].surface.case_ids,
                strict=True,
            )
        )
        jobs.append(
            OuterEndpointJob(
                center,
                prepared[center],
                build_center_case_outcomes(prepared[center], sufficient_stat_labels),
                outer_plans,
                tuple(donor_priors[center].items()),
            )
        )
    return tuple(jobs), MappingProxyType(support_by_route)


def _merge_consistent_labels(
    observed: dict[tuple[str, str, str], int], labels: Sequence[BinaryLabel]
) -> None:
    for row in labels:
        previous = observed.setdefault(row.key, row.value)
        if previous != row.value:
            raise ProtocolError("PCSI repeated support capabilities disagree.")


def _fit_outer_utility_models(
    donor_products: Mapping[tuple[str, str], OuterEndpointProducts],
    donor_labels: Mapping[tuple[str, str], Sequence[BinaryLabel]],
) -> tuple[
    Mapping[str, tuple[DonorUtilityRow, ...]],
    Mapping[str, SignedUtilityModel],
    Mapping[str, Mapping[str, SignedUtilityModel]],
]:
    donor_rows: dict[str, tuple[DonorUtilityRow, ...]] = {}
    full_models: dict[str, SignedUtilityModel] = {}
    delete_models: dict[str, Mapping[str, SignedUtilityModel]] = {}
    for outer in CENTERS:
        rows: list[DonorUtilityRow] = []
        for donor in CENTERS:
            if donor == outer:
                continue
            labels = tuple(donor_labels[(outer, donor)])
            by_case = {
                case: tuple(row for row in labels if row.case_id == case)
                for case in dict.fromkeys(row.case_id for row in labels)
            }
            n_positive = sum(row.value == 1 for row in labels)
            n_negative = sum(row.value == 0 for row in labels)
            products = donor_products[(outer, donor)]
            descriptors = build_utility_descriptor_surface(products.predictions)
            descriptors_by_case = _group_descriptors_by_case(descriptors)
            if set(by_case) != {row.case_id for row in products.predictions}:
                raise ProtocolError("PCSI donor cases do not align with endpoints.")
            for prediction in products.predictions:
                rows.extend(
                    build_donor_utility_rows(
                        outer_target_center=outer,
                        prediction=prediction,
                        descriptors=descriptors_by_case[prediction.case_id],
                        case_labels=by_case[prediction.case_id],
                        center_n_positive=n_positive,
                        center_n_negative=n_negative,
                    )
                )
        donor_rows[outer] = tuple(sorted(rows, key=lambda row: row.key))
        full, deleted = fit_response_model_family(
            donor_rows[outer], outer_target_center=outer
        )
        full_models[outer], delete_models[outer] = full, deleted
    return (
        MappingProxyType(donor_rows),
        MappingProxyType(full_models),
        MappingProxyType(delete_models),
    )


def _group_descriptors_by_case(
    descriptors: Sequence[UtilityDescriptor],
) -> Mapping[str, tuple[UtilityDescriptor, ...]]:
    cases = dict.fromkeys(row.case_id for row in descriptors)
    return MappingProxyType(
        {case: tuple(row for row in descriptors if row.case_id == case) for case in cases}
    )


def _utility_model_fit_count(
    full: Mapping[str, SignedUtilityModel],
    deleted: Mapping[str, Mapping[str, SignedUtilityModel]],
) -> int:
    return sum(
        1 + len(deleted[center])
        for center in CENTERS
        if center in full
    )


__all__ = ("LabelLoader", "PreterminalResult", "build_preterminal_result")
