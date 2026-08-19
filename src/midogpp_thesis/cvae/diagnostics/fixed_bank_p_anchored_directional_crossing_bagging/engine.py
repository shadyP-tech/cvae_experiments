"""Phase-separated, preterminal PDCB scientific engine."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Callable, Mapping, Sequence

from ...protocol import ProtocolError
from .composition import compose_case_probabilities
from .constants import (
    CENTERS,
    COMPOSED_POLICY_IDS,
    EXPECTED_CROSSING_MODEL_FIT_COUNT,
    EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
    FULL_ONLY_METHOD_ID,
    MODEL_BASED_METHOD_ID,
    PERMUTATION_METHOD_ID,
    candidate_sources,
)
from .contracts import BinaryLabel, EndpointCasePrediction, PhysicalProbabilitySurface
from .crossing_contracts import (
    ComposedCasePrediction,
    CrossingDescriptor,
    CrossingHelpfulnessModel,
    CrossingPrediction,
    DonorCrossingRow,
)
from .crossing_features import build_crossing_descriptor_surface
from .crossing_model import fit_full_and_delete_donor_models
from .crossing_responses import blocked_feature_permutation, build_donor_crossing_rows
from .donor_center_bagging import predict_crossing_surface
from .endpoint_reconstruction import PreparedCenter, build_center_case_outcomes, compute_donor_priors, prepare_center
from .hashing import canonical_hash
from .label_capabilities import PDCBLabelFirewall
from .outer_endpoint_runtime import (
    OuterEndpointJob,
    OuterEndpointProducts,
    compute_outer_endpoint_products,
    execute_outer_endpoint_jobs,
    recompose_outer_endpoint_products,
)
from .outer_plans import OuterPlanSeal, build_outer_plans
from .protocol import FrozenProtocol, build_frozen_protocol


LabelLoader = Callable[[frozenset[tuple[str, str, str]], str], Sequence[object]]


@dataclass(frozen=True)
class PreterminalResult:
    protocol: FrozenProtocol
    surface: PhysicalProbabilitySurface
    plans: OuterPlanSeal
    endpoint_products: tuple[OuterEndpointProducts, ...]
    donor_endpoint_products: Mapping[tuple[str, str], OuterEndpointProducts]
    predictions_by_center: Mapping[str, tuple[EndpointCasePrediction, ...]]
    crossing_descriptors_by_center: Mapping[str, tuple[CrossingDescriptor, ...]]
    donor_crossing_rows_by_target: Mapping[str, tuple[DonorCrossingRow, ...]]
    full_models_by_target: Mapping[str, CrossingHelpfulnessModel]
    delete_models_by_target: Mapping[str, Mapping[str, CrossingHelpfulnessModel]]
    permutation_full_models_by_target: Mapping[str, CrossingHelpfulnessModel]
    permutation_delete_models_by_target: Mapping[str, Mapping[str, CrossingHelpfulnessModel]]
    crossing_predictions_by_policy: Mapping[str, tuple[CrossingPrediction, ...]]
    composed_predictions_by_policy: Mapping[str, tuple[ComposedCasePrediction, ...]]
    decision_barrier: Mapping[str, object]
    aggregate_seal: Mapping[str, object]
    label_firewall: PDCBLabelFirewall


def build_preterminal_result(
    surface: PhysicalProbabilitySurface,
    label_loader: LabelLoader,
    *,
    use_processes: bool = True,
) -> PreterminalResult:
    """Fit donor models and freeze every route before target labels open."""

    protocol = build_frozen_protocol()
    prepared = MappingProxyType(
        {center: prepare_center(surface.centers[center]) for center in CENTERS}
    )
    plans = build_outer_plans(
        _surface_identities(surface),
        probability_surface_hash=surface.surface_hash,
        strict_canonical_topology=surface.strict_canonical_topology,
    )
    firewall = PDCBLabelFirewall(plans, label_loader)

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

    donor_labels: dict[tuple[str, str], tuple[BinaryLabel, ...]] = {}
    for outer in CENTERS:
        for donor in CENTERS:
            if donor != outer:
                donor_labels[(outer, donor)] = firewall.open_crossing_donor_labels(
                    outer,
                    donor,
                )

    jobs = _build_outer_jobs(prepared, plans, donor_priors, firewall)
    endpoint_products = (
        execute_outer_endpoint_jobs(jobs, use_processes=True)
        if surface.strict_canonical_topology and use_processes
        else tuple(compute_outer_endpoint_products(job) for job in jobs)
    )
    if surface.strict_canonical_topology and sum(
        row.endpoint_model_fit_count for row in endpoint_products
    ) != EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT:
        raise ProtocolError("PDCB canonical endpoint workload drifted.")
    for product in endpoint_products:
        for case, digest in product.state_hashes:
            firewall.record_outer_state_seal(product.target_center, case, digest)

    predictions_by_center = MappingProxyType(
        {row.target_center: row.predictions for row in endpoint_products}
    )
    crossing_descriptors_by_center = MappingProxyType(
        {
            center: build_crossing_descriptor_surface(predictions_by_center[center])
            for center in CENTERS
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
    (
        donor_rows,
        full_models,
        delete_models,
        permutation_full_models,
        permutation_delete_models,
    ) = _fit_outer_crossing_models(donor_endpoint_products, donor_labels)

    crossing_predictions: dict[str, tuple[CrossingPrediction, ...]] = {}
    composed_predictions: dict[str, tuple[ComposedCasePrediction, ...]] = {}
    for policy_id in COMPOSED_POLICY_IDS:
        predictions: list[CrossingPrediction] = []
        compositions: list[ComposedCasePrediction] = []
        for outer in CENTERS:
            descriptors = crossing_descriptors_by_center[outer]
            if policy_id == PERMUTATION_METHOD_ID:
                full_model = permutation_full_models[outer]
                deleted = permutation_delete_models[outer]
            else:
                full_model = full_models[outer]
                deleted = delete_models[outer]
            outer_predictions = predict_crossing_surface(
                descriptors,
                full_model=full_model,
                delete_models=deleted,
            )
            predictions.extend(outer_predictions)
            crossing_by_hash = {row.descriptor_hash: row for row in outer_predictions}
            descriptor_by_case = _group_descriptors_by_case(descriptors)
            for endpoint in predictions_by_center[outer]:
                case_descriptors = descriptor_by_case.get(endpoint.case_id, ())
                compositions.append(
                    compose_case_probabilities(
                        endpoint,
                        case_descriptors,
                        tuple(crossing_by_hash[row.descriptor_hash] for row in case_descriptors),
                        policy_id=policy_id,
                    )
                )
        crossing_predictions[policy_id] = tuple(predictions)
        composed_predictions[policy_id] = tuple(compositions)

    policy_payload = {
        "schema_version": "fixed_bank_pdcb_policy_menu_v1",
        "policy_ids": list(COMPOSED_POLICY_IDS),
        "primary_policy_id": MODEL_BASED_METHOD_ID,
        "full_only_control_id": FULL_ONLY_METHOD_ID,
        "blocked_feature_control_id": PERMUTATION_METHOD_ID,
        "protected_fallback": "P_PROTECTED",
        "selected_from_terminal_labels": False,
    }
    composition_by_key = {
        policy: {(row.target_center, row.case_id): row for row in rows}
        for policy, rows in composed_predictions.items()
    }
    descriptor_hash_by_case = {
        (center, case): canonical_hash(
            [
                row.descriptor_hash
                for row in crossing_descriptors_by_center[center]
                if row.case_id == case
            ]
        )
        for center in CENTERS
        for case in {row.case_id for row in predictions_by_center[center]}
    }
    for center in CENTERS:
        for endpoint in predictions_by_center[center]:
            key = (center, endpoint.case_id)
            firewall.record_route_decision_seal(
                *key,
                canonical_hash(
                    {
                        "schema_version": "fixed_bank_pdcb_case_route_seal_v1",
                        "endpoint_prediction_hash": endpoint.prediction_hash,
                        "crossing_descriptor_hash": descriptor_hash_by_case[key],
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
        "schema_version": "fixed_bank_pdcb_preterminal_aggregate_v1",
        "protocol_hash": protocol.protocol_hash,
        "probability_surface_hash": surface.surface_hash,
        "plan_seal_hash": plans.seal_hash,
        "decision_barrier_hash": barrier["decision_barrier_hash"],
        "crossing_descriptor_hash": canonical_hash(
            [
                row.descriptor_hash
                for center in CENTERS
                for row in crossing_descriptors_by_center[center]
            ]
        ),
        "donor_crossing_row_hash": canonical_hash(
            [row.to_payload() for center in CENTERS for row in donor_rows[center]]
        ),
        "crossing_model_hash": canonical_hash(
            [
                model.model_hash
                for center in CENTERS
                for model in (
                    full_models[center],
                    *delete_models[center].values(),
                    permutation_full_models[center],
                    *permutation_delete_models[center].values(),
                )
            ]
        ),
        "crossing_prediction_hash": canonical_hash(
            [
                row.prediction_hash
                for policy in COMPOSED_POLICY_IDS
                for row in crossing_predictions[policy]
            ]
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
    aggregate = {**aggregate_payload, "aggregate_seal_hash": canonical_hash(aggregate_payload)}
    firewall.record_aggregate_seal(aggregate)
    if _crossing_model_fit_count(full_models, delete_models, permutation_full_models, permutation_delete_models) != EXPECTED_CROSSING_MODEL_FIT_COUNT:
        raise ProtocolError("PDCB crossing model workload drifted.")
    return PreterminalResult(
        protocol,
        surface,
        plans,
        endpoint_products,
        donor_endpoint_products,
        predictions_by_center,
        crossing_descriptors_by_center,
        donor_rows,
        full_models,
        delete_models,
        permutation_full_models,
        permutation_delete_models,
        MappingProxyType(crossing_predictions),
        MappingProxyType(composed_predictions),
        MappingProxyType(dict(barrier)),
        MappingProxyType(aggregate),
        firewall,
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
    firewall: PDCBLabelFirewall,
) -> tuple[OuterEndpointJob, ...]:
    jobs: list[OuterEndpointJob] = []
    for center in CENTERS:
        outer_plans = tuple(row for row in plans.outer_plans if row.target_center == center)
        observed: dict[tuple[str, str, str], int] = {}
        for plan in outer_plans:
            _merge_consistent_labels(
                observed,
                firewall.open_outer_support_labels(center, plan.case_id, plan_hash=plan.plan_hash),
            )
        expected = {
            (center, case_id, sample_id)
            for sample_id, case_id in zip(
                prepared[center].surface.sample_ids,
                prepared[center].surface.case_ids,
                strict=True,
            )
        }
        if set(observed) != expected:
            raise ProtocolError("PDCB support capability union does not cover one center.")
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
        outcomes = build_center_case_outcomes(
            prepared[center],
            sufficient_stat_labels,
        )
        jobs.append(
            OuterEndpointJob(
                center,
                prepared[center],
                outcomes,
                outer_plans,
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
            raise ProtocolError("PDCB repeated support capabilities disagree.")


def _fit_outer_crossing_models(
    donor_products: Mapping[tuple[str, str], OuterEndpointProducts],
    donor_labels: Mapping[tuple[str, str], Sequence[BinaryLabel]],
) -> tuple[
    Mapping[str, tuple[DonorCrossingRow, ...]],
    Mapping[str, CrossingHelpfulnessModel],
    Mapping[str, Mapping[str, CrossingHelpfulnessModel]],
    Mapping[str, CrossingHelpfulnessModel],
    Mapping[str, Mapping[str, CrossingHelpfulnessModel]],
]:
    donor_rows: dict[str, tuple[DonorCrossingRow, ...]] = {}
    full_models: dict[str, CrossingHelpfulnessModel] = {}
    delete_models: dict[str, Mapping[str, CrossingHelpfulnessModel]] = {}
    permutation_full: dict[str, CrossingHelpfulnessModel] = {}
    permutation_delete: dict[str, Mapping[str, CrossingHelpfulnessModel]] = {}
    for outer in CENTERS:
        rows: list[DonorCrossingRow] = []
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
            descriptors = build_crossing_descriptor_surface(products.predictions)
            descriptors_by_case = _group_descriptors_by_case(descriptors)
            if set(by_case) != {row.case_id for row in products.predictions}:
                raise ProtocolError("PDCB donor cases do not align with endpoints.")
            for prediction in products.predictions:
                rows.extend(
                    build_donor_crossing_rows(
                        outer_target_center=outer,
                        prediction=prediction,
                        descriptors=descriptors_by_case.get(prediction.case_id, ()),
                        case_labels=by_case[prediction.case_id],
                        center_n_positive=n_positive,
                        center_n_negative=n_negative,
                    )
                )
        donor_rows[outer] = tuple(sorted(rows, key=lambda row: row.key))
        full, deleted = fit_full_and_delete_donor_models(donor_rows[outer], outer_target_center=outer)
        permuted = blocked_feature_permutation(donor_rows[outer])
        perm_full, perm_deleted = fit_full_and_delete_donor_models(permuted, outer_target_center=outer)
        full_models[outer], delete_models[outer] = full, deleted
        permutation_full[outer], permutation_delete[outer] = perm_full, perm_deleted
    return (
        MappingProxyType(donor_rows),
        MappingProxyType(full_models),
        MappingProxyType(delete_models),
        MappingProxyType(permutation_full),
        MappingProxyType(permutation_delete),
    )


def _group_descriptors_by_case(
    descriptors: Sequence[CrossingDescriptor],
) -> Mapping[str, tuple[CrossingDescriptor, ...]]:
    cases = dict.fromkeys(row.case_id for row in descriptors)
    return MappingProxyType(
        {case: tuple(row for row in descriptors if row.case_id == case) for case in cases}
    )


def _crossing_model_fit_count(*groups: Mapping[str, object]) -> int:
    total = 0
    for group in groups:
        for value in group.values():
            total += len(value) if isinstance(value, Mapping) else 1
    return total


__all__ = ("LabelLoader", "PreterminalResult", "build_preterminal_result")
