"""Phase-separated preterminal engine for posterior-utility margin routing."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Callable, Mapping, Sequence

from ...protocol import ProtocolError
from .composition import compose_case_probabilities
from .constants import (
    BLOCKED_FINGERPRINT_CONTROL_ID,
    CENTERS,
    COMPOSED_POLICY_IDS,
    EXPECTED_MARGIN_CALIBRATION_COUNT,
    EXPECTED_INNER_DONOR_REPLAY_COUNT,
    EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
    FULL_ONLY_METHOD_ID,
    MODEL_BASED_METHOD_ID,
    PERMUTATION_METHOD_ID,
    PRIMARY_FINGERPRINT_CONTROL_ID,
    candidate_sources,
)
from .contracts import BinaryLabel, EndpointCasePrediction, PhysicalProbabilitySurface
from .donor_calibration_runtime import build_donor_calibrations
from .endpoint_reconstruction import (
    PreparedCenter,
    build_center_case_outcomes,
    compute_donor_priors,
    prepare_center,
)
from .hashing import canonical_hash
from .label_capabilities import PUMRLabelFirewall
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
from .posterior_contracts import (
    CasePosteriorPrediction,
    PhysicalFingerprintSurface,
    RoutePosteriorEnsemble,
    TargetLocalPosteriorModel,
)
from .posterior_utility import score_posterior_utilities
from .protocol import FrozenProtocol, build_frozen_protocol
from .target_local_runtime import (
    TargetCenterPosteriorJob,
    execute_target_center_posterior_jobs,
)
from .utility_contracts import (
    ComposedCasePrediction,
    DonorUtilityRow,
    MarginCalibration,
    PosteriorUtilityPrediction,
    UtilityDescriptor,
)
from .utility_features import build_utility_descriptor_surface


LabelLoader = Callable[[frozenset[tuple[str, str, str]], str], Sequence[object]]
CONTROL_IDS = (
    PRIMARY_FINGERPRINT_CONTROL_ID,
    BLOCKED_FINGERPRINT_CONTROL_ID,
)


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
        str, tuple[CasePosteriorPrediction, ...]
    ]
    route_posterior_ensembles_by_control: Mapping[
        str, tuple[RoutePosteriorEnsemble, ...]
    ]
    posterior_utility_predictions_by_control: Mapping[
        str, tuple[PosteriorUtilityPrediction, ...]
    ]
    donor_utility_rows_by_target: Mapping[str, tuple[DonorUtilityRow, ...]]
    donor_posterior_utilities_by_target_control: Mapping[
        tuple[str, str], tuple[PosteriorUtilityPrediction, ...]
    ]
    margin_calibrations: Mapping[tuple[str, str], MarginCalibration]
    composed_predictions_by_policy: Mapping[str, tuple[ComposedCasePrediction, ...]]
    decision_barrier: Mapping[str, object]
    aggregate_seal: Mapping[str, object]
    label_firewall: PUMRLabelFirewall


def build_preterminal_result(
    surface: PhysicalProbabilitySurface,
    label_loader: LabelLoader,
    *,
    use_processes: bool = True,
) -> PreterminalResult:
    """Freeze all routes and compositions before terminal labels can open."""

    protocol = build_frozen_protocol()
    prepared = MappingProxyType(
        {center: prepare_center(surface.centers[center]) for center in CENTERS}
    )
    plans = build_outer_plans(
        _surface_identities(surface),
        probability_surface_hash=surface.surface_hash,
        strict_canonical_topology=surface.strict_canonical_topology,
    )
    firewall = PUMRLabelFirewall(plans, label_loader)
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
            prepared, labels_by_source, heldout_center=endpoint_target
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
        raise ProtocolError("PUMR canonical endpoint workload drifted.")
    for product in endpoint_products:
        for case, digest in product.state_hashes:
            firewall.record_outer_state_seal(product.target_center, case, digest)
    predictions_by_center = MappingProxyType(
        {row.target_center: row.predictions for row in endpoint_products}
    )
    descriptors_by_center = MappingProxyType(
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
                    (case_id, route_support_labels[(center, case_id)])
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
        raise ProtocolError("PUMR canonical posterior workload drifted.")
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
    posterior_ensembles = MappingProxyType(
        {
            PRIMARY_FINGERPRINT_CONTROL_ID: tuple(
                ensemble
                for row in posterior_products
                for ensemble in row.primary_ensembles
            ),
            BLOCKED_FINGERPRINT_CONTROL_ID: tuple(
                ensemble
                for row in posterior_products
                for ensemble in row.blocked_ensembles
            ),
        }
    )
    ensemble_index = {
        control: {
            (row.target_center, row.held_case_id): row
            for row in posterior_ensembles[control]
        }
        for control in CONTROL_IDS
    }

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
    donor_rows, donor_utilities, margin_calibrations = build_donor_calibrations(
        donor_endpoint_products,
        donor_labels,
        ensemble_index,
        control_ids=CONTROL_IDS,
    )
    if len(margin_calibrations) != EXPECTED_MARGIN_CALIBRATION_COUNT:
        raise ProtocolError("PUMR margin-calibration workload drifted.")
    if (
        sum(len(row.inner_replays) for row in margin_calibrations.values())
        != EXPECTED_INNER_DONOR_REPLAY_COUNT
    ):
        raise ProtocolError("PUMR inner donor replay workload drifted.")

    target_utilities: dict[str, tuple[PosteriorUtilityPrediction, ...]] = {}
    for control in CONTROL_IDS:
        target_utilities[control] = tuple(
            utility
            for center in CENTERS
            for endpoint in predictions_by_center[center]
            for utility in score_posterior_utilities(
                endpoint,
                _group_descriptors_by_case(descriptors_by_center[center])[
                    endpoint.case_id
                ],
                ensemble_index[control][(center, endpoint.case_id)],
            )
        )
    target_utilities_proxy = MappingProxyType(target_utilities)

    target_utility_index = {
        control: {row.descriptor_hash: row for row in target_utilities[control]}
        for control in CONTROL_IDS
    }
    composed: dict[str, tuple[ComposedCasePrediction, ...]] = {}
    for policy_id in COMPOSED_POLICY_IDS:
        control = (
            BLOCKED_FINGERPRINT_CONTROL_ID
            if policy_id == PERMUTATION_METHOD_ID
            else PRIMARY_FINGERPRINT_CONTROL_ID
        )
        rows: list[ComposedCasePrediction] = []
        for center in CENTERS:
            grouped = _group_descriptors_by_case(descriptors_by_center[center])
            calibration = margin_calibrations[(center, control)]
            for endpoint in predictions_by_center[center]:
                descriptors = grouped[endpoint.case_id]
                rows.append(
                    compose_case_probabilities(
                        endpoint,
                        descriptors,
                        tuple(
                            target_utility_index[control][row.descriptor_hash]
                            for row in descriptors
                        ),
                        calibration,
                        policy_id=policy_id,
                    )
                )
        composed[policy_id] = tuple(rows)

    policy_payload = {
        "schema_version": "fixed_bank_pumr_policy_menu_v1",
        "policy_ids": list(COMPOSED_POLICY_IDS),
        "primary_policy_id": MODEL_BASED_METHOD_ID,
        "bacc_only_control_id": "PUMR_BACC_ONLY",
        "zero_margin_control_id": FULL_ONLY_METHOD_ID,
        "blocked_fingerprint_control_id": PERMUTATION_METHOD_ID,
        "protected_fallback": "P_PROTECTED",
        "score": "five_fold_posterior_expected_BACC_lower",
        "proper_safety": "five_fold_posterior_Brier_and_LogLoss_upper_nonpositive",
        "margin": "nested_leave_one_donor_scalar_abstention_margin",
        "selected_from_terminal_labels": False,
    }
    composition_index = {
        policy: {(row.target_center, row.case_id): row for row in rows}
        for policy, rows in composed.items()
    }
    descriptors_by_case_hash = {
        (center, case): canonical_hash([row.descriptor_hash for row in rows])
        for center in CENTERS
        for case, rows in _group_descriptors_by_case(
            descriptors_by_center[center]
        ).items()
    }
    target_utility_case_hash = {
        control: {
            (center, case): canonical_hash(
                [
                    row.utility_hash
                    for row in target_utilities[control]
                    if row.target_center == center and row.case_id == case
                ]
            )
            for center in CENTERS
            for case in primary_fingerprints[center].cases
        }
        for control in CONTROL_IDS
    }
    for center in CENTERS:
        for endpoint in predictions_by_center[center]:
            key = (center, endpoint.case_id)
            firewall.record_route_decision_seal(
                *key,
                canonical_hash(
                    {
                        "schema_version": "fixed_bank_pumr_case_route_seal_v1",
                        "endpoint_prediction_hash": endpoint.prediction_hash,
                        "utility_descriptor_hash": descriptors_by_case_hash[key],
                        "route_posterior_ensemble_hashes": {
                            control: ensemble_index[control][key].ensemble_hash
                            for control in CONTROL_IDS
                        },
                        "posterior_utility_hashes": {
                            control: target_utility_case_hash[control][key]
                            for control in CONTROL_IDS
                        },
                        "margin_calibration_hashes": {
                            control: margin_calibrations[(center, control)].calibration_hash
                            for control in CONTROL_IDS
                        },
                        "policy_menu": policy_payload,
                        "composition_hashes": {
                            policy: composition_index[policy][key].prediction_hash
                            for policy in COMPOSED_POLICY_IDS
                        },
                        "terminal_labels_used": False,
                    }
                ),
            )
    barrier = firewall.decision_barrier_payload()
    aggregate_payload = {
        "schema_version": "fixed_bank_pumr_preterminal_aggregate_v1",
        "protocol_hash": protocol.protocol_hash,
        "probability_surface_hash": surface.surface_hash,
        "plan_seal_hash": plans.seal_hash,
        "decision_barrier_hash": barrier["decision_barrier_hash"],
        "utility_descriptor_hash": canonical_hash(
            [row.descriptor_hash for center in CENTERS for row in descriptors_by_center[center]]
        ),
        "fingerprint_hash": canonical_hash(
            [primary_fingerprints[center].fingerprint_hash for center in CENTERS]
            + [blocked_fingerprints[center].fingerprint_hash for center in CENTERS]
        ),
        "target_posterior_model_hash": canonical_hash(
            [row.model_hash for control in CONTROL_IDS for row in posterior_models[control]]
        ),
        "target_posterior_prediction_hash": canonical_hash(
            [
                row.prediction_hash
                for control in CONTROL_IDS
                for row in posterior_predictions[control]
            ]
        ),
        "route_posterior_ensemble_hash": canonical_hash(
            [row.ensemble_hash for control in CONTROL_IDS for row in posterior_ensembles[control]]
        ),
        "posterior_utility_prediction_hash": canonical_hash(
            [row.utility_hash for control in CONTROL_IDS for row in target_utilities[control]]
        ),
        "donor_utility_row_hash": canonical_hash(
            [row.to_payload() for center in CENTERS for row in donor_rows[center]]
        ),
        "donor_posterior_utility_hash": canonical_hash(
            [
                row.utility_hash
                for key in sorted(donor_utilities)
                for row in donor_utilities[key]
            ]
        ),
        "margin_calibration_hash": canonical_hash(
            [margin_calibrations[key].calibration_hash for key in sorted(margin_calibrations)]
        ),
        "composition_hash": canonical_hash(
            [
                row.prediction_hash
                for policy in COMPOSED_POLICY_IDS
                for row in composed[policy]
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
    return PreterminalResult(
        protocol,
        surface,
        plans,
        endpoint_products,
        donor_endpoint_products,
        predictions_by_center,
        descriptors_by_center,
        primary_fingerprints,
        blocked_fingerprints,
        posterior_models,
        posterior_predictions,
        posterior_ensembles,
        target_utilities_proxy,
        donor_rows,
        donor_utilities,
        margin_calibrations,
        MappingProxyType(composed),
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
    firewall: PUMRLabelFirewall,
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
            raise ProtocolError("PUMR support capability union does not cover one center.")
        sufficient_stat_labels = tuple(
            BinaryLabel(
                center,
                case_id,
                sample_id,
                observed[(center, case_id, sample_id)],
                f"derived_support_sufficient_stats::H={center}",
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
            raise ProtocolError("PUMR repeated support capabilities disagree.")


def _group_descriptors_by_case(
    descriptors: Sequence[UtilityDescriptor],
) -> Mapping[str, tuple[UtilityDescriptor, ...]]:
    cases = dict.fromkeys(row.case_id for row in descriptors)
    return MappingProxyType(
        {case: tuple(row for row in descriptors if row.case_id == case) for case in cases}
    )


__all__ = ("LabelLoader", "PreterminalResult", "build_preterminal_result")
