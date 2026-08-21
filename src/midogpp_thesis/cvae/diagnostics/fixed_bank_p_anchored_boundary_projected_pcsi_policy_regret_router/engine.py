"""Thin phase orchestrator for the one-shot PCSI-PARC diagnostic."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace

from ...protocol import ProtocolError
from .constants import (
    BLOCKED_FINGERPRINT_CONTROL_ID,
    CENTERS,
    COMPOSED_POLICY_IDS,
    EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
    EXPECTED_POLICY_REPLAY_COUNT,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    EXPECTED_UTILITY_MODEL_FIT_COUNT,
    PRIMARY_FINGERPRINT_CONTROL_ID,
    candidate_sources,
)
from .contracts import (
    BinaryLabel,
    EndpointCasePrediction,
    PhysicalProbabilitySurface,
)
from .donor_runtime import (
    DonorRuntimeResult,
    DoubleExcludedDonorPriorProvenance,
    build_donor_runtime,
)
from .endpoint_reconstruction import (
    PreparedCenter,
    build_center_case_outcomes,
    compute_donor_priors,
    prepare_center,
)
from .hashing import canonical_hash
from .label_capabilities import PCSIPARCLabelFirewall
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
from .policy_replay_runtime import (
    PolicyReplayRuntimeResult,
    build_policy_replay_runtime,
)
from .protocol import FrozenProtocol, build_frozen_protocol
from .sample_influence_contracts import (
    PhysicalFingerprintSurface,
    TargetLocalPosteriorModel,
    TargetLocalPosteriorPrediction,
)
from .target_local_runtime import (
    TargetCenterPosteriorJob,
    compute_target_center_posteriors,
    execute_target_center_posterior_jobs,
)
from .transport import (
    LEGACY_TRANSPORT_PROTOCOL_FIELDS,
    TRANSPORT_PROTOCOL_CONTRACT,
)


LabelLoader = Callable[
    [frozenset[tuple[str, str, str]], str], Sequence[object]
]
PhaseObserver = Callable[[str, Mapping[str, int]], None]


@dataclass(frozen=True)
class PreterminalResult:
    protocol: FrozenProtocol
    surface: PhysicalProbabilitySurface
    plans: OuterPlanSeal
    endpoint_products: tuple[OuterEndpointProducts, ...]
    donor_endpoint_products: Mapping[tuple[str, str], OuterEndpointProducts]
    predictions_by_center: Mapping[str, tuple[EndpointCasePrediction, ...]]
    primary_fingerprints_by_center: Mapping[str, PhysicalFingerprintSurface]
    blocked_fingerprints_by_center: Mapping[str, PhysicalFingerprintSurface]
    target_posterior_models_by_control: Mapping[
        str, tuple[TargetLocalPosteriorModel, ...]
    ]
    target_posterior_predictions_by_control: Mapping[
        str, tuple[TargetLocalPosteriorPrediction, ...]
    ]
    donor_runtime: DonorRuntimeResult
    policy_runtime: PolicyReplayRuntimeResult
    decision_barrier: Mapping[str, object]
    aggregate_seal: Mapping[str, object]
    label_firewall: PCSIPARCLabelFirewall


def build_preterminal_result(
    surface: PhysicalProbabilitySurface,
    label_loader: LabelLoader,
    *,
    use_processes: bool = True,
    phase_observer: PhaseObserver | None = None,
) -> PreterminalResult:
    """Build and seal every decision before the global terminal-label grant."""

    strict = surface.strict_canonical_topology
    use_parallel = strict and use_processes
    protocol = build_frozen_protocol()
    assert_transport_contract_executable(
        protocol,
        strict_canonical_topology=strict,
    )
    prepared = MappingProxyType(
        {center: prepare_center(surface.centers[center]) for center in CENTERS}
    )
    plans = build_outer_plans(
        _surface_identities(surface),
        probability_surface_hash=surface.surface_hash,
        strict_canonical_topology=strict,
    )
    firewall = PCSIPARCLabelFirewall(plans, label_loader)

    _emit_phase(
        phase_observer,
        "physical_started",
        {"physical_cell_count": len(CENTERS) * 10 * 9},
    )
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
    _emit_phase(
        phase_observer,
        "physical_completed",
        {"physical_cell_count": len(CENTERS) * 10 * 9},
    )

    (
        source_prior_labels,
        donor_priors,
        crossfit_donor_priors,
        pseudo_donor_prior_provenance,
    ) = _open_prior_roles(prepared, firewall)
    del source_prior_labels
    donor_labels = MappingProxyType(
        {
            (outer, donor): firewall.open_utility_donor_labels(outer, donor)
            for outer in CENTERS
            for donor in CENTERS
            if donor != outer
        }
    )
    jobs, route_support_labels = _build_outer_jobs(
        prepared, plans, donor_priors, firewall
    )

    _emit_phase(
        phase_observer,
        "endpoint_started",
        {
            "outer_route_count": len(plans.outer_plans),
            "model_fit_count": EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
        },
    )
    endpoint_products = (
        execute_outer_endpoint_jobs(jobs, use_processes=True)
        if use_parallel
        else tuple(compute_outer_endpoint_products(job) for job in jobs)
    )
    endpoint_fit_count = sum(row.endpoint_model_fit_count for row in endpoint_products)
    if strict and endpoint_fit_count != EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT:
        raise ProtocolError("PCSI-PARC canonical endpoint workload drifted.")
    for product in endpoint_products:
        for case_id, digest in product.state_hashes:
            firewall.record_outer_state_seal(product.target_center, case_id, digest)
    _emit_phase(
        phase_observer,
        "endpoint_completed",
        {
            "outer_route_count": sum(len(row.predictions) for row in endpoint_products),
            "model_fit_count": endpoint_fit_count,
        },
    )
    predictions_by_center = MappingProxyType(
        {row.target_center: row.predictions for row in endpoint_products}
    )

    _emit_phase(
        phase_observer,
        "posterior_started",
        {
            "outer_route_count": len(plans.outer_plans),
            "model_fit_count": EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
        },
    )
    posterior_jobs = tuple(
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
    )
    posterior_products = (
        execute_target_center_posterior_jobs(posterior_jobs, use_processes=True)
        if use_parallel
        else tuple(compute_target_center_posteriors(job) for job in posterior_jobs)
    )
    posterior_fit_count = sum(row.model_fit_count for row in posterior_products)
    if strict and posterior_fit_count != EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT:
        raise ProtocolError("PCSI-PARC canonical posterior workload drifted.")
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
    _emit_phase(
        phase_observer,
        "posterior_completed",
        {
            "outer_route_count": len(plans.outer_plans),
            "model_fit_count": posterior_fit_count,
        },
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
    pseudo_donor_endpoint_products = MappingProxyType(
        {
            (outer, pseudo, donor): recompose_outer_endpoint_products(
                job_by_center[donor],
                product_by_center[donor],
                donor_priors=pseudo_donor_prior_provenance[
                    (outer, pseudo, donor)
                ].prior_values,
            )
            for outer in CENTERS
            for pseudo in CENTERS
            for donor in CENTERS
            if len({outer, pseudo, donor}) == 3
        }
    )

    _emit_phase(
        phase_observer,
        "donor_utility_started",
        {"model_fit_count": EXPECTED_UTILITY_MODEL_FIT_COUNT},
    )
    donor_runtime = build_donor_runtime(
        predictions_by_center=predictions_by_center,
        donor_endpoint_products=donor_endpoint_products,
        pseudo_prior_provenance=pseudo_donor_prior_provenance,
        pseudo_donor_endpoint_products=pseudo_donor_endpoint_products,
        donor_labels=donor_labels,
        use_processes=use_parallel,
        strict_canonical_topology=strict,
    )
    _emit_phase(
        phase_observer,
        "donor_utility_completed",
        {"model_fit_count": donor_runtime.model_fit_count},
    )

    _emit_phase(
        phase_observer,
        "policy_replay_started",
        {"policy_replay_count": EXPECTED_POLICY_REPLAY_COUNT},
    )
    policy_runtime = build_policy_replay_runtime(
        predictions_by_center=predictions_by_center,
        endpoint_products=endpoint_products,
        donor_endpoint_products=donor_endpoint_products,
        prepared_by_center=prepared,
        donor_runtime=donor_runtime,
        target_posterior_models_by_control=posterior_models,
        target_posterior_predictions_by_control=posterior_predictions,
        label_firewall=firewall,
        strict_canonical_topology=strict,
    )
    _emit_phase(
        phase_observer,
        "policy_replay_completed",
        {"policy_replay_count": len(policy_runtime.replays)},
    )

    barrier = firewall.decision_barrier_payload()
    aggregate_payload = {
        "schema_version": "fixed_bank_pcsi_parc_preterminal_aggregate_v1",
        "protocol_hash": protocol.protocol_hash,
        "probability_surface_hash": surface.surface_hash,
        "plan_seal_hash": plans.seal_hash,
        "decision_barrier_hash": barrier["decision_barrier_hash"],
        "donor_runtime_hash": donor_runtime.runtime_hash,
        "transport_hash": policy_runtime.transport_hash,
        "policy_menu_hash": policy_runtime.policy_menu_seal[
            "policy_menu_seal_hash"
        ],
        "policy_replay_hash": canonical_hash(
            [
                policy_runtime.replays[(geometry, outer, pseudo)].replay_hash
                for geometry in donor_runtime.geometry_results
                for outer in CENTERS
                for pseudo in CENTERS
                if pseudo != outer
            ]
        ),
        "authorization_hash": canonical_hash(
            [
                row.authorization_hash
                for _key, row in sorted(policy_runtime.authorizations.items())
            ]
        ),
        "final_prediction_hash": canonical_hash(
            [
                row.prediction_hash
                for policy in COMPOSED_POLICY_IDS
                for row in policy_runtime.final_predictions_by_policy[policy]
            ]
        ),
        "workload_hash": canonical_hash(
            {
                "physical_cell_count": len(CENTERS) * 10 * 9,
                "outer_endpoint_model_fit_count": endpoint_fit_count,
                "target_posterior_model_fit_count": posterior_fit_count,
                "utility_model_fit_count": donor_runtime.model_fit_count,
                "policy_replay_count": len(policy_runtime.replays),
                "whole_case_route_count": len(plans.outer_plans),
            }
        ),
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
        primary_fingerprints,
        blocked_fingerprints,
        posterior_models,
        posterior_predictions,
        donor_runtime,
        policy_runtime,
        MappingProxyType(dict(barrier)),
        MappingProxyType(aggregate),
        firewall,
    )


def assert_transport_contract_executable(
    protocol: FrozenProtocol,
    *,
    strict_canonical_topology: bool,
) -> None:
    """Require the exact authorized support-conditioned transport contract."""

    if type(strict_canonical_topology) is not bool:
        raise ProtocolError("PCSI-PARC transport topology flag is not boolean.")
    observed = {
        key: protocol.payload.get(key)
        for key in TRANSPORT_PROTOCOL_CONTRACT
    }
    legacy = LEGACY_TRANSPORT_PROTOCOL_FIELDS.intersection(protocol.payload)
    if observed != dict(TRANSPORT_PROTOCOL_CONTRACT) or legacy:
        raise ProtocolError(
            "PCSI-PARC transport contract is not the exact audited "
            "support-conditioned endpoint-reconstructed P/B/I/R lineage."
        )
    if (
        observed["transport_identity_level_route_noninterference_proven"]
        is not True
        or observed["transport_authorization_valid"] is not True
    ):
        raise ProtocolError(
            "PCSI-PARC canonical execution is blocked: center-wide OOF "
            "transport feeds held-case and pseudo-case identities through "
            "other route-support states, so identity-level route "
            "noninterference is false and transport authorization is invalid."
        )


def _open_prior_roles(
    prepared: Mapping[str, PreparedCenter],
    firewall: PCSIPARCLabelFirewall,
) -> tuple[
    Mapping[tuple[str, str], tuple[BinaryLabel, ...]],
    Mapping[str, Mapping[tuple[str, str], float]],
    Mapping[tuple[str, str], Mapping[tuple[str, str], float]],
    Mapping[tuple[str, str, str], DoubleExcludedDonorPriorProvenance],
]:
    source_labels: dict[tuple[str, str], tuple[BinaryLabel, ...]] = {}
    priors: dict[str, Mapping[tuple[str, str], float]] = {}
    for endpoint_target in CENTERS:
        labels_by_source: dict[str, Mapping[str, tuple[BinaryLabel, ...]]] = {}
        for source in candidate_sources(endpoint_target):
            labels = firewall.open_source_prior_labels(endpoint_target, source)
            source_labels[(endpoint_target, source)] = labels
            legal = tuple(
                center
                for center in CENTERS
                if center not in {endpoint_target, source}
            )
            labels_by_source[source] = MappingProxyType(
                {
                    center: tuple(row for row in labels if row.center == center)
                    for center in legal
                }
            )
        priors[endpoint_target] = compute_donor_priors(
            prepared,
            labels_by_source,
            heldout_center=endpoint_target,
        )
    crossfit: dict[tuple[str, str], Mapping[tuple[str, str], float]] = {}
    for outer in CENTERS:
        for donor in CENTERS:
            if donor == outer:
                continue
            labels_by_source = {}
            for source in candidate_sources(donor):
                grant = source_labels[(donor, source)]
                legal = tuple(
                    center
                    for center in CENTERS
                    if center not in {outer, donor, source}
                )
                labels_by_source[source] = MappingProxyType(
                    {
                        center: tuple(row for row in grant if row.center == center)
                        for center in legal
                    }
                )
            crossfit[(outer, donor)] = compute_donor_priors(
                prepared,
                labels_by_source,
                heldout_center=donor,
                excluded_query_centers=(outer,),
            )
    pseudo_crossfit: dict[
        tuple[str, str, str], DoubleExcludedDonorPriorProvenance
    ] = {}
    for outer in CENTERS:
        for pseudo in CENTERS:
            if pseudo == outer:
                continue
            for donor in CENTERS:
                if donor in {outer, pseudo}:
                    continue
                labels_by_source = {}
                query_centers_by_source: list[tuple[str, tuple[str, ...]]] = []
                for source in candidate_sources(donor):
                    grant = source_labels[(donor, source)]
                    legal = tuple(
                        center
                        for center in CENTERS
                        if center not in {outer, pseudo, donor, source}
                    )
                    query_centers_by_source.append((source, legal))
                    labels_by_source[source] = MappingProxyType(
                        {
                            center: tuple(
                                row for row in grant if row.center == center
                            )
                            for center in legal
                        }
                    )
                values = compute_donor_priors(
                    prepared,
                    labels_by_source,
                    heldout_center=donor,
                    excluded_query_centers=(outer, pseudo),
                )
                pseudo_crossfit[(outer, pseudo, donor)] = (
                    DoubleExcludedDonorPriorProvenance.create(
                        outer_target_center=outer,
                        pseudo_target_center=pseudo,
                        donor_center=donor,
                        query_centers_by_source=query_centers_by_source,
                        prior_values=values,
                    )
                )
    return (
        MappingProxyType(source_labels),
        MappingProxyType(priors),
        MappingProxyType(crossfit),
        MappingProxyType(pseudo_crossfit),
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


def _build_outer_jobs(
    prepared: Mapping[str, PreparedCenter],
    plans: OuterPlanSeal,
    donor_priors: Mapping[str, Mapping[tuple[str, str], float]],
    firewall: PCSIPARCLabelFirewall,
) -> tuple[
    tuple[OuterEndpointJob, ...],
    Mapping[tuple[str, str], tuple[BinaryLabel, ...]],
]:
    jobs: list[OuterEndpointJob] = []
    support_by_route: dict[tuple[str, str], tuple[BinaryLabel, ...]] = {}
    for center in CENTERS:
        center_plans = tuple(
            row for row in plans.outer_plans if row.target_center == center
        )
        observed: dict[tuple[str, str, str], int] = {}
        for plan in center_plans:
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
            raise ProtocolError(
                "PCSI-PARC support-capability union does not cover one center."
            )
        scope = f"derived_support_sufficient_stats::H={center}"
        sufficient_labels = tuple(
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
                build_center_case_outcomes(prepared[center], sufficient_labels),
                center_plans,
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
            raise ProtocolError("PCSI-PARC repeated support capabilities disagree.")


def _emit_phase(
    observer: PhaseObserver | None,
    phase: str,
    counts: Mapping[str, int],
) -> None:
    if observer is not None:
        observer(phase, MappingProxyType({str(key): int(value) for key, value in counts.items()}))


__all__ = (
    "LabelLoader",
    "PhaseObserver",
    "PreterminalResult",
    "assert_transport_contract_executable",
    "build_preterminal_result",
)
