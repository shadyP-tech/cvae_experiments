"""Phase-ordered construction of the sealed target and pseudo candidates.

This module owns orchestration only.  Endpoint fitting, posterior fitting and
candidate scoring remain in their small scientific kernels.  Raw labels enter
only through :class:`CBPUPRLabelFirewall` and are reduced to route-scoped
sufficient statistics before a spawned worker is started.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .candidate_runtime import CandidateRuntimeResult, build_case_candidates
from .constants import (
    ALTERNATIVE_METHOD_IDS,
    CENTERS,
    EXPECTED_OUTER_PLAN_COUNT,
    EXPECTED_PSEUDO_ROUTE_COUNT,
)
from .contracts import BinaryLabel, PhysicalProbabilitySurface
from .endpoint_preparation import (
    PreparedCenter,
    build_center_case_outcomes,
    compute_donor_priors,
    prepare_center,
)
from .hashing import canonical_hash
from .label_capabilities import CBPUPRLabelFirewall
from .outer_endpoint_runtime import (
    OuterEndpointJob,
    OuterEndpointProducts,
    build_outer_endpoint_job,
    execute_outer_endpoint_jobs,
    recompose_outer_endpoint_products,
)
from .outer_plans import OuterPlanSeal, build_outer_plans
from .physical_fingerprint import (
    blocked_within_case_fingerprint,
    build_physical_fingerprint_surface,
)
from .posterior_contracts import (
    CONTROL_IDS,
    CasePosteriorPrediction,
    PhysicalFingerprintSurface,
    PseudoPosteriorReference,
    TargetLocalPosteriorModel,
    build_pseudo_posterior_references,
    index_predictions,
)
from .pseudo_endpoint_evidence import (
    PseudoEndpointEvidence,
    PseudoSourcePriorEvidence,
    build_pseudo_source_prior_evidence,
)
from .target_local_runtime import (
    TargetCenterPosteriorJob,
    execute_target_posterior_jobs,
)


LabelLoader = Callable[
    [frozenset[tuple[str, str, str]], str], Sequence[object]
]


@dataclass(frozen=True)
class SealedCandidateProducts:
    plan_seal: OuterPlanSeal
    firewall: CBPUPRLabelFirewall
    prepared_centers: tuple[tuple[str, PreparedCenter], ...]
    primary_fingerprints: tuple[PhysicalFingerprintSurface, ...]
    blocked_fingerprints: tuple[PhysicalFingerprintSurface, ...]
    endpoint_jobs: tuple[OuterEndpointJob, ...]
    endpoint_products: tuple[OuterEndpointProducts, ...]
    pseudo_source_prior_evidence: tuple[PseudoSourcePriorEvidence, ...]
    pseudo_endpoint_evidence: tuple[PseudoEndpointEvidence, ...]
    posterior_models: tuple[TargetLocalPosteriorModel, ...]
    posterior_predictions: tuple[CasePosteriorPrediction, ...]
    pseudo_posterior_references: tuple[PseudoPosteriorReference, ...]
    target_portfolios: tuple[tuple[str, str, tuple[float, ...]], ...]
    pseudo_portfolios: tuple[tuple[str, str, str, tuple[float, ...]], ...]
    target_candidates: tuple[CandidateRuntimeResult, ...]
    pseudo_candidates: tuple[CandidateRuntimeResult, ...]
    target_candidate_seal_hash: str
    pre_evaluation_seal_hash: str


def build_outer_endpoint_jobs(
    surface: PhysicalProbabilitySurface,
    *,
    plan_seal: OuterPlanSeal,
    prepared_centers: Mapping[str, PreparedCenter],
    support_by_route: Mapping[tuple[str, str], Sequence[BinaryLabel]],
    ordinary_priors: Mapping[str, Mapping[tuple[str, str], float]],
) -> tuple[OuterEndpointJob, ...]:
    """Build the production endpoint jobs with both surface hash roles bound."""

    if plan_seal.probability_surface_hash != surface.surface_hash:
        raise ProtocolError("CBPUPR endpoint factory global surface lineage drifted.")
    try:
        jobs = tuple(
            build_outer_endpoint_job(
                surface,
                target_center=center,
                prepared=prepared_centers[center],
                route_outcomes=tuple(
                    (
                        plan.case_id,
                        build_center_case_outcomes(
                            prepared_centers[center],
                            support_by_route[plan.key],
                            expected_scope=(
                                f"outer_support::H={center}::excluded_c={plan.case_id}"
                            ),
                        ),
                    )
                    for plan in plan_seal.outer_plans
                    if plan.target_center == center
                ),
                outer_plans=tuple(
                    plan
                    for plan in plan_seal.outer_plans
                    if plan.target_center == center
                ),
                donor_priors=ordinary_priors[center],
            )
            for center in CENTERS
        )
    except KeyError as exc:
        raise ProtocolError("CBPUPR endpoint factory input rectangle drifted.") from exc
    if tuple(job.target_center for job in jobs) != CENTERS:
        raise ProtocolError("CBPUPR endpoint factory job order drifted.")
    return jobs


def build_sealed_candidates(
    surface: PhysicalProbabilitySurface,
    label_loader: LabelLoader,
    *,
    use_processes: bool = True,
) -> SealedCandidateProducts:
    """Construct the full 218/1,744 candidate rectangle before evaluation."""

    identities = tuple(
        SimpleNamespace(
            center=center,
            case_id=case,
            sample_id=sample,
            group_id=case,
        )
        for center in CENTERS
        for sample, case in zip(
            surface.centers[center].sample_ids,
            surface.centers[center].case_ids,
            strict=True,
        )
    )
    plans = build_outer_plans(
        identities,
        probability_surface_hash=surface.surface_hash,
        strict_canonical_topology=surface.strict_canonical_topology,
    )
    firewall = CBPUPRLabelFirewall(plans, label_loader)
    prepared = {center: prepare_center(surface.centers[center]) for center in CENTERS}

    # Open the 72 ordinary and 504 H/J/source prior capabilities exactly once.
    ordinary_priors: dict[str, Mapping[tuple[str, str], float]] = {}
    pseudo_priors: dict[tuple[str, str], Mapping[tuple[str, str], float]] = {}
    for target in CENTERS:
        labels_by_source = {
            source: _split_labels_by_center(
                firewall.open_source_prior_labels(target, source),
                excluded_centers={target, source},
            )
            for source in CENTERS
            if source != target
        }
        ordinary_priors[target] = compute_donor_priors(
            prepared, labels_by_source, heldout_center=target
        )
    for outer in CENTERS:
        for pseudo in CENTERS:
            if pseudo == outer:
                continue
            labels_by_source = {
                source: _split_labels_by_center(
                    firewall.open_source_prior_labels(
                        pseudo, source, outer_excluded_center=outer
                    ),
                    excluded_centers={outer, pseudo, source},
                )
                for source in CENTERS
                if source not in {outer, pseudo}
            }
            pseudo_priors[(outer, pseudo)] = compute_donor_priors(
                prepared,
                labels_by_source,
                heldout_center=pseudo,
                excluded_query_centers=(outer,),
                excluded_source_centers=(outer,),
            )

    support_by_route: dict[tuple[str, str], tuple[BinaryLabel, ...]] = {}
    for plan in plans.outer_plans:
        support_by_route[plan.key] = firewall.open_outer_support_labels(*plan.key)
    if len(support_by_route) != EXPECTED_OUTER_PLAN_COUNT:
        raise ProtocolError("CBPUPR support capability rectangle drifted.")

    primary = {
        center: build_physical_fingerprint_surface(surface.centers[center])
        for center in CENTERS
    }
    blocked = {
        center: blocked_within_case_fingerprint(primary[center])
        for center in CENTERS
    }
    endpoint_jobs = build_outer_endpoint_jobs(
        surface,
        plan_seal=plans,
        prepared_centers=prepared,
        support_by_route=support_by_route,
        ordinary_priors=ordinary_priors,
    )
    endpoint_products = execute_outer_endpoint_jobs(
        endpoint_jobs, use_processes=use_processes
    )
    posterior_products = execute_target_posterior_jobs(
        tuple(
            TargetCenterPosteriorJob(
                center,
                primary[center],
                blocked[center],
                tuple(
                    (plan.case_id, support_by_route[plan.key])
                    for plan in plans.outer_plans
                    if plan.target_center == center
                ),
            )
            for center in CENTERS
        ),
        use_processes=use_processes,
    )
    models = tuple(row for product in posterior_products for row in product.models)
    predictions = tuple(
        row for product in posterior_products for row in product.predictions
    )
    prediction_by_key = index_predictions(predictions)
    model_by_key = {
        (row.target_center, row.held_case_id, row.control_id): row for row in models
    }
    pseudo_references = build_pseudo_posterior_references(predictions)
    reference_by_key = {
        (
            row.outer_target_center,
            row.pseudo_target_center,
            row.held_case_id,
            row.control_id,
        ): row
        for row in pseudo_references
    }
    endpoints_by_center = {row.target_center: row for row in endpoint_products}
    endpoint_by_key = {
        (product.target_center, prediction.case_id): prediction
        for product in endpoint_products
        for prediction in product.predictions
    }
    capability_events = tuple(candidates_event for candidates_event in firewall.audit_payload()["events"])
    pseudo_prior_evidence_by_key = {
        (outer, pseudo): build_pseudo_source_prior_evidence(
            outer_center=outer,
            target_center=pseudo,
            priors=pseudo_priors[(outer, pseudo)],
            capability_events=capability_events,
        )
        for outer in CENTERS
        for pseudo in CENTERS
        if pseudo != outer
    }

    target_candidates: list[CandidateRuntimeResult] = []
    target_portfolios: list[tuple[str, str, tuple[float, ...]]] = []
    for plan in plans.outer_plans:
        endpoint = endpoint_by_key[plan.key]
        target_portfolios.append(
            (
                plan.target_center,
                plan.case_id,
                tuple(endpoint.probabilities["P_PROTECTED"]),
            )
        )
        for control in CONTROL_IDS:
            posterior = prediction_by_key[(plan.target_center, plan.case_id, control)]
            model = model_by_key[(plan.target_center, plan.case_id, control)]
            target_candidates.append(
                _build_candidate_runtime(
                    outer=plan.target_center,
                    center=plan.target_center,
                    case=plan.case_id,
                    control=control,
                    endpoint=endpoint,
                    posterior=posterior,
                    model=model,
                    capability_hash=_support_capability_hash(
                        support_by_route[plan.key]
                    ),
                    excluded_sources=(plan.target_center,),
                )
            )

    pseudo_candidates: list[CandidateRuntimeResult] = []
    pseudo_portfolios: list[tuple[str, str, str, tuple[float, ...]]] = []
    pseudo_endpoint_evidence: list[PseudoEndpointEvidence] = []
    for outer in CENTERS:
        for pseudo in CENTERS:
            if pseudo == outer:
                continue
            recomposed = recompose_outer_endpoint_products(
                next(job for job in endpoint_jobs if job.target_center == pseudo),
                endpoints_by_center[pseudo],
                donor_priors=pseudo_priors[(outer, pseudo)],
                excluded_source_centers=(outer,),
            )
            recomposed_by_case = {row.case_id: row for row in recomposed.predictions}
            for case, endpoint in recomposed_by_case.items():
                endpoint_evidence = PseudoEndpointEvidence(
                    outer,
                    endpoint,
                    pseudo_prior_evidence_by_key[(outer, pseudo)].source_prior_hash,
                )
                pseudo_endpoint_evidence.append(endpoint_evidence)
                pseudo_portfolios.append(
                    (
                        outer,
                        pseudo,
                        case,
                        tuple(endpoint.probabilities["P_PROTECTED"]),
                    )
                )
                for control in CONTROL_IDS:
                    posterior = prediction_by_key[(pseudo, case, control)]
                    model = model_by_key[(pseudo, case, control)]
                    reference = reference_by_key[(outer, pseudo, case, control)]
                    pseudo_candidates.append(
                        _build_candidate_runtime(
                            outer=outer,
                            center=pseudo,
                            case=case,
                            control=control,
                            endpoint=endpoint,
                            posterior=posterior,
                            model=model,
                            capability_hash=reference.reference_hash,
                            excluded_sources=(outer, pseudo),
                            endpoint_lineage_hash=endpoint_evidence.evidence_hash,
                        )
                    )
    if (
        len(pseudo_prior_evidence_by_key) != len(CENTERS) * (len(CENTERS) - 1)
        or len(pseudo_endpoint_evidence) != EXPECTED_PSEUDO_ROUTE_COUNT
        or len(pseudo_candidates) != 2 * EXPECTED_PSEUDO_ROUTE_COUNT
    ):
        raise ProtocolError("CBPUPR pseudo candidate workload drifted.")

    target_hashes = {
        (row.center, row.case_id, row.control_id): row.runtime_hash
        for row in target_candidates
    }
    pseudo_hashes = {
        (row.outer_center, row.center, row.case_id, row.control_id): row.runtime_hash
        for row in pseudo_candidates
    }
    candidate_seal = firewall.seal_candidates(target_hashes, pseudo_hashes)
    lineage_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_cbpupr_structural_lineage_v1",
            "physical_surface_hash": surface.surface_hash,
            "outer_plan_seal_hash": plans.seal_hash,
            "candidate_seal_hash": candidate_seal,
            "pseudo_posterior_reference_hashes": sorted(
                row.reference_hash for row in pseudo_references
            ),
            "numeric_transport_is_authorization_gate": False,
        }
    )
    pre_evaluation = firewall.seal_pre_evaluation(lineage_hash)
    return SealedCandidateProducts(
        plans,
        firewall,
        tuple((center, prepared[center]) for center in CENTERS),
        tuple(primary[center] for center in CENTERS),
        tuple(blocked[center] for center in CENTERS),
        endpoint_jobs,
        endpoint_products,
        tuple(pseudo_prior_evidence_by_key.values()),
        tuple(pseudo_endpoint_evidence),
        models,
        predictions,
        pseudo_references,
        tuple(target_portfolios),
        tuple(pseudo_portfolios),
        tuple(target_candidates),
        tuple(pseudo_candidates),
        candidate_seal,
        pre_evaluation,
    )


def _split_labels_by_center(
    labels: Sequence[BinaryLabel], *, excluded_centers: set[str]
) -> dict[str, tuple[BinaryLabel, ...]]:
    rows = tuple(labels)
    return {
        center: tuple(row for row in rows if row.center == center)
        for center in CENTERS
        if center not in excluded_centers
    }


def _support_capability_hash(labels: Sequence[BinaryLabel]) -> str:
    rows = tuple(labels)
    return canonical_hash(
        {
            "schema_version": "fixed_bank_cbpupr_support_reference_v1",
            "scope": rows[0].scope if rows else None,
            "identities": [list(row.key) for row in rows],
            "values_persisted": False,
        }
    )


def _build_candidate_runtime(
    *,
    outer: str,
    center: str,
    case: str,
    control: str,
    endpoint: object,
    posterior: CasePosteriorPrediction,
    model: TargetLocalPosteriorModel,
    capability_hash: str,
    excluded_sources: Sequence[str],
    endpoint_lineage_hash: str | None = None,
) -> CandidateRuntimeResult:
    probabilities = getattr(endpoint, "probabilities")
    alternatives = {
        method: probabilities[method] for method in ALTERNATIVE_METHOD_IDS
    }
    return build_case_candidates(
        center=center,
        case_id=case,
        portfolio_probabilities=probabilities["P_PROTECTED"],
        alternative_probabilities=alternatives,
        posterior_eta=posterior.natural_probabilities,
        control_id=control,
        support_n_positive=model.training_n_positive,
        support_n_negative=model.training_n_negative,
        support_row_count=model.training_row_count,
        posterior_model_hash=model.model_hash,
        support_capability_hash=capability_hash,
        outer_center=outer,
        source_excluded_centers=excluded_sources,
        endpoint_lineage_hash=(
            getattr(endpoint, "prediction_hash")
            if endpoint_lineage_hash is None
            else endpoint_lineage_hash
        ),
    )


__all__ = (
    "SealedCandidateProducts",
    "build_outer_endpoint_jobs",
    "build_sealed_candidates",
)
