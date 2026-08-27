"""Exact H/J/K(/Q)-excluded donor and pseudo-replay worker phase."""

from __future__ import annotations

from ..capability_scoring import score_scoped_action_rectangle
from ..execution.dtos import OuterCenterTask
from ..hashing import canonical_hash
from ..identity import CENTERS, GovernanceError
from ..label_capabilities import DONOR, DelegatedWorkerLabelJournal, LabelCapability
from ..manifest_labels import ScopedCaseLabels
from ..physical.contracts import ACTION_IDS
from ..physical.endpoints import reconstruct_case_surface
from ..physical.planning import (
    DonorDirectionalPriorSurface,
    build_protected_route_plan_from_prior,
)
from ..physical_memmaps import MappedPhysicalStore
from ..posterior.contracts import DonorFitScope, DonorObservation
from ..posterior.donor import (
    DonorActionModel,
    DonorDeleteCenterFold,
    fit_donor_action_model,
)
from ..posterior.uncertainty import build_preargmax_bounds
from ..pseudo.scope import PseudoRouteKey, PseudoRouteScope
from ..pseudo.universe import PseudoActionRecord
from ..routing.admission import (
    AdmissionObservation,
    evaluate_admission,
)
from ..routing.selection import select_action
from ..utility.actions import build_action_rectangle
from .common import (
    cached_prior,
    canonical_centers,
    donor_observations,
    estimate,
    labels_for_cases,
    planning_scope_hash,
    subset_scoped_labels,
)
from .contracts import (
    DonorPhaseOutput,
    DonorSurfaceBundle,
    PseudoRouteData,
    ScienceSettings,
)
from .route_phase import build_local_model


def run_donor_phase(
    task: OuterCenterTask,
    store: MappedPhysicalStore,
    journal: DelegatedWorkerLabelJournal,
    capability: LabelCapability,
    labels: ScopedCaseLabels,
    settings: ScienceSettings,
) -> DonorPhaseOutput:
    """Build final and pseudo donor models while only donor labels are open."""

    journal.assert_active(capability, kind=DONOR)
    outer = task.target_center
    donor_centers = tuple(center for center in CENTERS if center != outer)
    if tuple(labels.labels_by_center_case) != donor_centers:
        raise GovernanceError("SCALE-BP v2 donor phase label inventory drifted.")
    prior_cache: dict[
        tuple[str, tuple[str, ...]], DonorDirectionalPriorSurface
    ] = {}
    surface_cache: dict[
        tuple[str, tuple[str, ...]], tuple[DonorSurfaceBundle, ...]
    ] = {}
    final_prior = cached_prior(
        prior_cache,
        store,
        labels,
        target_center=outer,
        source_excluded_centers=canonical_centers({outer}),
        donor_scope_hash=capability.scope_hash,
    )
    final_scope = DonorFitScope(
        outer_center=outer,
        prediction_center=outer,
        held_case_id=f"CENTER_{outer}_FULLY_EXCLUDED_ALL_CASES",
        training_case_ids_by_center={
            center: store.case_ids(center) for center in donor_centers
        },
        source_excluded_centers=canonical_centers({outer}),
        role="FINAL_H_C",
    )
    final_model, final_model_manifest_hash = fit_donor_model_with_delete_folds(
        store,
        journal,
        capability,
        labels,
        scope=final_scope,
        settings=settings,
        prior_cache=prior_cache,
        surface_cache=surface_cache,
    )
    admission_observations: list[AdmissionObservation] = []
    pseudo_record_hashes: list[str] = []
    pseudo_model_hashes: list[str] = []
    pseudo_prior_hashes: list[str] = []

    for donor_center in donor_centers:
        source_exclusions = canonical_centers({outer, donor_center})
        pseudo_prior = cached_prior(
            prior_cache,
            store,
            labels,
            target_center=donor_center,
            source_excluded_centers=source_exclusions,
            donor_scope_hash=capability.scope_hash,
        )
        pseudo_prior_hashes.append(pseudo_prior.prior_hash)
        pseudo_routes: list[PseudoRouteData] = []
        donor_cases = store.case_ids(donor_center)
        donor_surfaces = donor_query_surfaces(
            surface_cache,
            prior_cache,
            store,
            journal,
            capability,
            labels,
            query_center=donor_center,
            source_exclusions=source_exclusions,
        )
        by_case = {bundle.rectangle.case_id: bundle for bundle in donor_surfaces}
        if tuple(by_case) != donor_cases:
            raise GovernanceError("SCALE-BP v2 pseudo donor surface order drifted.")
        for held_case in donor_cases:
            support_cases = tuple(
                case for case in donor_cases if case != held_case
            )
            route_scope = PseudoRouteScope(
                PseudoRouteKey(outer, donor_center, held_case),
                support_cases,
                {
                    center: store.case_ids(center)
                    for center in CENTERS
                    if center not in {outer, donor_center}
                },
                source_exclusions,
            )
            bundle = by_case[held_case]
            pseudo_routes.append(
                PseudoRouteData(route_scope, bundle.rectangle, bundle.scored)
            )

        pseudo_scope = DonorFitScope(
            outer_center=outer,
            prediction_center=donor_center,
            held_case_id=f"CENTER_{donor_center}_FULLY_EXCLUDED_ALL_D",
            training_case_ids_by_center={
                center: store.case_ids(center)
                for center in CENTERS
                if center not in {outer, donor_center}
            },
            source_excluded_centers=source_exclusions,
            role="PSEUDO_H_J_D",
        )
        # Seed this J-local cache only with its already reconstructed H/J/K
        # base surfaces. Deeper H/J/K/Q folds are freed immediately after the
        # one reusable pseudo model for J is fitted.
        pseudo_surface_cache = {
            key: value
            for key, value in surface_cache.items()
            if key[0] != donor_center
            and set(key[1]) == {outer, donor_center, key[0]}
        }
        pseudo_model, nested_manifest_hash = fit_donor_model_with_delete_folds(
            store,
            journal,
            capability,
            labels,
            scope=pseudo_scope,
            settings=settings,
            prior_cache=prior_cache,
            surface_cache=pseudo_surface_cache,
        )
        pseudo_model_hashes.extend((pseudo_model.model_hash, nested_manifest_hash))
        route_support_labels = subset_scoped_labels(
            labels,
            center=donor_center,
            case_ids=donor_cases,
            identity_role=f"pseudo_center_{donor_center}",
        )
        for route in pseudo_routes:
            held_case = route.scope.key.case_id
            support_cases = route.scope.donor_support_case_ids
            local_scope_labels = subset_scoped_labels(
                route_support_labels,
                center=donor_center,
                case_ids=support_cases,
                identity_role=f"pseudo_route_{donor_center}_{held_case}",
            )
            local_model, local_observations, _ = build_local_model(
                task,
                store,
                journal,
                capability,
                local_scope_labels,
                target_center=donor_center,
                route_case_id=held_case,
                donor_prior=pseudo_prior,
                donor_model=pseudo_model,
                local_ridge_alpha=settings.local_ridge_alpha,
                role="PSEUDO_LOCAL_H_J_D",
            )
            estimates = estimate(
                route.rectangle,
                pseudo_model,
                local_model,
                settings,
            )
            bounds = build_preargmax_bounds(
                estimates,
                base_multiplier=settings.uncertainty_base_multiplier,
            )
            decision = select_action(
                route.rectangle,
                estimates,
                bounds,
                thresholds=settings.safety_thresholds,
            )
            realized = {value.action_id: value for value in route.scored.values}
            admission_observations.append(
                AdmissionObservation(
                    donor_center,
                    f"H={outer}::J={donor_center}::d={held_case}",
                    {row.action_id: row.mean for row in estimates},
                    {
                        action_id: realized[action_id].value
                        for action_id in ACTION_IDS
                    },
                    decision.selected_action_id,
                )
            )
            estimate_by_id = {row.action_id: row for row in estimates}
            for action_id in ACTION_IDS:
                cell = route.rectangle.cell(action_id)
                record = PseudoActionRecord(
                    route.scope,
                    action_id,
                    estimate_by_id[action_id].mean,
                    realized[action_id].value,
                    cell.evidence.descriptor.feature_hash,
                    estimate_by_id[action_id].estimate_hash,
                    realized[action_id].value_hash,
                    decision.selected_action_id == action_id,
                    cell.structural_noop,
                )
                pseudo_record_hashes.append(record.record_hash)
            # Retain only fitted/hash state after this route; raw arrays remain
            # capability-local and die with the scoped label objects.
            del local_observations, local_scope_labels
        del pseudo_surface_cache
        removable = tuple(
            key
            for key in surface_cache
            if (
                key == (donor_center, source_exclusions)
                or (
                    key[0] != donor_center
                    and set(key[1]) == {outer, donor_center, key[0]}
                )
            )
        )
        for key in removable:
            del surface_cache[key]

    admission = evaluate_admission(
        admission_observations,
        thresholds=settings.admission_thresholds,
    )
    pseudo_replay_hash = canonical_hash(
        {
            "schema_version": "scale_bp_v2_outer_pseudo_replay_v1",
            "outer_center": outer,
            "record_hashes": pseudo_record_hashes,
            "record_count": len(pseudo_record_hashes),
            "admission_observation_count": len(admission_observations),
            "admission_metrics_hash": admission.metrics_hash,
            "complete_six_action_rectangles": True,
            "source_exclusions_H_J_K_enforced": True,
            "raw_labels_persisted": False,
        }
    )
    pseudo_model_manifest_hash = canonical_hash(
        {
            "schema_version": "scale_bp_v2_outer_pseudo_model_manifest_v1",
            "outer_center": outer,
            "final_model_manifest_hash": final_model_manifest_hash,
            "pseudo_model_and_nested_manifest_hashes": pseudo_model_hashes,
            "pseudo_prior_hashes": pseudo_prior_hashes,
            "one_model_per_fully_excluded_prediction_center": True,
        }
    )
    donor_phase_hash = canonical_hash(
        {
            "schema_version": "scale_bp_v2_outer_donor_phase_v1",
            "outer_center": outer,
            "final_prior_hash": final_prior.prior_hash,
            "final_model_hash": final_model.model_hash,
            "pseudo_replay_hash": pseudo_replay_hash,
            "pseudo_model_manifest_hash": pseudo_model_manifest_hash,
            "admission_metrics_hash": admission.metrics_hash,
            "donor_capability_scope_hash": capability.scope_hash,
            "raw_labels_retained": False,
        }
    )
    return DonorPhaseOutput(
        final_prior,
        final_model,
        admission,
        pseudo_replay_hash,
        len(pseudo_record_hashes),
        pseudo_model_manifest_hash,
        donor_phase_hash,
    )


def fit_donor_model_with_delete_folds(
    store: MappedPhysicalStore,
    journal: DelegatedWorkerLabelJournal,
    capability: LabelCapability,
    labels: ScopedCaseLabels,
    *,
    scope: DonorFitScope,
    settings: ScienceSettings,
    prior_cache: dict[
        tuple[str, tuple[str, ...]], DonorDirectionalPriorSurface
    ],
    surface_cache: dict[
        tuple[str, tuple[str, ...]], tuple[DonorSurfaceBundle, ...]
    ],
) -> tuple[DonorActionModel, str]:
    """Fit with exact source-and-query delete-center reconstructions."""

    observations: list[DonorObservation] = []
    base_by_query: dict[str, tuple[DonorObservation, ...]] = {}
    base_surface_hashes: list[str] = []
    for query_center in scope.training_centers:
        query_exclusions = canonical_centers(
            {*scope.source_excluded_centers, query_center}
        )
        bundles = donor_query_surfaces(
            surface_cache,
            prior_cache,
            store,
            journal,
            capability,
            labels,
            query_center=query_center,
            source_exclusions=query_exclusions,
        )
        query_observations = tuple(
            observation
            for bundle in bundles
            for observation in donor_observations(
                bundle.rectangle,
                bundle.scored,
                scope=scope,
                source_centers=bundle.source_centers,
            )
        )
        base_by_query[query_center] = query_observations
        observations.extend(query_observations)
        base_surface_hashes.extend(bundle.plan_hash for bundle in bundles)

    folds: list[DonorDeleteCenterFold] = []
    fold_surface_manifest_hashes: list[str] = []
    for deleted_center in scope.training_centers:
        fold_exclusions = canonical_centers(
            {*scope.source_excluded_centers, deleted_center}
        )
        training_scope = DonorFitScope(
            outer_center=scope.outer_center,
            prediction_center=scope.prediction_center,
            held_case_id=scope.held_case_id,
            training_case_ids_by_center={
                center: case_ids
                for center, case_ids in scope.training_case_ids_by_center.items()
                if center != deleted_center
            },
            source_excluded_centers=fold_exclusions,
            role=scope.role,
        )
        training_observations: list[DonorObservation] = []
        fold_plan_hashes: list[str] = []
        for query_center in training_scope.training_centers:
            query_exclusions = canonical_centers(
                {*fold_exclusions, query_center}
            )
            bundles = donor_query_surfaces(
                surface_cache,
                prior_cache,
                store,
                journal,
                capability,
                labels,
                query_center=query_center,
                source_exclusions=query_exclusions,
            )
            fold_plan_hashes.extend(bundle.plan_hash for bundle in bundles)
            training_observations.extend(
                observation
                for bundle in bundles
                for observation in donor_observations(
                    bundle.rectangle,
                    bundle.scored,
                    scope=training_scope,
                    source_centers=bundle.source_centers,
                )
            )
        fold = DonorDeleteCenterFold(
            deleted_center=deleted_center,
            base_scope=scope,
            training_scope=training_scope,
            training_observations=tuple(training_observations),
            validation_observations=base_by_query[deleted_center],
        )
        folds.append(fold)
        fold_surface_manifest_hashes.append(
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_delete_center_surface_fold_v1",
                    "deleted_center": deleted_center,
                    "training_scope_hash": training_scope.scope_hash,
                    "plan_hashes": fold_plan_hashes,
                    "complete_exclusion_tuple_reconstruction": True,
                    "base_table_filtering_used": False,
                }
            )
        )
    model = fit_donor_action_model(
        observations,
        scope=scope,
        delete_center_folds=folds,
        ridge_alpha=settings.donor_ridge_alpha,
    )
    manifest_hash = canonical_hash(
        {
            "schema_version": "scale_bp_v2_strict_donor_model_manifest_v1",
            "outer_center": scope.outer_center,
            "prediction_center": scope.prediction_center,
            "scope_hash": scope.scope_hash,
            "model_hash": model.model_hash,
            "base_surface_plan_hashes": base_surface_hashes,
            "delete_center_fold_hashes": [fold.fold_hash for fold in folds],
            "delete_center_surface_manifest_hashes": fold_surface_manifest_hashes,
            "source_and_query_delete_center_reconstruction": True,
            "less_restrictive_base_table_filtering_used": False,
        }
    )
    return model, manifest_hash


def donor_query_surfaces(
    surface_cache: dict[
        tuple[str, tuple[str, ...]], tuple[DonorSurfaceBundle, ...]
    ],
    prior_cache: dict[
        tuple[str, tuple[str, ...]], DonorDirectionalPriorSurface
    ],
    store: MappedPhysicalStore,
    journal: DelegatedWorkerLabelJournal,
    capability: LabelCapability,
    labels: ScopedCaseLabels,
    *,
    query_center: str,
    source_exclusions: tuple[str, ...],
) -> tuple[DonorSurfaceBundle, ...]:
    """Reconstruct an exact exclusion-scoped query surface inventory."""

    key = (query_center, source_exclusions)
    cached = surface_cache.get(key)
    if cached is not None:
        return cached
    if query_center not in source_exclusions:
        raise GovernanceError("SCALE-BP v2 donor query must be source-excluded.")
    prior = cached_prior(
        prior_cache,
        store,
        labels,
        target_center=query_center,
        source_excluded_centers=source_exclusions,
        donor_scope_hash=capability.scope_hash,
    )
    rows: list[DonorSurfaceBundle] = []
    query_cases = store.case_ids(query_center)
    for held_case in query_cases:
        support_cases = tuple(case for case in query_cases if case != held_case)
        plan = build_protected_route_plan_from_prior(
            store,
            target_center=query_center,
            case_id=held_case,
            support_labels_by_case=labels_for_cases(
                labels, query_center, support_cases
            ),
            donor_prior=prior,
            support_scope_hash=planning_scope_hash(
                "DONOR_DELETE_CENTER_SURFACE",
                capability,
                target_center=query_center,
                case_id=held_case,
                source_exclusions=source_exclusions,
                support_exclusions=(held_case,),
                parent_hash=prior.prior_hash,
            ),
        )
        surface = reconstruct_case_surface(store, plan)
        rectangle = build_action_rectangle(surface)
        scored = score_scoped_action_rectangle(
            journal,
            capability,
            labels,
            rectangle,
        )
        rows.append(
            DonorSurfaceBundle(
                rectangle,
                scored,
                surface.available_sources,
                prior.prior_hash,
                plan.plan_hash,
            )
        )
    result = tuple(rows)
    surface_cache[key] = result
    return result


__all__ = (
    "donor_query_surfaces",
    "fit_donor_model_with_delete_folds",
    "run_donor_phase",
)
