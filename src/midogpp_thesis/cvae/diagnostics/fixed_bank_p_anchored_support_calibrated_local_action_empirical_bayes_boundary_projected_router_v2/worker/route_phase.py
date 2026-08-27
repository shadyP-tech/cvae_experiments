"""Ephemeral H-minus-c/J-minus-d local adaptation and route selection."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import numpy as np

from ..capability_scoring import score_scoped_action_rectangle
from ..execution.dtos import OuterCenterTask
from ..hashing import canonical_hash
from ..identity import GovernanceError
from ..label_capabilities import SUPPORT, DelegatedWorkerLabelJournal, LabelCapability
from ..manifest_labels import ScopedCaseLabels
from ..physical.contracts import (
    ACTION_IDS,
    PRIMARY_METHOD_ID,
    P_METHOD_ID,
    MetricVector,
)
from ..physical.endpoints import reconstruct_case_surface
from ..physical.planning import (
    DonorDirectionalPriorSurface,
    build_protected_route_plan_from_prior,
)
from ..physical_memmaps import MappedPhysicalStore
from ..posterior.contracts import ScaleVector
from ..posterior.donor import DonorActionModel, predict_donor_action
from ..posterior.empirical_bayes import ActionEstimate
from ..posterior.local import (
    LOCAL_FOLD_COUNT,
    LocalResidualModel,
    assign_local_support_folds,
    build_local_residual_observations,
    fit_route_local_residual,
)
from ..posterior.uncertainty import build_preargmax_bounds
from ..routing.admission import AdmissionMetrics
from ..routing.controls import (
    CONTROL_METHOD_IDS,
    cyclically_poison_action_identities,
    donor_only_estimates,
    local_only_estimates,
    permute_local_residuals,
)
from ..routing.selection import RouteDecision, select_action
from ..terminal.scoring import sealed_probability_hash
from ..utility.actions import ActionRectangle, build_action_rectangle
from ..utility.metrics import ScoredActionRectangle
from .common import (
    estimate,
    labels_for_cases,
    planning_scope_hash,
    select_control,
)
from .contracts import FinalRouteOutput, METHOD_IDS, ScienceSettings
from .emission import decision_payload


def run_final_route(
    task: OuterCenterTask,
    store: MappedPhysicalStore,
    journal: DelegatedWorkerLabelJournal,
    capability: LabelCapability,
    labels: ScopedCaseLabels,
    *,
    case_id: str,
    final_prior: DonorDirectionalPriorSurface,
    final_donor_model: DonorActionModel,
    admission: AdmissionMetrics,
    settings: ScienceSettings,
) -> FinalRouteOutput:
    """Fit one ephemeral H-minus-c adapter and emit primary plus controls."""

    journal.assert_active(capability, kind=SUPPORT)
    target = task.target_center
    all_cases = store.case_ids(target)
    support_cases = tuple(case for case in all_cases if case != case_id)
    if labels.case_ids(target) != support_cases:
        raise GovernanceError("SCALE-BP v2 final support decoder order drifted.")
    final_plan = build_protected_route_plan_from_prior(
        store,
        target_center=target,
        case_id=case_id,
        support_labels_by_case=labels_for_cases(labels, target, support_cases),
        donor_prior=final_prior,
        support_scope_hash=capability.scope_hash,
    )
    surface = reconstruct_case_surface(store, final_plan)
    rectangle = build_action_rectangle(surface)
    local_model, local_observations, local_plan_manifest_hash = build_local_model(
        task,
        store,
        journal,
        capability,
        labels,
        target_center=target,
        route_case_id=case_id,
        donor_prior=final_prior,
        donor_model=final_donor_model,
        local_ridge_alpha=settings.local_ridge_alpha,
        role="FINAL_LOCAL_H_C",
    )
    estimates = estimate(rectangle, final_donor_model, local_model, settings)
    bounds = build_preargmax_bounds(
        estimates,
        base_multiplier=settings.uncertainty_base_multiplier,
    )
    ungated_primary = select_action(
        rectangle,
        estimates,
        bounds,
        thresholds=settings.safety_thresholds,
    )
    primary = (
        ungated_primary
        if admission.passed
        else force_exact_p(
            rectangle,
            ungated_primary,
            reason="EXACT_P_OUTER_PSEUDO_ADMISSION_FAILED",
        )
    )

    donor_estimates = donor_only_estimates(estimates)
    donor_decision = select_control(rectangle, donor_estimates, settings)
    local_estimates = local_only_estimates(estimates)
    local_decision = select_control(rectangle, local_estimates, settings)

    permuted_observations = permute_local_residuals(local_observations)
    permuted_model = fit_route_local_residual(
        permuted_observations,
        ridge_alpha=settings.local_ridge_alpha,
    )
    permuted_estimates = estimate(
        rectangle,
        final_donor_model,
        permuted_model,
        settings,
    )
    permutation_decision = select_control(
        rectangle, permuted_estimates, settings
    )

    poison_estimates = normalize_poison_estimates(
        rectangle,
        cyclically_poison_action_identities(estimates),
    )
    poison_decision = select_control(rectangle, poison_estimates, settings)

    method_probabilities = MappingProxyType(
        {
            P_METHOD_ID: np.asarray(surface.protected_p, dtype=np.float64),
            PRIMARY_METHOD_ID: np.asarray(
                primary.emitted_probabilities, dtype=np.float64
            ),
            CONTROL_METHOD_IDS[0]: np.asarray(
                donor_decision.emitted_probabilities, dtype=np.float64
            ),
            CONTROL_METHOD_IDS[1]: np.asarray(
                local_decision.emitted_probabilities, dtype=np.float64
            ),
            CONTROL_METHOD_IDS[2]: np.asarray(
                permutation_decision.emitted_probabilities, dtype=np.float64
            ),
            CONTROL_METHOD_IDS[3]: np.asarray(
                poison_decision.emitted_probabilities, dtype=np.float64
            ),
            CONTROL_METHOD_IDS[4]: np.asarray(
                primary.full_endpoint_probabilities, dtype=np.float64
            ),
        }
    )
    per_method_hashes = {
        method_id: sealed_probability_hash(method_probabilities[method_id])
        for method_id in METHOD_IDS
    }
    body = {
        "schema_version": "scale_bp_v2_route_record_v1",
        "target_center": target,
        "case_id": case_id,
        "sample_identity_hash": canonical_hash(surface.sample_ids),
        "support_scope_hash": capability.scope_hash,
        "final_plan_hash": final_plan.plan_hash,
        "endpoint_surface_hash": surface.surface_hash,
        "rectangle_hash": rectangle.rectangle_hash,
        "donor_model_hash": final_donor_model.model_hash,
        "local_model_hash": local_model.model_hash,
        "local_plan_manifest_hash": local_plan_manifest_hash,
        "permuted_local_model_hash": permuted_model.model_hash,
        "admission_metrics_hash": admission.metrics_hash,
        "admission_passed": admission.passed,
        "primary": decision_payload(primary, rectangle),
        "ungated_primary_decision_hash": ungated_primary.decision_hash,
        "controls": {
            CONTROL_METHOD_IDS[0]: decision_payload(donor_decision, rectangle),
            CONTROL_METHOD_IDS[1]: decision_payload(local_decision, rectangle),
            CONTROL_METHOD_IDS[2]: decision_payload(
                permutation_decision, rectangle
            ),
            CONTROL_METHOD_IDS[3]: decision_payload(poison_decision, rectangle),
            CONTROL_METHOD_IDS[4]: {
                "source_decision_hash": primary.decision_hash,
                "selected_action_id": primary.selected_action_id,
                "reason": primary.reason,
                "full_endpoint_only": True,
            },
        },
        "method_probability_hashes": per_method_hashes,
        "support_fold_count": LOCAL_FOLD_COUNT,
        "support_fold_plans_exclude_outer_case_and_own_fold": True,
        "controls_may_authorize_primary": False,
        "raw_labels_persisted": False,
        "fresh_evidence": False,
        "promotion_eligible": False,
    }
    route_hash = canonical_hash(body)
    return FinalRouteOutput(
        case_id,
        surface.sample_ids,
        method_probabilities,
        MappingProxyType({**body, "route_hash": route_hash}),
        route_hash,
    )


def build_local_model(
    task: OuterCenterTask,
    store: MappedPhysicalStore,
    journal: DelegatedWorkerLabelJournal,
    capability: LabelCapability,
    labels: ScopedCaseLabels,
    *,
    target_center: str,
    route_case_id: str,
    donor_prior: DonorDirectionalPriorSurface,
    donor_model: DonorActionModel,
    local_ridge_alpha: float,
    role: str,
) -> tuple[LocalResidualModel, tuple[object, ...], str]:
    """Create OOF descriptors with exclusions {route case} union fold(s)."""

    if task.support_fold_ids != tuple(range(LOCAL_FOLD_COUNT)):
        raise GovernanceError("SCALE-BP v2 local support fold inventory drifted.")
    all_cases = store.case_ids(target_center)
    support_cases = tuple(case for case in all_cases if case != route_case_id)
    if labels.case_ids(target_center) != support_cases:
        raise GovernanceError(
            "SCALE-BP v2 local scoring scope must be exactly H-minus-c or J-minus-d."
        )
    fold_assignments = assign_local_support_folds(support_cases)
    sorted_support = tuple(case for case, _ in fold_assignments)
    fold_by_case = dict(fold_assignments)
    fold_cases = {
        fold_id: tuple(
            case for case in sorted_support if fold_by_case[case] == fold_id
        )
        for fold_id in task.support_fold_ids
    }
    local_scope_hash = canonical_hash(
        {
            "schema_version": "scale_bp_v2_route_local_scope_v1",
            "role": role,
            "target_center": target_center,
            "route_case_id": route_case_id,
            "support_case_ids": sorted_support,
            "fold_assignments": tuple(
                (case, fold_by_case[case]) for case in sorted_support
            ),
            "capability_scope_hash": capability.scope_hash,
            "donor_prior_hash": donor_prior.prior_hash,
            "donor_model_hash": donor_model.model_hash,
        }
    )
    scored_rectangles: list[ScoredActionRectangle] = []
    donor_predictions: dict[tuple[str, str], object] = {}
    plan_hashes: list[str] = []
    for support_case in sorted_support:
        fold_id = fold_by_case[support_case]
        excluded_set = {route_case_id, *fold_cases[fold_id]}
        exclusions = tuple(case for case in all_cases if case in excluded_set)
        legal_support = tuple(case for case in all_cases if case not in excluded_set)
        plan = build_protected_route_plan_from_prior(
            store,
            target_center=target_center,
            case_id=support_case,
            support_labels_by_case=labels_for_cases(
                labels, target_center, legal_support
            ),
            donor_prior=donor_prior,
            support_scope_hash=planning_scope_hash(
                role,
                capability,
                target_center=target_center,
                case_id=support_case,
                source_exclusions=donor_prior.source_excluded_centers,
                support_exclusions=exclusions,
                parent_hash=local_scope_hash,
            ),
            support_excluded_case_ids=exclusions,
            outer_held_case_id=route_case_id,
        )
        if (
            plan.outer_held_case_id != route_case_id
            or plan.support_excluded_case_ids != exclusions
            or support_case not in exclusions
        ):
            raise GovernanceError("SCALE-BP v2 local OOF plan escaped its fold.")
        plan_hashes.append(plan.plan_hash)
        surface = reconstruct_case_surface(store, plan)
        rectangle = build_action_rectangle(surface)
        scored_rectangles.append(
            score_scoped_action_rectangle(
                journal,
                capability,
                labels,
                rectangle,
            )
        )
        for cell in rectangle.cells:
            donor_predictions[(support_case, cell.action_id)] = predict_donor_action(
                donor_model,
                action_id=cell.action_id,
                descriptor=cell.evidence.descriptor,
            )
    observations = build_local_residual_observations(
        scored_rectangles,
        donor_predictions,  # type: ignore[arg-type]
        target_center=target_center,
        route_case_id=route_case_id,
        support_scope_hash=local_scope_hash,
    )
    model = fit_route_local_residual(
        observations,
        ridge_alpha=local_ridge_alpha,
    )
    expected_assignments = fold_assignments
    if model.fold_assignments != expected_assignments:
        raise GovernanceError("SCALE-BP v2 local OOF fit assignment drifted.")
    manifest_hash = canonical_hash(
        {
            "schema_version": "scale_bp_v2_local_plan_manifest_v1",
            "role": role,
            "target_center": target_center,
            "route_case_id": route_case_id,
            "local_scope_hash": local_scope_hash,
            "plan_hashes": plan_hashes,
            "fold_assignments": expected_assignments,
            "outer_case_and_own_fold_excluded_before_plan_derivation": True,
        }
    )
    return model, tuple(observations), manifest_hash


def normalize_poison_estimates(
    rectangle: ActionRectangle,
    poisoned: tuple[ActionEstimate, ...],
) -> tuple[ActionEstimate, ...]:
    """Rebind cyclic values to legal target cells without defeating no-op gates."""

    by_action = {row.action_id: row for row in poisoned}
    if set(by_action) != set(ACTION_IDS) or len(by_action) != len(tuple(poisoned)):
        raise GovernanceError("SCALE-BP v2 poisoned action inventory drifted.")
    zero_metric = MetricVector.zeros()
    zero_scale = ScaleVector.zeros()
    output: list[ActionEstimate] = []
    for action_id in ACTION_IDS:
        row = by_action[action_id]
        target_noop = rectangle.cell(action_id).structural_noop
        if target_noop:
            row = replace(
                row,
                mean=zero_metric,
                donor_mean=zero_metric,
                local_correction=zero_metric,
                shrinkage_weights=(0.0, 0.0, 0.0),
                transport_rmse=zero_scale,
                donor_heterogeneity=zero_scale,
                donor_estimator_se=zero_scale,
                local_oof_rmse=zero_scale,
                local_fold_heterogeneity=zero_scale,
                local_estimator_se=zero_scale,
                combined_estimator_se=zero_scale,
                structural_noop=True,
            )
        else:
            row = replace(row, structural_noop=False)
        output.append(row)
    return tuple(output)


def force_exact_p(
    rectangle: ActionRectangle,
    source: RouteDecision,
    *,
    reason: str,
) -> RouteDecision:
    protected = rectangle.cells[0].action.protected_p
    return RouteDecision(
        rectangle.target_center,
        rectangle.case_id,
        P_METHOD_ID,
        None,
        str(reason),
        protected,
        protected,
        source.assessments,
        rectangle.rectangle_hash,
        None,
    )


__all__ = (
    "build_local_model",
    "force_exact_p",
    "normalize_poison_estimates",
    "run_final_route",
)
