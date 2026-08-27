"""Pure SCALE-BP case-route composition root.

This module connects the independent scientific kernels.  It accepts only
already scoped aggregate support records and sealed label-free endpoint
receipts; it owns no label capability and performs no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .action_geometry import canonical_probabilities, probability_hash
from .composition import ComposedAction, compose_selection
from .donor_prior import DonorPriorPrediction
from .empirical_bayes import EmpiricalBayesEstimate, shrink_action_value
from .hashing import canonical_hash
from .influence.contracts import ActionDescriptor
from .local_residual.contracts import (
    LocalCrossfitResult,
    LocalResidualModel,
    LocalResidualRecord,
)
from .local_residual.crossfit import (
    crossfit_local_residuals,
    fit_local_residual_model,
    predict_local_residual,
)
from .physical.endpoint_surface import EndpointProjectionReceipt
from .protocol import ProtocolError
from .replay_scope import DonorScope, FinalDonorScope, PseudoReplayScope
from .selection import ActionCandidate, SelectionDecision, select_case_actions
from .support_calibration import LocalActionCalibration, calibrate_local_action
from .support_folds import SupportFoldPlan
from .uncertainty import (
    SelectionAwareRadius,
    build_action_envelope,
    fit_selection_aware_radius,
)


_DONOR_FIT_ROLES = frozenset({"FINAL_H_C", "PSEUDO_H_J_D"})


@dataclass(frozen=True, slots=True)
class RouteActionInput:
    endpoint_projection: EndpointProjectionReceipt
    descriptor: ActionDescriptor
    donor_prediction: DonorPriorPrediction
    input_hash: str = field(init=False)

    def __post_init__(self) -> None:
        projection = self.endpoint_projection.projection
        endpoint = self.endpoint_projection.endpoint
        if (
            endpoint.case_id != self.descriptor.case_id
            or endpoint.action_id != self.descriptor.action_id
            or projection.action_id != self.descriptor.action_id
            or projection.is_exact_p
            or self.descriptor.baseline_probability_hash
            != projection.baseline_probability_hash
            or self.descriptor.action_probability_hash
            != projection.projected_probability_hash
            or self.descriptor.endpoint_probability_hash
            != endpoint.endpoint_probability_hash
            or self.donor_prediction.descriptor_hash != self.descriptor.descriptor_hash
            or self.donor_prediction.fit_role not in _DONOR_FIT_ROLES
        ):
            raise ProtocolError("SCALE-BP route-action input lineage drifted.")
        object.__setattr__(
            self,
            "input_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_route_action_input_v1",
                    "endpoint_projection_receipt_hash": (
                        self.endpoint_projection.receipt_hash
                    ),
                    "descriptor_hash": self.descriptor.descriptor_hash,
                    "donor_prediction_hash": self.donor_prediction.prediction_hash,
                }
            ),
        )

    @property
    def action_id(self) -> str:
        return self.descriptor.action_id


@dataclass(frozen=True, slots=True)
class CaseRouteRequest:
    case_id: str
    route_scope: DonorScope
    portfolio_probabilities: tuple[float, ...]
    support_plan: SupportFoldPlan
    support_records: tuple[LocalResidualRecord, ...]
    action_inputs: tuple[RouteActionInput, ...]
    request_hash: str = field(init=False)

    def __post_init__(self) -> None:
        case = str(self.case_id)
        portfolio = canonical_probabilities(self.portfolio_probabilities)
        records = tuple(self.support_records)
        actions = tuple(sorted(self.action_inputs, key=lambda row: row.action_id))
        baseline_hash = probability_hash(portfolio)
        scope = self.route_scope
        plan = self.support_plan
        if not isinstance(scope, (FinalDonorScope, PseudoReplayScope)) or not isinstance(
            plan, SupportFoldPlan
        ):
            raise ProtocolError("SCALE-BP case-route request scope drifted.")
        witness = scope.route_witness
        witness_inventory = witness.identity_inventory
        witness_case_inventory = witness_inventory.case_inventory
        expected_support_cases = set(witness.support_case_ids)
        if (
            not case
            or scope.held_case_id != case
            or scope.prediction_center != plan.held_center
            or witness.witness_hash != plan.route_witness.witness_hash
            or scope.route_scope_hash != plan.route_scope_hash
            or witness_inventory.inventory_hash
            != plan.route_witness.identity_inventory.inventory_hash
            or witness.evaluation_binding.binding_hash
            != plan.route_witness.evaluation_binding.binding_hash
            or witness.support_sample_key_hash
            != plan.route_witness.support_sample_key_hash
            or witness_case_inventory.inventory_hash
            != scope.case_inventory.inventory_hash
            or witness_case_inventory.cache_content_hash
            != scope.case_inventory.cache_content_hash
            or witness_case_inventory.row_order_hash
            != scope.case_inventory.row_order_hash
            or witness_case_inventory.manifest_hash
            != scope.case_inventory.manifest_hash
            or witness.evaluation_binding.row_count != len(portfolio)
            or plan.held_case_id != case
            or {member.case_id for member in plan.members}
            != expected_support_cases
            or any(not isinstance(row, LocalResidualRecord) for row in records)
            or any(not isinstance(row, RouteActionInput) for row in actions)
            or len({row.action_id for row in actions}) != len(actions)
            or any(
                row.descriptor.case_id != case
                or row.endpoint_projection.endpoint.target_center
                != scope.prediction_center
                or row.endpoint_projection.endpoint.cache_content_hash
                != scope.case_inventory.cache_content_hash
                or row.endpoint_projection.endpoint.row_order_hash
                != scope.case_inventory.row_order_hash
                or row.endpoint_projection.projection.baseline_probability_hash
                != baseline_hash
                or row.donor_prediction.scope_hash != scope.scope_hash
                or row.donor_prediction.fit_role != scope.fit_role
                for row in actions
            )
        ):
            raise ProtocolError("SCALE-BP case-route request lineage drifted.")
        portfolio_values = tuple(float(value) for value in portfolio)
        payload = {
            "schema_version": "scale_bp_case_route_request_v2",
            "case_id": case,
            "donor_scope_hash": scope.scope_hash,
            "donor_fit_role": scope.fit_role,
            "prediction_center": scope.prediction_center,
            "route_scope_hash": plan.route_scope_hash,
            "route_identity_inventory_hash": (
                plan.route_witness.identity_inventory.inventory_hash
            ),
            "evaluation_sample_key_hash": (
                plan.route_witness.evaluation_binding.sample_key_hash
            ),
            "support_sample_key_hash": (
                plan.route_witness.support_sample_key_hash
            ),
            "support_plan_hash": plan.plan_hash,
            "support_record_hashes": tuple(sorted(row.record_hash for row in records)),
            "action_input_hashes": tuple(row.input_hash for row in actions),
            "baseline_probability_hash": baseline_hash,
            "raw_labels_present": False,
        }
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "portfolio_probabilities", portfolio_values)
        object.__setattr__(self, "support_records", records)
        object.__setattr__(self, "action_inputs", actions)
        object.__setattr__(self, "request_hash", canonical_hash(payload))


@dataclass(frozen=True, slots=True)
class CaseRouteResult:
    request_hash: str
    case_id: str
    crossfit: LocalCrossfitResult | None
    final_local_model: LocalResidualModel | None
    selection_radius: SelectionAwareRadius | None
    calibrations: tuple[LocalActionCalibration, ...]
    estimates: tuple[EmpiricalBayesEstimate, ...]
    candidates: tuple[ActionCandidate, ...]
    decision: SelectionDecision
    boundary_action: ComposedAction
    full_endpoint_sensitivity: ComposedAction
    route_hash: str = field(init=False)

    def __post_init__(self) -> None:
        calibrations = tuple(self.calibrations)
        estimates = tuple(self.estimates)
        candidates = tuple(self.candidates)
        if (
            not self.case_id
            or tuple(row.action_id for row in calibrations)
            != tuple(row.action_id for row in estimates)
            or tuple(row.action_id for row in estimates)
            != tuple(row.action_id for row in candidates)
            or self.decision.case_id != self.case_id
            or self.boundary_action.case_id != self.case_id
            or self.full_endpoint_sensitivity.case_id != self.case_id
            or self.boundary_action.decision_hash != self.decision.decision_hash
            or self.full_endpoint_sensitivity.decision_hash
            != self.decision.decision_hash
            or (
                not candidates
                and any(
                    value is not None
                    for value in (
                        self.crossfit,
                        self.final_local_model,
                        self.selection_radius,
                    )
                )
            )
            or (
                candidates
                and any(
                    value is None
                    for value in (
                        self.crossfit,
                        self.final_local_model,
                        self.selection_radius,
                    )
                )
            )
        ):
            raise ProtocolError("SCALE-BP case-route result drifted.")
        payload = {
            "schema_version": "scale_bp_case_route_result_v1",
            "request_hash": self.request_hash,
            "case_id": self.case_id,
            "crossfit_hash": None if self.crossfit is None else self.crossfit.crossfit_hash,
            "final_local_model_hash": (
                None if self.final_local_model is None else self.final_local_model.model_hash
            ),
            "selection_radius_hash": (
                None if self.selection_radius is None else self.selection_radius.radius_hash
            ),
            "calibration_hashes": tuple(row.calibration_hash for row in calibrations),
            "estimate_hashes": tuple(row.estimate_hash for row in estimates),
            "candidate_hashes": tuple(row.candidate_hash for row in candidates),
            "decision_hash": self.decision.decision_hash,
            "boundary_composition_hash": self.boundary_action.composition_hash,
            "full_endpoint_composition_hash": (
                self.full_endpoint_sensitivity.composition_hash
            ),
            "exact_p_fallback": self.decision.is_exact_p,
            "raw_labels_persisted": False,
        }
        object.__setattr__(self, "calibrations", calibrations)
        object.__setattr__(self, "estimates", estimates)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "route_hash", canonical_hash(payload))


def build_case_route(request: CaseRouteRequest) -> CaseRouteResult:
    """Cross-fit, shrink, envelope, select, and compose one held case."""

    if not request.action_inputs:
        decision = select_case_actions(
            case_id=request.case_id,
            baseline_probability_hash=probability_hash(request.portfolio_probabilities),
            candidates=(),
        )
        boundary = compose_selection(
            request.portfolio_probabilities, (), decision, mode="boundary"
        )
        full = compose_selection(
            request.portfolio_probabilities, (), decision, mode="full_endpoint"
        )
        return CaseRouteResult(
            request.request_hash,
            request.case_id,
            None,
            None,
            None,
            (),
            (),
            (),
            decision,
            boundary,
            full,
        )

    crossfit = crossfit_local_residuals(
        request.support_records, request.support_plan
    )
    final_model = fit_local_residual_model(request.support_records)
    if final_model.route_scope_hash != request.support_plan.route_scope_hash:
        raise ProtocolError("SCALE-BP final local model escaped its route scope.")
    radius = fit_selection_aware_radius(crossfit)
    calibrations: list[LocalActionCalibration] = []
    estimates: list[EmpiricalBayesEstimate] = []
    candidates: list[ActionCandidate] = []
    for row in request.action_inputs:
        calibration = calibrate_local_action(
            row.descriptor,
            records=request.support_records,
            crossfit=crossfit,
            final_model=final_model,
        )
        local = predict_local_residual(final_model, row.descriptor)
        donor_se = row.donor_prediction.between_center_standard_error
        estimate = shrink_action_value(
            action_id=row.action_id,
            donor_metrics=row.donor_prediction.mean,
            local_residual=local,
            donor_standard_error=donor_se,
            local_standard_error=calibration.local_standard_error,
            between_center_variance=tuple(value * value for value in donor_se.as_tuple()),
        )
        candidate = ActionCandidate(
            case_id=request.case_id,
            projection=row.endpoint_projection.projection,
            envelope=build_action_envelope(estimate, radius),
            within_support=calibration.within_support,
            bank_viable=calibration.bank_viable,
        )
        calibrations.append(calibration)
        estimates.append(estimate)
        candidates.append(candidate)
    ordered_candidates = tuple(candidates)
    decision = select_case_actions(
        case_id=request.case_id,
        baseline_probability_hash=probability_hash(request.portfolio_probabilities),
        candidates=ordered_candidates,
    )
    boundary = compose_selection(
        request.portfolio_probabilities,
        ordered_candidates,
        decision,
        mode="boundary",
    )
    full = compose_selection(
        request.portfolio_probabilities,
        ordered_candidates,
        decision,
        mode="full_endpoint",
    )
    return CaseRouteResult(
        request.request_hash,
        request.case_id,
        crossfit,
        final_model,
        radius,
        tuple(calibrations),
        tuple(estimates),
        ordered_candidates,
        decision,
        boundary,
        full,
    )


__all__ = (
    "CaseRouteRequest",
    "CaseRouteResult",
    "RouteActionInput",
    "build_case_route",
)
