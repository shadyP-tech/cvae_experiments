"""Main-process preterminal route construction for authorized P-DCAPS v2.

Support capabilities are consumed here and never placed in a worker DTO.  The
output contains only typed scientific products, immutable hashes, and sealed
action surfaces.  Pseudo responses are opened only after the joint
identity/cyclic surface seal exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ....protocol import ProtocolError
from ..action_surface import (
    ActionResponse,
    RouteActionDraftSurface,
    build_route_action_draft_surface,
)
from ..contracts import RouteKey
from ..endpoint_runtime import build_case_endpoints
from ..identity import DIRECTIONS
from ..inventory import ExpectedRouteInventory
from ..label_firewall import LabelPhase
from ..lifecycle import PDCAPSLabelLifecycle
from ..physical_adapter import PhysicalSurface, candidate_sources
from ..surface_set import SealedActionSurfaceSet
from ..target_local_runtime import (
    CasePosteriorPrediction,
    POSTERIOR_CONTROL_IDS,
    TargetPosteriorModel,
    bind_pseudo_reference,
    build_fingerprint_surface,
    fit_route_posterior,
)
from .identity import canonical_hash, require_sha256
from .route_planning import (
    RoutePlan,
    RoutePlanInventory,
    build_route_plan_inventory,
)
from .viability import CanonicalBankViability, build_canonical_bank_viability


ZERO_DONOR_PRIOR_POLICY_ID = "ZERO_VECTOR_NO_FITTED_PRIOR"


@dataclass(frozen=True)
class ZeroDonorPriorPolicy:
    """Explicit, route-scoped all-zero prior vector used by every endpoint."""

    route_key: RouteKey
    values: tuple[tuple[str, str, float], ...]
    policy_id: str = ZERO_DONOR_PRIOR_POLICY_ID
    policy_hash: str = field(init=False)

    def __post_init__(self) -> None:
        route = self.route_key
        excluded = (
            () if route.surface_role == "target" else (route.outer_center,)
        )
        sources = tuple(
            source
            for source in candidate_sources(route.route_center)
            if source not in excluded
        )
        expected = tuple(
            (source, direction, 0.0)
            for source in sources
            for direction in DIRECTIONS
        )
        values = tuple(
            (str(source), str(direction), float(value))
            for source, direction, value in self.values
        )
        if (
            str(self.policy_id) != ZERO_DONOR_PRIOR_POLICY_ID
            or values != expected
            or any(value != 0.0 for _source, _direction, value in values)
        ):
            raise ProtocolError("P-DCAPS v2 zero donor-prior policy drifted.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "policy_id", ZERO_DONOR_PRIOR_POLICY_ID)
        object.__setattr__(
            self,
            "policy_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v2_zero_donor_prior_policy_v1",
                    "policy_id": ZERO_DONOR_PRIOR_POLICY_ID,
                    "route_key": route.to_payload(),
                    "values": values,
                    "fitted_prior": False,
                    "all_values_exact_zero": True,
                    "labels_used": False,
                }
            ),
        )

    def as_mapping(self) -> dict[tuple[str, str], float]:
        """Return a normal pickle-safe mapping for the endpoint kernel."""

        return {
            (source, direction): value
            for source, direction, value in self.values
        }

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_v2_zero_donor_prior_policy_v1",
            "policy_id": self.policy_id,
            "route_key": self.route_key.to_payload(),
            "values": [list(row) for row in self.values],
            "fitted_prior": False,
            "all_values_exact_zero": True,
            "labels_used": False,
            "policy_hash": self.policy_hash,
        }


def build_zero_donor_prior_policy(plan: RoutePlan) -> ZeroDonorPriorPolicy:
    excluded = set(plan.endpoint_excluded_source_centers)
    values = tuple(
        (source, direction, 0.0)
        for source in candidate_sources(plan.route_key.route_center)
        if source not in excluded
        for direction in DIRECTIONS
    )
    return ZeroDonorPriorPolicy(plan.route_key, values)


@dataclass(frozen=True)
class PosteriorFitRecord:
    control_id: str
    model: TargetPosteriorModel
    prediction: CasePosteriorPrediction
    support_n_positive: int
    support_n_negative: int
    record_hash: str = field(init=False)

    def __post_init__(self) -> None:
        control = str(self.control_id)
        model = self.model
        prediction = self.prediction
        if (
            control not in POSTERIOR_CONTROL_IDS
            or model.control_id != control
            or prediction.control_id != control
            or model.target_center != prediction.target_center
            or model.held_case_id != prediction.held_case_id
            or prediction.model_hash != model.model_hash
            or int(self.support_n_positive) != model.training_n_positive
            or int(self.support_n_negative) != model.training_n_negative
            or min(int(self.support_n_positive), int(self.support_n_negative)) <= 0
        ):
            raise ProtocolError("P-DCAPS v2 posterior fit record drifted.")
        object.__setattr__(self, "control_id", control)
        object.__setattr__(self, "support_n_positive", int(self.support_n_positive))
        object.__setattr__(self, "support_n_negative", int(self.support_n_negative))
        object.__setattr__(
            self,
            "record_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v2_posterior_fit_record_v1",
                    "control_id": control,
                    "model_hash": model.model_hash,
                    "prediction_hash": prediction.prediction_hash,
                    "support_n_positive": int(self.support_n_positive),
                    "support_n_negative": int(self.support_n_negative),
                    "whole_case_excluded": True,
                    "held_case_labels_used": False,
                }
            ),
        )

    @property
    def key(self) -> tuple[str, str, str]:
        return self.control_id, self.model.target_center, self.model.held_case_id

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_v2_posterior_fit_record_v1",
            "control_id": self.control_id,
            "model": self.model.to_payload(),
            "prediction": self.prediction.to_payload(),
            "support_n_positive": self.support_n_positive,
            "support_n_negative": self.support_n_negative,
            "whole_case_excluded": True,
            "held_case_labels_used": False,
            "record_hash": self.record_hash,
        }


@dataclass(frozen=True)
class RouteScienceBinding:
    """Audit binding omitted by the v1 draft DTO, including zero-prior lineage."""

    route_key: RouteKey
    route_plan_hash: str
    endpoint_hash: str
    donor_prior_policy: ZeroDonorPriorPolicy
    bank_viability: CanonicalBankViability
    posterior_prediction_hashes: tuple[tuple[str, str], ...]
    pseudo_reference_hashes: tuple[tuple[str, str | None], ...]
    route_draft_hashes: tuple[tuple[str, str], ...]
    binding_hash: str = field(init=False)

    def __post_init__(self) -> None:
        route = self.route_key
        plan_hash = require_sha256(self.route_plan_hash, "v2 route plan")
        endpoint_hash = require_sha256(self.endpoint_hash, "v2 endpoint")
        posterior = tuple(
            (str(control), require_sha256(value, "v2 posterior prediction"))
            for control, value in self.posterior_prediction_hashes
        )
        pseudo = tuple(
            (
                str(control),
                None if value is None else require_sha256(value, "v2 pseudo reference"),
            )
            for control, value in self.pseudo_reference_hashes
        )
        drafts = tuple(
            (str(control), require_sha256(value, "v2 route draft"))
            for control, value in self.route_draft_hashes
        )
        expected_pseudo_presence = route.surface_role == "pseudo"
        if (
            self.donor_prior_policy.route_key != route
            or self.bank_viability.target_center != route.route_center
            or self.bank_viability.excluded_source_centers
            != (
                () if route.surface_role == "target" else (route.outer_center,)
            )
            or tuple(control for control, _value in posterior)
            != POSTERIOR_CONTROL_IDS
            or tuple(control for control, _value in pseudo)
            != POSTERIOR_CONTROL_IDS
            or tuple(control for control, _value in drafts)
            != POSTERIOR_CONTROL_IDS
            or any((value is not None) != expected_pseudo_presence for _c, value in pseudo)
        ):
            raise ProtocolError("P-DCAPS v2 route science binding drifted.")
        object.__setattr__(self, "route_plan_hash", plan_hash)
        object.__setattr__(self, "endpoint_hash", endpoint_hash)
        object.__setattr__(self, "posterior_prediction_hashes", posterior)
        object.__setattr__(self, "pseudo_reference_hashes", pseudo)
        object.__setattr__(self, "route_draft_hashes", drafts)
        object.__setattr__(
            self,
            "binding_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v2_route_science_binding_v1",
                    "route_key": route.to_payload(),
                    "route_plan_hash": plan_hash,
                    "endpoint_hash": endpoint_hash,
                    "donor_prior_policy_hash": self.donor_prior_policy.policy_hash,
                    "bank_viability_set_hash": self.bank_viability.viability_set_hash,
                    "posterior_prediction_hashes": posterior,
                    "pseudo_reference_hashes": pseudo,
                    "route_draft_hashes": drafts,
                    "held_case_labels_used": False,
                    "target_labels_used": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_v2_route_science_binding_v1",
            "route_key": self.route_key.to_payload(),
            "route_plan_hash": self.route_plan_hash,
            "endpoint_hash": self.endpoint_hash,
            "donor_prior_policy": self.donor_prior_policy.to_payload(),
            "bank_viability": self.bank_viability.to_payload(),
            "posterior_prediction_hashes": [
                list(row) for row in self.posterior_prediction_hashes
            ],
            "pseudo_reference_hashes": [
                list(row) for row in self.pseudo_reference_hashes
            ],
            "route_draft_hashes": [list(row) for row in self.route_draft_hashes],
            "held_case_labels_used": False,
            "target_labels_used": False,
            "binding_hash": self.binding_hash,
        }


@dataclass(frozen=True)
class RouteRuntimeResult:
    expected_inventory: ExpectedRouteInventory
    route_plans: RoutePlanInventory
    physical_surface_hash: str
    posterior_fits: tuple[PosteriorFitRecord, ...]
    route_bindings: tuple[RouteScienceBinding, ...]
    center_sample_orders: tuple[tuple[str, tuple[str, ...]], ...]
    surface_set: SealedActionSurfaceSet
    runtime_hash: str = field(init=False)

    def __post_init__(self) -> None:
        physical_hash = require_sha256(
            self.physical_surface_hash, "v2 route runtime physical surface"
        )
        fits = tuple(self.posterior_fits)
        bindings = tuple(self.route_bindings)
        orders = tuple(
            (str(center), tuple(str(value) for value in sample_ids))
            for center, sample_ids in self.center_sample_orders
        )
        expected_fit_keys = tuple(
            (control, case.center, case.case_id)
            for control in POSTERIOR_CONTROL_IDS
            for case in self.expected_inventory.cases
        )
        if (
            self.route_plans.expected_inventory_hash
            != self.expected_inventory.inventory_hash
            or self.surface_set.expected_inventory_hash
            != self.expected_inventory.inventory_hash
            or self.surface_set.identity.physical_surface_hash != physical_hash
            or tuple(row.key for row in fits) != expected_fit_keys
            or tuple(row.route_key for row in bindings)
            != self.route_plans.route_keys
            or tuple(center for center, _values in orders)
            != self.expected_inventory.centers
            or any(not values or len(values) != len(set(values)) for _c, values in orders)
            or len(self.surface_set.identity.routes) != len(bindings)
            or len(self.surface_set.cyclic.routes) != len(bindings)
        ):
            raise ProtocolError("P-DCAPS v2 route runtime inventory drifted.")
        object.__setattr__(self, "physical_surface_hash", physical_hash)
        object.__setattr__(self, "posterior_fits", fits)
        object.__setattr__(self, "route_bindings", bindings)
        object.__setattr__(self, "center_sample_orders", orders)
        object.__setattr__(
            self,
            "runtime_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v2_route_runtime_result_v1",
                    "expected_inventory_hash": self.expected_inventory.inventory_hash,
                    "route_plan_inventory_hash": (
                        self.route_plans.route_plan_inventory_hash
                    ),
                    "physical_surface_hash": physical_hash,
                    "posterior_fit_record_hashes": tuple(
                        row.record_hash for row in fits
                    ),
                    "route_science_binding_hashes": tuple(
                        row.binding_hash for row in bindings
                    ),
                    "center_sample_orders": orders,
                    "surface_set_seal_hash": self.surface_set.surface_set_seal_hash,
                    "zero_donor_prior_policy": ZERO_DONOR_PRIOR_POLICY_ID,
                    "support_capabilities_retained": False,
                    "pseudo_labels_used": False,
                    "target_labels_used": False,
                }
            ),
        )

    def sample_order(self, center: str) -> tuple[str, ...]:
        try:
            return dict(self.center_sample_orders)[str(center)]
        except KeyError as exc:
            raise ProtocolError("P-DCAPS v2 center sample order is absent.") from exc

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_v2_route_runtime_result_v1",
            "expected_inventory_hash": self.expected_inventory.inventory_hash,
            "route_plan_inventory_hash": self.route_plans.route_plan_inventory_hash,
            "physical_surface_hash": self.physical_surface_hash,
            "posterior_fits": [row.to_payload() for row in self.posterior_fits],
            "route_bindings": [row.to_payload() for row in self.route_bindings],
            "center_sample_orders": [
                [center, list(values)] for center, values in self.center_sample_orders
            ],
            "surface_set": self.surface_set.to_payload(),
            "zero_donor_prior_policy": ZERO_DONOR_PRIOR_POLICY_ID,
            "support_capabilities_retained": False,
            "pseudo_labels_used": False,
            "target_labels_used": False,
            "runtime_hash": self.runtime_hash,
        }


@dataclass(frozen=True)
class PseudoResponseRuntime:
    route_plan_inventory_hash: str
    action_surface_set_seal_hash: str
    opened_route_keys: tuple[RouteKey, ...]
    responses_by_control: tuple[tuple[str, tuple[ActionResponse, ...]], ...]
    runtime_hash: str = field(init=False)

    def __post_init__(self) -> None:
        plan_hash = require_sha256(
            self.route_plan_inventory_hash, "v2 pseudo response route plan"
        )
        surface_hash = require_sha256(
            self.action_surface_set_seal_hash, "v2 pseudo response surface set"
        )
        routes = tuple(self.opened_route_keys)
        rows = tuple(
            (str(control), tuple(responses))
            for control, responses in self.responses_by_control
        )
        if (
            not routes
            or any(route.surface_role != "pseudo" for route in routes)
            or len(set(routes)) != len(routes)
            or tuple(control for control, _responses in rows)
            != POSTERIOR_CONTROL_IDS
            or any(
                response.key.route_key.surface_role != "pseudo"
                or response.key.route_key not in set(routes)
                for _control, responses in rows
                for response in responses
            )
            or any(
                len({response.response_hash for response in responses})
                != len(responses)
                for _control, responses in rows
            )
        ):
            raise ProtocolError("P-DCAPS v2 pseudo response runtime drifted.")
        object.__setattr__(self, "route_plan_inventory_hash", plan_hash)
        object.__setattr__(self, "action_surface_set_seal_hash", surface_hash)
        object.__setattr__(self, "opened_route_keys", routes)
        object.__setattr__(self, "responses_by_control", rows)
        object.__setattr__(
            self,
            "runtime_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v2_pseudo_response_runtime_v1",
                    "route_plan_inventory_hash": plan_hash,
                    "action_surface_set_seal_hash": surface_hash,
                    "opened_route_exclusion_hashes": tuple(
                        route.exclusion_hash for route in routes
                    ),
                    "response_hashes_by_control": tuple(
                        (
                            control,
                            tuple(response.response_hash for response in responses),
                        )
                        for control, responses in rows
                    ),
                    "response_denominators_derived_inside_label_lifecycle": True,
                    "target_labels_used": False,
                }
            ),
        )

    def responses(self, control_id: str) -> tuple[ActionResponse, ...]:
        try:
            return dict(self.responses_by_control)[str(control_id)]
        except KeyError as exc:
            raise ProtocolError("P-DCAPS v2 posterior-control responses are absent.") from exc

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_v2_pseudo_response_runtime_v1",
            "route_plan_inventory_hash": self.route_plan_inventory_hash,
            "action_surface_set_seal_hash": self.action_surface_set_seal_hash,
            "opened_route_exclusion_hashes": [
                route.exclusion_hash for route in self.opened_route_keys
            ],
            "response_hashes_by_control": [
                [control, [response.response_hash for response in responses]]
                for control, responses in self.responses_by_control
            ],
            "response_denominators_derived_inside_label_lifecycle": True,
            "target_labels_used": False,
            "runtime_hash": self.runtime_hash,
        }


def _support_keys(surface: object, held_case_id: str) -> tuple[tuple[str, str, str], ...]:
    center = str(getattr(surface, "center"))
    return tuple(
        (center, str(case_id), str(sample_id))
        for case_id, sample_id in zip(
            getattr(surface, "case_ids"),
            getattr(surface, "sample_ids"),
            strict=True,
        )
        if str(case_id) != str(held_case_id)
    )


def build_route_runtime(
    *,
    physical_surface: PhysicalSurface,
    lifecycle: PDCAPSLabelLifecycle,
    route_plans: RoutePlanInventory | None = None,
) -> RouteRuntimeResult:
    """Fit 2 x case posteriors, build every route, and jointly seal controls."""

    if not isinstance(physical_surface, PhysicalSurface) or not isinstance(
        lifecycle, PDCAPSLabelLifecycle
    ):
        raise ProtocolError("P-DCAPS v2 route runtime requires typed inputs.")
    inventory = lifecycle.expected_inventory
    plans = (
        build_route_plan_inventory(inventory, physical_surface)
        if route_plans is None
        else route_plans
    )
    if (
        plans.expected_inventory_hash != inventory.inventory_hash
        or lifecycle.phase != LabelPhase.LABEL_FREE
    ):
        raise ProtocolError("P-DCAPS v2 route runtime lifecycle drifted.")

    lifecycle.begin_support()
    fingerprints = {
        (control, center): build_fingerprint_surface(
            physical_surface.center(center),
            physical_surface_hash=physical_surface.physical_surface_hash,
            control_id=control,
        )
        for control in POSTERIOR_CONTROL_IDS
        for center in inventory.centers
    }
    fit_by_key: dict[tuple[str, str, str], PosteriorFitRecord] = {}
    draft_by_control_route: dict[
        tuple[str, RouteKey], RouteActionDraftSurface
    ] = {}
    binding_by_route: dict[RouteKey, RouteScienceBinding] = {}
    viability_by_scope: dict[
        tuple[str, tuple[str, ...]], CanonicalBankViability
    ] = {}

    for case in inventory.cases:
        center_surface = physical_surface.center(case.center)
        capability = lifecycle.open_support_labels(
            center=case.center,
            held_case_id=case.case_id,
            keys=_support_keys(center_surface, case.case_id),
        )
        n_positive = sum(value == 1 for value in capability.values)
        n_negative = sum(value == 0 for value in capability.values)
        case_predictions: dict[str, CasePosteriorPrediction] = {}
        for control in POSTERIOR_CONTROL_IDS:
            model, prediction = fit_route_posterior(
                fingerprints[(control, case.center)],
                held_case_id=case.case_id,
                support_capability=capability,
            )
            record = PosteriorFitRecord(
                control,
                model,
                prediction,
                n_positive,
                n_negative,
            )
            fit_by_key[record.key] = record
            case_predictions[control] = prediction

        for plan in plans.plans_for_case(case.center, case.case_id):
            prior = build_zero_donor_prior_policy(plan)
            viability_key = (
                case.center,
                plan.endpoint_excluded_source_centers,
            )
            if viability_key not in viability_by_scope:
                viability_by_scope[viability_key] = build_canonical_bank_viability(
                    case.center,
                    excluded_source_centers=(
                        plan.endpoint_excluded_source_centers
                    ),
                )
            viability = viability_by_scope[viability_key]
            endpoint = build_case_endpoints(
                center_surface,
                physical_surface_hash=physical_surface.physical_surface_hash,
                held_case_id=case.case_id,
                support_capability=capability,
                donor_priors=prior.as_mapping(),
                excluded_source_centers=plan.endpoint_excluded_source_centers,
            )
            posterior_hashes: list[tuple[str, str]] = []
            reference_hashes: list[tuple[str, str | None]] = []
            draft_hashes: list[tuple[str, str]] = []
            for control in POSTERIOR_CONTROL_IDS:
                prediction = case_predictions[control]
                draft = build_route_action_draft_surface(
                    endpoint,
                    prediction,
                    plan.route_key,
                    support_n_positive=n_positive,
                    support_n_negative=n_negative,
                    bank_viability_by_family=viability.as_mapping(),
                )
                draft_by_control_route[(control, plan.route_key)] = draft
                posterior_hashes.append((control, prediction.prediction_hash))
                reference_hashes.append(
                    (
                        control,
                        None
                        if plan.route_key.surface_role == "target"
                        else bind_pseudo_reference(
                            prediction, outer_center=plan.route_key.outer_center
                        ).reference_hash,
                    )
                )
                draft_hashes.append((control, draft.route_draft_hash))
            binding_by_route[plan.route_key] = RouteScienceBinding(
                plan.route_key,
                plan.plan_hash,
                endpoint.endpoint_hash,
                prior,
                viability,
                tuple(posterior_hashes),
                tuple(reference_hashes),
                tuple(draft_hashes),
            )

    identity_routes = tuple(
        draft_by_control_route[(POSTERIOR_CONTROL_IDS[0], plan.route_key)]
        for plan in plans.plans
    )
    cyclic_routes = tuple(
        draft_by_control_route[(POSTERIOR_CONTROL_IDS[1], plan.route_key)]
        for plan in plans.plans
    )
    lifecycle.seal_actions(
        identity_routes,
        cyclic_control_routes=cyclic_routes,
    )
    fits = tuple(
        fit_by_key[(control, case.center, case.case_id)]
        for control in POSTERIOR_CONTROL_IDS
        for case in inventory.cases
    )
    bindings = tuple(binding_by_route[plan.route_key] for plan in plans.plans)
    orders = tuple(
        (center, physical_surface.center(center).sample_ids)
        for center in inventory.centers
    )
    return RouteRuntimeResult(
        inventory,
        plans,
        physical_surface.physical_surface_hash,
        fits,
        bindings,
        orders,
        lifecycle.action_surface_set,
    )


def open_all_pseudo_responses(
    lifecycle: PDCAPSLabelLifecycle,
    route_runtime: RouteRuntimeResult,
) -> PseudoResponseRuntime:
    """Open every pseudo H/J/d response with lifecycle-derived denominators."""

    if (
        lifecycle.expected_inventory.inventory_hash
        != route_runtime.expected_inventory.inventory_hash
        or lifecycle.action_surface_set.surface_set_seal_hash
        != route_runtime.surface_set.surface_set_seal_hash
        or lifecycle.phase != LabelPhase.ACTION_SURFACE_SEALED
    ):
        raise ProtocolError("P-DCAPS v2 pseudo response lifecycle drifted.")
    lifecycle.begin_pseudo_responses()
    opened: list[RouteKey] = []
    by_control: dict[str, list[ActionResponse]] = {
        control: [] for control in POSTERIOR_CONTROL_IDS
    }
    for plan in route_runtime.route_plans.plans:
        if plan.route_key.surface_role != "pseudo":
            continue
        rows = lifecycle.open_pseudo_control_action_responses_derived(
            plan.route_key
        )
        if tuple(control for control, _responses in rows) != POSTERIOR_CONTROL_IDS:
            raise ProtocolError("P-DCAPS v2 pseudo response control drifted.")
        for control, responses in rows:
            by_control[control].extend(responses)
        opened.append(plan.route_key)
    return PseudoResponseRuntime(
        route_runtime.route_plans.route_plan_inventory_hash,
        route_runtime.surface_set.surface_set_seal_hash,
        tuple(opened),
        tuple((control, tuple(by_control[control])) for control in POSTERIOR_CONTROL_IDS),
    )


__all__ = (
    "PosteriorFitRecord",
    "PseudoResponseRuntime",
    "RouteRuntimeResult",
    "RouteScienceBinding",
    "ZERO_DONOR_PRIOR_POLICY_ID",
    "ZeroDonorPriorPolicy",
    "build_route_runtime",
    "build_zero_donor_prior_policy",
    "open_all_pseudo_responses",
)
