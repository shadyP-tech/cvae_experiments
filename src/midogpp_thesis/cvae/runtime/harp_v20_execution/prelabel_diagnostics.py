"""Label-free accounting for the HARP v20 pooled selected-policy routes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from .contracts import ActionKind, PrelabelRouteSet


def build_prelabel_diagnostics(routes: PrelabelRouteSet) -> Mapping[str, object]:
    if not isinstance(routes, PrelabelRouteSet):
        raise ProtocolError("HARP v20 prelabel diagnostics require typed routes.")
    reasons = Counter(case.reason for case in routes.cases)
    kinds = Counter(case.selected_kind.value for case in routes.cases)
    directions = Counter(case.direction or "OFF" for case in routes.cases)
    recipe_kinds = Counter(str(case.recipe_kind) for case in routes.cases)
    component_counts = Counter(len(case.component_action_ids) for case in routes.cases)
    admission_states: set[bool] = set()
    selection_statuses: Counter[str] = Counter()
    probability_statuses: Counter[str] = Counter()
    prediction_statuses: Counter[str] = Counter()
    utility_statuses: Counter[str] = Counter()
    for case in routes.cases:
        payload = case.decision_payload
        if (
            payload.get("router_hash") != routes.policy_hash
            or type(payload.get("support_policy_admitted")) is not bool
        ):
            raise ProtocolError("HARP v20 route lacks its pooled admission state.")
        admission_states.add(bool(payload["support_policy_admitted"]))
        for field, counter in (
            ("selection_status", selection_statuses),
            ("probability_status", probability_statuses),
            ("prediction_status", prediction_statuses),
            ("utility_status", utility_statuses),
        ):
            value = payload.get(field)
            if type(value) is not str or not value:
                raise ProtocolError(f"HARP v20 route lacks {field}.")
            counter[value] += 1
    if len(admission_states) != 1:
        raise ProtocolError("HARP v20 pooled admission changed across target cases.")
    exact_b = sum(
        case.selected_kind is ActionKind.B
        and case.routed_probabilities.tobytes(order="C")
        == case.baseline_probabilities.tobytes(order="C")
        for case in routes.cases
    )
    probability_changed = sum(
        case.routed_probabilities.tobytes(order="C")
        != case.baseline_probabilities.tobytes(order="C")
        for case in routes.cases
    )
    prediction_changed = sum(
        bool(case.decision_payload.get("prediction_changed"))
        for case in routes.cases
    )
    body = {
        "schema_version": "midogpp_harp_v20_prelabel_pooled_diagnostics_v1",
        "route_hash": routes.route_hash,
        "policy_hash": routes.policy_hash,
        "case_count": len(routes.cases),
        "selected_kind_counts": dict(sorted(kinds.items())),
        "recipe_kind_counts": dict(sorted(recipe_kinds.items())),
        "direction_counts": dict(sorted(directions.items())),
        "selected_component_count_distribution": {
            str(key): value for key, value in sorted(component_counts.items())
        },
        "reason_counts": dict(sorted(reasons.items())),
        "selection_status_counts": dict(sorted(selection_statuses.items())),
        "probability_status_counts": dict(sorted(probability_statuses.items())),
        "prediction_status_counts": dict(sorted(prediction_statuses.items())),
        "utility_status_counts": dict(sorted(utility_statuses.items())),
        "pooled_policy_count": 1,
        "pooled_policy_admitted": next(iter(admission_states)),
        "exact_b_fallback_count": exact_b,
        "route_selected_count": len(routes.cases) - exact_b,
        "probability_changed_count": probability_changed,
        "prediction_changed_count": prediction_changed,
        "utility_success_count": None,
        "route_selected_is_probability_changed": False,
        "probability_changed_is_prediction_changed": False,
        "all_k_lambda_probability_matrices_persisted": False,
        "evaluation_labels_opened": False,
    }
    return {**body, "diagnostic_hash": canonical_hash(body)}


__all__ = ("build_prelabel_diagnostics",)
