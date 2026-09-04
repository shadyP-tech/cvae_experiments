"""Label-free accounting for HARP v15 routing and abstention reasons.

The v15 policy exposes a hierarchical certificate trace rather than the flat
``action_scores``/``failed_gates`` payload used by older HARP routers.  Keep
the report aligned with that public contract so it remains useful when the
terminal result contains only exact-B fallbacks.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from .contracts import ActionKind, PrelabelRouteSet


def build_prelabel_diagnostics(routes: PrelabelRouteSet) -> Mapping[str, object]:
    if not isinstance(routes, PrelabelRouteSet):
        raise ProtocolError("HARP v15 prelabel diagnostics require typed routes.")
    reasons = Counter(case.reason for case in routes.cases)
    kinds = Counter(case.selected_kind.value for case in routes.cases)
    directions = Counter(case.direction or "OFF" for case in routes.cases)
    selected_directions: Counter[str] = Counter()
    selected_families: Counter[str] = Counter()
    eligible_action_count = 0
    cases_with_eligible_actions = 0
    selected_certificate_count = 0
    router_admission_by_hash: dict[str, bool] = {}
    for case in routes.cases:
        trace = case.decision_payload.get("hierarchy_trace", {})
        if not isinstance(trace, Mapping):
            raise ProtocolError("HARP v15 route lacks a typed hierarchy trace.")
        eligible = trace.get("eligible_action_ids", ())
        if not isinstance(eligible, (list, tuple)):
            raise ProtocolError("HARP v15 route eligibility trace is malformed.")
        eligible_action_count += len(eligible)
        cases_with_eligible_actions += bool(eligible)
        direction = trace.get("selected_direction")
        family = trace.get("selected_family")
        if direction is not None:
            selected_directions[str(direction)] += 1
        if family is not None:
            selected_families[str(family)] += 1
        selected_certificate_count += (
            case.decision_payload.get("selected_certificate_hash") is not None
        )
        router_hash = case.decision_payload.get("router_hash")
        policy_admitted = case.decision_payload.get("support_policy_admitted")
        if not isinstance(router_hash, str) or type(policy_admitted) is not bool:
            raise ProtocolError("HARP v15 route lacks target-local admission state.")
        prior = router_admission_by_hash.setdefault(router_hash, policy_admitted)
        if prior is not policy_admitted:
            raise ProtocolError("HARP v15 target-local admission state drifted.")
    exact_b = sum(
        case.selected_kind is ActionKind.B
        and case.routed_probabilities.tobytes(order="C")
        == case.baseline_probabilities.tobytes(order="C")
        for case in routes.cases
    )
    body = {
        "schema_version": "midogpp_harp_v15_prelabel_rejection_diagnostics_v2",
        "route_hash": routes.route_hash,
        "case_count": len(routes.cases),
        "selected_kind_counts": dict(sorted(kinds.items())),
        "direction_counts": dict(sorted(directions.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "hierarchy_selected_direction_counts": dict(sorted(selected_directions.items())),
        "hierarchy_selected_family_counts": dict(sorted(selected_families.items())),
        "eligible_action_count": eligible_action_count,
        "cases_with_eligible_actions": cases_with_eligible_actions,
        "selected_certificate_count": selected_certificate_count,
        "distinct_target_local_router_count": len(router_admission_by_hash),
        "admitted_target_local_router_count": sum(router_admission_by_hash.values()),
        "cases_with_admitted_router_count": sum(
            case.decision_payload["support_policy_admitted"] for case in routes.cases
        ),
        "target_local_support_certificate_used": True,
        "per_action_worst_center_certificate_used": False,
        "exact_b_fallback_count": exact_b,
        "routed_case_count": len(routes.cases) - exact_b,
        "evaluation_labels_opened": False,
    }
    return {**body, "diagnostic_hash": canonical_hash(body)}


__all__ = ("build_prelabel_diagnostics",)
