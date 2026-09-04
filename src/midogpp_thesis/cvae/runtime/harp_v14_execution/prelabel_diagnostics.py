"""Label-free accounting for HARP v14 routing and abstention reasons."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from .contracts import ActionKind, PrelabelRouteSet


def build_prelabel_diagnostics(routes: PrelabelRouteSet) -> Mapping[str, object]:
    if not isinstance(routes, PrelabelRouteSet):
        raise ProtocolError("HARP v14 prelabel diagnostics require typed routes.")
    reasons = Counter(case.reason for case in routes.cases)
    kinds = Counter(case.selected_kind.value for case in routes.cases)
    directions = Counter(case.direction or "OFF" for case in routes.cases)
    gate_failures: Counter[str] = Counter()
    action_score_count = 0
    accepted_score_count = 0
    for case in routes.cases:
        raw = case.decision_payload.get("failed_gates", ())
        if isinstance(raw, (list, tuple)):
            gate_failures.update(str(value) for value in raw)
        scores = case.decision_payload.get("action_scores", ())
        if isinstance(scores, (list, tuple)):
            action_score_count += len(scores)
            threshold = case.decision_payload.get("acceptance_threshold")
            accepted_score_count += sum(
                isinstance(value, Mapping)
                and type(threshold) in (int, float)
                and float(value.get("acceptance_probability", -1.0))
                >= float(threshold)
                for value in scores
            )
    exact_b = sum(
        case.selected_kind is ActionKind.B
        and case.routed_probabilities.tobytes(order="C")
        == case.baseline_probabilities.tobytes(order="C")
        for case in routes.cases
    )
    body = {
        "schema_version": "midogpp_harp_v14_prelabel_rejection_diagnostics_v1",
        "route_hash": routes.route_hash,
        "case_count": len(routes.cases),
        "selected_kind_counts": dict(sorted(kinds.items())),
        "direction_counts": dict(sorted(directions.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "failed_gate_counts": dict(sorted(gate_failures.items())),
        "action_score_count": action_score_count,
        "scores_meeting_policy_threshold_count": accepted_score_count,
        "per_action_worst_center_certificate_used": False,
        "exact_b_fallback_count": exact_b,
        "routed_case_count": len(routes.cases) - exact_b,
        "evaluation_labels_opened": False,
    }
    return {**body, "diagnostic_hash": canonical_hash(body)}


__all__ = ("build_prelabel_diagnostics",)
