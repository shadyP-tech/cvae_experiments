"""Prelabel route selection, composition, and policy sealing for HARP v6."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.compatibility_conditioned_directional_router import (
    ActionKind as RouterActionKind,
    RoutingDecision,
    compose_route,
    select_baseline_anchored_route,
)
from ...routing.harp_protocol import canonical_hash
from .contracts import (
    ActionKind,
    ArtifactValue,
    LabelFreeOuterMenu,
    PrelabelRouteSet,
    RoutedCase,
    compose_directional_soft_probability,
)
from .model_adapter import RouterAdmissionState, RouterFitState, TargetEvidenceState
from .production_validation import (
    case_ids,
    decode_cells,
    float32_cells,
    require_sha256,
    require_state,
    target_case_blocks,
)


def _disabled_global_decision(
    *,
    outer_target_id: str,
    case_id: str,
    admission_hash: str,
) -> RoutingDecision:
    return RoutingDecision(
        outer_target_id=outer_target_id,
        case_id=case_id,
        enabled=False,
        selected_direction=None,
        selected_action_ids=(),
        selected_weights=(),
        mixture_lambda=0.0,
        reason="GLOBAL_SOURCE_ONLY_LEARNABILITY_ADMISSION_FAILED",
        admission_hash=admission_hash,
        evidence_hashes=(),
    )


def _compose_routed_case(
    *,
    menu: LabelFreeOuterMenu,
    case_id: str,
    decision: RoutingDecision,
    target_state: TargetEvidenceState,
) -> tuple[RoutedCase, dict[str, object]]:
    scoped_actions = target_state.case_actions(menu.outer_target_id, case_id)
    samples, baseline, uniform = target_case_blocks(menu, case_id)
    composition = compose_route(
        decision=decision,
        baseline_sample_ids=samples,
        baseline_probability_bytes=float32_cells(baseline),
        actions=scoped_actions,
    )
    if not decision.enabled:
        runtime_kind = ActionKind.B
        source = None
        direction = None
        components: tuple[np.ndarray, ...] = ()
        selected = baseline.copy()
        output = baseline.copy()
    else:
        selected_actions = tuple(
            next(
                row
                for row in scoped_actions
                if row.feature.action_id == action_id
            )
            for action_id in decision.selected_action_ids
        )
        router_kind = selected_actions[0].feature.action_kind
        if any(row.feature.action_kind is not router_kind for row in selected_actions):
            raise ProtocolError("HARP v6 route mixed U and HXE components.")
        runtime_kind = (
            ActionKind.U if router_kind is RouterActionKind.U else ActionKind.HXE
        )
        source = (
            None
            if runtime_kind is ActionKind.U
            else selected_actions[0].feature.candidate_source_id
        )
        direction = decision.selected_direction.value
        components = tuple(
            decode_cells(row.probability_bytes) for row in selected_actions
        )
        selected, output = compose_directional_soft_probability(
            baseline,
            components,
            decision.selected_weights,
            direction=direction,
            shrinkage=decision.mixture_lambda,
        )
        science_output = decode_cells(composition.output_probability_bytes)
        if output.tobytes(order="C") != science_output.tobytes(order="C"):
            raise ProtocolError(
                "HARP v6 runtime and neutral composition bytes disagree."
            )
    decision_payload = {
        "decision_hash": decision.decision_hash,
        "enabled": decision.enabled,
        "selected_direction": direction,
        "selected_action_ids": list(decision.selected_action_ids),
        "selected_weights": list(decision.selected_weights),
        "mixture_lambda": decision.mixture_lambda,
        "admission_hash": decision.admission_hash,
        "evidence_hashes": list(decision.evidence_hashes),
        "composition_hash": composition.receipt.composition_hash,
        "exact_baseline_fallback": composition.receipt.exact_baseline_fallback,
        "evaluation_labels_used": False,
    }
    routed = RoutedCase(
        outer_target_id=menu.outer_target_id,
        case_id=case_id,
        sample_ids=samples,
        selected_kind=runtime_kind,
        selected_source_id=source,
        reason=decision.reason,
        baseline_probabilities=baseline,
        uniform_probabilities=uniform,
        selected_probabilities=selected,
        routed_probabilities=output,
        direction=direction,
        shrinkage=decision.mixture_lambda,
        component_action_ids=decision.selected_action_ids,
        component_weights=decision.selected_weights,
        component_probabilities=components,
        decision_payload=decision_payload,
    )
    return routed, decision_payload


def build_prelabel_route_set(
    menus: Sequence[LabelFreeOuterMenu],
    target_actions: ArtifactValue,
    fit: ArtifactValue,
    admission: ArtifactValue,
    *,
    config: object,
    select_fn: Callable[..., RoutingDecision] = select_baseline_anchored_route,
) -> PrelabelRouteSet:
    """Select case routes, verify exact composition, and freeze the policy hash."""

    menu_rows = tuple(menus)
    target_state = require_state(
        target_actions, TargetEvidenceState, role="target evidence"
    )
    require_state(fit, RouterFitState, role="fitted router")
    admission_state = require_state(
        admission, RouterAdmissionState, role="learnability admission"
    )
    model = getattr(config, "model")
    policy = model["policy"]
    routed: list[RoutedCase] = []
    decision_rows: list[dict[str, object]] = []
    for menu in menu_rows:
        baseline_block = menu.target_block(ActionKind.B)
        for case_id in case_ids(baseline_block):
            outer_admission = admission_state.for_outer(menu.outer_target_id)
            evidence = target_state.case_evidence(menu.outer_target_id, case_id)
            if not admission_state.router_admitted:
                decision = _disabled_global_decision(
                    outer_target_id=menu.outer_target_id,
                    case_id=case_id,
                    admission_hash=outer_admission.admission_hash,
                )
            else:
                decision = select_fn(
                    evidence,
                    admission=outer_admission,
                    outer_target_id=menu.outer_target_id,
                    case_id=case_id,
                    top_k=int(model["soft_top_k"]),
                    mixture_lambda=float(model["soft_mixture_lambda"]),
                    opportunity_threshold=float(
                        model["opportunity_probability_threshold"]
                    ),
                    temperature=float(model["softmax_temperature"]),
                )
            route, payload = _compose_routed_case(
                menu=menu,
                case_id=case_id,
                decision=decision,
                target_state=target_state,
            )
            decision_rows.append(payload)
            routed.append(route)
    model_hash = require_sha256(fit.manifest.get("model_hash"), role="model hash")
    target_hash = require_sha256(
        target_actions.manifest.get("target_action_hash"), role="target action hash"
    )
    admission_hash = require_sha256(
        admission.manifest.get("admission_hash"), role="admission hash"
    )
    policy_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v6_frozen_directional_policy_v1",
            "model_hash": model_hash,
            "target_action_hash": target_hash,
            "admission_hash": admission_hash,
            "router_admitted": admission_state.router_admitted,
            "soft_top_k": model["soft_top_k"],
            "soft_mixture_lambda": model["soft_mixture_lambda"],
            "softmax_temperature": model["softmax_temperature"],
            "opportunity_probability_threshold": model[
                "opportunity_probability_threshold"
            ],
            "endpoint_thresholds": dict(policy),
            "decision_payloads": decision_rows,
            "evaluation_labels_used": False,
        }
    )
    return PrelabelRouteSet(
        cases=tuple(sorted(routed, key=lambda row: (row.outer_target_id, row.case_id))),
        policy_hash=policy_hash,
        model_hash=model_hash,
        target_action_hash=target_hash,
    )


__all__ = ("build_prelabel_route_set",)
