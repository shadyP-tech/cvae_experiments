"""Per-outer exact-top1 prelabel routing for HARP v7."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...routing.source_active_selective_router_v7 import select_exact_top1
from .contracts import (
    ActionKind,
    ArtifactValue,
    LabelFreeOuterMenu,
    PrelabelRouteSet,
    RoutedCase,
)
from .model_adapter import RouterAdmissionState, RouterFitState, TargetEvidenceState
from .production_validation import (
    case_ids,
    require_sha256,
    require_state,
    target_case_blocks,
)


def _decode_hex(values: tuple[str, ...]) -> np.ndarray:
    try:
        raw = b"".join(bytes.fromhex(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("HARP v7 selected probability hex is malformed.") from exc
    output = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
    if not len(output) or not np.isfinite(output).all():
        raise ProtocolError("HARP v7 selected probability vector is nonfinite.")
    return output


def _routed_case(
    *,
    physical_menu: LabelFreeOuterMenu,
    case_id: str,
    target_state: TargetEvidenceState,
    admission_state: RouterAdmissionState,
) -> tuple[RoutedCase, dict[str, object]]:
    effective, prediction = target_state.case(physical_menu.outer_target_id, case_id)
    policy = admission_state.for_outer(physical_menu.outer_target_id)
    decision = select_exact_top1(
        effective,
        prediction,
        policy.admission,
        policy.calibration,
    )
    failed_gates: tuple[str, ...]
    if decision.reason == "EXACT_B_NO_ACTIVE_ACTION":
        failed_gates = ("NO_ACTIVE_PHYSICAL_ACTION",)
    elif decision.reason == "EXACT_B_OUTER_ADMISSION_FAILED":
        failed_gates = tuple(policy.admission.reasons) or ("OUTER_ADMISSION_FAILED",)
    elif decision.reason == "EXACT_B_POLICY_CALIBRATION_FAILED":
        failed_gates = ("WHOLE_POLICY_RISK_COVERAGE_FAILED",)
    elif decision.reason == "EXACT_B_OPPORTUNITY_BELOW_THRESHOLD":
        failed_gates = ("OPPORTUNITY_BELOW_THRESHOLD",)
    elif decision.reason == "EXACT_B_RANK_MARGIN_BELOW_THRESHOLD":
        failed_gates = ("RANK_MARGIN_BELOW_THRESHOLD",)
    else:
        failed_gates = ()
    samples, baseline, uniform = target_case_blocks(physical_menu, case_id)
    selected = _decode_hex(decision.probability_hex)
    if len(selected) != len(samples):
        raise ProtocolError("HARP v7 exact-top1 route/sample geometry drifted.")
    if decision.exact_b_fallback:
        kind = ActionKind.B
        source = None
        direction = None
        components: tuple[np.ndarray, ...] = ()
        action_ids: tuple[str, ...] = ()
        weights: tuple[float, ...] = ()
        shrinkage = 0.0
        if selected.tobytes(order="C") != baseline.tobytes(order="C"):
            raise ProtocolError("HARP v7 exact-B decision changed baseline bytes.")
    else:
        action = next(
            (row for row in effective.actions if row.action_id == decision.selected_action_id),
            None,
        )
        if action is None:
            raise ProtocolError("HARP v7 selected action escaped its effective menu.")
        kind = ActionKind.U if action.action_kind == "U" else ActionKind.HXE
        source = action.candidate_source_id
        direction = action.direction.value
        components = (selected.copy(),)
        action_ids = (action.action_id,)
        weights = (1.0,)
        shrinkage = 1.0
    payload = {
        "route_hash": decision.route_hash,
        "selected_action_id": decision.selected_action_id,
        "exact_b_fallback": decision.exact_b_fallback,
        "reason": decision.reason,
        "prediction_hash": decision.prediction_hash,
        "admission_hash": decision.admission_hash,
        "calibration_hash": decision.calibration_hash,
        "policy_scope": "PER_OUTER_LOCAL",
        "deployed_action": "EXACT_TOP1_PHYSICAL_OR_EXACT_B",
        "failed_gates": failed_gates,
        "opportunity_probability": prediction.opportunity_probability,
        "opportunity_threshold": policy.calibration.opportunity_threshold,
        "rank_margin": prediction.rank_margin,
        "rank_margin_threshold": policy.calibration.rank_margin_threshold,
        "outer_policy_enabled": policy.policy_enabled,
        "evaluation_labels_used": False,
    }
    return (
        RoutedCase(
            outer_target_id=physical_menu.outer_target_id,
            case_id=case_id,
            sample_ids=samples,
            selected_kind=kind,
            selected_source_id=source,
            reason=decision.reason,
            baseline_probabilities=baseline,
            uniform_probabilities=uniform,
            selected_probabilities=selected,
            routed_probabilities=selected.copy(),
            direction=direction,
            shrinkage=shrinkage,
            component_action_ids=action_ids,
            component_weights=weights,
            component_probabilities=components,
            decision_payload=payload,
        ),
        payload,
    )


def build_prelabel_route_set(
    menus: Sequence[LabelFreeOuterMenu],
    target_actions: ArtifactValue,
    fit: ArtifactValue,
    admission: ArtifactValue,
    *,
    config: object,
) -> PrelabelRouteSet:
    """Apply source-frozen local policies with exact-B fallback per case."""

    physical_menus = tuple(menus)
    target_state = require_state(
        target_actions, TargetEvidenceState, role="target evidence"
    )
    require_state(fit, RouterFitState, role="fitted router")
    admission_state = require_state(
        admission, RouterAdmissionState, role="per-outer policy admission"
    )
    routed: list[RoutedCase] = []
    decisions: list[dict[str, object]] = []
    for menu in physical_menus:
        baseline = menu.target_block(ActionKind.B)
        for case_id in case_ids(baseline):
            route, payload = _routed_case(
                physical_menu=menu,
                case_id=case_id,
                target_state=target_state,
                admission_state=admission_state,
            )
            routed.append(route)
            decisions.append(payload)
    model_hash = require_sha256(fit.manifest.get("model_hash"), role="model hash")
    target_hash = require_sha256(
        target_actions.manifest.get("target_action_hash"), role="target action hash"
    )
    admission_hash = require_sha256(
        admission.manifest.get("admission_hash"), role="admission hash"
    )
    body = {
        "schema_version": "midogpp_harp_v7_frozen_source_active_policy_v1",
        "model_hash": model_hash,
        "target_action_hash": target_hash,
        "admission_hash": admission_hash,
        "outer_policy_hashes": {
            row.outer_target_id: {
                "admission_hash": row.admission.admission_hash,
                "calibration_hash": row.calibration.calibration_hash,
                "policy_enabled": row.policy_enabled,
            }
            for row in admission_state.by_outer
        },
        "decision_payloads": decisions,
        "global_kill_switch_used": False,
        "exact_top1_physical_action": True,
        "unevaluated_action_mixture_used": False,
        "exact_b_byte_identical_fallback": True,
        "evaluation_labels_used": False,
    }
    policy_hash = canonical_hash(body)
    return PrelabelRouteSet(
        cases=tuple(sorted(routed, key=lambda row: (row.outer_target_id, row.case_id))),
        policy_hash=policy_hash,
        model_hash=model_hash,
        target_action_hash=target_hash,
    )


__all__ = ("build_prelabel_route_set",)
