"""Typed, fixed-menu terminal method controls for P-DCAPS.

The public builders accept scientific source DTOs, never caller-selected action
hashes.  Each decision therefore derives its composition inventory from the
sealed result that produced it.  The cyclic poison is additionally bound to a
distinct cyclic-posterior result while the protected, primary, ablation, and
legacy methods are restricted to the identity-posterior result.

These controls are terminal consumed-test diagnostics.  Constructing or
composing one does not authorize execution, promotion, deployment, or a fresh
routing claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ...protocol import ProtocolError
from .action_surface import SealedRouteActionSurface
from .composition import ComposedCenterPrediction, compose_center_prediction
from .engine import OuterActionPolicyResult
from .identity import (
    ACTION_ONLY_METHOD_ID,
    CYCLIC_METHOD_ID,
    LEGACY_METHOD_ID,
    METHOD_MENU,
    POLICY_ONLY_METHOD_ID,
    PRIMARY_METHOD_ID,
    PUBLICATION_STATUS,
    P_METHOD_ID,
    TERMINAL_DECISION,
    canonical_hash,
    require_sha256,
)
from .legacy_control import LegacyControlSeal, resolve_legacy_control
from .routing import (
    AuthorizedOuterPolicy,
    authorize_primary_policy,
    build_admission_from_pseudo_policies,
)
from .surface_set import SealedActionSurfaceSet


IDENTITY_POSTERIOR_CONTROL_ID = "IDENTITY"
CYCLIC_POSTERIOR_CONTROL_ID = "WITHIN_CASE_CYCLIC_SHIFT"


_SELECTION_SEMANTICS = {
    P_METHOD_ID: "EXACT_P_PROTECTED",
    PRIMARY_METHOD_ID: "ACTION_THEN_POLICY_PREFIX_THEN_ADMISSION_H",
    ACTION_ONLY_METHOD_ID: "ACTION_SELECTION_ONLY_NO_POLICY_OR_ADMISSION_H",
    POLICY_ONLY_METHOD_ID: "ACTION_THEN_POLICY_PREFIX_NO_ADMISSION_H",
    LEGACY_METHOD_ID: "LEGACY_CENTER_POOLED_PREFIX_NO_ADMISSION_H",
    CYCLIC_METHOD_ID: (
        "CYCLIC_POSTERIOR_ACTION_THEN_POLICY_PREFIX_THEN_ADMISSION_H"
    ),
}


def _require_result(value: object, *, role: str) -> OuterActionPolicyResult:
    if not isinstance(value, OuterActionPolicyResult):
        raise ProtocolError(f"P-DCAPS {role} is not a sealed outer result.")
    return value


def _decision_route_keys(
    result: OuterActionPolicyResult,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    return (
        tuple(row.route_key for row in result.target_action_decisions),
        tuple(row.route_key for row in result.pseudo_action_decisions),
    )


def _require_identity_result(result: OuterActionPolicyResult) -> None:
    if result.posterior_control_id != IDENTITY_POSTERIOR_CONTROL_ID:
        raise ProtocolError(
            "P-DCAPS non-cyclic method requires the identity posterior control."
        )


def _require_cyclic_pair(
    identity: OuterActionPolicyResult,
    cyclic: OuterActionPolicyResult,
) -> None:
    _require_identity_result(identity)
    if (
        cyclic.posterior_control_id != CYCLIC_POSTERIOR_CONTROL_ID
        or cyclic.outer_center != identity.outer_center
        or cyclic.physical_surface_hash != identity.physical_surface_hash
        or cyclic.action_surface_seal_hash == identity.action_surface_seal_hash
        or cyclic.result_hash == identity.result_hash
        or _decision_route_keys(cyclic) != _decision_route_keys(identity)
    ):
        raise ProtocolError(
            "P-DCAPS cyclic method lacks a distinct same-topology cyclic result."
        )


def _action_only_selection_hash(result: OuterActionPolicyResult) -> str:
    return canonical_hash(
        {
            "schema_version": "pdcaps_action_only_selection_source_v1",
            "outer_result_hash": result.result_hash,
            "target_action_decision_hashes": tuple(
                row.decision_hash for row in result.target_action_decisions
            ),
            "selected_action_hashes": result.target_action_only_actions,
            "policy_surface_used": False,
            "outer_admission_used": False,
            "target_labels_used": False,
        }
    )


@dataclass(frozen=True)
class MethodControlDecision:
    """One fixed-menu decision derived from typed, sealed in-run sources.

    The selected hashes and every provenance field are ``init=False``.  A
    caller may choose a declared method and provide its required typed sources,
    but cannot inject an arbitrary action list or opaque seal.
    """

    method_id: str
    identity_result: OuterActionPolicyResult = field(repr=False, compare=False)
    source_result: OuterActionPolicyResult = field(repr=False, compare=False)
    legacy_control: LegacyControlSeal | None = field(
        default=None, repr=False, compare=False
    )
    surface_set: SealedActionSurfaceSet | None = field(
        default=None, repr=False, compare=False
    )
    outer_center: str = field(init=False)
    identity_result_hash: str = field(init=False)
    identity_action_surface_seal_hash: str = field(init=False)
    source_result_hash: str = field(init=False)
    source_action_surface_seal_hash: str = field(init=False)
    physical_surface_hash: str = field(init=False)
    posterior_control_id: str = field(init=False)
    source_selection_hash: str = field(init=False)
    source_authorization_hash: str | None = field(init=False)
    legacy_control_seal_hash: str | None = field(init=False)
    joint_surface_set_seal_hash: str | None = field(init=False)
    outer_admission_applied: bool = field(init=False)
    outer_admission_passed: bool | None = field(init=False)
    outer_admission_hash: str | None = field(init=False)
    selected_action_hashes: tuple[str, ...] = field(init=False)
    exact_p_fallback: bool = field(init=False)
    reason: str = field(init=False)
    selection_semantics: str = field(init=False)
    decision_hash: str = field(init=False)

    def __post_init__(self) -> None:
        method = str(self.method_id)
        if method not in METHOD_MENU:
            raise ProtocolError("P-DCAPS method control is outside the fixed menu.")
        identity = _require_result(self.identity_result, role="identity result")
        source = _require_result(self.source_result, role="method source result")
        _require_identity_result(identity)

        if method == CYCLIC_METHOD_ID:
            _require_cyclic_pair(identity, source)
            if (
                not isinstance(self.surface_set, SealedActionSurfaceSet)
                or self.surface_set.control_ids
                != (IDENTITY_POSTERIOR_CONTROL_ID, CYCLIC_POSTERIOR_CONTROL_ID)
                or self.surface_set.identity.action_surface_seal_hash
                != identity.action_surface_seal_hash
                or self.surface_set.identity.physical_surface_hash
                != identity.physical_surface_hash
                or self.surface_set.cyclic is None
                or self.surface_set.cyclic.action_surface_seal_hash
                != source.action_surface_seal_hash
                or self.surface_set.cyclic.physical_surface_hash
                != source.physical_surface_hash
            ):
                raise ProtocolError(
                    "P-DCAPS cyclic method lacks its typed joint surface-set seal."
                )
        elif (
            source.posterior_control_id != IDENTITY_POSTERIOR_CONTROL_ID
            or source.result_hash != identity.result_hash
            or source.action_surface_seal_hash
            != identity.action_surface_seal_hash
            or source.physical_surface_hash != identity.physical_surface_hash
        ):
            raise ProtocolError(
                "P-DCAPS identity method was bound to another result surface."
            )
        elif self.surface_set is not None:
            raise ProtocolError(
                "P-DCAPS non-cyclic method cannot consume a cyclic surface set."
            )

        control: LegacyControlSeal | None = None
        authorization: AuthorizedOuterPolicy | None = None
        selected: tuple[str, ...]
        selection_hash: str
        reason: str

        if method == P_METHOD_ID:
            if self.legacy_control is not None:
                raise ProtocolError("P-DCAPS protected P cannot consume a control seal.")
            p_cell = source.target_policy_surface.cells[0]
            if p_cell.k != 0 or p_cell.ordered_action_hashes:
                raise ProtocolError("P-DCAPS exact-P policy cell drifted.")
            selected = ()
            selection_hash = p_cell.cell_hash
            reason = "EXACT_P_PROTECTED"
        elif method == ACTION_ONLY_METHOD_ID:
            if self.legacy_control is not None:
                raise ProtocolError(
                    "P-DCAPS action-only ablation cannot consume a control seal."
                )
            selected = tuple(source.target_action_only_actions)
            selected_from_decisions = tuple(
                row.selection.selected_action_key.action_key_hash
                for row in source.target_action_decisions
                if row.selection.selected_action_key is not None
            )
            if (
                len(selected) != len(set(selected))
                or set(selected) != set(selected_from_decisions)
            ):
                raise ProtocolError("P-DCAPS action-only source inventory drifted.")
            selection_hash = _action_only_selection_hash(source)
            reason = (
                "P_DCAPS_ACTION_ONLY_SELECTED"
                if selected
                else "EXACT_P_ACTION_LAYER_SELECTED_NO_ACTION"
            )
        elif method == POLICY_ONLY_METHOD_ID:
            if self.legacy_control is not None:
                raise ProtocolError(
                    "P-DCAPS policy-only ablation cannot consume a control seal."
                )
            selected = tuple(source.target_selected_policy_actions)
            selection_hash = source.target_policy_selection.selection_hash
            reason = (
                "P_DCAPS_POLICY_ONLY_SELECTED"
                if selected
                else "EXACT_P_POLICY_SELECTED_K0"
            )
        elif method == LEGACY_METHOD_ID:
            if not isinstance(self.legacy_control, LegacyControlSeal):
                raise ProtocolError(
                    "P-DCAPS legacy method requires a typed same-run control seal."
                )
            control, _references = resolve_legacy_control(
                source, self.legacy_control
            )
            target = control.surface.target_decision
            selected = tuple(target.selected_action_hashes)
            selection_hash = target.decision_hash
            reason = target.reason
        else:
            if not isinstance(self.legacy_control, LegacyControlSeal):
                raise ProtocolError(
                    "P-DCAPS admitted method requires a typed same-run control seal."
                )
            control, _references = resolve_legacy_control(
                source, self.legacy_control
            )
            admission = build_admission_from_pseudo_policies(source, control)
            authorization = authorize_primary_policy(source, admission)
            selected = tuple(authorization.selected_action_hashes)
            selection_hash = source.target_policy_selection.selection_hash
            if method == PRIMARY_METHOD_ID:
                reason = authorization.reason
            else:
                reason = (
                    "P_DCAPS_CYCLIC_POISON_ADMITTED"
                    if selected
                    else f"CYCLIC_CONTROL::{authorization.reason}"
                )

        selected = tuple(
            require_sha256(value, "method selected-action hash")
            for value in selected
        )
        if len(selected) != len(set(selected)):
            raise ProtocolError("P-DCAPS method selected duplicate actions.")
        source_actions = {
            action.action_hash for action in source.target_policy_surface.ranked_actions
        }
        if not set(selected).issubset(source_actions):
            raise ProtocolError("P-DCAPS method selected action left its source result.")

        control_hash = (
            None
            if control is None
            else require_sha256(
                control.legacy_control_seal_hash, "legacy control seal hash"
            )
        )
        surface_set_hash = (
            None
            if self.surface_set is None
            else require_sha256(
                self.surface_set.surface_set_seal_hash,
                "joint action-surface-set seal hash",
            )
        )
        authorization_hash = (
            None if authorization is None else authorization.authorization_hash
        )
        admission_hash = (
            None if authorization is None else authorization.admission.admission_hash
        )
        admission_passed = (
            None if authorization is None else authorization.admission.passed
        )
        exact_p = not selected
        semantics = _SELECTION_SEMANTICS[method]

        values = {
            "outer_center": source.outer_center,
            "identity_result_hash": identity.result_hash,
            "identity_action_surface_seal_hash": (
                identity.action_surface_seal_hash
            ),
            "source_result_hash": source.result_hash,
            "source_action_surface_seal_hash": source.action_surface_seal_hash,
            "physical_surface_hash": source.physical_surface_hash,
            "posterior_control_id": source.posterior_control_id,
            "source_selection_hash": require_sha256(
                selection_hash, "method selection-source hash"
            ),
            "source_authorization_hash": authorization_hash,
            "legacy_control_seal_hash": control_hash,
            "joint_surface_set_seal_hash": surface_set_hash,
            "outer_admission_applied": authorization is not None,
            "outer_admission_passed": admission_passed,
            "outer_admission_hash": admission_hash,
            "selected_action_hashes": selected,
            "exact_p_fallback": exact_p,
            "reason": str(reason),
            "selection_semantics": semantics,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "method_id", method)
        object.__setattr__(
            self,
            "decision_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_method_control_decision_v1",
                    "method_id": method,
                    **values,
                    "publication_status": PUBLICATION_STATUS,
                    "terminal_decision": TERMINAL_DECISION,
                    "routing_authorized": False,
                    "promotion_allowed": False,
                    "target_labels_used": False,
                }
            ),
        )

    @property
    def composition_selection_enabled(self) -> bool:
        """Whether composition should replay the derived non-P actions."""

        if self.outer_admission_applied:
            return bool(self.outer_admission_passed and self.selected_action_hashes)
        return bool(self.selected_action_hashes)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_method_control_decision_v1",
            "method_id": self.method_id,
            "outer_center": self.outer_center,
            "identity_result_hash": self.identity_result_hash,
            "identity_action_surface_seal_hash": (
                self.identity_action_surface_seal_hash
            ),
            "source_result_hash": self.source_result_hash,
            "source_action_surface_seal_hash": (
                self.source_action_surface_seal_hash
            ),
            "physical_surface_hash": self.physical_surface_hash,
            "posterior_control_id": self.posterior_control_id,
            "source_selection_hash": self.source_selection_hash,
            "source_authorization_hash": self.source_authorization_hash,
            "legacy_control_seal_hash": self.legacy_control_seal_hash,
            "joint_surface_set_seal_hash": self.joint_surface_set_seal_hash,
            "outer_admission_applied": self.outer_admission_applied,
            "outer_admission_passed": self.outer_admission_passed,
            "outer_admission_hash": self.outer_admission_hash,
            "selected_action_hashes": list(self.selected_action_hashes),
            "exact_p_fallback": self.exact_p_fallback,
            "reason": self.reason,
            "selection_semantics": self.selection_semantics,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "routing_authorized": False,
            "promotion_allowed": False,
            "target_labels_used": False,
            "decision_hash": self.decision_hash,
        }


@dataclass(frozen=True)
class ComposedMethodPrediction:
    """Typed composition result retaining method-source and admission meaning."""

    decision: MethodControlDecision
    prediction: ComposedCenterPrediction
    method_composition_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.decision, MethodControlDecision)
            or not isinstance(self.prediction, ComposedCenterPrediction)
            or self.prediction.center != self.decision.outer_center
            or self.prediction.method_id != self.decision.method_id
            or self.prediction.selected_action_hashes
            != self.decision.selected_action_hashes
            or self.prediction.selection_enabled
            != self.decision.composition_selection_enabled
        ):
            raise ProtocolError("P-DCAPS typed method composition drifted.")
        object.__setattr__(
            self,
            "method_composition_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_composed_method_prediction_v1",
                    "method_decision_hash": self.decision.decision_hash,
                    "prediction_composition_hash": self.prediction.composition_hash,
                    "outer_admission_applied": (
                        self.decision.outer_admission_applied
                    ),
                    "outer_admission_passed": (
                        self.decision.outer_admission_passed
                    ),
                    "selection_enabled": (
                        self.decision.composition_selection_enabled
                    ),
                    "terminal_diagnostic_only": True,
                    "target_labels_used": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_composed_method_prediction_v1",
            "decision": self.decision.to_payload(),
            "prediction": self.prediction.to_payload(),
            "outer_admission_applied": self.decision.outer_admission_applied,
            "outer_admission_passed": self.decision.outer_admission_passed,
            "selection_enabled": self.decision.composition_selection_enabled,
            "terminal_diagnostic_only": True,
            "target_labels_used": False,
            "method_composition_hash": self.method_composition_hash,
        }


def build_protected_method_decision(
    result: OuterActionPolicyResult,
) -> MethodControlDecision:
    return MethodControlDecision(P_METHOD_ID, result, result)


def build_primary_method_decision(
    result: OuterActionPolicyResult,
    legacy_control: LegacyControlSeal,
) -> MethodControlDecision:
    return MethodControlDecision(
        PRIMARY_METHOD_ID, result, result, legacy_control
    )


def build_action_only_method_decision(
    result: OuterActionPolicyResult,
) -> MethodControlDecision:
    return MethodControlDecision(ACTION_ONLY_METHOD_ID, result, result)


def build_policy_only_method_decision(
    result: OuterActionPolicyResult,
) -> MethodControlDecision:
    return MethodControlDecision(POLICY_ONLY_METHOD_ID, result, result)


def build_legacy_method_decision(
    result: OuterActionPolicyResult,
    legacy_control: LegacyControlSeal,
) -> MethodControlDecision:
    return MethodControlDecision(LEGACY_METHOD_ID, result, result, legacy_control)


def build_cyclic_poison_method_decision(
    identity_result: OuterActionPolicyResult,
    cyclic_result: OuterActionPolicyResult,
    surface_set: SealedActionSurfaceSet,
    cyclic_legacy_control: LegacyControlSeal,
) -> MethodControlDecision:
    return MethodControlDecision(
        CYCLIC_METHOD_ID,
        identity_result,
        cyclic_result,
        cyclic_legacy_control,
        surface_set,
    )


def compose_method_prediction(
    routes: Sequence[SealedRouteActionSurface],
    *,
    center_sample_order: Sequence[str],
    decision: MethodControlDecision,
) -> ComposedMethodPrediction:
    """Compose one typed method; raw method IDs/action lists are not accepted."""

    if not isinstance(decision, MethodControlDecision):
        raise ProtocolError("P-DCAPS composition requires a typed method decision.")
    rows = tuple(sorted(tuple(routes), key=lambda row: row.route_key))
    expected_route_keys = tuple(
        row.route_key for row in decision.source_result.target_action_decisions
    )
    if (
        not rows
        or tuple(row.route_key for row in rows) != expected_route_keys
        or any(row.route_key.surface_role != "target" for row in rows)
        or any(row.route_key.outer_center != decision.outer_center for row in rows)
        or any(
            row.action_surface_seal_hash
            != decision.source_action_surface_seal_hash
            for row in rows
        )
        or any(
            row.physical_surface_hash != decision.physical_surface_hash
            for row in rows
        )
        or any(
            row.posterior_control_id != decision.posterior_control_id
            for row in rows
        )
    ):
        raise ProtocolError("P-DCAPS method composition source surface drifted.")
    prediction = compose_center_prediction(
        rows,
        center_sample_order=center_sample_order,
        selected_action_hashes=decision.selected_action_hashes,
        method_id=decision.method_id,
        selection_enabled=decision.composition_selection_enabled,
    )
    return ComposedMethodPrediction(decision, prediction)


__all__ = (
    "CYCLIC_POSTERIOR_CONTROL_ID",
    "ComposedMethodPrediction",
    "IDENTITY_POSTERIOR_CONTROL_ID",
    "MethodControlDecision",
    "build_action_only_method_decision",
    "build_cyclic_poison_method_decision",
    "build_legacy_method_decision",
    "build_policy_only_method_decision",
    "build_primary_method_decision",
    "build_protected_method_decision",
    "compose_method_prediction",
)
