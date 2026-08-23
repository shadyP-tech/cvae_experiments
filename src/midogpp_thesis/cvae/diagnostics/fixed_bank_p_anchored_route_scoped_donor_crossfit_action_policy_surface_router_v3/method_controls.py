"""Complete fixed-menu v3 adapters over frozen P-DCAPS scientific DTOs.

Only the nullable admission/authorization layer is new.  Physical surfaces,
outer-result DTOs, legacy controls, and the composition kernel remain the
source-sealed v2/base implementation. Primary and cyclic controls
deterministically compose byte-exact P whenever v3 Admission_H is undefined or
otherwise fails; the other four methods retain their frozen v2 semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ...protocol import ProtocolError
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.action_surface import (
    SealedRouteActionSurface,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.composition import (
    ComposedCenterPrediction,
    compose_center_prediction,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.engine import (
    OuterActionPolicyResult,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.legacy_control import (
    LegacyControlSeal,
    resolve_legacy_control,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.surface_set import (
    SealedActionSurfaceSet,
)
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
from .routing import authorize_primary_policy, build_admission_from_pseudo_policies


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


def _decision_route_keys(
    result: OuterActionPolicyResult,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    return (
        tuple(row.route_key for row in result.target_action_decisions),
        tuple(row.route_key for row in result.pseudo_action_decisions),
    )


def _require_identity_result(result: object) -> OuterActionPolicyResult:
    if (
        not isinstance(result, OuterActionPolicyResult)
        or result.posterior_control_id != IDENTITY_POSTERIOR_CONTROL_ID
    ):
        raise ProtocolError(
            "P-DCAPS v3 primary method requires an identity outer result."
        )
    return result


def _require_cyclic_pair(
    identity: OuterActionPolicyResult,
    cyclic: object,
    surface_set: object,
) -> tuple[OuterActionPolicyResult, SealedActionSurfaceSet]:
    identity = _require_identity_result(identity)
    if not isinstance(cyclic, OuterActionPolicyResult):
        raise ProtocolError("P-DCAPS v3 cyclic result DTO drifted.")
    if (
        cyclic.posterior_control_id != CYCLIC_POSTERIOR_CONTROL_ID
        or cyclic.outer_center != identity.outer_center
        or cyclic.physical_surface_hash != identity.physical_surface_hash
        or cyclic.action_surface_seal_hash == identity.action_surface_seal_hash
        or cyclic.result_hash == identity.result_hash
        or _decision_route_keys(cyclic) != _decision_route_keys(identity)
    ):
        raise ProtocolError(
            "P-DCAPS v3 cyclic method lacks a distinct same-topology result."
        )
    if (
        not isinstance(surface_set, SealedActionSurfaceSet)
        or surface_set.control_ids
        != (IDENTITY_POSTERIOR_CONTROL_ID, CYCLIC_POSTERIOR_CONTROL_ID)
        or surface_set.identity.action_surface_seal_hash
        != identity.action_surface_seal_hash
        or surface_set.identity.physical_surface_hash
        != identity.physical_surface_hash
        or surface_set.cyclic is None
        or surface_set.cyclic.action_surface_seal_hash
        != cyclic.action_surface_seal_hash
        or surface_set.cyclic.physical_surface_hash
        != cyclic.physical_surface_hash
    ):
        raise ProtocolError(
            "P-DCAPS v3 cyclic method lacks its joint surface-set seal."
        )
    return cyclic, surface_set


def _action_only_selection_hash(result: OuterActionPolicyResult) -> str:
    return canonical_hash(
        {
            "schema_version": "pdcaps_v3_action_only_selection_source_v1",
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
class AdmissionControlledMethodDecision:
    """One fixed-menu decision derived only from typed sealed sources.

    The historical class name remains as the compatibility facade, but the DTO
    now covers the complete frozen six-method menu.  Only primary and cyclic
    apply repaired Admission_H; the other four retain their unchanged v2
    selection semantics.
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
    source_result_hash: str = field(init=False)
    source_action_surface_seal_hash: str = field(init=False)
    physical_surface_hash: str = field(init=False)
    posterior_control_id: str = field(init=False)
    source_selection_hash: str = field(init=False)
    source_authorization_hash: str | None = field(init=False)
    legacy_control_seal_hash: str | None = field(init=False)
    outer_admission_applied: bool = field(init=False)
    outer_admission_passed: bool | None = field(init=False)
    outer_admission_hash: str | None = field(init=False)
    selected_action_hashes: tuple[str, ...] = field(init=False)
    exact_p_fallback: bool = field(init=False)
    reason: str = field(init=False)
    selection_semantics: str = field(init=False)
    joint_surface_set_seal_hash: str | None = field(init=False)
    decision_hash: str = field(init=False)

    def __post_init__(self) -> None:
        method = str(self.method_id)
        if method not in METHOD_MENU:
            raise ProtocolError("P-DCAPS v3 method is outside the fixed menu.")
        identity = _require_identity_result(self.identity_result)
        if method == CYCLIC_METHOD_ID:
            source, sealed_set = _require_cyclic_pair(
                identity, self.source_result, self.surface_set
            )
            surface_set_hash = require_sha256(
                sealed_set.surface_set_seal_hash,
                "joint surface-set seal hash",
            )
        else:
            source = _require_identity_result(self.source_result)
            if (
                source.result_hash != identity.result_hash
                or source.action_surface_seal_hash
                != identity.action_surface_seal_hash
                or source.physical_surface_hash != identity.physical_surface_hash
                or self.surface_set is not None
            ):
                raise ProtocolError("P-DCAPS v3 identity result lineage drifted.")
            surface_set_hash = None

        control: LegacyControlSeal | None = None
        authorization = None
        admission = None
        selection_hash: str
        selected: tuple[str, ...]
        reason: str

        if method == P_METHOD_ID:
            if self.legacy_control is not None:
                raise ProtocolError(
                    "P-DCAPS v3 protected P cannot consume a legacy seal."
                )
            p_cell = source.target_policy_surface.cells[0]
            if p_cell.k != 0 or p_cell.ordered_action_hashes:
                raise ProtocolError("P-DCAPS v3 exact-P policy cell drifted.")
            selected = ()
            selection_hash = p_cell.cell_hash
            reason = "EXACT_P_PROTECTED"
        elif method == ACTION_ONLY_METHOD_ID:
            if self.legacy_control is not None:
                raise ProtocolError(
                    "P-DCAPS v3 action-only cannot consume a legacy seal."
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
                raise ProtocolError(
                    "P-DCAPS v3 action-only source inventory drifted."
                )
            selection_hash = _action_only_selection_hash(source)
            reason = (
                "P_DCAPS_ACTION_ONLY_SELECTED"
                if selected
                else "EXACT_P_ACTION_LAYER_SELECTED_NO_ACTION"
            )
        elif method == POLICY_ONLY_METHOD_ID:
            if self.legacy_control is not None:
                raise ProtocolError(
                    "P-DCAPS v3 policy-only cannot consume a legacy seal."
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
                    "P-DCAPS v3 legacy method requires a typed same-run seal."
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
                    "P-DCAPS v3 admitted method requires a typed same-run seal."
                )
            control, _references = resolve_legacy_control(
                source, self.legacy_control
            )
            admission = build_admission_from_pseudo_policies(source, control)
            authorization = authorize_primary_policy(source, admission)
            selected = tuple(authorization.selected_action_hashes)
            selection_hash = source.target_policy_selection.selection_hash
            reason = (
                authorization.reason
                if method == PRIMARY_METHOD_ID
                else (
                    "P_DCAPS_CYCLIC_POISON_ADMITTED"
                    if selected
                    else f"CYCLIC_CONTROL::{authorization.reason}"
                )
            )

        selected = tuple(
            require_sha256(value, "method selected-action hash")
            for value in selected
        )
        source_actions = {
            action.action_hash
            for action in source.target_policy_surface.ranked_actions
        }
        if (
            len(selected) != len(set(selected))
            or not set(selected).issubset(source_actions)
        ):
            raise ProtocolError("P-DCAPS v3 selected action inventory drifted.")
        control_hash = (
            None
            if control is None
            else require_sha256(
                control.legacy_control_seal_hash, "legacy control seal hash"
            )
        )
        authorization_hash = (
            None
            if authorization is None
            else require_sha256(
                authorization.authorization_hash,
                "source authorization hash",
            )
        )
        admission_hash = None if admission is None else admission.admission_hash
        admission_passed = None if admission is None else admission.passed
        admission_applied = admission is not None
        semantics = _SELECTION_SEMANTICS[method]
        exact_p = not selected
        source_selection_hash = require_sha256(
            selection_hash, "method selection-source hash"
        )
        values: dict[str, object] = {
            "outer_center": source.outer_center,
            "identity_result_hash": identity.result_hash,
            "source_result_hash": source.result_hash,
            "source_action_surface_seal_hash": source.action_surface_seal_hash,
            "physical_surface_hash": source.physical_surface_hash,
            "posterior_control_id": source.posterior_control_id,
            "source_selection_hash": source_selection_hash,
            "source_authorization_hash": authorization_hash,
            "legacy_control_seal_hash": control_hash,
            "outer_admission_applied": admission_applied,
            "outer_admission_passed": admission_passed,
            "outer_admission_hash": admission_hash,
            "selected_action_hashes": selected,
            "exact_p_fallback": exact_p,
            "reason": reason,
            "selection_semantics": semantics,
            "joint_surface_set_seal_hash": surface_set_hash,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "method_id", method)
        object.__setattr__(self, "identity_result", identity)
        object.__setattr__(self, "source_result", source)
        object.__setattr__(
            self,
            "decision_hash",
            canonical_hash(
                {
                    "schema_version": (
                        "pdcaps_v3_admission_controlled_method_decision_v1"
                    ),
                    "method_id": method,
                    **values,
                    "nullable_admission_statistics": admission_applied,
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
        if self.outer_admission_applied:
            return bool(
                self.outer_admission_passed and self.selected_action_hashes
            )
        return bool(self.selected_action_hashes)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                "pdcaps_v3_admission_controlled_method_decision_v1"
            ),
            "method_id": self.method_id,
            "outer_center": self.outer_center,
            "identity_result_hash": self.identity_result_hash,
            "source_result_hash": self.source_result_hash,
            "source_action_surface_seal_hash": (
                self.source_action_surface_seal_hash
            ),
            "physical_surface_hash": self.physical_surface_hash,
            "posterior_control_id": self.posterior_control_id,
            "source_selection_hash": self.source_selection_hash,
            "source_authorization_hash": self.source_authorization_hash,
            "legacy_control_seal_hash": self.legacy_control_seal_hash,
            "outer_admission_applied": self.outer_admission_applied,
            "outer_admission_passed": self.outer_admission_passed,
            "outer_admission_hash": self.outer_admission_hash,
            "selected_action_hashes": list(self.selected_action_hashes),
            "exact_p_fallback": self.exact_p_fallback,
            "reason": self.reason,
            "selection_semantics": self.selection_semantics,
            "joint_surface_set_seal_hash": self.joint_surface_set_seal_hash,
            "nullable_admission_statistics": self.outer_admission_applied,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "routing_authorized": False,
            "promotion_allowed": False,
            "target_labels_used": False,
            "decision_hash": self.decision_hash,
        }


@dataclass(frozen=True)
class ComposedAdmissionControlledPrediction:
    decision: AdmissionControlledMethodDecision
    prediction: ComposedCenterPrediction
    method_composition_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.decision, AdmissionControlledMethodDecision)
            or not isinstance(self.prediction, ComposedCenterPrediction)
            or self.prediction.center != self.decision.outer_center
            or self.prediction.method_id != self.decision.method_id
            or self.prediction.selected_action_hashes
            != self.decision.selected_action_hashes
            or self.prediction.selection_enabled
            != self.decision.composition_selection_enabled
        ):
            raise ProtocolError("P-DCAPS v3 typed composition drifted.")
        object.__setattr__(
            self,
            "method_composition_hash",
            canonical_hash(
                {
                    "schema_version": (
                        "pdcaps_v3_admission_controlled_composition_v1"
                    ),
                    "method_decision_hash": self.decision.decision_hash,
                    "prediction_composition_hash": self.prediction.composition_hash,
                    "exact_p_fallback": self.decision.exact_p_fallback,
                    "terminal_diagnostic_only": True,
                    "target_labels_used": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_v3_admission_controlled_composition_v1",
            "decision": self.decision.to_payload(),
            "prediction": self.prediction.to_payload(),
            "exact_p_fallback": self.decision.exact_p_fallback,
            "terminal_diagnostic_only": True,
            "target_labels_used": False,
            "method_composition_hash": self.method_composition_hash,
        }


def build_primary_method_decision(
    result: OuterActionPolicyResult,
    legacy_control: LegacyControlSeal,
) -> AdmissionControlledMethodDecision:
    return AdmissionControlledMethodDecision(
        PRIMARY_METHOD_ID, result, result, legacy_control
    )


def build_protected_method_decision(
    result: OuterActionPolicyResult,
) -> AdmissionControlledMethodDecision:
    return AdmissionControlledMethodDecision(P_METHOD_ID, result, result)


def build_action_only_method_decision(
    result: OuterActionPolicyResult,
) -> AdmissionControlledMethodDecision:
    return AdmissionControlledMethodDecision(
        ACTION_ONLY_METHOD_ID, result, result
    )


def build_policy_only_method_decision(
    result: OuterActionPolicyResult,
) -> AdmissionControlledMethodDecision:
    return AdmissionControlledMethodDecision(
        POLICY_ONLY_METHOD_ID, result, result
    )


def build_legacy_method_decision(
    result: OuterActionPolicyResult,
    legacy_control: LegacyControlSeal,
) -> AdmissionControlledMethodDecision:
    return AdmissionControlledMethodDecision(
        LEGACY_METHOD_ID, result, result, legacy_control
    )


def build_cyclic_poison_method_decision(
    identity_result: OuterActionPolicyResult,
    cyclic_result: OuterActionPolicyResult,
    surface_set: SealedActionSurfaceSet,
    cyclic_legacy_control: LegacyControlSeal,
) -> AdmissionControlledMethodDecision:
    return AdmissionControlledMethodDecision(
        CYCLIC_METHOD_ID,
        identity_result,
        cyclic_result,
        cyclic_legacy_control,
        surface_set,
    )


def build_fixed_method_menu(
    *,
    identity_result: OuterActionPolicyResult,
    cyclic_result: OuterActionPolicyResult,
    surface_set: SealedActionSurfaceSet,
    identity_legacy_control: LegacyControlSeal,
    cyclic_legacy_control: LegacyControlSeal,
) -> tuple[AdmissionControlledMethodDecision, ...]:
    """Construct the complete frozen menu in canonical method order."""

    decisions = (
        build_protected_method_decision(identity_result),
        build_primary_method_decision(
            identity_result, identity_legacy_control
        ),
        build_action_only_method_decision(identity_result),
        build_policy_only_method_decision(identity_result),
        build_legacy_method_decision(
            identity_result, identity_legacy_control
        ),
        build_cyclic_poison_method_decision(
            identity_result,
            cyclic_result,
            surface_set,
            cyclic_legacy_control,
        ),
    )
    if tuple(row.method_id for row in decisions) != METHOD_MENU:
        raise ProtocolError("P-DCAPS v3 fixed method inventory drifted.")
    return decisions


def compose_method_prediction(
    routes: Sequence[SealedRouteActionSurface],
    *,
    center_sample_order: Sequence[str],
    decision: AdmissionControlledMethodDecision,
) -> ComposedAdmissionControlledPrediction:
    if not isinstance(decision, AdmissionControlledMethodDecision):
        raise ProtocolError(
            "P-DCAPS v3 composition requires a typed method decision."
        )
    rows = tuple(sorted(tuple(routes), key=lambda row: row.route_key))
    expected_route_keys = tuple(
        row.route_key for row in decision.source_result.target_action_decisions
    )
    if (
        not rows
        or tuple(row.route_key for row in rows) != expected_route_keys
        or any(row.route_key.surface_role != "target" for row in rows)
        or any(
            row.route_key.outer_center != decision.outer_center for row in rows
        )
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
        raise ProtocolError("P-DCAPS v3 composition source surface drifted.")
    prediction = compose_center_prediction(
        rows,
        center_sample_order=center_sample_order,
        selected_action_hashes=decision.selected_action_hashes,
        method_id=decision.method_id,
        selection_enabled=decision.composition_selection_enabled,
    )
    # The frozen composition kernel reconstructs P by sample ID in the caller's
    # requested order and performs its own byte-exact no-selection assertion.
    return ComposedAdmissionControlledPrediction(decision, prediction)


__all__ = (
    "AdmissionControlledMethodDecision",
    "CYCLIC_POSTERIOR_CONTROL_ID",
    "ComposedAdmissionControlledPrediction",
    "IDENTITY_POSTERIOR_CONTROL_ID",
    "build_action_only_method_decision",
    "build_cyclic_poison_method_decision",
    "build_fixed_method_menu",
    "build_legacy_method_decision",
    "build_policy_only_method_decision",
    "build_primary_method_decision",
    "build_protected_method_decision",
    "compose_method_prediction",
)
