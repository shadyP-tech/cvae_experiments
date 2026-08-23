"""Pseudo-only Admission_H and final preterminal policy authorization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from .admission import OuterAdmission, PseudoPolicyEvidence, build_outer_admission
from .engine import OuterActionPolicyResult
from .identity import PRIMARY_METHOD_ID, canonical_hash
from .legacy_control import (
    LegacyControlSeal,
    LegacyPseudoReference,
    resolve_legacy_control,
)


@dataclass(frozen=True)
class AuthorizedOuterPolicy:
    outer_center: str
    method_id: str
    admission: OuterAdmission
    proposed_selection_hash: str
    selected_action_hashes: tuple[str, ...]
    exact_p_fallback: bool
    reason: str
    authorization_hash: str = field(init=False)

    def __post_init__(self) -> None:
        selected = tuple(str(value) for value in self.selected_action_hashes)
        if (
            self.method_id != PRIMARY_METHOD_ID
            or self.admission.outer_center != self.outer_center
            or bool(self.exact_p_fallback) != (not selected)
            or (selected and not self.admission.passed)
            or not self.reason
        ):
            raise ProtocolError("P-DCAPS outer policy authorization drifted.")
        object.__setattr__(self, "selected_action_hashes", selected)
        object.__setattr__(
            self,
            "authorization_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_authorized_outer_policy_v1",
                    "outer_center": self.outer_center,
                    "method_id": self.method_id,
                    "admission_hash": self.admission.admission_hash,
                    "proposed_selection_hash": self.proposed_selection_hash,
                    "selected_action_hashes": selected,
                    "exact_p_fallback": self.exact_p_fallback,
                    "reason": self.reason,
                    "target_labels_used": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_authorized_outer_policy_v1",
            "outer_center": self.outer_center,
            "method_id": self.method_id,
            "admission": self.admission.to_payload(),
            "proposed_selection_hash": self.proposed_selection_hash,
            "selected_action_hashes": list(self.selected_action_hashes),
            "exact_p_fallback": self.exact_p_fallback,
            "reason": self.reason,
            "target_labels_used": False,
            "authorization_hash": self.authorization_hash,
        }


def build_admission_from_pseudo_policies(
    result: OuterActionPolicyResult,
    legacy_control: LegacyControlSeal | Sequence[LegacyPseudoReference],
) -> OuterAdmission:
    """Construct Admission_H without touching target-H terminal labels."""

    donors = tuple(center for center in CENTERS if center != result.outer_center)
    _seal, resolved = resolve_legacy_control(result, legacy_control)
    references = tuple(sorted(resolved, key=lambda row: row.donor_center))
    if (
        tuple(row.donor_center for row in references) != donors
        or any(row.outer_center != result.outer_center for row in references)
    ):
        raise ProtocolError("P-DCAPS admission legacy control inventory drifted.")
    surfaces = {
        row.provenance.route_center: row
        for row in result.pseudo_policy_response_surfaces
    }
    selections = dict(result.pseudo_policy_selections_by_center)
    evidence: list[PseudoPolicyEvidence] = []
    for reference in references:
        surface = surfaces[reference.donor_center]
        if (
            surface.response_surface_hash is None
            or reference.pseudo_response_surface_hash
            != surface.response_surface_hash
            or surface.provenance.action_surface_seal_hash
            != result.action_surface_seal_hash
        ):
            raise ProtocolError(
                "P-DCAPS admission legacy pseudo-response lineage drifted."
            )
        selection = selections[reference.donor_center]
        selected = selection.selected_cell
        realized = surface.cells[selection.selected_k].realized_utility
        if realized is None:
            raise ProtocolError("P-DCAPS pseudo admission response is absent.")
        safe = (
            realized.bacc_gain > 0.0
            and realized.brier_gain >= 0.0
            and realized.log_gain >= 0.0
        )
        evidence.append(
            PseudoPolicyEvidence(
                result.outer_center,
                reference.donor_center,
                selected.corrected_utility,
                realized,
                selection.authorized,
                bool(selection.authorized and safe),
                reference.endpoint_oracle_bacc_gain,
                abs(reference.endpoint_oracle_bacc_gain - realized.bacc_gain),
                reference.legacy_realized,
                reference.legacy_routed,
                reference.legacy_jointly_safe,
                reference.legacy_absolute_oracle_regret,
            )
        )
    return build_outer_admission(result.outer_center, evidence)


def authorize_primary_policy(
    result: OuterActionPolicyResult,
    admission: OuterAdmission,
) -> AuthorizedOuterPolicy:
    proposed = result.target_selected_policy_actions
    if admission.outer_center != result.outer_center:
        raise ProtocolError("P-DCAPS Admission_H was applied to another center.")
    if admission.passed and proposed:
        selected = proposed
        fallback = False
        reason = "P_DCAPS_PRIMARY_ADMITTED"
    elif not admission.passed:
        selected = ()
        fallback = True
        reason = "EXACT_P_OUTER_ADMISSION_FAILED"
    else:
        selected = ()
        fallback = True
        reason = "EXACT_P_POLICY_SELECTED_K0"
    return AuthorizedOuterPolicy(
        result.outer_center,
        PRIMARY_METHOD_ID,
        admission,
        result.target_policy_selection.selection_hash,
        selected,
        fallback,
        reason,
    )


__all__ = (
    "AuthorizedOuterPolicy",
    "LegacyPseudoReference",
    "authorize_primary_policy",
    "build_admission_from_pseudo_policies",
)
