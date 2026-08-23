"""Same-run legacy control sealing, references, and admission replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ....protocol import ProtocolError
from ..engine import OuterActionPolicyResult
from ..identity import canonical_hash
from .contracts import (
    LegacyControlSurface,
    LegacyPseudoReference,
    LegacyTargetPolicyDecision,
)
from .selection import build_legacy_control_surface


@dataclass(frozen=True)
class LegacyControlSeal:
    """Final same-run seal accepted by P-DCAPS Admission_H."""

    surface: LegacyControlSurface
    legacy_control_seal_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.surface, LegacyControlSurface):
            raise ProtocolError("P-DCAPS legacy control seal DTO drifted.")
        surface = self.surface
        object.__setattr__(
            self,
            "legacy_control_seal_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_legacy_control_seal_v1",
                    "outer_center": surface.outer_center,
                    "outer_result_hash": surface.outer_result_hash,
                    "physical_surface_hash": surface.physical_surface_hash,
                    "action_surface_seal_hash": surface.action_surface_seal_hash,
                    "pseudo_response_surface_hashes": (
                        surface.pseudo_response_surface_hashes
                    ),
                    "legacy_decision_hashes": tuple(
                        row.decision_hash for row in surface.decisions
                    ),
                    "legacy_target_decision_hash": (
                        surface.target_decision.decision_hash
                    ),
                    "control_surface_hash": surface.control_surface_hash,
                    "same_run_control": True,
                    "pseudo_only": True,
                    "target_labels_used": False,
                }
            ),
        )

    @property
    def references(self) -> tuple[LegacyPseudoReference, ...]:
        return tuple(
            LegacyPseudoReference(
                self.surface.outer_result_hash,
                self.surface.physical_surface_hash,
                self.surface.action_surface_seal_hash,
                self.surface.control_surface_hash,
                self.legacy_control_seal_hash,
                self.surface.target_decision,
                decision,
            )
            for decision in self.surface.decisions
        )

    @property
    def target_decision(self) -> LegacyTargetPolicyDecision:
        return self.surface.target_decision

    @property
    def target_selected_action_hashes(self) -> tuple[str, ...]:
        return self.surface.target_decision.selected_action_hashes

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_legacy_control_seal_v1",
            "surface": self.surface.to_payload(),
            "references": [row.to_payload() for row in self.references],
            "same_run_control": True,
            "pseudo_only": True,
            "target_labels_used": False,
            "legacy_control_seal_hash": self.legacy_control_seal_hash,
        }


def seal_legacy_control(
    result: OuterActionPolicyResult,
) -> LegacyControlSeal:
    """Build and seal one same-run legacy control without target labels."""

    return LegacyControlSeal(build_legacy_control_surface(result))


def _validate_surface_against_result(
    result: OuterActionPolicyResult,
    surface: LegacyControlSurface,
) -> None:
    current_surfaces = tuple(result.pseudo_policy_response_surfaces)
    response_hashes = tuple(
        (row.provenance.route_center, str(row.response_surface_hash))
        for row in current_surfaces
    )
    if (
        surface.outer_center != result.outer_center
        or surface.outer_result_hash != result.result_hash
        or surface.physical_surface_hash != result.physical_surface_hash
        or surface.action_surface_seal_hash != result.action_surface_seal_hash
        or surface.pseudo_response_surface_hashes != response_hashes
    ):
        raise ProtocolError("P-DCAPS admission legacy same-run lineage drifted.")
    expected = build_legacy_control_surface(result)
    if expected != surface:
        raise ProtocolError(
            "P-DCAPS admission legacy decision/response lineage drifted."
        )


def resolve_legacy_control(
    result: OuterActionPolicyResult,
    control: LegacyControlSeal | Sequence[LegacyPseudoReference],
) -> tuple[LegacyControlSeal, tuple[LegacyPseudoReference, ...]]:
    """Validate a seal or reconstruct it from references emitted by that seal."""

    if isinstance(control, LegacyControlSeal):
        seal = control
        references = seal.references
    else:
        try:
            references = tuple(control)
        except TypeError as exc:
            raise ProtocolError(
                "P-DCAPS admission legacy reference inventory drifted."
            ) from exc
        donors = tuple(center for center in CENTERS if center != result.outer_center)
        if (
            any(not isinstance(row, LegacyPseudoReference) for row in references)
            or tuple(row.donor_center for row in references) != donors
            or any(row.outer_center != result.outer_center for row in references)
            or len({row.outer_result_hash for row in references}) != 1
            or len({row.physical_surface_hash for row in references}) != 1
            or len({row.action_surface_seal_hash for row in references}) != 1
            or len({row.control_surface_hash for row in references}) != 1
            or len({row.legacy_control_seal_hash for row in references}) != 1
            or len(
                {row.target_decision.decision_hash for row in references}
            )
            != 1
        ):
            raise ProtocolError(
                "P-DCAPS admission legacy reference inventory drifted."
            )
        rebuilt_surface = LegacyControlSurface(
            result.outer_center,
            references[0].outer_result_hash,
            references[0].physical_surface_hash,
            references[0].action_surface_seal_hash,
            tuple(
                (row.donor_center, row.pseudo_response_surface_hash)
                for row in references
            ),
            tuple(row.decision for row in references),
            references[0].target_decision,
        )
        seal = LegacyControlSeal(rebuilt_surface)
        if (
            any(
                row.control_surface_hash != rebuilt_surface.control_surface_hash
                for row in references
            )
            or any(
                row.legacy_control_seal_hash != seal.legacy_control_seal_hash
                for row in references
            )
            or tuple(row.reference_hash for row in references)
            != tuple(row.reference_hash for row in seal.references)
        ):
            raise ProtocolError("P-DCAPS admission legacy reference seal drifted.")
    _validate_surface_against_result(result, seal.surface)
    return seal, references


__all__ = (
    "LegacyControlSeal",
    "resolve_legacy_control",
    "seal_legacy_control",
)
