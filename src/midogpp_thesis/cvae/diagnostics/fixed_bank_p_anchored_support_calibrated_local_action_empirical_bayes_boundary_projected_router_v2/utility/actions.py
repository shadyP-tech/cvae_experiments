"""Complete six-cell action rectangles, including structural no-op rows."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..hashing import canonical_hash
from ..protocol import GovernanceError
from ..physical.contracts import ACTION_FAMILIES, ACTION_IDS, DIRECTIONS
from ..physical.endpoints import CaseEndpointSurface
from ..physical.evidence import CaseEvidencePacket, build_case_evidence_packet
from ..physical.geometry import BoundaryAction, build_boundary_action


@dataclass(frozen=True, slots=True)
class ActionCell:
    target_center: str
    case_id: str
    action: BoundaryAction
    evidence: CaseEvidencePacket
    cell_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.action.action_id != self.evidence.action_id
            or self.target_center != self.evidence.target_center
            or self.case_id != self.evidence.case_id
            or self.action.action_hash != self.evidence.action_hash
        ):
            raise GovernanceError("SCALE-BP v2 action cell lineage drifted.")
        object.__setattr__(
            self,
            "cell_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_action_cell_v1",
                    "target_center": self.target_center,
                    "case_id": self.case_id,
                    "action_id": self.action.action_id,
                    "action_hash": self.action.action_hash,
                    "evidence_packet_hash": self.evidence.packet_hash,
                    "threshold_switch_count": self.evidence.threshold_switch_count,
                    "harmful_switch_count": self.evidence.harmful_switch_count,
                    "harmful_switch_count_status": (
                        self.evidence.harmful_switch_count_status
                    ),
                    "structural_noop": self.action.structural_noop,
                }
            ),
        )

    @property
    def action_id(self) -> str:
        return self.action.action_id

    @property
    def structural_noop(self) -> bool:
        return self.action.structural_noop


@dataclass(frozen=True, slots=True)
class ActionRectangle:
    target_center: str
    case_id: str
    cells: tuple[ActionCell, ...]
    endpoint_surface_hash: str
    endpoint_plan_hash: str
    support_excluded_case_ids: tuple[str, ...]
    outer_held_case_id: str
    rectangle_hash: str = field(init=False)

    def __post_init__(self) -> None:
        cells = tuple(self.cells)
        excluded_cases = tuple(str(value) for value in self.support_excluded_case_ids)
        outer_held = str(self.outer_held_case_id)
        if (
            tuple(cell.action_id for cell in cells) != ACTION_IDS
            or len({cell.cell_hash for cell in cells}) != len(ACTION_IDS)
            or any(
                cell.target_center != self.target_center or cell.case_id != self.case_id
                for cell in cells
            )
            or any(
                cell.evidence.endpoint_surface_hash != self.endpoint_surface_hash
                for cell in cells
            )
            or not self.endpoint_plan_hash
            or self.case_id not in excluded_cases
            or outer_held not in excluded_cases
            or len(excluded_cases) != len(set(excluded_cases))
        ):
            raise GovernanceError("SCALE-BP v2 six-action rectangle drifted.")
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "support_excluded_case_ids", excluded_cases)
        object.__setattr__(self, "outer_held_case_id", outer_held)
        object.__setattr__(
            self,
            "rectangle_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_action_rectangle_v2",
                    "target_center": self.target_center,
                    "case_id": self.case_id,
                    "action_ids": ACTION_IDS,
                    "cell_hashes": tuple(cell.cell_hash for cell in cells),
                    "endpoint_surface_hash": self.endpoint_surface_hash,
                    "endpoint_plan_hash": self.endpoint_plan_hash,
                    "support_excluded_case_ids": excluded_cases,
                    "outer_held_case_id": outer_held,
                    "structural_noops_retained": True,
                }
            ),
        )

    def cell(self, action_id: object) -> ActionCell:
        action = str(action_id)
        try:
            return next(cell for cell in self.cells if cell.action_id == action)
        except StopIteration as exc:
            raise GovernanceError("SCALE-BP v2 action cell is absent.") from exc


def build_action_rectangle(surface: CaseEndpointSurface) -> ActionRectangle:
    cells: list[ActionCell] = []
    for family in ACTION_FAMILIES:
        for direction in DIRECTIONS:
            endpoint = surface.challenger(family, direction)
            action = build_boundary_action(
                surface.protected_p,
                endpoint,
                family=family,
                direction=direction,
            )
            evidence = build_case_evidence_packet(surface, action)
            cells.append(
                ActionCell(surface.target_center, surface.case_id, action, evidence)
            )
    return ActionRectangle(
        surface.target_center,
        surface.case_id,
        tuple(cells),
        surface.surface_hash,
        surface.plan_hash,
        surface.support_excluded_case_ids,
        surface.outer_held_case_id,
    )


__all__ = ("ActionCell", "ActionRectangle", "build_action_rectangle")
