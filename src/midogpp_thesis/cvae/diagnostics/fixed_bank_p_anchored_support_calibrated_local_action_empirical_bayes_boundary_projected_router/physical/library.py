"""Independent 810-cell action inventory over neutral fixed-bank runtime data."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..hashing import canonical_hash
from ..identity import (
    CENTERS,
    GENERATION_SEEDS,
    PHYSICAL_CELL_COUNT,
    TRAINING_SEEDS,
)
from ..protocol import ProtocolError


B_ACTION_ID = "B"
U_ACTION_ID = "U"
A1_PREFIX = "A1::source="


def action_ids_for_target(target_center: str) -> tuple[str, ...]:
    target = str(target_center)
    if target not in CENTERS:
        raise ProtocolError("SCALE-BP physical target center is unknown.")
    return (
        B_ACTION_ID,
        U_ACTION_ID,
        *(f"{A1_PREFIX}{source}" for source in CENTERS if source != target),
    )


@dataclass(frozen=True, slots=True, order=True)
class PhysicalCellIdentity:
    target_center: str
    action_id: str
    training_seed: int
    generation_seed: int
    cell_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        actions = action_ids_for_target(self.target_center)
        if (
            self.action_id not in actions
            or self.training_seed not in TRAINING_SEEDS
            or self.generation_seed not in GENERATION_SEEDS
            or (
                self.action_id.startswith(A1_PREFIX)
                and self.action_id == f"{A1_PREFIX}{self.target_center}"
            )
        ):
            raise ProtocolError("SCALE-BP physical cell identity drifted.")
        object.__setattr__(
            self,
            "cell_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_physical_cell_identity_v1",
                    "target_center": self.target_center,
                    "action_id": self.action_id,
                    "training_seed": self.training_seed,
                    "generation_seed": self.generation_seed,
                    "target_expert_excluded": True,
                    "labels_used": False,
                }
            ),
        )


def build_physical_cell_inventory() -> tuple[PhysicalCellIdentity, ...]:
    cells = tuple(
        PhysicalCellIdentity(target, action, training_seed, generation_seed)
        for target in CENTERS
        for action in action_ids_for_target(target)
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )
    if (
        len(cells) != PHYSICAL_CELL_COUNT
        or len({row.cell_hash for row in cells}) != PHYSICAL_CELL_COUNT
    ):
        raise ProtocolError("SCALE-BP physical 810-cell inventory drifted.")
    return cells


__all__ = (
    "A1_PREFIX",
    "B_ACTION_ID",
    "PhysicalCellIdentity",
    "U_ACTION_ID",
    "action_ids_for_target",
    "build_physical_cell_inventory",
)
