"""Immutable, manifest-bound donor scopes for final and pseudo replay."""

from __future__ import annotations

from dataclasses import dataclass, field

from .case_inventory import DatasetCaseInventory
from .hashing import canonical_hash
from .identity import CENTERS
from .protocol import ProtocolError
from .route_identity import RouteScopeWitness


def _ordered_centers(values: set[str]) -> tuple[str, ...]:
    return tuple(center for center in CENTERS if center in values)


def _donor_case_ids(
    inventory: DatasetCaseInventory, centers: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(sorted(case for center in centers for case in inventory.cases(center)))


@dataclass(frozen=True, slots=True)
class FinalDonorScope:
    """Legal donor population for one final target route ``H,c``."""

    target_center: str
    held_case_id: str
    route_witness: RouteScopeWitness
    case_inventory: DatasetCaseInventory
    scope_hash: str = field(init=False)

    def __post_init__(self) -> None:
        target = str(self.target_center)
        held = str(self.held_case_id)
        if (
            target not in CENTERS
            or not held
            or not isinstance(self.route_witness, RouteScopeWitness)
            or not isinstance(self.case_inventory, DatasetCaseInventory)
            or held not in self.case_inventory.cases(target)
            or self.route_witness.target_center != target
            or self.route_witness.held_case_id != held
            or self.route_witness.identity_inventory.case_inventory.inventory_hash
            != self.case_inventory.inventory_hash
        ):
            raise ProtocolError("SCALE-BP final H/c donor scope drifted.")
        payload = {
            "schema_version": "scale_bp_final_donor_scope_v3",
            "target_center": target,
            "held_case_id": held,
            "route_scope_witness_hash": self.route_witness.witness_hash,
            "evaluation_sample_key_hash": (
                self.route_witness.evaluation_binding.sample_key_hash
            ),
            "support_sample_key_hash": self.route_witness.support_sample_key_hash,
            "case_inventory_hash": self.case_inventory.inventory_hash,
            "donor_training_centers": self.donor_training_centers,
            "donor_training_case_ids": self.donor_training_case_ids,
            "source_excluded_centers": self.source_excluded_centers,
            "outer_H_excluded": True,
            "held_c_excluded": True,
            "pseudo_replay": False,
        }
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "held_case_id", held)
        object.__setattr__(self, "scope_hash", canonical_hash(payload))

    @property
    def route_scope_hash(self) -> str:
        return self.route_witness.witness_hash

    @property
    def donor_training_centers(self) -> tuple[str, ...]:
        return _ordered_centers(set(CENTERS) - {self.target_center})

    @property
    def donor_training_case_ids(self) -> tuple[str, ...]:
        return _donor_case_ids(self.case_inventory, self.donor_training_centers)

    @property
    def source_excluded_centers(self) -> tuple[str, ...]:
        return (self.target_center,)

    @property
    def fit_role(self) -> str:
        return "FINAL_H_C"

    @property
    def prediction_center(self) -> str:
        return self.target_center


@dataclass(frozen=True, slots=True)
class PseudoReplayScope:
    """Legal pseudo-case replay population for one ``H,J,d`` triple."""

    outer_center: str
    pseudo_center: str
    held_case_id: str
    route_witness: RouteScopeWitness
    case_inventory: DatasetCaseInventory
    scope_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = str(self.outer_center)
        pseudo = str(self.pseudo_center)
        held = str(self.held_case_id)
        if (
            outer not in CENTERS
            or pseudo not in CENTERS
            or outer == pseudo
            or not held
            or not isinstance(self.route_witness, RouteScopeWitness)
            or not isinstance(self.case_inventory, DatasetCaseInventory)
            or held not in self.case_inventory.cases(pseudo)
            or self.route_witness.target_center != pseudo
            or self.route_witness.held_case_id != held
            or self.route_witness.identity_inventory.case_inventory.inventory_hash
            != self.case_inventory.inventory_hash
        ):
            raise ProtocolError("SCALE-BP H/J/d replay scope drifted.")
        payload = {
            "schema_version": "scale_bp_pseudo_replay_scope_v3",
            "outer_center": outer,
            "pseudo_center": pseudo,
            "held_case_id": held,
            "route_scope_witness_hash": self.route_witness.witness_hash,
            "evaluation_sample_key_hash": (
                self.route_witness.evaluation_binding.sample_key_hash
            ),
            "support_sample_key_hash": self.route_witness.support_sample_key_hash,
            "case_inventory_hash": self.case_inventory.inventory_hash,
            "donor_training_centers": self.donor_training_centers,
            "donor_training_case_ids": self.donor_training_case_ids,
            "support_case_ids": self.support_case_ids,
            "source_excluded_centers": self.source_excluded_centers,
            "outer_H_excluded": True,
            "pseudo_J_excluded": True,
            "held_d_excluded": True,
            "candidate_source_exclusions_recomputed": True,
            "pseudo_replay": True,
        }
        object.__setattr__(self, "outer_center", outer)
        object.__setattr__(self, "pseudo_center", pseudo)
        object.__setattr__(self, "held_case_id", held)
        object.__setattr__(self, "scope_hash", canonical_hash(payload))

    @property
    def support_scope_hash(self) -> str:
        return self.route_witness.witness_hash

    @property
    def route_scope_hash(self) -> str:
        """Target-local ``J\\d`` route identity used by support cross-fitting."""

        return self.route_witness.witness_hash

    @property
    def donor_training_centers(self) -> tuple[str, ...]:
        return _ordered_centers(set(CENTERS) - {self.outer_center, self.pseudo_center})

    @property
    def donor_training_case_ids(self) -> tuple[str, ...]:
        return _donor_case_ids(self.case_inventory, self.donor_training_centers)

    @property
    def support_case_ids(self) -> tuple[str, ...]:
        return tuple(
            case
            for case in self.case_inventory.cases(self.pseudo_center)
            if case != self.held_case_id
        )

    @property
    def source_excluded_centers(self) -> tuple[str, ...]:
        return _ordered_centers({self.outer_center, self.pseudo_center})

    @property
    def fit_role(self) -> str:
        return "PSEUDO_H_J_D"

    @property
    def prediction_center(self) -> str:
        return self.pseudo_center


DonorScope = FinalDonorScope | PseudoReplayScope


__all__ = ("DonorScope", "FinalDonorScope", "PseudoReplayScope")
