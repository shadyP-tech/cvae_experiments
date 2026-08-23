"""Closed-world preterminal seal over every fixed-menu P-DCAPS output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from .engine import OuterActionPolicyResult
from .identity import (
    CYCLIC_METHOD_ID,
    LEGACY_METHOD_ID,
    METHOD_MENU,
    PRIMARY_METHOD_ID,
    canonical_hash,
    require_sha256,
)
from .legacy_control import LegacyControlSeal, resolve_legacy_control
from .method_controls import ComposedMethodPrediction, MethodControlDecision
from .surface_set import SealedActionSurfaceSet
from .target_local_runtime import POSTERIOR_CONTROL_IDS


@dataclass(frozen=True)
class PreterminalOutputHashes:
    """Hashes for both controls and all six methods, frozen before target labels."""

    action_surface_set_seal_hash: str
    action_surface_seals: tuple[tuple[str, str], ...]
    expected_inventory_hash: str
    control_result_hashes: tuple[tuple[str, str, str], ...]
    legacy_control_seal_hashes: tuple[tuple[str, str, str], ...]
    method_decision_hashes: tuple[tuple[str, str, str], ...]
    method_composition_hashes: tuple[tuple[str, str, str], ...]
    output_bundle_hash: str = field(init=False)

    def __post_init__(self) -> None:
        set_hash = require_sha256(
            self.action_surface_set_seal_hash, "action surface set seal"
        )
        inventory_hash = require_sha256(
            self.expected_inventory_hash, "expected route inventory hash"
        )
        surface_seals = tuple(
            (str(control), require_sha256(value, "action surface seal"))
            for control, value in self.action_surface_seals
        )
        result_rows = _control_rows(
            self.control_result_hashes, role="control result"
        )
        legacy_rows = _control_rows(
            self.legacy_control_seal_hashes, role="legacy control seal"
        )
        decision_rows = _method_rows(
            self.method_decision_hashes, role="method decision"
        )
        composition_rows = _method_rows(
            self.method_composition_hashes, role="method composition"
        )
        centers = tuple(
            center
            for control, center, _value in result_rows
            if control == POSTERIOR_CONTROL_IDS[0]
        )
        expected_control_keys = tuple(
            (control, center)
            for control in POSTERIOR_CONTROL_IDS
            for center in centers
        )
        expected_method_keys = tuple(
            (center, method) for center in centers for method in METHOD_MENU
        )
        if (
            tuple(control for control, _value in surface_seals)
            != POSTERIOR_CONTROL_IDS
            or not centers
            or centers != tuple(center for center in CENTERS if center in set(centers))
            or tuple((control, center) for control, center, _value in result_rows)
            != expected_control_keys
            or tuple((control, center) for control, center, _value in legacy_rows)
            != expected_control_keys
            or tuple((center, method) for center, method, _value in decision_rows)
            != expected_method_keys
            or tuple((center, method) for center, method, _value in composition_rows)
            != expected_method_keys
        ):
            raise ProtocolError("P-DCAPS preterminal fixed-menu inventory drifted.")
        object.__setattr__(self, "action_surface_set_seal_hash", set_hash)
        object.__setattr__(self, "action_surface_seals", surface_seals)
        object.__setattr__(self, "expected_inventory_hash", inventory_hash)
        object.__setattr__(self, "control_result_hashes", result_rows)
        object.__setattr__(self, "legacy_control_seal_hashes", legacy_rows)
        object.__setattr__(self, "method_decision_hashes", decision_rows)
        object.__setattr__(self, "method_composition_hashes", composition_rows)
        object.__setattr__(
            self,
            "output_bundle_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_preterminal_output_hashes_v3",
                    "action_surface_set_seal_hash": set_hash,
                    "action_surface_seals": surface_seals,
                    "expected_inventory_hash": inventory_hash,
                    "control_result_hashes": result_rows,
                    "legacy_control_seal_hashes": legacy_rows,
                    "method_decision_hashes": decision_rows,
                    "method_composition_hashes": composition_rows,
                    "complete_fixed_method_menu": True,
                    "target_labels_used": False,
                }
            ),
        )

    @property
    def centers(self) -> tuple[str, ...]:
        return tuple(
            center
            for control, center, _value in self.control_result_hashes
            if control == POSTERIOR_CONTROL_IDS[0]
        )

    def to_payload(self) -> dict[str, object]:
        """Return the complete target-label-free attestation payload."""

        return {
            "schema_version": "pdcaps_preterminal_output_hashes_v3",
            "action_surface_set_seal_hash": self.action_surface_set_seal_hash,
            "action_surface_seals": [list(row) for row in self.action_surface_seals],
            "expected_inventory_hash": self.expected_inventory_hash,
            "control_result_hashes": [list(row) for row in self.control_result_hashes],
            "legacy_control_seal_hashes": [
                list(row) for row in self.legacy_control_seal_hashes
            ],
            "method_decision_hashes": [list(row) for row in self.method_decision_hashes],
            "method_composition_hashes": [
                list(row) for row in self.method_composition_hashes
            ],
            "complete_fixed_method_menu": True,
            "target_labels_used": False,
            "output_bundle_hash": self.output_bundle_hash,
        }

    @classmethod
    def from_runtime(
        cls,
        *,
        surface_set: SealedActionSurfaceSet,
        identity_results: Sequence[OuterActionPolicyResult],
        cyclic_results: Sequence[OuterActionPolicyResult],
        identity_legacy_controls: Sequence[LegacyControlSeal],
        cyclic_legacy_controls: Sequence[LegacyControlSeal],
        method_decisions: Sequence[MethodControlDecision],
        method_compositions: Sequence[ComposedMethodPrediction],
    ) -> "PreterminalOutputHashes":
        """Recompute the complete typed lineage instead of trusting hash tables."""

        if (
            not isinstance(surface_set, SealedActionSurfaceSet)
            or surface_set.control_ids != POSTERIOR_CONTROL_IDS
        ):
            raise ProtocolError("P-DCAPS preterminal requires both sealed controls.")
        results_by_control = {
            POSTERIOR_CONTROL_IDS[0]: _result_map(identity_results),
            POSTERIOR_CONTROL_IDS[1]: _result_map(cyclic_results),
        }
        centers = tuple(
            center for center in CENTERS if center in results_by_control["IDENTITY"]
        )
        if (
            not centers
            or set(results_by_control["IDENTITY"]) != set(centers)
            or set(results_by_control["WITHIN_CASE_CYCLIC_SHIFT"]) != set(centers)
        ):
            raise ProtocolError("P-DCAPS preterminal result center inventory drifted.")
        for control_id, result_by_center in results_by_control.items():
            surface = surface_set.surface(control_id)
            for center in centers:
                result = result_by_center[center]
                if (
                    result.posterior_control_id != control_id
                    or result.action_surface_seal_hash
                    != surface.action_surface_seal_hash
                    or result.physical_surface_hash != surface.physical_surface_hash
                ):
                    raise ProtocolError("P-DCAPS preterminal result surface drifted.")

        legacy_by_control = {
            POSTERIOR_CONTROL_IDS[0]: _legacy_map(identity_legacy_controls),
            POSTERIOR_CONTROL_IDS[1]: _legacy_map(cyclic_legacy_controls),
        }
        for control_id, controls in legacy_by_control.items():
            if set(controls) != set(centers):
                raise ProtocolError("P-DCAPS preterminal legacy inventory drifted.")
            for center in centers:
                resolve_legacy_control(
                    results_by_control[control_id][center], controls[center]
                )

        decisions = tuple(method_decisions)
        decision_by_key = {
            (row.outer_center, row.method_id): row for row in decisions
        }
        compositions = tuple(method_compositions)
        composition_by_key = {
            (row.decision.outer_center, row.decision.method_id): row
            for row in compositions
        }
        expected_method_keys = tuple(
            (center, method) for center in centers for method in METHOD_MENU
        )
        if (
            len(decision_by_key) != len(decisions)
            or len(composition_by_key) != len(compositions)
            or tuple(decision_by_key) != expected_method_keys
            or tuple(composition_by_key) != expected_method_keys
        ):
            raise ProtocolError("P-DCAPS preterminal method inventory drifted.")
        for center, method in expected_method_keys:
            decision = decision_by_key[(center, method)]
            composition = composition_by_key[(center, method)]
            identity = results_by_control[POSTERIOR_CONTROL_IDS[0]][center]
            source_control = (
                POSTERIOR_CONTROL_IDS[1]
                if method == CYCLIC_METHOD_ID
                else POSTERIOR_CONTROL_IDS[0]
            )
            source = results_by_control[source_control][center]
            expected_legacy = (
                legacy_by_control[source_control][center]
                if method
                in {
                    PRIMARY_METHOD_ID,
                    LEGACY_METHOD_ID,
                    CYCLIC_METHOD_ID,
                }
                else None
            )
            if (
                decision.identity_result_hash != identity.result_hash
                or decision.source_result_hash != source.result_hash
                or decision.legacy_control_seal_hash
                != (
                    None
                    if expected_legacy is None
                    else expected_legacy.legacy_control_seal_hash
                )
                or decision.joint_surface_set_seal_hash
                != (
                    surface_set.surface_set_seal_hash
                    if source_control == POSTERIOR_CONTROL_IDS[1]
                    else None
                )
                or composition.decision.decision_hash != decision.decision_hash
            ):
                raise ProtocolError("P-DCAPS preterminal method lineage drifted.")

        return cls(
            surface_set.surface_set_seal_hash,
            tuple(
                (surface.posterior_control_id, surface.action_surface_seal_hash)
                for surface in surface_set.surfaces
            ),
            surface_set.expected_inventory_hash,
            tuple(
                (control, center, results_by_control[control][center].result_hash)
                for control in POSTERIOR_CONTROL_IDS
                for center in centers
            ),
            tuple(
                (
                    control,
                    center,
                    legacy_by_control[control][center].legacy_control_seal_hash,
                )
                for control in POSTERIOR_CONTROL_IDS
                for center in centers
            ),
            tuple(
                (center, method, decision_by_key[(center, method)].decision_hash)
                for center, method in expected_method_keys
            ),
            tuple(
                (
                    center,
                    method,
                    composition_by_key[(center, method)].method_composition_hash,
                )
                for center, method in expected_method_keys
            ),
        )


def _control_rows(
    rows: Sequence[tuple[str, str, str]], *, role: str
) -> tuple[tuple[str, str, str], ...]:
    output = tuple(
        (str(control), str(center), require_sha256(value, role))
        for control, center, value in rows
    )
    if len(output) != len(set((control, center) for control, center, _ in output)):
        raise ProtocolError(f"P-DCAPS duplicate {role} inventory.")
    return output


def _method_rows(
    rows: Sequence[tuple[str, str, str]], *, role: str
) -> tuple[tuple[str, str, str], ...]:
    output = tuple(
        (str(center), str(method), require_sha256(value, role))
        for center, method, value in rows
    )
    if len(output) != len(set((center, method) for center, method, _ in output)):
        raise ProtocolError(f"P-DCAPS duplicate {role} inventory.")
    return output


def _result_map(
    rows: Sequence[OuterActionPolicyResult],
) -> dict[str, OuterActionPolicyResult]:
    values = tuple(rows)
    if any(not isinstance(row, OuterActionPolicyResult) for row in values):
        raise ProtocolError("P-DCAPS preterminal result DTO drifted.")
    output = {row.outer_center: row for row in values}
    if len(output) != len(values):
        raise ProtocolError("P-DCAPS preterminal result inventory duplicated.")
    return output


def _legacy_map(rows: Sequence[LegacyControlSeal]) -> dict[str, LegacyControlSeal]:
    values = tuple(rows)
    if any(not isinstance(row, LegacyControlSeal) for row in values):
        raise ProtocolError("P-DCAPS preterminal legacy DTO drifted.")
    output = {row.surface.outer_center: row for row in values}
    if len(output) != len(values):
        raise ProtocolError("P-DCAPS preterminal legacy inventory duplicated.")
    return output


__all__ = ("PreterminalOutputHashes",)
