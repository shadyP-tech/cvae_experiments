"""Workstation resource contract and fail-before-allocation estimates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.preflight import run_label_free_workstation_preflight
from .identity import canonical_hash


@dataclass(frozen=True)
class WorkstationEstimate:
    outer_centers: int
    pseudo_routes: int
    attempted_action_cells: int
    maximum_crossing_actions: int
    maximum_prefix_cells: int
    action_ridge_fits: int
    policy_ridge_fits: int
    estimated_dense_bytes: int
    estimate_hash: str = field(init=False)

    @property
    def estimate_role(self) -> str:
        """Describe the deliberately conservative scope of the byte estimate."""

        return (
            "pessimistic_dense_calibration_record_cap_"
            "excluding_process_and_library_overhead"
        )

    def __post_init__(self) -> None:
        values = (
            self.outer_centers,
            self.pseudo_routes,
            self.attempted_action_cells,
            self.maximum_crossing_actions,
            self.maximum_prefix_cells,
            self.action_ridge_fits,
            self.policy_ridge_fits,
            self.estimated_dense_bytes,
        )
        if any(int(value) <= 0 for value in values):
            raise ProtocolError("P-DCAPS workstation estimate drifted.")
        object.__setattr__(
            self,
            "estimate_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_workstation_estimate_v1",
                    "values": values,
                    "estimate_role": self.estimate_role,
                    "measured_peak_memory": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "outer_centers": self.outer_centers,
            "pseudo_routes": self.pseudo_routes,
            "attempted_action_cells": self.attempted_action_cells,
            "maximum_crossing_actions": self.maximum_crossing_actions,
            "maximum_prefix_cells": self.maximum_prefix_cells,
            "action_ridge_fits": self.action_ridge_fits,
            "policy_ridge_fits": self.policy_ridge_fits,
            "estimated_dense_bytes": self.estimated_dense_bytes,
            "estimate_role": self.estimate_role,
            "measured_peak_memory": False,
            "estimate_hash": self.estimate_hash,
        }


def estimate_workstation_surface(
    *,
    case_count: int = 218,
    outer_centers: int = 9,
    action_strata: int = 6,
    feature_count: int = 14,
) -> WorkstationEstimate:
    pseudo_routes = (outer_centers - 1) * case_count
    attempted = pseudo_routes * action_strata
    maximum_prefix = pseudo_routes * (case_count + 1)
    donor_count = outer_centers - 1
    ridge_per_layer = outer_centers * 3 * (
        1 + donor_count + donor_count * (donor_count - 1) // 2
    )
    dense_bytes = (
        attempted * (feature_count * 8 + 3 * 8 + 8)
        + maximum_prefix * (12 * 8 + 3 * 8 + 8)
    )
    return WorkstationEstimate(
        outer_centers,
        pseudo_routes,
        attempted,
        attempted,
        maximum_prefix,
        ridge_per_layer,
        ridge_per_layer,
        dense_bytes,
    )


def run_workstation_preflight(
    root: Path,
    *,
    runtime: Mapping[str, object],
) -> Mapping[str, object]:
    estimate = estimate_workstation_surface()
    memory_limit = int(runtime.get("maximum_dense_surface_bytes", 0))
    if memory_limit <= 0 or estimate.estimated_dense_bytes > memory_limit:
        raise ProtocolError("P-DCAPS dense surface exceeds the frozen RAM budget.")
    payload = dict(
        run_label_free_workstation_preflight(
            root,
            runtime=runtime,
            expected_scratch_root=str(runtime["scratch_preference"][0]),
            expected_target_action_identity_count=90,
            expected_target_probability_cell_count=810,
            expected_unique_classifier_fit_count=810,
            expected_resume_policy=str(runtime["resume_policy"]),
        )
    )
    payload["pdcaps_surface_estimate"] = estimate.to_payload()
    payload["outer_process_workers"] = int(runtime["outer_process_workers"])
    payload["nested_process_pools"] = False
    return payload


__all__ = (
    "WorkstationEstimate",
    "estimate_workstation_surface",
    "run_workstation_preflight",
)
