"""Corrected policy-cell contracts and deterministic exact-P selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ....protocol import ProtocolError
from ..contracts import FavorableUtility
from ..identity import TIE_TOLERANCE, canonical_hash
from .contracts import PrefixCell, PrefixSurface


@dataclass(frozen=True)
class CalibratedPrefixCell:
    cell: PrefixCell
    model_predicted_utility: FavorableUtility
    envelope_correction: FavorableUtility
    corrected_utility: FavorableUtility
    policy_calibration_hash: str
    policy_envelope_hash: str
    correction_applied_count: int
    calibrated_cell_hash: str = field(init=False)

    def __post_init__(self) -> None:
        expected = np.asarray(self.model_predicted_utility.as_tuple()) - np.asarray(
            self.envelope_correction.as_tuple()
        )
        if self.cell.k == 0:
            valid = (
                self.correction_applied_count == 0
                and self.model_predicted_utility == FavorableUtility.zeros()
                and self.envelope_correction == FavorableUtility.zeros()
                and self.corrected_utility == FavorableUtility.zeros()
            )
        else:
            valid = self.correction_applied_count == 1 and np.allclose(
                expected,
                np.asarray(self.corrected_utility.as_tuple()),
                rtol=0.0,
                atol=1.0e-15,
            )
        if not valid:
            raise ProtocolError("P-DCAPS policy correction count or value drifted.")
        object.__setattr__(
            self,
            "calibrated_cell_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_calibrated_prefix_cell_v1",
                    "cell_hash": self.cell.cell_hash,
                    "model_predicted_utility": self.model_predicted_utility.to_payload(),
                    "envelope_correction": self.envelope_correction.to_payload(),
                    "corrected_utility": self.corrected_utility.to_payload(),
                    "policy_calibration_hash": self.policy_calibration_hash,
                    "policy_envelope_hash": self.policy_envelope_hash,
                    "correction_applied_count": self.correction_applied_count,
                }
            ),
        )

    @property
    def feasible(self) -> bool:
        if self.cell.k == 0:
            return True
        value = self.corrected_utility
        return value.bacc_gain > 0.0 and value.brier_gain >= 0.0 and value.log_gain >= 0.0

    @property
    def reason_codes(self) -> tuple[str, ...]:
        if self.cell.k == 0:
            return ("EXACT_P_PROTECTED",)
        reasons = []
        if self.corrected_utility.bacc_gain <= 0.0:
            reasons.append("NONPOSITIVE_CORRECTED_BACC")
        if self.corrected_utility.brier_gain < 0.0:
            reasons.append("NEGATIVE_CORRECTED_BRIER_GAIN")
        if self.corrected_utility.log_gain < 0.0:
            reasons.append("NEGATIVE_CORRECTED_LOG_GAIN")
        return ("CORRECTED_POLICY_PASS",) if not reasons else tuple(reasons)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_calibrated_prefix_cell_v1",
            "cell": self.cell.to_payload(),
            "model_predicted_utility": self.model_predicted_utility.to_payload(),
            "envelope_correction": self.envelope_correction.to_payload(),
            "corrected_utility": self.corrected_utility.to_payload(),
            "policy_calibration_hash": self.policy_calibration_hash,
            "policy_envelope_hash": self.policy_envelope_hash,
            "correction_applied_count": self.correction_applied_count,
            "feasible": self.feasible,
            "reason_codes": list(self.reason_codes),
            "calibrated_cell_hash": self.calibrated_cell_hash,
        }


@dataclass(frozen=True)
class PolicySelection:
    surface_hash: str
    calibrated_cells: tuple[CalibratedPrefixCell, ...]
    selected_k: int
    selected_calibrated_cell_hash: str
    selection_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows = tuple(self.calibrated_cells)
        if (
            tuple(row.cell.k for row in rows) != tuple(range(len(rows)))
            or not 0 <= self.selected_k < len(rows)
            or rows[self.selected_k].calibrated_cell_hash
            != self.selected_calibrated_cell_hash
            or not rows[self.selected_k].feasible
        ):
            raise ProtocolError("P-DCAPS policy selection topology drifted.")
        object.__setattr__(self, "calibrated_cells", rows)
        object.__setattr__(
            self,
            "selection_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_policy_selection_v1",
                    "surface_hash": self.surface_hash,
                    "calibrated_cell_hashes": tuple(
                        row.calibrated_cell_hash for row in rows
                    ),
                    "selected_k": self.selected_k,
                    "selected_calibrated_cell_hash": self.selected_calibrated_cell_hash,
                    "exact_p_fallback_required": True,
                }
            ),
        )

    @property
    def authorized(self) -> bool:
        return self.selected_k > 0

    @property
    def selected_cell(self) -> CalibratedPrefixCell:
        return self.calibrated_cells[self.selected_k]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_policy_selection_v1",
            "surface_hash": self.surface_hash,
            "calibrated_cells": [row.to_payload() for row in self.calibrated_cells],
            "selected_k": self.selected_k,
            "selected_calibrated_cell_hash": self.selected_calibrated_cell_hash,
            "authorized": self.authorized,
            "selection_hash": self.selection_hash,
        }


def select_policy_prefix(
    surface: PrefixSurface,
    calibrated_cells: Sequence[CalibratedPrefixCell],
    *,
    tolerance: float = TIE_TOLERANCE,
) -> PolicySelection:
    rows = tuple(calibrated_cells)
    if (
        not np.isfinite(float(tolerance))
        or float(tolerance) < 0.0
        or len(rows) != len(surface.cells)
        or tuple(row.cell.cell_hash for row in rows)
        != tuple(cell.cell_hash for cell in surface.cells)
    ):
        raise ProtocolError("P-DCAPS policy selection input drifted.")
    feasible = tuple(row for row in rows if row.feasible)
    if not feasible or feasible[0].cell.k != 0:
        raise ProtocolError("P-DCAPS exact P is absent from policy selection.")
    maximum = max(row.corrected_utility.bacc_gain for row in feasible)
    tied = tuple(
        row
        for row in feasible
        if abs(row.corrected_utility.bacc_gain - maximum) <= float(tolerance)
    )
    selected = min(tied, key=lambda row: (row.cell.k, row.calibrated_cell_hash))
    return PolicySelection(
        surface.surface_hash,
        rows,
        selected.cell.k,
        selected.calibrated_cell_hash,
    )


__all__ = (
    "CalibratedPrefixCell",
    "PolicySelection",
    "select_policy_prefix",
)
