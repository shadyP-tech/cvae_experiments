"""Label-free exact-nine physical-action fingerprints and blocked control."""

from __future__ import annotations

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    BLOCKED_FINGERPRINT_CONTROL_ID,
    FINGERPRINT_STATISTIC_IDS,
    HARD_THRESHOLD,
    PRIMARY_FINGERPRINT_CONTROL_ID,
    physical_action_ids,
)
from .contracts import CenterProbabilitySurface
from .hashing import canonical_hash
from .sample_influence_contracts import PhysicalFingerprintSurface


def fingerprint_feature_names(target_center: object) -> tuple[str, ...]:
    return tuple(
        f"{action}::{statistic}"
        for action in physical_action_ids(target_center)
        for statistic in FINGERPRINT_STATISTIC_IDS
    )


def build_physical_fingerprint_surface(
    surface: CenterProbabilitySurface,
) -> PhysicalFingerprintSurface:
    """Reduce every physical action's exact-nine probabilities without labels."""

    columns: list[np.ndarray] = []
    for action in physical_action_ids(surface.center):
        exact_nine = surface.seed_probabilities[action].astype(np.float64, copy=False)
        columns.extend(
            (
                np.mean(exact_nine, axis=0, dtype=np.float64),
                np.std(exact_nine, axis=0, ddof=0, dtype=np.float64),
                np.mean(exact_nine >= HARD_THRESHOLD, axis=0, dtype=np.float64),
            )
        )
    values = np.column_stack(columns)
    return PhysicalFingerprintSurface(
        surface.center,
        surface.sample_ids,
        surface.case_ids,
        fingerprint_feature_names(surface.center),
        values,
        surface.surface_hash,
        PRIMARY_FINGERPRINT_CONTROL_ID,
    )


def blocked_within_case_fingerprint(
    surface: PhysicalFingerprintSurface,
) -> PhysicalFingerprintSurface:
    """Rotate complete feature rows inside cases, preserving case topology.

    This deterministic negative control destroys sample-to-feature alignment
    without consulting labels, changing feature columns independently, or
    moving information between whole cases.
    """

    if surface.control_id != PRIMARY_FINGERPRINT_CONTROL_ID:
        raise ProtocolError("PCSI-PARC can block only the primary fingerprint.")
    values = np.array(surface.feature_values, dtype=np.float64, copy=True)
    for case_id in surface.cases:
        positions = surface.positions(case_id)
        if len(positions) <= 1:
            continue
        shift = 1 + (
            int(canonical_hash([surface.center, case_id, "PCSI_BLOCK"])[:16], 16)
            % (len(positions) - 1)
        )
        values[positions] = surface.feature_values[np.roll(positions, shift)]
    return PhysicalFingerprintSurface(
        surface.center,
        surface.sample_ids,
        surface.case_ids,
        surface.feature_names,
        values,
        surface.source_surface_hash,
        BLOCKED_FINGERPRINT_CONTROL_ID,
    )


__all__ = (
    "blocked_within_case_fingerprint",
    "build_physical_fingerprint_surface",
    "fingerprint_feature_names",
)
