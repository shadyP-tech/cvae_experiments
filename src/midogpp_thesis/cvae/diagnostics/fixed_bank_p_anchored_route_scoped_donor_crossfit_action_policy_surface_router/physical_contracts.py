"""Immutable physical prediction surfaces for P-DCAPS."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_array
from ...runtime.fixed_bank_a1_action_predictions import GlobalPredictionSeal
from ...runtime.frozen_source_streams import FrozenSourceStreamCache
from .identity import canonical_hash
from .physical_actions import action_library_by_target


@dataclass(frozen=True)
class CenterPhysicalSurface:
    """Canonical exact-nine action predictions for one target center."""

    center: str
    sample_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    seed_probabilities: tuple[tuple[str, np.ndarray], ...]
    prediction_store_hash: str
    center_surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.center not in CENTERS:
            raise ProtocolError("P-DCAPS center surface topology drifted.")
        samples = tuple(str(value) for value in self.sample_ids)
        cases = tuple(str(value) for value in self.case_ids)
        arrays: list[tuple[str, np.ndarray]] = []
        for action_id, values in self.seed_probabilities:
            array = np.ascontiguousarray(values, dtype=np.float32)
            if array.shape != (9, len(samples)) or not np.isfinite(array).all():
                raise ProtocolError("P-DCAPS physical action array drifted.")
            array.setflags(write=False)
            arrays.append((str(action_id), array))
        expected_actions = tuple(
            action.action_id for action in action_library_by_target()[self.center]
        )
        if (
            len(samples) != len(cases)
            or not samples
            or tuple(action for action, _ in arrays) != expected_actions
        ):
            raise ProtocolError("P-DCAPS center surface topology drifted.")
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "seed_probabilities", tuple(arrays))
        object.__setattr__(
            self,
            "center_surface_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_center_physical_surface_v1",
                    "center": self.center,
                    "sample_ids": samples,
                    "case_ids": cases,
                    "arrays": [
                        [action, sha256_array(values)] for action, values in arrays
                    ],
                    "prediction_store_hash": self.prediction_store_hash,
                    "labels_used": False,
                }
            ),
        )

    def exact_nine_mean(self, action_id: str) -> np.ndarray:
        try:
            values = dict(self.seed_probabilities)[str(action_id)]
        except KeyError as exc:
            raise ProtocolError("P-DCAPS physical action is absent.") from exc
        result = np.ascontiguousarray(
            np.mean(values.astype(np.float64), axis=0, dtype=np.float64),
            dtype=np.float32,
        )
        result.setflags(write=False)
        return result

    def positions(self, case_id: object) -> np.ndarray:
        result = np.flatnonzero(np.asarray(self.case_ids) == str(case_id))
        if not len(result):
            raise ProtocolError("P-DCAPS physical case is absent.")
        return result


@dataclass(frozen=True)
class PhysicalSurface:
    """Canonical physical surfaces for all nine target centers."""

    centers: tuple[CenterPhysicalSurface, ...]
    prediction_store_hash: str
    physical_surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if tuple(row.center for row in self.centers) != CENTERS:
            raise ProtocolError("P-DCAPS physical center order drifted.")
        object.__setattr__(
            self,
            "physical_surface_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_physical_surface_v1",
                    "prediction_store_hash": self.prediction_store_hash,
                    "center_surface_hashes": [
                        [row.center, row.center_surface_hash] for row in self.centers
                    ],
                    "labels_used": False,
                }
            ),
        )

    def center(self, center: object) -> CenterPhysicalSurface:
        try:
            return {row.center: row for row in self.centers}[str(center)]
        except KeyError as exc:
            raise ProtocolError("P-DCAPS physical center is absent.") from exc


@dataclass(frozen=True)
class MaterializedPhysicalBank:
    """Neutral source cache, global prediction seal, and typed surface bundle."""

    source_cache: FrozenSourceStreamCache
    prediction: GlobalPredictionSeal
    surface: PhysicalSurface


__all__ = (
    "CenterPhysicalSurface",
    "MaterializedPhysicalBank",
    "PhysicalSurface",
)
