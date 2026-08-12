"""Typed phase products shared by the science-runtime phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ...routing.threshold_flip_case_router import (
    DirectionSharedCalibration,
    QueryFixedEffectStaticFit,
    StaticSelection,
    TwoHeadRidgeModel,
)
from .products import DecisionBundle


@dataclass(frozen=True)
class DonorPhaseResult:
    """Nine independent H-specific ordinary and permutation models."""

    contribution_targets: tuple[Mapping[str, object], ...]
    models: tuple[Mapping[str, object], ...]
    seals: Mapping[str, Mapping[str, object]]
    permutation_payload: Mapping[str, object]
    model_by_target: Mapping[str, TwoHeadRidgeModel]
    permutation_model_by_target: Mapping[str, TwoHeadRidgeModel]
    global_selection_by_target: Mapping[str, StaticSelection]
    global_selection_fit_by_target: Mapping[str, QueryFixedEffectStaticFit]


@dataclass(frozen=True)
class DecisionPhaseResult:
    """All 45 pre-evaluation fold decisions and their typed replay inputs."""

    static_rows: tuple[Mapping[str, object], ...]
    calibration_rows: tuple[Mapping[str, object], ...]
    static_seal_payload: Mapping[str, object]
    calibration_seal_payload: Mapping[str, object]
    bundle: DecisionBundle
    static_by_fold: Mapping[tuple[str, int], Mapping[str, StaticSelection]]
    calibration_by_fold: Mapping[
        tuple[str, int], Mapping[str, DirectionSharedCalibration]
    ]
