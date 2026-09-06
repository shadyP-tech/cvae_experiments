"""Prespecified evidence comparison, frozen calibration split, and admission controls."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Sequence
from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256
from .contract_values import PROBABILITY_CLIP


@dataclass(frozen=True, slots=True)
class RouterFitConfig:
    outer_folds: int = 5
    inner_folds: int = 4
    opportunity_ridge_alphas: tuple[float, ...] = (1.0,)  # Serialized compatibility only.
    ranker_ridge_alphas: tuple[float, ...] = (1.0,)
    evidence_variants: tuple[str, ...] = ("baseline", "calibrated_baseline", "embedding_residual")
    evidence_variant: str = "embedding_residual"
    calibration_partition_folds: int = 3
    calibration_partition_index: int = 0
    winner_gate_ridge_alpha: float = 0.01
    k_values: tuple[int, ...] = (1, 2, 4)
    lambda_values: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    route_thresholds: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
    maximum_numeric_features: int = 20
    required_source_case_count: int | None = 216
    required_source_center_count: int | None = 9
    minimum_cases_per_center: int = 2
    minimum_routed_oof_cases: int = 18
    minimum_routed_oof_centers: int = 6
    minimum_routed_oof_cases_per_center: int = 2
    bootstrap_replicates: int = 1024
    bootstrap_alpha: float = 0.05
    bootstrap_seed: int = 21021
    probability_clip: float = PROBABILITY_CLIP

    def __post_init__(self) -> None:
        numeric_grids = (
            self.opportunity_ridge_alphas,
            self.ranker_ridge_alphas,
            self.lambda_values,
            self.route_thresholds,
        )
        if (
            not self.evidence_variants
            or len(self.evidence_variants) != len(set(self.evidence_variants))
            or any(v not in ("baseline", "calibrated_baseline", "embedding_residual") for v in self.evidence_variants)
            or self.evidence_variant not in self.evidence_variants
            or type(self.outer_folds) is not int
            or self.outer_folds < 2
            or type(self.inner_folds) is not int
            or self.inner_folds < 2
            or any(not grid for grid in numeric_grids)
            or any(not math.isfinite(value) for grid in numeric_grids for value in grid)
            or self.opportunity_ridge_alphas != (1.0,)
            or self.ranker_ridge_alphas != (1.0,)
            or self.winner_gate_ridge_alpha != 0.01
            or self.calibration_partition_folds != 3
            or self.calibration_partition_index != 0
            or any(value <= 0.0 for value in self.opportunity_ridge_alphas)
            or any(value <= 0.0 for value in self.ranker_ridge_alphas)
            or tuple(sorted(set(self.k_values))) != self.k_values
            or any(type(value) is not int or value < 1 for value in self.k_values)
            or tuple(sorted(set(self.lambda_values))) != self.lambda_values
            or any(not 0.0 < value <= 1.0 for value in self.lambda_values)
            or tuple(sorted(set(self.route_thresholds))) != self.route_thresholds
            or any(not 0.0 <= value <= 1.0 for value in self.route_thresholds)
            or type(self.maximum_numeric_features) is not int
            or self.maximum_numeric_features < 1
            or type(self.minimum_cases_per_center) is not int
            or self.minimum_cases_per_center < 1
            or type(self.minimum_routed_oof_cases) is not int
            or self.minimum_routed_oof_cases < 1
            or type(self.minimum_routed_oof_centers) is not int
            or self.minimum_routed_oof_centers < 1
            or type(self.minimum_routed_oof_cases_per_center) is not int
            or self.minimum_routed_oof_cases_per_center < 1
            or type(self.bootstrap_replicates) is not int
            or self.bootstrap_replicates < 32
            or not 0.0 < self.bootstrap_alpha < 0.5
            or type(self.bootstrap_seed) is not int
            or self.probability_clip != PROBABILITY_CLIP
        ):
            raise ProtocolError("HARP v21 router fit configuration is malformed.")
        for value, name in (
            (self.required_source_case_count, "required source case count"),
            (self.required_source_center_count, "required source center count"),
        ):
            if value is not None and (type(value) is not int or value < 2):
                raise ProtocolError(f"HARP v21 {name} is malformed.")

    def public_payload(self) -> dict[str, object]:
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in (
                (name, getattr(self, name)) for name in self.__dataclass_fields__
            )
        }


