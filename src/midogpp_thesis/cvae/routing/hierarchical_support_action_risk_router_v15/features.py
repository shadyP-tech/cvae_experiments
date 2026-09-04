"""Predeclared, label-free feature map for the HARP v15 support router."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import ActionFamily, Direction, LabelFreeAction, canonical_text
from .hashing import canonical_hash


# Ordered before outcomes are opened.  The list favors mechanism-facing
# quantities that exist on both train-support and test-target physical menus.
MECHANISM_FEATURE_PRIORITY = (
    "active_mask_fraction",
    "threshold_flip_fraction",
    "direction_aligned_branch_mass",
    "action_delta_mean",
    "action_delta_std",
    "action_delta_abs_mean",
    "baseline_probability_mean",
    "baseline_positive_branch_fraction",
    "baseline_boundary_distance_mean",
    "baseline_boundary_distance_min",
    "surface_boundary_distance_mean",
    "surface_boundary_distance_min",
    "boundary_distance_change_mean",
    "baseline_seed_dispersion_mean",
    "surface_seed_dispersion_mean",
    "surface_seed_dispersion_change_mean",
    "compatibility_mean_z",
    "compatibility_std_z",
    "compatibility_reciprocal_rank",
    "compatibility_rank_margin",
    "compatibility_available",
    "geometry_action_maximum_source_weight",
    "geometry_action_effective_source_count",
    "geometry_density_excess_over_quarter",
    "geometry_effective_sources_shortfall_from_six",
)


def _common_schema(actions: Sequence[LabelFreeAction]) -> tuple[str, ...]:
    rows = tuple(actions)
    if not rows:
        raise ProtocolError("HARP v15 feature fitting requires support actions.")
    schema = rows[0].feature_names
    if any(row.feature_names != schema for row in rows):
        raise ProtocolError("HARP v15 support actions do not share one feature schema.")
    return schema


def _case_balanced_weights(actions: Sequence[LabelFreeAction]) -> np.ndarray:
    counts: dict[str, int] = {}
    for row in actions:
        counts[row.case_id] = counts.get(row.case_id, 0) + 1
    return np.asarray([1.0 / counts[row.case_id] for row in actions], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class FittedFeatureMap:
    numeric_feature_names: tuple[str, ...]
    numeric_means: tuple[float, ...]
    numeric_scales: tuple[float, ...]
    candidate_source_ids: tuple[str, ...]
    fitted_case_ids: tuple[str, ...]
    feature_map_hash: str = field(init=False)

    def __post_init__(self) -> None:
        names = tuple(self.numeric_feature_names)
        means = tuple(float(value) for value in self.numeric_means)
        scales = tuple(float(value) for value in self.numeric_scales)
        sources = tuple(sorted(str(value) for value in self.candidate_source_ids))
        cases = tuple(sorted(str(value) for value in self.fitted_case_ids))
        if (
            not names
            or len(names) != len(set(names))
            or len(names) != len(means)
            or len(names) != len(scales)
            or any(not math.isfinite(value) for value in (*means, *scales))
            or any(value <= 0.0 for value in scales)
            or len(sources) != len(set(sources))
            or not cases
            or len(cases) != len(set(cases))
        ):
            raise ProtocolError("HARP v15 fitted feature map is malformed.")
        object.__setattr__(self, "numeric_feature_names", names)
        object.__setattr__(self, "numeric_means", means)
        object.__setattr__(self, "numeric_scales", scales)
        object.__setattr__(self, "candidate_source_ids", sources)
        object.__setattr__(self, "fitted_case_ids", cases)
        object.__setattr__(
            self,
            "feature_map_hash",
            canonical_hash(
                {
                    "schema_version": "hierarchical_support_feature_map_v15",
                    "numeric_feature_names": names,
                    "numeric_means": means,
                    "numeric_scales": scales,
                    "candidate_source_ids": sources,
                    "candidate_universe_declared_before_labels": True,
                    "fitted_case_ids": cases,
                    "feature_selection_outcome_adaptive": False,
                    "normalization_fit_on_support_only": True,
                    "case_balanced_normalization": True,
                }
            ),
        )

    @property
    def vector_names(self) -> tuple[str, ...]:
        return (
            "intercept",
            *(f"numeric::{name}" for name in self.numeric_feature_names),
            "direction::D10",
            "family::HXE",
            *(f"candidate::{source}" for source in self.candidate_source_ids),
        )

    def transform(self, action: LabelFreeAction) -> np.ndarray:
        if not isinstance(action, LabelFreeAction):
            raise ProtocolError("HARP v15 feature transforms require a label-free action.")
        values = dict(zip(action.feature_names, action.feature_values, strict=True))
        if any(name not in values for name in self.numeric_feature_names):
            raise ProtocolError("HARP v15 target feature schema drifted from support.")
        if (
            action.candidate_source_id is not None
            and action.candidate_source_id not in self.candidate_source_ids
        ):
            raise ProtocolError("HARP v15 target menu contains an unseen candidate expert.")
        numeric = [
            (values[name] - mean) / scale
            for name, mean, scale in zip(
                self.numeric_feature_names,
                self.numeric_means,
                self.numeric_scales,
                strict=True,
            )
        ]
        vector = np.asarray(
            [
                1.0,
                *numeric,
                float(action.direction is Direction.D10),
                float(action.family is ActionFamily.HXE),
                *(
                    float(action.candidate_source_id == source)
                    for source in self.candidate_source_ids
                ),
            ],
            dtype=np.float64,
        )
        if vector.shape != (len(self.vector_names),) or not np.isfinite(vector).all():
            raise ProtocolError("HARP v15 transformed features are malformed.")
        return vector

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "hierarchical_support_feature_map_v15",
            "feature_map_hash": self.feature_map_hash,
            "numeric_feature_names": list(self.numeric_feature_names),
            "numeric_means": list(self.numeric_means),
            "numeric_scales": list(self.numeric_scales),
            "candidate_source_ids": list(self.candidate_source_ids),
            "candidate_universe_declared_before_labels": True,
            "fitted_case_ids": list(self.fitted_case_ids),
            "vector_names": list(self.vector_names),
            "feature_selection_outcome_adaptive": False,
            "normalization_fit_on_support_only": True,
            "case_balanced_normalization": True,
        }


def fit_feature_map(
    actions: Sequence[LabelFreeAction],
    *,
    maximum_numeric_features: int,
    candidate_source_ids: Sequence[str] | None = None,
    fitted_case_ids: Sequence[str] | None = None,
) -> FittedFeatureMap:
    rows = tuple(actions)
    schema = _common_schema(rows)
    if int(maximum_numeric_features) < 1:
        raise ProtocolError("HARP v15 numeric feature limit must be positive.")
    selected = [name for name in MECHANISM_FEATURE_PRIORITY if name in schema]
    selected.extend(name for name in sorted(schema) if name not in selected)
    names = tuple(selected[: int(maximum_numeric_features)])
    positions = {name: index for index, name in enumerate(schema)}
    matrix = np.asarray(
        [[row.feature_values[positions[name]] for name in names] for row in rows],
        dtype=np.float64,
    )
    weights = _case_balanced_weights(rows)
    weight_sum = float(np.sum(weights, dtype=np.float64))
    means = np.sum(matrix * weights[:, None], axis=0, dtype=np.float64) / weight_sum
    variance = (
        np.sum(((matrix - means) ** 2) * weights[:, None], axis=0, dtype=np.float64)
        / weight_sum
    )
    scales = np.sqrt(np.maximum(variance, 0.0).astype(np.float64, copy=False))
    scales = np.where(scales > 1e-12, scales, 1.0)
    observed_sources = {
        row.candidate_source_id
        for row in rows
        if row.candidate_source_id is not None
    }
    declared_sources = tuple(
        sorted(
            canonical_text(value, name="declared candidate source")
            for value in (
                observed_sources
                if candidate_source_ids is None
                else tuple(candidate_source_ids)
            )
        )
    )
    declared_cases = tuple(
        sorted(
            canonical_text(value, name="fitted support case id")
            for value in (
                {row.case_id for row in rows}
                if fitted_case_ids is None
                else tuple(fitted_case_ids)
            )
        )
    )
    if (
        len(declared_sources) != len(set(declared_sources))
        or not observed_sources.issubset(declared_sources)
        or len(declared_cases) != len(set(declared_cases))
        or not {row.case_id for row in rows}.issubset(declared_cases)
    ):
        raise ProtocolError(
            "HARP v15 declared candidate or support-case universe is incomplete."
        )
    return FittedFeatureMap(
        numeric_feature_names=names,
        numeric_means=tuple(float(value) for value in means.tolist()),
        numeric_scales=tuple(float(value) for value in scales.tolist()),
        candidate_source_ids=declared_sources,
        fitted_case_ids=declared_cases,
    )


def case_balanced_weights(actions: Sequence[LabelFreeAction]) -> tuple[float, ...]:
    """Expose the audited weights used by every v15 endpoint head."""

    return tuple(float(value) for value in _case_balanced_weights(tuple(actions)))


__all__ = (
    "FittedFeatureMap",
    "MECHANISM_FEATURE_PRIORITY",
    "case_balanced_weights",
    "fit_feature_map",
)
