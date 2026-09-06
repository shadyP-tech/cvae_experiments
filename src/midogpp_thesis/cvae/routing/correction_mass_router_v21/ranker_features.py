"""Source-scope transforms for pairwise donor features."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    CompositeKind,
    Direction,
    LabelFreeAction,
    LabelFreeCaseMenu,
    SupportActionOutcome,
    SupportCaseClassProfile,
    canonical_text,
)
from .hashing import canonical_hash

from .ranker_numerics import _DIRECTIONS, _case_weights


@dataclass(frozen=True, slots=True)
class FittedFeatureTransform:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    donor_ids: tuple[str, ...]
    training_case_keys: tuple[tuple[str, str], ...]
    transform_hash: str = field(init=False)

    def __post_init__(self) -> None:
        names = tuple(canonical_text(value, name="model feature name") for value in self.feature_names)
        means = tuple(float(value) for value in self.means)
        scales = tuple(float(value) for value in self.scales)
        donors = tuple(sorted(canonical_text(value, name="model donor id") for value in self.donor_ids))
        keys = tuple(sorted((canonical_text(c, name="training center"), canonical_text(k, name="training case")) for c, k in self.training_case_keys))
        if (
            not names
            or len(names) != len(set(names))
            or len(names) != len(means)
            or len(names) != len(scales)
            or any(not math.isfinite(value) for value in (*means, *scales))
            or any(value <= 0.0 for value in scales)
            or len(donors) != len(set(donors))
            or not keys
            or len(keys) != len(set(keys))
        ):
            raise ProtocolError("HARP v21 fitted feature transform is malformed.")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "donor_ids", donors)
        object.__setattr__(self, "training_case_keys", keys)
        object.__setattr__(
            self,
            "transform_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_pairwise_feature_transform_v21",
                    "feature_names": names,
                    "means": means,
                    "scales": scales,
                    "donor_ids": donors,
                    "training_case_keys": keys,
                    "center_identity_is_not_a_feature": True,
                    "center_case_equal_weighted": True,
                }
            ),
        )

    @property
    def pairwise_design_names(self) -> tuple[str, ...]:
        return (
            *(f"D01::{name}" for name in self.feature_names),
            *(f"D10::{name}" for name in self.feature_names),
            *(f"donor::{donor}" for donor in self.donor_ids),
        )

    def numeric(self, action: LabelFreeAction) -> np.ndarray:
        values = dict(zip(action.feature_names, action.feature_values, strict=True))
        if any(name not in values for name in self.feature_names):
            raise ProtocolError("HARP v21 action feature schema drifted from its fitted fold.")
        return (
            np.asarray([values[name] for name in self.feature_names], dtype=np.float64)
            - np.asarray(self.means, dtype=np.float64)
        ) / np.asarray(self.scales, dtype=np.float64)

    def action_vector(self, action: LabelFreeAction) -> np.ndarray:
        if action.direction not in _DIRECTIONS:
            raise ProtocolError("HARP v21 pairwise action has an unseen direction/donor.")
        numeric = self.numeric(action)
        zeros = np.zeros(len(numeric), dtype=np.float64)
        donor = np.asarray([float(action.donor_id == value) for value in self.donor_ids], dtype=np.float64)
        blocks = (numeric, zeros) if action.direction is Direction.D01 else (zeros, numeric)
        return np.concatenate((*blocks, donor)).astype(np.float64, copy=False)

    def public_payload(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "donor_ids": list(self.donor_ids),
            "training_case_keys": [list(value) for value in self.training_case_keys],
            "pairwise_design_names": list(self.pairwise_design_names),
            "transform_hash": self.transform_hash,
            "center_identity_is_not_a_feature": True,
            "center_case_equal_weighted": True,
        }


def fit_feature_transform(
    menus: Sequence[LabelFreeCaseMenu], *, maximum_numeric_features: int
) -> FittedFeatureTransform:
    rows = tuple(sorted(menus, key=lambda row: (row.center_id, row.case_id)))
    actions = tuple(action for menu in rows for action in menu.actions)
    if not rows or not actions or type(maximum_numeric_features) is not int or maximum_numeric_features < 1:
        raise ProtocolError("HARP v21 feature fitting requires directional source actions.")
    schemas = {action.feature_names for action in actions}
    if len(schemas) != 1:
        raise ProtocolError("HARP v21 source action feature schema is not singular.")
    schema = next(iter(schemas))
    names = tuple(schema[:maximum_numeric_features])
    positions = {name: index for index, name in enumerate(schema)}
    keys = tuple((row.center_id, row.case_id) for row in rows)
    case_weight = _case_weights(keys)
    action_counts = Counter((action.center_id, action.case_id) for action in actions)
    weights = np.asarray(
        [case_weight[(action.center_id, action.case_id)] / action_counts[(action.center_id, action.case_id)] for action in actions],
        dtype=np.float64,
    )
    matrix = np.asarray(
        [[action.feature_values[positions[name]] for name in names] for action in actions],
        dtype=np.float64,
    )
    weight_sum = float(np.sum(weights, dtype=np.float64))
    means = np.sum(weights[:, None] * matrix, axis=0, dtype=np.float64) / weight_sum
    variance = np.sum(weights[:, None] * (matrix - means) ** 2, axis=0, dtype=np.float64) / weight_sum
    scales = np.sqrt(np.maximum(variance, 0.0))
    scales[scales <= math.sqrt(np.finfo(np.float64).eps)] = 1.0
    donors = tuple(sorted({action.donor_id for action in actions if action.donor_id is not None}))
    return FittedFeatureTransform(
        feature_names=names,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        donor_ids=donors,
        training_case_keys=tuple(sorted(set(keys))),
    )
