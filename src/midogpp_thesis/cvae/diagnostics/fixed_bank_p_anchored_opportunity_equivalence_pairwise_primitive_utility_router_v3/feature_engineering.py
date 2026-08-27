"""Frozen label-free feature maps for OE-PPUR v3.

Only probability, margin, entropy, crossing, and disagreement evidence enters
the learned source models. Center, case, target, and identity values are used
solely as fold/join keys and never as predictors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.pairwise_primitive_utility import (
    ActionQuery,
    RowPosteriorObservation,
    assert_label_free_feature_names,
    canonical_sha256,
)
from .candidate_pools import ALL_ACTION_IDS, CANDIDATE_ACTION_IDS, P_ACTION_ID
from .source_supervision import DERIVED_FEATURE_DIM, SourceSupervisionRow


ROW_FEATURE_NAMES = (
    "protected_probability",
    "protected_margin",
    "candidate_probability_mean",
    "candidate_probability_std",
    "candidate_disagreement",
    "probability_entropy",
)
ACTION_FEATURE_NAMES = (
    "protected_probability_mean",
    "candidate_probability_mean",
    "probability_difference_mean",
    "probability_difference_std",
    "crossing_fraction",
    "candidate_disagreement",
)

assert len(ROW_FEATURE_NAMES) == DERIVED_FEATURE_DIM
assert len(ACTION_FEATURE_NAMES) == DERIVED_FEATURE_DIM
assert_label_free_feature_names(ROW_FEATURE_NAMES)
assert_label_free_feature_names(ACTION_FEATURE_NAMES)


@dataclass(frozen=True, slots=True)
class LabelFreeFeatureVector:
    names: tuple[str, ...]
    values: tuple[float, ...]
    feature_hash: str = field(init=False)

    def __post_init__(self) -> None:
        names = assert_label_free_feature_names(self.names)
        values = tuple(float(value) for value in self.values)
        if len(names) != len(values) or not all(math.isfinite(value) for value in values):
            raise ProtocolError("OE-PPUR v3 label-free feature vector drifted.")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self,
            "feature_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v3_label_free_feature_vector_v1",
                    "names": names,
                    "values": values,
                    "labels_used": False,
                    "identity_features_used": False,
                }
            ),
        )


def _binary_entropy(probability: float) -> float:
    value = min(max(float(probability), 1.0e-12), 1.0 - 1.0e-12)
    return float(-(value * math.log(value) + (1.0 - value) * math.log1p(-value)))


def build_row_features(
    action_probabilities: Sequence[float],
) -> LabelFreeFeatureVector:
    """Build six fixed row-posterior predictors from one seven-action row."""

    values = np.asarray(action_probabilities, dtype=np.float64)
    if (
        values.shape != (len(ALL_ACTION_IDS),)
        or not np.isfinite(values).all()
        or np.any((values < 0.0) | (values > 1.0))
    ):
        raise ProtocolError("OE-PPUR v3 row feature input is not one N-by-7 row.")
    protected = float(values[0])
    candidates = values[1:]
    return LabelFreeFeatureVector(
        ROW_FEATURE_NAMES,
        (
            protected,
            abs(protected - 0.5),
            float(np.mean(candidates, dtype=np.float64)),
            float(np.std(candidates, dtype=np.float64)),
            float(np.max(candidates) - np.min(candidates)),
            _binary_entropy(protected),
        ),
    )


def build_action_features(
    protected_probabilities: Sequence[float],
    candidate_probabilities: Sequence[float],
) -> LabelFreeFeatureVector:
    """Build six fixed case/action predictors with float64 reductions."""

    protected = np.asarray(protected_probabilities, dtype=np.float64)
    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    if (
        protected.ndim != 1
        or candidate.shape != protected.shape
        or len(protected) == 0
        or not np.isfinite(protected).all()
        or not np.isfinite(candidate).all()
        or np.any((protected < 0.0) | (protected > 1.0))
        or np.any((candidate < 0.0) | (candidate > 1.0))
    ):
        raise ProtocolError("OE-PPUR v3 action-feature probability vectors drifted.")
    difference = candidate - protected
    crossing = (protected >= 0.5) != (candidate >= 0.5)
    return LabelFreeFeatureVector(
        ACTION_FEATURE_NAMES,
        (
            float(np.mean(protected, dtype=np.float64)),
            float(np.mean(candidate, dtype=np.float64)),
            float(np.mean(difference, dtype=np.float64)),
            float(np.std(difference, dtype=np.float64)),
            float(np.mean(crossing, dtype=np.float64)),
            float(np.mean(np.abs(difference), dtype=np.float64)),
        ),
    )


def build_row_posterior_observations(
    rows: Sequence[SourceSupervisionRow],
) -> tuple[RowPosteriorObservation, ...]:
    """Adapt source outcomes to the neutral fixed-capacity posterior core."""

    values = tuple(rows)
    if not values or any(not isinstance(row, SourceSupervisionRow) for row in values):
        raise ProtocolError("OE-PPUR v3 row-posterior source rows are empty or untyped.")
    result = tuple(
        RowPosteriorObservation(
            center_id=row.query_center,
            case_id=row.case_id,
            row_id=row.source_row_id,
            feature_names=ROW_FEATURE_NAMES,
            feature_values=build_row_features(row.action_probabilities).values,
            outcome=row.outcome,
        )
        for row in values
    )
    keys = tuple((row.center_id, row.case_id, row.row_id) for row in result)
    if len(set(keys)) != len(keys):
        raise ProtocolError(
            "OE-PPUR v3 row-posterior observations duplicated source row identities."
        )
    return tuple(sorted(result, key=lambda row: (row.center_id, row.case_id, row.row_id)))


def build_case_action_feature(
    rows: Sequence[SourceSupervisionRow], *, action_id: object
) -> LabelFreeFeatureVector:
    action = str(action_id)
    if action not in CANDIDATE_ACTION_IDS:
        raise ProtocolError("OE-PPUR v3 action features exclude exact protected P.")
    values = tuple(rows)
    case_keys = {(row.outer_target_center, row.query_center, row.case_id) for row in values}
    if not values or len(case_keys) != 1:
        raise ProtocolError("OE-PPUR v3 case/action feature rows mixed case scopes.")
    index = ALL_ACTION_IDS.index(action)
    return build_action_features(
        [row.action_probabilities[0] for row in values],
        [row.action_probabilities[index] for row in values],
    )


def build_action_query(
    rows: Sequence[SourceSupervisionRow], *, action_id: object
) -> ActionQuery:
    action = str(action_id)
    feature = build_case_action_feature(rows, action_id=action)
    family, direction = action.split("::", maxsplit=1)
    return ActionQuery(action, family, direction, feature.names, feature.values)


def build_case_action_queries(
    rows: Sequence[SourceSupervisionRow],
) -> tuple[ActionQuery, ...]:
    return tuple(build_action_query(rows, action_id=action) for action in CANDIDATE_ACTION_IDS)


FEATURE_DEFINITION_RECEIPT_HASH = canonical_sha256(
    {
        "schema": "oe_ppur_v3_frozen_label_free_feature_definitions_v1",
        "row_feature_names": ROW_FEATURE_NAMES,
        "action_feature_names": ACTION_FEATURE_NAMES,
        "row_formulas": (
            "P", "abs(P-0.5)", "mean(challengers)", "std(challengers)",
            "max(challengers)-min(challengers)", "binary_entropy(P)",
        ),
        "action_formulas": (
            "mean(P)", "mean(A)", "mean(A-P)", "std(A-P)",
            "mean(class_crossing(P,A))", "mean(abs(A-P))",
        ),
        "float64_reductions": True,
        "labels_used": False,
        "identity_features_used": False,
    }
)


__all__ = (
    "ACTION_FEATURE_NAMES",
    "FEATURE_DEFINITION_RECEIPT_HASH",
    "LabelFreeFeatureVector",
    "ROW_FEATURE_NAMES",
    "build_action_features",
    "build_action_query",
    "build_case_action_feature",
    "build_case_action_queries",
    "build_row_features",
    "build_row_posterior_observations",
)
