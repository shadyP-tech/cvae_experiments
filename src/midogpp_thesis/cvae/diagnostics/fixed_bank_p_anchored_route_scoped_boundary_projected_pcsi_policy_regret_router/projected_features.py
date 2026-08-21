"""Twelve frozen descriptors computed from final emitted action vectors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import DIRECTION_IDS, PORTFOLIO_METHOD_ID, UTILITY_FEATURE_NAMES
from .contracts import EndpointCasePrediction
from .projected_contracts import ProjectedUtilityDescriptor
from .projection import ActionEquivalenceClass
from .projection_lattice import THRESHOLD, as_binary32


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values, dtype=np.float64)) if len(values) else 0.0


def _entropy(values: np.ndarray) -> float:
    if not len(values):
        return 0.0
    clipped = np.clip(values.astype(np.float64), 1.0e-12, 1.0 - 1.0e-12)
    return _mean(-(clipped * np.log(clipped) + (1.0 - clipped) * np.log1p(-clipped)))


def build_projected_descriptors(
    endpoint: EndpointCasePrediction,
    actions: Sequence[ActionEquivalenceClass],
) -> tuple[ProjectedUtilityDescriptor, ...]:
    rows = tuple(actions)
    portfolio = as_binary32(endpoint.probabilities[PORTFOLIO_METHOD_ID], name="descriptor P")
    p_hard = portfolio >= THRESHOLD
    output: list[ProjectedUtilityDescriptor] = []
    for action in rows:
        if (
            action.target_center != endpoint.center
            or action.case_id != endpoint.case_id
            or action.endpoint_prediction_hash != endpoint.prediction_hash
            or action.sample_ids != endpoint.sample_ids
        ):
            raise ProtocolError("PCSI-RACR action/endpoint descriptor binding drifted.")
        emitted = as_binary32(action.probabilities, name="descriptor action")
        branch = ~p_hard if action.direction == DIRECTION_IDS[0] else p_hard
        crossing = (emitted >= THRESHOLD) != p_hard
        expected_crossings = tuple(
            endpoint.sample_ids[int(index)] for index in np.flatnonzero(crossing)
        )
        if expected_crossings != action.crossing_sample_ids:
            raise ProtocolError("PCSI-RACR crossing identities drifted after projection.")
        values = (
            float(np.log1p(len(portfolio))),
            _mean(branch.astype(np.float64)),
            _mean(crossing.astype(np.float64)),
            _mean(portfolio[branch]),
            _mean(emitted[branch]),
            _mean(np.abs(portfolio[branch].astype(np.float64) - float(THRESHOLD))),
            _mean(np.abs(emitted[branch].astype(np.float64) - float(THRESHOLD))),
            _mean((emitted.astype(np.float64) - portfolio.astype(np.float64))[crossing]),
            _mean(np.abs(emitted.astype(np.float64) - portfolio.astype(np.float64))[crossing]),
            _entropy(portfolio[crossing]),
            _entropy(emitted[crossing]),
            float(np.log1p(action.crossing_count)),
        )
        output.append(
            ProjectedUtilityDescriptor(
                endpoint.center,
                endpoint.case_id,
                action.geometry_id,
                action.direction,
                action.representative,
                action.members,
                UTILITY_FEATURE_NAMES,
                values,
                action.crossing_sample_ids,
                action.action_hash,
                endpoint.prediction_hash,
            )
        )
    result = tuple(sorted(output, key=lambda row: row.key))
    if len(result) != len(rows) or len({row.key for row in result}) != len(result):
        raise ProtocolError("PCSI-RACR projected descriptor surface drifted.")
    return result


__all__ = ("build_projected_descriptors",)
