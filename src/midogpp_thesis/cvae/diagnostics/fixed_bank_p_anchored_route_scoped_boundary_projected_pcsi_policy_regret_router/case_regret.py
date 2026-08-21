"""Case-local pseudo replay and favorable utility identities for PCSI-RACR."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    LOG_LOSS_CLIP_EPSILON,
    PORTFOLIO_METHOD_ID,
    PROJECTION_GEOMETRY_ID,
    UNPROJECTED_GEOMETRY_ID,
)
from .contracts import BinaryLabel, EndpointCasePrediction, PseudoRouteKey
from .hashing import canonical_hash, require_sha256
from .policy_selection import CaseCandidatePolicy
from .projection_lattice import THRESHOLD, as_binary32


@dataclass(frozen=True, order=True)
class PseudoCaseReplay:
    geometry_id: str
    route: PseudoRouteKey
    candidate_policy_hash: str
    endpoint_prediction_hash: str
    predicted_favorable_vector: tuple[float, float, float]
    realized_favorable_vector: tuple[float, float, float]
    overprediction_residual: tuple[float, float, float]
    label_identity_hash: str
    replay_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if self.geometry_id not in {PROJECTION_GEOMETRY_ID, UNPROJECTED_GEOMETRY_ID}:
            raise ProtocolError("PCSI-RACR pseudo replay geometry drifted.")
        for digest in (
            self.candidate_policy_hash,
            self.endpoint_prediction_hash,
            self.label_identity_hash,
        ):
            require_sha256(digest, "pseudo_replay_binding_hash")
        predicted = tuple(float(value) for value in self.predicted_favorable_vector)
        realized = tuple(float(value) for value in self.realized_favorable_vector)
        residual = tuple(float(value) for value in self.overprediction_residual)
        if (
            len(predicted) != 3
            or len(realized) != 3
            or len(residual) != 3
            or any(not math.isfinite(value) for value in (*predicted, *realized, *residual))
            or any(abs(residual[k] - (predicted[k] - realized[k])) > 1.0e-12 for k in range(3))
        ):
            raise ProtocolError("PCSI-RACR pseudo replay vector drifted.")
        object.__setattr__(self, "predicted_favorable_vector", predicted)
        object.__setattr__(self, "realized_favorable_vector", realized)
        object.__setattr__(self, "overprediction_residual", residual)
        object.__setattr__(self, "replay_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_pseudo_case_replay_v1",
            "geometry_id": self.geometry_id,
            "outer_center": self.route.outer_center,
            "donor_center": self.route.donor_center,
            "case_id": self.route.case_id,
            "candidate_policy_hash": self.candidate_policy_hash,
            "endpoint_prediction_hash": self.endpoint_prediction_hash,
            "predicted_favorable_vector": list(self.predicted_favorable_vector),
            "realized_favorable_vector": list(self.realized_favorable_vector),
            "overprediction_residual": list(self.overprediction_residual),
            "label_identity_hash": self.label_identity_hash,
            "pseudo_policy_sealed_before_replay_label": True,
            "conformal": False,
            "finite_sample_coverage": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "replay_hash": self.replay_hash}


def realized_case_favorable_vector(
    endpoint: EndpointCasePrediction,
    candidate: CaseCandidatePolicy,
    case_labels: Sequence[BinaryLabel],
    *,
    center_n_positive: int,
    center_n_negative: int,
) -> tuple[float, float, float]:
    labels = {row.sample_id: row.value for row in case_labels}
    if (
        endpoint.center != candidate.target_center
        or endpoint.case_id != candidate.case_id
        or set(labels) != set(endpoint.sample_ids)
        or center_n_positive <= 0
        or center_n_negative <= 0
    ):
        raise ProtocolError("PCSI-RACR realized case utility scope drifted.")
    y = np.asarray([labels[sample] for sample in endpoint.sample_ids], dtype=np.int8)
    baseline = as_binary32(endpoint.probabilities[PORTFOLIO_METHOD_ID], name="replay P").astype(np.float64)
    action = as_binary32(candidate.probabilities, name="replay action").astype(np.float64)
    baseline_hard = baseline >= float(THRESHOLD)
    action_hard = action >= float(THRESHOLD)
    positive = y == 1
    negative = ~positive
    bacc = 0.5 * (
        float(
            np.sum(
                action_hard[positive].astype(np.int8)
                - baseline_hard[positive].astype(np.int8),
                dtype=np.int64,
            )
        )
        / center_n_positive
        + float(
            np.sum(
                (~action_hard[negative]).astype(np.int8)
                - (~baseline_hard[negative]).astype(np.int8),
                dtype=np.int64,
            )
        )
        / center_n_negative
    )
    n_total = center_n_positive + center_n_negative
    brier = float(
        np.sum(
            (baseline - y) ** 2 - (action - y) ** 2,
            dtype=np.float64,
        )
        / n_total
    )
    base_clip = np.clip(
        baseline, LOG_LOSS_CLIP_EPSILON, 1.0 - LOG_LOSS_CLIP_EPSILON
    )
    action_clip = np.clip(
        action, LOG_LOSS_CLIP_EPSILON, 1.0 - LOG_LOSS_CLIP_EPSILON
    )
    base_log = -(y * np.log(base_clip) + (1 - y) * np.log1p(-base_clip))
    action_log = -(y * np.log(action_clip) + (1 - y) * np.log1p(-action_clip))
    log_gain = float(np.sum(base_log - action_log, dtype=np.float64) / n_total)
    return float(bacc), brier, log_gain


def build_pseudo_case_replay(
    *,
    outer_center: str,
    candidate: CaseCandidatePolicy,
    endpoint: EndpointCasePrediction,
    case_labels: Sequence[BinaryLabel],
    center_n_positive: int,
    center_n_negative: int,
) -> PseudoCaseReplay:
    if outer_center not in CENTERS or outer_center == endpoint.center:
        raise ProtocolError("PCSI-RACR pseudo replay outer scope drifted.")
    realized = realized_case_favorable_vector(
        endpoint,
        candidate,
        case_labels,
        center_n_positive=center_n_positive,
        center_n_negative=center_n_negative,
    )
    predicted = tuple(float(value) for value in candidate.predicted_favorable_endpoint_vector)
    residual = tuple(predicted[k] - realized[k] for k in range(3))
    label_hash = canonical_hash(
        [
            [row.center, row.case_id, row.sample_id, row.value, row.scope]
            for row in sorted(case_labels, key=lambda row: row.key)
        ]
    )
    return PseudoCaseReplay(
        candidate.geometry_id,
        PseudoRouteKey(outer_center, endpoint.center, endpoint.case_id),
        candidate.policy_hash,
        endpoint.prediction_hash,
        predicted,
        realized,
        residual,
        label_hash,
    )


__all__ = (
    "PseudoCaseReplay",
    "build_pseudo_case_replay",
    "realized_case_favorable_vector",
)
