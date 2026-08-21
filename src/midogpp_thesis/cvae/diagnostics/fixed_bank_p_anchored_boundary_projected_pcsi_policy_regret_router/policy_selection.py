"""Proper-safe class selection and exact P-anchored case composition."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    ALTERNATIVE_METHOD_IDS,
    COMPOSED_POLICY_IDS,
    DIRECTION_IDS,
    PORTFOLIO_METHOD_ID,
    UTILITY_ZERO_TOLERANCE,
)
from .contracts import EndpointCasePrediction
from .hashing import canonical_hash, require_sha256
from .projected_contracts import ProjectedUtilityDescriptor, ProjectedUtilityPrediction
from .projection import ActionEquivalenceClass
from .projection_lattice import THRESHOLD, as_binary32, canonical_bytes
from .sample_influence_contracts import InfluencePrediction


_ALTERNATIVE_ORDER = MappingProxyType(
    {name: index for index, name in enumerate(ALTERNATIVE_METHOD_IDS)}
)


@dataclass(frozen=True, order=True)
class DirectionalClassDecision:
    target_center: str
    case_id: str
    policy_id: str
    direction: str
    selected_action_hash: str | None
    selected_representative: str
    target_influence: float
    predicted_favorable_endpoint_vector: tuple[float, float, float]
    candidate_binding_hashes: tuple[str, ...]
    decision_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        vector = tuple(float(value) for value in self.predicted_favorable_endpoint_vector)
        if (
            self.policy_id not in COMPOSED_POLICY_IDS
            or self.direction not in DIRECTION_IDS
            or self.selected_representative not in (*ALTERNATIVE_METHOD_IDS, PORTFOLIO_METHOD_ID)
            or len(vector) != 3
            or any(not math.isfinite(value) for value in (*vector, float(self.target_influence)))
            or (self.selected_action_hash is None) != (self.selected_representative == PORTFOLIO_METHOD_ID)
        ):
            raise ProtocolError("PCSI-PARC directional decision drifted.")
        if self.selected_action_hash is not None:
            require_sha256(self.selected_action_hash, "selected_action_hash")
        for digest in self.candidate_binding_hashes:
            require_sha256(digest, "candidate_binding_hash")
        object.__setattr__(self, "predicted_favorable_endpoint_vector", vector)
        object.__setattr__(self, "decision_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_directional_decision_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "policy_id": self.policy_id,
            "direction": self.direction,
            "selected_action_hash": self.selected_action_hash,
            "selected_representative": self.selected_representative,
            "target_influence": self.target_influence,
            "predicted_favorable_endpoint_vector": list(self.predicted_favorable_endpoint_vector),
            "candidate_binding_hashes": list(self.candidate_binding_hashes),
            "terminal_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "decision_hash": self.decision_hash}


@dataclass(frozen=True, order=True)
class CaseCandidatePolicy:
    target_center: str
    case_id: str
    policy_id: str
    geometry_id: str
    sample_ids: tuple[str, ...]
    probabilities: tuple[float, ...]
    decisions: tuple[DirectionalClassDecision, ...]
    predicted_favorable_endpoint_vector: tuple[float, float, float]
    endpoint_prediction_hash: str
    probability_bytes_sha256: str
    policy_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        samples = tuple(str(value) for value in self.sample_ids)
        probabilities = as_binary32(self.probabilities, name="candidate case policy")
        vector = tuple(float(value) for value in self.predicted_favorable_endpoint_vector)
        if (
            self.policy_id not in COMPOSED_POLICY_IDS
            or not samples
            or len(samples) != len(set(samples))
            or len(probabilities) != len(samples)
            or tuple(row.direction for row in self.decisions) != DIRECTION_IDS
            or any(
                row.target_center != self.target_center
                or row.case_id != self.case_id
                or row.policy_id != self.policy_id
                for row in self.decisions
            )
            or len(vector) != 3
            or any(not math.isfinite(value) for value in vector)
        ):
            raise ProtocolError("PCSI-PARC candidate case policy drifted.")
        require_sha256(self.endpoint_prediction_hash, "endpoint_prediction_hash")
        require_sha256(self.probability_bytes_sha256, "policy_probability_bytes_hash")
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "probabilities", tuple(float(value) for value in probabilities))
        object.__setattr__(self, "predicted_favorable_endpoint_vector", vector)
        object.__setattr__(self, "policy_hash", canonical_hash(self._unhashed()))

    @property
    def changed(self) -> bool:
        return any(row.selected_action_hash is not None for row in self.decisions)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_case_candidate_policy_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "policy_id": self.policy_id,
            "geometry_id": self.geometry_id,
            "sample_ids": list(self.sample_ids),
            "probability_bytes_sha256": self.probability_bytes_sha256,
            "decisions": [row.to_payload() for row in self.decisions],
            "predicted_favorable_endpoint_vector": list(self.predicted_favorable_endpoint_vector),
            "endpoint_prediction_hash": self.endpoint_prediction_hash,
            "terminal_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._unhashed(),
            "probabilities": list(self.probabilities),
            "changed": self.changed,
            "policy_hash": self.policy_hash,
        }


@dataclass(frozen=True, order=True)
class FinalCasePolicyPrediction:
    target_center: str
    case_id: str
    policy_id: str
    sample_ids: tuple[str, ...]
    probabilities: tuple[float, ...]
    authorized: bool
    candidate_policy_hash: str
    authorization_hash: str
    prediction_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        samples = tuple(str(value) for value in self.sample_ids)
        probabilities = as_binary32(self.probabilities, name="final case policy")
        if not samples or len(samples) != len(probabilities):
            raise ProtocolError("PCSI-PARC final prediction topology drifted.")
        require_sha256(self.candidate_policy_hash, "candidate_policy_hash")
        require_sha256(self.authorization_hash, "policy_authorization_hash")
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "probabilities", tuple(float(value) for value in probabilities))
        object.__setattr__(self, "prediction_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_final_case_prediction_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "policy_id": self.policy_id,
            "sample_ids": list(self.sample_ids),
            "probability_bytes_sha256": hashlib.sha256(
                canonical_bytes(self.probabilities)
            ).hexdigest(),
            "authorized": self.authorized,
            "candidate_policy_hash": self.candidate_policy_hash,
            "authorization_hash": self.authorization_hash,
            "terminal_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "probabilities": list(self.probabilities), "prediction_hash": self.prediction_hash}


def select_and_compose_case_policy(
    endpoint: EndpointCasePrediction,
    actions: Sequence[ActionEquivalenceClass],
    descriptors: Sequence[ProjectedUtilityDescriptor],
    influences: Sequence[InfluencePrediction],
    utilities: Sequence[ProjectedUtilityPrediction],
    *,
    policy_id: str,
    require_positive_bacc_prediction: bool,
) -> CaseCandidatePolicy:
    action_by_hash = {row.action_hash: row for row in actions}
    descriptor_by_hash = {row.descriptor_hash: row for row in descriptors}
    influence_by_hash = {row.descriptor_hash: row for row in influences}
    utility_by_hash = {row.descriptor_hash: row for row in utilities}
    if (
        policy_id not in COMPOSED_POLICY_IDS
        or not descriptor_by_hash
        or set(descriptor_by_hash) != set(influence_by_hash)
        or set(descriptor_by_hash) != set(utility_by_hash)
        or {row.action_hash for row in descriptors} != set(action_by_hash)
        or len(descriptor_by_hash) != len(tuple(descriptors))
    ):
        raise ProtocolError("PCSI-PARC case selection surface drifted.")
    portfolio = as_binary32(endpoint.probabilities[PORTFOLIO_METHOD_ID], name="composition P")
    p_hard = portfolio >= THRESHOLD
    output = portfolio.copy()
    occupied = np.zeros(len(output), dtype=bool)
    decisions: list[DirectionalClassDecision] = []
    ghat = np.zeros(3, dtype=np.float64)
    for direction in DIRECTION_IDS:
        candidates: list[tuple[float, ProjectedUtilityDescriptor]] = []
        direction_rows = tuple(row for row in descriptors if row.direction == direction)
        for descriptor in direction_rows:
            influence = influence_by_hash[descriptor.descriptor_hash]
            utility = utility_by_hash[descriptor.descriptor_hash]
            if (
                influence.target_center != descriptor.target_center
                or influence.case_id != descriptor.case_id
                or influence.direction != descriptor.direction
                or influence.alternative != descriptor.representative
                or influence.crossing_count != descriptor.crossing_count
                or utility.geometry_id != descriptor.geometry_id
            ):
                raise ProtocolError("PCSI-PARC influence/utility binding drifted.")
            score = float(influence.target_score)
            proper_safe = (
                utility.robust("brier_contribution_delta") <= 0.0
                and utility.robust("log_loss_contribution_delta") <= 0.0
            )
            bacc_safe = utility.robust("bacc_contribution_delta") > 0.0
            admissible = (
                descriptor.crossing_count > 0
                and score > UTILITY_ZERO_TOLERANCE
                and proper_safe
                and (bacc_safe or not require_positive_bacc_prediction)
            )
            if admissible:
                candidates.append(
                    (
                        score,
                        descriptor,
                    )
                )
        # Eligibility already enforces score > 1e-15.  Among eligible actions,
        # maximize the exact score; only exact score ties use B < I < R and the
        # action hash as a final total-order key.  P is chosen iff none qualify.
        selected = min(
            candidates,
            default=None,
            key=lambda row: (
                -row[0],
                _ALTERNATIVE_ORDER[row[1].representative],
                row[1].action_hash,
            ),
        )
        if selected is None:
            vector = (0.0, 0.0, 0.0)
            selected_hash = None
            representative = PORTFOLIO_METHOD_ID
            selected_score = 0.0
        else:
            descriptor = selected[1]
            action = action_by_hash[descriptor.action_hash]
            emitted = as_binary32(action.probabilities, name="selected action")
            mask = (emitted >= THRESHOLD) != p_hard
            if np.any(mask & occupied):
                raise ProtocolError("PCSI-PARC opposing direction masks overlap.")
            output[mask] = emitted[mask]
            occupied |= mask
            utility = utility_by_hash[descriptor.descriptor_hash]
            vector = (
                utility.robust("bacc_contribution_delta"),
                -utility.robust("brier_contribution_delta"),
                -utility.robust("log_loss_contribution_delta"),
            )
            ghat += np.asarray(vector, dtype=np.float64)
            selected_hash = descriptor.action_hash
            representative = descriptor.representative
            selected_score = float(selected[0])
        bindings = tuple(
            digest
            for row in sorted(
                direction_rows,
                key=lambda item: (_ALTERNATIVE_ORDER[item.representative], item.action_hash),
            )
            for digest in (
                row.descriptor_hash,
                influence_by_hash[row.descriptor_hash].influence_hash,
                utility_by_hash[row.descriptor_hash].prediction_hash,
            )
        )
        decisions.append(
            DirectionalClassDecision(
                endpoint.center,
                endpoint.case_id,
                policy_id,
                direction,
                selected_hash,
                representative,
                selected_score,
                tuple(float(value) for value in vector),
                bindings,
            )
        )
    probability_hash = hashlib.sha256(canonical_bytes(output)).hexdigest()
    geometry_ids = {row.geometry_id for row in descriptors}
    if len(geometry_ids) != 1:
        raise ProtocolError("PCSI-PARC one case spans action geometries.")
    return CaseCandidatePolicy(
        endpoint.center,
        endpoint.case_id,
        policy_id,
        next(iter(geometry_ids)),
        endpoint.sample_ids,
        tuple(float(value) for value in output),
        tuple(decisions),
        tuple(float(value) for value in ghat),
        endpoint.prediction_hash,
        probability_hash,
    )


def finalize_case_policy(
    candidate: CaseCandidatePolicy,
    endpoint: EndpointCasePrediction,
    *,
    authorized: bool,
    authorization_hash: str,
) -> FinalCasePolicyPrediction:
    if candidate.target_center != endpoint.center or candidate.case_id != endpoint.case_id:
        raise ProtocolError("PCSI-PARC finalization crossed a case.")
    probabilities = (
        candidate.probabilities
        if authorized
        else endpoint.probabilities[PORTFOLIO_METHOD_ID]
    )
    return FinalCasePolicyPrediction(
        candidate.target_center,
        candidate.case_id,
        candidate.policy_id,
        candidate.sample_ids,
        probabilities,
        bool(authorized),
        candidate.policy_hash,
        authorization_hash,
    )


__all__ = (
    "CaseCandidatePolicy",
    "DirectionalClassDecision",
    "FinalCasePolicyPrediction",
    "finalize_case_policy",
    "select_and_compose_case_policy",
)
