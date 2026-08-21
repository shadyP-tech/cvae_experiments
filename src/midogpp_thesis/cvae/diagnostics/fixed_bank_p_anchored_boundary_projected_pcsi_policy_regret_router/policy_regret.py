"""Whole-policy endpoint-vector regret correction for PCSI-PARC."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import CENTERS, LOG_LOSS_CLIP_EPSILON, PORTFOLIO_METHOD_ID
from .contracts import BinaryLabel, EndpointCasePrediction
from .hashing import canonical_hash, require_sha256
from .policy_selection import CaseCandidatePolicy
from .projection_lattice import THRESHOLD, as_binary32
from .transport import TransportScreen


FAVORABLE_ENDPOINT_IDS = (
    "bacc_delta",
    "negative_brier_delta",
    "negative_log_loss_delta",
)


def _vector(values: object, *, name: str) -> tuple[float, float, float]:
    converted = tuple(float(value) for value in values)
    if len(converted) != 3 or any(not math.isfinite(value) for value in converted):
        raise ProtocolError(f"PCSI-PARC {name} vector drifted.")
    return converted  # type: ignore[return-value]


@dataclass(frozen=True, order=True)
class CenterCandidatePolicy:
    center: str
    policy_id: str
    geometry_id: str
    cases: tuple[CaseCandidatePolicy, ...]
    predicted_favorable_endpoint_vector: tuple[float, float, float]
    policy_seal_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        rows = tuple(sorted(self.cases, key=lambda row: row.case_id))
        vector = _vector(
            self.predicted_favorable_endpoint_vector,
            name="predicted center policy",
        )
        if (
            self.center not in CENTERS
            or not rows
            or len({row.case_id for row in rows}) != len(rows)
            or any(
                row.target_center != self.center
                or row.policy_id != self.policy_id
                or row.geometry_id != self.geometry_id
                for row in rows
            )
        ):
            raise ProtocolError("PCSI-PARC center candidate policy drifted.")
        recomputed = tuple(
            float(
                np.sum(
                    [row.predicted_favorable_endpoint_vector[index] for row in rows],
                    dtype=np.float64,
                )
            )
            for index in range(3)
        )
        if recomputed != vector:
            raise ProtocolError("PCSI-PARC center Ghat is not the selected-case sum.")
        object.__setattr__(self, "cases", rows)
        object.__setattr__(self, "predicted_favorable_endpoint_vector", vector)
        object.__setattr__(self, "policy_seal_hash", canonical_hash(self._unhashed()))

    @property
    def changed_case_count(self) -> int:
        return sum(row.changed for row in self.cases)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_center_candidate_policy_v1",
            "center": self.center,
            "policy_id": self.policy_id,
            "geometry_id": self.geometry_id,
            "case_policy_hashes": [row.policy_hash for row in self.cases],
            "predicted_favorable_endpoint_vector": list(
                self.predicted_favorable_endpoint_vector
            ),
            "changed_case_count": self.changed_case_count,
            "terminal_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "policy_seal_hash": self.policy_seal_hash}


@dataclass(frozen=True, order=True)
class WholePolicyReplay:
    outer_target_center: str
    pseudo_target_center: str
    geometry_id: str
    predicted_favorable_endpoint_vector: tuple[float, float, float]
    actual_favorable_endpoint_vector: tuple[float, float, float]
    residual_vector: tuple[float, float, float]
    pseudo_policy_seal_hash: str
    evaluation_identity_hash: str
    transport_screen_hash: str
    replay_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        predicted = _vector(self.predicted_favorable_endpoint_vector, name="pseudo Ghat")
        actual = _vector(self.actual_favorable_endpoint_vector, name="pseudo G")
        residual = _vector(self.residual_vector, name="pseudo residual")
        expected = tuple(float(predicted[index] - actual[index]) for index in range(3))
        if (
            self.outer_target_center not in CENTERS
            or self.pseudo_target_center not in CENTERS
            or self.outer_target_center == self.pseudo_target_center
            or residual != expected
        ):
            raise ProtocolError("PCSI-PARC whole-policy replay drifted.")
        for value, name in (
            (self.pseudo_policy_seal_hash, "pseudo_policy_seal_hash"),
            (self.evaluation_identity_hash, "pseudo_evaluation_identity_hash"),
            (self.transport_screen_hash, "pseudo_transport_screen_hash"),
        ):
            require_sha256(value, name)
        object.__setattr__(self, "predicted_favorable_endpoint_vector", predicted)
        object.__setattr__(self, "actual_favorable_endpoint_vector", actual)
        object.__setattr__(self, "residual_vector", residual)
        object.__setattr__(self, "replay_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_whole_policy_replay_v1",
            "outer_target_center": self.outer_target_center,
            "pseudo_target_center": self.pseudo_target_center,
            "geometry_id": self.geometry_id,
            "predicted_favorable_endpoint_vector": list(
                self.predicted_favorable_endpoint_vector
            ),
            "actual_favorable_endpoint_vector": list(self.actual_favorable_endpoint_vector),
            "residual_vector": list(self.residual_vector),
            "pseudo_policy_seal_hash": self.pseudo_policy_seal_hash,
            "evaluation_identity_hash": self.evaluation_identity_hash,
            "transport_screen_hash": self.transport_screen_hash,
            "calibration_unit": "whole_center",
            "conformal_or_coverage_claimed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "replay_hash": self.replay_hash}


@dataclass(frozen=True, order=True)
class PolicyAuthorization:
    target_center: str
    policy_id: str
    geometry_id: str
    predicted_favorable_endpoint_vector: tuple[float, float, float]
    regret_radius_vector: tuple[float, float, float]
    diagnostic_lower_vector: tuple[float, float, float]
    effective_donor_count: int
    target_transport_passed: bool
    pseudo_transport_passes: tuple[tuple[str, bool], ...]
    authorized: bool
    target_policy_seal_hash: str
    target_transport_screen_hash: str
    replay_hashes: tuple[str, ...]
    authorization_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        predicted = _vector(self.predicted_favorable_endpoint_vector, name="target Ghat")
        radius = _vector(self.regret_radius_vector, name="regret radius")
        lower = _vector(self.diagnostic_lower_vector, name="diagnostic lower")
        passes = tuple((str(center), bool(value)) for center, value in self.pseudo_transport_passes)
        if (
            self.target_center not in CENTERS
            or len(passes) != len(CENTERS) - 1
            or {center for center, _value in passes} != set(CENTERS).difference({self.target_center})
            or any(value < 0.0 for value in radius)
            or lower != tuple(predicted[index] - radius[index] for index in range(3))
            or self.effective_donor_count != sum(value for _center, value in passes)
            or self.authorized
            != bool(
                self.target_transport_passed
                and self.effective_donor_count == len(CENTERS) - 1
                and all(value > 0.0 for value in lower)
            )
        ):
            raise ProtocolError("PCSI-PARC policy authorization drifted.")
        require_sha256(self.target_policy_seal_hash, "target_policy_seal_hash")
        require_sha256(self.target_transport_screen_hash, "target_transport_screen_hash")
        for digest in self.replay_hashes:
            require_sha256(digest, "policy_replay_hash")
        object.__setattr__(self, "predicted_favorable_endpoint_vector", predicted)
        object.__setattr__(self, "regret_radius_vector", radius)
        object.__setattr__(self, "diagnostic_lower_vector", lower)
        object.__setattr__(self, "pseudo_transport_passes", passes)
        object.__setattr__(self, "authorization_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_policy_authorization_v1",
            "target_center": self.target_center,
            "policy_id": self.policy_id,
            "geometry_id": self.geometry_id,
            "endpoint_ids": list(FAVORABLE_ENDPOINT_IDS),
            "predicted_favorable_endpoint_vector": list(
                self.predicted_favorable_endpoint_vector
            ),
            "regret_radius_vector": list(self.regret_radius_vector),
            "diagnostic_lower_vector": list(self.diagnostic_lower_vector),
            "effective_donor_count": self.effective_donor_count,
            "target_transport_passed": self.target_transport_passed,
            "pseudo_transport_passes": dict(self.pseudo_transport_passes),
            "authorized": self.authorized,
            "target_policy_seal_hash": self.target_policy_seal_hash,
            "target_transport_screen_hash": self.target_transport_screen_hash,
            "replay_hashes": list(self.replay_hashes),
            "regret_rule": "coordinatewise_max_zero_and_eight_observed_donors",
            "equality_abstains": True,
            "conformal_or_coverage_claimed": False,
            "terminal_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "authorization_hash": self.authorization_hash}


def build_center_candidate_policy(
    cases: Sequence[CaseCandidatePolicy],
) -> CenterCandidatePolicy:
    rows = tuple(cases)
    if not rows:
        raise ProtocolError("PCSI-PARC cannot seal an empty center policy.")
    vector = tuple(
        float(
            np.sum(
                [row.predicted_favorable_endpoint_vector[index] for row in rows],
                dtype=np.float64,
            )
        )
        for index in range(3)
    )
    return CenterCandidatePolicy(
        rows[0].target_center,
        rows[0].policy_id,
        rows[0].geometry_id,
        rows,
        vector,
    )


def score_actual_center_policy(
    policy: CenterCandidatePolicy,
    endpoints: Sequence[EndpointCasePrediction],
    labels: Sequence[BinaryLabel],
) -> tuple[tuple[float, float, float], str]:
    """Score a sealed complete policy in the exact favorable endpoint units."""

    endpoint_by_case = {row.case_id: row for row in endpoints}
    label_by_key = {row.key: row for row in labels}
    if (
        set(endpoint_by_case) != {row.case_id for row in policy.cases}
        or len(endpoint_by_case) != len(tuple(endpoints))
        or len(label_by_key) != len(tuple(labels))
        or {row.center for row in labels} != {policy.center}
    ):
        raise ProtocolError("PCSI-PARC whole-policy evaluation scope drifted.")
    expected = {
        (policy.center, case.case_id, sample_id)
        for case in policy.cases
        for sample_id in case.sample_ids
    }
    if set(label_by_key) != expected:
        raise ProtocolError("PCSI-PARC whole-policy evaluation identities drifted.")
    y = np.asarray([label_by_key[key].value for key in sorted(expected)], dtype=np.int8)
    candidate_by_key: dict[tuple[str, str, str], float] = {}
    portfolio_by_key: dict[tuple[str, str, str], float] = {}
    for case in policy.cases:
        endpoint = endpoint_by_case[case.case_id]
        if endpoint.sample_ids != case.sample_ids:
            raise ProtocolError("PCSI-PARC evaluation case rows drifted.")
        for sample_id, candidate, portfolio in zip(
            case.sample_ids,
            case.probabilities,
            endpoint.probabilities[PORTFOLIO_METHOD_ID],
            strict=True,
        ):
            key = (policy.center, case.case_id, sample_id)
            candidate_by_key[key] = float(candidate)
            portfolio_by_key[key] = float(portfolio)
    ordered = sorted(expected)
    candidate = as_binary32([candidate_by_key[key] for key in ordered], name="actual candidate").astype(np.float64)
    portfolio = as_binary32([portfolio_by_key[key] for key in ordered], name="actual P").astype(np.float64)
    hard = candidate >= float(THRESHOLD)
    p_hard = portfolio >= float(THRESHOLD)
    positive = y == 1
    negative = ~positive
    n_positive = int(np.sum(positive, dtype=np.int64))
    n_negative = int(np.sum(negative, dtype=np.int64))
    if min(n_positive, n_negative) <= 0:
        raise ProtocolError("PCSI-PARC center evaluation lacks one class.")
    bacc = 0.5 * (
        float(np.sum(hard[positive].astype(np.int8) - p_hard[positive].astype(np.int8), dtype=np.int64)) / n_positive
        + float(np.sum((~hard[negative]).astype(np.int8) - (~p_hard[negative]).astype(np.int8), dtype=np.int64)) / n_negative
    )
    brier = float(np.mean((candidate - y) ** 2 - (portfolio - y) ** 2, dtype=np.float64))
    c_clip = np.clip(candidate, LOG_LOSS_CLIP_EPSILON, 1.0 - LOG_LOSS_CLIP_EPSILON)
    p_clip = np.clip(portfolio, LOG_LOSS_CLIP_EPSILON, 1.0 - LOG_LOSS_CLIP_EPSILON)
    c_log = -(y * np.log(c_clip) + (1 - y) * np.log1p(-c_clip))
    p_log = -(y * np.log(p_clip) + (1 - y) * np.log1p(-p_clip))
    log_delta = float(np.mean(c_log - p_log, dtype=np.float64))
    identity_hash = canonical_hash([list(key) for key in ordered])
    return (float(bacc), -brier, -log_delta), identity_hash


def build_whole_policy_replay(
    *,
    outer_target_center: str,
    pseudo_policy: CenterCandidatePolicy,
    endpoints: Sequence[EndpointCasePrediction],
    evaluation_labels: Sequence[BinaryLabel],
    transport_screen: TransportScreen,
) -> WholePolicyReplay:
    if (
        pseudo_policy.center == outer_target_center
        or transport_screen.candidate_center != pseudo_policy.center
    ):
        raise ProtocolError("PCSI-PARC pseudo-policy replay topology drifted.")
    actual, identity_hash = score_actual_center_policy(
        pseudo_policy,
        endpoints,
        evaluation_labels,
    )
    predicted = pseudo_policy.predicted_favorable_endpoint_vector
    residual = tuple(predicted[index] - actual[index] for index in range(3))
    return WholePolicyReplay(
        outer_target_center,
        pseudo_policy.center,
        pseudo_policy.geometry_id,
        predicted,
        actual,
        residual,
        pseudo_policy.policy_seal_hash,
        identity_hash,
        transport_screen.screen_hash,
    )


def authorize_center_policy(
    target_policy: CenterCandidatePolicy,
    replays: Sequence[WholePolicyReplay],
    *,
    target_transport_screen: TransportScreen,
    pseudo_transport_screens: Mapping[str, TransportScreen],
) -> PolicyAuthorization:
    rows = tuple(sorted(replays, key=lambda row: CENTERS.index(row.pseudo_target_center)))
    expected = tuple(center for center in CENTERS if center != target_policy.center)
    if (
        tuple(row.pseudo_target_center for row in rows) != expected
        or any(
            row.outer_target_center != target_policy.center
            or row.geometry_id != target_policy.geometry_id
            for row in rows
        )
        or set(pseudo_transport_screens) != set(expected)
        or target_transport_screen.candidate_center != target_policy.center
    ):
        raise ProtocolError("PCSI-PARC authorization replay matrix drifted.")
    residuals = np.asarray([row.residual_vector for row in rows], dtype=np.float64)
    radius = tuple(
        float(max(0.0, np.max(residuals[:, index]))) for index in range(3)
    )
    predicted = target_policy.predicted_favorable_endpoint_vector
    lower = tuple(predicted[index] - radius[index] for index in range(3))
    passes = tuple(
        (center, bool(pseudo_transport_screens[center].passed)) for center in expected
    )
    effective = sum(value for _center, value in passes)
    authorized = bool(
        target_transport_screen.passed
        and effective == len(expected)
        and all(value > 0.0 for value in lower)
    )
    return PolicyAuthorization(
        target_policy.center,
        target_policy.policy_id,
        target_policy.geometry_id,
        predicted,
        radius,
        lower,
        effective,
        bool(target_transport_screen.passed),
        passes,
        authorized,
        target_policy.policy_seal_hash,
        target_transport_screen.screen_hash,
        tuple(row.replay_hash for row in rows),
    )


__all__ = (
    "CenterCandidatePolicy",
    "FAVORABLE_ENDPOINT_IDS",
    "PolicyAuthorization",
    "WholePolicyReplay",
    "authorize_center_policy",
    "build_center_candidate_policy",
    "build_whole_policy_replay",
    "score_actual_center_policy",
)
