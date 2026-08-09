"""Post-seal exact and smooth response DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .constants import (
    CENTERS,
    EXACT_BACC_DELTA,
    RESPONSE_ROW_SCHEMA,
    SMOOTH_BACC_DELTA,
)
from .contract_validation import bounded, hash_sequence, hash_token, sha256


@dataclass(frozen=True)
class CaseAwareResponseRow:
    """Post-seal exact and smooth responses for one candidate endpoint."""

    outer_target_id: str
    query_id: str
    candidate_source: str
    support_partition_hash: str
    feature_row_hash: str
    feature_surface_seal_hash: str
    evaluation_partition_hash: str
    evaluation_case_hashes: tuple[str, ...]
    evaluation_row_hash: str
    evaluation_label_sha256: str
    response_prediction_hash: str
    exact_base_bacc: float
    exact_tail_bacc: float
    exact_bacc_delta: float
    smooth_base_bacc: float
    smooth_tail_bacc: float
    smooth_bacc_delta: float
    support_eval_disjoint: bool = True
    features_sealed_before_label_access: bool = True
    exact_response_is_primary: bool = True
    smooth_response_is_diagnostic_only: bool = True
    policy_update_authorized: bool = False
    response_row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for value, name in (
            (self.outer_target_id, "outer_target_id"),
            (self.query_id, "query_id"),
            (self.candidate_source, "candidate_source"),
        ):
            if value not in CENTERS:
                raise ProtocolError(f"{name} is outside the frozen center geometry.")
        outer = self.outer_target_id
        query = self.query_id
        source = self.candidate_source
        if outer == query or source in {outer, query}:
            raise ProtocolError("Response row requires distinct H/q/e domains.")
        hashes = {
            "support_partition_hash": hash_token(
                self.support_partition_hash, "support_partition_hash"
            ),
            "feature_row_hash": sha256(self.feature_row_hash, "feature_row_hash"),
            "feature_surface_seal_hash": sha256(
                self.feature_surface_seal_hash, "feature_surface_seal_hash"
            ),
            "evaluation_partition_hash": hash_token(
                self.evaluation_partition_hash, "evaluation_partition_hash"
            ),
            "evaluation_row_hash": hash_token(
                self.evaluation_row_hash, "evaluation_row_hash"
            ),
            "evaluation_label_sha256": sha256(
                self.evaluation_label_sha256, "evaluation_label_sha256"
            ),
            "response_prediction_hash": hash_token(
                self.response_prediction_hash, "response_prediction_hash"
            ),
        }
        case_hashes = hash_sequence(
            self.evaluation_case_hashes,
            "evaluation_case_hashes",
            len(self.evaluation_case_hashes),
        )
        if not case_hashes or len(set(case_hashes)) != len(case_hashes):
            raise ProtocolError("Response requires distinct remaining evaluation cases.")
        exact_base = bounded(self.exact_base_bacc, "exact_base_bacc", 0.0, 1.0)
        exact_tail = bounded(self.exact_tail_bacc, "exact_tail_bacc", 0.0, 1.0)
        exact_delta = bounded(self.exact_bacc_delta, "exact_bacc_delta", -1.0, 1.0)
        smooth_base = bounded(self.smooth_base_bacc, "smooth_base_bacc", 0.0, 1.0)
        smooth_tail = bounded(self.smooth_tail_bacc, "smooth_tail_bacc", 0.0, 1.0)
        smooth_delta = bounded(
            self.smooth_bacc_delta, "smooth_bacc_delta", -1.0, 1.0
        )
        if not np.isclose(exact_tail - exact_base, exact_delta, atol=1.0e-12):
            raise ProtocolError("Exact BACC response delta is internally inconsistent.")
        if not np.isclose(smooth_tail - smooth_base, smooth_delta, atol=1.0e-12):
            raise ProtocolError("Smooth BACC response delta is internally inconsistent.")
        if (
            self.support_eval_disjoint is not True
            or self.features_sealed_before_label_access is not True
            or self.exact_response_is_primary is not True
            or self.smooth_response_is_diagnostic_only is not True
            or self.policy_update_authorized is not False
        ):
            raise ProtocolError("Response row violates the terminal scoring boundary.")
        object.__setattr__(self, "evaluation_case_hashes", case_hashes)
        for name, value in hashes.items():
            object.__setattr__(self, name, value)
        for name, value in (
            ("exact_base_bacc", exact_base),
            ("exact_tail_bacc", exact_tail),
            ("exact_bacc_delta", exact_delta),
            ("smooth_base_bacc", smooth_base),
            ("smooth_tail_bacc", smooth_tail),
            ("smooth_bacc_delta", smooth_delta),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "response_row_hash", canonical_sha256(self._unhashed_payload())
        )

    @property
    def row_key(self) -> tuple[str, str, str]:
        return self.outer_target_id, self.query_id, self.candidate_source

    def response_value(self, response_name: str) -> float:
        if response_name == EXACT_BACC_DELTA:
            return self.exact_bacc_delta
        if response_name == SMOOTH_BACC_DELTA:
            return self.smooth_bacc_delta
        raise ProtocolError("Unknown case-aware response name.")

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": RESPONSE_ROW_SCHEMA,
            **{
                name: getattr(self, name)
                for name in (
                    "outer_target_id",
                    "query_id",
                    "candidate_source",
                    "support_partition_hash",
                    "feature_row_hash",
                    "feature_surface_seal_hash",
                    "evaluation_partition_hash",
                    "evaluation_row_hash",
                    "evaluation_label_sha256",
                    "response_prediction_hash",
                    "exact_base_bacc",
                    "exact_tail_bacc",
                    "exact_bacc_delta",
                    "smooth_base_bacc",
                    "smooth_tail_bacc",
                    "smooth_bacc_delta",
                    "support_eval_disjoint",
                    "features_sealed_before_label_access",
                    "exact_response_is_primary",
                    "smooth_response_is_diagnostic_only",
                    "policy_update_authorized",
                )
            },
            "evaluation_case_hashes": list(self.evaluation_case_hashes),
            "response_unit": "candidate_H_q_e_after_exact_nine_probability_mean",
            "technical_seed_rows_are_independent_observations": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._unhashed_payload(),
            "response_row_hash": self.response_row_hash,
        }


@dataclass(frozen=True)
class CaseAwareResponseSurface:
    rows: tuple[CaseAwareResponseRow, ...]
    row_keys: tuple[tuple[str, str, str], ...]
    feature_surface_hash: str
    feature_surface_seal_hash: str
    surface_hash: str


__all__ = ("CaseAwareResponseRow", "CaseAwareResponseSurface")
