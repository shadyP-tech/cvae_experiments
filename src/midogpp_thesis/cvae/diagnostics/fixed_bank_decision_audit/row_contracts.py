"""Typed, hash-bound rows for the fixed-bank scientific core."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    DATASET_SCHEMA,
    FEATURE_ROW_SCHEMA,
    RESPONSE_ROW_SCHEMA,
    candidate_sources,
    expected_row_keys,
)
from .serialization import (
    canonical_hash,
    finite,
    probability_delta,
    require_exact_keys,
    require_sha256,
)


_FEATURE_FIELDS = (
    "schema_version",
    "outer_target_id",
    "query_id",
    "candidate_source",
    "source_feature_row_hash",
    "metadata_similarity",
    "pooled_row_weighted_abs_shift",
    "equal_case_abs_shift",
    "case_abs_shift_sd",
    "equal_case_signed_margin",
    "case_balanced_flip_rate",
    "case_balanced_entropy_change",
    "case_balanced_reconstruction",
    "case_balanced_kl",
    "case_balanced_log_mmd",
    "probability_role_used",
    "labels_used",
    "evaluation_probabilities_used_as_features",
    "known_fixed_bank_reuse",
    "unseen_expert_transfer",
    "feature_row_hash",
)

_RESPONSE_FIELDS = (
    "schema_version",
    "outer_target_id",
    "query_id",
    "candidate_source",
    "feature_row_hash",
    "source_response_row_hash",
    "exact_bacc_delta",
    "smooth_bacc_delta",
    "support_eval_disjoint",
    "features_sealed_before_label_access",
    "exact_response_is_primary",
    "smooth_response_is_descriptive_only",
    "policy_update_authorized",
    "response_row_hash",
)


def _row_key(outer: object, query: object, source: object) -> tuple[str, str, str]:
    values = (outer, query, source)
    if any(type(value) is not str or value not in CENTERS for value in values):
        raise ProtocolError("Fixed-bank row IDs drifted from MIDOG++ centers.")
    if len(set(values)) != 3 or source not in candidate_sources(outer, query):
        raise ProtocolError("Fixed-bank row requires distinct legal H/q/e IDs.")
    return outer, query, source  # type: ignore[return-value]


def _bounded(value: object, name: str, lower: float, upper: float) -> float:
    result = finite(value, name)
    if result < lower or result > upper:
        raise ProtocolError(f"{name} must be in [{lower}, {upper}].")
    return result


@dataclass(frozen=True)
class FixedBankFeatureRow:
    """Label-free support features for one known-bank candidate endpoint."""

    outer_target_id: str
    query_id: str
    candidate_source: str
    source_feature_row_hash: str
    metadata_similarity: float
    pooled_row_weighted_abs_shift: float
    equal_case_abs_shift: float
    case_abs_shift_sd: float
    equal_case_signed_margin: float
    case_balanced_flip_rate: float
    case_balanced_entropy_change: float
    case_balanced_reconstruction: float
    case_balanced_kl: float
    case_balanced_log_mmd: float
    probability_role_used: str = "support_only"
    labels_used: bool = False
    evaluation_probabilities_used_as_features: bool = False
    known_fixed_bank_reuse: bool = True
    unseen_expert_transfer: bool = False
    feature_row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _row_key(*self.row_key)
        source_hash = require_sha256(
            self.source_feature_row_hash, "source_feature_row_hash"
        )
        values = {
            "metadata_similarity": _bounded(
                self.metadata_similarity, "metadata_similarity", 0.0, 1.0
            ),
            "pooled_row_weighted_abs_shift": _bounded(
                self.pooled_row_weighted_abs_shift,
                "pooled_row_weighted_abs_shift",
                0.0,
                1.0,
            ),
            "equal_case_abs_shift": _bounded(
                self.equal_case_abs_shift, "equal_case_abs_shift", 0.0, 1.0
            ),
            "case_abs_shift_sd": _bounded(
                self.case_abs_shift_sd, "case_abs_shift_sd", 0.0, 1.0
            ),
            "equal_case_signed_margin": _bounded(
                self.equal_case_signed_margin,
                "equal_case_signed_margin",
                -1.0,
                1.0,
            ),
            "case_balanced_flip_rate": _bounded(
                self.case_balanced_flip_rate,
                "case_balanced_flip_rate",
                0.0,
                1.0,
            ),
            "case_balanced_entropy_change": _bounded(
                self.case_balanced_entropy_change,
                "case_balanced_entropy_change",
                -math.log(2.0),
                math.log(2.0),
            ),
            "case_balanced_reconstruction": finite(
                self.case_balanced_reconstruction,
                "case_balanced_reconstruction",
            ),
            "case_balanced_kl": finite(self.case_balanced_kl, "case_balanced_kl"),
            "case_balanced_log_mmd": finite(
                self.case_balanced_log_mmd, "case_balanced_log_mmd"
            ),
        }
        if any(
            values[name] < 0.0
            for name in (
                "case_balanced_reconstruction",
                "case_balanced_kl",
                "case_balanced_log_mmd",
            )
        ):
            raise ProtocolError("Rich compatibility summaries must be nonnegative.")
        if (
            self.probability_role_used != "support_only"
            or self.labels_used is not False
            or self.evaluation_probabilities_used_as_features is not False
            or self.known_fixed_bank_reuse is not True
            or self.unseen_expert_transfer is not False
        ):
            raise ProtocolError("Fixed-bank features crossed the claim boundary.")
        object.__setattr__(self, "source_feature_row_hash", source_hash)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "feature_row_hash", canonical_hash(self._unhashed()))

    @property
    def row_key(self) -> tuple[str, str, str]:
        return self.outer_target_id, self.query_id, self.candidate_source

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": FEATURE_ROW_SCHEMA,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "feature_row_hash"
            },
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "feature_row_hash": self.feature_row_hash}


@dataclass(frozen=True)
class FixedBankResponseRow:
    """Post-seal terminal exact response plus isolated smooth description."""

    outer_target_id: str
    query_id: str
    candidate_source: str
    feature_row_hash: str
    source_response_row_hash: str
    exact_bacc_delta: float
    smooth_bacc_delta: float
    support_eval_disjoint: bool = True
    features_sealed_before_label_access: bool = True
    exact_response_is_primary: bool = True
    smooth_response_is_descriptive_only: bool = True
    policy_update_authorized: bool = False
    response_row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _row_key(*self.row_key)
        feature_hash = require_sha256(self.feature_row_hash, "feature_row_hash")
        source_hash = require_sha256(
            self.source_response_row_hash, "source_response_row_hash"
        )
        exact = probability_delta(self.exact_bacc_delta, "exact_bacc_delta")
        smooth = probability_delta(self.smooth_bacc_delta, "smooth_bacc_delta")
        if (
            self.support_eval_disjoint is not True
            or self.features_sealed_before_label_access is not True
            or self.exact_response_is_primary is not True
            or self.smooth_response_is_descriptive_only is not True
            or self.policy_update_authorized is not False
        ):
            raise ProtocolError("Fixed-bank response crossed the terminal boundary.")
        object.__setattr__(self, "feature_row_hash", feature_hash)
        object.__setattr__(self, "source_response_row_hash", source_hash)
        object.__setattr__(self, "exact_bacc_delta", exact)
        object.__setattr__(self, "smooth_bacc_delta", smooth)
        object.__setattr__(self, "response_row_hash", canonical_hash(self._unhashed()))

    @property
    def row_key(self) -> tuple[str, str, str]:
        return self.outer_target_id, self.query_id, self.candidate_source

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": RESPONSE_ROW_SCHEMA,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "response_row_hash"
            },
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "response_row_hash": self.response_row_hash}


@dataclass(frozen=True)
class FixedBankDataset:
    """Complete aligned 504-row feature/response surface."""

    feature_rows: tuple[FixedBankFeatureRow, ...]
    response_rows: tuple[FixedBankResponseRow, ...]
    feature_surface_hash: str = field(init=False)
    exact_response_surface_hash: str = field(init=False)
    smooth_response_surface_hash: str = field(init=False)
    dataset_hash: str = field(init=False)

    def __post_init__(self) -> None:
        keys = expected_row_keys()
        if (
            tuple(row.row_key for row in self.feature_rows) != keys
            or tuple(row.row_key for row in self.response_rows) != keys
            or len({row.feature_row_hash for row in self.feature_rows}) != len(keys)
            or len({row.response_row_hash for row in self.response_rows}) != len(keys)
        ):
            raise ProtocolError("Fixed-bank dataset geometry/order drifted.")
        for feature, response in zip(
            self.feature_rows, self.response_rows, strict=True
        ):
            if response.feature_row_hash != feature.feature_row_hash:
                raise ProtocolError("Fixed-bank feature/response hash linkage drifted.")
        object.__setattr__(
            self,
            "feature_surface_hash",
            canonical_hash(
                {
                    "schema_version": f"{DATASET_SCHEMA}_feature_surface",
                    "row_keys": [list(key) for key in keys],
                    "feature_row_hashes": [
                        row.feature_row_hash for row in self.feature_rows
                    ],
                }
            ),
        )
        object.__setattr__(
            self,
            "exact_response_surface_hash",
            canonical_hash(
                {
                    "schema_version": f"{DATASET_SCHEMA}_exact_response_surface",
                    "row_keys": [list(key) for key in keys],
                    "feature_row_hashes": [
                        row.feature_row_hash for row in self.response_rows
                    ],
                    "exact_bacc_deltas": [
                        row.exact_bacc_delta for row in self.response_rows
                    ],
                }
            ),
        )
        object.__setattr__(
            self,
            "smooth_response_surface_hash",
            canonical_hash(
                {
                    "schema_version": f"{DATASET_SCHEMA}_smooth_response_surface",
                    "row_keys": [list(key) for key in keys],
                    "feature_row_hashes": [
                        row.feature_row_hash for row in self.response_rows
                    ],
                    "smooth_bacc_deltas": [
                        row.smooth_bacc_delta for row in self.response_rows
                    ],
                }
            ),
        )
        object.__setattr__(self, "dataset_hash", canonical_hash(self._unhashed()))

    @property
    def row_keys(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(row.row_key for row in self.feature_rows)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": DATASET_SCHEMA,
            "centers": list(CENTERS),
            "row_keys": [list(key) for key in self.row_keys],
            "feature_row_hashes": [row.feature_row_hash for row in self.feature_rows],
            "response_row_hashes": [
                row.response_row_hash for row in self.response_rows
            ],
            "feature_surface_hash": self.feature_surface_hash,
            "exact_response_surface_hash": self.exact_response_surface_hash,
            "smooth_response_surface_hash": self.smooth_response_surface_hash,
            "known_fixed_bank_reuse": True,
            "unseen_expert_transfer": False,
            "exact_response_is_primary": True,
            "smooth_response_is_descriptive_only": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "dataset_hash": self.dataset_hash}


def feature_row_from_payload(payload: Mapping[str, object]) -> FixedBankFeatureRow:
    require_exact_keys(payload, _FEATURE_FIELDS, "Fixed-bank feature row")
    if payload["schema_version"] != FEATURE_ROW_SCHEMA:
        raise ProtocolError("Fixed-bank feature row schema version drifted.")
    supplied_hash = require_sha256(payload["feature_row_hash"], "feature_row_hash")
    values = {key: payload[key] for key in _FEATURE_FIELDS if key not in {"schema_version", "feature_row_hash"}}
    row = FixedBankFeatureRow(**values)  # type: ignore[arg-type]
    if row.feature_row_hash != supplied_hash:
        raise ProtocolError("Fixed-bank feature row hash drifted.")
    return row


def response_row_from_payload(payload: Mapping[str, object]) -> FixedBankResponseRow:
    require_exact_keys(payload, _RESPONSE_FIELDS, "Fixed-bank response row")
    if payload["schema_version"] != RESPONSE_ROW_SCHEMA:
        raise ProtocolError("Fixed-bank response row schema version drifted.")
    supplied_hash = require_sha256(payload["response_row_hash"], "response_row_hash")
    values = {key: payload[key] for key in _RESPONSE_FIELDS if key not in {"schema_version", "response_row_hash"}}
    row = FixedBankResponseRow(**values)  # type: ignore[arg-type]
    if row.response_row_hash != supplied_hash:
        raise ProtocolError("Fixed-bank response row hash drifted.")
    return row


def ordered_rows_by_key(
    rows: Sequence[FixedBankFeatureRow | FixedBankResponseRow],
) -> tuple[FixedBankFeatureRow | FixedBankResponseRow, ...]:
    by_key = {row.row_key: row for row in rows}
    if len(by_key) != len(rows) or set(by_key) != set(expected_row_keys()):
        raise ProtocolError("Fixed-bank rows do not cover the complete surface.")
    return tuple(by_key[key] for key in expected_row_keys())


__all__ = (
    "FixedBankDataset",
    "FixedBankFeatureRow",
    "FixedBankResponseRow",
    "feature_row_from_payload",
    "ordered_rows_by_key",
    "response_row_from_payload",
)
