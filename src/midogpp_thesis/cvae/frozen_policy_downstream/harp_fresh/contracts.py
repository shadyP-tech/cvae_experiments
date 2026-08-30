"""Fresh MIDOG++ target, policy, and prediction contracts for HARP."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from ...runtime.harp_probability_menu.hashing import (
    canonical_sha256,
    identity_sequence_sha256,
    raw_array_sha256,
    require_digest,
    require_sha256,
)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value or "\x00" in value:
        raise ProtocolError(f"Fresh HARP {name} must be a canonical identity.")
    return value


def _case_map(
    value: Mapping[object, object], *, role: str
) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or tuple(str(key) for key in value) != CENTERS:
        raise ProtocolError(f"Fresh HARP {role} cases must cover all centers in order.")
    normalized: dict[str, tuple[str, ...]] = {}
    for center in CENTERS:
        raw = value[center]
        if not isinstance(raw, (tuple, list)):
            raise ProtocolError(f"Fresh HARP {role} cases must be sequences.")
        rows = tuple(_identity(item, name=f"{role} case") for item in raw)
        if not rows or len(rows) != len(set(rows)):
            raise ProtocolError(f"Fresh HARP {role} cases must be nonempty and unique.")
        normalized[center] = rows
    return MappingProxyType(normalized)


@dataclass(frozen=True, kw_only=True)
class HarpFreshReservation:
    reservation_id: str
    support_case_ids_by_center: Mapping[str, tuple[str, ...]]
    evaluation_case_ids_by_center: Mapping[str, tuple[str, ...]]
    dataset_family: str = "MIDOG++"
    status: str = "ACTIVE"
    fresh_unconsumed_surface: bool = True
    labels_opened: bool = False
    previously_evaluated: bool = False
    upstream_reservation_hash: str | None = field(default=None, repr=False)
    reservation_semantic_hash: str = field(init=False)
    reservation_hash: str = field(init=False)

    def __post_init__(self) -> None:
        reservation_id = _identity(self.reservation_id, name="reservation")
        support = _case_map(self.support_case_ids_by_center, role="support")
        evaluation = _case_map(self.evaluation_case_ids_by_center, role="evaluation")
        support_all = tuple(case for center in CENTERS for case in support[center])
        evaluation_all = tuple(case for center in CENTERS for case in evaluation[center])
        if (
            self.dataset_family != "MIDOG++"
            or self.status != "ACTIVE"
            or self.fresh_unconsumed_surface is not True
            or self.labels_opened is not False
            or self.previously_evaluated is not False
            or len(support_all) != len(set(support_all))
            or len(evaluation_all) != len(set(evaluation_all))
            or set(support_all).intersection(evaluation_all)
        ):
            raise ProtocolError(
                "Fresh HARP requires one active, unopened, case-disjoint reservation."
            )
        payload = {
            "schema_version": "midogpp_harp_fresh_reservation_v1",
            "reservation_id": reservation_id,
            "dataset_family": "MIDOG++",
            "status": "ACTIVE",
            "support_case_ids_by_center": {
                center: list(support[center]) for center in CENTERS
            },
            "evaluation_case_ids_by_center": {
                center: list(evaluation[center]) for center in CENTERS
            },
            "support_evaluation_case_disjoint": True,
            "fresh_unconsumed_surface": True,
            "labels_opened": False,
            "previously_evaluated": False,
        }
        semantic_hash = canonical_sha256(payload)
        upstream_hash = self.upstream_reservation_hash
        if upstream_hash is not None:
            upstream_hash = require_sha256(
                upstream_hash, name="upstream fresh reservation"
            )
        object.__setattr__(self, "reservation_id", reservation_id)
        object.__setattr__(self, "support_case_ids_by_center", support)
        object.__setattr__(self, "evaluation_case_ids_by_center", evaluation)
        object.__setattr__(self, "upstream_reservation_hash", upstream_hash)
        object.__setattr__(self, "reservation_semantic_hash", semantic_hash)
        object.__setattr__(
            self,
            "reservation_hash",
            semantic_hash if upstream_hash is None else upstream_hash,
        )


@dataclass(frozen=True, kw_only=True)
class HarpFreshTargetFrame:
    center: str
    embeddings: np.ndarray
    row_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    frame_hash: str = field(init=False)

    def __post_init__(self) -> None:
        center = str(self.center)
        if center not in CENTERS:
            raise ProtocolError("Fresh HARP target frame center is unknown.")
        rows = tuple(_identity(value, name="row") for value in self.row_ids)
        cases = tuple(_identity(value, name="case") for value in self.case_ids)
        values = np.asarray(self.embeddings)
        if (
            values.dtype != np.dtype("float32")
            or values.ndim != 2
            or values.shape != (len(rows), COMMON_OUTPUT_DIM)
            or not rows
            or len(rows) != len(set(rows))
            or len(cases) != len(rows)
            or not np.isfinite(values).all()
        ):
            raise ProtocolError("Fresh HARP target frame geometry drifted.")
        array = np.ascontiguousarray(values, dtype=np.float32)
        payload = {
            "schema_version": "midogpp_harp_fresh_target_frame_v1",
            "center": center,
            "row_identity_sha256": identity_sequence_sha256(rows, identity_kind="row"),
            "case_identity_sha256": identity_sequence_sha256(cases, identity_kind="case"),
            "embedding_bytes_sha256": raw_array_sha256(array),
            "shape": list(array.shape),
            "dtype": "float32",
            "labels_persisted": False,
        }
        array.setflags(write=False)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "embeddings", array)
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "frame_hash", canonical_sha256(payload))


@dataclass(frozen=True, kw_only=True)
class HarpFreshTargetCache:
    reservation: HarpFreshReservation
    frames_by_center: Mapping[str, HarpFreshTargetFrame]
    labels_persisted: bool = False
    cache_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.reservation, HarpFreshReservation):
            raise ProtocolError("Fresh HARP cache lacks a typed reservation.")
        if self.labels_persisted is not False:
            raise ProtocolError("Fresh HARP feature caches cannot persist labels.")
        if not isinstance(self.frames_by_center, Mapping) or tuple(self.frames_by_center) != CENTERS:
            raise ProtocolError("Fresh HARP cache must cover all centers in order.")
        frames = {
            center: self.frames_by_center[center]
            for center in CENTERS
        }
        if any(
            not isinstance(frame, HarpFreshTargetFrame) or frame.center != center
            for center, frame in frames.items()
        ):
            raise ProtocolError("Fresh HARP cache frame identity drifted.")
        all_rows: list[str] = []
        for center, frame in frames.items():
            if set(frame.case_ids) != set(
                self.reservation.evaluation_case_ids_by_center[center]
            ):
                raise ProtocolError("Fresh HARP cache does not match reserved cases.")
            all_rows.extend(frame.row_ids)
        if len(all_rows) != len(set(all_rows)):
            raise ProtocolError("Fresh HARP row identities must be globally unique.")
        payload = {
            "schema_version": "midogpp_harp_fresh_target_cache_v1",
            "dataset_family": "MIDOG++",
            "reservation_hash": self.reservation.reservation_hash,
            "frames": {center: frames[center].frame_hash for center in CENTERS},
            "labels_persisted": False,
        }
        object.__setattr__(self, "frames_by_center", MappingProxyType(frames))
        object.__setattr__(self, "cache_hash", canonical_sha256(payload))


@dataclass(frozen=True, kw_only=True)
class HarpFrozenPolicyMetadata:
    policy_lock_hash: str
    fresh_reservation_hash: str
    bank_hash: str
    generation_lock_hash: str
    source_cache_hash: str
    classifier_hash: str
    dataset_family: str = "MIDOG++"
    status: str = "FROZEN_BEFORE_TARGET_EVALUATION"
    target_outcomes_used: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_lock_hash", require_sha256(self.policy_lock_hash, name="policy lock")
        )
        object.__setattr__(
            self,
            "fresh_reservation_hash",
            require_sha256(self.fresh_reservation_hash, name="fresh reservation"),
        )
        for name in (
            "bank_hash",
            "generation_lock_hash",
            "source_cache_hash",
            "classifier_hash",
        ):
            object.__setattr__(self, name, require_digest(getattr(self, name), name=name))
        if (
            self.dataset_family != "MIDOG++"
            or self.status != "FROZEN_BEFORE_TARGET_EVALUATION"
            or self.target_outcomes_used is not False
        ):
            raise ProtocolError("Fresh HARP accepts only a frozen outcome-free MIDOG++ policy.")


@dataclass(frozen=True, kw_only=True)
class HarpFrozenExecutionLineage:
    """Explicit Stage-60 semantic identities and authoritative byte receipts."""

    bank_semantic_lock_hash: str
    generation_semantic_lock_hash: str
    source_stream_lock_hash: str
    source_stream_index_hash: str
    source_stream_content_hash: str
    classifier_config_hash: str
    expert_bank_index_sha256: str
    generation_lock_file_sha256: str
    source_cache_lock_sha256: str
    source_cache_index_sha256: str
    source_stream_artifact_binding_hash: str
    classifier_contract_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "bank_semantic_lock_hash",
            "generation_semantic_lock_hash",
            "source_stream_lock_hash",
            "source_stream_index_hash",
            "classifier_config_hash",
        ):
            object.__setattr__(
                self, name, require_digest(getattr(self, name), name=name)
            )
        for name in (
            "source_stream_content_hash",
            "expert_bank_index_sha256",
            "generation_lock_file_sha256",
            "source_cache_lock_sha256",
            "source_cache_index_sha256",
            "source_stream_artifact_binding_hash",
            "classifier_contract_sha256",
        ):
            object.__setattr__(
                self, name, require_sha256(getattr(self, name), name=name)
            )

    def to_payload(self) -> dict[str, str]:
        return {
            name: str(getattr(self, name))
            for name in (
                "bank_semantic_lock_hash",
                "generation_semantic_lock_hash",
                "source_stream_lock_hash",
                "source_stream_index_hash",
                "source_stream_content_hash",
                "classifier_config_hash",
                "expert_bank_index_sha256",
                "generation_lock_file_sha256",
                "source_cache_lock_sha256",
                "source_cache_index_sha256",
                "source_stream_artifact_binding_hash",
                "classifier_contract_sha256",
            )
        }


@dataclass(frozen=True, kw_only=True)
class HarpFreshPredictionOutput:
    probabilities: np.ndarray
    composition_hash: str
    scaler_state_hash: str

    def __post_init__(self) -> None:
        values = np.asarray(self.probabilities)
        if (
            values.dtype != np.dtype("float32")
            or values.ndim != 1
            or not len(values)
            or not np.isfinite(values).all()
            or np.any((values < 0.0) | (values > 1.0))
        ):
            raise ProtocolError("Fresh HARP predictor output must be float32 probabilities.")
        values = np.ascontiguousarray(values, dtype=np.float32)
        values.setflags(write=False)
        object.__setattr__(self, "probabilities", values)
        object.__setattr__(
            self,
            "composition_hash",
            require_digest(self.composition_hash, name="composition hash"),
        )
        object.__setattr__(
            self,
            "scaler_state_hash",
            require_digest(self.scaler_state_hash, name="scaler-state hash"),
        )


__all__ = (
    "HarpFreshPredictionOutput",
    "HarpFreshReservation",
    "HarpFreshTargetCache",
    "HarpFreshTargetFrame",
    "HarpFrozenExecutionLineage",
    "HarpFrozenPolicyMetadata",
)
