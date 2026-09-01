"""Candidate-aware label-free feature and opportunity construction."""

from __future__ import annotations

import struct
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ..pairwise_primitive_utility import ActionSurface, OpportunitySet, build_opportunity_set
from .contracts import (
    ActionKind,
    CandidateFeatureVector,
    CandidatePoolReceipt,
    CompatibilityReceipt,
    Direction,
    TargetAction,
    canonical_probability_bytes,
    finite,
)
from .hashing import probability_bytes_hash


COMPATIBILITY_FEATURE_NAMES = (
    "compatibility_mean_z",
    "compatibility_std_z",
    "compatibility_reciprocal_rank",
    "compatibility_rank_margin",
    "compatibility_available",
)
_FORBIDDEN_FEATURE_TOKENS = (
    "label",
    "truth",
    "outcome",
    "oracle",
    "bacc",
    "brier",
    "log_loss",
    "evaluation_endpoint",
)


def _label_free_base_features(values: Mapping[str, float]) -> tuple[tuple[str, ...], tuple[float, ...]]:
    rows = tuple(sorted((str(name), finite(value, name=f"feature {name}")) for name, value in values.items()))
    names = tuple(name for name, _ in rows)
    if (
        not rows
        or len(set(names)) != len(names)
        or set(names).intersection(COMPATIBILITY_FEATURE_NAMES)
        or any(any(token in name.lower() for token in _FORBIDDEN_FEATURE_TOKENS) for name in names)
    ):
        raise ProtocolError("Candidate base features are empty, duplicated, or outcome-bearing.")
    return names, tuple(value for _, value in rows)


def build_candidate_feature(
    *,
    candidate_pool: CandidatePoolReceipt,
    case_id: str,
    action_id: str,
    action_kind: ActionKind,
    direction: Direction,
    candidate_source_id: str | None,
    base_features: Mapping[str, float],
    probability_hash: str,
    compatibility: CompatibilityReceipt | None,
) -> CandidateFeatureVector:
    """Append calibrated compatibility to one label-free action descriptor.

    Uniform ``U`` receives explicit zero compatibility plus an availability
    bit, keeping the feature schema identical without pretending that U is an
    expert replica.  HXE requires a receipt from the same H/q/pool lineage.
    """

    if not isinstance(candidate_pool, CandidatePoolReceipt):
        raise ProtocolError("Candidate features require a typed candidate pool.")
    try:
        kind = ActionKind(action_kind)
        action_direction = Direction(direction)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Candidate feature action kind or direction is unknown.") from exc
    base_names, base_values = _label_free_base_features(base_features)
    if kind is ActionKind.HXE:
        if (
            not isinstance(compatibility, CompatibilityReceipt)
            or compatibility.outer_target_id != candidate_pool.outer_target_id
            or compatibility.query_center_id != candidate_pool.query_center_id
            or compatibility.candidate_pool_hash != candidate_pool.pool_hash
            or compatibility.candidate_source_id != candidate_source_id
            or candidate_source_id not in candidate_pool.candidate_center_ids
        ):
            raise ProtocolError("HXE feature compatibility escaped its H/q candidate pool.")
        compatibility_values = (
            compatibility.mean_z,
            compatibility.std_z,
            1.0 / float(compatibility.rank),
            compatibility.rank_margin,
            1.0,
        )
        compatibility_hash = compatibility.receipt_hash
    elif kind is ActionKind.U:
        if compatibility is not None or candidate_source_id is not None:
            raise ProtocolError("Uniform U cannot carry candidate compatibility.")
        compatibility_values = (0.0, 0.0, 0.0, 0.0, 0.0)
        compatibility_hash = None
    else:
        raise ProtocolError("Protected B is the implicit zero anchor, not a feature row.")
    return CandidateFeatureVector(
        outer_target_id=candidate_pool.outer_target_id,
        query_center_id=candidate_pool.query_center_id,
        case_id=str(case_id),
        action_id=str(action_id),
        action_kind=kind,
        direction=action_direction,
        candidate_source_id=candidate_source_id,
        feature_names=(*base_names, *COMPATIBILITY_FEATURE_NAMES),
        feature_values=(*base_values, *compatibility_values),
        candidate_pool_hash=candidate_pool.pool_hash,
        probability_hash=probability_hash,
        compatibility_receipt_hash=compatibility_hash,
    )


def build_label_free_opportunity(
    *, baseline_probability_bytes: Sequence[bytes], actions: Sequence[TargetAction]
) -> OpportunitySet:
    """Use the neutral exact-surface primitive to remove no-ops/duplicates."""

    baseline_raw = canonical_probability_bytes(baseline_probability_bytes)
    baseline = tuple(float(struct.unpack("<f", value)[0]) for value in baseline_raw)
    rows = tuple(sorted(tuple(actions), key=lambda row: row.feature.action_id))
    if (
        not rows
        or any(not isinstance(row, TargetAction) for row in rows)
        or any(len(row.probability_bytes) != len(baseline_raw) for row in rows)
        or len({row.feature.action_id for row in rows}) != len(rows)
    ):
        raise ProtocolError("Opportunity actions are empty, duplicated, or misaligned.")
    surfaces = tuple(
        ActionSurface(
            action_id=row.feature.action_id,
            family=row.feature.action_kind.value,
            direction=row.feature.direction.value,
            probabilities=tuple(struct.unpack("<f", value)[0] for value in row.probability_bytes),
        )
        for row in rows
    )
    return build_opportunity_set(
        baseline,
        surfaces,
        candidate_action_ids=tuple(row.action_id for row in surfaces),
    )


def probability_hash(values: Sequence[bytes]) -> str:
    """Public exact-byte hash helper for feature construction."""

    return probability_bytes_hash(canonical_probability_bytes(values))


__all__ = (
    "COMPATIBILITY_FEATURE_NAMES",
    "build_candidate_feature",
    "build_label_free_opportunity",
    "probability_hash",
)
