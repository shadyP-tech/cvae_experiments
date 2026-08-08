"""Label-free Stage-90 feature-surface assembly.

The numerical feature definition is stage-neutral and lives in
``routing.utility_aligned``.  This module adds only the consumed MIDOG++
center/cardinality/support locks required by the Stage-90 diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from ...routing.utility_aligned import (
    INNER_ROLE,
    TARGET_ROLE,
    CandidateFeatureRow,
    FeatureSurface,
    build_distributional_feature_surface,
)
from .contracts import (
    CENTERS,
    EXPECTED_INNER_UTILITY_ROW_COUNT,
    EXPECTED_TARGET_FEATURE_ROW_COUNT,
    FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
    SEED_PAIR_COUNT,
    candidate_sources,
)


@dataclass(frozen=True)
class HeldoutFeatureSurfaces:
    """Source-inner training features and two-case target features for one H."""

    outer_target_id: str
    inner: FeatureSurface
    target: FeatureSurface
    surface_hash: str

    def __post_init__(self) -> None:
        target = str(self.outer_target_id)
        sources = candidate_sources(target)
        if (
            self.inner.role != INNER_ROLE
            or self.target.role != TARGET_ROLE
            or self.inner.outer_target_id != target
            or self.target.outer_target_id != target
            or self.inner.candidate_sources != sources
            or self.target.candidate_sources != sources
            or len(self.inner.rows) != len(sources) * (len(sources) - 1) * SEED_PAIR_COUNT
            or len(self.target.rows) != len(sources) * SEED_PAIR_COUNT
            or any(
                row.support_case_count != FIXED_SUPPORT_CASE_COUNT_PER_CENTER
                for row in (*self.inner.rows, *self.target.rows)
            )
        ):
            raise ProtocolError("Stage-90 held-out feature geometry drifted.")
        expected = canonical_sha256(self._unhashed_payload(target=target))
        if self.surface_hash != expected:
            raise ProtocolError("Stage-90 held-out feature hash drifted.")
        object.__setattr__(self, "outer_target_id", target)

    def _unhashed_payload(self, *, target: str | None = None) -> dict[str, object]:
        return {
            "schema_version": "midogpp_utility_aligned_stage90_heldout_features_v1",
            "outer_target_id": target or self.outer_target_id,
            "candidate_sources": list(self.inner.candidate_sources),
            "inner_feature_surface_hash": self.inner.surface_hash,
            "target_feature_surface_hash": self.target.surface_hash,
            "inner_row_count": len(self.inner.rows),
            "target_row_count": len(self.target.rows),
            "fixed_support_case_count": FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
            "all_nine_seed_pairs_required": True,
            "labels_used": False,
            "evaluation_embeddings_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "surface_hash": self.surface_hash}


@dataclass(frozen=True)
class Stage90FeatureSurfaceSet:
    """Complete immutable H-indexed feature input to model fitting."""

    by_target: Mapping[str, HeldoutFeatureSurfaces]
    surface_hash: str

    def __post_init__(self) -> None:
        values = {str(key): value for key, value in self.by_target.items()}
        if (
            tuple(values) != CENTERS
            or any(
                not isinstance(value, HeldoutFeatureSurfaces)
                or value.outer_target_id != target
                for target, value in values.items()
            )
            or sum(len(value.inner.rows) for value in values.values())
            != EXPECTED_INNER_UTILITY_ROW_COUNT
            or sum(len(value.target.rows) for value in values.values())
            != EXPECTED_TARGET_FEATURE_ROW_COUNT
        ):
            raise ProtocolError("Stage-90 feature-surface set is incomplete.")
        expected = canonical_sha256(_surface_set_payload(values))
        if self.surface_hash != expected:
            raise ProtocolError("Stage-90 feature-surface set hash drifted.")
        object.__setattr__(self, "by_target", MappingProxyType(values))

    def to_payload(self) -> dict[str, object]:
        return {**_surface_set_payload(self.by_target), "surface_hash": self.surface_hash}


def build_stage90_feature_surface_set(
    inner_rows: Sequence[CandidateFeatureRow],
    target_rows: Sequence[CandidateFeatureRow],
) -> Stage90FeatureSurfaceSet:
    """Build all nine H-specific surfaces without accepting label arguments."""

    inner = _typed_rows(inner_rows, role=INNER_ROLE, expected=EXPECTED_INNER_UTILITY_ROW_COUNT)
    target = _typed_rows(
        target_rows,
        role=TARGET_ROLE,
        expected=EXPECTED_TARGET_FEATURE_ROW_COUNT,
    )
    by_target: dict[str, HeldoutFeatureSurfaces] = {}
    for heldout in CENTERS:
        by_target[heldout] = build_heldout_feature_surfaces(
            tuple(row for row in inner if row.outer_target_id == heldout),
            tuple(row for row in target if row.outer_target_id == heldout),
            outer_target_id=heldout,
        )
    payload = _surface_set_payload(by_target)
    return Stage90FeatureSurfaceSet(
        by_target=by_target,
        surface_hash=canonical_sha256(payload),
    )


def build_heldout_feature_surfaces(
    inner_rows: Sequence[CandidateFeatureRow],
    target_rows: Sequence[CandidateFeatureRow],
    *,
    outer_target_id: object,
) -> HeldoutFeatureSurfaces:
    """Build one H surface, allowing bounded per-H worker execution."""

    heldout = str(outer_target_id)
    sources = candidate_sources(heldout)
    inner = tuple(inner_rows)
    target = tuple(target_rows)
    if (
        len(inner) != len(sources) * (len(sources) - 1) * SEED_PAIR_COUNT
        or len(target) != len(sources) * SEED_PAIR_COUNT
        or any(
            not isinstance(row, CandidateFeatureRow)
            or row.role != INNER_ROLE
            or row.outer_target_id != heldout
            or row.support_case_count != FIXED_SUPPORT_CASE_COUNT_PER_CENTER
            for row in inner
        )
        or any(
            not isinstance(row, CandidateFeatureRow)
            or row.role != TARGET_ROLE
            or row.outer_target_id != heldout
            or row.support_case_count != FIXED_SUPPORT_CASE_COUNT_PER_CENTER
            for row in target
        )
    ):
        raise ProtocolError("Stage-90 per-H feature rows are incomplete or illegal.")
    inner_surface = build_distributional_feature_surface(inner)
    target_surface = build_distributional_feature_surface(target)
    unhashed = {
        "schema_version": "midogpp_utility_aligned_stage90_heldout_features_v1",
        "outer_target_id": heldout,
        "candidate_sources": list(inner_surface.candidate_sources),
        "inner_feature_surface_hash": inner_surface.surface_hash,
        "target_feature_surface_hash": target_surface.surface_hash,
        "inner_row_count": len(inner_surface.rows),
        "target_row_count": len(target_surface.rows),
        "fixed_support_case_count": FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
        "all_nine_seed_pairs_required": True,
        "labels_used": False,
        "evaluation_embeddings_used": False,
    }
    return HeldoutFeatureSurfaces(
        outer_target_id=heldout,
        inner=inner_surface,
        target=target_surface,
        surface_hash=canonical_sha256(unhashed),
    )


def build_inner_feature_surfaces(
    rows: Sequence[CandidateFeatureRow],
) -> Mapping[str, FeatureSurface]:
    """Build only the nine source-inner surfaces for staged execution."""

    typed = _typed_rows(rows, role=INNER_ROLE, expected=EXPECTED_INNER_UTILITY_ROW_COUNT)
    return MappingProxyType(
        {
            target: build_distributional_feature_surface(
                tuple(row for row in typed if row.outer_target_id == target)
            )
            for target in CENTERS
        }
    )


def build_target_feature_surfaces(
    rows: Sequence[CandidateFeatureRow],
) -> Mapping[str, FeatureSurface]:
    """Build only the nine unlabeled two-case target surfaces."""

    typed = _typed_rows(
        rows,
        role=TARGET_ROLE,
        expected=EXPECTED_TARGET_FEATURE_ROW_COUNT,
    )
    return MappingProxyType(
        {
            target: build_distributional_feature_surface(
                tuple(row for row in typed if row.outer_target_id == target)
            )
            for target in CENTERS
        }
    )


def _typed_rows(
    rows: Sequence[CandidateFeatureRow],
    *,
    role: str,
    expected: int,
) -> tuple[CandidateFeatureRow, ...]:
    values = tuple(rows)
    if (
        len(values) != expected
        or any(not isinstance(row, CandidateFeatureRow) or row.role != role for row in values)
        or {row.outer_target_id for row in values} != set(CENTERS)
        or any(
            row.support_case_count != FIXED_SUPPORT_CASE_COUNT_PER_CENTER
            for row in values
        )
    ):
        raise ProtocolError("Stage-90 label-free feature rows are incomplete or illegal.")
    return values


def _surface_set_payload(
    values: Mapping[str, HeldoutFeatureSurfaces],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_utility_aligned_stage90_feature_surface_set_v1",
        "centers": list(CENTERS),
        "surface_hashes_by_target": {
            target: values[target].surface_hash for target in CENTERS
        },
        "inner_row_count": sum(len(values[target].inner.rows) for target in CENTERS),
        "target_row_count": sum(len(values[target].target.rows) for target in CENTERS),
        "fixed_support_case_count": FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
        "target_support_labels_used": False,
        "target_evaluation_embeddings_used": False,
        "target_or_query_identity_features_used": False,
    }


__all__ = (
    "HeldoutFeatureSurfaces",
    "Stage90FeatureSurfaceSet",
    "build_heldout_feature_surfaces",
    "build_inner_feature_surfaces",
    "build_stage90_feature_surface_set",
    "build_target_feature_surfaces",
)
