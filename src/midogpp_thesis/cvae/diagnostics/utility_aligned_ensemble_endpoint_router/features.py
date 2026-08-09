"""Label-free Stage-90 M0/M1 feature production.

Seed-level distribution summaries remain diagnostics.  The fitted designs use
one candidate row after exact-nine collapse, with at most one local predictor:
the versioned ensemble-first support probability shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from ...routing.utility_aligned.ensemble_endpoint_contracts import (
    SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
    SupportActionProbabilityShift,
)
from ...routing.utility_aligned.ensemble_feature_contracts import (
    GLOBAL_SOURCE_CONTROL_NAME,
    EnsembleFeatureSurface,
    GlobalSourceControl,
)
from ...routing.utility_aligned.ensemble_features import (
    aggregate_candidate_seed_features,
    build_ensemble_feature_surface,
    cyclically_permute_target_scalar,
)
from ...routing.utility_aligned.ensemble_target import (
    derive_label_free_global_source_control,
)
from ...routing.utility_aligned.row_contracts import INNER_ROLE, TARGET_ROLE
from ...routing.utility_aligned.surface_contracts import CandidateFeatureRow
from .contracts import (
    CENTERS,
    EXPECTED_INNER_FEATURE_SEED_ROW_COUNT,
    EXPECTED_TARGET_FEATURE_SEED_ROW_COUNT,
    FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
    INNER_CANDIDATE_COUNT,
    PERMUTATION_SEED,
    SEED_PAIR_COUNT,
    TARGET_CANDIDATE_COUNT,
    candidate_sources,
)


@dataclass(frozen=True)
class HeldoutEnsembleFeatureSurfaces:
    """M0/M1/P source-inner and two-case target surfaces for one ``H``."""

    outer_target_id: str
    global_source_control: GlobalSourceControl
    inner_support_shift_lock_hash: str
    target_support_shift_lock_hash: str
    target_probe_seal_hash: str
    inner_m0: EnsembleFeatureSurface
    inner_m1: EnsembleFeatureSurface
    inner_permuted: EnsembleFeatureSurface
    target_m0: EnsembleFeatureSurface
    target_m1: EnsembleFeatureSurface
    target_permuted: EnsembleFeatureSurface
    surface_hash: str

    def __post_init__(self) -> None:
        target = str(self.outer_target_id)
        sources = candidate_sources(target)
        surfaces = (
            self.inner_m0,
            self.inner_m1,
            self.inner_permuted,
            self.target_m0,
            self.target_m1,
            self.target_permuted,
        )
        locks = (
            self.inner_support_shift_lock_hash,
            self.target_support_shift_lock_hash,
            self.target_probe_seal_hash,
        )
        if (
            not isinstance(self.global_source_control, GlobalSourceControl)
            or self.global_source_control.outer_target_id != target
            or any(not isinstance(surface, EnsembleFeatureSurface) for surface in surfaces)
            or any(surface.outer_target_id != target for surface in surfaces)
            or any(surface.candidate_sources != sources for surface in surfaces)
            or any(
                row.support_case_count != FIXED_SUPPORT_CASE_COUNT_PER_CENTER
                for surface in surfaces
                for row in surface.rows
            )
            or len(self.inner_m0.rows) != TARGET_CANDIDATE_COUNT * INNER_CANDIDATE_COUNT
            or len(self.inner_m1.rows) != TARGET_CANDIDATE_COUNT * INNER_CANDIDATE_COUNT
            or len(self.target_m0.rows) != TARGET_CANDIDATE_COUNT
            or len(self.target_m1.rows) != TARGET_CANDIDATE_COUNT
            or self.inner_m0.role != INNER_ROLE
            or self.inner_m1.role != INNER_ROLE
            or self.inner_permuted.role != INNER_ROLE
            or self.target_m0.role != TARGET_ROLE
            or self.target_m1.role != TARGET_ROLE
            or self.target_permuted.role != TARGET_ROLE
            or self.inner_m0.feature_names != (GLOBAL_SOURCE_CONTROL_NAME,)
            or self.target_m0.feature_names != (GLOBAL_SOURCE_CONTROL_NAME,)
            or self.inner_m1.feature_names
            != (
                GLOBAL_SOURCE_CONTROL_NAME,
                f"target_local::{SUPPORT_ACTION_PROBABILITY_SHIFT_NAME}",
            )
            or self.target_m1.feature_names != self.inner_m1.feature_names
            or self.inner_permuted.feature_names != self.inner_m1.feature_names
            or self.target_permuted.feature_names != self.target_m1.feature_names
            or self.inner_m0.permutation_seed is not None
            or self.inner_m1.permutation_seed is not None
            or self.target_m0.permutation_seed is not None
            or self.target_m1.permutation_seed is not None
            or self.inner_permuted.permutation_seed != PERMUTATION_SEED
            or self.target_permuted.permutation_seed != PERMUTATION_SEED
            or any(not _is_hash(value) for value in locks)
        ):
            raise ProtocolError("Held-out ensemble feature boundary drifted.")
        expected = canonical_sha256(self._unhashed_payload(target=target))
        if self.surface_hash != expected:
            raise ProtocolError("Held-out ensemble feature hash drifted.")
        object.__setattr__(self, "outer_target_id", target)

    def _unhashed_payload(self, *, target: str | None = None) -> dict[str, object]:
        return {
            "schema_version": (
                "midogpp_utility_aligned_stage90_ensemble_heldout_features_v1"
            ),
            "outer_target_id": target or self.outer_target_id,
            "global_source_control_hash": self.global_source_control.provenance_hash,
            "inner_support_shift_lock_hash": self.inner_support_shift_lock_hash,
            "target_support_shift_lock_hash": self.target_support_shift_lock_hash,
            "target_probe_seal_hash": self.target_probe_seal_hash,
            "inner_m0_surface_hash": self.inner_m0.surface_hash,
            "inner_m1_surface_hash": self.inner_m1.surface_hash,
            "inner_permuted_surface_hash": self.inner_permuted.surface_hash,
            "target_m0_surface_hash": self.target_m0.surface_hash,
            "target_m1_surface_hash": self.target_m1.surface_hash,
            "target_permuted_surface_hash": self.target_permuted.surface_hash,
            "inner_candidate_response_count": len(self.inner_m1.rows),
            "target_candidate_count": len(self.target_m1.rows),
            "fixed_support_case_count": FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
            "seed_rows_are_independent_observations": False,
            "technical_seed_spread_may_feed_model": False,
            "labels_used": False,
            "evaluation_embeddings_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "surface_hash": self.surface_hash}


@dataclass(frozen=True)
class Stage90EnsembleFeatureSurfaceSet:
    by_target: Mapping[str, HeldoutEnsembleFeatureSurfaces]
    surface_hash: str

    def __post_init__(self) -> None:
        values = {str(key): value for key, value in self.by_target.items()}
        if (
            tuple(values) != CENTERS
            or any(
                not isinstance(value, HeldoutEnsembleFeatureSurfaces)
                or value.outer_target_id != target
                for target, value in values.items()
            )
            or sum(len(value.inner_m1.rows) for value in values.values()) != 504
            or sum(len(value.target_m1.rows) for value in values.values()) != 72
        ):
            raise ProtocolError("Stage-90 ensemble feature set is incomplete.")
        expected = canonical_sha256(_surface_set_payload(values))
        if self.surface_hash != expected:
            raise ProtocolError("Stage-90 ensemble feature-set hash drifted.")
        object.__setattr__(self, "by_target", MappingProxyType(values))

    def to_payload(self) -> dict[str, object]:
        return {**_surface_set_payload(self.by_target), "surface_hash": self.surface_hash}


def build_heldout_ensemble_feature_surfaces(
    inner_seed_feature_rows: Sequence[CandidateFeatureRow],
    target_seed_feature_rows: Sequence[CandidateFeatureRow],
    *,
    inner_support_shift_by_candidate: Mapping[
        tuple[str, str, str], SupportActionProbabilityShift
    ],
    target_support_shift_by_candidate: Mapping[
        tuple[str, str, str], SupportActionProbabilityShift
    ],
    inner_support_shift_lock_hash: str,
    target_support_shift_lock_hash: str,
    target_probe_seal_hash: str,
    outer_target_id: object,
    permutation_seed: int = PERMUTATION_SEED,
) -> HeldoutEnsembleFeatureSurfaces:
    """Build one label-free held-out feature bundle for bounded worker use."""

    target = str(outer_target_id)
    sources = candidate_sources(target)
    inner = tuple(inner_seed_feature_rows)
    target_rows = tuple(target_seed_feature_rows)
    if permutation_seed != PERMUTATION_SEED:
        raise ProtocolError("Stage-90 ensemble permutation seed cannot be tuned.")
    if any(
        not _is_hash(value)
        for value in (
            inner_support_shift_lock_hash,
            target_support_shift_lock_hash,
            target_probe_seal_hash,
        )
    ):
        raise ProtocolError("Stage-90 ensemble support shifts require sealed locks.")
    if (
        len(inner) != TARGET_CANDIDATE_COUNT * INNER_CANDIDATE_COUNT * SEED_PAIR_COUNT
        or len(target_rows) != TARGET_CANDIDATE_COUNT * SEED_PAIR_COUNT
        or any(
            not isinstance(row, CandidateFeatureRow)
            or row.role != INNER_ROLE
            or row.outer_target_id != target
            or row.support_case_count != FIXED_SUPPORT_CASE_COUNT_PER_CENTER
            for row in inner
        )
        or any(
            not isinstance(row, CandidateFeatureRow)
            or row.role != TARGET_ROLE
            or row.outer_target_id != target
            or row.query_id != target
            or row.support_case_count != FIXED_SUPPORT_CASE_COUNT_PER_CENTER
            for row in target_rows
        )
        or {row.query_id for row in inner} != set(sources)
    ):
        raise ProtocolError("Held-out ensemble seed feature geometry drifted.")
    global_control = derive_label_free_global_source_control(inner)
    inner_m0_rows = aggregate_candidate_seed_features(inner)
    inner_m1_rows = aggregate_candidate_seed_features(
        inner,
        support_action_shift_by_candidate=inner_support_shift_by_candidate,
    )
    target_m0_rows = aggregate_candidate_seed_features(target_rows)
    target_m1_rows = aggregate_candidate_seed_features(
        target_rows,
        support_action_shift_by_candidate=target_support_shift_by_candidate,
    )
    surface_kwargs = {
        "global_source_control_by_source": global_control.value_by_source,
        "global_source_control_semantics": global_control.semantics,
        "global_source_control_provenance_hash": global_control.provenance_hash,
    }
    inner_m0 = build_ensemble_feature_surface(inner_m0_rows, **surface_kwargs)
    inner_m1 = build_ensemble_feature_surface(inner_m1_rows, **surface_kwargs)
    target_m0 = build_ensemble_feature_surface(target_m0_rows, **surface_kwargs)
    target_m1 = build_ensemble_feature_surface(target_m1_rows, **surface_kwargs)
    inner_permuted = cyclically_permute_target_scalar(
        inner_m1, permutation_seed=PERMUTATION_SEED
    )
    target_permuted = cyclically_permute_target_scalar(
        target_m1, permutation_seed=PERMUTATION_SEED
    )
    unhashed = {
        "schema_version": "midogpp_utility_aligned_stage90_ensemble_heldout_features_v1",
        "outer_target_id": target,
        "global_source_control_hash": global_control.provenance_hash,
        "inner_support_shift_lock_hash": inner_support_shift_lock_hash,
        "target_support_shift_lock_hash": target_support_shift_lock_hash,
        "target_probe_seal_hash": target_probe_seal_hash,
        "inner_m0_surface_hash": inner_m0.surface_hash,
        "inner_m1_surface_hash": inner_m1.surface_hash,
        "inner_permuted_surface_hash": inner_permuted.surface_hash,
        "target_m0_surface_hash": target_m0.surface_hash,
        "target_m1_surface_hash": target_m1.surface_hash,
        "target_permuted_surface_hash": target_permuted.surface_hash,
        "inner_candidate_response_count": len(inner_m1.rows),
        "target_candidate_count": len(target_m1.rows),
        "fixed_support_case_count": FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
        "seed_rows_are_independent_observations": False,
        "technical_seed_spread_may_feed_model": False,
        "labels_used": False,
        "evaluation_embeddings_used": False,
    }
    return HeldoutEnsembleFeatureSurfaces(
        outer_target_id=target,
        global_source_control=global_control,
        inner_support_shift_lock_hash=inner_support_shift_lock_hash,
        target_support_shift_lock_hash=target_support_shift_lock_hash,
        target_probe_seal_hash=target_probe_seal_hash,
        inner_m0=inner_m0,
        inner_m1=inner_m1,
        inner_permuted=inner_permuted,
        target_m0=target_m0,
        target_m1=target_m1,
        target_permuted=target_permuted,
        surface_hash=canonical_sha256(unhashed),
    )


def build_stage90_ensemble_feature_surface_set(
    inner_seed_feature_rows: Sequence[CandidateFeatureRow],
    target_seed_feature_rows: Sequence[CandidateFeatureRow],
    *,
    inner_support_shift_by_candidate: Mapping[
        tuple[str, str, str], SupportActionProbabilityShift
    ],
    target_support_shift_by_candidate: Mapping[
        tuple[str, str, str], SupportActionProbabilityShift
    ],
    inner_support_shift_lock_hash: str,
    target_support_shift_lock_hash: str,
    target_probe_seal_hash: str,
) -> Stage90EnsembleFeatureSurfaceSet:
    """Build all nine H-indexed bundles without a label-bearing argument."""

    inner = tuple(inner_seed_feature_rows)
    target = tuple(target_seed_feature_rows)
    if (
        len(inner) != EXPECTED_INNER_FEATURE_SEED_ROW_COUNT
        or len(target) != EXPECTED_TARGET_FEATURE_SEED_ROW_COUNT
    ):
        raise ProtocolError("Stage-90 ensemble seed feature set is incomplete.")
    inner_shifts = dict(inner_support_shift_by_candidate)
    target_shifts = dict(target_support_shift_by_candidate)
    by_target: dict[str, HeldoutEnsembleFeatureSurfaces] = {}
    for heldout in CENTERS:
        by_target[heldout] = build_heldout_ensemble_feature_surfaces(
            tuple(row for row in inner if row.outer_target_id == heldout),
            tuple(row for row in target if row.outer_target_id == heldout),
            inner_support_shift_by_candidate={
                key: value for key, value in inner_shifts.items() if key[0] == heldout
            },
            target_support_shift_by_candidate={
                key: value for key, value in target_shifts.items() if key[0] == heldout
            },
            inner_support_shift_lock_hash=inner_support_shift_lock_hash,
            target_support_shift_lock_hash=target_support_shift_lock_hash,
            target_probe_seal_hash=target_probe_seal_hash,
            outer_target_id=heldout,
        )
    payload = _surface_set_payload(by_target)
    return Stage90EnsembleFeatureSurfaceSet(
        by_target=by_target,
        surface_hash=canonical_sha256(payload),
    )


def _surface_set_payload(
    values: Mapping[str, HeldoutEnsembleFeatureSurfaces],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_utility_aligned_stage90_ensemble_feature_set_v1",
        "centers": list(CENTERS),
        "surface_hashes_by_target": {
            target: values[target].surface_hash for target in CENTERS
        },
        "inner_support_shift_lock_hashes": sorted(
            {values[target].inner_support_shift_lock_hash for target in CENTERS}
        ),
        "target_support_shift_lock_hashes": sorted(
            {values[target].target_support_shift_lock_hash for target in CENTERS}
        ),
        "target_probe_seal_hashes": sorted(
            {values[target].target_probe_seal_hash for target in CENTERS}
        ),
        "candidate_response_count": sum(
            len(values[target].inner_m1.rows) for target in CENTERS
        ),
        "target_candidate_count": sum(
            len(values[target].target_m1.rows) for target in CENTERS
        ),
        "fixed_support_case_count": FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
        "seed_rows_are_independent_observations": False,
        "technical_seed_spread_may_feed_model": False,
        "target_support_labels_used": False,
        "target_evaluation_embeddings_used": False,
    }


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 16 and value.strip() == value


__all__ = (
    "HeldoutEnsembleFeatureSurfaces",
    "Stage90EnsembleFeatureSurfaceSet",
    "build_heldout_ensemble_feature_surfaces",
    "build_stage90_ensemble_feature_surface_set",
)
