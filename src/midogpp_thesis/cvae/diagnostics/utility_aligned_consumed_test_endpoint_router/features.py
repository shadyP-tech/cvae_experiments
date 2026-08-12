"""Label-free M0/M1/P feature production for the target-static router."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from ...routing.utility_aligned import (
    CaseBootstrapPlan,
    CandidateFeatureRow,
    EnsembleFeatureSurface,
    GlobalSourceControl,
    SupportActionProbabilityShift,
    TargetEnsembleFeatureProduction,
    TargetSupportActionShiftCase,
    aggregate_candidate_seed_features,
    build_case_bootstrap_plan,
    build_ensemble_feature_surface,
    build_target_ensemble_feature_surfaces,
    cyclically_permute_target_scalar,
    derive_label_free_global_source_control,
)
from ...routing.utility_aligned.ensemble_endpoint_contracts import (
    SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
)
from ...routing.utility_aligned.ensemble_feature_contracts import (
    GLOBAL_SOURCE_CONTROL_NAME,
)
from ...routing.utility_aligned.row_contracts import INNER_ROLE, TARGET_ROLE
from .contracts import (
    CENTERS,
    INNER_CANDIDATE_COUNT,
    M0_PREDICTOR_NAMES,
    M1_PREDICTOR_NAMES,
    PERMUTATION_SEED,
    SEED_PAIR_COUNT,
    SUPPORT_BOOTSTRAP_REPLICATES,
    SUPPORT_BOOTSTRAP_SEED,
    SUPPORT_CASE_COUNT_PER_CENTER,
    TARGET_CANDIDATE_COUNT,
    candidate_sources,
)


@dataclass(frozen=True)
class SourceInnerFeatureSurfaces:
    """One H-excluded M0, M1, and deterministic cyclic-P design."""

    outer_target_id: str
    global_source_control: GlobalSourceControl
    m0: EnsembleFeatureSurface
    m1: EnsembleFeatureSurface
    permutation: EnsembleFeatureSurface
    feature_input_seal_hash: str
    surface_hash: str

    def __post_init__(self) -> None:
        target = str(self.outer_target_id)
        sources = candidate_sources(target)
        if (
            not isinstance(self.global_source_control, GlobalSourceControl)
            or self.global_source_control.outer_target_id != target
            or any(
                not isinstance(surface, EnsembleFeatureSurface)
                or surface.outer_target_id != target
                or surface.role != INNER_ROLE
                or surface.candidate_sources != sources
                for surface in (self.m0, self.m1, self.permutation)
            )
            or self.m0.feature_names != M0_PREDICTOR_NAMES
            or self.m1.feature_names != M1_PREDICTOR_NAMES
            or self.permutation.feature_names != M1_PREDICTOR_NAMES
            or self.m0.permutation_seed is not None
            or self.m1.permutation_seed is not None
            or self.permutation.permutation_seed != PERMUTATION_SEED
            or len(self.m0.rows) != TARGET_CANDIDATE_COUNT * INNER_CANDIDATE_COUNT
            or len(self.m1.rows) != TARGET_CANDIDATE_COUNT * INNER_CANDIDATE_COUNT
            or not _text(self.feature_input_seal_hash)
        ):
            raise ProtocolError("Source-inner feature surface boundary drifted.")
        if self.surface_hash != canonical_sha256(self._unhashed_payload(target)):
            raise ProtocolError("Source-inner feature surface hash drifted.")
        object.__setattr__(self, "outer_target_id", target)

    def _unhashed_payload(self, target: str | None = None) -> dict[str, object]:
        return {
            "schema_version": "midogpp_consumed_test_source_inner_features_v1",
            "outer_target_id": target or self.outer_target_id,
            "global_source_control_hash": self.global_source_control.provenance_hash,
            "m0_surface_hash": self.m0.surface_hash,
            "m1_surface_hash": self.m1.surface_hash,
            "permutation_surface_hash": self.permutation.surface_hash,
            "feature_input_seal_hash": self.feature_input_seal_hash,
            "candidate_response_count": len(self.m1.rows),
            "predictor_names": {
                "M0": list(self.m0.feature_names),
                "M1": list(self.m1.feature_names),
                "P": list(self.permutation.feature_names),
            },
            "permutation_seed": PERMUTATION_SEED,
            "metadata_control_provenance": "experiment_manifest_only",
            "metadata_identity_or_labels_used_as_predictors": False,
            "technical_seed_rows_are_independent_observations": False,
            "support_labels_used": False,
            "same_outer_H_evaluation_rows_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "surface_hash": self.surface_hash}


@dataclass(frozen=True)
class TargetFeatureSurfaces:
    """Point plus exactly 32 label-free whole-case bootstrap feature surfaces."""

    target_id: str
    source_feature_surface_hash: str
    support_partition_lock_hash: str
    bootstrap_plan: CaseBootstrapPlan
    production: TargetEnsembleFeatureProduction
    target_feature_seal_hash: str
    feature_hash: str

    def __post_init__(self) -> None:
        target = str(self.target_id)
        if (
            target not in CENTERS
            or not _text(self.source_feature_surface_hash)
            or not _text(self.support_partition_lock_hash)
            or not _text(self.target_feature_seal_hash)
            or not isinstance(self.bootstrap_plan, CaseBootstrapPlan)
            or self.bootstrap_plan.target_id != target
            or self.bootstrap_plan.bootstrap_seed != SUPPORT_BOOTSTRAP_SEED
            or self.bootstrap_plan.replicate_count != SUPPORT_BOOTSTRAP_REPLICATES
            or len(self.bootstrap_plan.support_case_ids) != SUPPORT_CASE_COUNT_PER_CENTER
            or not isinstance(self.production, TargetEnsembleFeatureProduction)
            or self.production.target_id != target
            or self.production.case_bootstrap_plan_hash != self.bootstrap_plan.plan_hash
            or len(self.production.bootstrap_surfaces) != SUPPORT_BOOTSTRAP_REPLICATES
            or self.production.point_surface.feature_names != M1_PREDICTOR_NAMES
        ):
            raise ProtocolError("Target feature/bootstrap boundary drifted.")
        if self.feature_hash != canonical_sha256(self._unhashed_payload(target)):
            raise ProtocolError("Target feature surface hash drifted.")
        object.__setattr__(self, "target_id", target)

    @property
    def point_surface(self) -> EnsembleFeatureSurface:
        return self.production.point_surface

    @property
    def bootstrap_surfaces(self) -> tuple[EnsembleFeatureSurface, ...]:
        return self.production.bootstrap_surfaces

    def _unhashed_payload(self, target: str | None = None) -> dict[str, object]:
        return {
            "schema_version": "midogpp_consumed_test_target_features_v1",
            "target_id": target or self.target_id,
            "source_feature_surface_hash": self.source_feature_surface_hash,
            "support_partition_lock_hash": self.support_partition_lock_hash,
            "case_bootstrap_plan_hash": self.bootstrap_plan.plan_hash,
            "target_feature_production_hash": self.production.production_hash,
            "target_feature_seal_hash": self.target_feature_seal_hash,
            "support_case_count": len(self.bootstrap_plan.support_case_ids),
            "bootstrap_seed": self.bootstrap_plan.bootstrap_seed,
            "bootstrap_replicate_count": len(self.bootstrap_surfaces),
            "resampling_unit": "independent_whole_support_case",
            "labels_used": False,
            "utility_responses_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "feature_hash": self.feature_hash}


@dataclass(frozen=True)
class SourceInnerFeatureSurfaceSet:
    by_target: Mapping[str, SourceInnerFeatureSurfaces]
    surface_set_hash: str

    def __post_init__(self) -> None:
        values = {str(key): value for key, value in self.by_target.items()}
        if (
            tuple(values) != CENTERS
            or any(value.outer_target_id != target for target, value in values.items())
            or sum(len(value.m1.rows) for value in values.values()) != 504
        ):
            raise ProtocolError("Source-inner feature set requires complete nine-H coverage.")
        payload = _source_set_payload(values)
        if self.surface_set_hash != canonical_sha256(payload):
            raise ProtocolError("Source-inner feature-set hash drifted.")
        object.__setattr__(self, "by_target", MappingProxyType(values))

    def to_payload(self) -> dict[str, object]:
        return {**_source_set_payload(self.by_target), "surface_set_hash": self.surface_set_hash}


def build_source_inner_feature_surfaces(
    seed_feature_rows: Sequence[CandidateFeatureRow],
    *,
    support_action_shift_by_candidate: Mapping[
        tuple[str, str, str], SupportActionProbabilityShift
    ],
    outer_target_id: object,
    feature_input_seal_hash: str,
    permutation_seed: int = PERMUTATION_SEED,
) -> SourceInnerFeatureSurfaces:
    """Build M0/M1/P for one outer H without accepting a response or label."""

    target = str(outer_target_id)
    sources = candidate_sources(target)
    rows = tuple(seed_feature_rows)
    if permutation_seed != PERMUTATION_SEED:
        raise ProtocolError("Endpoint-router permutation seed cannot be tuned.")
    if (
        len(rows) != TARGET_CANDIDATE_COUNT * INNER_CANDIDATE_COUNT * SEED_PAIR_COUNT
        or not _text(feature_input_seal_hash)
        or any(
            not isinstance(row, CandidateFeatureRow)
            or row.role != INNER_ROLE
            or row.outer_target_id != target
            or row.query_id not in sources
            or row.candidate_source in {target, row.query_id}
            or row.support_case_count != SUPPORT_CASE_COUNT_PER_CENTER
            for row in rows
        )
    ):
        raise ProtocolError("Source-inner seed feature geometry drifted.")
    global_control = derive_label_free_global_source_control(rows)
    m0_rows = aggregate_candidate_seed_features(rows)
    m1_rows = aggregate_candidate_seed_features(
        rows,
        support_action_shift_by_candidate=support_action_shift_by_candidate,
    )
    kwargs = {
        "global_source_control_by_source": global_control.value_by_source,
        "global_source_control_semantics": global_control.semantics,
        "global_source_control_provenance_hash": global_control.provenance_hash,
    }
    m0 = build_ensemble_feature_surface(m0_rows, **kwargs)
    m1 = build_ensemble_feature_surface(m1_rows, **kwargs)
    permutation = cyclically_permute_target_scalar(
        m1, permutation_seed=PERMUTATION_SEED
    )
    provisional = {
        "schema_version": "midogpp_consumed_test_source_inner_features_v1",
        "outer_target_id": target,
        "global_source_control_hash": global_control.provenance_hash,
        "m0_surface_hash": m0.surface_hash,
        "m1_surface_hash": m1.surface_hash,
        "permutation_surface_hash": permutation.surface_hash,
        "feature_input_seal_hash": feature_input_seal_hash,
        "candidate_response_count": len(m1.rows),
        "predictor_names": {
            "M0": list(m0.feature_names),
            "M1": list(m1.feature_names),
            "P": list(permutation.feature_names),
        },
        "permutation_seed": PERMUTATION_SEED,
        "metadata_control_provenance": "experiment_manifest_only",
        "metadata_identity_or_labels_used_as_predictors": False,
        "technical_seed_rows_are_independent_observations": False,
        "support_labels_used": False,
        "same_outer_H_evaluation_rows_used": False,
    }
    return SourceInnerFeatureSurfaces(
        outer_target_id=target,
        global_source_control=global_control,
        m0=m0,
        m1=m1,
        permutation=permutation,
        feature_input_seal_hash=feature_input_seal_hash,
        surface_hash=canonical_sha256(provisional),
    )


def build_target_case_bootstrap_plan(
    *, target_id: object, support_case_ids: Sequence[object]
) -> CaseBootstrapPlan:
    """Build the frozen exact-eight, exact-32 neutral bootstrap plan."""

    case_ids = tuple(str(value) for value in support_case_ids)
    if len(case_ids) != SUPPORT_CASE_COUNT_PER_CENTER:
        raise ProtocolError("Target bootstrap requires exactly eight support cases.")
    return build_case_bootstrap_plan(
        target_id=target_id,
        support_case_ids=case_ids,
        bootstrap_seed=SUPPORT_BOOTSTRAP_SEED,
        replicate_count=SUPPORT_BOOTSTRAP_REPLICATES,
    )


def build_target_feature_production(
    target_seed_feature_rows: Sequence[CandidateFeatureRow],
    target_case_shift_rows: Sequence[TargetSupportActionShiftCase],
    *,
    source_features: SourceInnerFeatureSurfaces,
    case_bootstrap_plan: CaseBootstrapPlan,
    support_partition_lock_hash: str,
    target_feature_seal_hash: str,
) -> TargetFeatureSurfaces:
    """Build target point and bootstrap features with no label-bearing input."""

    if not isinstance(source_features, SourceInnerFeatureSurfaces):
        raise ProtocolError("Target features require the typed source feature surface.")
    target = source_features.outer_target_id
    if (
        not isinstance(case_bootstrap_plan, CaseBootstrapPlan)
        or case_bootstrap_plan.target_id != target
        or case_bootstrap_plan.bootstrap_seed != SUPPORT_BOOTSTRAP_SEED
        or case_bootstrap_plan.replicate_count != SUPPORT_BOOTSTRAP_REPLICATES
        or len(case_bootstrap_plan.support_case_ids) != SUPPORT_CASE_COUNT_PER_CENTER
        or not _text(support_partition_lock_hash)
        or not _text(target_feature_seal_hash)
    ):
        raise ProtocolError("Target feature/bootstrap binding drifted.")
    rows = tuple(target_seed_feature_rows)
    if (
        len(rows) != TARGET_CANDIDATE_COUNT * SEED_PAIR_COUNT
        or any(
            not isinstance(row, CandidateFeatureRow)
            or row.role != TARGET_ROLE
            or row.outer_target_id != target
            or row.query_id != target
            or row.candidate_source not in candidate_sources(target)
            or row.support_case_count != SUPPORT_CASE_COUNT_PER_CENTER
            or row.support_partition_hash != case_bootstrap_plan.support_partition_hash
            for row in rows
        )
    ):
        raise ProtocolError("Target seed feature geometry drifted.")
    production = build_target_ensemble_feature_surfaces(
        rows,
        tuple(target_case_shift_rows),
        case_bootstrap_plan,
        global_source_control=source_features.global_source_control,
    )
    payload = {
        "schema_version": "midogpp_consumed_test_target_features_v1",
        "target_id": target,
        "source_feature_surface_hash": source_features.surface_hash,
        "support_partition_lock_hash": support_partition_lock_hash,
        "case_bootstrap_plan_hash": case_bootstrap_plan.plan_hash,
        "target_feature_production_hash": production.production_hash,
        "target_feature_seal_hash": target_feature_seal_hash,
        "support_case_count": len(case_bootstrap_plan.support_case_ids),
        "bootstrap_seed": case_bootstrap_plan.bootstrap_seed,
        "bootstrap_replicate_count": len(production.bootstrap_surfaces),
        "resampling_unit": "independent_whole_support_case",
        "labels_used": False,
        "utility_responses_used": False,
    }
    return TargetFeatureSurfaces(
        target_id=target,
        source_feature_surface_hash=source_features.surface_hash,
        support_partition_lock_hash=support_partition_lock_hash,
        bootstrap_plan=case_bootstrap_plan,
        production=production,
        target_feature_seal_hash=target_feature_seal_hash,
        feature_hash=canonical_sha256(payload),
    )


def build_source_inner_feature_surface_set(
    values: Mapping[str, SourceInnerFeatureSurfaces],
) -> SourceInnerFeatureSurfaceSet:
    normalized = {center: values[center] for center in CENTERS}
    payload = _source_set_payload(normalized)
    return SourceInnerFeatureSurfaceSet(
        by_target=normalized,
        surface_set_hash=canonical_sha256(payload),
    )


def _source_set_payload(
    values: Mapping[str, SourceInnerFeatureSurfaces],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_consumed_test_source_inner_feature_set_v1",
        "centers": list(CENTERS),
        "surface_hashes_by_target": {
            target: values[target].surface_hash for target in CENTERS
        },
        "candidate_response_count": sum(len(values[target].m1.rows) for target in CENTERS),
        "labels_used": False,
        "support_labels_used": False,
    }


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value.strip() == value


__all__ = (
    "SourceInnerFeatureSurfaceSet",
    "SourceInnerFeatureSurfaces",
    "TargetFeatureSurfaces",
    "build_source_inner_feature_surface_set",
    "build_source_inner_feature_surfaces",
    "build_target_case_bootstrap_plan",
    "build_target_feature_production",
)
