"""Label-free source controls and whole-case target feature production."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256
from .ensemble_endpoint_contracts import (
    ENSEMBLE_SEED_KEYS,
    SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
    SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
    SeedProbabilityVector,
)
from .ensemble_feature_contracts import (
    GLOBAL_SOURCE_CONTROL_SEMANTICS,
    EnsembleCandidateFeatureRow,
    GlobalSourceControl,
    TargetEnsembleFeatureProduction,
    TargetSupportActionShiftCase,
)
from .ensemble_endpoint import support_action_probability_shift
from .ensemble_features import (
    aggregate_candidate_seed_features,
    build_ensemble_feature_surface,
)
from .row_contracts import (
    INNER_CANDIDATE_COUNT,
    INNER_ROLE,
    TARGET_CANDIDATE_COUNT,
    TARGET_ROLE,
    CaseBootstrapPlan,
)
from .surface_contracts import CandidateFeatureRow


def derive_label_free_global_source_control(
    rows: Sequence[CandidateFeatureRow],
) -> GlobalSourceControl:
    """Derive M0 only from locked source-inner metadata similarity.

    For each H-excluded candidate source, metadata similarity is averaged over
    its seven legal pseudoqueries and all nine paired seed cells.  No response,
    downstream label, bank performance, or utility object is accepted.
    """

    seed_rows = tuple(rows)
    if (
        len(seed_rows)
        != TARGET_CANDIDATE_COUNT
        * INNER_CANDIDATE_COUNT
        * len(ENSEMBLE_SEED_KEYS)
        or any(not isinstance(row, CandidateFeatureRow) for row in seed_rows)
        or {row.role for row in seed_rows} != {INNER_ROLE}
    ):
        raise ProtocolError("Global source control requires one complete source-inner surface.")
    outer_ids = {row.outer_target_id for row in seed_rows}
    if len(outer_ids) != 1:
        raise ProtocolError("Global source control requires exactly one outer H.")
    outer = next(iter(outer_ids))
    # Reuse the exact-nine candidate validator; its output is deliberately not
    # used as a response or as nine independent observations.
    aggregated = aggregate_candidate_seed_features(seed_rows)
    sources = tuple(sorted({row.candidate_source for row in aggregated}))
    if len(sources) != TARGET_CANDIDATE_COUNT or outer in sources:
        raise ProtocolError("Global source control candidate universe drifted.")
    values: dict[str, float] = {}
    for source in sources:
        source_rows = tuple(row for row in seed_rows if row.candidate_source == source)
        if (
            len(source_rows) != INNER_CANDIDATE_COUNT * len(ENSEMBLE_SEED_KEYS)
            or len({row.query_id for row in source_rows}) != INNER_CANDIDATE_COUNT
        ):
            raise ProtocolError("Global source control pseudoquery coverage drifted.")
        values[source] = float(
            np.mean(
                np.asarray(
                    [row.metadata_similarity for row in source_rows], dtype=np.float64
                ),
                dtype=np.float64,
            )
        )
    ordered = tuple(sorted(seed_rows, key=lambda row: row.row_key))
    input_hash = canonical_sha256([row.row_hash for row in ordered])
    unhashed = {
        "schema_version": "midogpp_utility_aligned_global_source_control_v1",
        "outer_target_id": outer,
        "value_by_source": values,
        "source_inner_seed_row_count": len(ordered),
        "input_row_hashes_hash": input_hash,
        "semantics": GLOBAL_SOURCE_CONTROL_SEMANTICS,
        "labels_used": False,
        "utility_responses_used": False,
    }
    return GlobalSourceControl(
        outer_target_id=outer,
        value_by_source=values,
        source_inner_seed_row_count=len(ordered),
        input_row_hashes_hash=input_hash,
        provenance_hash=canonical_sha256(unhashed),
    )


def build_target_support_action_shift_case(
    *,
    target_id: str,
    candidate_source: str,
    case_id: str,
    base_vectors: Sequence[SeedProbabilityVector],
    tail_vectors: Sequence[SeedProbabilityVector],
) -> TargetSupportActionShiftCase:
    """Seal one independent case's label-free exact-nine probability shift."""

    shift = support_action_probability_shift(base_vectors, tail_vectors)
    return TargetSupportActionShiftCase(
        target_id=target_id,
        candidate_source=candidate_source,
        case_id=case_id,
        support_row_identity_hash=shift.row_identity_hash,
        support_row_count=len(base_vectors[0].positive_class_probabilities),
        seed_keys=shift.seed_keys,
        per_seed_mean_absolute_shifts=shift.per_seed_mean_absolute_shifts,
        base_component_vector_hashes=shift.base_component_vector_hashes,
        tail_component_vector_hashes=shift.tail_component_vector_hashes,
        ensemble_mean_absolute_shift=shift.value,
        base_ensemble_probability_hash=shift.base_ensemble_probability_hash,
        tail_ensemble_probability_hash=shift.tail_ensemble_probability_hash,
        ensemble_absolute_difference_hash=(
            shift.ensemble_absolute_difference_hash
        ),
    )


def build_target_ensemble_feature_surfaces(
    rows: Sequence[CandidateFeatureRow],
    case_shift_rows: Sequence[TargetSupportActionShiftCase],
    case_bootstrap_plan: CaseBootstrapPlan,
    *,
    global_source_control: GlobalSourceControl,
) -> TargetEnsembleFeatureProduction:
    """Build target M1 point + whole-case bootstrap surfaces.

    Each bootstrap scalar is recomputed from the plan's sampled case IDs with
    multiplicity.  The global source control remains the same locked M0 value;
    seed spread stays diagnostic and no utility response is accepted.
    """

    if not isinstance(case_bootstrap_plan, CaseBootstrapPlan):
        raise ProtocolError("Target ensemble features require a typed case bootstrap plan.")
    if not isinstance(global_source_control, GlobalSourceControl):
        raise ProtocolError("Target ensemble features require a typed global control.")
    seed_rows = tuple(rows)
    expected_seed_row_count = TARGET_CANDIDATE_COUNT * len(ENSEMBLE_SEED_KEYS)
    if (
        len(seed_rows) != expected_seed_row_count
        or any(not isinstance(row, CandidateFeatureRow) for row in seed_rows)
        or {row.role for row in seed_rows} != {TARGET_ROLE}
    ):
        raise ProtocolError("Target ensemble point features require eight exact-nine candidates.")
    targets = {row.outer_target_id for row in seed_rows}
    if len(targets) != 1:
        raise ProtocolError("Target ensemble point features require one target.")
    target = next(iter(targets))
    if (
        case_bootstrap_plan.target_id != target
        or global_source_control.outer_target_id != target
        or {row.query_id for row in seed_rows} != {target}
    ):
        raise ProtocolError("Target feature/bootstrap/global-control binding drifted.")
    if {row.support_partition_hash for row in seed_rows} != {
        case_bootstrap_plan.support_partition_hash
    }:
        raise ProtocolError("Target point feature support partition does not match the plan.")
    if {row.support_case_count for row in seed_rows} != {
        len(case_bootstrap_plan.support_case_ids)
    }:
        raise ProtocolError("Target point feature case count does not match the plan.")
    base_rows = aggregate_candidate_seed_features(seed_rows)
    sources = tuple(sorted(row.candidate_source for row in base_rows))
    if set(sources) != set(global_source_control.value_by_source):
        raise ProtocolError("Target/global-control candidate universes drifted.")

    shifts = tuple(case_shift_rows)
    if not shifts or any(not isinstance(row, TargetSupportActionShiftCase) for row in shifts):
        raise ProtocolError("Target per-case shifts require typed nonempty rows.")
    by_source_case: dict[tuple[str, str], TargetSupportActionShiftCase] = {}
    for row in shifts:
        if row.target_id != target or row.candidate_source not in sources:
            raise ProtocolError("Target per-case shift domain binding drifted.")
        key = (row.candidate_source, row.case_id)
        if key in by_source_case:
            raise ProtocolError("Target per-case shifts contain duplicate source/case rows.")
        by_source_case[key] = row
    expected_case_ids = set(case_bootstrap_plan.support_case_ids)
    if set(by_source_case) != {
        (source, case_id) for source in sources for case_id in expected_case_ids
    }:
        raise ProtocolError("Every target candidate requires every independent support case.")
    per_case_surface_hash = canonical_sha256(
        {
            "schema_version": "midogpp_utility_aligned_target_support_shift_surface_v1",
            "target_id": target,
            "candidate_sources": list(sources),
            "support_case_ids": list(case_bootstrap_plan.support_case_ids),
            "case_hashes": [
                by_source_case[(source, case_id)].case_hash
                for source in sources
                for case_id in case_bootstrap_plan.support_case_ids
            ],
            "labels_used": False,
        }
    )
    point_rows = _attach_case_resampled_scalar(
        base_rows,
        by_source_case,
        sampled_case_ids=case_bootstrap_plan.support_case_ids,
        support_partition_hash=case_bootstrap_plan.support_partition_hash,
        scalar_parent_hash=per_case_surface_hash,
        replicate_hash=None,
    )
    point_surface = build_ensemble_feature_surface(
        point_rows,
        global_source_control_by_source=global_source_control.value_by_source,
        global_source_control_semantics=global_source_control.semantics,
        global_source_control_provenance_hash=global_source_control.provenance_hash,
    )
    bootstrap_surfaces = tuple(
        build_ensemble_feature_surface(
            _attach_case_resampled_scalar(
                base_rows,
                by_source_case,
                sampled_case_ids=replicate.sampled_case_ids,
                support_partition_hash=replicate.support_partition_hash,
                scalar_parent_hash=per_case_surface_hash,
                replicate_hash=replicate.replicate_hash,
            ),
            global_source_control_by_source=global_source_control.value_by_source,
            global_source_control_semantics=global_source_control.semantics,
            global_source_control_provenance_hash=global_source_control.provenance_hash,
        )
        for replicate in case_bootstrap_plan.replicates
    )
    if len({surface.surface_hash for surface in bootstrap_surfaces}) != len(
        bootstrap_surfaces
    ):
        raise ProtocolError("Whole-case target bootstrap surfaces are not uniquely bound.")
    unhashed = {
        "schema_version": "midogpp_utility_aligned_target_ensemble_feature_production_v1",
        "target_id": target,
        "global_source_control_provenance_hash": global_source_control.provenance_hash,
        "case_bootstrap_plan_hash": case_bootstrap_plan.plan_hash,
        "per_case_shift_surface_hash": per_case_surface_hash,
        "point_surface_hash": point_surface.surface_hash,
        "bootstrap_surface_hashes": [
            surface.surface_hash for surface in bootstrap_surfaces
        ],
        "bootstrap_replicate_count": len(bootstrap_surfaces),
        "resampling_unit": "independent_target_support_case",
        "labels_used": False,
        "utility_responses_used": False,
    }
    return TargetEnsembleFeatureProduction(
        target_id=target,
        global_source_control=global_source_control,
        case_bootstrap_plan_hash=case_bootstrap_plan.plan_hash,
        per_case_shift_surface_hash=per_case_surface_hash,
        point_surface=point_surface,
        bootstrap_surfaces=bootstrap_surfaces,
        production_hash=canonical_sha256(unhashed),
    )


def _attach_case_resampled_scalar(
    base_rows: Sequence[EnsembleCandidateFeatureRow],
    by_source_case: dict[tuple[str, str], TargetSupportActionShiftCase],
    *,
    sampled_case_ids: tuple[str, ...],
    support_partition_hash: str,
    scalar_parent_hash: str,
    replicate_hash: str | None,
) -> tuple[EnsembleCandidateFeatureRow, ...]:
    output: list[EnsembleCandidateFeatureRow] = []
    for row in base_rows:
        selected = tuple(
            by_source_case[(row.candidate_source, case_id)]
            for case_id in sampled_case_ids
        )
        per_case_seed = np.asarray(
            [item.per_seed_mean_absolute_shifts for item in selected],
            dtype=np.float64,
        )
        row_counts = np.asarray(
            [item.support_row_count for item in selected], dtype=np.float64
        )
        seed_means = np.average(per_case_seed, axis=0, weights=row_counts)
        value = float(
            np.average(
                np.asarray(
                    [item.ensemble_mean_absolute_shift for item in selected],
                    dtype=np.float64,
                ),
                weights=row_counts,
            )
        )
        seed_sd = float(np.std(seed_means, ddof=0, dtype=np.float64))
        scalar_hash = canonical_sha256(
            {
                "schema_version": "midogpp_utility_aligned_resampled_target_scalar_v2",
                "candidate_key": list(row.row_key),
                "scalar_parent_hash": scalar_parent_hash,
                "replicate_hash": replicate_hash,
                "sampled_case_ids": list(sampled_case_ids),
                "sampled_case_hashes": [item.case_hash for item in selected],
                "sampled_case_row_counts": [
                    item.support_row_count for item in selected
                ],
                "sampled_case_ensemble_shifts": [
                    item.ensemble_mean_absolute_shift for item in selected
                ],
                "per_seed_means": seed_means.tolist(),
                "value": value,
                "seed_standard_deviation": seed_sd,
                "technical_seed_values_may_feed_model": False,
                "labels_used": False,
            }
        )
        output.append(
            EnsembleCandidateFeatureRow(
                role=row.role,
                outer_target_id=row.outer_target_id,
                query_id=row.query_id,
                candidate_source=row.candidate_source,
                candidate_source_count=row.candidate_source_count,
                support_partition_hash=support_partition_hash,
                support_case_count=len(sampled_case_ids),
                seed_row_hashes=row.seed_row_hashes,
                feature_mean_by_name=row.feature_mean_by_name,
                feature_seed_standard_deviation_by_name=(
                    row.feature_seed_standard_deviation_by_name
                ),
                target_local_scalar=value,
                target_local_scalar_name=SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
                target_local_scalar_semantics=(
                    SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS
                ),
                target_local_scalar_seed_standard_deviation=seed_sd,
                target_local_scalar_provenance_hash=scalar_hash,
            )
        )
    return tuple(output)


__all__ = (
    "build_target_ensemble_feature_surfaces",
    "build_target_support_action_shift_case",
    "derive_label_free_global_source_control",
)
