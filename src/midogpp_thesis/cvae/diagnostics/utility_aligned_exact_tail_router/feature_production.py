"""Label-free two-case feature production from the experiment-local cache."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from ...routing.utility_aligned import (
    INNER_ROLE,
    TARGET_ROLE,
    CandidateFeatureRow,
)
from .actions import (
    build_inner_exact_tail_action_library,
    build_inner_exact_tail_actions,
)
from .contracts import (
    CENTERS,
    EXPECTED_INNER_UTILITY_ROW_COUNT,
    EXPECTED_TARGET_FEATURE_ROW_COUNT,
    FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    candidate_sources,
    inner_candidate_sources,
)
from .development_prediction_contracts import action_library_for
from .features import Stage90FeatureSurfaceSet, build_stage90_feature_surface_set
from .input_contracts import FixedPartitionSurface, LabelFreeValidationFrame, row_identity_hash
from .source_cache_contracts import LabelFreeComponentRecord, SourceCache


ACTION_BINDING_COLUMNS = (
    "schema_version",
    "outer_target",
    "query_center",
    "execution_action_id",
    "execution_action_hash",
    "canonical_action_id",
    "canonical_action_hash",
    "geometry_equal",
    "labels_used",
)


@dataclass(frozen=True)
class Stage90FeatureProduction:
    inner_rows: tuple[CandidateFeatureRow, ...]
    target_rows: tuple[CandidateFeatureRow, ...]
    surfaces: Stage90FeatureSurfaceSet
    development_action_bindings: tuple[Mapping[str, object], ...]
    canonical_inner_action_library_hash: str
    action_binding_hash: str
    production_hash: str

    def __post_init__(self) -> None:
        bindings = tuple(MappingProxyType(dict(row)) for row in self.development_action_bindings)
        if (
            len(self.inner_rows) != EXPECTED_INNER_UTILITY_ROW_COUNT
            or len(self.target_rows) != EXPECTED_TARGET_FEATURE_ROW_COUNT
            or not isinstance(self.surfaces, Stage90FeatureSurfaceSet)
            or len(bindings) != len(CENTERS) * 8 * 8
            or any(set(row) != set(ACTION_BINDING_COLUMNS) for row in bindings)
            or self.action_binding_hash
            != canonical_sha256([dict(row) for row in bindings])
        ):
            raise ProtocolError("Stage-90 feature production coverage drifted.")
        expected = canonical_sha256(self._unhashed_payload(bindings=bindings))
        if self.production_hash != expected:
            raise ProtocolError("Stage-90 feature-production hash drifted.")
        object.__setattr__(self, "development_action_bindings", bindings)

    def _unhashed_payload(
        self, *, bindings: Sequence[Mapping[str, object]] | None = None
    ) -> dict[str, object]:
        values = self.development_action_bindings if bindings is None else bindings
        return {
            "schema_version": "midogpp_utility_aligned_stage90_feature_production_v1",
            "feature_surface_set_hash": self.surfaces.surface_hash,
            "inner_row_hashes": [row.row_hash for row in self.inner_rows],
            "target_row_hashes": [row.row_hash for row in self.target_rows],
            "inner_row_count": len(self.inner_rows),
            "target_row_count": len(self.target_rows),
            "canonical_inner_action_library_hash": self.canonical_inner_action_library_hash,
            "action_binding_hash": self.action_binding_hash,
            "action_binding_count": len(values),
            "support_case_count": FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
            "labels_used": False,
            "evaluation_embeddings_used": False,
            "all_nine_seed_pairs_retained": True,
            "strict_H_q_e_exclusion": True,
            "routing_status": "INSUFFICIENT_SUPPORT_FOR_POLICY",
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "production_hash": self.production_hash}


def produce_label_free_features(
    source_cache: SourceCache,
    frame: LabelFreeValidationFrame,
    partitions: FixedPartitionSurface,
    metadata_similarity: Mapping[str, Mapping[str, float]],
) -> Stage90FeatureProduction:
    """Produce the exact 4,536+648 feature grid without a label parameter."""

    if not isinstance(source_cache, SourceCache) or not isinstance(
        frame, LabelFreeValidationFrame
    ) or not isinstance(partitions, FixedPartitionSurface):
        raise ProtocolError("Stage-90 feature production requires typed label-free inputs.")
    metadata = {
        str(query): {str(source): float(value) for source, value in rows.items()}
        for query, rows in metadata_similarity.items()
    }
    if tuple(metadata) != CENTERS or any(
        tuple(metadata[query]) != candidate_sources(query) for query in CENTERS
    ):
        raise ProtocolError("Stage-90 metadata-similarity grid drifted.")
    case_ids_by_query: dict[str, tuple[str, ...]] = {}
    support_hash_by_query: dict[str, str] = {}
    for query in CENTERS:
        support_rows = partitions.support_rows_by_center[query]
        if (
            len({row.case_id for row in support_rows})
            != FIXED_SUPPORT_CASE_COUNT_PER_CENTER
            or any(row.partition_role != "support" for row in support_rows)
        ):
            raise ProtocolError("Stage-90 feature support is not exactly two cases.")
        # This also proves that the requested rows belong to the admitted frame.
        frame.embeddings_for(support_rows)
        case_ids_by_query[query] = tuple(row.case_id for row in support_rows)
        support_hash_by_query[query] = row_identity_hash(support_rows)

    component_rows: dict[tuple[str, str, int, int], CandidateFeatureRow] = {}
    for query in CENTERS:
        for source in candidate_sources(query):
            records = {
                seed: source_cache.component_by_key[(query, source, seed)]
                for seed in TRAINING_SEEDS
            }
            if any(
                record.support_partition_hash != support_hash_by_query[query]
                or record.support_case_count != FIXED_SUPPORT_CASE_COUNT_PER_CENTER
                for record in records.values()
            ):
                raise ProtocolError("Stage-90 feature/cache support binding drifted.")
            replica_energies = np.asarray(
                [records[seed].case_equal_energy for seed in TRAINING_SEEDS],
                dtype=np.float64,
            )
            disagreement = float(np.std(replica_energies, ddof=0))
            for training_seed in TRAINING_SEEDS:
                record = records[training_seed]
                reconstruction, kl = source_cache.component_arrays(
                    query_center=query,
                    source_center=source,
                    training_seed=training_seed,
                )
                reconstruction_stats = _case_equal_stats(
                    reconstruction,
                    case_ids_by_query[query],
                    role="reconstruction",
                )
                kl_stats = _case_equal_stats(
                    kl,
                    case_ids_by_query[query],
                    role="KL",
                )
                for generation_seed in GENERATION_SEEDS:
                    component_rows[(query, source, training_seed, generation_seed)] = (
                        _candidate_row(
                            role=TARGET_ROLE,
                            outer=query,
                            query=query,
                            source=source,
                            training_seed=training_seed,
                            generation_seed=generation_seed,
                            support_hash=record.support_partition_hash,
                            reconstruction_stats=reconstruction_stats,
                            kl_stats=kl_stats,
                            disagreement=disagreement,
                            mmd=record.linear_kernel_mmd2_by_generation_seed[
                                generation_seed
                            ],
                            metadata_similarity=metadata[query][source],
                        )
                    )

    target_rows = tuple(
        component_rows[(target, source, training_seed, generation_seed)]
        for target in CENTERS
        for source in candidate_sources(target)
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )
    inner_rows = tuple(
        _as_inner(component_rows[(query, source, training_seed, generation_seed)], outer)
        for outer in CENTERS
        for query in candidate_sources(outer)
        for source in inner_candidate_sources(outer, query)
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )
    if (
        len(inner_rows) != EXPECTED_INNER_UTILITY_ROW_COUNT
        or len(target_rows) != EXPECTED_TARGET_FEATURE_ROW_COUNT
    ):
        raise ProtocolError("Stage-90 produced feature row count drifted.")
    surfaces = build_stage90_feature_surface_set(inner_rows, target_rows)
    bindings, canonical_hash = build_development_action_bindings()
    binding_hash = canonical_sha256([dict(row) for row in bindings])
    provisional = {
        "schema_version": "midogpp_utility_aligned_stage90_feature_production_v1",
        "feature_surface_set_hash": surfaces.surface_hash,
        "inner_row_hashes": [row.row_hash for row in inner_rows],
        "target_row_hashes": [row.row_hash for row in target_rows],
        "inner_row_count": len(inner_rows),
        "target_row_count": len(target_rows),
        "canonical_inner_action_library_hash": canonical_hash,
        "action_binding_hash": binding_hash,
        "action_binding_count": len(bindings),
        "support_case_count": FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
        "labels_used": False,
        "evaluation_embeddings_used": False,
        "all_nine_seed_pairs_retained": True,
        "strict_H_q_e_exclusion": True,
        "routing_status": "INSUFFICIENT_SUPPORT_FOR_POLICY",
    }
    return Stage90FeatureProduction(
        inner_rows=inner_rows,
        target_rows=target_rows,
        surfaces=surfaces,
        development_action_bindings=bindings,
        canonical_inner_action_library_hash=canonical_hash,
        action_binding_hash=binding_hash,
        production_hash=canonical_sha256(provisional),
    )


def build_development_action_bindings(
) -> tuple[tuple[Mapping[str, object], ...], str]:
    """Bind execution DTO hashes to canonical neutral-primitive actions."""

    canonical_library = build_inner_exact_tail_action_library()
    rows: list[Mapping[str, object]] = []
    for outer in CENTERS:
        for query in candidate_sources(outer):
            execution = action_library_for(outer_target=outer, query_center=query)
            canonical = build_inner_exact_tail_actions(outer, query)
            if len(execution) != len(canonical):
                raise ProtocolError("Development action binding cardinality drifted.")
            for execution_action, canonical_action in zip(
                execution, canonical, strict=True
            ):
                execution_counts = dict(execution_action.counts_per_class)
                canonical_counts = dict(canonical_action.final_counts_by_class[0])
                if (
                    execution_counts != canonical_counts
                    or dict(canonical_action.final_counts_by_class[1])
                    != canonical_counts
                ):
                    raise ProtocolError("Development action DTO geometry drifted from canonical.")
                rows.append(
                    MappingProxyType(
                        {
                            "schema_version": "midogpp_utility_aligned_stage90_action_binding_v1",
                            "outer_target": outer,
                            "query_center": query,
                            "execution_action_id": execution_action.action_id,
                            "execution_action_hash": execution_action.action_hash,
                            "canonical_action_id": canonical_action.action_id,
                            "canonical_action_hash": canonical_action.action_hash,
                            "geometry_equal": True,
                            "labels_used": False,
                        }
                    )
                )
    return tuple(rows), canonical_library.action_library_hash


def _candidate_row(
    *,
    role: str,
    outer: str,
    query: str,
    source: str,
    training_seed: int,
    generation_seed: int,
    support_hash: str,
    reconstruction_stats: tuple[float, float, float, float, float],
    kl_stats: tuple[float, float, float, float, float],
    disagreement: float,
    mmd: float,
    metadata_similarity: float,
) -> CandidateFeatureRow:
    return CandidateFeatureRow(
        role=role,
        outer_target_id=outer,
        query_id=query,
        candidate_source=source,
        training_seed=training_seed,
        generation_seed=generation_seed,
        candidate_source_count=8 if role == TARGET_ROLE else 7,
        support_partition_hash=support_hash,
        support_case_count=FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
        reconstruction_mean=reconstruction_stats[0],
        reconstruction_std=reconstruction_stats[1],
        reconstruction_q25=reconstruction_stats[2],
        reconstruction_q50=reconstruction_stats[3],
        reconstruction_q75=reconstruction_stats[4],
        kl_mean=kl_stats[0],
        kl_std=kl_stats[1],
        kl_q25=kl_stats[2],
        kl_q50=kl_stats[3],
        kl_q75=kl_stats[4],
        replica_disagreement=disagreement,
        distribution_mmd=float(mmd),
        metadata_similarity=float(metadata_similarity),
    )


def _as_inner(row: CandidateFeatureRow, outer: str) -> CandidateFeatureRow:
    return _candidate_row(
        role=INNER_ROLE,
        outer=outer,
        query=row.query_id,
        source=row.candidate_source,
        training_seed=row.training_seed,
        generation_seed=row.generation_seed,
        support_hash=row.support_partition_hash,
        reconstruction_stats=(
            row.reconstruction_mean,
            row.reconstruction_std,
            row.reconstruction_q25,
            row.reconstruction_q50,
            row.reconstruction_q75,
        ),
        kl_stats=(row.kl_mean, row.kl_std, row.kl_q25, row.kl_q50, row.kl_q75),
        disagreement=row.replica_disagreement,
        mmd=row.distribution_mmd,
        metadata_similarity=row.metadata_similarity,
    )


def _case_equal_stats(
    per_class: Mapping[int, np.ndarray],
    case_ids: Sequence[str],
    *,
    role: str,
) -> tuple[float, float, float, float, float]:
    if set(per_class) != {0, 1}:
        raise ProtocolError(f"Stage-90 {role} requires both class hypotheses.")
    values = 0.5 * (
        np.asarray(per_class[0], dtype=np.float64)
        + np.asarray(per_class[1], dtype=np.float64)
    )
    cases = tuple(str(value) for value in case_ids)
    if (
        values.shape != (len(cases),)
        or len(set(cases)) != FIXED_SUPPORT_CASE_COUNT_PER_CENTER
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
    ):
        raise ProtocolError(f"Stage-90 {role} component geometry drifted.")
    grouped: dict[str, list[float]] = defaultdict(list)
    for case_id, value in zip(cases, values, strict=True):
        grouped[case_id].append(float(value))
    case_values = np.asarray(
        [np.mean(grouped[case_id], dtype=np.float64) for case_id in sorted(grouped)],
        dtype=np.float64,
    )
    quantiles = np.quantile(case_values, (0.25, 0.5, 0.75))
    return (
        float(np.mean(case_values, dtype=np.float64)),
        float(np.std(case_values, ddof=0)),
        float(quantiles[0]),
        float(quantiles[1]),
        float(quantiles[2]),
    )


__all__ = (
    "ACTION_BINDING_COLUMNS",
    "Stage90FeatureProduction",
    "build_development_action_bindings",
    "produce_label_free_features",
)
