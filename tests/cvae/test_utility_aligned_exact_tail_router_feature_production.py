from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    candidate_sources,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.feature_production import (
    produce_label_free_features,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.input_contracts import (
    FixedPartitionSurface,
    LabelFreeValidationFrame,
    ValidationRowIdentity,
    row_identity_hash,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.source_cache_contracts import (
    SourceCache,
)


class _FeatureCache(SourceCache):
    @property
    def component_by_key(self):
        return self._components

    def component_arrays(self, *, query_center, source_center, training_seed):
        query = CENTERS.index(str(query_center))
        source = CENTERS.index(str(source_center))
        seed = TRAINING_SEEDS.index(int(training_seed))
        base = 0.1 + 0.01 * query + 0.002 * source + 0.0001 * seed
        reconstruction = {
            0: np.asarray([base, base + 0.02], dtype=np.float32),
            1: np.asarray([base + 0.01, base + 0.03], dtype=np.float32),
        }
        kl = {
            0: np.asarray([base / 2, base / 2 + 0.01], dtype=np.float32),
            1: np.asarray([base / 2 + 0.005, base / 2 + 0.015], dtype=np.float32),
        }
        return reconstruction, kl


def test_feature_production_is_exact_label_free_and_binds_execution_actions() -> None:
    frame, partitions = _frame_and_partitions()
    cache = object.__new__(_FeatureCache)
    components = {}
    ordinal = 0
    for query in CENTERS:
        support_hash = row_identity_hash(partitions.support_rows_by_center[query])
        for source in candidate_sources(query):
            for training_seed in TRAINING_SEEDS:
                components[(query, source, training_seed)] = SimpleNamespace(
                    component_ordinal=ordinal,
                    support_partition_hash=support_hash,
                    support_case_count=2,
                    case_equal_energy=(
                        0.2
                        + 0.01 * CENTERS.index(source)
                        + 0.001 * TRAINING_SEEDS.index(training_seed)
                    ),
                    linear_kernel_mmd2_by_generation_seed={
                        seed: 0.3 + 0.001 * index
                        for index, seed in enumerate(GENERATION_SEEDS)
                    },
                )
                ordinal += 1
    object.__setattr__(cache, "_components", components)
    metadata = {
        query: {
            source: 1.0 - abs(CENTERS.index(query) - CENTERS.index(source)) / 9.0
            for source in candidate_sources(query)
        }
        for query in CENTERS
    }

    first = produce_label_free_features(cache, frame, partitions, metadata)
    second = produce_label_free_features(cache, frame, partitions, metadata)

    assert len(first.inner_rows) == 4_536
    assert len(first.target_rows) == 648
    assert len(first.development_action_bindings) == 576
    assert first.production_hash == second.production_hash
    assert first.surfaces.surface_hash == second.surfaces.surface_hash
    assert all(row.support_case_count == 2 for row in (*first.inner_rows, *first.target_rows))
    assert all(row.outer_target_id != row.query_id for row in first.inner_rows)
    assert all(
        row.candidate_source not in {row.outer_target_id, row.query_id}
        for row in first.inner_rows
    )
    assert all(row.outer_target_id == row.query_id for row in first.target_rows)
    assert all(row.candidate_source != row.outer_target_id for row in first.target_rows)
    assert all(
        binding["execution_action_id"] == binding["canonical_action_id"]
        and binding["execution_action_hash"] == binding["canonical_action_hash"]
        and binding["geometry_equal"] is True
        and binding["labels_used"] is False
        for binding in first.development_action_bindings
    )
    assert first.to_payload()["labels_used"] is False
    assert first.to_payload()["evaluation_embeddings_used"] is False
    assert first.to_payload()["routing_status"] == "INSUFFICIENT_SUPPORT_FOR_POLICY"


def _frame_and_partitions() -> tuple[LabelFreeValidationFrame, FixedPartitionSurface]:
    unassigned = []
    by_center = {}
    support = {}
    evaluation = {}
    ordinal = 0
    for center in CENTERS:
        center_unassigned = []
        center_support = []
        for local in range(2):
            raw = ValidationRowIdentity(
                row_ordinal=ordinal,
                manifest_row_index=ordinal,
                sample_id=f"sample::{center}::{local}",
                case_id=f"case::{center}::{local}",
                center=center,
            )
            center_unassigned.append(raw)
            unassigned.append(raw)
            center_support.append(
                ValidationRowIdentity(
                    row_ordinal=ordinal,
                    manifest_row_index=ordinal,
                    sample_id=raw.sample_id,
                    case_id=raw.case_id,
                    center=center,
                    partition_role="support",
                )
            )
            ordinal += 1
        by_center[center] = tuple(center_unassigned)
        support[center] = tuple(center_support)
        evaluation[center] = ()
    frame = LabelFreeValidationFrame(
        embeddings=np.zeros((len(unassigned), 3_840), dtype=np.float32),
        rows=tuple(unassigned),
        rows_by_center=by_center,
        cache_binding={"schema_version": "test"},
    )
    partitions = FixedPartitionSurface(
        support_rows_by_center=support,
        evaluation_rows_by_center=evaluation,
        table_rows=(),
        lock_payload={"support_partition_lock_hash": "support-lock"},
    )
    return frame, partitions
