"""Label-free seed-feature production adapter.

The shared leaf computes CVAE component summaries from this experiment's own
source cache.  Its older aggregate surfaces are discarded; all model-facing
surfaces are rebuilt by the ensemble-endpoint core.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...routing.residual_topup.hashing import canonical_sha256
from ...routing.utility_aligned.surface_contracts import CandidateFeatureRow
from ..utility_aligned_exact_tail_router.feature_production import (
    produce_label_free_features as _produce_seed_features,
)


@dataclass(frozen=True)
class EnsembleSeedFeatureProduction:
    inner_rows: tuple[CandidateFeatureRow, ...]
    target_rows: tuple[CandidateFeatureRow, ...]
    production_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage90_ensemble_endpoint_seed_feature_production_v1",
            "inner_row_count": len(self.inner_rows),
            "target_row_count": len(self.target_rows),
            "inner_row_hashes": [row.row_hash for row in self.inner_rows],
            "target_row_hashes": [row.row_hash for row in self.target_rows],
            "labels_used": False,
            "evaluation_embeddings_used": False,
            "production_hash": self.production_hash,
        }


def produce_label_free_seed_features(
    source_cache: object, frame: object, partitions: object, metadata_similarity: object
) -> EnsembleSeedFeatureProduction:
    shared = _produce_seed_features(source_cache, frame, partitions, metadata_similarity)
    unhashed = {
        "schema_version": "midogpp_stage90_ensemble_endpoint_seed_feature_production_v1",
        "inner_row_count": len(shared.inner_rows),
        "target_row_count": len(shared.target_rows),
        "inner_row_hashes": [row.row_hash for row in shared.inner_rows],
        "target_row_hashes": [row.row_hash for row in shared.target_rows],
        "labels_used": False,
        "evaluation_embeddings_used": False,
    }
    return EnsembleSeedFeatureProduction(
        inner_rows=shared.inner_rows,
        target_rows=shared.target_rows,
        production_hash=canonical_sha256(unhashed),
    )


__all__ = ("EnsembleSeedFeatureProduction", "produce_label_free_seed_features")
