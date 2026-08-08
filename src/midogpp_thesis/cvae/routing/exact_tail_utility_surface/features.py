"""Label-free distributional features aligned to exact-tail utility cells."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ..utility_aligned import (
    INNER_ROLE,
    CandidateFeatureRow,
    build_distributional_feature_surface,
)
from .contracts import (
    CENTERS,
    EXPECTED_UTILITY_ROW_COUNT,
    MINIMUM_SUPPORT_CASE_COUNT,
    TRAINING_SEEDS,
    expected_utility_keys,
)
from .scoring import ScoredExactTailUtilityRow
from .production_inputs import PreparedDevelopmentInputs
from .source_generation import GeneratedDevelopmentCache, load_component_arrays


def build_candidate_feature_row(
    *,
    outer_target: str,
    pseudo_query: str,
    candidate_source: str,
    training_seed: int,
    generation_seed: int,
    support_partition_hash: str,
    case_ids: Sequence[str],
    per_class_reconstruction_mse: Mapping[int, Sequence[float] | np.ndarray],
    per_class_normalized_ps_kl: Mapping[int, Sequence[float] | np.ndarray],
    replica_case_equal_energies: Mapping[int, float],
    linear_kernel_mmd2: float,
    metadata_similarity: float,
) -> CandidateFeatureRow:
    """Reduce row-level label-free components using independent cases as units."""

    cases = tuple(str(value) for value in case_ids)
    unique_cases = tuple(sorted(set(cases)))
    if len(unique_cases) < MINIMUM_SUPPORT_CASE_COUNT:
        raise ProtocolError(
            "Utility-aligned features require at least eight independent support cases."
        )
    reconstruction = _class_marginal_component(
        per_class_reconstruction_mse, len(cases), "reconstruction"
    )
    kl = _class_marginal_component(
        per_class_normalized_ps_kl, len(cases), "KL"
    )
    reconstruction_cases = _case_equal_values(reconstruction, cases, unique_cases)
    kl_cases = _case_equal_values(kl, cases, unique_cases)
    if set(replica_case_equal_energies) != set(TRAINING_SEEDS):
        raise ProtocolError(
            "Utility-aligned features require exact replica seeds 17,42,101."
        )
    replicas = np.asarray(
        [float(replica_case_equal_energies[key]) for key in TRAINING_SEEDS],
        dtype=np.float64,
    )
    if len(replicas) != 3 or not np.isfinite(replicas).all():
        raise ProtocolError("Utility-aligned features require all three expert replicas.")
    recon_stats = _distribution_stats(reconstruction_cases)
    kl_stats = _distribution_stats(kl_cases)
    return CandidateFeatureRow(
        role=INNER_ROLE,
        outer_target_id=str(outer_target),
        query_id=str(pseudo_query),
        candidate_source=str(candidate_source),
        training_seed=int(training_seed),
        generation_seed=int(generation_seed),
        candidate_source_count=7,
        support_partition_hash=str(support_partition_hash),
        support_case_count=len(unique_cases),
        reconstruction_mean=recon_stats[0],
        reconstruction_std=recon_stats[1],
        reconstruction_q25=recon_stats[2],
        reconstruction_q50=recon_stats[3],
        reconstruction_q75=recon_stats[4],
        kl_mean=kl_stats[0],
        kl_std=kl_stats[1],
        kl_q25=kl_stats[2],
        kl_q50=kl_stats[3],
        kl_q75=kl_stats[4],
        replica_disagreement=float(np.std(replicas, ddof=0)),
        distribution_mmd=float(linear_kernel_mmd2),
        metadata_similarity=float(metadata_similarity),
    )


def validate_aligned_candidate_features(
    feature_rows: Sequence[CandidateFeatureRow],
    utility_rows: Sequence[ScoredExactTailUtilityRow],
) -> tuple[tuple[CandidateFeatureRow, ...], str]:
    """Require a label-free feature row for every utility key, in exact order."""

    features = tuple(sorted(feature_rows, key=lambda row: row.row_key))
    utilities = tuple(
        sorted(
            utility_rows,
            key=lambda row: (
                row.outer_target,
                row.pseudo_query,
                row.candidate_source,
                row.training_seed,
                row.generation_seed,
            ),
        )
    )
    expected = tuple(sorted(expected_utility_keys()))
    feature_keys = tuple(row.row_key for row in features)
    utility_keys = tuple(
        (
            row.outer_target,
            row.pseudo_query,
            row.candidate_source,
            row.training_seed,
            row.generation_seed,
        )
        for row in utilities
    )
    if (
        len(features) != EXPECTED_UTILITY_ROW_COUNT
        or feature_keys != expected
        or utility_keys != expected
        or feature_keys != utility_keys
    ):
        raise ProtocolError("Candidate features and utility rows are not key-aligned.")
    if any(row.support_case_count < MINIMUM_SUPPORT_CASE_COUNT for row in features):
        raise ProtocolError("Candidate feature support is below the eight-case floor.")
    surfaces = []
    for outer in CENTERS:
        outer_rows = tuple(row for row in features if row.outer_target_id == outer)
        surfaces.append(build_distributional_feature_surface(outer_rows))
    feature_hash = stable_hash(
        {
            "schema_version": "midogpp_exact_tail_candidate_features_v1",
            "row_hashes": [row.row_hash for row in features],
            "outer_feature_surface_hashes": [surface.surface_hash for surface in surfaces],
            "row_count": len(features),
            "utility_values_available_to_feature_builder": False,
            "query_and_case_clusters_are_uncertainty_units": True,
            "seed_selection_performed": False,
        }
    )
    return features, feature_hash


def materialize_candidate_feature_rows(
    inputs: PreparedDevelopmentInputs,
    generated: GeneratedDevelopmentCache,
) -> tuple[CandidateFeatureRow, ...]:
    """Build the complete label-free feature table from sealed cache components."""

    components = generated.component_by_key
    rows: list[CandidateFeatureRow] = []
    for outer, query, source, training_seed, generation_seed in expected_utility_keys():
        record = components[(query, source, training_seed)]
        reconstruction, kl = load_component_arrays(generated, record)
        replica_energies = {
            seed: components[(query, source, seed)].case_equal_energy
            for seed in sorted({key[2] for key in components if key[:2] == (query, source)})
        }
        rows.append(
            build_candidate_feature_row(
                outer_target=outer,
                pseudo_query=query,
                candidate_source=source,
                training_seed=training_seed,
                generation_seed=generation_seed,
                support_partition_hash=record.support_partition_hash,
                case_ids=inputs.support_case_ids_by_center[query],
                per_class_reconstruction_mse=reconstruction,
                per_class_normalized_ps_kl=kl,
                replica_case_equal_energies=replica_energies,
                linear_kernel_mmd2=record.linear_kernel_mmd2_by_generation_seed[
                    generation_seed
                ],
                metadata_similarity=inputs.reservation.metadata_similarity_by_query_source[
                    query
                ][source],
            )
        )
    if len(rows) != EXPECTED_UTILITY_ROW_COUNT or tuple(
        row.row_key for row in rows
    ) != expected_utility_keys():
        raise ProtocolError("Exact-tail candidate feature production coverage drifted.")
    return tuple(rows)


def _class_marginal_component(
    raw: Mapping[int, Sequence[float] | np.ndarray], row_count: int, role: str
) -> np.ndarray:
    if set(raw) != {0, 1}:
        raise ProtocolError(f"Utility-aligned {role} components require both classes.")
    values = [np.asarray(raw[label], dtype=np.float64) for label in (0, 1)]
    if any(value.shape != (row_count,) for value in values) or not all(
        np.isfinite(value).all() and np.all(value >= 0.0) for value in values
    ):
        raise ProtocolError(f"Utility-aligned {role} components are invalid.")
    return 0.5 * (values[0] + values[1])


def _case_equal_values(
    values: np.ndarray, cases: Sequence[str], case_order: Sequence[str]
) -> np.ndarray:
    grouped: dict[str, list[float]] = defaultdict(list)
    for case_id, value in zip(cases, values, strict=True):
        grouped[case_id].append(float(value))
    result = np.asarray(
        [np.mean(grouped[case_id], dtype=np.float64) for case_id in case_order],
        dtype=np.float64,
    )
    if not np.isfinite(result).all():
        raise ProtocolError("Utility-aligned case summaries are non-finite.")
    return result


def _distribution_stats(values: np.ndarray) -> tuple[float, float, float, float, float]:
    quantiles = np.quantile(values, [0.25, 0.5, 0.75])
    return (
        float(np.mean(values, dtype=np.float64)),
        float(np.std(values, ddof=0)),
        float(quantiles[0]),
        float(quantiles[1]),
        float(quantiles[2]),
    )


__all__ = (
    "build_candidate_feature_row",
    "materialize_candidate_feature_rows",
    "validate_aligned_candidate_features",
)
