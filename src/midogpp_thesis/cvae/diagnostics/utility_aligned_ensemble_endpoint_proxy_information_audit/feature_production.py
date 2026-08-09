"""Label-free proxy primitive production from the audit's sealed support vectors."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from ...routing.utility_aligned.ensemble_endpoint import (
    mean_exact_nine_positive_class_probabilities,
)
from .contracts import (
    CENTERS,
    EXPECTED_PROXY_FEATURE_ROW_COUNT,
    PROXY_FEATURE_SCHEMA,
)
from .execution_adapter import (
    DevelopmentPredictionCapability,
    EnsembleSeedFeatureProduction,
    validate_global_development_seal,
)
from .input_contracts import row_identity_hash


def produce_label_free_proxy_feature_payloads(
    seed_features: EnsembleSeedFeatureProduction,
    development: DevelopmentPredictionCapability,
    partitions: object,
) -> tuple[dict[str, object], ...]:
    """Build 504 candidate rows using support probabilities and no labels.

    Evaluation probability vectors are deliberately unreachable: this adapter
    calls ``CombinedPredictionStore.vectors`` only with role ``support``.
    """

    seal = validate_global_development_seal(development)
    grouped: dict[tuple[str, str, str], list[object]] = defaultdict(list)
    for row in seed_features.inner_rows:
        grouped[(row.outer_target_id, row.query_id, row.candidate_source)].append(row)
    if len(grouped) != EXPECTED_PROXY_FEATURE_ROW_COUNT:
        raise ProtocolError("Proxy-audit seed feature coverage drifted.")

    raw: dict[tuple[str, str, str], dict[str, object]] = {}
    for key in sorted(grouped):
        outer, query, source = key
        if outer == query or source in {outer, query}:
            raise ProtocolError("Proxy-audit requires distinct H/q/e domains.")
        rows = tuple(sorted(grouped[key], key=lambda item: (item.training_seed, item.generation_seed)))
        if len(rows) != 9 or len({(row.training_seed, row.generation_seed) for row in rows}) != 9:
            raise ProtocolError("Proxy-audit feature collapse requires exact-nine seed cells.")
        support_rows = partitions.support_rows_by_center[query]
        expected_support_hash = row_identity_hash(support_rows)
        if (
            len({row.support_partition_hash for row in rows}) != 1
            or len({row.support_case_count for row in rows}) != 1
            or rows[0].support_partition_hash != expected_support_hash
        ):
            raise ProtocolError("Proxy-audit support identity drifted across seed features.")

        scope = f"{outer}::{query}"
        base_vectors = development.store.vectors(scope, "B", "support")
        tail_vectors = development.store.vectors(scope, f"Hxe::{source}", "support")
        base = mean_exact_nine_positive_class_probabilities(base_vectors)
        tail = mean_exact_nine_positive_class_probabilities(tail_vectors)
        if len(base) != len(support_rows) or base.shape != tail.shape:
            raise ProtocolError("Proxy-audit support probability geometry drifted.")
        delta = tail - base
        pseudo_sign = np.where(base >= 0.5, 1.0, -1.0)
        base_entropy = _binary_entropy(base)
        tail_entropy = _binary_entropy(tail)
        raw[key] = {
            "outer_target_id": outer,
            "query_id": query,
            "candidate_source": source,
            "support_partition_hash": expected_support_hash,
            "support_case_count": len({row.case_id for row in support_rows}),
            "support_row_count": len(support_rows),
            "seed_feature_row_hashes": [row.row_hash for row in rows],
            "base_support_vector_hashes": [item.vector_hash for item in base_vectors],
            "tail_support_vector_hashes": [item.vector_hash for item in tail_vectors],
            "metadata_similarity": _mean(rows, "metadata_similarity"),
            "reconstruction_mean": _mean(rows, "reconstruction_mean"),
            "kl_mean": _mean(rows, "kl_mean"),
            "distribution_mmd": _mean(rows, "distribution_mmd"),
            "absolute_ensemble_shift": float(np.mean(np.abs(delta), dtype=np.float64)),
            "signed_margin_projection": float(
                np.mean(pseudo_sign * delta, dtype=np.float64)
            ),
            "threshold_flip_rate": float(
                np.mean((base >= 0.5) != (tail >= 0.5), dtype=np.float64)
            ),
            "mean_entropy_change": float(
                np.mean(tail_entropy - base_entropy, dtype=np.float64)
            ),
            "development_prediction_seal_hash": str(seal["prediction_seal_hash"]),
        }

    payloads: list[dict[str, object]] = []
    by_query: dict[tuple[str, str], list[tuple[str, dict[str, object]]]] = defaultdict(list)
    for (outer, query, source), value in raw.items():
        by_query[(outer, query)].append((source, value))
    for outer in CENTERS:
        for query in tuple(center for center in CENTERS if center != outer):
            values = sorted(by_query[(outer, query)])
            if len(values) != 7:
                raise ProtocolError("Proxy-audit candidate list is incomplete.")
            rec_z = _within_group_z([float(value["reconstruction_mean"]) for _, value in values])
            kl_z = _within_group_z([float(value["kl_mean"]) for _, value in values])
            mmd_z = _within_group_z(
                [float(np.log1p(float(value["distribution_mmd"]))) for _, value in values]
            )
            for index, (source, value) in enumerate(values):
                unhashed = {
                    "schema_version": PROXY_FEATURE_SCHEMA,
                    "outer_target_id": outer,
                    "query_id": query,
                    "candidate_source": source,
                    "candidate_source_count": 7,
                    "support_partition_hash": value["support_partition_hash"],
                    "support_case_count": value["support_case_count"],
                    "support_row_count": value["support_row_count"],
                    "seed_pair_count": 9,
                    "seed_feature_row_hashes": value["seed_feature_row_hashes"],
                    "base_support_vector_hashes": value["base_support_vector_hashes"],
                    "tail_support_vector_hashes": value["tail_support_vector_hashes"],
                    "metadata_similarity": value["metadata_similarity"],
                    "absolute_ensemble_shift": value["absolute_ensemble_shift"],
                    "reconstruction_mean_within_query_z": float(rec_z[index]),
                    "kl_mean_within_query_z": float(kl_z[index]),
                    "log_distribution_mmd_within_query_z": float(mmd_z[index]),
                    "signed_margin_projection": value["signed_margin_projection"],
                    "threshold_flip_rate": value["threshold_flip_rate"],
                    "mean_entropy_change": value["mean_entropy_change"],
                    "development_prediction_seal_hash": value[
                        "development_prediction_seal_hash"
                    ],
                    "probability_role_used": "support_only",
                    "labels_used": False,
                    "evaluation_probabilities_used_as_features": False,
                    "technical_seed_rows_are_independent_observations": False,
                }
                payloads.append(
                    {**unhashed, "proxy_feature_row_hash": canonical_sha256(unhashed)}
                )
    result = tuple(payloads)
    if len(result) != EXPECTED_PROXY_FEATURE_ROW_COUNT:
        raise ProtocolError("Proxy-audit feature row count drifted.")
    return result


def build_proxy_feature_lock(
    payloads: Sequence[Mapping[str, object]], *, partition_lock_hash: str,
    development_prediction_seal_hash: str,
) -> dict[str, object]:
    rows = tuple(payloads)
    if len(rows) != EXPECTED_PROXY_FEATURE_ROW_COUNT:
        raise ProtocolError("Proxy-audit lock requires exactly 504 feature rows.")
    unhashed = {
        "schema_version": "midogpp_stage90_proxy_information_feature_lock_v1",
        "status": "SEALED_LABEL_FREE_PROXY_FEATURES_BEFORE_LABEL_ACCESS",
        "support_partition_lock_hash": partition_lock_hash,
        "development_prediction_seal_hash": development_prediction_seal_hash,
        "feature_row_schema": PROXY_FEATURE_SCHEMA,
        "feature_row_count": len(rows),
        "ordered_feature_row_hashes": [row["proxy_feature_row_hash"] for row in rows],
        "feature_families_predeclared": [
            "equal_union_null",
            "metadata_only_control",
            "absolute_shift_control",
            "rich_distributional_compact",
            "directional_action_compact",
            "hybrid_compact",
            "cyclic_directional_permutation_control",
        ],
        "support_probabilities_only": True,
        "support_labels_used": False,
        "evaluation_probabilities_used_as_features": False,
        "development_labels_opened": False,
        "target_actions_built": False,
    }
    return {**unhashed, "proxy_feature_lock_hash": canonical_sha256(unhashed)}


def _mean(rows: Sequence[object], name: str) -> float:
    values = np.asarray([float(getattr(row, name)) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ProtocolError(f"Proxy-audit primitive {name} is non-finite.")
    return float(np.mean(values, dtype=np.float64))


def _within_group_z(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    centered = array - float(np.mean(array, dtype=np.float64))
    rms = float(np.sqrt(np.mean(centered * centered, dtype=np.float64)))
    if rms <= float(np.sqrt(np.finfo(np.float64).eps)):
        return np.zeros_like(centered)
    return centered / rms


def _binary_entropy(probability: np.ndarray) -> np.ndarray:
    epsilon = np.finfo(np.float64).eps
    value = np.clip(np.asarray(probability, dtype=np.float64), epsilon, 1.0 - epsilon)
    return -(value * np.log(value) + (1.0 - value) * np.log(1.0 - value))


__all__ = (
    "PROXY_FEATURE_SCHEMA",
    "build_proxy_feature_lock",
    "produce_label_free_proxy_feature_payloads",
)
