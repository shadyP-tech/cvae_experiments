"""Label-free execution for the local marginal-utility Stage-90 diagnostic.

Nothing in this module can open the validation manifest.  It constructs the
case-level support partition from cache identities, evaluates label-free CVAE
compatibility features, and materializes every predeclared perturbation arm.
The sibling label-access module is the only route to outcomes and requires the
global prediction seal written after this module returns.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ....data.features.uniform_b_routing_validation import (
    load_unlabeled_validation_shard,
    validate_uniform_b_routing_validation_cache,
)
from ....real_features.classifier_reference.classifiers import fit_logistic_classifier
from ...expert_bank.uniform_b_v2_promotion.serialization import (
    load_routing_authorized_expert,
)
from ...generation.contracts import GenerationLock
from ...generation.generation import (
    derived_composition_seed,
    generate_source_block,
    source_generation_plan,
)
from ...protocol import ProtocolError
from ...routing.dense_residual_soft_router import (
    DEFAULT_SCALE_FLOOR,
    ReplicaKey,
    calibrate_own_source_energies,
    compose_prefix_blocks,
    deterministic_case_partitions,
    score_variational_compatibility,
)
from .contracts import (
    CENTERS,
    DEVELOPMENT_TOTAL_PER_CLASS,
    EPSILON,
    EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT,
    GENERATION_SEEDS,
    SUPPORT_CASE_COUNT,
    SUPPORT_PARTITION_NAMESPACE,
    SUPPORT_SPLIT_SEED,
    TOTAL_PER_CLASS,
    TRAINING_SEEDS,
    ValidationRowIdentity,
    legal_sources,
    perturbation_library_for,
    row_identity_hash,
)
from .prediction_io import FlatPredictionStore, PredictionAccumulator

if TYPE_CHECKING:  # pragma: no cover
    from .config import LocalMarginalUtilityRouterConfig


SUPPORT_PARTITION_COLUMNS = (
    "schema_version",
    "row_ordinal",
    "manifest_row_index",
    "sample_id",
    "case_id",
    "center",
    "split",
    "partition_role",
    "center_partition_hash",
    "support_split_seed",
    "label_present",
)

COMPATIBILITY_CASE_COLUMNS = (
    "schema_version",
    "source_center",
    "training_seed",
    "query_center",
    "case_id",
    "query_partition_role",
    "row_count",
    "marginal_variational_energy",
    "class_0_energy",
    "class_1_energy",
    "class_0_common_reconstruction_mse",
    "class_1_common_reconstruction_mse",
    "class_0_normalized_ps_kl",
    "class_1_normalized_ps_kl",
    "class_prior_json",
    "labels_used",
    "exact_nelbo_claimed",
)

COMPATIBILITY_SCORE_COLUMNS = (
    "schema_version",
    "query_center",
    "source_center",
    "training_seed_17_z",
    "training_seed_42_z",
    "training_seed_101_z",
    "mean_calibrated_energy_z",
    "query_support_case_count",
    "replica_aggregation",
    "legal_development_outer_targets_json",
    "legal_target_candidate",
    "query_support_labels_used",
    "exact_nelbo_claimed",
)


@dataclass(frozen=True)
class LabelFreeValidationFrame:
    embeddings: np.ndarray
    rows: tuple[ValidationRowIdentity, ...]
    rows_by_center: Mapping[str, tuple[ValidationRowIdentity, ...]]
    cache_binding: Mapping[str, object]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.embeddings)
        if (
            matrix.shape != (len(self.rows), 3840)
            or matrix.dtype != np.float32
            or not np.isfinite(matrix).all()
            or tuple(row.row_ordinal for row in self.rows)
            != tuple(range(len(self.rows)))
        ):
            raise ProtocolError("Local-utility label-free frame geometry drifted.")
        if set(self.rows_by_center) != set(CENTERS):
            raise ProtocolError("Local-utility label-free center coverage drifted.")
        if len({row.sample_id for row in self.rows}) != len(self.rows):
            raise ProtocolError("Local-utility sample identities duplicate.")

    @property
    def cache_binding_hash(self) -> str:
        return stable_hash(dict(self.cache_binding))

    def embeddings_for(self, rows: Sequence[ValidationRowIdentity]) -> np.ndarray:
        ordinals = np.asarray([row.row_ordinal for row in rows], dtype=np.int64)
        if not len(ordinals) or np.any(ordinals < 0) or np.any(ordinals >= len(self.rows)):
            raise ProtocolError("Local-utility row slice is invalid.")
        expected = tuple(self.rows[int(index)].sample_id for index in ordinals)
        if expected != tuple(row.sample_id for row in rows):
            raise ProtocolError("Local-utility row slice identity drifted.")
        return np.ascontiguousarray(self.embeddings[ordinals], dtype=np.float32)


@dataclass(frozen=True)
class PartitionSurface:
    support_rows_by_center: Mapping[str, tuple[ValidationRowIdentity, ...]]
    evaluation_rows_by_center: Mapping[str, tuple[ValidationRowIdentity, ...]]
    table_rows: tuple[Mapping[str, object], ...]
    lock_payload: Mapping[str, object]

    @property
    def lock_hash(self) -> str:
        return str(self.lock_payload["support_partition_lock_hash"])


@dataclass(frozen=True)
class CompatibilitySurface:
    calibrated_energy_by_query: Mapping[str, Mapping[str, float]]
    case_rows: tuple[Mapping[str, object], ...]
    score_rows: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class DevelopmentPredictionSurface:
    store: FlatPredictionStore


def load_label_free_validation_frame(
    config: "LocalMarginalUtilityRouterConfig",
) -> LabelFreeValidationFrame:
    """Validate and load feature shards without opening their label manifest."""

    checks = validate_uniform_b_routing_validation_cache(config.validation_cache_root)
    if checks.get("status") != "PASS" or checks.get("label_fields_absent") is not True:
        raise ProtocolError("Local-utility validation cache is not label-free and valid.")
    arrays: list[np.ndarray] = []
    rows: list[ValidationRowIdentity] = []
    by_center: dict[str, tuple[ValidationRowIdentity, ...]] = {}
    shard_hashes: dict[str, str] = {}
    ordinal = 0
    for center in CENTERS:
        shard = load_unlabeled_validation_shard(
            config.validation_cache_root / f"embeddings/by_center/center_{center}.pt",
            expected_center=center,
        )
        center_rows: list[ValidationRowIdentity] = []
        for metadata in shard.metadata:
            row = ValidationRowIdentity(
                row_ordinal=ordinal,
                manifest_row_index=int(metadata["manifest_row_index"]),
                sample_id=str(metadata["sample_id"]),
                case_id=str(metadata["case_id"]),
                center=center,
                split=str(metadata["split"]),
                partition_role="evaluation",
            )
            center_rows.append(row)
            rows.append(row)
            ordinal += 1
        arrays.append(np.asarray(shard.embeddings, dtype=np.float32))
        by_center[center] = tuple(center_rows)
        shard_hashes[center] = shard.cache_sha256
    protocol = _json(config.validation_cache_root / "manifests/frozen_build_protocol.json")
    content = _json(config.validation_cache_root / "manifests/content_index.json")
    if (
        protocol.get("cache_name") != config.expected_validation_cache_semantic_id
        or protocol.get("representation_id")
        != config.expected_validation_cache_representation_id
        or protocol.get("validation_split") != "val"
    ):
        raise ProtocolError("Local-utility validation-cache identity drifted.")
    binding = {
        "schema_version": "midogpp_local_marginal_utility_cache_binding_v1",
        "cache_artifact_id": config.validation_cache_artifact_id,
        "cache_name": protocol.get("cache_name"),
        "representation_id": protocol.get("representation_id"),
        "validation_split": protocol.get("validation_split"),
        "feature_dim": 3840,
        "row_count": len(rows),
        "center_count": len(CENTERS),
        "cache_protocol_hash": protocol.get("frozen_build_protocol_hash"),
        "cache_content_hash": content.get("content_hash"),
        "shard_sha256_by_center": shard_hashes,
        "labels_persisted": False,
        "manifest_opened": False,
    }
    return LabelFreeValidationFrame(
        embeddings=np.ascontiguousarray(np.concatenate(arrays, axis=0), dtype=np.float32),
        rows=tuple(rows),
        rows_by_center=by_center,
        cache_binding=binding,
    )


def build_partition_surface(
    frame: LabelFreeValidationFrame,
    *,
    config_contract_hash: str,
) -> PartitionSurface:
    """Create a deterministic whole-case support/evaluation split from IDs."""

    support_by_center: dict[str, tuple[ValidationRowIdentity, ...]] = {}
    evaluation_by_center: dict[str, tuple[ValidationRowIdentity, ...]] = {}
    table: list[dict[str, object]] = []
    center_payloads: dict[str, object] = {}
    for center in CENTERS:
        original = frame.rows_by_center[center]
        partition = deterministic_case_partitions(
            [row.sample_id for row in original],
            [row.case_id for row in original],
            target_center=center,
            support_case_count=SUPPORT_CASE_COUNT,
            namespace=SUPPORT_PARTITION_NAMESPACE,
            split_seed=SUPPORT_SPLIT_SEED,
        )
        support_indices = set(partition.support_indices)
        support_rows: list[ValidationRowIdentity] = []
        evaluation_rows: list[ValidationRowIdentity] = []
        for local_index, original_row in enumerate(original):
            role = "support" if local_index in support_indices else "evaluation"
            row = ValidationRowIdentity(
                row_ordinal=original_row.row_ordinal,
                manifest_row_index=original_row.manifest_row_index,
                sample_id=original_row.sample_id,
                case_id=original_row.case_id,
                center=original_row.center,
                split=original_row.split,
                partition_role=role,
            )
            (support_rows if role == "support" else evaluation_rows).append(row)
            table.append(
                {
                    "schema_version": "midogpp_local_marginal_utility_partition_row_v1",
                    "row_ordinal": row.row_ordinal,
                    "manifest_row_index": row.manifest_row_index,
                    "sample_id": row.sample_id,
                    "case_id": row.case_id,
                    "center": row.center,
                    "split": row.split,
                    "partition_role": role,
                    "center_partition_hash": partition.partition_hash,
                    "support_split_seed": SUPPORT_SPLIT_SEED,
                    "label_present": False,
                }
            )
        if set(row.case_id for row in support_rows).intersection(
            row.case_id for row in evaluation_rows
        ):
            raise ProtocolError("Local-utility support/evaluation cases overlap.")
        support_by_center[center] = tuple(support_rows)
        evaluation_by_center[center] = tuple(evaluation_rows)
        center_payloads[center] = {
            "partition_hash": partition.partition_hash,
            "support_cases": list(partition.support_cases),
            "evaluation_cases": list(partition.evaluation_cases),
            "support_row_identity_hash": row_identity_hash(tuple(support_rows)),
            "evaluation_row_identity_hash": row_identity_hash(tuple(evaluation_rows)),
            "support_row_count": len(support_rows),
            "evaluation_row_count": len(evaluation_rows),
        }
    unhashed = {
        "schema_version": "midogpp_local_marginal_utility_partition_lock_v1",
        "status": "LOCKED_FROM_LABEL_FREE_CACHE_IDENTITIES",
        "config_contract_hash": config_contract_hash,
        "validation_cache_binding_hash": frame.cache_binding_hash,
        "support_case_count_per_center": SUPPORT_CASE_COUNT,
        "support_split_seed": SUPPORT_SPLIT_SEED,
        "support_partition_namespace": SUPPORT_PARTITION_NAMESPACE,
        "centers": center_payloads,
        "manifest_opened": False,
        "labels_used": False,
        "whole_case": True,
        "support_evaluation_case_disjoint": True,
        "support_evaluation_sample_disjoint": True,
    }
    lock = {**unhashed, "support_partition_lock_hash": stable_hash(unhashed)}
    return PartitionSurface(
        support_rows_by_center=support_by_center,
        evaluation_rows_by_center=evaluation_by_center,
        table_rows=tuple(table),
        lock_payload=lock,
    )


def compute_compatibility_surface(
    config: "LocalMarginalUtilityRouterConfig",
    frame: LabelFreeValidationFrame,
    partitions: PartitionSurface,
) -> CompatibilitySurface:
    """Compute label-free compatibility features for all ordered q/e pairs."""

    energies: dict[tuple[str, int, str], object] = {}
    case_rows: list[dict[str, object]] = []
    for source in CENTERS:
        for training_seed in TRAINING_SEEDS:
            expert = load_routing_authorized_expert(
                config.expert_bank_root,
                source_center=source,
                training_seed=training_seed,
                device=config.compatibility_device,
            )
            try:
                for query in CENTERS:
                    query_rows = partitions.support_rows_by_center[query]
                    energy = score_variational_compatibility(
                        expert,
                        frame.embeddings_for(query_rows),
                        [row.case_id for row in query_rows],
                    )
                    energies[(source, training_seed, query)] = energy
                    cases = np.asarray([row.case_id for row in query_rows], dtype=str)
                    for case_id in energy.case_order:
                        mask = cases == case_id
                        case_rows.append(
                            {
                                "schema_version": "midogpp_local_marginal_utility_compatibility_case_v1",
                                "source_center": source,
                                "training_seed": training_seed,
                                "query_center": query,
                                "case_id": case_id,
                                "query_partition_role": "support",
                                "row_count": int(np.sum(mask)),
                                "marginal_variational_energy": float(energy.per_case[case_id]),
                                "class_0_energy": float(np.mean(energy.per_class_energy[0][mask])),
                                "class_1_energy": float(np.mean(energy.per_class_energy[1][mask])),
                                "class_0_common_reconstruction_mse": float(np.mean(energy.per_class_reconstruction_mse[0][mask])),
                                "class_1_common_reconstruction_mse": float(np.mean(energy.per_class_reconstruction_mse[1][mask])),
                                "class_0_normalized_ps_kl": float(np.mean(energy.per_class_normalized_ps_kl[0][mask])),
                                "class_1_normalized_ps_kl": float(np.mean(energy.per_class_normalized_ps_kl[1][mask])),
                                "class_prior_json": _json_compact([0.5, 0.5]),
                                "labels_used": False,
                                "exact_nelbo_claimed": False,
                            }
                        )
            finally:
                del expert
                _empty_device_cache(config.compatibility_device)

    calibrated_by_query: dict[str, dict[str, float]] = {}
    score_rows: list[dict[str, object]] = []
    for query in CENTERS:
        candidates = tuple(center for center in CENTERS if center != query)
        query_support_cases = {row.case_id for row in partitions.support_rows_by_center[query]}
        query_map: dict[ReplicaKey, Mapping[str, float]] = {}
        own_map: dict[ReplicaKey, Mapping[str, float]] = {}
        for source in candidates:
            for training_seed in TRAINING_SEEDS:
                key = ReplicaKey(source, training_seed)
                query_energy = energies[(source, training_seed, query)]
                own_energy = energies[(source, training_seed, source)]
                query_map[key] = {
                    case_id: float(query_energy.per_case[case_id])
                    for case_id in sorted(query_support_cases)
                }
                own_map[key] = dict(own_energy.per_case)
        calibration = calibrate_own_source_energies(
            query_map,
            own_map,
            candidate_sources=candidates,
            training_seeds=TRAINING_SEEDS,
            scale_floor=DEFAULT_SCALE_FLOOR,
        )
        calibrated_by_query[query] = dict(calibration.mean_z_by_source)
        replica_by_key = {
            (row.source_center, row.training_seed): row for row in calibration.replicas
        }
        for source in candidates:
            score_rows.append(
                {
                    "schema_version": "midogpp_local_marginal_utility_compatibility_score_v1",
                    "query_center": query,
                    "source_center": source,
                    "training_seed_17_z": replica_by_key[(source, 17)].calibrated_z,
                    "training_seed_42_z": replica_by_key[(source, 42)].calibrated_z,
                    "training_seed_101_z": replica_by_key[(source, 101)].calibrated_z,
                    "mean_calibrated_energy_z": calibration.mean_z_by_source[source],
                    "query_support_case_count": len(query_support_cases),
                    "replica_aggregation": "arithmetic_mean_all_three_no_seed_selection",
                    "legal_development_outer_targets_json": _json_compact(
                        [center for center in CENTERS if center not in {query, source}]
                    ),
                    "legal_target_candidate": True,
                    "query_support_labels_used": False,
                    "exact_nelbo_claimed": False,
                }
            )
    return CompatibilitySurface(
        calibrated_energy_by_query=calibrated_by_query,
        case_rows=tuple(case_rows),
        score_rows=tuple(score_rows),
    )


def materialize_development_predictions(
    config: "LocalMarginalUtilityRouterConfig",
    generation_lock: GenerationLock,
    frame: LabelFreeValidationFrame,
    partitions: PartitionSurface,
) -> DevelopmentPredictionSurface:
    """Predict all 5,184 fixed cells before a development label can open."""

    key_map = _generation_key_map(generation_lock)
    accumulator = PredictionAccumulator()
    for training_seed in TRAINING_SEEDS:
        for generation_seed in GENERATION_SEEDS:
            blocks = _generate_seed_cell_blocks(
                config,
                key_map,
                training_seed=training_seed,
                generation_seed=generation_seed,
            )
            try:
                for outer_target in CENTERS:
                    for query_center in CENTERS:
                        if query_center == outer_target:
                            continue
                        candidates = legal_sources(
                            outer_target=outer_target,
                            query_center=query_center,
                        )
                        eval_rows = partitions.evaluation_rows_by_center[query_center]
                        eval_matrix = frame.embeddings_for(eval_rows)
                        for plan in perturbation_library_for(
                            outer_target=outer_target,
                            query_center=query_center,
                        ):
                            if plan.candidate_sources != candidates:
                                raise ProtocolError("Local-utility perturbation pool drifted.")
                            shuffle_seeds = {
                                str(label): derived_composition_seed(
                                    generation_lock_hash=generation_lock.generation_lock_hash,
                                    target_center=query_center,
                                    training_seed=training_seed,
                                    generation_seed=generation_seed,
                                    class_label=label,
                                )
                                for label in (0, 1)
                            }
                            composition = compose_prefix_blocks(
                                {source: blocks[source] for source in candidates},
                                plan.allocations,
                                shuffle_seed_by_class=shuffle_seeds,
                                total_per_class=DEVELOPMENT_TOTAL_PER_CLASS,
                            )
                            fitted = _fit_classifier(
                                config,
                                composition.embeddings,
                                composition.labels,
                                eval_matrix,
                            )
                            accumulator.append(
                                predictions=fitted["predictions"],
                                probabilities=fitted["probabilities"],
                                metadata=_prediction_metadata(
                                    outer_target=outer_target,
                                    query_center=query_center,
                                    plan=plan,
                                    training_seed=training_seed,
                                    generation_seed=generation_seed,
                                    shuffle_seeds=shuffle_seeds,
                                    composition_hash=composition.composition_hash,
                                    fitted=fitted,
                                    eval_rows=eval_rows,
                                ),
                            )
            finally:
                del blocks
                gc.collect()
                _empty_device_cache(config.generation_device)
    store = accumulator.finish()
    if len(store.index_rows) != EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT:
        raise ProtocolError("Local-utility prediction coverage drifted.")
    return DevelopmentPredictionSurface(store=store)


def _generation_key_map(
    generation_lock: GenerationLock,
) -> dict[tuple[str, int, int], object]:
    keys = source_generation_plan(generation_lock)
    output = {
        (key.source_center, key.training_seed, key.generation_seed): key for key in keys
    }
    expected = {
        (source, training_seed, generation_seed)
        for source in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    }
    if len(keys) != len(output) or set(output) != expected:
        raise ProtocolError("Local-utility GenerationLock source plan drifted.")
    return output


def _generate_seed_cell_blocks(
    config: "LocalMarginalUtilityRouterConfig",
    key_map: Mapping[tuple[str, int, int], object],
    *,
    training_seed: int,
    generation_seed: int,
) -> dict[str, object]:
    blocks: dict[str, object] = {}
    for source in CENTERS:
        expert = load_routing_authorized_expert(
            config.expert_bank_root,
            source_center=source,
            training_seed=training_seed,
            device=config.generation_device,
        )
        try:
            blocks[source] = generate_source_block(
                expert,
                key_map[(source, training_seed, generation_seed)],
                per_class=TOTAL_PER_CLASS,
                device=config.generation_device,
            )
        finally:
            del expert
            _empty_device_cache(config.generation_device)
    return blocks


def _fit_classifier(
    config: "LocalMarginalUtilityRouterConfig",
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    eval_embeddings: np.ndarray,
) -> dict[str, object]:
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Local-utility fitting requires threadpoolctl.") from exc
    with threadpool_limits(limits=config.threads_per_fit):
        fitted = fit_logistic_classifier(
            train_embeddings,
            train_labels,
            eval_embeddings,
            spec=config.classifier,
        )
    predictions = np.asarray(fitted.predictions)
    probabilities = np.asarray(fitted.probabilities, dtype=float)
    if (
        tuple(int(value) for value in fitted.classes) != (0, 1)
        or predictions.shape != (len(eval_embeddings),)
        or not np.isin(predictions, [0, 1]).all()
        or probabilities.shape != (len(eval_embeddings), 2)
        or not np.isfinite(probabilities).all()
        or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-7)
        or not fitted.converged
        or fitted.classifier_config_hash != config.classifier.config_hash
        or not fitted.scaler_state_hash
    ):
        raise ProtocolError("Local-utility classifier fit or prediction drifted.")
    return {
        "predictions": predictions.astype(np.uint8, copy=False),
        "probabilities": probabilities[:, 1].astype(np.float32, copy=False),
        "classes": tuple(int(value) for value in fitted.classes),
        "n_iter": tuple(int(value) for value in fitted.n_iter),
        "converged": bool(fitted.converged),
        "classifier_config_hash": fitted.classifier_config_hash,
        "scaler_state_hash": fitted.scaler_state_hash,
    }


def _prediction_metadata(
    *,
    outer_target: str,
    query_center: str,
    plan: object,
    training_seed: int,
    generation_seed: int,
    shuffle_seeds: Mapping[str, int],
    composition_hash: str,
    fitted: Mapping[str, object],
    eval_rows: Sequence[ValidationRowIdentity],
) -> dict[str, object]:
    boosted_source = getattr(plan, "boosted_source")
    return {
        "schema_version": "midogpp_local_marginal_utility_prediction_cell_v1",
        "phase": "development_utility_surface",
        "outer_target": outer_target,
        "query_center": query_center,
        "action_id": str(getattr(plan, "action_id")),
        "arm_role": str(getattr(plan, "arm_role")),
        "boosted_source": "" if boosted_source is None else str(boosted_source),
        "epsilon": EPSILON,
        "training_seed": training_seed,
        "generation_seed": generation_seed,
        "candidate_sources_json": _json_compact(list(getattr(plan, "candidate_sources"))),
        "weights_json": _json_compact(dict(getattr(plan, "weights"))),
        "allocations_json": _json_compact(dict(getattr(plan, "allocations"))),
        "effective_source_count": float(getattr(plan, "effective_source_count")),
        "shuffle_seed_by_class_json": _json_compact(dict(shuffle_seeds)),
        "composition_hash": composition_hash,
        "classifier_config_hash": str(fitted["classifier_config_hash"]),
        "scaler_state_hash": str(fitted["scaler_state_hash"]),
        "classifier_classes_json": _json_compact(list(fitted["classes"])),
        "classifier_n_iter_json": _json_compact(list(fitted["n_iter"])),
        "classifier_converged": bool(fitted["converged"]),
        "evaluation_row_ids_json": _json_compact([row.sample_id for row in eval_rows]),
        "evaluation_row_identity_hash": row_identity_hash(tuple(eval_rows)),
    }


def _json_compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read local-utility cache member: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Local-utility JSON must be an object: {path}.")
    return payload


def _empty_device_cache(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda"):
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


__all__ = (
    "COMPATIBILITY_CASE_COLUMNS",
    "COMPATIBILITY_SCORE_COLUMNS",
    "SUPPORT_PARTITION_COLUMNS",
    "CompatibilitySurface",
    "DevelopmentPredictionSurface",
    "LabelFreeValidationFrame",
    "PartitionSurface",
    "build_partition_surface",
    "compute_compatibility_surface",
    "load_label_free_validation_frame",
    "materialize_development_predictions",
)
