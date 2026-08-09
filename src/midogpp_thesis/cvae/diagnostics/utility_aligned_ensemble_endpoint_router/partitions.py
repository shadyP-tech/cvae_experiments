"""V2-owned deterministic two-case support and whole-case folds."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...routing.dense_residual_soft_router import deterministic_case_partitions
from .contracts import (
    CENTERS, EXPECTED_CASE_OOF_FOLD_COUNT, EXPECTED_TOTAL_CASE_COUNT,
    FIXED_SUPPORT_CASE_COUNT_PER_CENTER, SUPPORT_PARTITION_NAMESPACE, SUPPORT_SPLIT_SEED,
)
from .input_contracts import FixedPartitionSurface, LabelFreeValidationFrame, ValidationRowIdentity, row_identity_hash


SUPPORT_PARTITION_COLUMNS = (
    "schema_version", "row_ordinal", "manifest_row_index", "sample_id", "case_id",
    "center", "split", "partition_role", "center_partition_hash",
    "support_split_seed", "support_partition_namespace", "label_present",
)


@dataclass(frozen=True)
class EvaluationCaseFold:
    fold_ordinal: int
    fold_id: str
    target_center: str
    heldout_case_id: str
    fixed_support_rows: tuple[ValidationRowIdentity, ...]
    heldout_rows: tuple[ValidationRowIdentity, ...]
    fold_hash: str

    @property
    def heldout_row_identity_hash(self) -> str:
        return row_identity_hash(self.heldout_rows)


@dataclass(frozen=True)
class CaseFoldSurface:
    folds: tuple[EvaluationCaseFold, ...]
    folds_by_target: Mapping[str, tuple[EvaluationCaseFold, ...]]
    lock_payload: Mapping[str, object]

    def __post_init__(self) -> None:
        values = {str(key): tuple(rows) for key, rows in self.folds_by_target.items()}
        if (
            len(self.folds) != EXPECTED_CASE_OOF_FOLD_COUNT or tuple(values) != CENTERS
            or tuple(fold for target in CENTERS for fold in values[target]) != self.folds
        ):
            raise ProtocolError("Ensemble-endpoint case-fold surface drifted.")
        object.__setattr__(self, "folds_by_target", MappingProxyType(values))
        object.__setattr__(self, "lock_payload", MappingProxyType(dict(self.lock_payload)))

    @property
    def lock_hash(self) -> str:
        return str(self.lock_payload["case_fold_lock_hash"])


def build_fixed_partition_surface(
    frame: LabelFreeValidationFrame, *, config_contract_hash: str
) -> FixedPartitionSurface:
    support_by_center: dict[str, tuple[ValidationRowIdentity, ...]] = {}
    evaluation_by_center: dict[str, tuple[ValidationRowIdentity, ...]] = {}
    table: list[dict[str, object]] = []
    centers_payload: dict[str, object] = {}
    all_cases: set[str] = set()
    all_samples: set[str] = set()
    for center in CENTERS:
        original = frame.rows_by_center[center]
        partition = deterministic_case_partitions(
            [row.sample_id for row in original], [row.case_id for row in original],
            target_center=center, support_case_count=FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
            namespace=SUPPORT_PARTITION_NAMESPACE, split_seed=SUPPORT_SPLIT_SEED,
        )
        support_indices = set(partition.support_indices)
        support: list[ValidationRowIdentity] = []
        evaluation: list[ValidationRowIdentity] = []
        for local_index, raw in enumerate(original):
            role = "support" if local_index in support_indices else "evaluation"
            row = ValidationRowIdentity(
                row_ordinal=raw.row_ordinal, manifest_row_index=raw.manifest_row_index,
                sample_id=raw.sample_id, case_id=raw.case_id, center=center,
                partition_role=role,
            )
            (support if role == "support" else evaluation).append(row)
            table.append({
                "schema_version": "midogpp_stage90_ensemble_endpoint_partition_row_v1",
                **row.identity_payload(), "center_partition_hash": partition.partition_hash,
                "support_split_seed": SUPPORT_SPLIT_SEED,
                "support_partition_namespace": SUPPORT_PARTITION_NAMESPACE,
                "label_present": False,
            })
        support_cases = tuple(sorted({row.case_id for row in support}))
        evaluation_cases = tuple(sorted({row.case_id for row in evaluation}))
        if len(support_cases) != 2 or not evaluation_cases or set(support_cases) & set(evaluation_cases):
            raise ProtocolError("Ensemble-endpoint fixed support geometry drifted.")
        for row in (*support, *evaluation):
            if row.sample_id in all_samples: raise ProtocolError("Validation sample IDs are not unique.")
            all_samples.add(row.sample_id)
        for case_id in (*support_cases, *evaluation_cases):
            if case_id in all_cases: raise ProtocolError("Validation case IDs are not unique.")
            all_cases.add(case_id)
        support_by_center[center] = tuple(support)
        evaluation_by_center[center] = tuple(evaluation)
        centers_payload[center] = {
            "partition_hash": partition.partition_hash,
            "support_case_ids": list(support_cases), "evaluation_case_ids": list(evaluation_cases),
            "support_row_identity_hash": row_identity_hash(support),
            "evaluation_row_identity_hash": row_identity_hash(evaluation),
            "support_row_count": len(support), "evaluation_row_count": len(evaluation),
        }
    if len(all_cases) != EXPECTED_TOTAL_CASE_COUNT:
        raise ProtocolError("Ensemble-endpoint total-case geometry drifted.")
    unhashed = {
        "schema_version": "midogpp_stage90_ensemble_endpoint_support_partition_lock_v1",
        "status": "LOCKED_FROM_LABEL_FREE_CACHE_IDENTITIES",
        "config_contract_hash": config_contract_hash,
        "validation_cache_binding_hash": frame.cache_binding_hash,
        "fixed_support_case_count_per_center": 2,
        "support_split_seed": SUPPORT_SPLIT_SEED,
        "support_partition_namespace": SUPPORT_PARTITION_NAMESPACE,
        "centers": centers_payload, "manifest_opened": False, "labels_used": False,
        "whole_case": True, "support_evaluation_case_disjoint": True,
        "support_evaluation_sample_disjoint": True,
        "insufficient_support_for_fresh_policy": True,
    }
    return FixedPartitionSurface(
        support_rows_by_center=support_by_center, evaluation_rows_by_center=evaluation_by_center,
        table_rows=tuple(table),
        lock_payload={**unhashed, "support_partition_lock_hash": stable_hash(unhashed)},
    )


def build_case_fold_surface(
    partitions: FixedPartitionSurface, *, config_contract_hash: str
) -> CaseFoldSurface:
    folds: list[EvaluationCaseFold] = []
    by_target: dict[str, tuple[EvaluationCaseFold, ...]] = {}
    targets: dict[str, object] = {}
    for target in CENTERS:
        target_folds: list[EvaluationCaseFold] = []
        support = partitions.support_rows_by_center[target]
        for case_id in sorted({row.case_id for row in partitions.evaluation_rows_by_center[target]}):
            rows = tuple(row for row in partitions.evaluation_rows_by_center[target] if row.case_id == case_id)
            ordinal = len(folds)
            fold_id = f"ensemble_endpoint_fold_{ordinal:02d}_target_{target}_case_{case_id}"
            unhashed = {
                "schema_version": "midogpp_stage90_ensemble_endpoint_case_fold_v1",
                "fold_ordinal": ordinal, "fold_id": fold_id, "target_center": target,
                "heldout_case_id": case_id, "support_partition_lock_hash": partitions.lock_hash,
                "config_contract_hash": config_contract_hash,
                "support_partition_namespace": SUPPORT_PARTITION_NAMESPACE,
                "fixed_support_case_ids": sorted({row.case_id for row in support}),
                "fixed_support_row_identity_hash": row_identity_hash(support),
                "heldout_row_identity_hash": row_identity_hash(rows),
                "heldout_case_excluded_from_route": True,
                "other_evaluation_embeddings_used_for_route": False,
                "support_labels_used": False, "evaluation_labels_used_for_route": False,
            }
            fold = EvaluationCaseFold(
                fold_ordinal=ordinal, fold_id=fold_id, target_center=target,
                heldout_case_id=case_id, fixed_support_rows=support, heldout_rows=rows,
                fold_hash=stable_hash(unhashed),
            )
            folds.append(fold); target_folds.append(fold)
        by_target[target] = tuple(target_folds)
        targets[target] = {"fold_ids": [fold.fold_id for fold in target_folds], "fold_hashes": [fold.fold_hash for fold in target_folds]}
    if len(folds) != EXPECTED_CASE_OOF_FOLD_COUNT:
        raise ProtocolError("Ensemble-endpoint case-fold count drifted.")
    unhashed = {
        "schema_version": "midogpp_stage90_ensemble_endpoint_case_fold_lock_v1",
        "status": "LOCKED_BEFORE_LABEL_ACCESS", "config_contract_hash": config_contract_hash,
        "support_partition_lock_hash": partitions.lock_hash,
        "support_partition_namespace": SUPPORT_PARTITION_NAMESPACE,
        "fold_count": len(folds), "targets": targets,
        "each_evaluation_case_held_out_exactly_once": True,
        "other_evaluation_embeddings_used_for_route": False, "labels_used": False,
    }
    return CaseFoldSurface(
        folds=tuple(folds), folds_by_target=by_target,
        lock_payload={**unhashed, "case_fold_lock_hash": stable_hash(unhashed)},
    )


__all__ = (
    "CaseFoldSurface", "EvaluationCaseFold", "SUPPORT_PARTITION_COLUMNS",
    "build_case_fold_surface", "build_fixed_partition_surface",
)
