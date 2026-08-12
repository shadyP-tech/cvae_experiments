"""Immutable label-free feature-runtime products and task contracts.

The contracts deliberately have no outcome, response, utility, or evaluation
field.  Expensive variational components are keyed only by
``(query, candidate source, training seed)`` and are later expanded into the
strict ``H/q/e`` development geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from ...routing.utility_aligned import (
    CandidateFeatureRow,
    SupportActionProbabilityShift,
    TargetSupportActionShiftCase,
)
from .contracts import (
    CENTERS,
    GENERATION_SEEDS,
    SEED_PAIRS,
    TRAINING_SEEDS,
    candidate_sources,
    inner_candidate_sources,
)


FEATURE_CHECKPOINT_DIRECTORY = "checkpoints/feature_runtime"
FEATURE_COMPONENT_COUNT = 216
INNER_SEED_FEATURE_ROW_COUNT = 4_536
TARGET_SEED_FEATURE_ROW_COUNT = 648
SOURCE_INNER_SHIFT_COUNT = 504
TARGET_CASE_SHIFT_COUNT = 576


@dataclass(frozen=True)
class SupportSlice:
    """One staged exact-eight whole-case query-support array."""

    query_center: str
    relative_array_path: str
    array_sha256: str
    row_count: int
    case_ids: tuple[str, ...]
    support_case_ids: tuple[str, ...]
    row_identity_hash: str
    center_partition_hash: str
    feature_support_partition_hash: str
    slice_hash: str

    def __post_init__(self) -> None:
        cases = tuple(map(str, self.case_ids))
        support_cases = tuple(map(str, self.support_case_ids))
        payload = self.unhashed_payload(cases=cases, support_cases=support_cases)
        if (
            self.query_center not in CENTERS
            or Path(self.relative_array_path).is_absolute()
            or ".." in Path(self.relative_array_path).parts
            or isinstance(self.row_count, bool)
            or not isinstance(self.row_count, Integral)
            or int(self.row_count) != len(cases)
            or not cases
            or support_cases != tuple(sorted(set(cases)))
            or len(support_cases) != 8
            or any(not _sha256(value) for value in (
                self.array_sha256,
                self.row_identity_hash,
                self.center_partition_hash,
                self.feature_support_partition_hash,
            ))
            or self.slice_hash != canonical_sha256(payload)
        ):
            raise ProtocolError("Endpoint-router feature support slice drifted.")
        object.__setattr__(self, "row_count", int(self.row_count))
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "support_case_ids", support_cases)

    def unhashed_payload(
        self,
        *,
        cases: Sequence[str] | None = None,
        support_cases: Sequence[str] | None = None,
    ) -> dict[str, object]:
        return {
            "schema_version": "midogpp_endpoint_router_feature_support_slice_v1",
            "query_center": self.query_center,
            "relative_array_path": self.relative_array_path,
            "array_sha256": self.array_sha256,
            "row_count": self.row_count,
            "case_ids": list(self.case_ids if cases is None else cases),
            "support_case_ids": list(
                self.support_case_ids if support_cases is None else support_cases
            ),
            "row_identity_hash": self.row_identity_hash,
            "center_partition_hash": self.center_partition_hash,
            "feature_support_partition_hash": self.feature_support_partition_hash,
            "whole_case": True,
            "labels_available": False,
            "evaluation_rows_available": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self.unhashed_payload(), "slice_hash": self.slice_hash}


def build_support_slice(
    *,
    query_center: str,
    relative_array_path: str,
    array_sha256: str,
    case_ids: Sequence[str],
    row_identity_hash: str,
    center_partition_hash: str,
    feature_support_partition_hash: str,
) -> SupportSlice:
    cases = tuple(map(str, case_ids))
    support_cases = tuple(sorted(set(cases)))
    provisional = SupportSlice.__new__(SupportSlice)
    # Use the contract's single payload renderer without constructing a weak
    # partially validated public object.
    for name, value in {
        "query_center": query_center,
        "relative_array_path": relative_array_path,
        "array_sha256": array_sha256,
        "row_count": len(cases),
        "case_ids": cases,
        "support_case_ids": support_cases,
        "row_identity_hash": row_identity_hash,
        "center_partition_hash": center_partition_hash,
        "feature_support_partition_hash": feature_support_partition_hash,
        "slice_hash": "",
    }.items():
        object.__setattr__(provisional, name, value)
    payload = provisional.unhashed_payload()
    return SupportSlice(
        query_center=query_center,
        relative_array_path=relative_array_path,
        array_sha256=array_sha256,
        row_count=len(cases),
        case_ids=cases,
        support_case_ids=support_cases,
        row_identity_hash=row_identity_hash,
        center_partition_hash=center_partition_hash,
        feature_support_partition_hash=feature_support_partition_hash,
        slice_hash=canonical_sha256(payload),
    )


@dataclass(frozen=True)
class FeatureTask:
    """One promoted replica scored against its eight legal query centers."""

    source_center: str
    training_seed: int
    device: str
    expert_bank_root: str
    source_array_path: str
    source_block_ordinal_by_generation_seed: Mapping[int, int]
    source_block_output_sha256_by_generation_seed: Mapping[int, str]
    support_root: str
    support_slices: tuple[SupportSlice, ...]
    checkpoint_npz_path: str
    checkpoint_json_path: str
    config_contract_hash: str
    bank_lock_hash: str
    source_stream_lock_hash: str
    cache_binding_hash: str
    partition_lock_hash: str
    metadata_grid_hash: str
    task_hash: str

    def __post_init__(self) -> None:
        blocks = {
            int(key): int(value)
            for key, value in self.source_block_ordinal_by_generation_seed.items()
        }
        block_hashes = {
            int(key): value
            for key, value in self.source_block_output_sha256_by_generation_seed.items()
        }
        slices = tuple(self.support_slices)
        payload = self.unhashed_payload(
            blocks=blocks,
            block_hashes=block_hashes,
            slices=slices,
        )
        if (
            self.source_center not in CENTERS
            or self.training_seed not in TRAINING_SEEDS
            or self.device not in {"cuda:0", "cuda:1"}
            or set(blocks) != set(GENERATION_SEEDS)
            or set(block_hashes) != set(GENERATION_SEEDS)
            or any(value < 0 for value in blocks.values())
            or len(set(blocks.values())) != len(blocks)
            or any(not _sha256(value) for value in block_hashes.values())
            or tuple(item.query_center for item in slices)
            != candidate_sources(self.source_center)
            or any(not _stable_hash(value) for value in (
                self.config_contract_hash,
                self.bank_lock_hash,
                self.source_stream_lock_hash,
            ))
            or any(not _sha256(value) for value in (
                self.cache_binding_hash,
                self.partition_lock_hash,
                self.metadata_grid_hash,
            ))
            or self.task_hash != canonical_sha256(payload)
        ):
            raise ProtocolError("Endpoint-router feature task drifted.")
        object.__setattr__(
            self,
            "source_block_ordinal_by_generation_seed",
            MappingProxyType(blocks),
        )
        object.__setattr__(
            self,
            "source_block_output_sha256_by_generation_seed",
            MappingProxyType(block_hashes),
        )
        object.__setattr__(self, "support_slices", slices)

    def unhashed_payload(
        self,
        *,
        blocks: Mapping[int, int] | None = None,
        block_hashes: Mapping[int, str] | None = None,
        slices: Sequence[SupportSlice] | None = None,
    ) -> dict[str, object]:
        block_values = self.source_block_ordinal_by_generation_seed if blocks is None else blocks
        block_hash_values = (
            self.source_block_output_sha256_by_generation_seed
            if block_hashes is None
            else block_hashes
        )
        slice_values = self.support_slices if slices is None else slices
        return _feature_task_payload(
            source_center=self.source_center,
            training_seed=self.training_seed,
            device=self.device,
            expert_bank_root=self.expert_bank_root,
            source_array_path=self.source_array_path,
            blocks=block_values,
            block_hashes=block_hash_values,
            support_root=self.support_root,
            slices=slice_values,
            checkpoint_npz_path=self.checkpoint_npz_path,
            checkpoint_json_path=self.checkpoint_json_path,
            config_contract_hash=self.config_contract_hash,
            bank_lock_hash=self.bank_lock_hash,
            source_stream_lock_hash=self.source_stream_lock_hash,
            cache_binding_hash=self.cache_binding_hash,
            partition_lock_hash=self.partition_lock_hash,
            metadata_grid_hash=self.metadata_grid_hash,
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            type(self),
            (
                self.source_center,
                self.training_seed,
                self.device,
                self.expert_bank_root,
                self.source_array_path,
                dict(self.source_block_ordinal_by_generation_seed),
                dict(self.source_block_output_sha256_by_generation_seed),
                self.support_root,
                self.support_slices,
                self.checkpoint_npz_path,
                self.checkpoint_json_path,
                self.config_contract_hash,
                self.bank_lock_hash,
                self.source_stream_lock_hash,
                self.cache_binding_hash,
                self.partition_lock_hash,
                self.metadata_grid_hash,
                self.task_hash,
            ),
        )


def build_feature_task(
    *,
    source_center: str,
    training_seed: int,
    device: str,
    expert_bank_root: str,
    source_array_path: str,
    source_block_ordinal_by_generation_seed: Mapping[int, int],
    source_block_output_sha256_by_generation_seed: Mapping[int, str],
    support_root: str,
    support_slices: Sequence[SupportSlice],
    checkpoint_npz_path: str,
    checkpoint_json_path: str,
    config_contract_hash: str,
    bank_lock_hash: str,
    source_stream_lock_hash: str,
    cache_binding_hash: str,
    partition_lock_hash: str,
    metadata_grid_hash: str,
) -> FeatureTask:
    blocks = {int(key): int(value) for key, value in source_block_ordinal_by_generation_seed.items()}
    block_hashes = {
        int(key): value
        for key, value in source_block_output_sha256_by_generation_seed.items()
    }
    slices = tuple(support_slices)
    payload = _feature_task_payload(
        source_center=source_center,
        training_seed=training_seed,
        device=device,
        expert_bank_root=expert_bank_root,
        source_array_path=source_array_path,
        blocks=blocks,
        block_hashes=block_hashes,
        support_root=support_root,
        slices=slices,
        checkpoint_npz_path=checkpoint_npz_path,
        checkpoint_json_path=checkpoint_json_path,
        config_contract_hash=config_contract_hash,
        bank_lock_hash=bank_lock_hash,
        source_stream_lock_hash=source_stream_lock_hash,
        cache_binding_hash=cache_binding_hash,
        partition_lock_hash=partition_lock_hash,
        metadata_grid_hash=metadata_grid_hash,
    )
    return FeatureTask(
        source_center=source_center,
        training_seed=training_seed,
        device=device,
        expert_bank_root=expert_bank_root,
        source_array_path=source_array_path,
        source_block_ordinal_by_generation_seed=blocks,
        source_block_output_sha256_by_generation_seed=block_hashes,
        support_root=support_root,
        support_slices=slices,
        checkpoint_npz_path=checkpoint_npz_path,
        checkpoint_json_path=checkpoint_json_path,
        config_contract_hash=config_contract_hash,
        bank_lock_hash=bank_lock_hash,
        source_stream_lock_hash=source_stream_lock_hash,
        cache_binding_hash=cache_binding_hash,
        partition_lock_hash=partition_lock_hash,
        metadata_grid_hash=metadata_grid_hash,
        task_hash=canonical_sha256(payload),
    )


def _feature_task_payload(
    *,
    source_center: str,
    training_seed: int,
    device: str,
    expert_bank_root: str,
    source_array_path: str,
    blocks: Mapping[int, int],
    block_hashes: Mapping[int, str],
    support_root: str,
    slices: Sequence[SupportSlice],
    checkpoint_npz_path: str,
    checkpoint_json_path: str,
    config_contract_hash: str,
    bank_lock_hash: str,
    source_stream_lock_hash: str,
    cache_binding_hash: str,
    partition_lock_hash: str,
    metadata_grid_hash: str,
) -> dict[str, object]:
    return {
            "schema_version": "midogpp_endpoint_router_feature_task_v1",
            "source_center": source_center,
            "training_seed": training_seed,
            "device": device,
            "expert_bank_root": expert_bank_root,
            "source_array_path": source_array_path,
            "source_block_ordinal_by_generation_seed": {
                str(key): int(blocks[key]) for key in GENERATION_SEEDS
            },
            "source_block_output_sha256_by_generation_seed": {
                str(key): block_hashes[key] for key in GENERATION_SEEDS
            },
            "support_root": support_root,
            "support_slices": [item.to_payload() for item in slices],
            "checkpoint_npz_path": checkpoint_npz_path,
            "checkpoint_json_path": checkpoint_json_path,
            "config_contract_hash": config_contract_hash,
            "bank_lock_hash": bank_lock_hash,
            "source_stream_lock_hash": source_stream_lock_hash,
            "cache_binding_hash": cache_binding_hash,
            "partition_lock_hash": partition_lock_hash,
            "metadata_grid_hash": metadata_grid_hash,
            "labels_available": False,
            "evaluation_embeddings_available": False,
        }


@dataclass(frozen=True)
class FeatureComponentRecord:
    query_center: str
    candidate_source: str
    training_seed: int
    relative_npz_path: str
    npz_sha256: str
    array_prefix: str
    support_row_count: int
    support_case_count: int
    support_partition_hash: str
    support_row_identity_hash: str
    center_partition_hash: str
    case_equal_energy: float
    linear_kernel_mmd2_by_generation_seed: Mapping[int, float]
    task_hash: str
    component_hash: str

    def __post_init__(self) -> None:
        mmd = {
            int(key): float(value)
            for key, value in self.linear_kernel_mmd2_by_generation_seed.items()
        }
        payload = self.unhashed_payload(mmd=mmd)
        if (
            self.query_center not in CENTERS
            or self.candidate_source not in candidate_sources(self.query_center)
            or self.training_seed not in TRAINING_SEEDS
            or Path(self.relative_npz_path).is_absolute()
            or ".." in Path(self.relative_npz_path).parts
            or not self.array_prefix
            or self.support_row_count <= 0
            or self.support_case_count != 8
            or set(mmd) != set(GENERATION_SEEDS)
            or any(not np.isfinite(value) or value < 0.0 for value in mmd.values())
            or not np.isfinite(float(self.case_equal_energy))
            or any(not _sha256(value) for value in (
                self.npz_sha256,
                self.support_partition_hash,
                self.support_row_identity_hash,
                self.center_partition_hash,
                self.task_hash,
            ))
            or self.component_hash != canonical_sha256(payload)
        ):
            raise ProtocolError("Endpoint-router feature component record drifted.")
        object.__setattr__(
            self, "linear_kernel_mmd2_by_generation_seed", MappingProxyType(mmd)
        )
        object.__setattr__(self, "case_equal_energy", float(self.case_equal_energy))

    @property
    def key(self) -> tuple[str, str, int]:
        return self.query_center, self.candidate_source, self.training_seed

    def unhashed_payload(
        self, *, mmd: Mapping[int, float] | None = None
    ) -> dict[str, object]:
        values = self.linear_kernel_mmd2_by_generation_seed if mmd is None else mmd
        return {
            "schema_version": "midogpp_endpoint_router_feature_component_v1",
            "query_center": self.query_center,
            "candidate_source": self.candidate_source,
            "training_seed": self.training_seed,
            "relative_npz_path": self.relative_npz_path,
            "npz_sha256": self.npz_sha256,
            "array_prefix": self.array_prefix,
            "support_row_count": self.support_row_count,
            "support_case_count": self.support_case_count,
            "support_partition_hash": self.support_partition_hash,
            "support_row_identity_hash": self.support_row_identity_hash,
            "center_partition_hash": self.center_partition_hash,
            "case_equal_energy": self.case_equal_energy,
            "linear_kernel_mmd2_by_generation_seed": {
                str(key): float(values[key]) for key in GENERATION_SEEDS
            },
            "task_hash": self.task_hash,
            "labels_used": False,
            "evaluation_embeddings_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self.unhashed_payload(), "component_hash": self.component_hash}


ComponentArrayLoader = Callable[
    [FeatureComponentRecord], tuple[Mapping[int, np.ndarray], Mapping[int, np.ndarray]]
]


@dataclass(frozen=True)
class SeedFeatureProduction:
    inner_rows: tuple[CandidateFeatureRow, ...]
    target_rows: tuple[CandidateFeatureRow, ...]
    component_records: tuple[FeatureComponentRecord, ...]
    feature_input_seal_hash: str
    production_hash: str

    def __post_init__(self) -> None:
        inner = tuple(self.inner_rows)
        target = tuple(self.target_rows)
        components = tuple(self.component_records)
        if (
            len(inner) != INNER_SEED_FEATURE_ROW_COUNT
            or len(target) != TARGET_SEED_FEATURE_ROW_COUNT
            or len(components) != FEATURE_COMPONENT_COUNT
            or len({row.row_key for row in inner}) != len(inner)
            or len({row.row_key for row in target}) != len(target)
            or tuple(row.row_key for row in inner) != _expected_inner_row_keys()
            or tuple(row.row_key for row in target) != _expected_target_row_keys()
            or tuple(record.key for record in components) != _expected_component_keys()
            or not _sha256(self.feature_input_seal_hash)
            or self.production_hash != canonical_sha256(self._unhashed_payload())
        ):
            raise ProtocolError("Endpoint-router seed feature production drifted.")

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_endpoint_router_seed_feature_production_v1",
            "feature_input_seal_hash": self.feature_input_seal_hash,
            "component_count": len(self.component_records),
            "component_hashes": [row.component_hash for row in self.component_records],
            "inner_row_count": len(self.inner_rows),
            "inner_row_hashes": [row.row_hash for row in self.inner_rows],
            "target_row_count": len(self.target_rows),
            "target_row_hashes": [row.row_hash for row in self.target_rows],
            "support_case_count_per_query": 8,
            "technical_seed_rows_are_independent_observations": False,
            "strict_H_q_e_exclusion": True,
            "labels_used": False,
            "evaluation_embeddings_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "production_hash": self.production_hash}

    def inner_table_rows(self) -> tuple[Mapping[str, object], ...]:
        return tuple({**row.to_payload(), "row_hash": row.row_hash} for row in self.inner_rows)

    def target_table_rows(self) -> tuple[Mapping[str, object], ...]:
        return tuple({**row.to_payload(), "row_hash": row.row_hash} for row in self.target_rows)


@dataclass(frozen=True)
class SupportShiftProduction:
    source_inner_by_candidate: Mapping[
        tuple[str, str, str], SupportActionProbabilityShift
    ]
    target_case_rows: tuple[TargetSupportActionShiftCase, ...]
    seed_feature_production_hash: str
    development_prediction_seal_hash: str
    target_prediction_store_hash: str
    partition_lock_hash: str
    production_hash: str

    def __post_init__(self) -> None:
        source = {
            tuple(map(str, key)): value
            for key, value in self.source_inner_by_candidate.items()
        }
        target = tuple(self.target_case_rows)
        expected_source_keys = tuple(
            (outer, query, candidate)
            for outer in CENTERS
            for query in candidate_sources(outer)
            for candidate in inner_candidate_sources(outer, query)
        )
        case_ids_by_target: dict[str, tuple[str, ...]] = {}
        target_geometry_valid = True
        for center in CENTERS:
            center_rows = tuple(row for row in target if row.target_id == center)
            first_source = candidate_sources(center)[0]
            case_ids = tuple(
                row.case_id for row in center_rows if row.candidate_source == first_source
            )
            case_ids_by_target[center] = case_ids
            if (
                len(case_ids) != 8
                or case_ids != tuple(sorted(case_ids))
                or tuple(
                    (row.candidate_source, row.case_id) for row in center_rows
                )
                != tuple(
                    (source_id, case_id)
                    for source_id in candidate_sources(center)
                    for case_id in case_ids
                )
            ):
                target_geometry_valid = False
        if (
            tuple(source) != expected_source_keys
            or len(source) != SOURCE_INNER_SHIFT_COUNT
            or any(not isinstance(value, SupportActionProbabilityShift) for value in source.values())
            or len(target) != TARGET_CASE_SHIFT_COUNT
            or not target_geometry_valid
            or len({(row.target_id, row.candidate_source, row.case_id) for row in target})
            != len(target)
            or any(not _sha256(value) for value in (
                self.seed_feature_production_hash,
                self.development_prediction_seal_hash,
                self.target_prediction_store_hash,
                self.partition_lock_hash,
            ))
            or self.production_hash != canonical_sha256(
                self._unhashed_payload(source=source, target=target)
            )
        ):
            raise ProtocolError("Endpoint-router support-shift production drifted.")
        object.__setattr__(self, "source_inner_by_candidate", MappingProxyType(source))
        object.__setattr__(self, "target_case_rows", target)

    def _unhashed_payload(
        self,
        *,
        source: Mapping[tuple[str, str, str], SupportActionProbabilityShift] | None = None,
        target: Sequence[TargetSupportActionShiftCase] | None = None,
    ) -> dict[str, object]:
        source_values = self.source_inner_by_candidate if source is None else source
        target_values = self.target_case_rows if target is None else target
        return {
            "schema_version": "midogpp_endpoint_router_support_shift_production_v1",
            "seed_feature_production_hash": self.seed_feature_production_hash,
            "development_prediction_seal_hash": self.development_prediction_seal_hash,
            "target_prediction_store_hash": self.target_prediction_store_hash,
            "partition_lock_hash": self.partition_lock_hash,
            "source_inner_shift_count": len(source_values),
            "source_inner_shift_hashes": [value.shift_hash for value in source_values.values()],
            "target_case_shift_count": len(target_values),
            "target_case_hashes": [value.case_hash for value in target_values],
            "ensemble_first": True,
            "technical_seed_values_may_feed_model": False,
            "labels_used": False,
            "evaluation_embeddings_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "production_hash": self.production_hash}

    def source_table_rows(self) -> tuple[Mapping[str, object], ...]:
        return tuple(
            {
                "schema_version": "midogpp_endpoint_router_source_inner_support_shift_row_v1",
                "outer_target_id": key[0],
                "query_id": key[1],
                "candidate_source": key[2],
                **value.to_payload(),
            }
            for key, value in self.source_inner_by_candidate.items()
        )

    def target_case_table_rows(self) -> tuple[Mapping[str, object], ...]:
        return tuple({**row.to_payload(), "case_hash": row.case_hash} for row in self.target_case_rows)


@dataclass(frozen=True)
class FeatureRuntimeProducts:
    seed_features: SeedFeatureProduction
    support_shifts: SupportShiftProduction
    runtime_seal_hash: str

    def __post_init__(self) -> None:
        if (
            self.support_shifts.seed_feature_production_hash
            != self.seed_features.production_hash
            or self.runtime_seal_hash != canonical_sha256(self._unhashed_payload())
        ):
            raise ProtocolError("Endpoint-router feature runtime seal drifted.")

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_endpoint_router_feature_runtime_seal_v1",
            "seed_feature_production_hash": self.seed_features.production_hash,
            "feature_input_seal_hash": self.seed_features.feature_input_seal_hash,
            "support_shift_production_hash": self.support_shifts.production_hash,
            "development_prediction_seal_hash": self.support_shifts.development_prediction_seal_hash,
            "target_prediction_store_hash": self.support_shifts.target_prediction_store_hash,
            "labels_used": False,
            "evaluation_embeddings_used": False,
            "target_prediction_seal_binds_this_runtime_later": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "runtime_seal_hash": self.runtime_seal_hash}


def _expected_component_keys() -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (query, source, training_seed)
        for query in CENTERS
        for source in candidate_sources(query)
        for training_seed in TRAINING_SEEDS
    )


def _expected_target_row_keys() -> tuple[tuple[str, str, str, int, int], ...]:
    return tuple(
        (target, target, source, training_seed, generation_seed)
        for target in CENTERS
        for source in candidate_sources(target)
        for training_seed, generation_seed in SEED_PAIRS
    )


def _expected_inner_row_keys() -> tuple[tuple[str, str, str, int, int], ...]:
    return tuple(
        (outer, query, source, training_seed, generation_seed)
        for outer in CENTERS
        for query in candidate_sources(outer)
        for source in inner_candidate_sources(outer, query)
        for training_seed, generation_seed in SEED_PAIRS
    )


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _stable_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 16
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = (
    "ComponentArrayLoader",
    "FEATURE_CHECKPOINT_DIRECTORY",
    "FEATURE_COMPONENT_COUNT",
    "FeatureComponentRecord",
    "FeatureRuntimeProducts",
    "FeatureTask",
    "INNER_SEED_FEATURE_ROW_COUNT",
    "SOURCE_INNER_SHIFT_COUNT",
    "SeedFeatureProduction",
    "SupportShiftProduction",
    "SupportSlice",
    "TARGET_CASE_SHIFT_COUNT",
    "TARGET_SEED_FEATURE_ROW_COUNT",
    "build_feature_task",
    "build_support_slice",
)
