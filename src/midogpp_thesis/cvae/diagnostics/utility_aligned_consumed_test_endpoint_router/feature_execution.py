"""Default production runtime for the complete label-free feature grid."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
import sys
from typing import Callable, Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import array_sha256, canonical_sha256
from ...routing.utility_aligned import (
    INNER_ROLE,
    TARGET_ROLE,
    CandidateFeatureRow,
    build_case_bootstrap_plan,
)
from ...runtime.frozen_source_streams import FrozenSourceStreamCache
from .artifact_io import persist_or_validate_json, sha256_file
from .array_io import atomic_save_npy
from .contracts import (
    CENTERS,
    GENERATION_SEEDS,
    SEED_PAIRS,
    TRAINING_SEEDS,
    candidate_sources,
    inner_candidate_sources,
)
from .feature_checkpoint_store import load_component_arrays, load_feature_checkpoint
from .feature_runtime_contracts import (
    FEATURE_CHECKPOINT_DIRECTORY,
    FeatureComponentRecord,
    FeatureTask,
    SeedFeatureProduction,
    SupportSlice,
    build_feature_task,
    build_support_slice,
)
from .feature_worker import FEATURE_DEVICES, execute_feature_tasks
from .input_contracts import LabelFreeTestFrame, MetadataCompatibilityGrid, row_identity_hash
from .partitions import ConsumedTestPartitionSurface


FeatureTaskExecutor = Callable[[Sequence[FeatureTask]], object]


def materialize_label_free_seed_features(
    config: object,
    source_cache: FrozenSourceStreamCache,
    frame: LabelFreeTestFrame,
    partitions: ConsumedTestPartitionSurface,
    metadata: MetadataCompatibilityGrid,
    *,
    root: Path,
    task_executor: FeatureTaskExecutor | None = None,
    retain_checkpoints: bool = False,
) -> SeedFeatureProduction:
    """Compute 216 components once and expand them to 4,536+648 rows.

    This API intentionally has no label capability or evaluation-row input.
    It must run before the parent enters the CUDA-free classifier phase.
    """

    if not isinstance(retain_checkpoints, bool):
        raise ProtocolError("Endpoint-router feature retention flag is invalid.")
    _validate_inputs(config, source_cache, frame, partitions, metadata)
    checkpoint_root = Path(root) / FEATURE_CHECKPOINT_DIRECTORY
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    support_slices = _stage_support_arrays(
        frame, partitions, checkpoint_root=checkpoint_root
    )
    tasks = _build_tasks(
        config,
        source_cache,
        support_slices,
        frame=frame,
        partitions=partitions,
        metadata=metadata,
        checkpoint_root=checkpoint_root,
    )
    input_unhashed = {
        "schema_version": "midogpp_endpoint_router_feature_input_seal_v1",
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "expert_bank_lock_hash": str(getattr(config, "expected_bank_lock_hash")),
        "source_stream_lock_hash": source_cache.lock_hash,
        "cache_binding_hash": frame.cache_binding_hash,
        "partition_lock_hash": partitions.lock_hash,
        "metadata_grid_hash": metadata.grid_hash,
        "support_slice_hashes": [support_slices[center].slice_hash for center in CENTERS],
        "task_hashes": [task.task_hash for task in tasks],
        "feature_component_count": 216,
        "gpu_devices": list(FEATURE_DEVICES),
        "persistent_worker_count": 2,
        "multiprocessing_start_method": "spawn",
        "support_case_count_per_query": 8,
        "labels_used": False,
        "evaluation_embeddings_used": False,
    }
    input_seal = {
        **input_unhashed,
        "feature_input_seal_hash": canonical_sha256(input_unhashed),
    }
    persist_or_validate_json(checkpoint_root / "feature_input_seal.json", input_seal)

    completed: dict[tuple[str, str, int], FeatureComponentRecord] = {}
    pending: list[FeatureTask] = []
    for task in tasks:
        records = load_feature_checkpoint(task)
        if records is None:
            pending.append(task)
        else:
            completed.update((record.key, record) for record in records)
    if pending:
        executor = execute_feature_tasks if task_executor is None else task_executor
        executor(tuple(pending))
        for task in pending:
            records = load_feature_checkpoint(task, required=True)
            assert records is not None
            completed.update((record.key, record) for record in records)
            print(
                f"[endpoint-router:features] expert tasks {len(completed) // 8}/27",
                flush=True,
            )
    ordered = tuple(
        completed[(query, source, training_seed)]
        for query in CENTERS
        for source in candidate_sources(query)
        for training_seed in TRAINING_SEEDS
    )
    product = assemble_seed_feature_production(
        ordered,
        lambda record: load_component_arrays(checkpoint_root, record),
        support_case_ids_by_query={
            center: tuple(row.case_id for row in partitions.support_rows_by_center[center])
            for center in CENTERS
        },
        metadata_by_query=metadata.by_target,
        feature_input_seal_hash=str(input_seal["feature_input_seal_hash"]),
    )
    # The workstation runner retains these task-bound members until terminal
    # outputs have been persisted, allowing a crash in a later CPU phase to
    # resume without repeating the 216 GPU feature components.  Standalone
    # callers keep the historical closed-world default.
    if not retain_checkpoints:
        cleanup_feature_runtime_checkpoints(root)
    return product


def cleanup_feature_runtime_checkpoints(root: Path) -> None:
    """Delete only a complete, exact feature-checkpoint inventory."""

    base = Path(root)
    checkpoint_root = base / FEATURE_CHECKPOINT_DIRECTORY
    if not checkpoint_root.exists():
        return
    if (
        not base.is_absolute()
        or checkpoint_root.is_symlink()
        or not checkpoint_root.is_dir()
        or checkpoint_root.parent.name != "checkpoints"
        or checkpoint_root.name != "feature_runtime"
    ):
        raise ProtocolError("Endpoint-router feature checkpoint cleanup is unsafe.")
    expected = {
        "feature_input_seal.json",
        *(f"support_q{center}.npy" for center in CENTERS),
        *(
            f"feature_e{source}_train{training_seed}.{suffix}"
            for source in CENTERS
            for training_seed in TRAINING_SEEDS
            for suffix in ("json", "npz")
        ),
    }
    members = tuple(checkpoint_root.iterdir())
    observed = {path.name for path in members}
    if (
        not observed.issubset(expected)
        or any(path.is_symlink() or not path.is_file() for path in members)
    ):
        raise ProtocolError(
            "Endpoint-router feature checkpoint cleanup found an unowned inventory."
        )
    for name in sorted(observed):
        (checkpoint_root / name).unlink()
    checkpoint_root.rmdir()


def assemble_seed_feature_production(
    component_records: Sequence[FeatureComponentRecord],
    component_loader: Callable[
        [FeatureComponentRecord],
        tuple[Mapping[int, np.ndarray], Mapping[int, np.ndarray]],
    ],
    *,
    support_case_ids_by_query: Mapping[str, Sequence[str]],
    metadata_by_query: Mapping[str, Mapping[str, float]],
    feature_input_seal_hash: str,
) -> SeedFeatureProduction:
    """Pure deterministic assembly seam used by production and synthetic tests."""

    records = tuple(component_records)
    expected_keys = tuple(
        (query, source, training_seed)
        for query in CENTERS
        for source in candidate_sources(query)
        for training_seed in TRAINING_SEEDS
    )
    if tuple(record.key for record in records) != expected_keys:
        raise ProtocolError("Endpoint-router feature component coverage drifted.")
    cases = {
        str(query): tuple(map(str, values))
        for query, values in support_case_ids_by_query.items()
    }
    metadata = {
        str(query): {str(source): float(value) for source, value in values.items()}
        for query, values in metadata_by_query.items()
    }
    if (
        tuple(cases) != CENTERS
        or any(len(set(cases[query])) != 8 for query in CENTERS)
        or tuple(metadata) != CENTERS
        or any(tuple(metadata[query]) != candidate_sources(query) for query in CENTERS)
        or any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0
            for values in metadata.values()
            for value in values.values()
        )
    ):
        raise ProtocolError("Endpoint-router feature assembly inputs drifted.")
    by_key = {record.key: record for record in records}
    component_rows: dict[tuple[str, str, int, int], CandidateFeatureRow] = {}
    for query in CENTERS:
        for source in candidate_sources(query):
            replicas = tuple(by_key[(query, source, seed)] for seed in TRAINING_SEEDS)
            replica_energy = np.asarray(
                [record.case_equal_energy for record in replicas], dtype=np.float64
            )
            disagreement = float(np.std(replica_energy, ddof=0, dtype=np.float64))
            for record in replicas:
                if record.support_row_count != len(cases[query]):
                    raise ProtocolError("Endpoint-router component/support row alignment drifted.")
                reconstruction, kl = component_loader(record)
                reconstruction_stats = _case_equal_stats(
                    reconstruction, cases[query], role="reconstruction"
                )
                kl_stats = _case_equal_stats(kl, cases[query], role="KL")
                for generation_seed in GENERATION_SEEDS:
                    component_rows[
                        (query, source, record.training_seed, generation_seed)
                    ] = _candidate_row(
                        role=TARGET_ROLE,
                        outer=query,
                        query=query,
                        source=source,
                        training_seed=record.training_seed,
                        generation_seed=generation_seed,
                        support_hash=record.support_partition_hash,
                        reconstruction_stats=reconstruction_stats,
                        kl_stats=kl_stats,
                        disagreement=disagreement,
                        mmd=record.linear_kernel_mmd2_by_generation_seed[generation_seed],
                        metadata_similarity=metadata[query][source],
                    )
    target_rows = tuple(
        component_rows[(target, source, training_seed, generation_seed)]
        for target in CENTERS
        for source in candidate_sources(target)
        for training_seed, generation_seed in SEED_PAIRS
    )
    inner_rows = tuple(
        _as_inner(
            component_rows[(query, source, training_seed, generation_seed)],
            outer,
            support_hash=by_key[(query, source, training_seed)].support_row_identity_hash,
        )
        for outer in CENTERS
        for query in candidate_sources(outer)
        for source in inner_candidate_sources(outer, query)
        for training_seed, generation_seed in SEED_PAIRS
    )
    unhashed = {
        "schema_version": "midogpp_endpoint_router_seed_feature_production_v1",
        "feature_input_seal_hash": feature_input_seal_hash,
        "component_count": len(records),
        "component_hashes": [row.component_hash for row in records],
        "inner_row_count": len(inner_rows),
        "inner_row_hashes": [row.row_hash for row in inner_rows],
        "target_row_count": len(target_rows),
        "target_row_hashes": [row.row_hash for row in target_rows],
        "support_case_count_per_query": 8,
        "technical_seed_rows_are_independent_observations": False,
        "strict_H_q_e_exclusion": True,
        "labels_used": False,
        "evaluation_embeddings_used": False,
    }
    return SeedFeatureProduction(
        inner_rows=inner_rows,
        target_rows=target_rows,
        component_records=records,
        feature_input_seal_hash=feature_input_seal_hash,
        production_hash=canonical_sha256(unhashed),
    )


def _stage_support_arrays(
    frame: LabelFreeTestFrame,
    partitions: ConsumedTestPartitionSurface,
    *,
    checkpoint_root: Path,
) -> Mapping[str, SupportSlice]:
    slices: dict[str, SupportSlice] = {}
    for center in CENTERS:
        rows = partitions.support_rows_by_center[center]
        embeddings = frame.embeddings_for(rows)
        path = checkpoint_root / f"support_q{center}.npy"
        if path.is_file():
            observed = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                observed.shape != embeddings.shape
                or observed.dtype != np.float32
                or array_sha256(observed) != array_sha256(embeddings)
            ):
                raise ProtocolError("Endpoint-router staged support checkpoint drifted.")
        else:
            atomic_save_npy(path, embeddings)
        case_ids = tuple(row.case_id for row in rows)
        plan = build_case_bootstrap_plan(
            target_id=center,
            support_case_ids=tuple(sorted(set(case_ids))),
        )
        slices[center] = build_support_slice(
            query_center=center,
            relative_array_path=path.relative_to(checkpoint_root).as_posix(),
            array_sha256=sha256_file(path),
            case_ids=case_ids,
            row_identity_hash=row_identity_hash(rows),
            center_partition_hash=partitions.by_center[center].partition_hash,
            feature_support_partition_hash=plan.support_partition_hash,
        )
    return slices


def _build_tasks(
    config: object,
    source_cache: FrozenSourceStreamCache,
    support_slices: Mapping[str, SupportSlice],
    *,
    frame: LabelFreeTestFrame,
    partitions: ConsumedTestPartitionSurface,
    metadata: MetadataCompatibilityGrid,
    checkpoint_root: Path,
) -> tuple[FeatureTask, ...]:
    tasks: list[FeatureTask] = []
    for source in CENTERS:
        for training_seed in TRAINING_SEEDS:
            ordinal = len(tasks)
            stem = f"feature_e{source}_train{training_seed}"
            tasks.append(
                build_feature_task(
                    source_center=source,
                    training_seed=training_seed,
                    device=FEATURE_DEVICES[ordinal % 2],
                    expert_bank_root=str(Path(getattr(config, "expert_bank_root")).resolve()),
                    source_array_path=str(source_cache.source_array_path.resolve()),
                    source_block_ordinal_by_generation_seed={
                        generation_seed: source_cache.by_key[
                            (source, training_seed, generation_seed)
                        ].block_ordinal
                        for generation_seed in GENERATION_SEEDS
                    },
                    support_root=str(checkpoint_root.resolve()),
                    support_slices=tuple(
                        support_slices[query] for query in candidate_sources(source)
                    ),
                    checkpoint_npz_path=str((checkpoint_root / f"{stem}.npz").resolve()),
                    checkpoint_json_path=str((checkpoint_root / f"{stem}.json").resolve()),
                    config_contract_hash=str(getattr(config, "contract_hash")),
                    bank_lock_hash=str(getattr(config, "expected_bank_lock_hash")),
                    source_stream_lock_hash=source_cache.lock_hash,
                    cache_binding_hash=frame.cache_binding_hash,
                    partition_lock_hash=partitions.lock_hash,
                    metadata_grid_hash=metadata.grid_hash,
                )
            )
    return tuple(tasks)


def _validate_inputs(
    config: object,
    source_cache: FrozenSourceStreamCache,
    frame: LabelFreeTestFrame,
    partitions: ConsumedTestPartitionSurface,
    metadata: MetadataCompatibilityGrid,
) -> None:
    runtime = getattr(config, "runtime", None)
    torch_module = sys.modules.get("torch")
    if (
        not isinstance(source_cache, FrozenSourceStreamCache)
        or not isinstance(frame, LabelFreeTestFrame)
        or not isinstance(partitions, ConsumedTestPartitionSurface)
        or not isinstance(metadata, MetadataCompatibilityGrid)
        or not isinstance(runtime, Mapping)
        or source_cache.lock_payload.get("labels_consumed") is not False
        or frame.cache_binding.get("labels_persisted") is not False
        or frame.cache_binding.get("manifest_opened") is not False
        or (torch_module is not None and torch_module.cuda.is_initialized())
    ):
        raise ProtocolError("Endpoint-router feature runtime admission failed.")
    validate_feature_worker_topology(runtime)


def validate_feature_worker_topology(
    runtime: Mapping[str, object],
) -> tuple[str, str]:
    """Return the only workstation topology admitted for feature scoring."""

    if (
        tuple(runtime.get("generation_devices", ())) != FEATURE_DEVICES
        or int(runtime.get("generation_workers_per_device", -1)) != 1
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("parent_cuda_context_forbidden") is not True
        or runtime.get("tf32_enabled") is not False
        or runtime.get("amp_enabled") is not False
        or runtime.get("array_storage_dtype") != "float32"
        or runtime.get("scientific_reduction_dtype") != "float64"
    ):
        raise ProtocolError("Endpoint-router feature worker topology drifted.")
    return FEATURE_DEVICES


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
        support_case_count=8,
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


def _as_inner(
    row: CandidateFeatureRow, outer: str, *, support_hash: str
) -> CandidateFeatureRow:
    return _candidate_row(
        role=INNER_ROLE,
        outer=outer,
        query=row.query_id,
        source=row.candidate_source,
        training_seed=row.training_seed,
        generation_seed=row.generation_seed,
        support_hash=support_hash,
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
        raise ProtocolError(f"Endpoint-router {role} components require both hypotheses.")
    values = 0.5 * (
        np.asarray(per_class[0], dtype=np.float64)
        + np.asarray(per_class[1], dtype=np.float64)
    )
    cases = tuple(map(str, case_ids))
    if (
        values.shape != (len(cases),)
        or len(set(cases)) != 8
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
    ):
        raise ProtocolError(f"Endpoint-router {role} component geometry drifted.")
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
        float(np.std(case_values, ddof=0, dtype=np.float64)),
        float(quantiles[0]),
        float(quantiles[1]),
        float(quantiles[2]),
    )


# Stable orchestration imports: the runner can depend on one feature-runtime
# facade while the implementation remains split by responsibility.
from .support_shift_runtime import (  # noqa: E402
    combine_feature_runtime,
    materialize_label_free_support_shifts,
)


__all__ = (
    "FeatureTaskExecutor",
    "assemble_seed_feature_production",
    "cleanup_feature_runtime_checkpoints",
    "combine_feature_runtime",
    "materialize_label_free_seed_features",
    "materialize_label_free_support_shifts",
    "validate_feature_worker_topology",
)
