"""Public compatibility facade for SCEPTRE v5 physical source streams.

The implementation is split by responsibility across source contracts,
planning, checkpoint IO, GPU dispatch, materialization, and sealed-store
validation. This facade preserves the original import surface while keeping
the production data path label-free: it never opens target caches, manifests,
or outcome surfaces.
"""

from __future__ import annotations

from .gpu_dispatch import (
    execute_gpu_tasks as _execute_gpu_tasks,
    execute_injected_task as _execute_injected_task,
    production_generation_worker as _production_generation_worker,
)
from .source_checkpoints import (
    load_checkpoint_if_complete as _load_checkpoint_if_complete,
    persist_exact_json as _persist_exact_json,
    persist_exact_npy as _persist_exact_npy,
    publish_checkpoint as _publish_checkpoint,
    publish_source_array as _publish_source_array,
    validate_checkpoint_directory as _validate_checkpoint_directory,
)
from .source_contracts import (
    CHECKPOINT_DIRECTORY,
    PRODUCTION_SOURCE_GEOMETRY,
    SOURCE_ARRAY_MEMBER,
    SOURCE_INDEX_MEMBER,
    SOURCE_RECEIPT_MEMBER,
    SourceGeometry,
    SourceRuntimeConfig,
    SourceRuntimeTestMode,
    SourceStreamRecord,
    SourceStreamStore,
)
from .source_hashing import (
    block_bundle_sha256 as _block_bundle_sha256,
    canonical_sha256 as _canonical_sha256,
)
from .source_materialization import materialize_source_streams
from .source_planning import (
    assert_owned_root as _assert_owned_root,
    assert_parent_cuda_free as _assert_parent_cuda_free,
    assert_production_runtime as _assert_production_runtime,
    build_tasks as _build_tasks,
    config_hash as _config_hash,
    final_paths as _final_paths,
    generation_keys as _generation_keys,
    geometry_for as _geometry,
    resolve_attempt_id as _attempt_id,
    resolve_expert_bank_root as _resolve_expert_bank_root,
    task_identity as _task_identity,
    task_key as _task_key,
    validate_generation_grid as _validate_generation_grid,
)
from .source_store import load_source_streams
from .worker_runtime import GPU_DEVICES


__all__ = (
    "CHECKPOINT_DIRECTORY",
    "GPU_DEVICES",
    "PRODUCTION_SOURCE_GEOMETRY",
    "SOURCE_ARRAY_MEMBER",
    "SOURCE_INDEX_MEMBER",
    "SOURCE_RECEIPT_MEMBER",
    "SourceGeometry",
    "SourceRuntimeConfig",
    "SourceRuntimeTestMode",
    "SourceStreamRecord",
    "SourceStreamStore",
    "load_source_streams",
    "materialize_source_streams",
)
