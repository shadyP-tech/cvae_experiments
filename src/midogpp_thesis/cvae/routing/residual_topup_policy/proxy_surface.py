"""Stable facade for fresh, label-blind Stage-60 proxy-surface scoring.

Implementation responsibilities live in small sibling modules; this facade
preserves the original import surface for runners, tests, and workstation jobs.
"""

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from .contracts import FreshProxyScoreRow
from .proxy_surface_checkpoints import load_fresh_proxy_score_checkpoint
from .proxy_surface_contracts import (
    ArrayLoader,
    CHECKPOINT_SCHEMA_VERSION,
    COMMON_FEATURE_DIM,
    DEFAULT_DEVICES,
    EXPECTED_EXPERT_TASK_COUNT,
    EXPECTED_QUERY_SHARD_COUNT,
    FRESH_SURFACE_ATTESTATION_SCHEMA_VERSION,
    PROXY_SCORE_COLUMNS,
    SCORE_CHUNK_ROWS,
    SHARD_SCHEMA_VERSION,
    TASK_SCHEMA_VERSION,
    FreshProxyScoreSurface,
    FreshProxyScoreTask,
    FreshProxyTaskResult,
    FreshQueryShard,
    MaterializedFreshProxyInputs,
    embedding_array_sha256,
    make_fresh_query_shard,
)
from .proxy_surface_materialization import materialize_fresh_proxy_inputs
from .proxy_surface_planning import build_fresh_proxy_score_tasks
from .proxy_surface_runtime import (
    TaskExecutor,
    TaskWorker,
    build_fresh_proxy_score_surface,
)
from .proxy_surface_validation import validate_fresh_proxy_score_surface
from .proxy_surface_worker import (
    CompatibilityScorer,
    ExpertLoader,
    execute_fresh_proxy_score_task,
)


materialize_fresh_proxy_score_surface = materialize_fresh_proxy_inputs


__all__ = (
    "CHECKPOINT_SCHEMA_VERSION",
    "COMMON_FEATURE_DIM",
    "DEFAULT_DEVICES",
    "EXPECTED_EXPERT_TASK_COUNT",
    "EXPECTED_QUERY_SHARD_COUNT",
    "FreshProxyScoreRow",
    "FreshProxyScoreSurface",
    "FreshProxyScoreTask",
    "FreshProxyTaskResult",
    "FreshQueryShard",
    "MaterializedFreshProxyInputs",
    "SCORE_CHUNK_ROWS",
    "build_fresh_proxy_score_surface",
    "build_fresh_proxy_score_tasks",
    "embedding_array_sha256",
    "execute_fresh_proxy_score_task",
    "load_fresh_proxy_score_checkpoint",
    "make_fresh_query_shard",
    "materialize_fresh_proxy_inputs",
    "materialize_fresh_proxy_score_surface",
    "validate_fresh_proxy_score_surface",
)
