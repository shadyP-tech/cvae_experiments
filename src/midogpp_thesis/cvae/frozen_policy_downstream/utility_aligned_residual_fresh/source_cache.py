"""Action-neutral source generation with fail-closed resume admission."""

from pathlib import Path

from ...generation import source_generation_plan
from ...protocol import ProtocolError
from ..residual_topup_fresh import source_cache as _shared

from ..residual_topup_fresh.source_cache import (
    EXPECTED_EXPERT_TASK_COUNT,
    EXPECTED_SOURCE_BLOCK_COUNT,
    FreshSourceCache,
    SourceBlockRecord,
    SourceExpertTask,
    SourceTaskExecutor,
    load_resumable_source_checkpoint,
    load_source_cache,
    load_validated_generation_lock,
)


def materialize_source_cache(config, generation_lock, *, root, **kwargs):
    """Reuse neutral generation while refusing invalid COMPLETE checkpoints."""

    cache_root = Path(root)
    lock_path = cache_root / "source_cache.json"
    if lock_path.exists():
        # A published lock claims full completion.  Never overwrite or repair
        # it in place; validation failure is evidence drift/tampering.
        _shared.load_source_cache(cache_root)
    for key in source_generation_plan(generation_lock):
        metadata = cache_root / f"metadata/{key.stream_id}.json"
        if metadata.is_file() and load_resumable_source_checkpoint(cache_root, key) is None:
            raise ProtocolError(
                "Utility-aligned source resume found an invalid published checkpoint."
            )
    return _shared.materialize_source_cache(
        config, generation_lock, root=cache_root, **kwargs
    )

__all__ = (
    "EXPECTED_EXPERT_TASK_COUNT",
    "EXPECTED_SOURCE_BLOCK_COUNT",
    "FreshSourceCache",
    "SourceBlockRecord",
    "SourceExpertTask",
    "SourceTaskExecutor",
    "load_source_cache",
    "load_validated_generation_lock",
    "materialize_source_cache",
)
