"""Validated lock loading and resumable exact-tail source task planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ...generation.config import load_generation_lock_config
from ...generation.contracts import (
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    GenerationLock,
)
from ...generation.runner import read_generation_lock
from ...generation.validation import validate_generation_bundle
from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    GENERATION_SEEDS,
    SOURCE_PREFIX_ROWS_PER_CLASS,
    TRAINING_SEEDS,
)
from .runtime import GENERATION_DEVICES
from .source_checkpoint_store import load_component_record, load_source_record
from .source_contracts import (
    ExpertTask,
    FeatureComponentRecord,
    SourceBlockRecord,
    SourceFeatureInputs,
    SourceGenerationConfigProtocol,
)


@dataclass(frozen=True)
class SourceCachePlan:
    lock: GenerationLock
    existing_sources: Mapping[tuple[str, int, int], SourceBlockRecord]
    existing_components: Mapping[tuple[str, str, int], FeatureComponentRecord]
    tasks: tuple[ExpertTask, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "existing_sources", MappingProxyType(dict(self.existing_sources))
        )
        object.__setattr__(
            self,
            "existing_components",
            MappingProxyType(dict(self.existing_components)),
        )


def load_validated_generation_lock(
    config: SourceGenerationConfigProtocol,
) -> GenerationLock:
    config_path = config.generation_lock_root / "config.resolved.yaml"
    lock_path = config.generation_lock_root / "manifests/generation_lock.json"
    if not config_path.is_file() or not lock_path.is_file():
        raise ProtocolError("Exact-tail validated GenerationLock is absent.")
    generation_config = load_generation_lock_config(config_path)
    if generation_config.bank_root.resolve() != config.expert_bank_root.resolve():
        raise ProtocolError("Exact-tail bank and GenerationLock roots disagree.")
    validate_generation_bundle(config.generation_lock_root, config=generation_config)
    lock = read_generation_lock(lock_path)
    if (
        lock.generation_lock_hash != EXPECTED_GENERATION_LOCK_HASH
        or lock.bank_lock_hash != EXPECTED_BANK_LOCK_HASH
        or generation_config.classifier != config.classifier
    ):
        raise ProtocolError("Exact-tail GenerationLock identity drifted.")
    return lock


def build_source_cache_plan(
    config: SourceGenerationConfigProtocol,
    inputs: SourceFeatureInputs,
    *,
    root: Path,
) -> SourceCachePlan:
    """Bind the canonical generation plan to valid resumable checkpoints."""

    # Importing the tensor-backed generator is deferred until execution planning;
    # module import and workstation preflight remain CUDA-free in the parent.
    from ...generation.generation import source_generation_plan

    lock = load_validated_generation_lock(config)
    keys = tuple(source_generation_plan(lock))
    expected_source_keys = {
        (center, training_seed, generation_seed)
        for center in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    }
    if (
        {(key.source_center, key.training_seed, key.generation_seed) for key in keys}
        != expected_source_keys
        or any(key.max_samples_per_class < SOURCE_PREFIX_ROWS_PER_CLASS for key in keys)
    ):
        raise ProtocolError("Exact-tail source generation plan drifted.")
    key_by_tuple = {
        (key.source_center, key.training_seed, key.generation_seed): key for key in keys
    }

    sources: dict[tuple[str, int, int], SourceBlockRecord] = {}
    components: dict[tuple[str, str, int], FeatureComponentRecord] = {}
    tasks: list[ExpertTask] = []
    ordinal = 0
    for source in CENTERS:
        for training_seed in TRAINING_SEEDS:
            existing_paths: dict[int, str] = {}
            missing_keys = []
            for generation_seed in GENERATION_SEEDS:
                key = key_by_tuple[(source, training_seed, generation_seed)]
                record = load_source_record(root, key)
                if record is None:
                    missing_keys.append(key)
                else:
                    sources[record.key] = record
                    existing_paths[generation_seed] = str(
                        (root.resolve() / record.relative_path).resolve()
                    )
            missing_queries: list[str] = []
            for query in CENTERS:
                if query == source:
                    continue
                record = load_component_record(
                    root,
                    query=query,
                    source=source,
                    training_seed=training_seed,
                    support_partition_hash=inputs.support_partition_hash_by_center[
                        query
                    ],
                )
                if record is None:
                    missing_queries.append(query)
                else:
                    components[record.key] = record
            if missing_keys or missing_queries:
                tasks.append(
                    ExpertTask(
                        source_center=source,
                        training_seed=training_seed,
                        generation_keys=tuple(missing_keys),
                        existing_source_path_by_generation_seed=MappingProxyType(
                            existing_paths
                        ),
                        query_centers=tuple(missing_queries),
                        support_array_path_by_center=MappingProxyType(
                            {
                                query: str(inputs.support_array_path_by_center[query])
                                for query in CENTERS
                                if query != source
                            }
                        ),
                        support_case_ids_by_center=MappingProxyType(
                            {
                                query: inputs.support_case_ids_by_center[query]
                                for query in CENTERS
                                if query != source
                            }
                        ),
                        support_partition_hash_by_center=MappingProxyType(
                            {
                                query: inputs.support_partition_hash_by_center[query]
                                for query in CENTERS
                                if query != source
                            }
                        ),
                        device=GENERATION_DEVICES[ordinal % len(GENERATION_DEVICES)],
                    )
                )
            ordinal += 1
    return SourceCachePlan(
        lock=lock,
        existing_sources=sources,
        existing_components=components,
        tasks=tuple(tasks),
    )


__all__ = (
    "SourceCachePlan",
    "build_source_cache_plan",
    "load_validated_generation_lock",
)
