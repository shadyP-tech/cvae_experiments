"""Orchestration for the complete resumable exact-tail source cache."""

from __future__ import annotations

from pathlib import Path

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    GENERATION_SEEDS,
    SOURCE_PREFIX_ROWS_PER_CLASS,
    TRAINING_SEEDS,
)
from .production_inputs import PreparedDevelopmentInputs
from .runtime import GENERATION_DEVICES
from .source_checkpoint_store import (
    atomic_json,
    publish_component_record,
    publish_source_record,
)
from .source_contracts import (
    SOURCE_CACHE_SCHEMA,
    GeneratedDevelopmentCache,
    SourceFeatureInputs,
    SourceGenerationConfigProtocol,
)
from .source_gpu_worker import spawn_expert_tasks
from .source_planning import build_source_cache_plan


def materialize_generated_development_cache(
    config: SourceGenerationConfigProtocol,
    inputs: SourceFeatureInputs | PreparedDevelopmentInputs,
    *,
    root: Path,
    scratch_root: Path | None = None,
) -> GeneratedDevelopmentCache:
    """Generate/resume 81 streams and 216 q/e/replica component cells."""

    root.mkdir(parents=True, exist_ok=True)
    source_inputs = coerce_source_feature_inputs(inputs)
    plan = build_source_cache_plan(config, source_inputs, root=root)
    sources = dict(plan.existing_sources)
    components = dict(plan.existing_components)
    work_root = root if scratch_root is None else scratch_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    for payload in spawn_expert_tasks(plan.tasks, config.expert_bank_root, work_root):
        for raw in payload["sources"]:
            record = publish_source_record(
                raw, canonical_root=root, scratch_root=scratch_root
            )
            sources[record.key] = record
        for raw in payload["components"]:
            record = publish_component_record(
                raw, canonical_root=root, scratch_root=scratch_root
            )
            components[record.key] = record

    ordered_sources = tuple(
        sources[(center, training_seed, generation_seed)]
        for center in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )
    ordered_components = tuple(
        components[(query, source, training_seed)]
        for query in CENTERS
        for source in CENTERS
        if source != query
        for training_seed in TRAINING_SEEDS
    )
    if len(ordered_sources) != 81 or len(ordered_components) != 216:
        raise ProtocolError("Exact-tail generated development cache is incomplete.")
    unhashed = {
        "schema_version": SOURCE_CACHE_SCHEMA,
        "status": "COMPLETE",
        "generation_lock_hash": plan.lock.generation_lock_hash,
        "bank_lock_hash": plan.lock.bank_lock_hash,
        "source_records": [record.to_payload() for record in ordered_sources],
        "feature_component_records": [
            record.to_payload() for record in ordered_components
        ],
        "source_rows_per_class": SOURCE_PREFIX_ROWS_PER_CLASS,
        "generation_devices": list(GENERATION_DEVICES),
        "persistent_gpu_workers": 2,
        "tf32_enabled": False,
        "amp_enabled": False,
        "source_experts_updated": False,
        "support_labels_used": False,
        "scratch_authoritative": False,
    }
    cache_hash = stable_hash(unhashed)
    atomic_json(root / "source_cache.json", {**unhashed, "cache_hash": cache_hash})
    return GeneratedDevelopmentCache(
        root=root,
        generation_lock_hash=plan.lock.generation_lock_hash,
        bank_lock_hash=plan.lock.bank_lock_hash,
        source_records=ordered_sources,
        component_records=ordered_components,
        cache_hash=cache_hash,
    )


def coerce_source_feature_inputs(
    inputs: SourceFeatureInputs | PreparedDevelopmentInputs,
) -> SourceFeatureInputs:
    """Adapt the historical exact-tail input bundle to the neutral seam."""

    if isinstance(inputs, SourceFeatureInputs):
        return inputs
    if not isinstance(inputs, PreparedDevelopmentInputs):
        raise ProtocolError("Exact-tail source-feature inputs are not typed.")
    return SourceFeatureInputs(
        support_array_path_by_center=inputs.support_array_path_by_center,
        support_case_ids_by_center=inputs.support_case_ids_by_center,
        support_partition_hash_by_center={
            center: inputs.reservation.partitions[center].reservation_hash
            for center in CENTERS
        },
    )


__all__ = (
    "coerce_source_feature_inputs",
    "materialize_generated_development_cache",
)
