"""Typed source-reference generation and target feature construction."""

from __future__ import annotations

import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from ..exact_tail_utility_surface.config import CLASSIFIER
from ..exact_tail_utility_surface.source_contracts import SourceFeatureInputs, SourceGenerationConfig
from ..exact_tail_utility_surface.source_generation import (
    GeneratedDevelopmentCache,
    load_component_arrays,
    materialize_generated_development_cache,
)
from ..metadata_compatibility import derive_compatibility_scores, derive_metadata_profiles
from ..residual_topup.hashing import canonical_sha256
from ..utility_aligned.target_features import (
    TargetCandidateComponents,
    TargetFeatureProduction,
    build_target_feature_production,
    target_sources,
)
from ..utility_aligned_identities import (
    CENTERS,
    METADATA_PROFILE_MEMBER,
    METADATA_PROFILE_SHA256,
)
from .config import TargetSupportSurfaceConfig
from .inputs import TargetSupportInputs, load_target_support_inputs


def build_all_target_features(
    config: TargetSupportSurfaceConfig,
) -> tuple[
    TargetSupportInputs,
    GeneratedDevelopmentCache,
    tuple[TargetFeatureProduction, ...],
]:
    execution_root = execution_root_for(config)
    inputs = load_target_support_inputs(config, execution_root=execution_root)
    partition_hashes = {
        center: canonical_sha256({
            "schema_version": "midogpp_target_support_partition_bridge_v1",
            "parent_reservation_hash": inputs.reservation_hash,
            "target": center,
            "case_ids": sorted(set(inputs.case_ids_by_target[center])),
        }) for center in CENTERS
    }
    generated = materialize_generated_development_cache(
        SourceGenerationConfig(config.expert_bank_root, config.generation_lock_root, CLASSIFIER),
        SourceFeatureInputs(inputs.support_array_path_by_target, inputs.case_ids_by_target, partition_hashes),
        root=execution_root / "generated_feature_reference",
        scratch_root=execution_root / "generation_scratch",
    )
    metadata = metadata_similarity(config)
    productions = tuple(
        build_for_target(target, inputs, generated, metadata[target], bootstrap_seed=60_920_000 + ordinal)
        for ordinal, target in enumerate(CENTERS)
    )
    return inputs, generated, productions


def build_for_target(
    target: str,
    inputs: TargetSupportInputs,
    generated: GeneratedDevelopmentCache,
    metadata: Mapping[str, float],
    *,
    bootstrap_seed: int,
) -> TargetFeatureProduction:
    support = np.load(inputs.support_array_path_by_target[target], mmap_mode="r", allow_pickle=False)
    raw_cases = inputs.case_ids_by_target[target]
    support_means = {
        case: np.mean(support[np.asarray([value == case for value in raw_cases])], axis=0, dtype=np.float64)
        for case in sorted(set(raw_cases))
    }
    components = {}
    for source in target_sources(target):
        reconstruction = {}; kl = {}; generated_means = {}
        for training_seed in (17, 42, 101):
            record = generated.component_by_key[(target, source, training_seed)]
            reconstruction[training_seed], kl[training_seed] = load_component_arrays(generated, record)
            for generation_seed in (17, 42, 101):
                source_record = generated.source_by_key[(source, training_seed, generation_seed)]
                array = np.load(generated.root / source_record.relative_path, mmap_mode="r", allow_pickle=False)
                generated_means[(training_seed, generation_seed)] = np.mean(array, axis=0, dtype=np.float64)
        components[source] = TargetCandidateComponents(
            candidate_source=source,
            reconstruction_by_training_seed=reconstruction,
            normalized_ps_kl_by_training_seed=kl,
            support_case_mean_embeddings=support_means,
            generated_mean_by_seed_pair=generated_means,
            metadata_similarity=metadata[source],
        )
    return build_target_feature_production(target_id=target, case_ids=raw_cases, components_by_source=components, bootstrap_seed=bootstrap_seed, bootstrap_replicate_count=32)


def metadata_similarity(config: TargetSupportSurfaceConfig) -> Mapping[str, Mapping[str, float]]:
    profiles = derive_metadata_profiles(
        config.metadata_profile_root / METADATA_PROFILE_MEMBER,
        expected_sha256=METADATA_PROFILE_SHA256,
    )
    result = {target: {} for target in CENTERS}
    for score in derive_compatibility_scores(profiles):
        result[score.target_center][score.source_center] = float(score.exact_match_count) / 3.0
    if any(tuple(result[target]) != target_sources(target) for target in CENTERS):
        raise ProtocolError("Target-support metadata candidate grid drifted.")
    return MappingProxyType({key: MappingProxyType(value) for key, value in result.items()})


def execution_root_for(config: TargetSupportSurfaceConfig) -> Path:
    local = Path("/data/local")
    root = local / "midogpp_target_support" / config.contract_hash if local.is_dir() and os.access(local, os.W_OK) else config.artifact_root.parent / ".midogpp_checkpoints" / "target_support_surface" / config.contract_hash
    root.mkdir(parents=True, exist_ok=True)
    try: root.resolve().relative_to(config.artifact_root.resolve())
    except ValueError: return root
    raise ProtocolError("Target-support execution scratch must stay outside the artifact.")


__all__ = ("build_all_target_features", "build_for_target", "execution_root_for")
