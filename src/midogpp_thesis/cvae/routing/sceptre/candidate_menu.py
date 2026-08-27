"""Construction of target-excluded SCEPTRE source-family menus."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Iterable, Mapping

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    legal_routing_sources,
)
from ...generation.contracts import GenerationLock, SourceGenerationKey
from ...generation.generation import source_generation_plan
from ...protocol import ProtocolError
from .contracts import CandidateMenu, SourceFamily


def build_source_family(
    *,
    generation_lock: GenerationLock,
    target_center: str,
    source_center: str,
    stream_keys: Iterable[SourceGenerationKey],
) -> SourceFamily:
    """Bind one source-center candidate to its exact nine frozen streams."""

    return SourceFamily(
        target_center=str(target_center),
        source_center=str(source_center),
        generation_lock_hash=generation_lock.generation_lock_hash,
        stream_keys=tuple(stream_keys),
    )


def build_candidate_menu_from_keys(
    generation_lock: GenerationLock,
    target_center: str,
    stream_keys: Iterable[SourceGenerationKey],
) -> CandidateMenu:
    """Build a menu from an explicit candidate-only stream-key inventory.

    This entry point is intentionally strict: callers must remove the target's
    streams before admission.  That makes target leakage and seed-grid drift
    observable rather than silently filtering either condition.
    """

    target, expected_sources = _target_and_sources(target_center)
    _validate_locked_candidate_pool(generation_lock, target, expected_sources)
    rows = tuple(stream_keys)
    if not rows:
        raise ProtocolError("SCEPTRE candidate stream inventory is empty.")
    if any(not isinstance(row, SourceGenerationKey) for row in rows):
        raise ProtocolError("SCEPTRE candidate inventory requires source-generation keys.")
    if any(row.source_center == target for row in rows):
        raise ProtocolError("SCEPTRE candidate stream inventory includes the target expert.")
    if any(row.source_center not in expected_sources for row in rows):
        raise ProtocolError("SCEPTRE candidate stream inventory contains an unknown source.")
    if len({row.stream_id for row in rows}) != len(rows):
        raise ProtocolError("SCEPTRE candidate stream inventory contains duplicate stream ids.")

    grouped: dict[str, list[SourceGenerationKey]] = defaultdict(list)
    for row in rows:
        grouped[row.source_center].append(row)
    if set(grouped) != set(expected_sources):
        raise ProtocolError("SCEPTRE candidate source-family coverage is incomplete.")
    families = tuple(
        build_source_family(
            generation_lock=generation_lock,
            target_center=target,
            source_center=source,
            stream_keys=grouped[source],
        )
        for source in expected_sources
    )
    frozen_by_cell = {
        (row.source_center, row.training_seed, row.generation_seed): row
        for row in _locked_stream_plan(generation_lock)
        if row.source_center in set(expected_sources)
    }
    if len(frozen_by_cell) != len(rows):
        raise ProtocolError("SCEPTRE candidate stream inventory coverage drifted.")
    for family in families:
        for row in family.stream_keys:
            frozen = frozen_by_cell.get(
                (row.source_center, row.training_seed, row.generation_seed)
            )
            if frozen is None or frozen.to_payload() != row.to_payload():
                raise ProtocolError(
                    "SCEPTRE candidate stream identity differs from GenerationLock."
                )
    return CandidateMenu(
        target_center=target,
        generation_lock_hash=generation_lock.generation_lock_hash,
        families=families,
    )


def build_candidate_menu(
    generation_lock: GenerationLock,
    target_center: str,
) -> CandidateMenu:
    """Build the canonical eight-family menu from a GenerationLock."""

    target, canonical_sources = _target_and_sources(target_center)
    expected_sources = set(canonical_sources)
    keys = tuple(
        key
        for key in _locked_stream_plan(generation_lock)
        if key.source_center in expected_sources
    )
    return build_candidate_menu_from_keys(generation_lock, target, keys)


def _validate_locked_candidate_pool(
    generation_lock: GenerationLock,
    target: str,
    expected_sources: tuple[str, ...],
) -> None:
    payload = generation_lock.to_payload()
    bank = payload.get("bank")
    if not isinstance(bank, Mapping):
        raise ProtocolError("SCEPTRE GenerationLock lacks a bank payload.")
    raw_pools = bank.get("candidate_sources_by_target")
    if not isinstance(raw_pools, Mapping):
        raise ProtocolError("SCEPTRE GenerationLock lacks candidate pools.")
    raw_sources = raw_pools.get(target)
    if isinstance(raw_sources, (str, bytes)) or not isinstance(raw_sources, (list, tuple)):
        raise ProtocolError("SCEPTRE GenerationLock target candidate pool is invalid.")
    observed = tuple(str(source) for source in raw_sources)
    if observed != expected_sources:
        raise ProtocolError("SCEPTRE GenerationLock target candidate pool drifted.")


def _target_and_sources(target_center: object) -> tuple[str, tuple[str, ...]]:
    target = str(target_center)
    try:
        return target, legal_routing_sources(target)
    except ValueError as exc:
        raise ProtocolError("SCEPTRE target center is outside the promoted bank.") from exc


def _locked_stream_plan(
    generation_lock: GenerationLock,
) -> tuple[SourceGenerationKey, ...]:
    rows = source_generation_plan(generation_lock)
    expected_grid = set(product(CENTERS, TRAINING_SEEDS, GENERATION_SEEDS))
    observed_grid = {
        (row.source_center, row.training_seed, row.generation_seed) for row in rows
    }
    if (
        len(rows) != len(expected_grid)
        or observed_grid != expected_grid
        or len({row.stream_id for row in rows}) != len(rows)
    ):
        raise ProtocolError("SCEPTRE GenerationLock source-stream grid drifted.")
    return rows


# Descriptive aliases used by the diagnostic layer.
build_target_excluded_source_families = build_candidate_menu
bind_target_excluded_source_families = build_candidate_menu_from_keys


__all__ = (
    "bind_target_excluded_source_families",
    "build_candidate_menu",
    "build_candidate_menu_from_keys",
    "build_source_family",
    "build_target_excluded_source_families",
)
