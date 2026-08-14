"""Deterministic splitmix64 whole-candidate feature-block control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ...protocol import ProtocolError
from .constants import (
    CANDIDATE_FEATURE_PERMUTATION_ALGORITHM,
    CANDIDATE_FEATURE_PERMUTATION_SEED,
    DIRECTION_IDS,
    FEATURE_NAMES,
    candidate_sources,
)
from .correctness_products import LabelFreeDirectionalFeatures
from .hashing import canonical_hash
from .split_plans import WholeCaseLooPlan


_UINT64_MASK = (1 << 64) - 1


def _splitmix64(value: int) -> tuple[int, int]:
    state = (int(value) + 0x9E3779B97F4A7C15) & _UINT64_MASK
    mixed = state
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return state, (mixed ^ (mixed >> 31)) & _UINT64_MASK


def candidate_feature_permutation(
    target_center: object,
    case_id: object,
    direction: object,
    *,
    seed: int = CANDIDATE_FEATURE_PERMUTATION_SEED,
) -> Mapping[str, str]:
    target = str(target_center)
    case = str(case_id)
    direction_id = str(direction)
    if not case or direction_id not in DIRECTION_IDS:
        raise ProtocolError("OGDE feature permutation identity drifted.")
    sources = candidate_sources(target)
    identity = canonical_hash(
        {
            "algorithm": CANDIDATE_FEATURE_PERMUTATION_ALGORITHM,
            "seed": int(seed),
            "target_center": target,
            "case_id": case,
            "direction": direction_id,
            "candidate_sources": list(sources),
        }
    )
    state = (int(seed) ^ int(identity[:16], 16)) & _UINT64_MASK
    donors = list(sources)
    for index in range(len(donors) - 1, 0, -1):
        state, draw = _splitmix64(state)
        chosen = int(draw % (index + 1))
        donors[index], donors[chosen] = donors[chosen], donors[index]
    return {destination: donor for destination, donor in zip(sources, donors, strict=True)}


def permute_route_candidate_feature_blocks(
    features: Sequence[LabelFreeDirectionalFeatures],
    plan: WholeCaseLooPlan,
    *,
    seed: int = CANDIDATE_FEATURE_PERMUTATION_SEED,
) -> tuple[LabelFreeDirectionalFeatures, ...]:
    indexed = {row.key: row for row in features}
    if len(indexed) != len(features):
        raise ProtocolError("OGDE permutation input duplicated.")
    output: list[LabelFreeDirectionalFeatures] = []
    permutations = {
        direction: candidate_feature_permutation(
            plan.target_center, plan.case_id, direction, seed=seed
        )
        for direction in DIRECTION_IDS
    }
    # Preserve the selector's frozen case-major, source-major, DIRECTION_IDS
    # topology.  Lexical sorting is invalid because ``one_to_zero`` sorts
    # before the canonical first direction ``zero_to_one``.
    for case in (*plan.support_case_ids, plan.case_id):
        for destination in candidate_sources(plan.target_center):
            for direction in DIRECTION_IDS:
                permutation = permutations[direction]
                donor = permutation[destination]
                try:
                    block = indexed[(plan.target_center, case, donor, direction)]
                except KeyError as exc:
                    raise ProtocolError("OGDE permutation route feature surface is incomplete.") from exc
                output.append(
                    LabelFreeDirectionalFeatures(
                        plan.target_center,
                        case,
                        destination,
                        direction,
                        FEATURE_NAMES,
                        block.values,
                        block.directional_flip_count,
                        block.case_size,
                    )
                )
    return tuple(output)


__all__ = ("candidate_feature_permutation", "permute_route_candidate_feature_blocks")
