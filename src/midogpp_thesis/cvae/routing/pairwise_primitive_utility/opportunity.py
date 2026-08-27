"""Label-free no-op detection and deterministic action-surface collapse."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    ActionSurface,
    OpportunityMember,
    OpportunityCaseReceipt,
    OpportunitySet,
    canonical_sha256,
)
from .surface_hashing import crossing_hash, probability_hash


def _validate_baseline(values: Sequence[float]) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.float64)
    if (
        result.ndim != 1
        or result.size == 0
        or not np.isfinite(result).all()
        or np.any(result < 0.0)
        or np.any(result > 1.0)
    ):
        raise ProtocolError("Protected P must be a finite non-empty probability vector.")
    return result


def build_opportunity_set(
    protected_p: Sequence[float],
    actions: Sequence[ActionSurface],
    *,
    candidate_action_ids: Sequence[str],
) -> OpportunitySet:
    """Collapse label-free duplicate opportunities and exclude exact P no-ops.

    Two active actions share an equivalence class only when their projected
    probability surface is exactly identical (and therefore its crossing mask
    is identical too).  A crossing-mask match alone is not equivalence because
    proper-score safety can differ.  The lexicographically smallest action id
    is the class representative.

    Every supplied action remains in ``members`` for audit.  An action is a
    structural no-op whenever it changes no downstream threshold decision,
    even if its raw probability bytes differ from P; such rows have no
    representative and can never enter fitting or ranking.  Exact probability
    equality is retained as a separate audit flag.
    """

    baseline = _validate_baseline(protected_p)
    rows = tuple(sorted(tuple(actions), key=lambda row: row.action_id))
    frozen_inventory = tuple(sorted(str(value).strip() for value in candidate_action_ids))
    if (
        not rows
        or any(not value for value in frozen_inventory)
        or len(set(frozen_inventory)) != len(frozen_inventory)
        or len({row.action_id for row in rows}) != len(rows)
        or tuple(row.action_id for row in rows) != frozen_inventory
    ):
        raise ProtocolError("Opportunity construction requires unique candidate actions.")
    if any(len(row.probabilities) != len(baseline) for row in rows):
        raise ProtocolError("Candidate and protected probability surfaces are misaligned.")

    probability_hash_by_id: dict[str, str] = {}
    crossing_hash_by_id: dict[str, str] = {}
    noop_by_id: dict[str, bool] = {}
    exact_p_by_id: dict[str, bool] = {}
    for row in rows:
        candidate = np.asarray(row.probabilities, dtype=np.float64)
        probability_hash_by_id[row.action_id] = probability_hash(candidate)
        crossing_hash_by_id[row.action_id] = crossing_hash(baseline, candidate)
        noop_by_id[row.action_id] = bool(
            np.array_equal(candidate >= 0.5, baseline >= 0.5)
        )
        exact_p_by_id[row.action_id] = bool(np.array_equal(candidate, baseline))

    active_ids = tuple(row.action_id for row in rows if not noop_by_id[row.action_id])
    parent = {action_id: action_id for action_id in active_ids}

    def find(action_id: str) -> str:
        root = action_id
        while parent[root] != root:
            root = parent[root]
        while parent[action_id] != action_id:
            predecessor = parent[action_id]
            parent[action_id] = root
            action_id = predecessor
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    by_probability: dict[str, list[str]] = defaultdict(list)
    for action_id in active_ids:
        by_probability[probability_hash_by_id[action_id]].append(action_id)
    for group in by_probability.values():
        head = min(group)
        for action_id in group:
            union(head, action_id)

    class_members: dict[str, list[str]] = defaultdict(list)
    for action_id in active_ids:
        class_members[find(action_id)].append(action_id)
    representative_by_id = {
        action_id: min(group)
        for group in class_members.values()
        for action_id in group
    }
    members = tuple(
        OpportunityMember(
            action_id=row.action_id,
            family=row.family,
            direction=row.direction,
            probability_hash=probability_hash_by_id[row.action_id],
            crossing_hash=crossing_hash_by_id[row.action_id],
            structural_noop=noop_by_id[row.action_id],
            exact_p_probability=exact_p_by_id[row.action_id],
            representative_action_id=(
                None if noop_by_id[row.action_id] else representative_by_id[row.action_id]
            ),
        )
        for row in rows
    )
    representatives = tuple(
        sorted({member.representative_action_id for member in members if member.representative_action_id})
    )
    baseline_hash = probability_hash(baseline)
    payload = {
        "schema": "pairwise_primitive_opportunity_set_v2",
        "baseline_hash": baseline_hash,
        "candidate_action_ids": frozen_inventory,
        "members": tuple(
            {
                "action_id": member.action_id,
                "family": member.family,
                "direction": member.direction,
                "probability_hash": member.probability_hash,
                "crossing_hash": member.crossing_hash,
                "structural_noop": member.structural_noop,
                "exact_p_probability": member.exact_p_probability,
                "representative_action_id": member.representative_action_id,
            }
            for member in members
        ),
        "active_representatives": representatives,
        "labels_used": False,
    }
    return OpportunitySet(
        baseline_hash=baseline_hash,
        candidate_action_ids=frozen_inventory,
        members=members,
        active_representative_ids=representatives,
        opportunity_hash=canonical_sha256(payload),
    )


def build_opportunity_case_receipt(
    *, center_id: object, case_id: object, opportunity: OpportunitySet
) -> OpportunityCaseReceipt:
    if not isinstance(opportunity, OpportunitySet):
        raise ProtocolError("Opportunity receipt requires a typed opportunity set.")
    return OpportunityCaseReceipt(
        center_id=str(center_id),
        case_id=str(case_id),
        opportunity=opportunity,
    )


__all__ = ("build_opportunity_case_receipt", "build_opportunity_set")
