"""Blocked deterministic whole-case feature derangement for F_P."""

from __future__ import annotations

import hashlib
import random
from typing import Sequence

from ...protocol import ProtocolError
from .contracts import DonorRow
from .model import fit_two_head_ridge


def blocked_case_derangement(
    rows: Sequence[DonorRow],
    *,
    seed: int,
) -> tuple[DonorRow, ...]:
    """Derange complete feature cases within (H,q), preserving labeled targets."""

    records = tuple(rows)
    if not records:
        raise ProtocolError("Feature derangement requires donor rows.")
    blocks: dict[tuple[str, str], list[DonorRow]] = {}
    for row in records:
        blocks.setdefault((row.model_target, row.query_center), []).append(row)
    result: list[DonorRow] = []
    for block_key in sorted(blocks):
        block = blocks[block_key]
        by_case: dict[str, dict[tuple[str, str], DonorRow]] = {}
        for row in block:
            identity = (row.action_id, row.candidate_source)
            if identity in by_case.setdefault(row.case_id, {}):
                raise ProtocolError("Permutation block contains duplicate case/action rows.")
            by_case[row.case_id][identity] = row
        cases = sorted(by_case)
        if len(cases) < 2:
            raise ProtocolError("Every permutation block requires at least two whole cases.")
        schema = set(by_case[cases[0]])
        if any(set(by_case[case]) != schema for case in cases):
            raise ProtocolError("Whole-case permutation blocks must have aligned action schemas.")
        rng_seed = int.from_bytes(
            hashlib.sha256(f"{int(seed)}::{block_key[0]}::{block_key[1]}".encode()).digest()[:8],
            "big",
        )
        shuffled = cases[:]
        random.Random(rng_seed).shuffle(shuffled)
        # Convert any fixed points into one deterministic cycle.
        if any(left == right for left, right in zip(cases, shuffled, strict=True)):
            offset = 1 + rng_seed % (len(cases) - 1)
            shuffled = cases[offset:] + cases[:offset]
        for recipient_case, donor_case in zip(cases, shuffled, strict=True):
            for identity in sorted(schema):
                recipient = by_case[recipient_case][identity]
                feature_donor = by_case[donor_case][identity]
                result.append(
                    DonorRow(
                        model_target=recipient.model_target,
                        query_center=recipient.query_center,
                        candidate_source=recipient.candidate_source,
                        case_id=recipient.case_id,
                        action_id=recipient.action_id,
                        feature_case_id=feature_donor.case_id,
                        feature_names=feature_donor.feature_names,
                        values=feature_donor.values,
                        target=recipient.target,
                    )
                )
    return tuple(sorted(result))


def refit_blocked_permutation_control(
    rows: Sequence[DonorRow],
    *,
    heldout_h: str,
    seed: int,
):
    """Derange then refit the exact same two-head capacity used by F_S."""

    return fit_two_head_ridge(
        blocked_case_derangement(rows, seed=seed),
        heldout_h=heldout_h,
        alpha=1.0,
        variance_floor=1.0e-6,
    )


__all__ = ("blocked_case_derangement", "refit_blocked_permutation_control")
