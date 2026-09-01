"""Role-complete candidate-pool construction."""

from __future__ import annotations

from typing import Sequence

from .contracts import CandidatePoolReceipt


def build_source_candidate_pool(
    *,
    outer_target_id: str,
    pseudo_query_id: str,
    all_center_ids: Sequence[str],
    bank_lock_hash: str,
) -> CandidatePoolReceipt:
    """Seal the exact source-development pool ``C \\ {H, q}``."""

    centers = tuple(sorted(str(value) for value in all_center_ids))
    candidates = tuple(
        value for value in centers if value not in {str(outer_target_id), str(pseudo_query_id)}
    )
    return CandidatePoolReceipt(
        outer_target_id=str(outer_target_id),
        query_center_id=str(pseudo_query_id),
        all_center_ids=centers,
        candidate_center_ids=candidates,
        bank_lock_hash=bank_lock_hash,
    )


def build_target_candidate_pool(
    *,
    outer_target_id: str,
    all_center_ids: Sequence[str],
    bank_lock_hash: str,
) -> CandidatePoolReceipt:
    """Seal the exact target pool ``C \\ {H}``."""

    centers = tuple(sorted(str(value) for value in all_center_ids))
    candidates = tuple(value for value in centers if value != str(outer_target_id))
    return CandidatePoolReceipt(
        outer_target_id=str(outer_target_id),
        query_center_id=str(outer_target_id),
        all_center_ids=centers,
        candidate_center_ids=candidates,
        bank_lock_hash=bank_lock_hash,
    )


__all__ = ("build_source_candidate_pool", "build_target_candidate_pool")
