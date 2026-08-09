"""Within-support-case derangements of complete candidate statistic blocks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from ...protocol import ProtocolError
from .case_partitions import CaseFold
from .core_contracts import (
    CaseActionSufficientStatistics,
    SufficientStatisticSurface,
    make_statistics_surface,
)
from .core_hashing import canonical_hash, require_sha256
from .scientific_constants import BASELINE_ACTION_ID, candidate_actions


@dataclass(frozen=True)
class BlockedSupportPermutation:
    target_center: str
    fold_ordinal: int
    permutation_index: int
    permutation_seed: int
    case_recipient_donor_actions: tuple[tuple[str, str, str], ...]
    permutation_hash: str

    def __post_init__(self) -> None:
        mapping = tuple(
            sorted(
                (str(case), str(recipient), str(donor))
                for case, recipient, donor in self.case_recipient_donor_actions
            )
        )
        if (
            isinstance(self.permutation_index, bool)
            or not isinstance(self.permutation_index, int)
            or self.permutation_index < 0
            or isinstance(self.permutation_seed, bool)
            or not isinstance(self.permutation_seed, int)
            or not mapping
        ):
            raise ProtocolError("Blocked candidate permutation identity is malformed.")
        candidates = set(candidate_actions(self.target_center))
        cases = {case for case, _, _ in mapping}
        for case in cases:
            case_rows = tuple(row for row in mapping if row[0] == case)
            recipients = {recipient for _, recipient, _ in case_rows}
            donors = {donor for _, _, donor in case_rows}
            if (
                len(case_rows) != len(candidates)
                or recipients != candidates
                or donors != candidates
                or any(recipient == donor for _, recipient, donor in case_rows)
                or any(BASELINE_ACTION_ID in (recipient, donor) for _, recipient, donor in case_rows)
            ):
                raise ProtocolError("Each support case needs an eight-action candidate derangement.")
        require_sha256(self.permutation_hash, "permutation_hash")
        if canonical_hash(self._unhashed(mapping)) != self.permutation_hash:
            raise ProtocolError("Blocked candidate permutation hash drifted.")
        object.__setattr__(self, "case_recipient_donor_actions", mapping)

    def _unhashed(
        self, mapping: tuple[tuple[str, str, str], ...] | None = None
    ) -> dict[str, object]:
        values = self.case_recipient_donor_actions if mapping is None else mapping
        return {
            "schema_version": "fixed_bank_pooled_bacc_blocked_candidate_permutation_v2",
            "target_center": self.target_center,
            "fold_ordinal": self.fold_ordinal,
            "permutation_index": self.permutation_index,
            "permutation_seed": self.permutation_seed,
            "case_recipient_donor_actions": [list(row) for row in values],
            "blocking_unit": "full_candidate_sufficient_statistic_block_within_support_case",
            "baseline_action_permuted": False,
            "candidate_multiset_preserved_per_case": True,
            "evaluation_case_donor_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "permutation_hash": self.permutation_hash}


def build_blocked_support_permutation(
    fold: CaseFold,
    *,
    permutation_index: int,
    permutation_seed: int,
) -> BlockedSupportPermutation:
    if isinstance(permutation_index, bool) or not isinstance(permutation_index, int) or permutation_index < 0:
        raise ProtocolError("permutation_index must be a non-negative integer.")
    if isinstance(permutation_seed, bool) or not isinstance(permutation_seed, int):
        raise ProtocolError("permutation_seed must be an integer.")
    candidates = candidate_actions(fold.target_center)
    mapping: list[tuple[str, str, str]] = []
    for case_id in fold.support_case_ids:
        ordered = _case_candidate_order(
            candidates,
            permutation_seed=permutation_seed,
            fold_id=fold.fold_id,
            case_id=case_id,
        )
        shift = _case_shift(
            permutation_seed=permutation_seed,
            fold_id=fold.fold_id,
            permutation_index=permutation_index,
            case_id=case_id,
        )
        mapping.extend(
            (case_id, recipient, ordered[(index + shift) % len(ordered)])
            for index, recipient in enumerate(ordered)
        )
    canonical = tuple(sorted(mapping))
    payload = {
        "schema_version": "fixed_bank_pooled_bacc_blocked_candidate_permutation_v2",
        "target_center": fold.target_center,
        "fold_ordinal": fold.fold_ordinal,
        "permutation_index": permutation_index,
        "permutation_seed": permutation_seed,
        "case_recipient_donor_actions": [list(row) for row in canonical],
        "blocking_unit": "full_candidate_sufficient_statistic_block_within_support_case",
        "baseline_action_permuted": False,
        "candidate_multiset_preserved_per_case": True,
        "evaluation_case_donor_used": False,
    }
    return BlockedSupportPermutation(
        target_center=fold.target_center,
        fold_ordinal=fold.fold_ordinal,
        permutation_index=permutation_index,
        permutation_seed=permutation_seed,
        case_recipient_donor_actions=canonical,
        permutation_hash=canonical_hash(payload),
    )


def permute_fold_support_statistics(
    surface: SufficientStatisticSurface,
    fold: CaseFold,
    permutation: BlockedSupportPermutation,
) -> SufficientStatisticSurface:
    if (
        permutation.target_center != fold.target_center
        or permutation.fold_ordinal != fold.fold_ordinal
        or set(surface.allowed_case_keys)
        != {(fold.target_center, case) for case in fold.support_case_ids}
        or not surface.label_scope.startswith(f"fold_local_support::{fold.fold_id}")
    ):
        raise ProtocolError("Blocked permutation received a mismatched support surface.")
    donor_action = {
        (case, recipient): donor
        for case, recipient, donor in permutation.case_recipient_donor_actions
    }
    source = surface.by_key()
    rows: list[CaseActionSufficientStatistics] = []
    for row in surface.rows:
        donor = (
            row
            if row.action_id == BASELINE_ACTION_ID
            else source[
                (
                    row.target_center,
                    row.case_id,
                    donor_action[(row.case_id, row.action_id)],
                )
            ]
        )
        rows.append(
            CaseActionSufficientStatistics(
                target_center=row.target_center,
                case_id=row.case_id,
                action_id=row.action_id,
                n_positive=donor.n_positive,
                true_positive=donor.true_positive,
                n_negative=donor.n_negative,
                true_negative=donor.true_negative,
            )
        )
    return make_statistics_surface(
        rows,
        allowed_case_keys=surface.allowed_case_keys,
        label_scope=(
            f"fold_local_support::{fold.fold_id}::blocked_candidate_permutation::"
            f"{permutation.permutation_index}"
        ),
        prerequisite_seal_hash=surface.prerequisite_seal_hash,
    )


# Orchestration-friendly alias.
permute_fold_support_utilities = permute_fold_support_statistics


def _case_candidate_order(
    candidates: tuple[str, ...],
    *,
    permutation_seed: int,
    fold_id: str,
    case_id: str,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            candidates,
            key=lambda action: (
                hashlib.sha256(
                    f"{permutation_seed}::{fold_id}::{case_id}::{action}".encode()
                ).hexdigest(),
                action,
            ),
        )
    )


def _case_shift(
    *, permutation_seed: int, fold_id: str, permutation_index: int, case_id: str
) -> int:
    """O(1) counter-based shift shared by standalone and vector null paths."""

    base = _case_shift_base(
        permutation_seed=permutation_seed, fold_id=fold_id, case_id=case_id
    )
    return 1 + _splitmix64(
        base + (permutation_index + 1) * 0x9E3779B97F4A7C15
    ) % 7


def _case_shift_base(*, permutation_seed: int, fold_id: str, case_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(
            f"{permutation_seed}::{fold_id}::{case_id}::shift".encode()
        ).digest()[:8],
        "big",
    )


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value &= mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return value & mask


__all__ = (
    "BlockedSupportPermutation",
    "build_blocked_support_permutation",
    "permute_fold_support_statistics",
    "permute_fold_support_utilities",
)
