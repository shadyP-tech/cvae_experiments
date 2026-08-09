"""Deterministic candidate-label nulls blocked within support cases."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from ...protocol import ProtocolError
from .core_contracts import CaseActionUtility, CaseUtilitySurface, canonical_utility_rows
from .core_hashing import canonical_hash, require_sha256
from .partitions import CaseFold
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
        mapping = tuple(sorted((str(case), str(recipient), str(donor)) for case, recipient, donor in self.case_recipient_donor_actions))
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
        expected = canonical_hash(self._unhashed(mapping))
        if expected != self.permutation_hash:
            raise ProtocolError("Blocked candidate permutation hash drifted.")
        object.__setattr__(self, "case_recipient_donor_actions", mapping)

    def _unhashed(self, mapping=None) -> dict[str, object]:
        values = self.case_recipient_donor_actions if mapping is None else mapping
        return {
            "schema_version": "fixed_bank_label_aware_blocked_candidate_permutation_v1",
            "target_center": self.target_center,
            "fold_ordinal": self.fold_ordinal,
            "permutation_index": self.permutation_index,
            "permutation_seed": self.permutation_seed,
            "case_recipient_donor_actions": [list(row) for row in values],
            "blocking_unit": "candidate_source_labels_within_target_fold_support_case",
            "baseline_action_permuted": False,
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
    """Derange the eight candidate labels independently inside each support case."""

    if isinstance(permutation_index, bool) or not isinstance(permutation_index, int) or permutation_index < 0:
        raise ProtocolError("permutation_index must be a non-negative integer.")
    if isinstance(permutation_seed, bool) or not isinstance(permutation_seed, int):
        raise ProtocolError("permutation_seed must be an integer.")
    canonical_actions = candidate_actions(fold.target_center)
    mapping: list[tuple[str, str, str]] = []
    fold_seed = int.from_bytes(
        hashlib.sha256(
            f"{permutation_seed}::{fold.fold_id}".encode("utf-8")
        ).digest()[:8],
        "big",
    )
    rng = np.random.Generator(np.random.PCG64(fold_seed))
    shifts = rng.integers(
        1,
        len(canonical_actions),
        size=(permutation_index + 1, len(fold.support_case_ids)),
        dtype=np.int8,
    )[-1]
    for case_index, case_id in enumerate(fold.support_case_ids):
        ordered = tuple(
            sorted(
                canonical_actions,
                key=lambda action: (
                    hashlib.sha256(
                        f"{permutation_seed}::{fold.fold_id}::{case_id}::{action}".encode("utf-8")
                    ).hexdigest(),
                    action,
                ),
            )
        )
        shift = int(shifts[case_index])
        mapping.extend(
            (case_id, action, ordered[(index + shift) % len(ordered)])
            for index, action in enumerate(ordered)
        )
    canonical_mapping = tuple(sorted(mapping))
    unhashed = {
        "schema_version": "fixed_bank_label_aware_blocked_candidate_permutation_v1",
        "target_center": fold.target_center,
        "fold_ordinal": fold.fold_ordinal,
        "permutation_index": permutation_index,
        "permutation_seed": permutation_seed,
        "case_recipient_donor_actions": [list(row) for row in canonical_mapping],
        "blocking_unit": "candidate_source_labels_within_target_fold_support_case",
        "baseline_action_permuted": False,
        "evaluation_case_donor_used": False,
    }
    return BlockedSupportPermutation(
        target_center=fold.target_center,
        fold_ordinal=fold.fold_ordinal,
        permutation_index=permutation_index,
        permutation_seed=permutation_seed,
        case_recipient_donor_actions=canonical_mapping,
        permutation_hash=canonical_hash(unhashed),
    )


def permute_fold_support_utilities(
    surface: CaseUtilitySurface,
    fold: CaseFold,
    permutation: BlockedSupportPermutation,
) -> CaseUtilitySurface:
    """Break candidate identity while preserving each case's utility multiset and B."""

    if (
        permutation.target_center != fold.target_center
        or permutation.fold_ordinal != fold.fold_ordinal
        or set(surface.allowed_case_keys)
        != {(fold.target_center, case_id) for case_id in fold.support_case_ids}
        or not surface.label_scope.startswith(f"fold_local_support::{fold.fold_id}")
    ):
        raise ProtocolError("Blocked permutation received a mismatched support surface.")
    donor_action = {
        (case, recipient): donor
        for case, recipient, donor in permutation.case_recipient_donor_actions
    }
    source = surface.by_key()
    rows: list[CaseActionUtility] = []
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
            CaseActionUtility(
                target_center=row.target_center,
                case_id=row.case_id,
                action_id=row.action_id,
                sample_count=donor.sample_count,
                exact_bacc=donor.exact_bacc,
                smooth_bacc=donor.smooth_bacc,
                exact_gain_vs_b=donor.exact_gain_vs_b,
            )
        )
    canonical = canonical_utility_rows(rows)
    label_scope = (
        f"fold_local_support::{fold.fold_id}::blocked_candidate_permutation::"
        f"{permutation.permutation_index}"
    )
    exact_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_label_aware_case_utility_exact_v1",
            "label_scope": label_scope,
            "prerequisite_seal_hash": surface.prerequisite_seal_hash,
            "allowed_case_keys": [list(key) for key in surface.allowed_case_keys],
            "rows": [row.exact_payload() for row in canonical],
        }
    )
    return CaseUtilitySurface(
        rows=canonical,
        allowed_case_keys=surface.allowed_case_keys,
        label_scope=label_scope,
        prerequisite_seal_hash=surface.prerequisite_seal_hash,
        exact_surface_hash=exact_hash,
        descriptive_surface_hash=canonical_hash(
            {
                "exact_surface_hash": exact_hash,
                "smooth_bacc": [row.smooth_bacc for row in canonical],
            }
        ),
    )


__all__ = (
    "BlockedSupportPermutation",
    "build_blocked_support_permutation",
    "permute_fold_support_utilities",
)
