"""Fold-local normalization of primitive Train-H action outcomes.

Class-presence and class-specific recall deltas are case-local.  The weights
that turn them into the case-equal BACC contribution are deliberately rebuilt
from the training cases of each fold; a held-out case can therefore never
change another case's training target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ...protocol import ProtocolError
from .contracts import (
    LabelFreeCaseMenu,
    SupportActionOutcome,
    SupportCaseClassProfile,
)
from .hashing import canonical_hash


@dataclass(frozen=True, slots=True)
class SupportFoldNormalizer:
    """Hash-bound case-equal BACC normalization for one exact fit inventory."""

    outer_target_id: str
    case_ids: tuple[str, ...]
    profile_hashes: tuple[tuple[str, str], ...]
    class_support_counts: tuple[int, int]
    normalizer_hash: str = field(init=False)

    def __post_init__(self) -> None:
        cases = tuple(sorted(self.case_ids))
        profiles = tuple(sorted(self.profile_hashes))
        counts = tuple(self.class_support_counts)
        if (
            not self.outer_target_id
            or not cases
            or len(cases) != len(set(cases))
            or tuple(case for case, _ in profiles) != cases
            or len(counts) != 2
            or any(type(value) is not int or value < 1 or value > len(cases) for value in counts)
            or any(type(value) is not str or len(value) != 64 for _, value in profiles)
        ):
            raise ProtocolError("HARP v15 support-fold normalizer is malformed.")
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "profile_hashes", profiles)
        object.__setattr__(self, "class_support_counts", counts)
        object.__setattr__(
            self,
            "normalizer_hash",
            canonical_hash(
                {
                    "schema_version": "hierarchical_support_fold_normalizer_v15",
                    "outer_target_id": self.outer_target_id,
                    "case_ids": cases,
                    "profile_hashes": profiles,
                    "class_support_counts": counts,
                    "heldout_case_excluded_before_fit": True,
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    def normalize(self, outcome: SupportActionOutcome) -> SupportActionOutcome:
        return outcome.with_fold_normalization(
            case_count=len(self.case_ids),
            class_support_counts=self.class_support_counts,
            normalization_hash=self.normalizer_hash,
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "hierarchical_support_fold_normalizer_v15",
            "outer_target_id": self.outer_target_id,
            "case_ids": list(self.case_ids),
            "profile_hashes": [
                {"case_id": case, "profile_hash": value}
                for case, value in self.profile_hashes
            ],
            "class_support_counts": list(self.class_support_counts),
            "normalizer_hash": self.normalizer_hash,
            "evaluation_labels_consumed": False,
        }


def validate_support_case_profiles(
    menus: Sequence[LabelFreeCaseMenu],
    profiles: Sequence[SupportCaseClassProfile],
    *,
    require_complete: bool,
) -> tuple[SupportCaseClassProfile, ...]:
    menu_rows = tuple(menus)
    rows = tuple(sorted(profiles, key=lambda row: row.case_id))
    if not rows and not require_complete:
        return ()
    if (
        len(rows) != len(menu_rows)
        or len({row.case_id for row in rows}) != len(rows)
        or {row.case_id for row in rows} != {row.case_id for row in menu_rows}
        or any(
            row.outer_target_id != menu.outer_target_id
            for row in rows
            for menu in menu_rows
            if row.case_id == menu.case_id
        )
    ):
        raise ProtocolError(
            "HARP v15 support class profiles do not exactly cover all support cases."
        )
    return rows


def fold_class_support_counts(
    profiles: Sequence[SupportCaseClassProfile],
    case_ids: Sequence[str],
) -> tuple[int, int]:
    rows = {row.case_id: row for row in profiles}
    cases = tuple(sorted(str(value) for value in case_ids))
    if (
        not cases
        or len(cases) != len(set(cases))
        or any(case not in rows for case in cases)
    ):
        raise ProtocolError("HARP v15 fold-local class normalization is incomplete.")
    counts = tuple(
        sum(rows[case].class_support[label] for case in cases) for label in (0, 1)
    )
    if any(value < 1 for value in counts):
        raise ProtocolError(
            "HARP v15 fold-local BACC requires both classes in the training cases."
        )
    return counts  # type: ignore[return-value]


def fit_support_fold_normalizer(
    profiles: Sequence[SupportCaseClassProfile],
    case_ids: Sequence[str],
) -> SupportFoldNormalizer:
    rows = {row.case_id: row for row in profiles}
    cases = tuple(sorted(str(value) for value in case_ids))
    counts = fold_class_support_counts(tuple(rows.values()), cases)
    outer_ids = {rows[case].outer_target_id for case in cases}
    if len(outer_ids) != 1:
        raise ProtocolError("HARP v15 support normalizer crossed outer targets.")
    return SupportFoldNormalizer(
        outer_target_id=next(iter(outer_ids)),
        case_ids=cases,
        profile_hashes=tuple((case, rows[case].profile_hash) for case in cases),
        class_support_counts=counts,
    )


def normalize_action_outcomes(
    outcomes: Sequence[SupportActionOutcome],
    *,
    profiles: Sequence[SupportCaseClassProfile],
    normalization_case_ids: Sequence[str],
) -> tuple[SupportActionOutcome, ...]:
    """Return targets normalized only by the named training-case inventory.

    Scalar-only predecessor-style outcomes are rejected: the v15 fit surface
    must be reconstructible from primitive class-local components.
    """

    rows = tuple(
        sorted(outcomes, key=lambda row: (row.action.case_id, row.action.action_id))
    )
    if not rows:
        return ()
    primitive = tuple(row.has_class_local_components for row in rows)
    if not all(primitive):
        raise ProtocolError(
            "HARP v15 support outcomes require primitive class-local components."
        )
    profile_rows = tuple(profiles)
    profile_by_case = {row.case_id: row for row in profile_rows}
    if len(profile_by_case) != len(profile_rows):
        raise ProtocolError("HARP v15 support class profiles are duplicated.")
    if any(
        row.action.case_id not in profile_by_case
        or row.class_support != profile_by_case[row.action.case_id].class_support
        or row.action.outer_target_id
        != profile_by_case[row.action.case_id].outer_target_id
        for row in rows
    ):
        raise ProtocolError(
            "HARP v15 action outcome class support drifted from its case profile."
        )
    normalizer = fit_support_fold_normalizer(profile_rows, normalization_case_ids)
    return tuple(normalizer.normalize(row) for row in rows)


__all__ = (
    "SupportFoldNormalizer",
    "fit_support_fold_normalizer",
    "fold_class_support_counts",
    "normalize_action_outcomes",
    "validate_support_case_profiles",
)
