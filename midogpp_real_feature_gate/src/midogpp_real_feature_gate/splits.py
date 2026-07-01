"""Held-out center/domain split construction for the MIDOG++ gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import ELIGIBLE_CENTERS, QUARANTINE_CENTERS
from .data import ManifestRow


@dataclass(frozen=True)
class Fold:
    fold_unit: str
    heldout_center: str
    heldout_tumor_domain: str
    source_indices: tuple[int, ...]
    eval_indices: tuple[int, ...]


def is_eligible_center(center: str) -> bool:
    return str(center) in ELIGIBLE_CENTERS


def is_quarantine_center(center: str) -> bool:
    return str(center) in QUARANTINE_CENTERS


def heldout_center_folds(rows: Sequence[ManifestRow]) -> tuple[Fold, ...]:
    centers = sorted({row.center for row in rows if is_eligible_center(row.center) or is_quarantine_center(row.center)})
    folds: list[Fold] = []
    for center in centers:
        source = tuple(idx for idx, row in enumerate(rows) if row.center != center)
        eval_ = tuple(idx for idx, row in enumerate(rows) if row.center == center)
        folds.append(
            Fold(
                fold_unit="heldout_center",
                heldout_center=center,
                heldout_tumor_domain="",
                source_indices=source,
                eval_indices=eval_,
            )
        )
    return tuple(folds)
