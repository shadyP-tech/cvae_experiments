"""Strict outer-center views over the historical source-inner utility surface."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import fmean, pvariance
from typing import Iterable

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    TRAINING_SEEDS,
)
from midogpp_thesis.cvae.generation.contracts import GENERATION_SEEDS
from midogpp_thesis.cvae.protocol import ProtocolError

from .hashing import canonical_hash, require_sha256


@dataclass(frozen=True, slots=True)
class HistoricalUtilityCell:
    query_center: str
    candidate_center: str
    training_seed: int
    generation_seed: int
    bacc: float
    macro_f1: float

    def __post_init__(self) -> None:
        if self.query_center not in CENTERS or self.candidate_center not in CENTERS:
            raise ProtocolError("SCEPTRE utility cell has an unknown center.")
        if self.query_center == self.candidate_center:
            raise ProtocolError("SCEPTRE utility cell contains forbidden q == e.")
        if self.training_seed not in TRAINING_SEEDS:
            raise ProtocolError("SCEPTRE utility cell has an unknown training seed.")
        if self.generation_seed not in GENERATION_SEEDS:
            raise ProtocolError("SCEPTRE utility cell has an unknown generation seed.")
        if not math.isfinite(self.bacc) or not 0.0 <= self.bacc <= 1.0:
            raise ProtocolError("SCEPTRE utility BACC is invalid.")
        if not math.isfinite(self.macro_f1) or not 0.0 <= self.macro_f1 <= 1.0:
            raise ProtocolError("SCEPTRE utility macro-F1 is invalid.")

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (
            self.query_center,
            self.candidate_center,
            self.training_seed,
            self.generation_seed,
        )


@dataclass(frozen=True, slots=True)
class AggregatedUtility:
    query_center: str
    candidate_center: str
    mean_bacc: float
    seed_cell_variance: float
    seed_cell_count: int


@dataclass(frozen=True, slots=True)
class OuterDevelopmentView:
    """A target-specific surface after q/e deletion and before transforms."""

    outer_target: str
    cells: tuple[HistoricalUtilityCell, ...]
    aggregate_rows: tuple[AggregatedUtility, ...]
    exclusion_receipt_hash: str

    @property
    def query_centers(self) -> tuple[str, ...]:
        return tuple(center for center in CENTERS if center != self.outer_target)

    @property
    def candidate_centers(self) -> tuple[str, ...]:
        return tuple(center for center in CENTERS if center != self.outer_target)


@dataclass(frozen=True, slots=True)
class SourceInnerDevelopmentSurface:
    """Complete immutable historical surface; no fitting occurs at this level."""

    cells: tuple[HistoricalUtilityCell, ...]
    utility_lock_sha256: str
    utility_table_sha256: str
    case_confusions_sha256: str
    amendment_sha256: str

    def __post_init__(self) -> None:
        for value, role in (
            (self.utility_lock_sha256, "utility lock"),
            (self.utility_table_sha256, "utility table"),
            (self.case_confusions_sha256, "case-confusion table"),
            (self.amendment_sha256, "adaptive reuse amendment"),
        ):
            require_sha256(value, role)
        expected = {
            (q, e, training_seed, generation_seed)
            for q in CENTERS
            for e in CENTERS
            if q != e
            for training_seed in TRAINING_SEEDS
            for generation_seed in GENERATION_SEEDS
        }
        observed = {cell.key for cell in self.cells}
        if len(self.cells) != len(expected) or observed != expected:
            raise ProtocolError("SCEPTRE historical utility coverage is not exact.")

    def for_outer_target(self, outer_target: str) -> OuterDevelopmentView:
        """Delete q==H or e==H before any aggregation or model operation."""

        target = str(outer_target)
        if target not in CENTERS:
            raise ProtocolError("SCEPTRE outer target is unknown.")
        cells = tuple(
            sorted(
                (
                    cell
                    for cell in self.cells
                    if cell.query_center != target
                    and cell.candidate_center != target
                    and cell.query_center != cell.candidate_center
                ),
                key=lambda cell: cell.key,
            )
        )
        expected_count = 8 * 7 * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
        if len(cells) != expected_count:
            raise ProtocolError("SCEPTRE outer deletion did not yield 504 cells.")
        aggregate = aggregate_seed_cells(cells)
        receipt = canonical_hash(
            {
                "schema_version": "sceptre_outer_development_view_v1",
                "outer_target": target,
                "strict_filter": "q!=H_and_e!=H_and_q!=e_before_all_transforms",
                "cell_keys": [list(cell.key) for cell in cells],
                "seed_cells_are_nuisance_replications": True,
                "seed_selection_allowed": False,
                "utility_table_sha256": self.utility_table_sha256,
                "amendment_sha256": self.amendment_sha256,
            }
        )
        return OuterDevelopmentView(
            outer_target=target,
            cells=cells,
            aggregate_rows=aggregate,
            exclusion_receipt_hash=receipt,
        )


def aggregate_seed_cells(
    cells: Iterable[HistoricalUtilityCell],
) -> tuple[AggregatedUtility, ...]:
    """Average the exact 3x3 seed grid; cells are replications, not samples."""

    groups: dict[tuple[str, str], list[HistoricalUtilityCell]] = {}
    for cell in cells:
        groups.setdefault((cell.query_center, cell.candidate_center), []).append(cell)
    expected_seed_grid = {
        (training_seed, generation_seed)
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    }
    rows: list[AggregatedUtility] = []
    for (query, candidate), group in sorted(groups.items()):
        seeds = {(cell.training_seed, cell.generation_seed) for cell in group}
        if len(group) != 9 or seeds != expected_seed_grid:
            raise ProtocolError("SCEPTRE utility family lacks the exact 3x3 seed grid.")
        values = [cell.bacc for cell in group]
        rows.append(
            AggregatedUtility(
                query_center=query,
                candidate_center=candidate,
                mean_bacc=fmean(values),
                seed_cell_variance=pvariance(values),
                seed_cell_count=9,
            )
        )
    return tuple(rows)


__all__ = (
    "AggregatedUtility",
    "HistoricalUtilityCell",
    "OuterDevelopmentView",
    "SourceInnerDevelopmentSurface",
    "aggregate_seed_cells",
)
