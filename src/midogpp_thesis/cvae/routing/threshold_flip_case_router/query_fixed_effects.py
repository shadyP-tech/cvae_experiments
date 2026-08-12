"""Query-adjusted global static A1 selection for incomplete q/e panels."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import DonorRow, StaticSelection, canonical_hash
from .targets import pooled_gain


@dataclass(frozen=True)
class QueryFixedEffectStaticFit:
    """Sealed additive q/source fit under sum-to-zero identifiability."""

    heldout_h: str
    query_centers: tuple[str, ...]
    candidate_sources: tuple[str, ...]
    cell_gains: tuple[Mapping[str, object], ...]
    grand_mean: float
    query_effects: tuple[tuple[str, float], ...]
    source_effects: tuple[tuple[str, float], ...]
    adjusted_source_gains: tuple[tuple[str, float], ...]
    residual_sum_squares: float
    design_rank: int
    required_rank: int
    identifiable: bool
    identifiability_failure: str | None
    selection: StaticSelection
    fit_hash: str = ""

    def __post_init__(self) -> None:
        if (
            not self.heldout_h
            or len(self.query_centers) < 3
            or self.query_centers != self.candidate_sources
            or len(set(self.query_centers)) != len(self.query_centers)
            or self.required_rank != 2 * len(self.query_centers) - 1
            or len(self.cell_gains)
            != len(self.query_centers) * (len(self.query_centers) - 1)
            or tuple(name for name, _ in self.query_effects) != self.query_centers
            or tuple(name for name, _ in self.source_effects) != self.candidate_sources
            or tuple(name for name, _ in self.adjusted_source_gains)
            != self.candidate_sources
        ):
            raise ProtocolError("Query-fixed-effect static fit topology drifted.")
        if self.identifiable:
            if self.design_rank != self.required_rank or self.identifiability_failure is not None:
                raise ProtocolError("Identifiable query-fixed-effect fit metadata drifted.")
        elif (
            not self.selection.fallback_to_b
            or self.identifiability_failure is None
        ):
            raise ProtocolError("Nonidentifiable query-fixed-effect fit must fall back to B.")
        numbers = (
            self.grand_mean,
            self.residual_sum_squares,
            *(value for _, value in self.query_effects),
            *(value for _, value in self.source_effects),
            *(value for _, value in self.adjusted_source_gains),
        )
        if not all(math.isfinite(float(value)) for value in numbers):
            raise ProtocolError("Query-fixed-effect static fit is not finite.")
        expected = canonical_hash(self._unhashed_payload())
        if self.fit_hash and self.fit_hash != expected:
            raise ProtocolError("Query-fixed-effect static fit hash drifted.")
        object.__setattr__(self, "fit_hash", expected)

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "threshold_flip_query_fixed_effect_static_v1",
            "heldout_H": self.heldout_h,
            "objective": "unweighted_least_squares_exact_per_q_e_pooled_bacc_gain",
            "model": "gain_qe=grand_mean+query_effect_q+source_effect_e",
            "identifiability_constraints": [
                "sum_query_effects=0",
                "sum_source_effects=0",
            ],
            "query_centers": list(self.query_centers),
            "candidate_sources": list(self.candidate_sources),
            "cell_gains": [dict(row) for row in self.cell_gains],
            "grand_mean": self.grand_mean,
            "query_effects": {name: value for name, value in self.query_effects},
            "source_effects": {name: value for name, value in self.source_effects},
            "adjusted_source_gains": {
                name: value for name, value in self.adjusted_source_gains
            },
            "residual_sum_squares": self.residual_sum_squares,
            "observation_count": len(self.cell_gains),
            "design_rank": self.design_rank,
            "required_rank": self.required_rank,
            "identifiable": self.identifiable,
            "identifiability_failure": self.identifiability_failure,
            "selection": self.selection.to_payload(),
            "B_fallback_if_best_adjusted_gain_nonpositive": True,
            "heldout_H_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "fit_hash": self.fit_hash}


def select_query_fixed_effect_static_source(
    donors: Sequence[DonorRow],
    *,
    heldout_h: str,
) -> QueryFixedEffectStaticFit:
    """Select by adjusted source gain over the balanced off-diagonal q/e panel."""

    rows = tuple(donors)
    if not rows or any(row.model_target != heldout_h for row in rows):
        raise ProtocolError("Query-fixed-effect donors do not bind heldout H.")
    centers = tuple(
        sorted(
            {row.query_center for row in rows}
            | {row.candidate_source for row in rows}
        )
    )
    if heldout_h in centers or len(centers) < 3:
        raise ProtocolError("Query-fixed-effect donor center topology drifted.")
    grouped: dict[tuple[str, str], list[object]] = {}
    for row in rows:
        if row.query_center == row.candidate_source:
            raise ProtocolError("Query-fixed-effect panel contains q=e.")
        if row.action_id != f"A1::source={row.candidate_source}":
            raise ProtocolError("Query-fixed-effect donor action/source drifted.")
        grouped.setdefault((row.query_center, row.candidate_source), []).append(
            row.target
        )
    expected = {(query, source) for query in centers for source in centers if query != source}
    if set(grouped) != expected:
        raise ProtocolError("Query-fixed-effect panel is not the exact off-diagonal design.")
    cell_rows = []
    missing_class_cell = False
    for query, source in sorted(expected):
        targets = tuple(grouped[(query, source)])
        try:
            gain: float | None = pooled_gain(targets)
        except ProtocolError:
            gain = None
            missing_class_cell = True
        cell_rows.append({
            "query_center_q": query,
            "candidate_source_e": source,
            "action_id": f"A1::source={source}",
            "exact_pooled_bacc_gain": gain,
            "case_count": len(targets),
        })
    cells = tuple(cell_rows)
    n = len(centers)
    design = np.zeros((len(cells), 1 + 2 * n), dtype=np.float64)
    response = np.zeros(len(cells), dtype=np.float64)
    index = {center: ordinal for ordinal, center in enumerate(centers)}
    for ordinal, cell in enumerate(cells):
        design[ordinal, 0] = 1.0
        design[ordinal, 1 + index[str(cell["query_center_q"])]] = 1.0
        design[ordinal, 1 + n + index[str(cell["candidate_source_e"])]] = 1.0
        if cell["exact_pooled_bacc_gain"] is not None:
            response[ordinal] = float(cell["exact_pooled_bacc_gain"])
    constraints = np.zeros((2, 1 + 2 * n), dtype=np.float64)
    constraints[0, 1 : 1 + n] = 1.0
    constraints[1, 1 + n :] = 1.0
    normal = np.block(
        [
            [design.T @ design, constraints.T],
            [constraints, np.zeros((2, 2), dtype=np.float64)],
        ]
    )
    right = np.concatenate((design.T @ response, np.zeros(2, dtype=np.float64)))
    required_rank = 2 * n - 1
    design_rank = int(np.linalg.matrix_rank(design))
    if (
        missing_class_cell
        or design_rank != required_rank
        or int(np.linalg.matrix_rank(normal)) != len(normal)
    ):
        failure = (
            "per_q_e_exact_pooled_bacc_lacks_both_classes"
            if missing_class_cell
            else "additive_query_source_design_rank_deficient"
        )
        zeros = tuple((center, 0.0) for center in centers)
        return QueryFixedEffectStaticFit(
            heldout_h=str(heldout_h),
            query_centers=centers,
            candidate_sources=centers,
            cell_gains=cells,
            grand_mean=0.0,
            query_effects=zeros,
            source_effects=zeros,
            adjusted_source_gains=zeros,
            residual_sum_squares=0.0,
            design_rank=design_rank,
            required_rank=required_rank,
            identifiable=False,
            identifiability_failure=failure,
            selection=StaticSelection("B", 0.0, 0.0, True),
        )
    solution = np.linalg.solve(normal, right)[: 1 + 2 * n]
    fitted = design @ solution
    grand_mean = float(solution[0])
    query_effects = tuple(
        (center, float(solution[1 + index[center]])) for center in centers
    )
    source_effects = tuple(
        (center, float(solution[1 + n + index[center]])) for center in centers
    )
    adjusted = tuple(
        (center, grand_mean + effect) for center, effect in source_effects
    )
    ranked = sorted(adjusted, key=lambda item: (-item[1], f"A1::source={item[0]}"))
    best_source, best_gain = ranked[0]
    runner_up = max(0.0, ranked[1][1])
    selection = (
        StaticSelection("B", 0.0, max(0.0, best_gain), True)
        if best_gain <= 0.0
        else StaticSelection(
            f"A1::source={best_source}", float(best_gain), float(runner_up), False
        )
    )
    return QueryFixedEffectStaticFit(
        heldout_h=str(heldout_h),
        query_centers=centers,
        candidate_sources=centers,
        cell_gains=cells,
        grand_mean=grand_mean,
        query_effects=query_effects,
        source_effects=source_effects,
        adjusted_source_gains=adjusted,
        residual_sum_squares=float(np.sum((response - fitted) ** 2, dtype=np.float64)),
        design_rank=design_rank,
        required_rank=required_rank,
        identifiable=True,
        identifiability_failure=None,
        selection=selection,
    )


__all__ = (
    "QueryFixedEffectStaticFit",
    "select_query_fixed_effect_static_source",
)
