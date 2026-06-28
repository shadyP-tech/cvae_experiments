from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

from core.metrics import spearman
from core.protocol import ProtocolError


@dataclass(frozen=True)
class CalibrationStats:
    expert_id: str
    mean: float
    std: float
    source: str = "expert_source_val"


@dataclass(frozen=True)
class SupportScore:
    experiment_seed: int
    heldout_center: str
    support_seed: int
    support_size: int
    expert_id: str
    raw_support_nelbo: float
    calibrated_support_nelbo: float
    candidate_rank: int = 0
    selected_top1: bool = False
    selected_top2: bool = False
    selected_top3: bool = False
    selected_expert_count: int = 0
    selected_fraction: float = 0.0
    oracle_rank_diagnostic: int | None = None
    downstream_bacc: float = math.nan

    def to_csv_row(self) -> dict[str, object]:
        return {
            "experiment_seed": self.experiment_seed,
            "heldout_center": self.heldout_center,
            "support_seed": self.support_seed,
            "support_size": self.support_size,
            "expert_id": self.expert_id,
            "eligible_expert_count": 4,
            "candidate_rank": self.candidate_rank,
            "raw_support_nelbo": self.raw_support_nelbo,
            "calibrated_support_nelbo": self.calibrated_support_nelbo,
            "selected_top1": int(self.selected_top1),
            "selected_top2": int(self.selected_top2),
            "selected_top3": int(self.selected_top3),
            "selected_expert_count": self.selected_expert_count,
            "selected_fraction": self.selected_fraction,
            "oracle_rank_diagnostic": "" if self.oracle_rank_diagnostic is None else self.oracle_rank_diagnostic,
            "downstream_bacc": self.downstream_bacc,
        }


def calibration_stats(expert_id: str, source_val_marginal_nelbos: Sequence[float]) -> CalibrationStats:
    vals = [float(v) for v in source_val_marginal_nelbos if math.isfinite(float(v))]
    if not vals:
        raise ProtocolError(f"No finite source-validation NELBO values for expert {expert_id}.")
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var)
    return CalibrationStats(expert_id=str(expert_id), mean=mean, std=std if std > 1.0e-12 else 1.0)


def calibrate(raw_marginal_nelbo: float, stats: CalibrationStats) -> float:
    return (float(raw_marginal_nelbo) - float(stats.mean)) / float(stats.std)


def rank_support_scores(scores: Sequence[SupportScore], *, eligible_count: int = 4) -> tuple[SupportScore, ...]:
    valid = [row for row in scores if math.isfinite(float(row.calibrated_support_nelbo))]
    if len(valid) < int(eligible_count):
        raise ProtocolError("Fewer than 4 valid candidates after NELBO scoring.")
    ranked = sorted(valid, key=lambda row: (float(row.calibrated_support_nelbo), str(row.expert_id)))
    out = []
    for rank, row in enumerate(ranked, start=1):
        out.append(
            replace(
                row,
                candidate_rank=rank,
                selected_top1=rank <= 1,
                selected_top2=rank <= 2,
                selected_top3=rank <= 3,
            )
        )
    return tuple(out)


def selected_experts(ranked: Sequence[SupportScore], k: int) -> tuple[str, ...]:
    if k <= 0:
        raise ProtocolError("k must be positive.")
    ordered = sorted(ranked, key=lambda row: (int(row.candidate_rank), str(row.expert_id)))
    if len(ordered) < k:
        raise ProtocolError("Cannot select fewer candidates than k.")
    return tuple(row.expert_id for row in ordered[:k])


def annotate_selection_fraction(ranked: Sequence[SupportScore], *, k: int, eligible_count: int = 4) -> tuple[SupportScore, ...]:
    chosen = set(selected_experts(ranked, k))
    return tuple(
        replace(
            row,
            selected_expert_count=int(k) if row.expert_id in chosen else 0,
            selected_fraction=float(k) / float(eligible_count) if row.expert_id in chosen else 0.0,
        )
        for row in ranked
    )


def ranking_alignment(
    *,
    ranked_scores: Sequence[SupportScore],
    downstream_bacc_by_expert: dict[str, float],
    method_baccs: dict[str, float] | None = None,
) -> dict[str, float]:
    rows = [row for row in ranked_scores if row.expert_id in downstream_bacc_by_expert]
    if not rows:
        return {
            "top1_downstream_oracle_hit": math.nan,
            "top2_oracle_containment": math.nan,
            "top3_oracle_containment": math.nan,
            "spearman_support_nelbo_vs_downstream_bacc": math.nan,
            "mean_oracle_rank_of_selected_experts": math.nan,
            "oracle_gap_top1": math.nan,
            "oracle_gap_top2": math.nan,
            "oracle_gap_top3": math.nan,
            "oracle_gap_all4": math.nan,
        }
    oracle = max(downstream_bacc_by_expert, key=lambda key: (float(downstream_bacc_by_expert[key]), str(key)))
    top1 = set(selected_experts(rows, 1))
    top2 = set(selected_experts(rows, min(2, len(rows))))
    top3 = set(selected_experts(rows, min(3, len(rows))))
    support_scores = [-float(row.calibrated_support_nelbo) for row in rows]
    baccs = [float(downstream_bacc_by_expert[row.expert_id]) for row in rows]
    oracle_order = sorted(downstream_bacc_by_expert, key=lambda key: (-float(downstream_bacc_by_expert[key]), str(key)))
    selected_ranks = [oracle_order.index(expert_id) + 1 for expert_id in selected_experts(rows, min(2, len(rows)))]
    oracle_bacc = float(downstream_bacc_by_expert[oracle])
    method_baccs = method_baccs or {}
    return {
        "top1_downstream_oracle_hit": float(oracle in top1),
        "top2_oracle_containment": float(oracle in top2),
        "top3_oracle_containment": float(oracle in top3),
        "spearman_support_nelbo_vs_downstream_bacc": spearman(support_scores, baccs),
        "mean_oracle_rank_of_selected_experts": sum(selected_ranks) / float(len(selected_ranks)),
        "oracle_gap_top1": oracle_bacc - float(method_baccs.get("support_nelbo_top1", math.nan)),
        "oracle_gap_top2": oracle_bacc - float(method_baccs.get("support_nelbo_top2_geom", math.nan)),
        "oracle_gap_top3": oracle_bacc - float(method_baccs.get("support_nelbo_top3_geom", math.nan)),
        "oracle_gap_all4": oracle_bacc - float(method_baccs.get("all4_geom", math.nan)),
    }
