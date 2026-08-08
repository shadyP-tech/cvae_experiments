"""Terminal sealed H x e diagnostics with no policy-update output."""

from __future__ import annotations

import math

import numpy as np

from ...metrics import spearman
from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    OracleDiagnostic,
    SUPPORT_ACTION_ID,
    ScoredEvaluation,
    legal_sources,
    tail_action_id,
)
from .prediction_seal import (
    PredictionSealCapability,
    read_sealed_prediction_snapshot,
)
from .scored_validation import validate_scored_evaluation


def compute_oracle_diagnostics(
    scored: ScoredEvaluation,
    capability: PredictionSealCapability,
) -> tuple[OracleDiagnostic, ...]:
    """Summarize H x e utility without emitting an oracle-derived action."""

    validate_scored_evaluation(scored)
    state = read_sealed_prediction_snapshot(capability)
    if scored.prediction_seal_hash != state.seal_hash:
        raise ProtocolError("Fresh Stage-70 oracle diagnostics escaped their seal.")
    ensemble = {row.key: row for row in scored.ensemble_metrics}
    output: list[OracleDiagnostic] = []
    for target in CENTERS:
        sources = legal_sources(target)
        support_action = state.plan.action_for(target, SUPPORT_ACTION_ID)
        scores = support_action.mean_normalized_midrank_by_source
        if set(scores) != set(sources):
            raise ProtocolError("Fresh Stage-70 support-score coverage drifted.")
        utilities: dict[str, float] = {}
        for source in sources:
            try:
                metric = ensemble[(target, tail_action_id(source))]
            except KeyError as exc:
                raise ProtocolError(
                    "Fresh Stage-70 sealed H x e utility matrix is incomplete."
                ) from exc
            if metric.prediction_seal_hash != state.seal_hash:
                raise ProtocolError("Fresh Stage-70 H x e metric seal drifted.")
            utilities[source] = float(metric.bacc)

        support_top1 = min(sources, key=lambda source: (scores[source], source))
        oracle_top1 = min(
            sources,
            key=lambda source: (-utilities[source], source),
        )
        utility_values = np.asarray(
            [utilities[source] for source in sources],
            dtype=np.float64,
        )
        aligned_scores = [-scores[source] for source in sources]
        rank_correlation = float(
            spearman(aligned_scores, utility_values.tolist())
        )
        rank_defined = math.isfinite(rank_correlation)
        oracle_utility = float(np.max(utility_values))
        selected_utility = utilities[support_top1]
        utility_range = float(np.max(utility_values) - np.min(utility_values))
        headroom = max(0.0, oracle_utility - selected_utility)
        normalized_gap = headroom / utility_range if utility_range > 0.0 else 0.0
        output.append(
            OracleDiagnostic(
                target_center=target,
                source_count=len(sources),
                support_score_utility_spearman=(
                    rank_correlation if rank_defined else 0.0
                ),
                spearman_defined=rank_defined,
                top1_agreement=support_top1 == oracle_top1,
                oracle_headroom_bacc=headroom,
                normalized_oracle_gap=normalized_gap,
                oracle_utility_range_bacc=utility_range,
                prediction_seal_hash=state.seal_hash,
                diagnostic_only=True,
                may_update_frozen_policy=False,
            )
        )
    return tuple(output)


__all__ = ("compute_oracle_diagnostics",)
