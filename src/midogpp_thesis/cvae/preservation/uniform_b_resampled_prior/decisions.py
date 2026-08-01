"""Candidate-specific Pq decision; never direct recipe promotion."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .config import UniformBResampledPriorConfig
from .contracts import P0, PQ, PUBLICATION_STATE


def paired_deltas(
    metric_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    cells: dict[tuple[object, ...], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in metric_rows:
        key = (
            row["outer_center"], row["inner_center"], row["source_center"],
            row["training_seed"], row["generation_seed"],
        )
        cells[key][str(row["prior"])] = row
    output = []
    for key, priors in sorted(cells.items(), key=lambda item: str(item[0])):
        if set(priors) != {P0, PQ}:
            raise ProtocolError("P0/Pq paired cell is incomplete.")
        output.append(
            {
                "schema_version": "midogpp_resampled_prior_paired_delta_v1",
                "outer_center": key[0],
                "inner_center": key[1],
                "source_center": key[2],
                "training_seed": key[3],
                "generation_seed": key[4],
                "treatment": PQ,
                "control": P0,
                "bacc_delta": float(priors[PQ]["bacc"]) - float(priors[P0]["bacc"]),
                "macro_f1_delta": float(priors[PQ]["macro_f1"]) - float(priors[P0]["macro_f1"]),
                "planned_contrast": True,
            }
        )
    return output


def study_decision(
    unique_rows: Sequence[Mapping[str, object]],
    delta_rows: Sequence[Mapping[str, object]],
    generation_rows: Sequence[Mapping[str, object]],
    *,
    config: UniformBResampledPriorConfig,
) -> dict[str, object]:
    if not unique_rows or not delta_rows:
        raise ProtocolError("Cannot decide an empty P0/Pq study.")
    deltas = [float(row["bacc_delta"]) for row in delta_rows]
    by_source: dict[str, list[float]] = defaultdict(list)
    for row in delta_rows:
        by_source[str(row["source_center"])].append(float(row["bacc_delta"]))
    source_means = {source: sum(values) / len(values) for source, values in by_source.items()}
    diversity = {
        prior: _diversity_pass(
            [row for row in unique_rows if row["prior"] == prior],
            config=config,
        )
        for prior in (P0, PQ)
    }
    pq_audits = [row for row in generation_rows if row["prior"] == PQ]
    fallback_fraction = sum(bool(row["fallback_to_p0"]) for row in pq_audits) / len(pq_audits)
    converged = all(bool(row["classifier_converged"]) for row in unique_rows)
    mean_delta = sum(deltas) / len(deltas)
    no_source_regression = all(
        value >= -config.max_source_mean_regression
        for value in source_means.values()
    )
    candidate = bool(
        mean_delta > 0.0
        and no_source_regression
        and diversity[PQ]
        and converged
        and fallback_fraction < 1.0
    )
    return {
        "schema_version": "midogpp_resampled_prior_study_decision_v1",
        "publication_state": PUBLICATION_STATE,
        "decision": (
            "CANDIDATE_FOR_SEPARATE_PROMOTION" if candidate else "DO_NOT_PROMOTE"
        ),
        "mean_pq_minus_p0_bacc": mean_delta,
        "positive_cells": sum(value > 0.0 for value in deltas),
        "n_cells": len(deltas),
        "source_mean_deltas": source_means,
        "no_source_regression": no_source_regression,
        "candidate_specific_diversity": diversity,
        "classifier_convergence_pass": converged,
        "pq_fallback_fraction": fallback_fraction,
        "may_feed_recipe_selection": False,
        "may_feed_expert_bank": False,
        "may_feed_generation": False,
        "may_feed_routing": False,
        "may_feed_downstream_utility": False,
        "separate_promotion_artifact_required": True,
    }


def _diversity_pass(
    rows: Sequence[Mapping[str, object]],
    *,
    config: UniformBResampledPriorConfig,
) -> bool:
    for row in rows:
        for key, raw in row.items():
            if "effective_rank_ratio" in key and float(raw) < config.min_effective_rank_ratio:
                return False
            if "pairwise_" in key and not (
                config.min_pairwise_distance_ratio
                <= float(raw)
                <= config.max_pairwise_distance_ratio
            ):
                return False
    return bool(rows)


__all__ = ("paired_deltas", "study_decision")
