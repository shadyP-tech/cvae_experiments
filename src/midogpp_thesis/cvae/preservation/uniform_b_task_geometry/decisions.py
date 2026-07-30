"""Pure non-promotional aggregation for the bounded mechanism study."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .contracts import BF, BG, BM, BT, PUBLICATION_STATE


PLANNED_CONTRASTS = ((BG, BF), (BM, BG), (BT, BM))


def paired_deltas(
    metric_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_cell: dict[tuple[object, ...], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in metric_rows:
        key = (
            row["outer_center"],
            row["inner_center"],
            row.get("source_center", ""),
            row["training_seed"],
            row["generation_seed"],
            row["composition_mode"],
            row["generation_kind"],
        )
        by_cell[key][str(row["arm"])] = row
    output: list[dict[str, object]] = []
    for key, arms in sorted(by_cell.items(), key=lambda item: str(item[0])):
        for treatment, control in PLANNED_CONTRASTS:
            if key[6] != "prior":
                continue
            if treatment not in arms or control not in arms:
                continue
            output.append(
                {
                    "schema_version": "midogpp_uniform_b_paired_delta_v1",
                    "outer_center": key[0],
                    "inner_center": key[1],
                    "source_center": key[2],
                    "training_seed": key[3],
                    "generation_seed": key[4],
                    "composition_mode": key[5],
                    "generation_kind": key[6],
                    "treatment": treatment,
                    "control": control,
                    "bacc_delta": float(arms[treatment]["bacc"])
                    - float(arms[control]["bacc"]),
                    "macro_f1_delta": float(arms[treatment]["macro_f1"])
                    - float(arms[control]["macro_f1"]),
                    "planned_contrast": True,
                }
            )
    return output


def prior_posterior_gaps(
    metric_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_cell: dict[tuple[object, ...], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in metric_rows:
        key = (
            row["outer_center"],
            row["inner_center"],
            row.get("source_center", ""),
            row["training_seed"],
            row["generation_seed"],
            row["composition_mode"],
            row["arm"],
        )
        by_cell[key][str(row["generation_kind"])] = row
    output = []
    for key, kinds in sorted(by_cell.items(), key=lambda item: str(item[0])):
        if set(kinds) != {"prior", "posterior"}:
            continue
        output.append(
            {
                "schema_version": "midogpp_uniform_b_prior_posterior_gap_v1",
                "outer_center": key[0],
                "inner_center": key[1],
                "source_center": key[2],
                "training_seed": key[3],
                "generation_seed": key[4],
                "composition_mode": key[5],
                "arm": key[6],
                "treatment": "posterior",
                "control": "prior",
                "generation_kind": "paired_gap",
                "bacc_delta": float(kinds["posterior"]["bacc"])
                - float(kinds["prior"]["bacc"]),
                "macro_f1_delta": float(kinds["posterior"]["macro_f1"])
                - float(kinds["prior"]["macro_f1"]),
                "planned_contrast": True,
                "diagnostic_only": True,
            }
        )
    return output


def study_decision(
    metric_rows: Sequence[Mapping[str, object]],
    delta_rows: Sequence[Mapping[str, object]],
    *,
    diversity_pass: bool,
) -> dict[str, object]:
    if not metric_rows:
        raise ProtocolError("Cannot decide an empty Uniform-B study.")
    summaries = []
    for treatment, control in PLANNED_CONTRASTS:
        values = [
            float(row["bacc_delta"])
            for row in delta_rows
            if row["treatment"] == treatment and row["control"] == control
        ]
        summaries.append(
            {
                "treatment": treatment,
                "control": control,
                "n_cells": len(values),
                "mean_bacc_delta": (
                    sum(values) / len(values) if values else None
                ),
                "positive_cells": sum(value > 0.0 for value in values),
            }
        )
    return {
        "schema_version": "midogpp_uniform_b_task_geometry_decision_v1",
        "publication_state": PUBLICATION_STATE,
        "decision": "DO_NOT_PROMOTE",
        "mechanism_diagnostics_complete": True,
        "diversity_safety_pass": bool(diversity_pass),
        "planned_contrasts": summaries,
        "claim": (
            "held-out-inner discriminative prior-TSTR mechanism diagnostic only"
        ),
        "may_feed_recipe_selection": False,
        "may_feed_expert_bank": False,
        "may_feed_generation": False,
        "may_feed_routing": False,
        "may_feed_downstream_utility": False,
        "separate_promotion_artifact_required": True,
    }


__all__ = (
    "PLANNED_CONTRASTS",
    "paired_deltas",
    "prior_posterior_gaps",
    "study_decision",
)
