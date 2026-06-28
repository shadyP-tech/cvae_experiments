"""Rank and oracle-gap metrics for downstream selection reports."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..downstream import CandidateDownstreamRow, spearman
from ..protocol import ProtocolError
from ..schemas import LEARNED_UTILITY_ALIGNMENT_COLUMNS, SELECTION_ELIGIBLE


def normalized_oracle_gap(*, oracle_bacc: float, selected_bacc: float) -> float:
    if oracle_bacc == 0.0:
        return float("nan")
    return (float(oracle_bacc) - float(selected_bacc)) / abs(float(oracle_bacc))


def spearman_with_true_utility(scores: Sequence[float], utilities: Sequence[float]) -> float:
    return spearman(scores, utilities)


def build_learned_utility_alignment_rows(
    *,
    selection_rows: Sequence[Mapping[str, object]],
    downstream_rows: Sequence[CandidateDownstreamRow],
) -> list[dict[str, object]]:
    """Join deployable selections to downstream utility for final reporting only."""

    downstream_by_candidate = {
        _downstream_key(row): row
        for row in downstream_rows
        if row.status == "ok" and row.row_type == "single_expert"
    }
    oracle_by_context: dict[tuple[object, ...], CandidateDownstreamRow] = {}
    for row in downstream_rows:
        if row.status != "ok" or row.row_type != "single_expert":
            continue
        context = _context_key_from_downstream(row)
        current = oracle_by_context.get(context)
        if current is None or (float(row.bacc), float(row.macro_f1), row.candidate_expert) > (
            float(current.bacc),
            float(current.macro_f1),
            current.candidate_expert,
        ):
            oracle_by_context[context] = row

    aligned: list[dict[str, object]] = []
    for selection in selection_rows:
        if str(selection.get("eligibility")) != SELECTION_ELIGIBLE:
            raise ProtocolError(f"Selection row is not adoption eligible: {selection}")
        selected = downstream_by_candidate.get(_downstream_key_from_selection(selection))
        if selected is None:
            raise ProtocolError(f"Missing downstream row for selected candidate: {selection}")
        oracle = oracle_by_context.get(_context_key_from_selection(selection))
        if oracle is None:
            raise ProtocolError(f"Missing diagnostic oracle context for selected candidate: {selection}")
        aligned.append(
            {
                "fold_id": selection["fold_id"],
                "experiment_seed": selection["experiment_seed"],
                "target_domain": selection["target_domain"],
                "support_split_id": selection["support_split_id"],
                "eval_split_id": selection["eval_split_id"],
                "method": selection["method"],
                "candidate_id": selection["candidate_id"],
                "expert_checkpoint_id": selection["expert_checkpoint_id"],
                "generation_mode": selection["generation_mode"],
                "generation_seed": selection["generation_seed"],
                "classifier_seed": selection["classifier_seed"],
                "predicted_primary_utility": selection.get("predicted_primary_utility", ""),
                "selected_bacc": float(selected.bacc),
                "selected_macro_f1": float(selected.macro_f1),
                "downstream_oracle_candidate_id": _candidate_id_from_downstream(oracle),
                "oracle_bacc": float(oracle.bacc),
                "oracle_macro_f1": float(oracle.macro_f1),
                "downstream_oracle_gap_bacc": float(oracle.bacc) - float(selected.bacc),
                "downstream_oracle_gap_macro_f1": float(oracle.macro_f1) - float(selected.macro_f1),
                "top1_downstream_oracle_hit": int(_downstream_key(selected) == _downstream_key(oracle)),
                "eligibility": selection["eligibility"],
            }
        )
    return aligned


def learned_utility_alignment_columns() -> tuple[str, ...]:
    return LEARNED_UTILITY_ALIGNMENT_COLUMNS


def _downstream_key(row: CandidateDownstreamRow) -> tuple[object, ...]:
    return (
        int(row.experiment_seed),
        str(row.heldout_center),
        str(row.candidate_expert),
        str(row.generation_mode),
        int(row.generation_seed),
        int(row.classifier_seed),
    )


def _downstream_key_from_selection(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        int(row["experiment_seed"]),
        str(row["target_domain"]),
        str(row["expert_checkpoint_id"]),
        str(row["generation_mode"]),
        int(row["generation_seed"]),
        int(row["classifier_seed"]),
    )


def _context_key_from_downstream(row: CandidateDownstreamRow) -> tuple[object, ...]:
    return (
        int(row.experiment_seed),
        str(row.heldout_center),
        str(row.generation_mode),
        int(row.generation_seed),
        int(row.classifier_seed),
    )


def _context_key_from_selection(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        int(row["experiment_seed"]),
        str(row["target_domain"]),
        str(row["generation_mode"]),
        int(row["generation_seed"]),
        int(row["classifier_seed"]),
    )


def _candidate_id_from_downstream(row: CandidateDownstreamRow) -> str:
    return "|".join(
        [
            f"expert={row.candidate_expert}",
            f"target={row.heldout_center}",
            f"mode={row.generation_mode}",
            f"gseed={row.generation_seed}",
            f"cseed={row.classifier_seed}",
        ]
    )
