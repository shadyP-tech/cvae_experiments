"""Outer-fold filtering and paired source-inner regret construction."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    EXPECTED_CANDIDATE_SUMMARIES,
    EXPECTED_REGRET_CELLS,
    EXPECTED_UTILITY_ROWS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    CandidateSummary,
    RegretCell,
)


UtilityKey = tuple[str, str, int, int]


def validate_utility_rows(rows: Sequence[Mapping[str, object]]) -> dict[UtilityKey, Mapping[str, object]]:
    """Validate the complete q!=e surface before any outer-fold transform."""

    if len(rows) != EXPECTED_UTILITY_ROWS:
        raise ProtocolError(
            f"Candidate utility coverage drifted: {len(rows)} != {EXPECTED_UTILITY_ROWS}."
        )
    expected = {
        (query, candidate, training_seed, generation_seed)
        for query in CENTERS
        for candidate in CENTERS
        if candidate != query
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    }
    indexed: dict[UtilityKey, Mapping[str, object]] = {}
    for row in rows:
        key = (
            str(row.get("pseudo_target_center", "")),
            str(row.get("candidate_source_center", "")),
            _integer(row.get("training_seed"), "training_seed"),
            _integer(row.get("generation_seed"), "generation_seed"),
        )
        if key in indexed:
            raise ProtocolError(f"Duplicate candidate utility key: {key}.")
        query, candidate, training_seed, generation_seed = key
        if (
            query not in CENTERS
            or candidate not in CENTERS
            or query == candidate
            or training_seed not in TRAINING_SEEDS
            or generation_seed not in GENERATION_SEEDS
        ):
            raise ProtocolError(f"Illegal candidate utility key: {key}.")
        bacc = _metric(row.get("bacc"), "bacc")
        macro_f1 = _metric(row.get("macro_f1"), "macro_f1")
        if not str(row.get("source_stream_id", "")):
            raise ProtocolError("Candidate utility row lacks source-stream identity.")
        status = str(row.get("status", "PASS")).upper()
        if status not in {"PASS", "OK"}:
            raise ProtocolError("Failed candidate utility rows cannot enter selection.")
        if row.get("eval_labels_used_for_scoring_only") is not True:
            raise ProtocolError("Candidate utility labels are not marked scoring-only.")
        if row.get("outer_target_instantiated") is not False:
            raise ProtocolError("Candidate utility must not instantiate an outer target.")
        if row.get("candidate_ranking_performed") is not False:
            raise ProtocolError("Candidate utility must remain non-selecting.")
        if row.get("policy_selection_performed") is not False:
            raise ProtocolError("Candidate utility must remain non-selecting.")
        if row.get("seed_selection_performed") is not False:
            raise ProtocolError("Candidate utility performed seed selection.")
        indexed[key] = row
    if set(indexed) != expected:
        missing = sorted(expected.difference(indexed))[:5]
        extra = sorted(set(indexed).difference(expected))[:5]
        raise ProtocolError(f"Candidate utility lattice drifted; missing={missing}, extra={extra}.")
    return indexed


def build_outer_regret_cells(
    rows: Sequence[Mapping[str, object]],
) -> tuple[RegretCell, ...]:
    """Filter H first, then compute paired regret within each q/seed cell."""

    utility = validate_utility_rows(rows)
    cells: list[RegretCell] = []
    for outer in CENTERS:
        for query in CENTERS:
            if query == outer:
                continue
            legal_candidates = tuple(
                candidate
                for candidate in CENTERS
                if candidate not in {outer, query}
            )
            if len(legal_candidates) != 7:
                raise ProtocolError("Outer source-inner candidate pool is not seven-wide.")
            for training_seed in TRAINING_SEEDS:
                for generation_seed in GENERATION_SEEDS:
                    group = [
                        utility[(query, candidate, training_seed, generation_seed)]
                        for candidate in legal_candidates
                    ]
                    oracle = max(float(row["bacc"]) for row in group)
                    for candidate, row in zip(legal_candidates, group):
                        bacc = float(row["bacc"])
                        regret = oracle - bacc
                        if regret < -1.0e-15:
                            raise ProtocolError("Paired regret is negative.")
                        cells.append(
                            RegretCell(
                                outer_target_center=outer,
                                query_center=query,
                                candidate_source=candidate,
                                training_seed=training_seed,
                                generation_seed=generation_seed,
                                bacc=bacc,
                                macro_f1=float(row["macro_f1"]),
                                oracle_bacc=oracle,
                                regret=max(0.0, regret),
                                source_stream_id=str(row["source_stream_id"]),
                                utility_row_hash=stable_hash(dict(row)),
                            )
                        )
    if len(cells) != EXPECTED_REGRET_CELLS:
        raise ProtocolError("Outer regret cell coverage drifted.")
    return tuple(cells)


def summarize_candidates(cells: Sequence[RegretCell]) -> tuple[CandidateSummary, ...]:
    """Average every candidate over seven queries and all nine seed pairs."""

    expected_keys = {(outer, candidate) for outer in CENTERS for candidate in CENTERS if candidate != outer}
    grouped: dict[tuple[str, str], list[RegretCell]] = {key: [] for key in expected_keys}
    for cell in cells:
        key = (cell.outer_target_center, cell.candidate_source)
        if key not in grouped:
            raise ProtocolError("Regret cell contains the outer target as a candidate.")
        if cell.query_center == cell.outer_target_center:
            raise ProtocolError("Regret cell contains the outer target as a query.")
        grouped[key].append(cell)
    summaries: list[CandidateSummary] = []
    for outer in CENTERS:
        for candidate in CENTERS:
            if candidate == outer:
                continue
            group = grouped[(outer, candidate)]
            queries = {cell.query_center for cell in group}
            pairs = {(cell.training_seed, cell.generation_seed) for cell in group}
            if len(group) != 63 or len(queries) != 7 or len(pairs) != 9:
                raise ProtocolError("Candidate regret aggregation coverage drifted.")
            summaries.append(
                CandidateSummary(
                    outer_target_center=outer,
                    candidate_source=candidate,
                    mean_regret=_mean(cell.regret for cell in group),
                    mean_bacc=_mean(cell.bacc for cell in group),
                    mean_macro_f1=_mean(cell.macro_f1 for cell in group),
                    query_count=len(queries),
                    seed_pair_count=len(pairs),
                    cell_count=len(group),
                )
            )
    if len(summaries) != EXPECTED_CANDIDATE_SUMMARIES:
        raise ProtocolError("Candidate summary coverage drifted.")
    return tuple(summaries)


def utility_table_hash(rows: Sequence[Mapping[str, object]]) -> str:
    validate_utility_rows(rows)
    canonical = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            CENTERS.index(str(row["pseudo_target_center"])),
            CENTERS.index(str(row["candidate_source_center"])),
            int(row["training_seed"]),
            int(row["generation_seed"]),
        ),
    )
    return stable_hash(canonical)


def regret_table_hash(cells: Sequence[RegretCell]) -> str:
    return stable_hash([cell.to_payload() for cell in cells])


def summary_table_hash(summaries: Sequence[CandidateSummary]) -> str:
    return stable_hash([summary.to_payload() for summary in summaries])


def _integer(value: object, label: str) -> int:
    try:
        observed = int(str(value))
    except ValueError as exc:
        raise ProtocolError(f"Candidate utility {label} is invalid.") from exc
    return observed


def _metric(value: object, label: str) -> float:
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Candidate utility {label} is invalid.") from exc
    if not math.isfinite(observed) or not 0.0 <= observed <= 1.0:
        raise ProtocolError(f"Candidate utility {label} is outside [0,1].")
    return observed


def _mean(values: object) -> float:
    materialized = tuple(float(value) for value in values)  # type: ignore[union-attr]
    if not materialized:
        raise ProtocolError("Cannot average an empty regret group.")
    return sum(materialized) / len(materialized)


__all__ = (
    "build_outer_regret_cells",
    "regret_table_hash",
    "summarize_candidates",
    "summary_table_hash",
    "utility_table_hash",
    "validate_utility_rows",
)
