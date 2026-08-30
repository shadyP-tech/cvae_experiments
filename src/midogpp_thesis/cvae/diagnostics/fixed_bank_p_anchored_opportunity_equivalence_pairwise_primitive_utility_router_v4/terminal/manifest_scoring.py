"""Canonical manifest alignment and terminal-only aggregate score derivation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ....protocol import ProtocolError
from ..hashing import canonical_hash, require_sha256
from ..identity import (
    ACTION_IDS,
    CENTERS,
    EXPECTED_CASE_COUNT,
    EXPECTED_TEST_MANIFEST_SHA256,
    EXPECTED_TEST_ROW_COUNT,
    P_ACTION_ID,
)
from ..physical.compiled_matrix import CompiledProbabilityMatrix
from ..physical.frame import LabelFreeTestFrame
from ..science.target_decision import TargetDecisionLedger
from ..science.target_inventory import CANONICAL_TARGET_CASE_INVENTORY


@dataclass(frozen=True, slots=True)
class CaseRoutingDiagnostic:
    center_id: str
    case_id: str
    selected_action_id: str
    oracle_action_id: str
    spearman_rank_correlation: float | None
    normalized_oracle_gap: float

    def __post_init__(self) -> None:
        center_id = str(self.center_id)
        case_id = str(self.case_id)
        selected = str(self.selected_action_id)
        oracle = str(self.oracle_action_id)
        spearman = (
            None
            if self.spearman_rank_correlation is None
            else float(self.spearman_rank_correlation)
        )
        gap = float(self.normalized_oracle_gap)
        allowed = {P_ACTION_ID, *ACTION_IDS}
        if (
            center_id not in CENTERS
            or not case_id
            or selected not in allowed
            or oracle not in allowed
            or (
                spearman is not None
                and (not math.isfinite(spearman) or not -1.0 <= spearman <= 1.0)
            )
            or not math.isfinite(gap)
            or not 0.0 <= gap <= 1.0
        ):
            raise ProtocolError("OE-PPUR v4 terminal case diagnostic drifted.")
        object.__setattr__(self, "center_id", center_id)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "selected_action_id", selected)
        object.__setattr__(self, "oracle_action_id", oracle)
        object.__setattr__(self, "spearman_rank_correlation", spearman)
        object.__setattr__(self, "normalized_oracle_gap", gap)

    @property
    def rank_available(self) -> bool:
        return self.spearman_rank_correlation is not None

    @property
    def case_key(self) -> tuple[str, str]:
        return (self.center_id, self.case_id)


def _validate_matrix_ledger_frame_linkage(
    matrix: CompiledProbabilityMatrix,
    ledger: TargetDecisionLedger,
    frame: LabelFreeTestFrame,
) -> None:
    if (
        type(matrix) is not CompiledProbabilityMatrix
        or type(ledger) is not TargetDecisionLedger
        or type(frame) is not LabelFreeTestFrame
        or matrix.row_ids != tuple(row.sample_id for row in frame.rows)
        or ledger.expected_case_inventory != CANONICAL_TARGET_CASE_INVENTORY
    ):
        raise ProtocolError("OE-PPUR v4 terminal matrix/frame identity drifted.")
    covered: set[int] = set()
    for decision in ledger.decisions:
        start, stop = matrix.center_offsets[decision.center_id]
        global_indices = tuple(start + value for value in decision.row_indices)
        if (
            not global_indices
            or any(value < start or value >= stop for value in global_indices)
            or covered.intersection(global_indices)
        ):
            raise ProtocolError("OE-PPUR v4 terminal local/global row binding drifted.")
        for local, global_index in zip(
            decision.row_indices, global_indices, strict=True
        ):
            row = frame.rows[global_index]
            if (
                row.center != decision.center_id
                or row.case_id != decision.case_id
                or row.sample_id != matrix.row_ids[start + local]
            ):
                raise ProtocolError(
                    "OE-PPUR v4 terminal row ID drifted after local/global conversion."
                )
        covered.update(global_indices)
    if covered != set(range(EXPECTED_TEST_ROW_COUNT)):
        raise ProtocolError("OE-PPUR v4 terminal decisions do not cover the matrix.")


def _read_aligned_manifest_labels(
    raw: bytes,
    *,
    frame: LabelFreeTestFrame,
) -> tuple[int, ...]:
    try:
        reader = csv.DictReader(raw.decode("utf-8").splitlines())
        fieldnames = tuple(reader.fieldnames or ())
        manifest_rows = tuple(dict(row) for row in reader)
    except (UnicodeError, csv.Error) as exc:
        raise ProtocolError("OE-PPUR v4 canonical manifest is unreadable.") from exc
    required = {"sample_id", "case_id", "label", "center", "split"}
    if not required <= set(fieldnames) or not manifest_rows:
        raise ProtocolError("OE-PPUR v4 canonical manifest schema drifted.")
    labels = []
    for row in frame.rows:
        try:
            manifest = manifest_rows[row.manifest_row_index]
        except IndexError as exc:
            raise ProtocolError("OE-PPUR v4 manifest row index escaped.") from exc
        if (
            row.sample_id
            != _evaluation_row_id(EXPECTED_TEST_MANIFEST_SHA256, row.manifest_row_index)
            or manifest.get("case_id") != row.case_id
            or manifest.get("center") != row.center
            or manifest.get("split") != "test"
            or manifest.get("label") not in {"0", "1"}
        ):
            raise ProtocolError("OE-PPUR v4 manifest/cache row alignment drifted.")
        labels.append(int(manifest["label"]))
    if len(labels) != EXPECTED_TEST_ROW_COUNT:
        raise ProtocolError("OE-PPUR v4 canonical manifest label coverage drifted.")
    return tuple(labels)


def _evaluation_row_id(manifest_sha256: str, row_index: int) -> str:
    digest = require_sha256(manifest_sha256, "manifest SHA-256")
    if type(row_index) is not int or row_index < 0:
        raise ProtocolError("OE-PPUR v4 manifest row index is invalid.")
    return "eval_" + canonical_hash(
        {"manifest_sha256": digest, "contract_row_index": row_index}
    )


def _derive_terminal_values(
    matrix: CompiledProbabilityMatrix,
    ledger: TargetDecisionLedger,
    frame: LabelFreeTestFrame,
    labels: tuple[int, ...],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[CaseRoutingDiagnostic, ...]]:
    selected = np.empty(EXPECTED_TEST_ROW_COUNT, dtype=np.float64)
    protected = np.asarray(matrix.values[:, 0], dtype=np.float64)
    diagnostics = []
    for decision in ledger.decisions:
        start, stop = matrix.center_offsets[decision.center_id]
        global_indices = np.asarray(
            [start + value for value in decision.row_indices], dtype=np.int64
        )
        if np.any(global_indices < start) or np.any(global_indices >= stop):
            raise ProtocolError("OE-PPUR v4 terminal local row index escaped center.")
        case_labels = tuple(labels[int(value)] for value in global_indices)
        case_matrix = np.asarray(matrix.values[global_indices], dtype=np.float64)
        selected_column = matrix.action_ids.index(decision.selected_action_id)
        selected[global_indices] = case_matrix[:, selected_column]
        utilities = tuple(
            _balanced_accuracy(
                case_labels,
                tuple(float(value) for value in case_matrix[:, column]),
            )
            for column in range(len(matrix.action_ids))
        )
        oracle_index = max(range(len(utilities)), key=lambda value: utilities[value])
        spread = max(utilities) - min(utilities)
        selected_utility = utilities[selected_column]
        gap = 0.0 if spread <= 0.0 else (max(utilities) - selected_utility) / spread
        spearman = None
        if decision.rank_available:
            predicted = tuple(
                float(value)
                for _action, value in decision.predicted_action_scores
                if value is not None
            )
            if len(predicted) != len(utilities):
                raise ProtocolError("OE-PPUR v4 predicted ranking surface drifted.")
            spearman = _spearman_or_unavailable(predicted, utilities)
        diagnostics.append(
            CaseRoutingDiagnostic(
                center_id=decision.center_id,
                case_id=decision.case_id,
                selected_action_id=decision.selected_action_id,
                oracle_action_id=matrix.action_ids[oracle_index],
                spearman_rank_correlation=spearman,
                normalized_oracle_gap=gap,
            )
        )
    if not np.isfinite(selected).all():
        raise ProtocolError("OE-PPUR v4 selected probability coverage drifted.")
    return (
        tuple(float(value) for value in selected),
        tuple(float(value) for value in protected),
        tuple(diagnostics),
    )


def _spearman_or_unavailable(
    predicted: Sequence[float], actual: Sequence[float]
) -> float | None:
    left = _average_ranks(predicted)
    right = _average_ranks(actual)
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(
        np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered))
    )
    if denominator <= 0.0:
        return None
    return max(-1.0, min(1.0, float(np.dot(left_centered, right_centered) / denominator)))


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ProtocolError("OE-PPUR v4 rank input drifted.")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _balanced_accuracy(labels: tuple[int, ...], probabilities: tuple[float, ...]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ProtocolError("OE-PPUR v4 terminal labels lack both binary classes.")
    true_positive = sum(
        label == 1 and probability >= 0.5
        for label, probability in zip(labels, probabilities, strict=True)
    )
    true_negative = sum(
        label == 0 and probability < 0.5
        for label, probability in zip(labels, probabilities, strict=True)
    )
    return 0.5 * (true_positive / positives + true_negative / negatives)


def _brier(labels: tuple[int, ...], probabilities: tuple[float, ...]) -> float:
    return sum(
        (probability - label) ** 2
        for label, probability in zip(labels, probabilities, strict=True)
    ) / len(labels)


def _log_loss(labels: tuple[int, ...], probabilities: tuple[float, ...]) -> float:
    epsilon = 1e-7
    return -sum(
        label * math.log(min(1.0 - epsilon, max(epsilon, probability)))
        + (1 - label) * math.log(min(1.0 - epsilon, max(epsilon, 1.0 - probability)))
        for label, probability in zip(labels, probabilities, strict=True)
    ) / len(labels)
