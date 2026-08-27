"""Manager-owned terminal label reader and aggregate scoring for OE-PPUR v3.

The physical manifest factory remains closed while the amendment is absent.
The concrete evaluator is testable over an opaque, manager-owned input view.
Raw row/case values have no return or persistence method and are destroyed
immediately after the one authorized aggregate read.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import NoReturn, Sequence

from ....protocol import ProtocolError
from ..hashing import canonical_hash
from ..identity import (
    ACTION_IDS,
    EXPECTED_CASE_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    P_ACTION_ID,
)
from .contracts import (
    ALLOWED_AGGREGATE_METRICS,
    AggregateOnlyTerminalReceipt,
    AggregateTerminalScoreRequest,
    GuardedPreterminalBoundary,
)


_READER_TOKEN = object()
_VIEW_TOKEN = object()


@dataclass(frozen=True, slots=True)
class CaseRoutingDiagnostic:
    case_id: str
    selected_action_id: str
    oracle_action_id: str
    spearman_rank_correlation: float
    normalized_oracle_gap: float

    def __post_init__(self) -> None:
        case_id = str(self.case_id)
        selected = str(self.selected_action_id)
        oracle = str(self.oracle_action_id)
        spearman = float(self.spearman_rank_correlation)
        gap = float(self.normalized_oracle_gap)
        allowed = {P_ACTION_ID, *ACTION_IDS}
        if (
            not case_id
            or selected not in allowed
            or oracle not in allowed
            or not math.isfinite(spearman)
            or not -1.0 <= spearman <= 1.0
            or not math.isfinite(gap)
            or not 0.0 <= gap <= 1.0
        ):
            raise ProtocolError("OE-PPUR v3 terminal case diagnostic drifted.")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "selected_action_id", selected)
        object.__setattr__(self, "oracle_action_id", oracle)
        object.__setattr__(self, "spearman_rank_correlation", spearman)
        object.__setattr__(self, "normalized_oracle_gap", gap)


class _SealedTerminalInputView:
    __slots__ = (
        "boundary_hash",
        "case_diagnostics",
        "decision_ledger_hash",
        "identity_hash",
        "protected_probabilities",
        "row_case_ids",
        "row_labels",
        "selected_probabilities",
    )

    def __init__(
        self,
        *,
        boundary: GuardedPreterminalBoundary,
        row_case_ids: tuple[str, ...],
        row_labels: tuple[int, ...],
        selected_probabilities: tuple[float, ...],
        protected_probabilities: tuple[float, ...],
        case_diagnostics: tuple[CaseRoutingDiagnostic, ...],
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _VIEW_TOKEN:
            raise ProtocolError("OE-PPUR v3 terminal input view bypassed manager.")
        self.boundary_hash = boundary.receipt_hash
        self.decision_ledger_hash = boundary.decision_ledger_receipt_hash
        self.row_case_ids = row_case_ids
        self.row_labels = row_labels
        self.selected_probabilities = selected_probabilities
        self.protected_probabilities = protected_probabilities
        self.case_diagnostics = case_diagnostics
        self.identity_hash = canonical_hash(
            {
                "schema_version": "oe_ppur_v3_sealed_terminal_input_view_v1",
                "boundary_hash": boundary.receipt_hash,
                "decision_ledger_hash": boundary.decision_ledger_receipt_hash,
                "row_count": len(row_labels),
                "case_count": len(case_diagnostics),
                "row_case_inventory_hash": canonical_hash(row_case_ids),
                "row_label_hash": canonical_hash(row_labels),
                "selected_probability_hash": canonical_hash(selected_probabilities),
                "protected_probability_hash": canonical_hash(
                    protected_probabilities
                ),
                "case_diagnostic_hash": canonical_hash(
                    [
                        (
                            row.case_id,
                            row.selected_action_id,
                            row.oracle_action_id,
                            row.spearman_rank_correlation,
                            row.normalized_oracle_gap,
                        )
                        for row in case_diagnostics
                    ]
                ),
                "raw_values_exported": False,
            }
        )

    def __reduce__(self):  # pragma: no cover - explicit safety seam
        raise TypeError("OE-PPUR v3 sealed terminal inputs cannot be serialized.")


class AggregateOnlyLabelReader(ABC):
    """Nominal terminal reader; no method can return raw labels."""

    @abstractmethod
    def _score_aggregate_only(
        self, request: AggregateTerminalScoreRequest
    ) -> AggregateOnlyTerminalReceipt:
        """Open private values only under a live terminal capability."""

    def __reduce__(self):  # pragma: no cover - process-local capability
        raise TypeError("OE-PPUR v3 terminal label readers cannot be serialized.")


class ManagerOwnedManifestLabelReader(AggregateOnlyLabelReader):
    """Concrete no-callback aggregate evaluator over one sealed input view."""

    __slots__ = ("_consumed", "_view")

    def __init__(
        self,
        view: _SealedTerminalInputView,
        *,
        _factory_token: object | None = None,
    ) -> None:
        if (
            _factory_token is not _READER_TOKEN
            or type(view) is not _SealedTerminalInputView
        ):
            raise ProtocolError("OE-PPUR v3 manifest label reader bypassed manager.")
        self._view: _SealedTerminalInputView | None = view
        self._consumed = False

    def _score_aggregate_only(
        self, request: AggregateTerminalScoreRequest
    ) -> AggregateOnlyTerminalReceipt:
        if self._consumed or self._view is None:
            raise ProtocolError("OE-PPUR v3 manifest label reader was replayed.")
        view = self._view
        self._consumed = True
        self._view = None
        if (
            request.boundary_receipt_hash != view.boundary_hash
            or request.decision_ledger_receipt_hash != view.decision_ledger_hash
        ):
            raise ProtocolError("OE-PPUR v3 terminal input lineage drifted.")

        selected_bacc = _balanced_accuracy(
            view.row_labels, view.selected_probabilities
        )
        protected_bacc = _balanced_accuracy(
            view.row_labels, view.protected_probabilities
        )
        selected_brier = _brier(view.row_labels, view.selected_probabilities)
        protected_brier = _brier(view.row_labels, view.protected_probabilities)
        selected_log = _log_loss(view.row_labels, view.selected_probabilities)
        protected_log = _log_loss(view.row_labels, view.protected_probabilities)
        routed_count = sum(
            row.selected_action_id != P_ACTION_ID for row in view.case_diagnostics
        )
        exact_p_count = EXPECTED_CASE_COUNT - routed_count
        metrics = (
            ("selected_balanced_accuracy", selected_bacc),
            ("protected_p_balanced_accuracy", protected_bacc),
            ("selected_minus_p_balanced_accuracy", selected_bacc - protected_bacc),
            ("selected_brier_score", selected_brier),
            ("protected_p_brier_score", protected_brier),
            ("p_minus_selected_brier_score", protected_brier - selected_brier),
            ("selected_log_loss", selected_log),
            ("protected_p_log_loss", protected_log),
            ("p_minus_selected_log_loss", protected_log - selected_log),
            ("routing_coverage", routed_count / EXPECTED_CASE_COUNT),
            (
                "top1_oracle_agreement",
                sum(
                    row.selected_action_id == row.oracle_action_id
                    for row in view.case_diagnostics
                )
                / EXPECTED_CASE_COUNT,
            ),
            (
                "spearman_rank_correlation",
                sum(
                    row.spearman_rank_correlation for row in view.case_diagnostics
                )
                / EXPECTED_CASE_COUNT,
            ),
            (
                "normalized_oracle_gap",
                sum(row.normalized_oracle_gap for row in view.case_diagnostics)
                / EXPECTED_CASE_COUNT,
            ),
            ("exact_p_rate", exact_p_count / EXPECTED_CASE_COUNT),
        )
        if tuple(key for key, _ in metrics) != ALLOWED_AGGREGATE_METRICS:
            raise ProtocolError("OE-PPUR v3 terminal metric inventory drifted.")
        return AggregateOnlyTerminalReceipt(
            boundary_receipt_hash=request.boundary_receipt_hash,
            decision_ledger_receipt_hash=request.decision_ledger_receipt_hash,
            evaluated_case_count=EXPECTED_CASE_COUNT,
            routed_case_count=routed_count,
            exact_p_fallback_count=exact_p_count,
            aggregate_metrics=metrics,
        )


def seal_manager_owned_terminal_input(
    boundary: GuardedPreterminalBoundary,
    *,
    row_case_ids: Sequence[object],
    row_labels: Sequence[object],
    selected_probabilities: Sequence[object],
    protected_probabilities: Sequence[object],
    case_diagnostics: Sequence[CaseRoutingDiagnostic],
) -> _SealedTerminalInputView:
    """Validate full coverage and return an opaque non-persistable view."""

    if type(boundary) is not GuardedPreterminalBoundary:
        raise ProtocolError("OE-PPUR v3 terminal input boundary is untyped.")
    cases = tuple(str(value) for value in row_case_ids)
    labels = tuple(int(value) for value in row_labels)
    selected = tuple(float(value) for value in selected_probabilities)
    protected = tuple(float(value) for value in protected_probabilities)
    diagnostics = tuple(case_diagnostics)
    diagnostic_ids = tuple(row.case_id for row in diagnostics)
    if (
        len(cases) != EXPECTED_TEST_ROW_COUNT
        or len(labels) != EXPECTED_TEST_ROW_COUNT
        or len(selected) != EXPECTED_TEST_ROW_COUNT
        or len(protected) != EXPECTED_TEST_ROW_COUNT
        or any(value not in (0, 1) for value in labels)
        or not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in (*selected, *protected)
        )
        or len(diagnostics) != EXPECTED_CASE_COUNT
        or any(type(row) is not CaseRoutingDiagnostic for row in diagnostics)
        or len(set(diagnostic_ids)) != EXPECTED_CASE_COUNT
        or set(cases) != set(diagnostic_ids)
        or len(set(cases)) != EXPECTED_CASE_COUNT
    ):
        raise ProtocolError("OE-PPUR v3 terminal row/case coverage drifted.")
    exact_p_ids = {
        row.case_id for row in diagnostics if row.selected_action_id == P_ACTION_ID
    }
    if len(exact_p_ids) != boundary.exact_p_fallback_count:
        raise ProtocolError("OE-PPUR v3 terminal exact-P coverage drifted.")
    for case_id, selected_probability, protected_probability in zip(
        cases, selected, protected, strict=True
    ):
        if case_id in exact_p_ids and selected_probability != protected_probability:
            raise ProtocolError("OE-PPUR v3 exact-P row probability drifted.")
    return _SealedTerminalInputView(
        boundary=boundary,
        row_case_ids=cases,
        row_labels=labels,
        selected_probabilities=selected,
        protected_probabilities=protected,
        case_diagnostics=diagnostics,
        _factory_token=_VIEW_TOKEN,
    )


def build_manager_owned_manifest_label_reader(
    view: _SealedTerminalInputView,
) -> ManagerOwnedManifestLabelReader:
    return ManagerOwnedManifestLabelReader(view, _factory_token=_READER_TOKEN)


def build_physical_manifest_label_reader(
    *args: object, **kwargs: object
) -> NoReturn:
    """Remain closed until input #7 and resolved admission both exist."""

    raise ProtocolError(
        "OE-PPUR v3 physical terminal label reader is closed: the v3 amendment "
        "and resolved seven-input admission are absent."
    )


def _balanced_accuracy(
    labels: tuple[int, ...], probabilities: tuple[float, ...]
) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ProtocolError("OE-PPUR v3 terminal labels lack both binary classes.")
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
        + (1 - label)
        * math.log(min(1.0 - epsilon, max(epsilon, 1.0 - probability)))
        for label, probability in zip(labels, probabilities, strict=True)
    ) / len(labels)


def _read_authorized_aggregates(
    reader: AggregateOnlyLabelReader,
    request: AggregateTerminalScoreRequest,
    *,
    _token: object,
) -> AggregateOnlyTerminalReceipt:
    if _token is not _READER_TOKEN or not isinstance(
        reader, AggregateOnlyLabelReader
    ):
        raise ProtocolError("OE-PPUR v3 terminal label read was not authorized.")
    result = reader._score_aggregate_only(request)
    if type(result) is not AggregateOnlyTerminalReceipt:
        raise ProtocolError("OE-PPUR v3 terminal label reader leaked its return type.")
    return result


__all__ = (
    "AggregateOnlyLabelReader",
    "CaseRoutingDiagnostic",
    "ManagerOwnedManifestLabelReader",
    "build_manager_owned_manifest_label_reader",
    "build_physical_manifest_label_reader",
    "seal_manager_owned_terminal_input",
)
