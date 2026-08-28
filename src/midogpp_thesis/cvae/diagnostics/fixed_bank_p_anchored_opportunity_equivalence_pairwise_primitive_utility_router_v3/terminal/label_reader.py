"""Manager-owned terminal label reader and aggregate scoring for OE-PPUR v3.

The physical factory is executable only from an authorization-ready resolved
seven-input bundle and two independent artifact-only attestations. Raw row/case
values have no return or persistence method and are destroyed immediately after
the one authorized aggregate read.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import math
from typing import Sequence

from ....protocol import ProtocolError
from ..config import ResolvedV3ConfigBundle, validate_authorization_ready_config
from ..hashing import canonical_hash
from ..identity import (
    EXPECTED_CASE_COUNT,
    EXPECTED_TEST_MANIFEST_SHA256,
    EXPECTED_TEST_ROW_COUNT,
    P_ACTION_ID,
)
from .authority import _read_regular_file_bytes, validate_resolved_terminal_authority
from .contracts import (
    ALLOWED_AGGREGATE_METRICS,
    AggregateOnlyTerminalReceipt,
    AggregateTerminalScoreRequest,
    ArtifactOnlyPreterminalAttestationReceipt,
    GuardedPreterminalBoundary,
    _TERMINAL_RECEIPT_TOKEN,
    _issue_aggregate_only_terminal_receipt,
)
from .manifest_scoring import (
    CaseRoutingDiagnostic,
    _balanced_accuracy,
    _brier,
    _derive_terminal_values,
    _log_loss,
    _read_aligned_manifest_labels,
    _validate_matrix_ledger_frame_linkage,
)


_READER_TOKEN = object()
_VIEW_TOKEN = object()


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
        available_ranks = tuple(
            float(row.spearman_rank_correlation)
            for row in view.case_diagnostics
            if row.spearman_rank_correlation is not None
        )
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
                "available_case_mean_spearman_rank_correlation",
                sum(available_ranks) / len(available_ranks)
                if available_ranks
                else 0.0,
            ),
            ("rank_diagnostic_coverage", len(available_ranks) / EXPECTED_CASE_COUNT),
            (
                "normalized_oracle_gap",
                sum(row.normalized_oracle_gap for row in view.case_diagnostics)
                / EXPECTED_CASE_COUNT,
            ),
            ("exact_p_rate", exact_p_count / EXPECTED_CASE_COUNT),
        )
        if tuple(key for key, _ in metrics) != ALLOWED_AGGREGATE_METRICS:
            raise ProtocolError("OE-PPUR v3 terminal metric inventory drifted.")
        return _issue_aggregate_only_terminal_receipt(
            boundary_receipt_hash=request.boundary_receipt_hash,
            decision_ledger_receipt_hash=request.decision_ledger_receipt_hash,
            evaluated_case_count=EXPECTED_CASE_COUNT,
            routed_case_count=routed_count,
            exact_p_fallback_count=exact_p_count,
            aggregate_metrics=metrics,
            _manager_token=_TERMINAL_RECEIPT_TOKEN,
        )


def _seal_manager_owned_terminal_input(
    boundary: GuardedPreterminalBoundary,
    *,
    row_case_ids: Sequence[object],
    row_labels: Sequence[object],
    selected_probabilities: Sequence[object],
    protected_probabilities: Sequence[object],
    case_diagnostics: Sequence[CaseRoutingDiagnostic],
    _manager_token: object,
) -> _SealedTerminalInputView:
    """Manager-internal raw-value gate; never exported as a package API.

    The production caller is the physical manifest reader after resolved
    seven-input admission and the two independent artifact-only attestations.
    Keeping the construction token explicit prevents a caller from turning an
    arbitrary in-memory label vector into terminal authority.
    """

    if (
        _manager_token is not _VIEW_TOKEN
        or type(boundary) is not GuardedPreterminalBoundary
        or len(boundary.preterminal_attestation_hashes) != 2
        or len(set(boundary.preterminal_attestation_hashes)) != 2
    ):
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


def _build_manager_owned_manifest_label_reader(
    view: _SealedTerminalInputView,
) -> ManagerOwnedManifestLabelReader:
    return ManagerOwnedManifestLabelReader(view, _factory_token=_READER_TOKEN)


def build_physical_manifest_label_reader(
    bundle: ResolvedV3ConfigBundle,
    *,
    boundary: GuardedPreterminalBoundary,
    preterminal_result: object,
    execution_request: object,
    persisted_artifact: object,
    attestations: Sequence[ArtifactOnlyPreterminalAttestationReceipt],
) -> ManagerOwnedManifestLabelReader:
    """Open canonical labels only after all physical authority is complete.

    No caller supplies labels, selected probabilities, oracle decisions, ranks,
    or case diagnostics.  All of those values are derived inside this manager
    boundary from the attested result, canonical cache row bindings, and the
    exact manifest.
    """

    from ..execution.preterminal_artifact import PersistedPreterminalArtifact
    from ..execution.services import (
        CanonicalPreterminalResult,
        CanonicalRouterExecutionRequest,
    )

    if (
        type(bundle) is not ResolvedV3ConfigBundle
        or type(boundary) is not GuardedPreterminalBoundary
        or type(preterminal_result) is not CanonicalPreterminalResult
        or type(execution_request) is not CanonicalRouterExecutionRequest
        or type(persisted_artifact) is not PersistedPreterminalArtifact
    ):
        raise ProtocolError("OE-PPUR v3 physical terminal boundary is untyped.")
    config = validate_authorization_ready_config(bundle.config)
    result = preterminal_result
    request = execution_request
    artifact = persisted_artifact
    rows = tuple(attestations)
    if (
        len(rows) != 2
        or any(type(row) is not ArtifactOnlyPreterminalAttestationReceipt for row in rows)
        or tuple(row.receipt_hash for row in rows)
        != boundary.preterminal_attestation_hashes
        or len({row.process_pid for row in rows}) != 2
        or any(
            row.sealed_ledger_receipt_hash != result.decision_ledger.ledger_hash
            or row.artifact_file_sha256 != artifact.artifact_file_sha256
            or row.artifact_file_identity_sha256
            != artifact.artifact_file_identity_sha256
            for row in rows
        )
        or result.request_hash != request.request_hash
        or result.result_hash != artifact.result_hash
        or result.decision_ledger.ledger_hash != artifact.decision_ledger_hash
        or result.decision_ledger.ledger_hash
        != boundary.decision_ledger_receipt_hash
        or result.seven_input_contract_hash != config.seven_input_contract_hash
        or result.source_seal_hash != boundary.source_seal_hash
        or result.source_training_surface_receipt_hash
        != boundary.source_training_surface_receipt_hash
        or boundary.seven_input_contract_hash != config.seven_input_contract_hash
        or boundary.case_inventory_sha256
        != EXPECTED_TERMINAL_CASE_INVENTORY_SHA256
        or boundary.exact_p_fallback_count != result.decision_ledger.exact_p_count
    ):
        raise ProtocolError("OE-PPUR v3 terminal attestation lineage drifted.")
    frame = request.frame
    matrix = result.probability_matrix
    ledger = result.decision_ledger
    _validate_matrix_ledger_frame_linkage(matrix, ledger, frame)
    validate_resolved_terminal_authority(
        bundle,
        source_training_surface_receipt_hash=(
            result.source_training_surface_receipt_hash
        ),
    )

    manifest_path = bundle.input_bindings[4].path
    raw = _read_regular_file_bytes(
        manifest_path,
        maximum_bytes=16 * 1024 * 1024,
        role="canonical manifest",
    )
    if hashlib.sha256(raw).hexdigest() != EXPECTED_TEST_MANIFEST_SHA256:
        raise ProtocolError("OE-PPUR v3 canonical manifest bytes drifted.")
    labels = _read_aligned_manifest_labels(raw, frame=frame)
    selected, protected, diagnostics = _derive_terminal_values(
        matrix,
        ledger,
        frame,
        labels,
    )
    view = _seal_manager_owned_terminal_input(
        boundary,
        row_case_ids=tuple(row.case_id for row in frame.rows),
        row_labels=labels,
        selected_probabilities=selected,
        protected_probabilities=protected,
        case_diagnostics=diagnostics,
        _manager_token=_VIEW_TOKEN,
    )
    return _build_manager_owned_manifest_label_reader(view)


def _read_authorized_aggregates(
    reader: ManagerOwnedManifestLabelReader,
    request: AggregateTerminalScoreRequest,
    *,
    _token: object,
) -> AggregateOnlyTerminalReceipt:
    if (
        _token is not _READER_TOKEN
        or type(reader) is not ManagerOwnedManifestLabelReader
    ):
        raise ProtocolError("OE-PPUR v3 terminal label read was not authorized.")
    result = reader._score_aggregate_only(request)
    if type(result) is not AggregateOnlyTerminalReceipt:
        raise ProtocolError("OE-PPUR v3 terminal label reader leaked its return type.")
    return result


__all__ = (
    "build_physical_manifest_label_reader",
    "validate_resolved_terminal_authority",
)
