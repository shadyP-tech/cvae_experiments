"""One-shot process-local aggregate terminal capability for OE-PPUR v4."""

from __future__ import annotations

import secrets
from threading import Lock

from ....protocol import ProtocolError
from ..hashing import canonical_hash
from .contracts import (
    AggregateOnlyTerminalReceipt,
    GuardedPreterminalBoundary,
    _issue_terminal_request,
    assert_aggregate_only_payload,
)
from .label_reader import (
    ManagerOwnedManifestLabelReader,
    _READER_TOKEN,
    _read_authorized_aggregates,
)


_CAPABILITY_TOKEN = object()
_EVALUATOR_TOKEN = object()


class AggregateOnlyTerminalEvaluator:
    """Source-sealed evaluator owning one aggregate-only label reader."""

    __slots__ = ("_reader",)

    def __init__(
        self,
        reader: ManagerOwnedManifestLabelReader,
        *,
        _factory_token: object | None = None,
    ) -> None:
        if (
            _factory_token is not _EVALUATOR_TOKEN
            or type(reader) is not ManagerOwnedManifestLabelReader
        ):
            raise ProtocolError("OE-PPUR v4 terminal evaluator bypassed manager.")
        self._reader = reader

    def _score(self, request):
        result = _read_authorized_aggregates(
            self._reader,
            request,
            _token=_READER_TOKEN,
        )
        assert_aggregate_only_payload(result.to_payload())
        return result

    def __reduce__(self):  # pragma: no cover - process-local capability
        raise TypeError("OE-PPUR v4 terminal evaluators cannot be serialized.")


class TerminalAggregateCapability:
    """Consume-before-use authority for exactly one terminal evaluation."""

    __slots__ = ("_boundary", "_capability_hash", "_consumed", "_evaluator", "_lock")

    def __init__(
        self,
        boundary: GuardedPreterminalBoundary,
        evaluator: AggregateOnlyTerminalEvaluator,
        *,
        _factory_token: object | None = None,
    ) -> None:
        if (
            _factory_token is not _CAPABILITY_TOKEN
            or type(boundary) is not GuardedPreterminalBoundary
            or type(evaluator) is not AggregateOnlyTerminalEvaluator
        ):
            raise ProtocolError("OE-PPUR v4 terminal capability bypassed gating.")
        self._boundary = boundary
        self._evaluator: AggregateOnlyTerminalEvaluator | None = evaluator
        self._consumed = False
        self._lock = Lock()
        self._capability_hash = canonical_hash(
            {
                "schema_version": "oe_ppur_v4_terminal_capability_v1",
                "boundary_receipt_hash": boundary.receipt_hash,
                "nonce": secrets.token_hex(32),
                "one_shot": True,
                "process_local": True,
                "aggregate_only": True,
            }
        )

    @property
    def consumed(self) -> bool:
        with self._lock:
            return self._consumed

    @property
    def capability_hash(self) -> str:
        return self._capability_hash

    def score_aggregates(self) -> AggregateOnlyTerminalReceipt:
        with self._lock:
            if self._consumed or self._evaluator is None:
                raise ProtocolError("OE-PPUR v4 terminal capability was replayed.")
            self._consumed = True
            evaluator = self._evaluator
            self._evaluator = None
        request = _issue_terminal_request(
            self._boundary,
            capability_hash=self._capability_hash,
        )
        result = evaluator._score(request)
        if (
            result.boundary_receipt_hash != self._boundary.receipt_hash
            or result.decision_ledger_receipt_hash
            != self._boundary.decision_ledger_receipt_hash
            or result.evaluated_case_count != self._boundary.case_count
            or result.exact_p_fallback_count
            != self._boundary.exact_p_fallback_count
        ):
            raise ProtocolError("OE-PPUR v4 terminal aggregate lineage drifted.")
        return result

    def __reduce__(self):  # pragma: no cover - process-local capability
        raise TypeError("OE-PPUR v4 terminal capabilities cannot be serialized.")

    def __copy__(self):  # pragma: no cover - process-local capability
        raise TypeError("OE-PPUR v4 terminal capabilities cannot be copied.")

    def __deepcopy__(self, memo):  # pragma: no cover - process-local capability
        raise TypeError("OE-PPUR v4 terminal capabilities cannot be copied.")


def issue_terminal_aggregate_capability(
    boundary: GuardedPreterminalBoundary,
    *,
    reader: ManagerOwnedManifestLabelReader,
) -> TerminalAggregateCapability:
    if (
        type(boundary) is not GuardedPreterminalBoundary
        or type(reader) is not ManagerOwnedManifestLabelReader
    ):
        raise ProtocolError("OE-PPUR v4 terminal capability inputs are untyped.")
    evaluator = _build_manager_owned_terminal_evaluator(reader)
    return TerminalAggregateCapability(
        boundary,
        evaluator,
        _factory_token=_CAPABILITY_TOKEN,
    )


def _build_manager_owned_terminal_evaluator(
    reader: ManagerOwnedManifestLabelReader,
) -> AggregateOnlyTerminalEvaluator:
    """Build a concrete evaluator only around the physical manifest reader."""

    if type(reader) is not ManagerOwnedManifestLabelReader:
        raise ProtocolError("OE-PPUR v4 terminal evaluator reader is untyped.")
    return AggregateOnlyTerminalEvaluator(reader, _factory_token=_EVALUATOR_TOKEN)


__all__ = (
    "issue_terminal_aggregate_capability",
)
