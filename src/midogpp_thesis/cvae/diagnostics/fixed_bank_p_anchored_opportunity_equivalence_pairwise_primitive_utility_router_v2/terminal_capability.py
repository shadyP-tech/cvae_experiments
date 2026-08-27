"""One-shot aggregate-only terminal capability for OE-PPUR v2.

The capability is process-local and can be issued only after a typed
preterminal decision ledger has an exactly matching preterminal attestation.
It never contains or exposes manifest paths, raw labels, per-row labels, or
per-case labels.  Its scorer receives a hash-and-count request and may return
only the canonical aggregate terminal receipt.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
import secrets
from threading import Lock

from ...protocol import ProtocolError
from .execution.decision_receipts import (
    TypedPreterminalDecisionLedgerReceipt,
    validate_typed_preterminal_decision_ledger,
)
from .fresh_process_validation import (
    ArtifactFreshProcessAttestationReceipt,
    validate_artifact_fresh_process_attestation,
)
from .hashing import canonical_hash, require_sha256
from .phase_contracts import (
    AggregateOnlyTerminalReceipt,
    assert_aggregate_only_payload,
)


_BOUNDARY_FACTORY_TOKEN = object()
_REQUEST_FACTORY_TOKEN = object()
_CAPABILITY_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class GuardedPreterminalBoundary:
    """Hash-only proof that one ledger and preterminal attestation match."""

    preterminal_ledger_receipt_hash: str
    preterminal_attestation_receipt_hash: str
    preterminal_ledger_file_sha256: str
    preterminal_ledger_file_identity_sha256: str
    six_input_admission_hash: str
    input_binding_hash: str
    parsed_probability_matrix_receipt_hash: str
    matrix_content_sha256: str
    row_binding_hash: str
    outer_fold_receipt_hash: str
    decision_source_hash: str
    case_inventory_sha256: str
    opportunity_surface_hash: str
    outer_lineage_surface_hash: str
    case_count: int
    exact_p_fallback_count: int
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _BOUNDARY_FACTORY_TOKEN:
            raise ProtocolError(
                "OE-PPUR v2 terminal boundary requires guarded attestation."
            )
        for role in (
            "preterminal_ledger_receipt_hash",
            "preterminal_attestation_receipt_hash",
            "preterminal_ledger_file_sha256",
            "preterminal_ledger_file_identity_sha256",
            "six_input_admission_hash",
            "input_binding_hash",
            "parsed_probability_matrix_receipt_hash",
            "matrix_content_sha256",
            "row_binding_hash",
            "outer_fold_receipt_hash",
            "decision_source_hash",
            "case_inventory_sha256",
            "opportunity_surface_hash",
            "outer_lineage_surface_hash",
        ):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )
        if (
            type(self.case_count) is not int
            or self.case_count <= 0
            or type(self.exact_p_fallback_count) is not int
            or not 0 <= self.exact_p_fallback_count <= self.case_count
        ):
            raise ProtocolError("OE-PPUR v2 terminal boundary counts drifted.")
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v2_guarded_preterminal_boundary_v1",
            "preterminal_ledger_receipt_hash": (
                self.preterminal_ledger_receipt_hash
            ),
            "preterminal_attestation_receipt_hash": (
                self.preterminal_attestation_receipt_hash
            ),
            "preterminal_ledger_file_sha256": (
                self.preterminal_ledger_file_sha256
            ),
            "preterminal_ledger_file_identity_sha256": (
                self.preterminal_ledger_file_identity_sha256
            ),
            "six_input_admission_hash": self.six_input_admission_hash,
            "input_binding_hash": self.input_binding_hash,
            "parsed_probability_matrix_receipt_hash": (
                self.parsed_probability_matrix_receipt_hash
            ),
            "matrix_content_sha256": self.matrix_content_sha256,
            "row_binding_hash": self.row_binding_hash,
            "outer_fold_receipt_hash": self.outer_fold_receipt_hash,
            "decision_source_hash": self.decision_source_hash,
            "case_inventory_sha256": self.case_inventory_sha256,
            "opportunity_surface_hash": self.opportunity_surface_hash,
            "outer_lineage_surface_hash": self.outer_lineage_surface_hash,
            "case_count": self.case_count,
            "exact_p_fallback_count": self.exact_p_fallback_count,
            "preterminal_attestation_matched": True,
            "raw_paths_present": False,
            "raw_labels_present": False,
        }

    def __reduce__(self):  # pragma: no cover - explicit safety seam
        raise TypeError("OE-PPUR v2 terminal boundaries cannot be serialized.")


@dataclass(frozen=True, slots=True)
class AggregateTerminalScoreRequest:
    """The complete, label-free request visible to a terminal scorer."""

    preterminal_ledger_receipt_hash: str
    preterminal_attestation_receipt_hash: str
    preterminal_ledger_file_sha256: str
    preterminal_ledger_file_identity_sha256: str
    six_input_admission_hash: str
    input_binding_hash: str
    parsed_probability_matrix_receipt_hash: str
    matrix_content_sha256: str
    row_binding_hash: str
    outer_fold_receipt_hash: str
    decision_source_hash: str
    case_inventory_sha256: str
    opportunity_surface_hash: str
    outer_lineage_surface_hash: str
    case_count: int
    exact_p_fallback_count: int
    capability_hash: str
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _REQUEST_FACTORY_TOKEN:
            raise ProtocolError(
                "OE-PPUR v2 terminal score requests require a live capability."
            )
        for role in (
            "preterminal_ledger_receipt_hash",
            "preterminal_attestation_receipt_hash",
            "preterminal_ledger_file_sha256",
            "preterminal_ledger_file_identity_sha256",
            "six_input_admission_hash",
            "input_binding_hash",
            "parsed_probability_matrix_receipt_hash",
            "matrix_content_sha256",
            "row_binding_hash",
            "outer_fold_receipt_hash",
            "decision_source_hash",
            "case_inventory_sha256",
            "opportunity_surface_hash",
            "outer_lineage_surface_hash",
            "capability_hash",
        ):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )
        if (
            type(self.case_count) is not int
            or self.case_count <= 0
            or type(self.exact_p_fallback_count) is not int
            or not 0 <= self.exact_p_fallback_count <= self.case_count
        ):
            raise ProtocolError("OE-PPUR v2 terminal score request drifted.")

    def __reduce__(self):  # pragma: no cover - explicit safety seam
        raise TypeError("OE-PPUR v2 terminal score requests cannot be serialized.")


class AggregateOnlyTerminalScorer(ABC):
    """Nominal terminal interface; receives no raw input or label capability."""

    @abstractmethod
    def score_aggregates(
        self, request: AggregateTerminalScoreRequest
    ) -> AggregateOnlyTerminalReceipt:
        """Return only canonical aggregate terminal diagnostics."""


class TerminalAggregateCapability:
    """Process-local one-shot authority to invoke one aggregate scorer."""

    __slots__ = (
        "_boundary",
        "_capability_hash",
        "_consumed",
        "_lock",
        "_scorer",
    )

    def __init__(
        self,
        boundary: GuardedPreterminalBoundary,
        scorer: AggregateOnlyTerminalScorer,
        *,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _CAPABILITY_FACTORY_TOKEN:
            raise ProtocolError(
                "OE-PPUR v2 terminal capability bypassed guarded issuance."
            )
        if (
            not isinstance(boundary, GuardedPreterminalBoundary)
            or not isinstance(scorer, AggregateOnlyTerminalScorer)
        ):
            raise ProtocolError("OE-PPUR v2 terminal capability is untyped.")
        nonce_hash = canonical_hash(
            {
                "schema_version": "oe_ppur_v2_terminal_capability_nonce_v1",
                "entropy": secrets.token_hex(32),
            }
        )
        self._boundary = boundary
        self._capability_hash = canonical_hash(
            {
                "schema_version": "oe_ppur_v2_terminal_aggregate_capability_v1",
                "guarded_preterminal_boundary_hash": boundary.receipt_hash,
                "nonce_hash": nonce_hash,
                "one_shot": True,
                "process_local": True,
                "aggregate_only": True,
                "raw_paths_exposed": False,
                "raw_labels_exposed": False,
            }
        )
        self._consumed = False
        self._lock = Lock()
        self._scorer: AggregateOnlyTerminalScorer | None = scorer

    @property
    def capability_hash(self) -> str:
        return self._capability_hash

    @property
    def consumed(self) -> bool:
        with self._lock:
            return self._consumed

    def score_aggregates(self) -> AggregateOnlyTerminalReceipt:
        """Consume the capability before invoking its aggregate-only scorer."""

        with self._lock:
            if self._consumed or self._scorer is None:
                raise ProtocolError(
                    "OE-PPUR v2 terminal aggregate capability was replayed."
                )
            self._consumed = True
            scorer = self._scorer
            self._scorer = None
            boundary = self._boundary
        request = AggregateTerminalScoreRequest(
            preterminal_ledger_receipt_hash=(
                boundary.preterminal_ledger_receipt_hash
            ),
            preterminal_attestation_receipt_hash=(
                boundary.preterminal_attestation_receipt_hash
            ),
            preterminal_ledger_file_sha256=(
                boundary.preterminal_ledger_file_sha256
            ),
            preterminal_ledger_file_identity_sha256=(
                boundary.preterminal_ledger_file_identity_sha256
            ),
            six_input_admission_hash=boundary.six_input_admission_hash,
            input_binding_hash=boundary.input_binding_hash,
            parsed_probability_matrix_receipt_hash=(
                boundary.parsed_probability_matrix_receipt_hash
            ),
            matrix_content_sha256=boundary.matrix_content_sha256,
            row_binding_hash=boundary.row_binding_hash,
            outer_fold_receipt_hash=boundary.outer_fold_receipt_hash,
            decision_source_hash=boundary.decision_source_hash,
            case_inventory_sha256=boundary.case_inventory_sha256,
            opportunity_surface_hash=boundary.opportunity_surface_hash,
            outer_lineage_surface_hash=boundary.outer_lineage_surface_hash,
            case_count=boundary.case_count,
            exact_p_fallback_count=boundary.exact_p_fallback_count,
            capability_hash=self._capability_hash,
            _factory_token=_REQUEST_FACTORY_TOKEN,
        )
        result = scorer.score_aggregates(request)
        if type(result) is not AggregateOnlyTerminalReceipt:
            raise ProtocolError(
                "OE-PPUR v2 terminal scorer returned a non-aggregate result."
            )
        _validate_receipt_integrity(result, role="terminal aggregate")
        if (
            result.preterminal_attestation_hash
            != boundary.preterminal_attestation_receipt_hash
            or result.preterminal_ledger_receipt_hash
            != boundary.preterminal_ledger_receipt_hash
            or result.evaluated_case_count != boundary.case_count
        ):
            raise ProtocolError(
                "OE-PPUR v2 terminal aggregate coverage or lineage drifted."
            )
        assert_aggregate_only_payload(result.to_payload())
        return result

    def __reduce__(self):  # pragma: no cover - explicit safety seam
        raise TypeError("OE-PPUR v2 terminal capabilities cannot be serialized.")

    def __copy__(self):  # pragma: no cover - explicit safety seam
        raise TypeError("OE-PPUR v2 terminal capabilities cannot be copied.")

    def __deepcopy__(self, memo: object):  # pragma: no cover - safety seam
        raise TypeError("OE-PPUR v2 terminal capabilities cannot be copied.")


def issue_terminal_aggregate_capability(
    preterminal: TypedPreterminalDecisionLedgerReceipt,
    attestation: ArtifactFreshProcessAttestationReceipt | None,
    *,
    scorer: AggregateOnlyTerminalScorer,
) -> TerminalAggregateCapability:
    """Issue one ephemeral capability after an exact preterminal match."""

    boundary = _guard_preterminal_boundary(preterminal, attestation)
    if not isinstance(scorer, AggregateOnlyTerminalScorer):
        raise ProtocolError(
            "OE-PPUR v2 terminal scorer must implement the nominal interface."
        )
    return TerminalAggregateCapability(
        boundary,
        scorer,
        _factory_token=_CAPABILITY_FACTORY_TOKEN,
    )


def _guard_preterminal_boundary(
    preterminal: TypedPreterminalDecisionLedgerReceipt,
    attestation: ArtifactFreshProcessAttestationReceipt | None,
) -> GuardedPreterminalBoundary:
    if type(preterminal) is not TypedPreterminalDecisionLedgerReceipt:
        raise ProtocolError("OE-PPUR v2 terminal ledger is untyped.")
    if type(attestation) is not ArtifactFreshProcessAttestationReceipt:
        raise ProtocolError(
            "OE-PPUR v2 terminal access requires preterminal attestation."
        )
    validated = validate_typed_preterminal_decision_ledger(preterminal)
    try:
        validated_attestation = validate_artifact_fresh_process_attestation(
            attestation,
            expected_phase="preterminal",
            expected_sealed_receipt_hash=validated.receipt_hash,
        )
    except ProtocolError as exc:
        raise ProtocolError(
            "OE-PPUR v2 terminal access requires matching preterminal attestation."
        ) from exc
    return GuardedPreterminalBoundary(
        preterminal_ledger_receipt_hash=validated.receipt_hash,
        preterminal_attestation_receipt_hash=validated_attestation.receipt_hash,
        preterminal_ledger_file_sha256=(
            validated_attestation.sealed_file_sha256
        ),
        preterminal_ledger_file_identity_sha256=(
            validated_attestation.sealed_file_identity_sha256
        ),
        six_input_admission_hash=validated.six_input_admission_hash,
        input_binding_hash=validated.input_binding_hash,
        parsed_probability_matrix_receipt_hash=(
            validated.parsed_probability_matrix_receipt_hash
        ),
        matrix_content_sha256=validated.matrix_content_sha256,
        row_binding_hash=validated.row_binding_hash,
        outer_fold_receipt_hash=validated.outer_fold_receipt_hash,
        decision_source_hash=validated.decision_source_hash,
        case_inventory_sha256=validated.case_inventory_sha256,
        opportunity_surface_hash=validated.opportunity_surface_hash,
        outer_lineage_surface_hash=validated.outer_lineage_surface_hash,
        case_count=validated.case_count,
        exact_p_fallback_count=validated.exact_p_fallback_count,
        _factory_token=_BOUNDARY_FACTORY_TOKEN,
    )


def _validate_receipt_integrity(value: object, *, role: str) -> None:
    to_payload = getattr(value, "to_payload", None)
    if not callable(to_payload):
        raise ProtocolError(f"OE-PPUR v2 {role} receipt is uninspectable.")
    payload = to_payload()
    if not isinstance(payload, dict):
        raise ProtocolError(f"OE-PPUR v2 {role} receipt payload drifted.")
    body = dict(payload)
    receipt_hash = body.pop("receipt_hash", None)
    if receipt_hash != canonical_hash(body):
        raise ProtocolError(f"OE-PPUR v2 {role} receipt integrity drifted.")


__all__ = (
    "AggregateOnlyTerminalScorer",
    "AggregateTerminalScoreRequest",
    "GuardedPreterminalBoundary",
    "TerminalAggregateCapability",
    "issue_terminal_aggregate_capability",
)
