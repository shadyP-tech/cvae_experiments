"""Canonical admitted-row lineage for the executable OE-PPUR v2 router.

The probability surface may only be indexed by the immutable test-cache order
admitted by the exact-six-input gate.  This module converts that guarded
admission receipt into one guarded row-binding receipt; callers cannot nominate
an unrelated row index, alignment receipt, row count, or case inventory.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field

from ...protocol import ProtocolError
from .execution_admission import SixInputAdmissionReceipt
from .hashing import canonical_hash, require_sha256
from .identity import (
    EXPECTED_CASE_COUNT,
    EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_MANIFEST_SHA256,
    EXPECTED_TEST_ROW_COUNT,
)


ROW_ALIGNMENT_SCHEMA = "oe_ppur_v2_admitted_row_alignment_v1"
ROW_BINDING_SCHEMA = "oe_ppur_v2_canonical_admitted_row_binding_v1"
_ROW_BINDING_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class CanonicalAdmittedRowBindingReceipt:
    """Guarded proof of the only row/case inventory admitted for one run."""

    six_input_admission_hash: str
    input_binding_hash: str
    cache_content_sha256: str
    cache_row_order_sha256: str
    manifest_sha256: str
    _factory_token: InitVar[object | None] = None
    row_count: int = field(init=False)
    case_count: int = field(init=False)
    case_inventory_sha256: str = field(init=False)
    row_index_sha256: str = field(init=False)
    row_alignment_receipt_hash: str = field(init=False)
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _ROW_BINDING_FACTORY_TOKEN:
            raise ProtocolError(
                "OE-PPUR v2 row binding bypassed six-input admission."
            )
        admission_hash = require_sha256(
            self.six_input_admission_hash, "row-binding admission hash"
        )
        input_binding_hash = require_sha256(
            self.input_binding_hash, "row-binding input hash"
        )
        cache_content = require_sha256(
            self.cache_content_sha256, "row-binding cache-content hash"
        )
        cache_row_order = require_sha256(
            self.cache_row_order_sha256, "row-binding cache row-order hash"
        )
        manifest = require_sha256(
            self.manifest_sha256, "row-binding manifest hash"
        )
        if (
            cache_content != EXPECTED_TEST_CACHE_CONTENT_HASH
            or cache_row_order != EXPECTED_TEST_CACHE_ROW_ORDER_HASH
            or manifest != EXPECTED_TEST_MANIFEST_SHA256
        ):
            raise ProtocolError("OE-PPUR v2 admitted row identity drifted.")

        # The cache row-order digest is the canonical row-index digest.  The
        # alignment receipt additionally closes the cache/manifest/case joins.
        row_index = cache_row_order
        alignment = canonical_hash(
            {
                "schema_version": ROW_ALIGNMENT_SCHEMA,
                "six_input_admission_hash": admission_hash,
                "input_binding_hash": input_binding_hash,
                "cache_content_sha256": cache_content,
                "cache_row_order_sha256": cache_row_order,
                "manifest_sha256": manifest,
                "row_index_sha256": row_index,
                "row_count": EXPECTED_TEST_ROW_COUNT,
                "case_count": EXPECTED_CASE_COUNT,
                "case_inventory_sha256": (
                    EXPECTED_TERMINAL_CASE_INVENTORY_SHA256
                ),
                "split": "test",
                "excluded_center_ids": ["4"],
                "labels_present": False,
            }
        )
        object.__setattr__(self, "six_input_admission_hash", admission_hash)
        object.__setattr__(self, "input_binding_hash", input_binding_hash)
        object.__setattr__(self, "cache_content_sha256", cache_content)
        object.__setattr__(self, "cache_row_order_sha256", cache_row_order)
        object.__setattr__(self, "manifest_sha256", manifest)
        object.__setattr__(self, "row_count", EXPECTED_TEST_ROW_COUNT)
        object.__setattr__(self, "case_count", EXPECTED_CASE_COUNT)
        object.__setattr__(
            self,
            "case_inventory_sha256",
            EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
        )
        object.__setattr__(self, "row_index_sha256", row_index)
        object.__setattr__(self, "row_alignment_receipt_hash", alignment)
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": ROW_BINDING_SCHEMA,
            "six_input_admission_hash": self.six_input_admission_hash,
            "input_binding_hash": self.input_binding_hash,
            "cache_content_sha256": self.cache_content_sha256,
            "cache_row_order_sha256": self.cache_row_order_sha256,
            "manifest_sha256": self.manifest_sha256,
            "row_index_sha256": self.row_index_sha256,
            "row_alignment_receipt_hash": self.row_alignment_receipt_hash,
            "row_count": self.row_count,
            "case_count": self.case_count,
            "case_inventory_sha256": self.case_inventory_sha256,
            "split": "test",
            "labels_present": False,
            "terminal_capability_opened": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def derive_admitted_row_binding(
    admission_receipt: SixInputAdmissionReceipt,
) -> CanonicalAdmittedRowBindingReceipt:
    """Derive the closed-world row binding from one exact-six admission."""

    admission = _validate_six_input_admission_receipt(admission_receipt)
    return _issue_canonical_admitted_row_binding(
        six_input_admission_hash=admission.receipt_hash,
        input_binding_hash=admission.input_binding_hash,
        cache_content_sha256=admission.cache_content_sha256,
        cache_row_order_sha256=admission.cache_row_order_sha256,
        manifest_sha256=admission.manifest_sha256,
    )


def validate_admitted_row_binding(
    receipt: object,
    *,
    admission_receipt: SixInputAdmissionReceipt | None = None,
) -> CanonicalAdmittedRowBindingReceipt:
    """Recompute a typed binding and optionally exact-match its admission."""

    if not isinstance(receipt, CanonicalAdmittedRowBindingReceipt):
        raise ProtocolError("OE-PPUR v2 row binding is untyped.")
    rebuilt = _issue_canonical_admitted_row_binding(
        six_input_admission_hash=receipt.six_input_admission_hash,
        input_binding_hash=receipt.input_binding_hash,
        cache_content_sha256=receipt.cache_content_sha256,
        cache_row_order_sha256=receipt.cache_row_order_sha256,
        manifest_sha256=receipt.manifest_sha256,
    )
    if rebuilt != receipt:
        raise ProtocolError("OE-PPUR v2 row-binding receipt hash drifted.")
    if admission_receipt is not None:
        expected = derive_admitted_row_binding(admission_receipt)
        if expected != receipt:
            raise ProtocolError(
                "OE-PPUR v2 row binding belongs to an unrelated admission."
            )
    return receipt


def _validate_six_input_admission_receipt(
    receipt: object,
) -> SixInputAdmissionReceipt:
    if not isinstance(receipt, SixInputAdmissionReceipt):
        raise ProtocolError(
            "OE-PPUR v2 row binding requires a six-input admission receipt."
        )
    if receipt.receipt_hash != canonical_hash(receipt._body()):
        raise ProtocolError("OE-PPUR v2 six-input admission receipt drifted.")
    if (
        receipt.cache_content_sha256 != EXPECTED_TEST_CACHE_CONTENT_HASH
        or receipt.cache_row_order_sha256
        != EXPECTED_TEST_CACHE_ROW_ORDER_HASH
        or receipt.manifest_sha256 != EXPECTED_TEST_MANIFEST_SHA256
    ):
        raise ProtocolError("OE-PPUR v2 six-input row lineage drifted.")
    return receipt


def _issue_canonical_admitted_row_binding(
    **fields: object,
) -> CanonicalAdmittedRowBindingReceipt:
    return CanonicalAdmittedRowBindingReceipt(
        **fields,
        _factory_token=_ROW_BINDING_FACTORY_TOKEN,
    )


__all__ = (
    "CanonicalAdmittedRowBindingReceipt",
    "ROW_ALIGNMENT_SCHEMA",
    "ROW_BINDING_SCHEMA",
    "derive_admitted_row_binding",
    "validate_admitted_row_binding",
)
