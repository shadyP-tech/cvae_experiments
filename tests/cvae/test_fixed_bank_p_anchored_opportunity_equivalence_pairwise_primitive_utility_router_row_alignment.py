from __future__ import annotations

from dataclasses import replace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1 import identity
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.execution import memmap
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.hashing import canonical_hash
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.manifest_contract import (
    ANNOTATION_MANIFEST_CONTENT_SHA256,
    ANNOTATION_MANIFEST_MEMBER,
    CANONICAL_TERMINAL_CASE_COUNTS_BY_CENTER,
    CANONICAL_TERMINAL_CASE_INVENTORY,
    CANONICAL_TERMINAL_ROW_COUNTS_BY_CENTER,
    CANONICAL_TERMINAL_SPLIT,
    build_canonical_terminal_manifest_receipt,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _manifest_receipt():
    return build_canonical_terminal_manifest_receipt(
        annotation_artifact_id=identity.ANNOTATION_MANIFEST_ARTIFACT_ID,
        manifest_member=ANNOTATION_MANIFEST_MEMBER,
        manifest_content_sha256=ANNOTATION_MANIFEST_CONTENT_SHA256,
        split=CANONICAL_TERMINAL_SPLIT,
        eligible_center_ids=identity.CENTERS,
        row_count=identity.EXPECTED_TEST_ROW_COUNT,
        case_count=identity.EXPECTED_CASE_COUNT,
        row_counts_by_center=CANONICAL_TERMINAL_ROW_COUNTS_BY_CENTER,
        case_counts_by_center=CANONICAL_TERMINAL_CASE_COUNTS_BY_CENTER,
        case_inventory=CANONICAL_TERMINAL_CASE_INVENTORY,
    )


def _synthetic_manifest_rows() -> tuple[memmap.CanonicalRowIdentity, ...]:
    rows: list[memmap.CanonicalRowIdentity] = []
    ordinal = 0
    for center, center_count in CANONICAL_TERMINAL_ROW_COUNTS_BY_CENTER:
        cases = tuple(
            case
            for case_center, case in CANONICAL_TERMINAL_CASE_INVENTORY
            if case_center == center
        )
        base, remainder = divmod(center_count, len(cases))
        for case_index, case in enumerate(cases):
            count = base + int(case_index < remainder)
            for _ in range(count):
                rows.append(
                    memmap.CanonicalRowIdentity(
                        row_ordinal=ordinal,
                        manifest_row_index=ordinal,
                        evaluation_row_id=f"synthetic-terminal-row-{ordinal:05d}",
                        center_id=center,
                        case_id=case,
                    )
                )
                ordinal += 1
    assert ordinal == identity.EXPECTED_TEST_ROW_COUNT
    return tuple(rows)


@pytest.fixture()
def exact_alignment(monkeypatch: pytest.MonkeyPatch):
    manifest_rows = _synthetic_manifest_rows()
    row_order_hash = canonical_hash(
        [row.evaluation_row_id for row in manifest_rows]
    )
    monkeypatch.setattr(
        memmap,
        "EXPECTED_EXECUTABLE_TEST_CACHE_ROW_ORDER_SHA256",
        row_order_hash,
    )
    receipt = memmap.build_canonical_row_alignment_receipt(
        manifest_receipt=_manifest_receipt(),
        manifest_rows=manifest_rows,
        rows=manifest_rows,
        cache_content_sha256=memmap.EXPECTED_EXECUTABLE_TEST_CACHE_CONTENT_SHA256,
        cache_row_order_sha256=row_order_hash,
    )
    return receipt


def test_canonical_alignment_binds_all_rows_cases_centers_and_cache_pin(
    exact_alignment: memmap.CanonicalRowAlignmentReceipt,
) -> None:
    receipt = exact_alignment
    assert receipt.row_count == 9928
    assert receipt.case_count == 218
    assert receipt.manifest_receipt.case_inventory == (
        CANONICAL_TERMINAL_CASE_INVENTORY
    )
    assert receipt.to_row_index_receipt().row_index_sha256 == (
        receipt.row_index_sha256
    )
    assert receipt.to_payload()["labels_present"] is False
    assert memmap.validate_canonical_row_alignment_receipt(receipt) is receipt


def test_alignment_factory_rejects_bypass_cross_case_and_manifest_order_drift(
    exact_alignment: memmap.CanonicalRowAlignmentReceipt,
) -> None:
    with pytest.raises(ProtocolError, match="guarded factory"):
        memmap.CanonicalRowAlignmentReceipt(
            manifest_receipt=exact_alignment.manifest_receipt,
            manifest_rows=exact_alignment.manifest_rows,
            rows=exact_alignment.rows,
            cache_content_sha256=exact_alignment.cache_content_sha256,
            cache_row_order_sha256=exact_alignment.cache_row_order_sha256,
        )

    poisoned_cache_rows = list(exact_alignment.rows)
    first = poisoned_cache_rows[0]
    poisoned_cache_rows[0] = replace(first, case_id="302")
    with pytest.raises(ProtocolError, match="physical cache order drifted"):
        memmap.build_canonical_row_alignment_receipt(
            manifest_receipt=exact_alignment.manifest_receipt,
            manifest_rows=exact_alignment.manifest_rows,
            rows=poisoned_cache_rows,
            cache_content_sha256=exact_alignment.cache_content_sha256,
            cache_row_order_sha256=exact_alignment.cache_row_order_sha256,
        )

    poisoned_manifest_rows = list(exact_alignment.manifest_rows)
    poisoned_manifest_rows[0], poisoned_manifest_rows[1] = (
        poisoned_manifest_rows[1],
        poisoned_manifest_rows[0],
    )
    with pytest.raises(ProtocolError, match="inventory is not exact"):
        memmap.build_canonical_row_alignment_receipt(
            manifest_receipt=exact_alignment.manifest_receipt,
            manifest_rows=poisoned_manifest_rows,
            rows=exact_alignment.rows,
            cache_content_sha256=exact_alignment.cache_content_sha256,
            cache_row_order_sha256=exact_alignment.cache_row_order_sha256,
        )


def test_physical_row_reordering_is_rejected(
    exact_alignment: memmap.CanonicalRowAlignmentReceipt,
) -> None:
    reordered = list(exact_alignment.rows)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    reordered = [replace(row, row_ordinal=index) for index, row in enumerate(reordered)]
    with pytest.raises(ProtocolError, match="physical cache order drifted"):
        memmap.build_canonical_row_alignment_receipt(
            manifest_receipt=exact_alignment.manifest_receipt,
            manifest_rows=exact_alignment.manifest_rows,
            rows=reordered,
            cache_content_sha256=exact_alignment.cache_content_sha256,
            cache_row_order_sha256=exact_alignment.cache_row_order_sha256,
        )
