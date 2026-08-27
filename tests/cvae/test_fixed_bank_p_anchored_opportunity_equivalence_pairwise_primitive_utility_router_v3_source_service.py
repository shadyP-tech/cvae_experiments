from __future__ import annotations

import inspect

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.config import (
    build_planned_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.execution.services import (
    CanonicalScientificRouterService,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.service_factory import (
    prepare_canonical_scientific_service_factory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_seal import (
    SourceSealReceipt,
    build_source_seal,
    validate_source_seal,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def test_source_seal_covers_v3_and_neutral_core_without_predecessor_imports() -> None:
    receipt = build_source_seal()
    payload = receipt.to_payload()

    assert validate_source_seal(receipt) == receipt
    assert receipt.adapter_member_count > 0
    assert receipt.neutral_member_count > 0
    assert payload["predecessor_imports_present"] is False
    assert payload["unsealed_project_imports_present"] is False


def test_source_seal_receipt_cannot_be_fabricated() -> None:
    receipt = build_source_seal()
    with pytest.raises(ProtocolError, match="bypassed source admission"):
        SourceSealReceipt(
            repository_root=receipt.repository_root,
            adapter_member_count=receipt.adapter_member_count,
            adapter_tree_sha256=receipt.adapter_tree_sha256,
            neutral_member_count=receipt.neutral_member_count,
            neutral_tree_sha256=receipt.neutral_tree_sha256,
            shared_protocol_sha256=receipt.shared_protocol_sha256,
            combined_source_sha256=receipt.combined_source_sha256,
        )


def test_nominal_service_is_concrete_but_factory_requires_typed_source_surface() -> None:
    assert inspect.isabstract(CanonicalScientificRouterService) is False
    with pytest.raises(ProtocolError, match="bypassed its factory"):
        CanonicalScientificRouterService(
            source_surface=object(),
            source_seal_hash="a" * 64,
            seven_input_contract_hash="b" * 64,
            factory_identity_hash="c" * 64,
        )

    with pytest.raises(ProtocolError, match="source-training surface identity drifted"):
        prepare_canonical_scientific_service_factory(
            build_planned_config(),
            source_seal=build_source_seal(),
            source_surface=object(),
        )
