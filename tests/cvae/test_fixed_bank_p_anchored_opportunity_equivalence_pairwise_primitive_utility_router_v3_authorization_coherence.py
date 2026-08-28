from __future__ import annotations

import hashlib
import json

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.authorization_contract import (
    IMMUTABLE_PARENT_LEDGER_ARTIFACT_ID,
    authorization_amendment_bytes,
    build_authorization_amendment_payload,
    canonical_authorization_amendment_sha256,
    validate_authorization_amendment_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.config import (
    build_authorization_ready_config,
    build_planned_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.identity import (
    ORIGINAL_PARENT_LEDGER_ARTIFACT_ID,
)
from midogpp_thesis.cvae.protocol import ProtocolError


SOURCE_RECEIPT = "1" * 64
ROW_ORDER = "2" * 64
PRODUCER_SEAL = "3" * 64
RECOMPUTATION = "4" * 64
AMENDMENT_SHA256 = "5" * 64
LIFECYCLE_SEAL = "6" * 64


def _authorized_config():
    return build_authorization_ready_config(
        source_supervision_content_sha256=SOURCE_RECEIPT,
        source_supervision_row_order_sha256=ROW_ORDER,
        source_supervision_producer_seal_sha256=PRODUCER_SEAL,
        source_supervision_recomputation_receipt_sha256=RECOMPUTATION,
        authorization_amendment_sha256=AMENDMENT_SHA256,
    )


def _all_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_mapping_keys(row) for row in value.values()))
    if isinstance(value, (list, tuple)):
        return set().union(*(_all_mapping_keys(row) for row in value))
    return set()


def test_scientific_protocol_is_state_neutral_and_authority_projections_agree() -> None:
    planned = build_planned_config().to_payload()
    authorized = _authorized_config().to_payload()

    assert planned["protocol"] == authorized["protocol"]
    assert planned["protocol"]["protocol_hash"] == authorized["protocol"][
        "protocol_hash"
    ]
    assert planned["protocol"][
        "source_supervision_materialization_required_before_execution"
    ] is True
    assert planned["protocol"][
        "authorization_amendment_required_before_execution"
    ] is True
    assert planned["protocol"][
        "current_authority_state_owned_by_config_not_protocol"
    ] is True
    mutable_state_keys = {
        "source_supervision_materialized",
        "authorization_amendment_issued",
        "execution_authorized",
        "consumed_test_reuse_authorized",
    }
    assert not mutable_state_keys.intersection(
        _all_mapping_keys(planned["protocol"])
    )

    for payload, state in ((planned, False), (authorized, True)):
        exact = payload["inputs"]["exact_seven_input_contract"]
        assert payload["experiment"]["execution_authorized"] is state
        assert exact["source_supervision_materialized"] is state
        assert exact["authorization_amendment_issued"] is state
        assert exact["execution_authorized"] is state
        assert payload["claim_boundary"]["execution_authorized"] is state
        assert payload["claim_boundary"]["consumed_test_reuse_authorized"] is state


def test_amendment_names_true_immutable_parent_not_v3_resolution_alias() -> None:
    protocol_hash = _authorized_config().protocol_hash
    payload = build_authorization_amendment_payload(
        source_contract_hash=SOURCE_RECEIPT,
        protocol_hash=protocol_hash,
        lifecycle_source_seal_sha256=LIFECYCLE_SEAL,
    )

    assert IMMUTABLE_PARENT_LEDGER_ARTIFACT_ID == (
        "midogpp_uniform_b_test_consumption_ledger_v1"
    )
    assert ORIGINAL_PARENT_LEDGER_ARTIFACT_ID != IMMUTABLE_PARENT_LEDGER_ARTIFACT_ID
    assert payload["parent_artifact_id"] == IMMUTABLE_PARENT_LEDGER_ARTIFACT_ID
    assert payload["direct_original_parent_only"] is True
    assert validate_authorization_amendment_payload(
        payload,
        source_contract_hash=SOURCE_RECEIPT,
        protocol_hash=protocol_hash,
        lifecycle_source_seal_sha256=LIFECYCLE_SEAL,
    ) == payload

    drifted = {**payload, "parent_artifact_id": ORIGINAL_PARENT_LEDGER_ARTIFACT_ID}
    with pytest.raises(ProtocolError, match="amendment drifted"):
        validate_authorization_amendment_payload(
            drifted,
            source_contract_hash=SOURCE_RECEIPT,
            protocol_hash=protocol_hash,
            lifecycle_source_seal_sha256=LIFECYCLE_SEAL,
        )


def test_amendment_canonical_bytes_and_hash_are_deterministic() -> None:
    protocol_hash = _authorized_config().protocol_hash
    payload = build_authorization_amendment_payload(
        source_contract_hash=SOURCE_RECEIPT,
        protocol_hash=protocol_hash,
        lifecycle_source_seal_sha256=LIFECYCLE_SEAL,
    )
    raw = authorization_amendment_bytes(
        source_contract_hash=SOURCE_RECEIPT,
        protocol_hash=protocol_hash,
        lifecycle_source_seal_sha256=LIFECYCLE_SEAL,
    )

    assert raw == (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    assert hashlib.sha256(raw).hexdigest() == (
        canonical_authorization_amendment_sha256(
            source_contract_hash=SOURCE_RECEIPT,
            protocol_hash=protocol_hash,
            lifecycle_source_seal_sha256=LIFECYCLE_SEAL,
        )
    )

    with pytest.raises(ProtocolError, match="cannot be placeholders"):
        build_authorization_amendment_payload(
            source_contract_hash="0" * 64,
            protocol_hash=protocol_hash,
            lifecycle_source_seal_sha256=LIFECYCLE_SEAL,
        )
