"""Path-free planned configuration for the OE-PPUR v3 successor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ...protocol import ProtocolError
from .execution.inputs import build_planned_seven_input_contract
from .hashing import canonical_hash
from .identity import (
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPECTED_AUTHORIZATION_AMENDMENT_SHA256,
    EXPECTED_BANK_CONTENT_INDEX_SHA256,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
    EXPECTED_SOURCE_SUPERVISION_CONTENT_SHA256,
    EXPECTED_SOURCE_SUPERVISION_ROW_ORDER_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPECTED_TEST_MANIFEST_SHA256,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from .protocol import claim_boundary_payload, frozen_protocol_payload


PLANNED_STATE = "PLANNED_NOT_AUTHORIZED"


@dataclass(frozen=True, slots=True)
class RouterV3Config:
    """Immutable, path-free statement of the unissued v3 execution identity."""

    experiment_id: str
    output_artifact_id: str
    authorization_state: str
    execution_authorized: bool
    direct_input_roles: tuple[str, ...]
    direct_input_artifact_ids: tuple[str, ...]
    protocol_hash: str
    seven_input_contract_hash: str
    source_supervision_content_sha256: str | None
    source_supervision_row_order_sha256: str | None
    authorization_amendment_sha256: str | None
    contract_hash: str = field(init=False)

    def __post_init__(self) -> None:
        expected_protocol = frozen_protocol_payload()
        expected_inputs = build_planned_seven_input_contract()
        if (
            self.experiment_id != EXPERIMENT_ID
            or self.output_artifact_id != OUTPUT_ARTIFACT_ID
            or self.authorization_state != PLANNED_STATE
            or self.execution_authorized is not False
            or tuple(self.direct_input_roles) != DIRECT_INPUT_ROLES
            or tuple(self.direct_input_artifact_ids) != DIRECT_INPUT_ARTIFACT_IDS
            or self.protocol_hash != expected_protocol["protocol_hash"]
            or self.seven_input_contract_hash != expected_inputs.receipt_hash
            or self.source_supervision_content_sha256 is not None
            or self.source_supervision_row_order_sha256 is not None
            or self.authorization_amendment_sha256 is not None
        ):
            raise ProtocolError("OE-PPUR v3 planned config identity drifted.")
        object.__setattr__(self, "direct_input_roles", DIRECT_INPUT_ROLES)
        object.__setattr__(
            self, "direct_input_artifact_ids", DIRECT_INPUT_ARTIFACT_IDS
        )
        object.__setattr__(self, "contract_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        inputs = build_planned_seven_input_contract()
        return {
            "schema_version": "oe_ppur_v3_planned_config_v1",
            "experiment": {
                "id": EXPERIMENT_ID,
                "name": EXPERIMENT_NAME,
                "output_artifact_id": OUTPUT_ARTIFACT_ID,
                "authorization_state": PLANNED_STATE,
                "execution_authorized": False,
                "publication_status": PUBLICATION_STATUS,
                "terminal_decision": TERMINAL_DECISION,
            },
            "inputs": {
                "exact_seven_input_contract": inputs.to_payload(),
                "source_supervision": {
                    "direct_input_ordinal": 3,
                    "content_sha256": EXPECTED_SOURCE_SUPERVISION_CONTENT_SHA256,
                    "row_order_sha256": EXPECTED_SOURCE_SUPERVISION_ROW_ORDER_SHA256,
                },
                "bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
                "generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
                "bank_content_index_sha256": EXPECTED_BANK_CONTENT_INDEX_SHA256,
                "generation_content_index_sha256": EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
                "test_cache_semantic_id": EXPECTED_TEST_CACHE_SEMANTIC_ID,
                "test_cache_representation_id": EXPECTED_TEST_CACHE_REPRESENTATION_ID,
                "test_cache_content_sha256": EXPECTED_TEST_CACHE_CONTENT_HASH,
                "test_cache_row_order_sha256": EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
                "test_manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
                "original_parent_ledger_sha256": EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
                "authorization_amendment_sha256": EXPECTED_AUTHORIZATION_AMENDMENT_SHA256,
            },
            "protocol": frozen_protocol_payload(),
            "claim_boundary": claim_boundary_payload(),
            "paths_present": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "config_contract_hash": self.contract_hash}


def build_planned_config() -> RouterV3Config:
    protocol = frozen_protocol_payload()
    inputs = build_planned_seven_input_contract()
    return RouterV3Config(
        experiment_id=EXPERIMENT_ID,
        output_artifact_id=OUTPUT_ARTIFACT_ID,
        authorization_state=PLANNED_STATE,
        execution_authorized=False,
        direct_input_roles=DIRECT_INPUT_ROLES,
        direct_input_artifact_ids=DIRECT_INPUT_ARTIFACT_IDS,
        protocol_hash=str(protocol["protocol_hash"]),
        seven_input_contract_hash=inputs.receipt_hash,
        source_supervision_content_sha256=None,
        source_supervision_row_order_sha256=None,
        authorization_amendment_sha256=None,
    )


def frozen_config_contract_payload() -> dict[str, object]:
    return build_planned_config().to_payload()


def validate_planned_config(value: object) -> RouterV3Config:
    if type(value) is not RouterV3Config or value != build_planned_config():
        raise ProtocolError("OE-PPUR v3 planned config contract drifted.")
    return value


def load_config(path: str | Path) -> RouterV3Config:
    """Load only the exact path-free planned payload; never resolve artifacts."""

    source = Path(path)
    try:
        raw: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ProtocolError("OE-PPUR v3 planned config could not be loaded.") from exc
    if not isinstance(raw, Mapping) or dict(raw) != frozen_config_contract_payload():
        raise ProtocolError("OE-PPUR v3 planned config bytes drifted.")
    return build_planned_config()


__all__ = (
    "PLANNED_STATE",
    "RouterV3Config",
    "build_planned_config",
    "frozen_config_contract_payload",
    "load_config",
    "validate_planned_config",
)
