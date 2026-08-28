"""Path-free planned configuration for the OE-PPUR v3 successor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ...protocol import ProtocolError
from .execution.inputs import (
    ResolvedDirectInput,
    build_authorized_seven_input_contract,
    build_planned_seven_input_contract,
    validate_exact_resolved_input_bindings,
)
from .hashing import canonical_hash, require_sha256
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
AUTHORIZATION_READY_STATE = "AUTHORIZATION_READY_EXTERNAL_AMENDMENT"


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
    source_supervision_producer_seal_sha256: str | None
    source_supervision_recomputation_receipt_sha256: str | None
    authorization_amendment_sha256: str | None
    contract_hash: str = field(init=False)

    def __post_init__(self) -> None:
        expected_protocol = frozen_protocol_payload()
        authorized = self.authorization_state == AUTHORIZATION_READY_STATE
        expected_inputs = (
            build_authorized_seven_input_contract()
            if authorized
            else build_planned_seven_input_contract()
        )
        if (
            self.experiment_id != EXPERIMENT_ID
            or self.output_artifact_id != OUTPUT_ARTIFACT_ID
            or self.authorization_state
            not in {PLANNED_STATE, AUTHORIZATION_READY_STATE}
            or self.execution_authorized is not authorized
            or tuple(self.direct_input_roles) != DIRECT_INPUT_ROLES
            or tuple(self.direct_input_artifact_ids) != DIRECT_INPUT_ARTIFACT_IDS
            or self.protocol_hash != expected_protocol["protocol_hash"]
            or self.seven_input_contract_hash != expected_inputs.receipt_hash
        ):
            role = (
                "planned config"
                if self.authorization_state == PLANNED_STATE
                else "config"
            )
            raise ProtocolError(f"OE-PPUR v3 {role} identity drifted.")
        guarded_hashes = (
            "source_supervision_content_sha256",
            "source_supervision_row_order_sha256",
            "source_supervision_producer_seal_sha256",
            "source_supervision_recomputation_receipt_sha256",
            "authorization_amendment_sha256",
        )
        if authorized:
            for role in guarded_hashes:
                digest = require_sha256(getattr(self, role), role.replace("_", " "))
                if digest == "0" * 64:
                    raise ProtocolError(
                        "OE-PPUR v3 authorization-ready hashes cannot be placeholders."
                    )
                object.__setattr__(self, role, digest)
        elif any(getattr(self, role) is not None for role in guarded_hashes):
            raise ProtocolError("OE-PPUR v3 planned config identity drifted.")
        object.__setattr__(self, "direct_input_roles", DIRECT_INPUT_ROLES)
        object.__setattr__(
            self, "direct_input_artifact_ids", DIRECT_INPUT_ARTIFACT_IDS
        )
        object.__setattr__(self, "contract_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        authorized = self.execution_authorized
        inputs = (
            build_authorized_seven_input_contract()
            if authorized
            else build_planned_seven_input_contract()
        )
        return {
            "schema_version": (
                "oe_ppur_v3_authorization_ready_config_v1"
                if authorized
                else "oe_ppur_v3_planned_config_v1"
            ),
            "experiment": {
                "id": EXPERIMENT_ID,
                "name": EXPERIMENT_NAME,
                "output_artifact_id": OUTPUT_ARTIFACT_ID,
                "authorization_state": self.authorization_state,
                "execution_authorized": authorized,
                "publication_status": PUBLICATION_STATUS,
                "terminal_decision": TERMINAL_DECISION,
            },
            "inputs": {
                "exact_seven_input_contract": inputs.to_payload(),
                "source_supervision": {
                    "direct_input_ordinal": 3,
                    "content_sha256": self.source_supervision_content_sha256,
                    "row_order_sha256": self.source_supervision_row_order_sha256,
                    "producer_source_seal_sha256": (
                        self.source_supervision_producer_seal_sha256
                    ),
                    "recomputation_receipt_sha256": (
                        self.source_supervision_recomputation_receipt_sha256
                    ),
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
                "authorization_amendment_sha256": (
                    self.authorization_amendment_sha256
                ),
            },
            "protocol": frozen_protocol_payload(),
            "claim_boundary": claim_boundary_payload(
                execution_authorized=authorized
            ),
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
        source_supervision_producer_seal_sha256=None,
        source_supervision_recomputation_receipt_sha256=None,
        authorization_amendment_sha256=None,
    )


def build_authorization_ready_config(
    *,
    source_supervision_content_sha256: str,
    source_supervision_row_order_sha256: str,
    source_supervision_producer_seal_sha256: str,
    source_supervision_recomputation_receipt_sha256: str,
    authorization_amendment_sha256: str,
) -> RouterV3Config:
    """Build a future external-amendment state; never called by preparation."""

    protocol = frozen_protocol_payload()
    inputs = build_authorized_seven_input_contract()
    return RouterV3Config(
        experiment_id=EXPERIMENT_ID,
        output_artifact_id=OUTPUT_ARTIFACT_ID,
        authorization_state=AUTHORIZATION_READY_STATE,
        execution_authorized=True,
        direct_input_roles=DIRECT_INPUT_ROLES,
        direct_input_artifact_ids=DIRECT_INPUT_ARTIFACT_IDS,
        protocol_hash=str(protocol["protocol_hash"]),
        seven_input_contract_hash=inputs.receipt_hash,
        source_supervision_content_sha256=source_supervision_content_sha256,
        source_supervision_row_order_sha256=source_supervision_row_order_sha256,
        source_supervision_producer_seal_sha256=(
            source_supervision_producer_seal_sha256
        ),
        source_supervision_recomputation_receipt_sha256=(
            source_supervision_recomputation_receipt_sha256
        ),
        authorization_amendment_sha256=authorization_amendment_sha256,
    )


def frozen_config_contract_payload() -> dict[str, object]:
    return build_planned_config().to_payload()


def validate_planned_config(value: object) -> RouterV3Config:
    if type(value) is not RouterV3Config or value != build_planned_config():
        raise ProtocolError("OE-PPUR v3 planned config contract drifted.")
    return value


def validate_authorization_ready_config(value: object) -> RouterV3Config:
    if (
        type(value) is not RouterV3Config
        or value.authorization_state != AUTHORIZATION_READY_STATE
        or value.execution_authorized is not True
        or value
        != build_authorization_ready_config(
            source_supervision_content_sha256=str(
                value.source_supervision_content_sha256
            ),
            source_supervision_row_order_sha256=str(
                value.source_supervision_row_order_sha256
            ),
            source_supervision_producer_seal_sha256=str(
                value.source_supervision_producer_seal_sha256
            ),
            source_supervision_recomputation_receipt_sha256=str(
                value.source_supervision_recomputation_receipt_sha256
            ),
            authorization_amendment_sha256=str(
                value.authorization_amendment_sha256
            ),
        )
    ):
        raise ProtocolError("OE-PPUR v3 authorization-ready config drifted.")
    return value


@dataclass(frozen=True, slots=True)
class ResolvedV3ConfigBundle:
    """Future workspace-rendered paths paired with an authorized config."""

    config: RouterV3Config
    source_path: Path
    artifact_root: Path
    input_bindings: tuple[ResolvedDirectInput, ...]

    def __post_init__(self) -> None:
        config = validate_authorization_ready_config(self.config)
        source = Path(self.source_path)
        artifact = Path(self.artifact_root)
        bindings = validate_exact_resolved_input_bindings(self.input_bindings)
        if (
            not source.is_absolute()
            or source.name != "config.resolved.yaml"
            or not artifact.is_absolute()
            or artifact == Path(artifact.anchor)
            or source.parent != artifact
        ):
            raise ProtocolError("OE-PPUR v3 resolved config bundle drifted.")
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "source_path", source)
        object.__setattr__(self, "artifact_root", artifact)
        object.__setattr__(self, "input_bindings", bindings)


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
    "AUTHORIZATION_READY_STATE",
    "PLANNED_STATE",
    "ResolvedV3ConfigBundle",
    "RouterV3Config",
    "build_authorization_ready_config",
    "build_planned_config",
    "frozen_config_contract_payload",
    "load_config",
    "validate_authorization_ready_config",
    "validate_planned_config",
)
