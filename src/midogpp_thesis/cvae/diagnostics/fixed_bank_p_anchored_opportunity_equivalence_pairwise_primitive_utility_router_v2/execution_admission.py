"""Mutation-free six-input admission for the OE-PPUR v2 successor."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field
from pathlib import Path

from ...protocol import ProtocolError
from .authorization_contract import load_and_validate_authorization_amendment
from .config import AUTHORIZATION_READY_STATE, RouterV2Config
from .hashing import canonical_hash, require_sha256
from .identity import (
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPERIMENT_ID,
    EXPECTED_BANK_CONTENT_INDEX_SHA256,
    EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
    EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_MANIFEST_SHA256,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from .protocol import validate_claim_boundary, validate_protocol_payload
from .source_seal import (
    SourceContractReceipt,
    validate_source_contract_receipt,
)
from .workspace_inputs import (
    ValidatedWorkspaceInputs,
    WorkspaceInputBinding,
    validate_workspace_inputs,
)


ADMISSION_SCHEMA = "oe_ppur_v2_six_input_admission_receipt_v1"
_ADMISSION_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class SixInputAdmissionReceipt:
    """Guarded proof that all read-only authority gates exact-matched."""

    status: str
    experiment_id: str
    output_artifact_id: str
    input_artifact_ids: tuple[str, ...]
    input_roles: tuple[str, ...]
    config_contract_hash: str
    protocol_hash: str
    source_contract_hash: str
    authorization_amendment_sha256: str
    input_binding_hash: str
    input_location_binding_sha256: str
    bank_content_index_sha256: str
    generation_content_index_sha256: str
    cache_content_sha256: str
    cache_row_order_sha256: str
    manifest_sha256: str
    parent_ledger_sha256: str
    artifact_root: str
    scratch_root: str
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _ADMISSION_FACTORY_TOKEN:
            raise ProtocolError(
                "OE-PPUR v2 admission receipts require read-only validation."
            )
        if (
            self.status != "ADMITTED_SINGLE_USE_READ_ONLY"
            or self.experiment_id != EXPERIMENT_ID
            or self.output_artifact_id != OUTPUT_ARTIFACT_ID
            or tuple(self.input_artifact_ids) != DIRECT_INPUT_ARTIFACT_IDS
            or tuple(self.input_roles) != DIRECT_INPUT_ROLES
            or self.cache_content_sha256 != EXPECTED_TEST_CACHE_CONTENT_HASH
            or self.cache_row_order_sha256
            != EXPECTED_TEST_CACHE_ROW_ORDER_HASH
            or self.manifest_sha256 != EXPECTED_TEST_MANIFEST_SHA256
            or self.parent_ledger_sha256
            != EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256
            or self.bank_content_index_sha256
            != EXPECTED_BANK_CONTENT_INDEX_SHA256
            or self.generation_content_index_sha256
            != EXPECTED_GENERATION_CONTENT_INDEX_SHA256
        ):
            raise ProtocolError("OE-PPUR v2 admission receipt identity drifted.")
        for role, digest in (
            ("config contract hash", self.config_contract_hash),
            ("protocol hash", self.protocol_hash),
            ("source contract hash", self.source_contract_hash),
            ("authorization amendment hash", self.authorization_amendment_sha256),
            ("input binding hash", self.input_binding_hash),
            (
                "input location binding hash",
                self.input_location_binding_sha256,
            ),
            ("bank content-index hash", self.bank_content_index_sha256),
            (
                "GenerationLock content-index hash",
                self.generation_content_index_sha256,
            ),
            ("cache content hash", self.cache_content_sha256),
            ("cache row-order hash", self.cache_row_order_sha256),
            ("manifest hash", self.manifest_sha256),
            ("parent ledger hash", self.parent_ledger_sha256),
        ):
            require_sha256(digest, role)
        object.__setattr__(self, "input_artifact_ids", DIRECT_INPUT_ARTIFACT_IDS)
        object.__setattr__(self, "input_roles", DIRECT_INPUT_ROLES)
        object.__setattr__(self, "receipt_hash", canonical_hash(self._body()))

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": ADMISSION_SCHEMA,
            "status": self.status,
            "experiment_id": self.experiment_id,
            "output_artifact_id": self.output_artifact_id,
            "input_artifact_ids": list(self.input_artifact_ids),
            "input_roles": list(self.input_roles),
            "config_contract_hash": self.config_contract_hash,
            "protocol_hash": self.protocol_hash,
            "source_contract_hash": self.source_contract_hash,
            "authorization_amendment_sha256": (
                self.authorization_amendment_sha256
            ),
            "input_binding_hash": self.input_binding_hash,
            "input_location_binding_sha256": (
                self.input_location_binding_sha256
            ),
            "bank_content_index_sha256": self.bank_content_index_sha256,
            "generation_content_index_sha256": (
                self.generation_content_index_sha256
            ),
            "cache_content_sha256": self.cache_content_sha256,
            "cache_row_order_sha256": self.cache_row_order_sha256,
            "manifest_sha256": self.manifest_sha256,
            "parent_ledger_sha256": self.parent_ledger_sha256,
            "artifact_root": self.artifact_root,
            "scratch_root": self.scratch_root,
            "execution_authorized": True,
            "consumed_test_reuse_authorized": True,
            "single_use_execution_identity": True,
            "authorization_exhausted": False,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "target_labels_opened": False,
            "parsed_probability_matrix_science_receipt_required": True,
            "predecessor_state_used": False,
            "cross_run_recovery_used": False,
            "mutation_performed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._body(), "receipt_hash": self.receipt_hash}


def admit_six_input_execution(
    config: RouterV2Config,
    *,
    input_bindings: Sequence[WorkspaceInputBinding],
    artifact_root: str | Path,
    scratch_root: str | Path,
    source_contract_receipt: SourceContractReceipt,
) -> SixInputAdmissionReceipt:
    """Admit one externally authorized run without creating any state."""

    # Authority is deliberately checked before touching a path or iterating an
    # input binding.  The checked-in planned config therefore fails path-free.
    if not isinstance(config, RouterV2Config):
        raise ProtocolError("OE-PPUR v2 admission requires its exact config type.")
    if (
        config.authorization_state != AUTHORIZATION_READY_STATE
        or config.execution_authorized is not True
        or config.consumed_test_reuse_authorized is not True
        or config.authorization_exhausted is not False
        or config.experiment_id != EXPERIMENT_ID
        or config.output_artifact_id != OUTPUT_ARTIFACT_ID
        or tuple(config.input_artifact_ids) != DIRECT_INPUT_ARTIFACT_IDS
        or tuple(config.input_roles) != DIRECT_INPUT_ROLES
        or config.expected_authorization_amendment_sha256 is None
        or config.source_contract_hash is None
    ):
        raise ProtocolError(
            "OE-PPUR v2 execution is not authorized by a real amendment."
        )
    validate_protocol_payload(config.protocol)
    validate_claim_boundary(config.claim_boundary, execution_authorized=True)
    if config.contract_hash != canonical_hash(config.to_payload()):
        raise ProtocolError("OE-PPUR v2 config contract hash drifted.")
    source_receipt = validate_source_contract_receipt(
        source_contract_receipt,
        expected_source_contract_hash=config.source_contract_hash,
    )
    source_hash = require_sha256(
        source_receipt.combined_source_sha256, "combined source contract hash"
    )
    if source_hash != config.source_contract_hash:
        raise ProtocolError("OE-PPUR v2 source/config binding drifted.")
    protocol_hash = require_sha256(
        config.protocol.get("protocol_hash"), "protocol hash"
    )

    validated = validate_workspace_inputs(
        input_bindings,
        artifact_root=artifact_root,
        scratch_root=scratch_root,
        expected_amendment_sha256=(
            config.expected_authorization_amendment_sha256
        ),
    )
    amendment, amendment_sha256 = load_and_validate_authorization_amendment(
        validated.amendment_path, config=config
    )
    if (
        amendment_sha256 != validated.amendment_sha256
        or amendment.get("source_contract_hash") != source_hash
        or amendment.get("protocol_hash") != protocol_hash
    ):
        raise ProtocolError("OE-PPUR v2 amendment admission binding drifted.")
    return _issue_six_input_admission_receipt(
        config=config,
        validated=validated,
        protocol_hash=protocol_hash,
        source_hash=source_hash,
        amendment_sha256=amendment_sha256,
    )


def _issue_six_input_admission_receipt(
    *,
    config: RouterV2Config,
    validated: ValidatedWorkspaceInputs,
    protocol_hash: str,
    source_hash: str,
    amendment_sha256: str,
) -> SixInputAdmissionReceipt:
    return SixInputAdmissionReceipt(
        status="ADMITTED_SINGLE_USE_READ_ONLY",
        experiment_id=EXPERIMENT_ID,
        output_artifact_id=OUTPUT_ARTIFACT_ID,
        input_artifact_ids=DIRECT_INPUT_ARTIFACT_IDS,
        input_roles=DIRECT_INPUT_ROLES,
        config_contract_hash=config.contract_hash,
        protocol_hash=protocol_hash,
        source_contract_hash=source_hash,
        authorization_amendment_sha256=amendment_sha256,
        input_binding_hash=validated.input_binding_hash,
        input_location_binding_sha256=(
            validated.input_location_binding_sha256
        ),
        bank_content_index_sha256=validated.bank_content_index_sha256,
        generation_content_index_sha256=(
            validated.generation_content_index_sha256
        ),
        cache_content_sha256=validated.cache_content_sha256,
        cache_row_order_sha256=validated.cache_row_order_sha256,
        manifest_sha256=validated.manifest_sha256,
        parent_ledger_sha256=validated.parent_ledger_sha256,
        artifact_root=validated.artifact_root,
        scratch_root=validated.scratch_root,
        _factory_token=_ADMISSION_FACTORY_TOKEN,
    )


assert_execution_authorized = admit_six_input_execution
admit_execution = admit_six_input_execution


__all__ = (
    "ADMISSION_SCHEMA",
    "SixInputAdmissionReceipt",
    "admit_execution",
    "admit_six_input_execution",
    "assert_execution_authorized",
)
