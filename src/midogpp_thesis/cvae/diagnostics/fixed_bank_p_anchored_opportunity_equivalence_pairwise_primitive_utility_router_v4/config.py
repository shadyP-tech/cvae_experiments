"""Path-free lifecycle configuration for the OE-PPUR v4 successor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256
from .execution.inputs import (
    ResolvedDirectInput,
    validate_exact_resolved_input_bindings,
)
from .identity import (
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPECTED_BANK_CONTENT_INDEX_SHA256,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
    EXPECTED_SOURCE_PRODUCER_SEAL_SHA256,
    EXPECTED_SOURCE_RECEIPT_SHA256,
    EXPECTED_SOURCE_RECOMPUTATION_RECEIPT_SHA256,
    EXPECTED_SOURCE_ROW_ORDER_SHA256,
    EXPECTED_SOURCE_SURFACE_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_MANIFEST_SHA256,
    OUTPUT_ARTIFACT_ID,
    PRESERVED_V3_AMENDMENT_SHA256,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from .input_contract import (
    build_authorized_seven_input_contract,
    build_planned_seven_input_contract,
)
from .protocol import claim_boundary_payload, frozen_protocol_payload


PLANNED_STATE = "PLANNED_WORKSPACE_SEAL_REQUIRED"
SEALED_STATE = "WORKSPACE_SEALED_AMENDMENT_ISSUED_NO_LAUNCH"


@dataclass(frozen=True, slots=True)
class RouterV4Config:
    experiment_id: str
    output_artifact_id: str
    authorization_state: str
    execution_amendment_issued: bool
    launch_authorized: bool
    workspace_plan_sha256: str | None
    authorization_amendment_sha256: str | None
    direct_input_roles: tuple[str, ...]
    direct_input_artifact_ids: tuple[str, ...]
    protocol_hash: str
    seven_input_contract_hash: str
    contract_hash: str = field(init=False)

    def __post_init__(self) -> None:
        sealed = self.authorization_state == SEALED_STATE
        inputs = (
            build_authorized_seven_input_contract()
            if sealed
            else build_planned_seven_input_contract()
        )
        if (
            self.experiment_id != EXPERIMENT_ID
            or self.output_artifact_id != OUTPUT_ARTIFACT_ID
            or self.authorization_state not in {PLANNED_STATE, SEALED_STATE}
            or type(self.execution_amendment_issued) is not bool
            or self.execution_amendment_issued is not sealed
            or self.launch_authorized is not False
            or tuple(self.direct_input_roles) != DIRECT_INPUT_ROLES
            or tuple(self.direct_input_artifact_ids) != DIRECT_INPUT_ARTIFACT_IDS
            or self.protocol_hash != frozen_protocol_payload()["protocol_hash"]
            or self.seven_input_contract_hash != inputs.receipt_hash
        ):
            raise ProtocolError("OE-PPUR v4 config identity drifted.")
        if sealed:
            plan = require_sha256(self.workspace_plan_sha256, "workspace plan")
            amendment = require_sha256(
                self.authorization_amendment_sha256,
                "authorization amendment",
            )
            if plan == "0" * 64 or amendment == "0" * 64:
                raise ProtocolError("OE-PPUR v4 sealed hashes are placeholders.")
            object.__setattr__(self, "workspace_plan_sha256", plan)
            object.__setattr__(self, "authorization_amendment_sha256", amendment)
        elif (
            self.workspace_plan_sha256 is not None
            or self.authorization_amendment_sha256 is not None
        ):
            raise ProtocolError("OE-PPUR v4 planned config contains authority.")
        object.__setattr__(self, "contract_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        sealed = self.execution_amendment_issued
        inputs = (
            build_authorized_seven_input_contract()
            if sealed
            else build_planned_seven_input_contract()
        )
        return {
            "schema_version": (
                "oe_ppur_v4_workspace_sealed_config_v1"
                if sealed
                else "oe_ppur_v4_planned_config_v1"
            ),
            "experiment": {
                "id": EXPERIMENT_ID,
                "name": EXPERIMENT_NAME,
                "output_artifact_id": OUTPUT_ARTIFACT_ID,
                "authorization_state": self.authorization_state,
                "execution_amendment_issued": sealed,
                "launch_authorized": False,
                "publication_status": PUBLICATION_STATUS,
                "terminal_decision": TERMINAL_DECISION,
            },
            "inputs": {
                "exact_seven_input_contract": inputs.to_payload(),
                "source_supervision": {
                    "direct_input_ordinal": 3,
                    "content_lineage": "IMMUTABLE_SOURCE_ONLY_SUCCESSOR_ALIAS",
                    "receipt_sha256": EXPECTED_SOURCE_RECEIPT_SHA256,
                    "surface_sha256": EXPECTED_SOURCE_SURFACE_SHA256,
                    "row_order_sha256": EXPECTED_SOURCE_ROW_ORDER_SHA256,
                    "producer_source_seal_sha256": (
                        EXPECTED_SOURCE_PRODUCER_SEAL_SHA256
                    ),
                    "recomputation_receipt_sha256": (
                        EXPECTED_SOURCE_RECOMPUTATION_RECEIPT_SHA256
                    ),
                    "target_rows_present": False,
                    "target_labels_used": False,
                    "authority_inherited": False,
                },
                "bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
                "generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
                "bank_content_index_sha256": EXPECTED_BANK_CONTENT_INDEX_SHA256,
                "generation_content_index_sha256": (
                    EXPECTED_GENERATION_CONTENT_INDEX_SHA256
                ),
                "test_cache_content_sha256": EXPECTED_TEST_CACHE_CONTENT_HASH,
                "test_cache_row_order_sha256": EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
                "test_manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
                "original_parent_ledger_sha256": (
                    EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256
                ),
                "preserved_v3_amendment_sha256": PRESERVED_V3_AMENDMENT_SHA256,
                "workspace_plan_sha256": self.workspace_plan_sha256,
                "authorization_amendment_sha256": (
                    self.authorization_amendment_sha256
                ),
            },
            "protocol": frozen_protocol_payload(),
            "claim_boundary": claim_boundary_payload(False),
            "paths_present": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "config_contract_hash": self.contract_hash}

    @property
    def source_supervision_content_sha256(self) -> str:
        """Exact parsed source-training receipt bound by direct input #3."""

        return EXPECTED_SOURCE_RECEIPT_SHA256

    @property
    def source_supervision_row_order_sha256(self) -> str:
        return EXPECTED_SOURCE_ROW_ORDER_SHA256

    @property
    def source_supervision_producer_seal_sha256(self) -> str:
        return EXPECTED_SOURCE_PRODUCER_SEAL_SHA256

    @property
    def source_supervision_recomputation_receipt_sha256(self) -> str:
        return EXPECTED_SOURCE_RECOMPUTATION_RECEIPT_SHA256


def validate_workspace_sealed_config(value: object) -> RouterV4Config:
    """Validate amendment readiness without pretending it grants launch."""

    if (
        type(value) is not RouterV4Config
        or value.authorization_state != SEALED_STATE
        or value.execution_amendment_issued is not True
        or value.launch_authorized is not False
        or value.workspace_plan_sha256 is None
        or value.authorization_amendment_sha256 is None
        or value
        != build_workspace_sealed_config(
            workspace_plan_sha256=value.workspace_plan_sha256,
            authorization_amendment_sha256=value.authorization_amendment_sha256,
        )
    ):
        raise ProtocolError("OE-PPUR v4 workspace-sealed config drifted.")
    return value


@dataclass(frozen=True, slots=True)
class ResolvedV4ConfigBundle:
    """Prospective path bindings plus separately admitted launch authority.

    The contained config deliberately remains ``launch_authorized=False``.
    Execution permission is represented only by the type-gated run admission;
    this bundle records its authority-file hash for end-to-end lineage.
    """

    config: RouterV4Config
    source_path: Path
    artifact_root: Path
    input_bindings: tuple[ResolvedDirectInput, ...]
    input_manifest_path: Path
    final_envelope_path: Path
    workspace_snapshot_sha256: str
    workspace_plan_sha256: str
    final_envelope_sha256: str
    execution_launch_authority_sha256: str

    def __post_init__(self) -> None:
        config = validate_workspace_sealed_config(self.config)
        source = Path(self.source_path)
        artifact = Path(self.artifact_root)
        manifest = Path(self.input_manifest_path)
        envelope = Path(self.final_envelope_path)
        bindings = validate_exact_resolved_input_bindings(self.input_bindings)
        if (
            not source.is_absolute()
            or source.name != "config.resolved.yaml"
            or not artifact.is_absolute()
            or artifact == Path(artifact.anchor)
            or source != artifact / "config.resolved.yaml"
            or manifest != artifact / "provenance/input_artifacts.json"
            or envelope
            != artifact / "preparation/final_authorization_envelope.json"
            or any(path.is_symlink() for path in (source, artifact, manifest, envelope))
        ):
            raise ProtocolError("OE-PPUR v4 resolved config bundle drifted.")
        for role in (
            "workspace_snapshot_sha256",
            "workspace_plan_sha256",
            "final_envelope_sha256",
            "execution_launch_authority_sha256",
        ):
            digest = require_sha256(getattr(self, role), role.replace("_", " "))
            if digest == "0" * 64:
                raise ProtocolError("OE-PPUR v4 resolved authority is a placeholder.")
            object.__setattr__(self, role, digest)
        if self.workspace_plan_sha256 != config.workspace_plan_sha256:
            raise ProtocolError("OE-PPUR v4 resolved workspace plan drifted.")
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "source_path", source)
        object.__setattr__(self, "artifact_root", artifact)
        object.__setattr__(self, "input_bindings", bindings)
        object.__setattr__(self, "input_manifest_path", manifest)
        object.__setattr__(self, "final_envelope_path", envelope)


def build_planned_config() -> RouterV4Config:
    protocol = frozen_protocol_payload()
    inputs = build_planned_seven_input_contract()
    return RouterV4Config(
        experiment_id=EXPERIMENT_ID,
        output_artifact_id=OUTPUT_ARTIFACT_ID,
        authorization_state=PLANNED_STATE,
        execution_amendment_issued=False,
        launch_authorized=False,
        workspace_plan_sha256=None,
        authorization_amendment_sha256=None,
        direct_input_roles=DIRECT_INPUT_ROLES,
        direct_input_artifact_ids=DIRECT_INPUT_ARTIFACT_IDS,
        protocol_hash=str(protocol["protocol_hash"]),
        seven_input_contract_hash=inputs.receipt_hash,
    )


def build_workspace_sealed_config(
    *,
    workspace_plan_sha256: str,
    authorization_amendment_sha256: str,
) -> RouterV4Config:
    protocol = frozen_protocol_payload()
    inputs = build_authorized_seven_input_contract()
    return RouterV4Config(
        experiment_id=EXPERIMENT_ID,
        output_artifact_id=OUTPUT_ARTIFACT_ID,
        authorization_state=SEALED_STATE,
        execution_amendment_issued=True,
        launch_authorized=False,
        workspace_plan_sha256=workspace_plan_sha256,
        authorization_amendment_sha256=authorization_amendment_sha256,
        direct_input_roles=DIRECT_INPUT_ROLES,
        direct_input_artifact_ids=DIRECT_INPUT_ARTIFACT_IDS,
        protocol_hash=str(protocol["protocol_hash"]),
        seven_input_contract_hash=inputs.receipt_hash,
    )


def frozen_config_contract_payload() -> dict[str, object]:
    return build_planned_config().to_payload()


def load_config(path: str | Path) -> RouterV4Config:
    source = Path(path)
    try:
        raw: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ProtocolError("OE-PPUR v4 planned config could not be loaded.") from exc
    if not isinstance(raw, Mapping) or dict(raw) != frozen_config_contract_payload():
        raise ProtocolError("OE-PPUR v4 planned config bytes drifted.")
    return build_planned_config()


__all__ = (
    "PLANNED_STATE",
    "ResolvedV4ConfigBundle",
    "SEALED_STATE",
    "RouterV4Config",
    "build_planned_config",
    "build_workspace_sealed_config",
    "frozen_config_contract_payload",
    "load_config",
    "validate_workspace_sealed_config",
)
