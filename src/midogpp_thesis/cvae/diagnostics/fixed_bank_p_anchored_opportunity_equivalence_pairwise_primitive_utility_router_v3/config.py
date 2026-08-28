"""Planned and authorization-ready configuration for OE-PPUR v3.

The scientific protocol is lifecycle-state neutral.  Current authority is
projected consistently through the experiment, exact seven-input contract,
and claim-boundary sections and is never inferred from protocol metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import os
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
    EXPECTED_INPUT_KINDS,
    EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
    EXPECTED_SOURCE_SUPERVISION_CONTENT_SHA256,
    EXPECTED_SOURCE_SUPERVISION_ROW_ORDER_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPECTED_TEST_MANIFEST_SHA256,
    FORBIDDEN_INPUT_PATH_FRAGMENTS,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from .workspace_binding import assert_canonical_output_root
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
        payload = self._payload()
        _validate_authority_projection(payload, execution_authorized=authorized)
        object.__setattr__(self, "contract_hash", canonical_hash(payload))

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
    """Bind a separately issued amendment and parsed source receipt."""

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


def load_resolved_config(path: str | Path) -> ResolvedV3ConfigBundle:
    """Load a future workspace-rendered authorization-ready seven-input file.

    The only accepted path-bearing form is an absolute regular file named
    ``config.resolved.yaml`` inside its declared output root.  Removing the two
    path-only fields must yield the exact authorization-ready path-free config.
    This loader does not open or hash any direct input.
    """

    source = Path(path)
    candidate = Path(os.path.abspath(source))
    if not source.is_absolute() or source != candidate:
        raise ProtocolError("OE-PPUR v3 resolved config path is unsafe.")
    _reject_symlink_chain(candidate)
    try:
        resolved = source.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError("OE-PPUR v3 resolved config path is unsafe.") from exc
    if (
        source.name != "config.resolved.yaml"
        or resolved != source
        or source.is_symlink()
        or not source.is_file()
    ):
        raise ProtocolError("OE-PPUR v3 resolved config path is unsafe.")
    try:
        raw: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ProtocolError("OE-PPUR v3 resolved config could not be loaded.") from exc
    return parse_resolved_config_payload(raw, source_path=source)


def parse_resolved_config_payload(
    raw: object,
    *,
    source_path: str | Path,
) -> ResolvedV3ConfigBundle:
    """Validate a path-bearing candidate without requiring published bytes."""

    source = Path(source_path)
    candidate = Path(os.path.abspath(source))
    if (
        not source.is_absolute()
        or source != candidate
        or source.name != "config.resolved.yaml"
    ):
        raise ProtocolError("OE-PPUR v3 resolved config path is unsafe.")
    _reject_symlink_chain(source.parent)
    if not isinstance(raw, Mapping):
        raise ProtocolError("OE-PPUR v3 resolved config topology drifted.")
    normalized = dict(raw)
    experiment = normalized.get("experiment")
    inputs = normalized.get("inputs")
    if not isinstance(experiment, Mapping) or not isinstance(inputs, Mapping):
        raise ProtocolError("OE-PPUR v3 resolved config sections drifted.")
    experiment = dict(experiment)
    inputs = dict(inputs)
    artifact_root = _absolute_resolved_path(
        experiment.pop("artifact_root", None), role="artifact root"
    )
    assert_canonical_output_root(artifact_root)
    locations = inputs.pop("direct_input_locations", None)
    if source.parent != artifact_root:
        raise ProtocolError("OE-PPUR v3 resolved config escaped its artifact root.")
    if not isinstance(locations, Mapping) or tuple(locations) != DIRECT_INPUT_ROLES:
        raise ProtocolError("OE-PPUR v3 resolved input roles drifted.")
    source_section = inputs.get("source_supervision")
    if not isinstance(source_section, Mapping):
        raise ProtocolError("OE-PPUR v3 resolved source supervision is absent.")
    config = build_authorization_ready_config(
        source_supervision_content_sha256=str(source_section.get("content_sha256")),
        source_supervision_row_order_sha256=str(source_section.get("row_order_sha256")),
        source_supervision_producer_seal_sha256=str(
            source_section.get("producer_source_seal_sha256")
        ),
        source_supervision_recomputation_receipt_sha256=str(
            source_section.get("recomputation_receipt_sha256")
        ),
        authorization_amendment_sha256=str(
            inputs.get("authorization_amendment_sha256")
        ),
    )
    normalized["experiment"] = experiment
    normalized["inputs"] = inputs
    if normalized != config.to_payload():
        raise ProtocolError("OE-PPUR v3 resolved config contract drifted.")
    paths = tuple(
        _absolute_resolved_path(locations[role], role=role)
        for role in DIRECT_INPUT_ROLES
    )
    bindings = tuple(
        ResolvedDirectInput(role, artifact_id, kind, path_value)
        for role, artifact_id, kind, path_value in zip(
            DIRECT_INPUT_ROLES,
            DIRECT_INPUT_ARTIFACT_IDS,
            EXPECTED_INPUT_KINDS,
            paths,
            strict=True,
        )
    )
    return ResolvedV3ConfigBundle(
        config=config,
        source_path=source,
        artifact_root=artifact_root,
        input_bindings=bindings,
    )


def _absolute_resolved_path(value: object, *, role: str) -> Path:
    if not isinstance(value, str):
        raise ProtocolError(f"OE-PPUR v3 resolved {role} is not a string.")
    path = Path(value)
    if (
        not path.is_absolute()
        or path == Path(path.anchor)
        or ".." in path.parts
        or value.startswith(("artifact://", "output://", "file://"))
        or any(fragment in path.as_posix().lower() for fragment in FORBIDDEN_INPUT_PATH_FRAGMENTS)
    ):
        raise ProtocolError(f"OE-PPUR v3 resolved {role} is unsafe.")
    return path


def _reject_symlink_chain(path: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ProtocolError("OE-PPUR v3 resolved config path contains a symlink.")
        if current == current.parent:
            return
        current = current.parent


def _validate_authority_projection(
    payload: Mapping[str, object],
    *,
    execution_authorized: bool,
) -> None:
    """Reject cross-section authority contradictions before hashing a config."""

    experiment = payload.get("experiment")
    inputs = payload.get("inputs")
    claim = payload.get("claim_boundary")
    protocol = payload.get("protocol")
    if not all(
        isinstance(section, Mapping)
        for section in (experiment, inputs, claim, protocol)
    ):
        raise ProtocolError("OE-PPUR v3 config authority sections drifted.")
    exact = inputs.get("exact_seven_input_contract")
    if not isinstance(exact, Mapping):
        raise ProtocolError("OE-PPUR v3 exact authority projection is absent.")
    if (
        experiment.get("execution_authorized") is not execution_authorized
        or exact.get("source_supervision_materialized")
        is not execution_authorized
        or exact.get("authorization_amendment_issued")
        is not execution_authorized
        or exact.get("execution_authorized") is not execution_authorized
        or claim.get("execution_authorized") is not execution_authorized
        or claim.get("consumed_test_reuse_authorized")
        is not execution_authorized
    ):
        raise ProtocolError("OE-PPUR v3 config authority projection drifted.")
    mutable_state_keys = {
        "source_supervision_materialized",
        "authorization_amendment_issued",
        "execution_authorized",
        "consumed_test_reuse_authorized",
    }

    def protocol_keys(value: object) -> set[str]:
        if isinstance(value, Mapping):
            return set(value).union(
                *(protocol_keys(child) for child in value.values())
            )
        if isinstance(value, (list, tuple)):
            return set().union(*(protocol_keys(child) for child in value))
        return set()

    if mutable_state_keys.intersection(protocol_keys(protocol)):
        raise ProtocolError(
            "OE-PPUR v3 scientific protocol contains mutable authority state."
        )


__all__ = (
    "AUTHORIZATION_READY_STATE",
    "PLANNED_STATE",
    "ResolvedV3ConfigBundle",
    "RouterV3Config",
    "build_authorization_ready_config",
    "build_planned_config",
    "frozen_config_contract_payload",
    "load_config",
    "load_resolved_config",
    "parse_resolved_config_payload",
    "validate_authorization_ready_config",
    "validate_planned_config",
)
