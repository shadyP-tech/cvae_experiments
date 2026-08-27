"""Path-free two-state configuration for OE-PPUR v2.

Loading a config validates artifact *identities* only.  It never resolves an
artifact URI, output root, scratch root, or input path.  Filesystem resolution
belongs exclusively to the read-only execution-admission phase.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256
from .identity import (
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
    CLAIM_SCOPE,
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPECTED_BANK_CONTENT_INDEX_SHA256,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
    EXPECTED_INPUT_KINDS,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPECTED_TEST_MANIFEST_SHA256,
    FRESH_EVIDENCE,
    INPUT_RELATIVE_MEMBERS,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
)
from .protocol import (
    claim_boundary_payload,
    frozen_protocol_payload,
    validate_claim_boundary,
    validate_protocol_payload,
)


PLANNED_STATE = "PLANNED_NOT_AUTHORIZED"
AUTHORIZATION_READY_STATE = "AUTHORIZATION_READY_EXTERNAL_AMENDMENT"
CONFIG_TOP_LEVEL = frozenset(
    {"experiment", "inputs", "protocol", "source_provenance", "claim_boundary"}
)

if TYPE_CHECKING:
    from .workspace_inputs import WorkspaceInputBinding


def _input_locations_payload() -> dict[str, str]:
    return {
        role: (
            f"artifact://{artifact_id}"
            + (f"/{member}" if member else "")
        )
        for role, artifact_id, member in zip(
            DIRECT_INPUT_ROLES,
            DIRECT_INPUT_ARTIFACT_IDS,
            INPUT_RELATIVE_MEMBERS,
            strict=True,
        )
    }


def _inputs_payload(
    *, expected_authorization_amendment_sha256: str | None
) -> dict[str, object]:
    return {
        "schema_version": "oe_ppur_v2_exact_six_input_config_v1",
        "direct_input_count": 6,
        "direct_input_roles": list(DIRECT_INPUT_ROLES),
        "direct_input_artifact_ids": list(DIRECT_INPUT_ARTIFACT_IDS),
        "direct_input_locations": _input_locations_payload(),
        "input_path_resolution_deferred_until_admission": True,
        "expert_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
        "expert_bank_content_index_file_sha256": (
            EXPECTED_BANK_CONTENT_INDEX_SHA256
        ),
        "generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
        "generation_lock_content_index_file_sha256": (
            EXPECTED_GENERATION_CONTENT_INDEX_SHA256
        ),
        "test_cache_semantic_id": EXPECTED_TEST_CACHE_SEMANTIC_ID,
        "test_cache_representation_id": EXPECTED_TEST_CACHE_REPRESENTATION_ID,
        "test_cache_content_sha256": EXPECTED_TEST_CACHE_CONTENT_HASH,
        "test_cache_row_order_sha256": EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
        "test_manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
        "original_parent_ledger_sha256": EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
        "authorization_amendment_artifact_id": (
            AUTHORIZATION_AMENDMENT_ARTIFACT_ID
        ),
        "expected_authorization_amendment_sha256": (
            expected_authorization_amendment_sha256
        ),
        "predecessor_input_reuse_allowed": False,
        "cross_run_recovery_allowed": False,
    }


def _config_payload(
    *,
    authorization_state: str,
    source_contract_hash: str | None,
    expected_authorization_amendment_sha256: str | None,
) -> dict[str, object]:
    authorized = authorization_state == AUTHORIZATION_READY_STATE
    return {
        "experiment": {
            "schema_version": "oe_ppur_v2_experiment_config_v1",
            "id": EXPERIMENT_ID,
            "name": EXPERIMENT_NAME,
            "stage": "90_oracles_and_diagnostics",
            "status": "diagnostic" if authorized else "planned",
            "claim_scope": CLAIM_SCOPE,
            "publication_status": PUBLICATION_STATUS,
            "fresh_evidence": FRESH_EVIDENCE,
            "artifact_root": f"output://{OUTPUT_ARTIFACT_ID}",
            "authorization_state": authorization_state,
            "execution_authorized": authorized,
            "consumed_test_reuse_authorized": authorized,
            "single_use_execution_identity": authorized,
            "authorization_exhausted": False,
            "implementation_authorizes_execution": False,
        },
        "inputs": _inputs_payload(
            expected_authorization_amendment_sha256=(
                expected_authorization_amendment_sha256
            )
        ),
        "protocol": frozen_protocol_payload(),
        "source_provenance": {
            "schema_version": "oe_ppur_v2_source_contract_binding_v1",
            "source_contract_hash": source_contract_hash,
            "source_must_revalidate_before_admission": True,
            "source_hash_resolution_deferred_until_admission": True,
        },
        "claim_boundary": claim_boundary_payload(
            execution_authorized=authorized
        ),
    }


@dataclass(frozen=True, slots=True)
class RouterV2Config:
    """Validated path-free config in one of two monotone authority states."""

    source_path: Path | None
    authorization_state: str
    source_contract_hash: str | None
    expected_authorization_amendment_sha256: str | None
    contract_hash: str = field(init=False)
    experiment_id: str = EXPERIMENT_ID
    output_artifact_id: str = OUTPUT_ARTIFACT_ID
    input_artifact_ids: tuple[str, ...] = DIRECT_INPUT_ARTIFACT_IDS
    input_roles: tuple[str, ...] = DIRECT_INPUT_ROLES

    def __post_init__(self) -> None:
        if (
            self.experiment_id != EXPERIMENT_ID
            or self.output_artifact_id != OUTPUT_ARTIFACT_ID
            or tuple(self.input_artifact_ids) != DIRECT_INPUT_ARTIFACT_IDS
            or tuple(self.input_roles) != DIRECT_INPUT_ROLES
            or len(set(self.input_artifact_ids)) != 6
            or self.authorization_state
            not in {PLANNED_STATE, AUTHORIZATION_READY_STATE}
        ):
            raise ProtocolError("OE-PPUR v2 config identity drifted.")
        if self.authorization_state == PLANNED_STATE:
            if (
                self.source_contract_hash is not None
                or self.expected_authorization_amendment_sha256 is not None
            ):
                raise ProtocolError(
                    "OE-PPUR v2 planned config cannot carry authorization evidence."
                )
        else:
            source_hash = require_sha256(
                self.source_contract_hash, "source contract hash"
            )
            amendment_hash = require_sha256(
                self.expected_authorization_amendment_sha256,
                "authorization amendment hash",
            )
            if source_hash == "0" * 64 or amendment_hash == "0" * 64:
                raise ProtocolError(
                    "OE-PPUR v2 authorization-ready hashes cannot be placeholders."
                )
        validate_protocol_payload(frozen_protocol_payload())
        validate_claim_boundary(
            claim_boundary_payload(execution_authorized=self.execution_authorized),
            execution_authorized=self.execution_authorized,
        )
        object.__setattr__(self, "input_artifact_ids", DIRECT_INPUT_ARTIFACT_IDS)
        object.__setattr__(self, "input_roles", DIRECT_INPUT_ROLES)
        object.__setattr__(self, "contract_hash", canonical_hash(self.to_payload()))

    @property
    def execution_authorized(self) -> bool:
        return self.authorization_state == AUTHORIZATION_READY_STATE

    @property
    def consumed_test_reuse_authorized(self) -> bool:
        return self.execution_authorized

    @property
    def authorization_exhausted(self) -> bool:
        return False

    @property
    def artifact_root(self) -> str:
        return f"output://{OUTPUT_ARTIFACT_ID}"

    @property
    def protocol(self) -> dict[str, object]:
        return frozen_protocol_payload()

    @property
    def claim_boundary(self) -> dict[str, object]:
        return claim_boundary_payload(
            execution_authorized=self.execution_authorized
        )

    @property
    def inputs(self) -> dict[str, object]:
        return _inputs_payload(
            expected_authorization_amendment_sha256=(
                self.expected_authorization_amendment_sha256
            )
        )

    def to_payload(self) -> dict[str, object]:
        return _config_payload(
            authorization_state=self.authorization_state,
            source_contract_hash=self.source_contract_hash,
            expected_authorization_amendment_sha256=(
                self.expected_authorization_amendment_sha256
            ),
        )


RouterConfig = RouterV2Config


@dataclass(frozen=True, slots=True)
class ResolvedConfigBundle:
    """Workspace-rendered paths paired with the canonical path-free config."""

    config: RouterV2Config
    source_path: Path
    artifact_root: Path
    input_bindings: tuple["WorkspaceInputBinding", ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.config, RouterV2Config)
            or self.config.authorization_state != AUTHORIZATION_READY_STATE
            or self.config.execution_authorized is not True
            or not self.source_path.is_absolute()
            or self.source_path.name != "config.resolved.yaml"
            or not self.artifact_root.is_absolute()
            or self.source_path.parent != self.artifact_root
            or tuple(getattr(row, "role", None) for row in self.input_bindings)
            != DIRECT_INPUT_ROLES
            or tuple(
                getattr(row, "artifact_id", None) for row in self.input_bindings
            )
            != DIRECT_INPUT_ARTIFACT_IDS
        ):
            raise ProtocolError("OE-PPUR v2 resolved config bundle drifted.")

    @property
    def contract_hash(self) -> str:
        return self.config.contract_hash


def build_planned_config() -> RouterV2Config:
    """Build the only state shipped before a real amendment is authorized."""

    return RouterV2Config(
        source_path=None,
        authorization_state=PLANNED_STATE,
        source_contract_hash=None,
        expected_authorization_amendment_sha256=None,
    )


def build_authorization_ready_config(
    *,
    source_contract_hash: str,
    expected_authorization_amendment_sha256: str,
) -> RouterV2Config:
    """Bind externally issued evidence; this function does not issue authority."""

    return RouterV2Config(
        source_path=None,
        authorization_state=AUTHORIZATION_READY_STATE,
        source_contract_hash=source_contract_hash,
        expected_authorization_amendment_sha256=(
            expected_authorization_amendment_sha256
        ),
    )


def frozen_config_contract_payload() -> dict[str, object]:
    """Return the current checked-in, non-authorized config payload."""

    return build_planned_config().to_payload()


def load_config(path: str | Path) -> RouterV2Config:
    """Load an exact path-free payload without resolving any artifact URI."""

    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read OE-PPUR v2 config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != set(CONFIG_TOP_LEVEL):
        raise ProtocolError("OE-PPUR v2 top-level config drifted.")
    _reject_pending(raw)
    return _config_from_canonical_payload(raw, source=source)


def load_resolved_config(path: str | Path) -> ResolvedConfigBundle:
    """Load the workspace-rendered executable snapshot with exact six paths.

    Unlike :func:`load_config`, this entry point accepts absolute values only
    from a file named ``config.resolved.yaml`` and only when the normalized
    path-free payload is an authorization-ready contract.  Existence, hashes,
    symlinks, overlap, and immutable input bytes remain admission concerns.
    """

    source = Path(path)
    if (
        not source.is_absolute()
        or source.name != "config.resolved.yaml"
        or source.is_symlink()
    ):
        raise ProtocolError("OE-PPUR v2 resolved config path is unsafe.")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read OE-PPUR v2 resolved config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != set(CONFIG_TOP_LEVEL):
        raise ProtocolError("OE-PPUR v2 resolved config topology drifted.")
    _reject_pending(raw)
    experiment = _section(raw, "experiment")
    if (
        experiment.get("authorization_state") != AUTHORIZATION_READY_STATE
        or experiment.get("execution_authorized") is not True
    ):
        raise ProtocolError(
            "OE-PPUR v2 resolved paths require authorization-ready config."
        )
    raw_artifact_root = experiment.get("artifact_root")
    artifact_root = _absolute_resolved_location(
        raw_artifact_root, role="artifact root"
    )
    if source.parent != artifact_root:
        raise ProtocolError(
            "OE-PPUR v2 resolved config escaped its declared artifact root."
        )
    inputs = _section(raw, "inputs")
    raw_locations = inputs.get("direct_input_locations")
    if (
        not isinstance(raw_locations, Mapping)
        or tuple(raw_locations) != DIRECT_INPUT_ROLES
    ):
        raise ProtocolError("OE-PPUR v2 resolved input roles drifted.")
    locations = tuple(
        _absolute_resolved_location(raw_locations[role], role=role)
        for role in DIRECT_INPUT_ROLES
    )
    if len(set(locations)) != 6:
        raise ProtocolError("OE-PPUR v2 resolved input paths are duplicated.")
    for path_value, member, role in zip(
        locations, INPUT_RELATIVE_MEMBERS, DIRECT_INPUT_ROLES, strict=True
    ):
        if member and not _has_relative_suffix(path_value, Path(member)):
            raise ProtocolError(f"OE-PPUR v2 resolved {role} member drifted.")

    normalized = copy.deepcopy(dict(raw))
    normalized_experiment = normalized.get("experiment")
    normalized_inputs = normalized.get("inputs")
    if not isinstance(normalized_experiment, dict) or not isinstance(
        normalized_inputs, dict
    ):
        raise ProtocolError("OE-PPUR v2 resolved config sections drifted.")
    normalized_experiment["artifact_root"] = f"output://{OUTPUT_ARTIFACT_ID}"
    normalized_inputs["direct_input_locations"] = _input_locations_payload()
    config = _config_from_canonical_payload(normalized, source=source)

    # Local import keeps ordinary planned-config loading independent from the
    # filesystem admission adapter.
    from .workspace_inputs import WorkspaceInputBinding

    bindings = tuple(
        WorkspaceInputBinding(role, artifact_id, location, kind)
        for role, artifact_id, location, kind in zip(
            DIRECT_INPUT_ROLES,
            DIRECT_INPUT_ARTIFACT_IDS,
            locations,
            EXPECTED_INPUT_KINDS,
            strict=True,
        )
    )
    return ResolvedConfigBundle(
        config=config,
        source_path=source,
        artifact_root=artifact_root,
        input_bindings=bindings,
    )


def _config_from_canonical_payload(
    raw: Mapping[str, Any], *, source: Path
) -> RouterV2Config:
    experiment = _section(raw, "experiment")
    source_provenance = _section(raw, "source_provenance")
    state = experiment.get("authorization_state")
    if state not in {PLANNED_STATE, AUTHORIZATION_READY_STATE}:
        raise ProtocolError("OE-PPUR v2 authorization state drifted.")
    inputs = _section(raw, "inputs")
    config = RouterV2Config(
        source_path=source,
        authorization_state=str(state),
        source_contract_hash=source_provenance.get("source_contract_hash"),
        expected_authorization_amendment_sha256=inputs.get(
            "expected_authorization_amendment_sha256"
        ),
    )
    if dict(raw) != config.to_payload():
        raise ProtocolError("OE-PPUR v2 config contract drifted.")
    validate_protocol_payload(_section(raw, "protocol"))
    validate_claim_boundary(
        _section(raw, "claim_boundary"),
        execution_authorized=config.execution_authorized,
    )
    return config


def _absolute_resolved_location(value: object, *, role: str) -> Path:
    if not isinstance(value, str):
        raise ProtocolError(f"OE-PPUR v2 resolved {role} is not a string.")
    path = Path(value)
    if (
        not path.is_absolute()
        or path == Path(path.anchor)
        or ".." in path.parts
        or value.startswith(("artifact://", "output://", "file://"))
    ):
        raise ProtocolError(f"OE-PPUR v2 resolved {role} is unsafe.")
    return path


def _has_relative_suffix(path: Path, suffix: Path) -> bool:
    return len(path.parts) >= len(suffix.parts) and path.parts[-len(suffix.parts) :] == (
        suffix.parts
    )


def _section(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"OE-PPUR v2 {name} section is not a mapping.")
    return dict(value)


def _reject_pending(value: object) -> None:
    if isinstance(value, str) and (
        "__PENDING" in value or value.startswith("file://")
    ):
        raise ProtocolError("OE-PPUR v2 config contains an unsafe placeholder/path.")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_pending(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_pending(item)


__all__ = (
    "AUTHORIZATION_READY_STATE",
    "CONFIG_TOP_LEVEL",
    "PLANNED_STATE",
    "RouterConfig",
    "RouterV2Config",
    "ResolvedConfigBundle",
    "build_authorization_ready_config",
    "build_planned_config",
    "frozen_config_contract_payload",
    "load_config",
    "load_resolved_config",
)
