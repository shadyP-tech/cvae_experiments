"""Read-only execution-authorization gate for the one-shot CBPUPR v3 run."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_file
from .config import (
    PAnchoredRouteScopedCenterBalancedPosteriorUtilityPrefixRouterConfig,
    load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config,
)
from .constants import (
    EXECUTION_REVISION,
    EXECUTION_SCHEMA_REVISION,
    EXPERIMENT_ID,
    FAILED_V2_EXPERIMENT_ID,
    FAILED_V2_OUTPUT_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
    QUARANTINED_V1_EXPERIMENT_ID,
    QUARANTINED_V1_OUTPUT_ARTIFACT_ID,
    REPAIR_BASE_COMMIT,
    REPAIR_CODE_IDENTITY,
    V1_FAILURE_EXCEPTION,
    V1_FAILURE_PHASE,
    V2_FAILURE_EXCEPTION,
    V2_FAILURE_PHASE,
)
from .experiment_contracts import (
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    AUTHORIZED_INPUT_ROLES,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    LEDGER_AMENDMENT_SCHEMA_VERSION,
    WORKSPACE_ALIAS_PLACEHOLDER_IDS,
)
from .hashing import require_sha256
from .source_seal import SOURCE_MANIFEST_FILENAME
from .workspace_inputs import validate_active_workspace_binding


_FALSE_PREDECESSOR_REUSE_FIELDS = (
    "quarantined_v1_output_used",
    "quarantined_v1_scratch_or_checkpoint_used",
    "quarantined_v1_terminal_outputs_used",
    "prior_v1_label_capability_history_used",
    "prior_v1_amendment_used",
    "failed_v2_output_used",
    "failed_v2_scratch_or_checkpoint_used",
    "failed_v2_preterminal_outputs_used",
    "prior_v2_label_capability_history_used",
    "prior_v2_amendment_used",
    "prior_v2_execution_authorization_reused",
    "previous_prediction_surfaces_used",
    "previous_stage90_outputs_used",
    "previous_stage90_amendments_used",
    "previous_stage90_scratch_or_checkpoints_used",
)


def assert_v3_execution_authorized(config: object) -> Mapping[str, object]:
    """Require a separately frozen config and single-consumer ledger amendment.

    This check is deliberately read-only and runs before the runner creates any
    directory, lock, or run-state member.  The implementation request that
    created this package is not an execution authorization.
    """

    if not isinstance(
        config,
        PAnchoredRouteScopedCenterBalancedPosteriorUtilityPrefixRouterConfig,
    ):
        raise ProtocolError(
            "CBPUPR v3 execution requires the exact canonical config loader."
        )

    try:
        source = Path(getattr(config, "source_path"))
        amendment_path = Path(getattr(config, "ledger_amendment_path"))
        expected_amendment_hash = require_sha256(
            getattr(config, "expected_ledger_amendment_sha256"),
            "CBPUPR v3 expected ledger amendment hash",
        )
        protocol = dict(getattr(config, "protocol"))
        claim = dict(getattr(config, "claim_boundary"))
        input_ids = tuple(getattr(config, "input_artifact_ids"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProtocolError(
            "CBPUPR v3 requires its canonical authorized config and ledger."
        ) from exc

    if (
        str(getattr(config, "experiment_id", "")) != EXPERIMENT_ID
        or str(getattr(config, "output_artifact_id", "")) != OUTPUT_ARTIFACT_ID
        or input_ids != INPUT_ARTIFACT_IDS
        or source.is_symlink()
        or not source.is_file()
        or amendment_path.is_symlink()
        or not amendment_path.is_file()
    ):
        raise ProtocolError(
            "CBPUPR v3 requires its canonical authorized config and ledger."
        )

    reloaded = (
        load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config(
            source
        )
    )
    if reloaded != config or reloaded.contract_hash != config.contract_hash:
        raise ProtocolError("CBPUPR v3 canonical config snapshot drifted.")

    raw_config = _read_yaml_object(source)
    amendment = _read_json_object(amendment_path)
    raw_experiment = raw_config.get("experiment")
    raw_inputs = raw_config.get("inputs")
    raw_claim = raw_config.get("claim_boundary")
    if not all(
        isinstance(value, Mapping)
        for value in (raw_experiment, raw_inputs, raw_claim)
    ):
        raise ProtocolError("CBPUPR v3 authorization config is malformed.")

    if (
        raw_experiment.get("id") != EXPERIMENT_ID
        or raw_inputs.get("expected_ledger_amendment_sha256")
        != expected_amendment_hash
        or raw_inputs.get("ledger_amendment_artifact_id")
        != LEDGER_AMENDMENT_ARTIFACT_ID
        or raw_claim.get("execution_authorized") is not True
        or raw_claim.get("execution_authorization_basis") != AUTHORIZATION_BASIS
        or raw_claim.get(
            "source_code_or_implementation_request_alone_authorizes_execution"
        )
        is not False
        or claim.get("execution_authorized") is not True
        or claim.get("execution_authorization_basis") != AUTHORIZATION_BASIS
        or claim.get("execution_requires_external_authorized_config_and_ledger")
        is not True
        or claim.get(
            "source_code_or_implementation_request_alone_authorizes_execution"
        )
        is not False
    ):
        raise ProtocolError("CBPUPR v3 config does not carry execution authority.")

    if sha256_file(amendment_path) != expected_amendment_hash:
        raise ProtocolError("CBPUPR v3 ledger-amendment bytes drifted.")

    source_fields = (
        "repair_source_manifest_member",
        "repair_source_manifest_sha256",
        "repair_source_tree_sha256",
        "repair_source_member_count",
    )
    try:
        require_sha256(
            protocol.get("repair_source_manifest_sha256"),
            "CBPUPR v3 repair source manifest hash",
        )
        require_sha256(
            protocol.get("repair_source_tree_sha256"),
            "CBPUPR v3 repair source tree hash",
        )
    except ProtocolError as exc:
        raise ProtocolError("CBPUPR v3 source authorization identity drifted.") from exc
    if (
        protocol.get("experiment_id") != EXPERIMENT_ID
        or protocol.get("execution_revision") != EXECUTION_REVISION
        or protocol.get("execution_schema_revision") != EXECUTION_SCHEMA_REVISION
        or protocol.get("repair_code_identity") != REPAIR_CODE_IDENTITY
        or protocol.get("repair_base_commit") != REPAIR_BASE_COMMIT
        or protocol.get("repair_source_manifest_member")
        != SOURCE_MANIFEST_FILENAME
        or not isinstance(protocol.get("repair_source_member_count"), int)
        or isinstance(protocol.get("repair_source_member_count"), bool)
        or protocol.get("repair_source_member_count", 0) <= 0
        or amendment.get("schema_version") != LEDGER_AMENDMENT_SCHEMA_VERSION
        or amendment.get("amendment_id") != LEDGER_AMENDMENT_ARTIFACT_ID
        or amendment.get("authorized_consumer_experiment_ids") != [EXPERIMENT_ID]
        or amendment.get("authorization_scope") != AUTHORIZATION_SCOPE
        or amendment.get("authorization_basis") != AUTHORIZATION_BASIS
        or amendment.get("execution_authorized") is not True
        or amendment.get("authorization_is_separate_from_implementation_request")
        is not True
        or amendment.get("single_use_execution_identity") is not True
        or amendment.get("authorized_input_roles") != list(AUTHORIZED_INPUT_ROLES)
        or amendment.get("fresh_v3_workspace_aliases")
        != list(WORKSPACE_ALIAS_PLACEHOLDER_IDS)
        or amendment.get("repair_code_identity") != REPAIR_CODE_IDENTITY
        or amendment.get("repair_base_commit") != REPAIR_BASE_COMMIT
        or any(amendment.get(key) != protocol.get(key) for key in source_fields)
        or amendment.get("quarantined_v1_experiment_id")
        != QUARANTINED_V1_EXPERIMENT_ID
        or amendment.get("quarantined_v1_output_artifact_id")
        != QUARANTINED_V1_OUTPUT_ARTIFACT_ID
        or amendment.get("v1_failure_preterminal") is not False
        or amendment.get("v1_failure_phase") != V1_FAILURE_PHASE
        or amendment.get("v1_failure_exception") != V1_FAILURE_EXCEPTION
        or amendment.get("failed_v2_experiment_id") != FAILED_V2_EXPERIMENT_ID
        or amendment.get("failed_v2_output_artifact_id")
        != FAILED_V2_OUTPUT_ARTIFACT_ID
        or amendment.get("v2_failure_phase") != V2_FAILURE_PHASE
        or amendment.get("v2_failure_exception") != V2_FAILURE_EXCEPTION
        or amendment.get("v2_failure_preterminal") is not True
        or amendment.get("v2_target_terminal_access_intent_persisted") is not False
        or amendment.get("v2_target_terminal_capability_had_opened") is not False
        or amendment.get("v2_terminal_outputs_had_persisted") is not False
        or amendment.get("v2_final_validation_passed") is not False
        or any(amendment.get(key) is not False for key in _FALSE_PREDECESSOR_REUSE_FIELDS)
    ):
        raise ProtocolError("CBPUPR v3 ledger execution authority drifted.")

    # A self-authored config and amendment are not authority.  The exact v3
    # experiment, six aliases, output identity, paths, and source anchors must
    # also be registered in the canonical workspace before the gate can pass.
    # This remains read-only and executes before the runner creates a lock,
    # run-state member, or output directory.
    workspace_binding = validate_active_workspace_binding(config)

    return MappingProxyType(
        {
            "status": "PASS",
            "experiment_id": EXPERIMENT_ID,
            "output_artifact_id": OUTPUT_ARTIFACT_ID,
            "authorization_basis": AUTHORIZATION_BASIS,
            "ledger_amendment_sha256": expected_amendment_hash,
            "workspace_binding_status": workspace_binding["status"],
            "predecessor_outputs_or_authorizations_reused": False,
        }
    )


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Cannot read CBPUPR v3 authorization ledger.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("CBPUPR v3 authorization ledger must be an object.")
    return value


def _read_yaml_object(path: Path) -> dict[str, object]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read CBPUPR v3 authorization config.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("CBPUPR v3 authorization config must be an object.")
    return value


__all__ = ("assert_v3_execution_authorized",)
