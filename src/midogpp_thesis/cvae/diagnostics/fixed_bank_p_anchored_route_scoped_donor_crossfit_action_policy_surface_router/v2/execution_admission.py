"""Read-only execution authority for the one-shot P-DCAPS v2 diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

from ....protocol import ProtocolError
from ....runtime.artifact_io import sha256_file
from .config import (
    PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV2Config,
    load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2_config,
)
from .experiment_contracts import (
    AUTHORIZED_INPUT_ROLES,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    LEDGER_AMENDMENT_SCHEMA_VERSION,
    SOURCE_SNAPSHOT_SCHEMA,
    V1_EXPERIMENT_ID,
    V1_OUTPUT_ARTIFACT_ID,
)
from .identity import (
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    EXPERIMENT_ID,
    EXECUTION_REVISION,
    OUTPUT_ARTIFACT_ID,
    require_sha256,
)
from .input_contracts import validate_source_snapshot
from .inputs import assert_input_fence
from .protocol import PROTOCOL_SCHEMA, validate_protocol_payload
from .workspace_inputs import validate_active_workspace_binding


_REQUIRED_FALSE_REUSE_OR_HISTORY_FIELDS = (
    "v1_output_used",
    "v1_amendment_used",
    "v1_label_capability_history_used",
    "v1_scratch_or_checkpoint_used",
    "prior_v1_execution_authorization_reused",
    "prior_v2_output_used",
    "prior_v2_amendment_used",
    "prior_v2_label_capability_history_used",
    "prior_v2_scratch_or_checkpoint_used",
    "prior_v2_execution_authorization_reused",
    "v2_execution_attempted",
    "v2_run_history_used",
    "cross_run_recovery_used",
)


def assert_v2_execution_authorized(config: object) -> Mapping[str, object]:
    """Require the exact config, ledger, source, and active workspace identities.

    The function is deliberately read-only.  A runner must call it before it
    creates a lock, run state, scratch directory, preflight report, or any other
    filesystem member.
    """

    if not isinstance(
        config, PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV2Config
    ):
        raise ProtocolError("P-DCAPS v2 requires its exact canonical config loader.")
    try:
        source = Path(config.source_path)
        parent_ledger_path = Path(config.test_consumption_ledger_path)
        amendment_path = Path(config.ledger_amendment_path)
        expected_amendment_hash = require_sha256(
            config.expected_ledger_amendment_sha256,
            "v2 expected ledger amendment hash",
        )
        require_sha256(config.contract_hash, "v2 config contract hash")
        protocol = dict(config.protocol)
        claim = dict(config.claim_boundary)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProtocolError("P-DCAPS v2 authorization inputs are malformed.") from exc
    if (
        config.experiment_id != EXPERIMENT_ID
        or config.output_artifact_id != OUTPUT_ARTIFACT_ID
        or tuple(config.input_artifact_ids) != INPUT_ARTIFACT_IDS
        or config.authorization_basis != AUTHORIZATION_BASIS
        or config.authorization_scope != AUTHORIZATION_SCOPE
        or config.execution_authorized is not True
        or source.is_symlink()
        or not source.is_file()
        or parent_ledger_path.is_symlink()
        or not parent_ledger_path.is_file()
        or amendment_path.is_symlink()
        or not amendment_path.is_file()
    ):
        raise ProtocolError("P-DCAPS v2 execution identity drifted.")
    assert_input_fence(config)

    reloaded = (
        load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2_config(
            source
        )
    )
    if reloaded != config or reloaded.contract_hash != config.contract_hash:
        raise ProtocolError("P-DCAPS v2 canonical config snapshot drifted.")
    validate_protocol_payload(protocol)

    raw_config = _read_yaml_object(source)
    parent_ledger = _read_json_object(parent_ledger_path)
    amendment = _read_json_object(amendment_path)
    experiment = raw_config.get("experiment")
    raw_inputs = raw_config.get("inputs")
    raw_claim = raw_config.get("claim_boundary")
    if not all(isinstance(row, Mapping) for row in (experiment, raw_inputs, raw_claim)):
        raise ProtocolError("P-DCAPS v2 authorization config is malformed.")
    if (
        experiment.get("id") != EXPERIMENT_ID
        or experiment.get("execution_authorized") is not True
        or experiment.get("execution_authorization_basis") != AUTHORIZATION_BASIS
        or experiment.get("single_use_execution_identity") is not True
        or raw_inputs.get("ledger_amendment_artifact_id")
        != LEDGER_AMENDMENT_ARTIFACT_ID
        or raw_inputs.get("expected_ledger_amendment_sha256")
        != expected_amendment_hash
        or raw_claim.get("execution_authorized") is not True
        or raw_claim.get("authorization_basis") != AUTHORIZATION_BASIS
        or raw_claim.get("authorization_scope") != AUTHORIZATION_SCOPE
        or raw_claim.get("single_use_execution_identity") is not True
        or raw_claim.get("authorization_exhausted") is not False
        or any(
            raw_claim.get(key) is not False
            for key in _REQUIRED_FALSE_REUSE_OR_HISTORY_FIELDS[:5]
        )
        or claim.get("execution_authorized") is not True
        or claim.get("authorization_basis") != AUTHORIZATION_BASIS
        or claim.get("authorization_scope") != AUTHORIZATION_SCOPE
        or claim.get("single_use_execution_identity") is not True
        or claim.get("authorization_exhausted") is not False
    ):
        raise ProtocolError("P-DCAPS v2 config does not carry exact execution authority.")
    if sha256_file(amendment_path) != expected_amendment_hash:
        raise ProtocolError("P-DCAPS v2 ledger-amendment bytes drifted.")
    if (
        sha256_file(parent_ledger_path) != EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        or parent_ledger.get("status") != "CONSUMED_FOR_REPRESENTATION_ADOPTION"
        or parent_ledger.get("split") != "test"
    ):
        raise ProtocolError("P-DCAPS v2 parent consumption ledger drifted.")

    source_identity = validate_source_snapshot(
        expected_manifest_sha256=config.expected_source_snapshot_manifest_sha256,
        expected_tree_sha256=config.expected_source_snapshot_tree_sha256,
        expected_member_count=config.expected_source_snapshot_member_count,
    )
    _validate_authorization_amendment(config, amendment)
    workspace = validate_active_workspace_binding(config)
    return MappingProxyType(
        {
            "status": "PASS",
            "experiment_id": EXPERIMENT_ID,
            "output_artifact_id": OUTPUT_ARTIFACT_ID,
            "authorization_basis": AUTHORIZATION_BASIS,
            "authorization_scope": AUTHORIZATION_SCOPE,
            "ledger_amendment_sha256": expected_amendment_hash,
            "workspace_binding_status": workspace["status"],
            "source_snapshot_manifest_sha256": source_identity[
                "source_snapshot_manifest_sha256"
            ],
            "source_snapshot_tree_sha256": source_identity[
                "source_snapshot_tree_sha256"
            ],
            "source_snapshot_member_count": source_identity[
                "source_snapshot_member_count"
            ],
            "predecessor_or_prior_v2_state_used": False,
        }
    )


def _validate_authorization_amendment(
    config: PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV2Config,
    amendment: Mapping[str, object],
) -> None:
    if (
        amendment.get("schema_version") != LEDGER_AMENDMENT_SCHEMA_VERSION
        or amendment.get("amendment_id") != LEDGER_AMENDMENT_ARTIFACT_ID
        or amendment.get("experiment_id") != EXPERIMENT_ID
        or amendment.get("execution_revision") != EXECUTION_REVISION
        or amendment.get("protocol_schema") != PROTOCOL_SCHEMA
        or amendment.get("parent_artifact_id")
        != "midogpp_uniform_b_test_consumption_ledger_v1"
        or amendment.get("parent_member") != "reports/test_consumption_ledger.json"
        or amendment.get("parent_sha256") != EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        or amendment.get("authorized_consumer_experiment_ids") != [EXPERIMENT_ID]
        or amendment.get("authorization_basis") != AUTHORIZATION_BASIS
        or amendment.get("authorization_scope") != AUTHORIZATION_SCOPE
        or amendment.get("execution_authorized") is not True
        or amendment.get("consumed_test_reuse_authorized") is not True
        or amendment.get("authorization_is_separate_from_implementation_request")
        is not True
        or amendment.get("implementation_request_alone_authorizes_execution")
        is not False
        or amendment.get(
            "source_code_or_implementation_request_alone_authorizes_execution"
        )
        is not False
        or amendment.get("single_use_execution_identity") is not True
        or amendment.get("authorization_exhausted") is not False
        or amendment.get("direct_input_artifact_ids") != list(INPUT_ARTIFACT_IDS)
        or amendment.get("authorized_input_roles") != list(AUTHORIZED_INPUT_ROLES)
        or amendment.get("fresh_v2_workspace_aliases") != list(INPUT_ARTIFACT_IDS[2:])
        or amendment.get("v1_experiment_id") != V1_EXPERIMENT_ID
        or amendment.get("v1_output_artifact_id") != V1_OUTPUT_ARTIFACT_ID
        or amendment.get("source_snapshot_manifest_sha256")
        != config.expected_source_snapshot_manifest_sha256
        or amendment.get("source_snapshot_schema") != SOURCE_SNAPSHOT_SCHEMA
        or amendment.get("source_snapshot_tree_sha256")
        != config.expected_source_snapshot_tree_sha256
        or amendment.get("source_snapshot_member_count")
        != config.expected_source_snapshot_member_count
        or amendment.get("fresh_evidence") is not False
        or amendment.get("terminal_decision")
        != "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
        or any(
            amendment.get(key) is not False
            for key in _REQUIRED_FALSE_REUSE_OR_HISTORY_FIELDS
        )
    ):
        raise ProtocolError("P-DCAPS v2 ledger execution authority drifted.")


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Cannot read P-DCAPS v2 authorization ledger.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("P-DCAPS v2 authorization ledger must be an object.")
    return value


def _read_yaml_object(path: Path) -> dict[str, object]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read P-DCAPS v2 authorization config.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("P-DCAPS v2 authorization config must be an object.")
    return value


__all__ = ("assert_v2_execution_authorized",)
