"""Strict, path-independent configuration for the fenced HARP diagnostic."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re

import yaml

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from .identity import (
    CLAIM_SCOPE,
    EXECUTION_REVISION,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    claim_boundary_payload,
)


INPUT_ARTIFACT_IDS = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1",
    "midogpp_output_uniform_b_v2_generation_lock_v1",
    "midogpp_stage90_harp_target_train_support_full_test_cache_v16",
    "midogpp_stage90_harp_target_train_support_label_capability_v16",
    "midogpp_stage90_harp_full_test_evaluation_release_v16",
    "midogpp_uniform_b_test_consumption_ledger_harp_parent_v16",
    "midogpp_uniform_b_test_consumption_ledger_harp_execution_amendment_v16",
)

_TOP = frozenset({"experiment", "inputs", "protocol", "model", "runtime", "claim_boundary"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PROTOCOL_CONTRACT_HASH = (
    "491673b695aef3eba310c029b662b8db6ced4beb5195791248ecc66363d16413"
)
_MODEL_CONTRACT_HASH = (
    "c03503e8d537478e785b59bfd1150267ee99d5fc51114eca3d9cb598808d3ca2"
)
_RUNTIME_CONTRACT_HASH = (
    "24f4f9cdb6dc46394fc445d837b9c6c83379bb3b66b6dc6d450ca26d800806d3"
)
_LOCATION_ROLES = (
    "expert_bank_root",
    "generation_lock_root",
    "test_cache_root",
    "development_manifest_path",
    "evaluation_manifest_path",
    "parent_ledger_path",
    "execution_amendment_path",
)
_HASH_ROLES = (
    "test_cache_content_sha256",
    "development_manifest_sha256",
    "evaluation_manifest_sha256",
    "parent_ledger_sha256",
    "execution_amendment_sha256",
)


def _section(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"HARP v16 config section is malformed: {key}.")
    return value


def _optional_sha256(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ProtocolError(f"HARP v16 {name} must be null or SHA-256.")
    return value


def _location(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ProtocolError(f"HARP v16 {name} is not a canonical location.")
    lowered = value.lower()
    forbidden = (
        "fixed_bank_sceptre",
        "source_inner_candidate_utility",
        "fixed_bank_harp_router/v1",
        "fixed_bank_harp_router/v2",
        "fixed_bank_harp_router/v3",
        "fixed_bank_harp_router/v4",
        "fixed_bank_harp_router/v5",
        "fixed_bank_harp_router/v6",
        "fixed_bank_harp_router/v7",
        "fixed_bank_harp_router/v8",
        "fixed_bank_harp_router/v9",
        "fixed_bank_harp_router/v10",
        "fixed_bank_harp_router/v11",
        "fixed_bank_harp_router/v12",
        "fixed_bank_harp_router/v13",
        "fixed_bank_harp_router/v14",
        "fixed_bank_harp_router/v15",
        "harp_router_v1",
        "harp_router_v2",
        "harp_router_v3",
        "harp_router_v4",
        "harp_router_v5",
        "harp_router_v6",
        "harp_router_v7",
        "harp_router_v8",
        "harp_router_v9",
        "harp_router_v10",
        "harp_router_v11",
        "harp_router_v12",
        "harp_router_v13",
        "harp_router_v14",
        "harp_router_v15",
        "harp_consumed_test_cache_v1",
        "harp_consumed_test_cache_v2",
        "harp_consumed_test_cache_v3",
        "harp_consumed_test_cache_v4",
        "harp_consumed_test_cache_v5",
        "harp_consumed_test_cache_v6",
        "harp_consumed_test_cache_v7",
        "harp_consumed_test_cache_v8",
        "harp_consumed_test_cache_v9",
        "harp_consumed_test_cache_v10",
        "harp_consumed_test_cache_v11",
        "harp_consumed_test_cache_v12",
        "harp_consumed_test_cache_v13",
        "harp_consumed_test_cache_v14",
        "harp_consumed_test_cache_v15",
        "harp_source_train_full_test_cache_v9",
        "harp_source_train_label_capability_v9",
        "harp_full_test_evaluation_release_v9",
        "harp_source_train_full_test_cache_v10",
        "harp_source_train_label_capability_v10",
        "harp_full_test_evaluation_release_v10",
        "harp_source_train_full_test_cache_v11",
        "harp_source_train_label_capability_v11",
        "harp_full_test_evaluation_release_v11",
        "harp_source_train_full_test_cache_v12",
        "harp_source_train_label_capability_v12",
        "harp_full_test_evaluation_release_v12",
        "harp_source_train_full_test_cache_v13",
        "harp_source_train_label_capability_v13",
        "harp_full_test_evaluation_release_v13",
        "harp_source_train_full_test_cache_v14",
        "harp_source_train_label_capability_v14",
        "harp_full_test_evaluation_release_v14",
        "harp_target_train_support_full_test_cache_v15",
        "harp_target_train_support_label_capability_v15",
        "harp_full_test_evaluation_release_v15",
        "harp_v1_execution",
        "harp_v2_execution",
        "harp_v3_execution",
        "harp_v4_execution",
        "harp_v5_execution",
        "harp_v6_execution",
        "harp_v7_execution",
        "harp_v8_execution",
        "harp_v9_execution",
        "harp_v10_execution",
        "harp_v11_execution",
        "harp_v12_execution",
        "harp_v13_execution",
        "harp_v14_execution",
        "harp_v15_execution",
        "source_active_selective_router_v7",
        "baseline_inclusive_action_safe_router_v8",
        "policy_calibrated_residual_router_v9",
        "policy_calibrated_residual_router_v10",
        "policy_calibrated_residual_router_v11",
        "policy_calibrated_residual_router_v12",
        "policy_calibrated_residual_router_v13",
        "policy_calibrated_residual_router_v14",
        "hierarchical_support_action_risk_router_v15",
        "dense_residual_soft_router",
        "compatibility_conditioned_directional_router",
    )
    if any(
        re.search(re.escape(fragment) + r"(?![0-9])", lowered) is not None
        for fragment in forbidden
    ):
        raise ProtocolError("HARP v16 cannot consume a predecessor path or policy surface.")
    return value


@dataclass(frozen=True, slots=True)
class HarpStage90V16Config:
    source_path: Path
    artifact_root: str
    input_locations: Mapping[str, str]
    expected_hashes: Mapping[str, str | None]
    execution_authorized: bool
    protocol: Mapping[str, object]
    model: Mapping[str, object]
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]
    config_hash: str
    experiment_id: str = EXPERIMENT_ID
    output_artifact_id: str = OUTPUT_ARTIFACT_ID
    input_artifact_ids: tuple[str, ...] = INPUT_ARTIFACT_IDS
    execution_revision: str = EXECUTION_REVISION

    def resolved_path(self, role: str) -> Path:
        try:
            raw = self.input_locations[role]
        except KeyError as exc:
            raise ProtocolError(f"HARP v16 input role is unknown: {role}.") from exc
        if "://" in raw:
            raise ProtocolError("HARP v16 production requires workspace-resolved inputs.")
        return Path(raw).resolve()

    @property
    def expected_execution_amendment_sha256(self) -> str | None:
        return self.expected_hashes["execution_amendment_sha256"]


def load_config(path: str | Path) -> HarpStage90V16Config:
    """Parse without resolving or opening any referenced artifact path."""

    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read HARP v16 config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != _TOP:
        raise ProtocolError("HARP v16 top-level config drifted.")

    experiment = _section(raw, "experiment")
    fixed_experiment = {
        "schema_version": "midogpp_harp_stage90_experiment_v16",
        "id": EXPERIMENT_ID,
        "name": EXPERIMENT_NAME,
        "stage": "90_oracles_and_diagnostics",
        "claim_scope": CLAIM_SCOPE,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "execution_revision": EXECUTION_REVISION,
        "implementation_authorizes_execution": False,
        "single_use_execution_identity": True,
        "consumed_test_reuse": True,
    }
    if set(experiment) != {*fixed_experiment, "status", "execution_authorized", "artifact_root"}:
        raise ProtocolError("HARP v16 experiment schema drifted.")
    if any(experiment.get(key) != value for key, value in fixed_experiment.items()):
        raise ProtocolError("HARP v16 experiment identity drifted.")
    authorized = experiment.get("execution_authorized")
    if type(authorized) is not bool:
        raise ProtocolError("HARP v16 authorization flag must be Boolean.")
    expected_status = "diagnostic" if authorized else "planned"
    if experiment.get("status") != expected_status:
        raise ProtocolError("HARP v16 status/authorization state drifted.")
    artifact_root = _location(experiment.get("artifact_root"), name="artifact root")
    if artifact_root.startswith("output://") and artifact_root != f"output://{OUTPUT_ARTIFACT_ID}":
        raise ProtocolError("HARP v16 output identity drifted.")

    inputs = _section(raw, "inputs")
    if set(inputs) != {
        "schema_version",
        "direct_input_artifact_ids",
        "expert_bank_lock_hash",
        "generation_lock_hash",
        *_LOCATION_ROLES,
        *_HASH_ROLES,
    }:
        raise ProtocolError("HARP v16 input schema drifted.")
    if inputs.get("schema_version") != "midogpp_harp_stage90_exact_seven_inputs_v16" or tuple(
        inputs.get("direct_input_artifact_ids", ())
    ) != INPUT_ARTIFACT_IDS:
        raise ProtocolError("HARP v16 direct input inventory drifted.")
    hashes: dict[str, str | None] = {
        role: _optional_sha256(inputs.get(role), name=role) for role in _HASH_ROLES
    }
    for role in ("expert_bank_lock_hash", "generation_lock_hash"):
        value = inputs.get(role)
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{16}", value) is None:
            raise ProtocolError(
                f"HARP v16 {role} must be an exact 16-hex semantic hash."
            )
        hashes[role] = value
    activated_input_roles = tuple(
        role for role in _HASH_ROLES if role != "execution_amendment_sha256"
    )
    if authorized and any(hashes[role] is None for role in activated_input_roles):
        raise ProtocolError("Activated HARP v16 execution requires every prepared-input hash.")
    if not authorized and any(hashes[role] is not None for role in _HASH_ROLES):
        raise ProtocolError("Planned HARP v16 config may not pre-bind execution hashes.")
    locations = {role: _location(inputs.get(role), name=role) for role in _LOCATION_ROLES}
    owned_locations = {
        "test_cache_root": "harp_target_train_support_full_test_cache_v16",
        "development_manifest_path": "harp_target_train_support_label_capability_v16",
        "evaluation_manifest_path": "harp_full_test_evaluation_release_v16",
        "parent_ledger_path": "harp_parent_v16",
        "execution_amendment_path": "harp_execution_amendment_v16",
    }
    for role, fragment in owned_locations.items():
        if "://" in locations[role] and fragment not in locations[role]:
            raise ProtocolError(f"HARP v16 {role} is not revision-owned.")

    protocol = dict(_section(raw, "protocol"))
    # Bind the complete support-local protocol byte-semantically while keeping
    # parsing independent of workstation paths and activation state.
    if canonical_hash(protocol) != _PROTOCOL_CONTRACT_HASH:
        raise ProtocolError("HARP v16 protocol contract drifted.")

    model = dict(_section(raw, "model"))
    if canonical_hash(model) != _MODEL_CONTRACT_HASH:
        raise ProtocolError("HARP v16 model contract drifted.")

    runtime = dict(_section(raw, "runtime"))
    if canonical_hash(runtime) != _RUNTIME_CONTRACT_HASH:
        raise ProtocolError("HARP v16 workstation runtime contract drifted.")
    boundary = dict(_section(raw, "claim_boundary"))
    if boundary != claim_boundary_payload(execution_authorized=authorized):
        raise ProtocolError("HARP v16 claim boundary drifted.")

    canonical = {
        "experiment": dict(experiment),
        "inputs": dict(inputs),
        "protocol": protocol,
        "model": model,
        "runtime": runtime,
        "claim_boundary": boundary,
    }
    return HarpStage90V16Config(
        source_path=source,
        artifact_root=artifact_root,
        input_locations=locations,
        expected_hashes=hashes,
        execution_authorized=authorized,
        protocol=protocol,
        model=model,
        runtime=runtime,
        claim_boundary=boundary,
        config_hash=canonical_hash(canonical),
    )


__all__ = ("HarpStage90V16Config", "INPUT_ARTIFACT_IDS", "load_config")
