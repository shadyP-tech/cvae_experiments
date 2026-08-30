"""Strict path-independent configuration for terminal HARP sensitivity v1."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_portfolio import HarpPolicyConfig
from ...routing.harp_protocol import canonical_hash
from .identity import (
    AUTHORIZATION_SCOPE,
    CLAIM_SCOPE,
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
    "midogpp_stage90_harp_consumed_test_cache_v1",
    "midogpp_stage90_harp_consumed_test_development_manifest_v1",
    "midogpp_stage90_harp_consumed_test_evaluation_manifest_v1",
    "midogpp_uniform_b_test_consumption_ledger_harp_parent_v1",
    "midogpp_uniform_b_test_consumption_ledger_harp_execution_amendment_v1",
)
_TOP = frozenset({"experiment", "inputs", "protocol", "model", "runtime", "claim_boundary"})
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _section(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"HARP Stage-90 config section is malformed: {key}.")
    return value


def _optional_sha256(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ProtocolError(f"HARP Stage-90 {name} must be null or a SHA-256 digest.")
    return value


def _location(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ProtocolError(f"HARP Stage-90 {name} is not a canonical location.")
    lowered = value.lower()
    if any(fragment in lowered for fragment in ("sceptre", "source_inner_candidate_utility")):
        raise ProtocolError("HARP Stage-90 cannot consume predecessor policy or utility surfaces.")
    return value


@dataclass(frozen=True, slots=True)
class HarpStage90Config:
    source_path: Path
    artifact_root: str
    input_locations: Mapping[str, str]
    expected_hashes: Mapping[str, str | None]
    execution_authorized: bool
    protocol: Mapping[str, object]
    policy: HarpPolicyConfig
    alpha_grid: tuple[float, ...]
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]
    config_hash: str
    experiment_id: str = EXPERIMENT_ID
    output_artifact_id: str = OUTPUT_ARTIFACT_ID
    input_artifact_ids: tuple[str, ...] = INPUT_ARTIFACT_IDS

    def resolved_path(self, role: str) -> Path:
        try:
            raw = self.input_locations[role]
        except KeyError as exc:
            raise ProtocolError(f"HARP Stage-90 input role is unknown: {role}.") from exc
        if "://" in raw:
            raise ProtocolError(
                "HARP Stage-90 production requires a workspace-resolved config."
            )
        return Path(raw).resolve()

    @property
    def expected_execution_amendment_sha256(self) -> str | None:
        return self.expected_hashes["execution_amendment_sha256"]


def load_config(path: str | Path) -> HarpStage90Config:
    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read HARP Stage-90 config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != _TOP:
        raise ProtocolError("HARP Stage-90 top-level config drifted.")

    experiment = _section(raw, "experiment")
    expected_experiment = {
        "schema_version": "midogpp_harp_stage90_experiment_v1",
        "id": EXPERIMENT_ID,
        "name": EXPERIMENT_NAME,
        "stage": "90_oracles_and_diagnostics",
        "claim_scope": CLAIM_SCOPE,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "implementation_authorizes_execution": False,
        "single_use_execution_identity": True,
        "consumed_test_reuse": True,
    }
    if set(experiment) != {*expected_experiment, "status", "execution_authorized", "artifact_root"}:
        raise ProtocolError("HARP Stage-90 experiment schema drifted.")
    if any(experiment.get(key) != value for key, value in expected_experiment.items()):
        raise ProtocolError("HARP Stage-90 experiment identity drifted.")
    execution_authorized = experiment.get("execution_authorized")
    if type(execution_authorized) is not bool:
        raise ProtocolError("HARP Stage-90 authorization flag must be Boolean.")
    if experiment.get("status") != ("diagnostic" if execution_authorized else "planned"):
        raise ProtocolError("HARP Stage-90 status/authorization state drifted.")
    artifact_root = _location(experiment.get("artifact_root"), name="artifact root")
    if artifact_root.startswith("output://") and artifact_root != f"output://{OUTPUT_ARTIFACT_ID}":
        raise ProtocolError("HARP Stage-90 output identity drifted.")

    inputs = _section(raw, "inputs")
    location_roles = (
        "expert_bank_root",
        "generation_lock_root",
        "test_cache_root",
        "development_manifest_path",
        "evaluation_manifest_path",
        "parent_ledger_path",
        "execution_amendment_path",
    )
    hash_roles = (
        "test_cache_content_sha256",
        "development_manifest_sha256",
        "evaluation_manifest_sha256",
        "parent_ledger_sha256",
        "execution_amendment_sha256",
    )
    if set(inputs) != {
        "schema_version",
        "direct_input_artifact_ids",
        "expert_bank_lock_hash",
        "generation_lock_hash",
        *location_roles,
        *hash_roles,
    }:
        raise ProtocolError("HARP Stage-90 input schema drifted.")
    if inputs.get("schema_version") != "midogpp_harp_stage90_exact_seven_inputs_v1" or tuple(
        inputs.get("direct_input_artifact_ids", ())
    ) != INPUT_ARTIFACT_IDS:
        raise ProtocolError("HARP Stage-90 direct input inventory drifted.")
    expected_hashes: dict[str, str | None] = {
        role: _optional_sha256(inputs.get(role), name=role) for role in hash_roles
    }
    for role in ("expert_bank_lock_hash", "generation_lock_hash"):
        value = inputs.get(role)
        if type(value) is not str or not value or not re.fullmatch(r"[0-9a-f]{16,64}", value):
            raise ProtocolError(f"HARP Stage-90 {role} is malformed.")
        expected_hashes[role] = value
    if execution_authorized and any(expected_hashes[role] is None for role in hash_roles):
        raise ProtocolError("Authorized HARP Stage-90 execution requires all input hashes.")
    locations = {role: _location(inputs.get(role), name=role) for role in location_roles}

    protocol = dict(_section(raw, "protocol"))
    required_protocol = {
        "schema_version": "midogpp_harp_stage90_terminal_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "dataset_family": "MIDOG++",
        "feature_backbone": "Virchow2_3840",
        "domain_axis": "scanner_center",
        "centers": list(CENTERS),
        "strict_outer_center_exclusion": True,
        "nested_center_lodo": True,
        "delete_donor_ensemble": True,
        "exact_nine_per_sample": True,
        "seed_cells_are_technical_replications": True,
        "complete_B_U_and_Hxe_physical_menu": True,
        "matched_budget_U_and_Hxe": True,
        "development_and_evaluation_cases_disjoint": True,
        "global_routes_sealed_before_evaluation_labels": True,
        "exact_b_byte_identical_fallback": True,
        "old_aggregate_utility_surface_used": False,
        "predecessor_policy_or_rank_used": False,
        "may_feed_stage60_or_stage70": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
    }
    if protocol != required_protocol:
        raise ProtocolError("HARP Stage-90 protocol contract drifted.")

    model = _section(raw, "model")
    if set(model) != {"schema_version", "alpha_grid", "policy", "action_semantics"} or model.get(
        "schema_version"
    ) != "midogpp_harp_stage90_model_v1":
        raise ProtocolError("HARP Stage-90 model schema drifted.")
    alpha_grid = tuple(float(value) for value in model.get("alpha_grid", ()))
    if alpha_grid != (0.01, 0.1, 1.0, 10.0):
        raise ProtocolError("HARP Stage-90 alpha grid drifted.")
    policy_raw = model.get("policy")
    if not isinstance(policy_raw, Mapping):
        raise ProtocolError("HARP Stage-90 policy thresholds are absent.")
    action_semantics = model.get("action_semantics")
    expected_action_semantics = {
        "routing_estimand": "frozen_predictive_probability_ensemble_over_frozen_generative_expert_actions",
        "matched_budget_reference_action": "U",
        "utility_deltas_reference_action": "U",
        "lambda_semantics": "post_classifier_predictive_probability_ensemble_not_generated_distribution",
        "physical_expert_routing_primary_lambda": 1.0,
        "exact_b_role": "byte_identical_abstention_and_operational_baseline",
    }
    if not isinstance(action_semantics, Mapping) or dict(action_semantics) != expected_action_semantics:
        raise ProtocolError("HARP Stage-90 action semantics drifted.")
    try:
        policy = HarpPolicyConfig(**dict(policy_raw))
    except TypeError as exc:
        raise ProtocolError("HARP Stage-90 policy threshold schema drifted.") from exc

    runtime = dict(_section(raw, "runtime"))
    expected_runtime = {
        "schema_version": "midogpp_harp_stage90_workstation_runtime_v1",
        "profile": "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        "gpu_devices": ["cuda:0", "cuda:1"],
        "persistent_gpu_workers": 2,
        "cpu_model_workers": 4,
        "blas_threads_per_worker": 3,
        "multiprocessing_start_method": "spawn",
        "parent_cuda_context_created": False,
        "late_torch_interop_setter_used": False,
        "probability_transport_dtype": "float32",
        "scientific_reduction_dtype": "float64",
        "scratch_root": "/data/local/fixed_bank_harp_router_v1",
    }
    if runtime != expected_runtime:
        raise ProtocolError("HARP Stage-90 workstation contract drifted.")

    boundary = dict(_section(raw, "claim_boundary"))
    if boundary != claim_boundary_payload(execution_authorized=execution_authorized):
        raise ProtocolError("HARP Stage-90 claim boundary drifted.")
    canonical = {
        "experiment": dict(experiment),
        "inputs": dict(inputs),
        "protocol": protocol,
        "model": {
            "schema_version": model["schema_version"],
            "alpha_grid": list(alpha_grid),
            "policy": dict(policy_raw),
            "action_semantics": dict(action_semantics),
        },
        "runtime": runtime,
        "claim_boundary": boundary,
    }
    return HarpStage90Config(
        source_path=source,
        artifact_root=artifact_root,
        input_locations=locations,
        expected_hashes=expected_hashes,
        execution_authorized=execution_authorized,
        protocol=protocol,
        policy=policy,
        alpha_grid=alpha_grid,
        runtime=runtime,
        claim_boundary=boundary,
        config_hash=canonical_hash(canonical),
    )


__all__ = ("HarpStage90Config", "INPUT_ARTIFACT_IDS", "load_config")
