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
    "midogpp_stage90_harp_consumed_test_cache_v6",
    "midogpp_stage90_harp_consumed_test_development_manifest_v6",
    "midogpp_stage90_harp_consumed_test_evaluation_manifest_v6",
    "midogpp_uniform_b_test_consumption_ledger_harp_parent_v6",
    "midogpp_uniform_b_test_consumption_ledger_harp_execution_amendment_v6",
)

_TOP = frozenset({"experiment", "inputs", "protocol", "model", "runtime", "claim_boundary"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
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
        raise ProtocolError(f"HARP v6 config section is malformed: {key}.")
    return value


def _optional_sha256(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ProtocolError(f"HARP v6 {name} must be null or SHA-256.")
    return value


def _location(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ProtocolError(f"HARP v6 {name} is not a canonical location.")
    lowered = value.lower()
    forbidden = (
        "fixed_bank_sceptre",
        "source_inner_candidate_utility",
        "fixed_bank_harp_router/v1",
        "fixed_bank_harp_router/v2",
        "fixed_bank_harp_router/v3",
        "fixed_bank_harp_router/v4",
        "fixed_bank_harp_router/v5",
        "harp_router_v1",
        "harp_router_v2",
        "harp_router_v3",
        "harp_router_v4",
        "harp_router_v5",
        "harp_consumed_test_cache_v1",
        "harp_consumed_test_cache_v2",
        "harp_consumed_test_cache_v3",
        "harp_consumed_test_cache_v4",
        "harp_consumed_test_cache_v5",
    )
    if any(fragment in lowered for fragment in forbidden):
        raise ProtocolError("HARP v6 cannot consume a predecessor path or policy surface.")
    return value


@dataclass(frozen=True, slots=True)
class HarpStage90V6Config:
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
            raise ProtocolError(f"HARP v6 input role is unknown: {role}.") from exc
        if "://" in raw:
            raise ProtocolError("HARP v6 production requires workspace-resolved inputs.")
        return Path(raw).resolve()

    @property
    def expected_execution_amendment_sha256(self) -> str | None:
        return self.expected_hashes["execution_amendment_sha256"]


def load_config(path: str | Path) -> HarpStage90V6Config:
    """Parse without resolving or opening any referenced artifact path."""

    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read HARP v6 config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != _TOP:
        raise ProtocolError("HARP v6 top-level config drifted.")

    experiment = _section(raw, "experiment")
    fixed_experiment = {
        "schema_version": "midogpp_harp_stage90_experiment_v6",
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
        raise ProtocolError("HARP v6 experiment schema drifted.")
    if any(experiment.get(key) != value for key, value in fixed_experiment.items()):
        raise ProtocolError("HARP v6 experiment identity drifted.")
    authorized = experiment.get("execution_authorized")
    if type(authorized) is not bool:
        raise ProtocolError("HARP v6 authorization flag must be Boolean.")
    expected_status = "diagnostic" if authorized else "planned"
    if experiment.get("status") != expected_status:
        raise ProtocolError("HARP v6 status/authorization state drifted.")
    artifact_root = _location(experiment.get("artifact_root"), name="artifact root")
    if artifact_root.startswith("output://") and artifact_root != f"output://{OUTPUT_ARTIFACT_ID}":
        raise ProtocolError("HARP v6 output identity drifted.")

    inputs = _section(raw, "inputs")
    if set(inputs) != {
        "schema_version",
        "direct_input_artifact_ids",
        "expert_bank_lock_hash",
        "generation_lock_hash",
        *_LOCATION_ROLES,
        *_HASH_ROLES,
    }:
        raise ProtocolError("HARP v6 input schema drifted.")
    if inputs.get("schema_version") != "midogpp_harp_stage90_exact_seven_inputs_v6" or tuple(
        inputs.get("direct_input_artifact_ids", ())
    ) != INPUT_ARTIFACT_IDS:
        raise ProtocolError("HARP v6 direct input inventory drifted.")
    hashes: dict[str, str | None] = {
        role: _optional_sha256(inputs.get(role), name=role) for role in _HASH_ROLES
    }
    for role in ("expert_bank_lock_hash", "generation_lock_hash"):
        value = inputs.get(role)
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{16}", value) is None:
            raise ProtocolError(
                f"HARP v6 {role} must be an exact 16-hex semantic hash."
            )
        hashes[role] = value
    activated_input_roles = tuple(
        role for role in _HASH_ROLES if role != "execution_amendment_sha256"
    )
    if authorized and any(hashes[role] is None for role in activated_input_roles):
        raise ProtocolError("Activated HARP v6 execution requires every prepared-input hash.")
    if not authorized and any(hashes[role] is not None for role in _HASH_ROLES):
        raise ProtocolError("Planned HARP v6 config may not pre-bind execution hashes.")
    locations = {role: _location(inputs.get(role), name=role) for role in _LOCATION_ROLES}
    owned_locations = {
        "test_cache_root": "harp_consumed_test_cache_v6",
        "development_manifest_path": "harp_consumed_test_development_manifest_v6",
        "evaluation_manifest_path": "harp_consumed_test_evaluation_manifest_v6",
        "parent_ledger_path": "harp_parent_v6",
        "execution_amendment_path": "harp_execution_amendment_v6",
    }
    for role, fragment in owned_locations.items():
        if "://" in locations[role] and fragment not in locations[role]:
            raise ProtocolError(f"HARP v6 {role} is not revision-owned.")

    protocol = dict(_section(raw, "protocol"))
    expected_protocol = {
        "schema_version": "midogpp_harp_stage90_terminal_protocol_v6",
        "experiment_id": EXPERIMENT_ID,
        "dataset_family": "MIDOG++",
        "feature_backbone": "Virchow2_3840",
        "domain_axis": "scanner_center",
        "utility_kind": "downstream_classifier_utility_not_NELBO",
        "routing_stage_compatibility_estimated": True,
        "compatibility_is_label_free_proxy_not_nelbo_or_utility": True,
        "generative_expert_compatibility_claimed": False,
        "centers": list(CENTERS),
        "strict_outer_center_exclusion": True,
        "nested_source_center_lodo": True,
        "delete_donor_ensemble": False,
        "seed_cells_are_technical_replications": True,
        "complete_B_U_and_Hxe_physical_menu": True,
        "directional_opportunity_surfaces_complete": True,
        "unlabeled_target_support_only_for_compatibility": True,
        "target_support_and_evaluation_cases_disjoint": True,
        "physical_expert_lambda_grid": [1.0],
        "case_level_decisions": True,
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
    if protocol != expected_protocol:
        raise ProtocolError("HARP v6 protocol contract drifted.")

    model = dict(_section(raw, "model"))
    expected_model = {
        "schema_version": "midogpp_harp_stage90_compatibility_directional_router_v6",
        "decision_unit": "case",
        "action_slate": ["B", "U", "Hxe_directional_soft_topk"],
        "directional_actions": ["D01", "D10", "ALL_MARGINS"],
        "physical_expert_lambda_grid": [1.0],
        "baseline_anchor": "exact_B_probability_vector",
        "soft_composition_reference": "exact_directional_action_surface",
        "soft_top_k": 2,
        "soft_mixture_lambda": 1.0,
        "softmax_temperature": 0.25,
        "endpoint_models": [
            "case_equal_bacc_contribution_gain",
            "brier_delta",
            "log_loss_delta",
        ],
        "primary_estimand": (
            "equal_centers_equal_classes_equal_supporting_cases_"
            "recall_at_threshold_0_5"
        ),
        "single_class_case_rule": (
            "sole_class_recall_weighted_by_total_cases_over_twice_"
            "class_supporting_cases"
        ),
        "opportunity_model": (
            "source_only_candidate_aware_hurdle_then_pairwise_directional_risk"
        ),
        "compatibility_feature": (
            "label_free_own_source_calibrated_variational_energy_proxy_"
            "mean_rank_margin_and_replica_dispersion"
        ),
        "compatibility_proxy_is_exact_nelbo": False,
        "compatibility_proxy_is_true_utility": False,
        "candidate_pool_indexed": True,
        "hurdle_target": "source_development_positive_opportunity_present",
        "pairwise_target": "action_endpoint_contrast_given_opportunity",
        "uncertainty_calibration": (
            "source_lodo_action_specific_endpoint_specific_one_sided_"
            "finite_sample_bounds_not_formal_conformal_v1"
        ),
        "uncertainty_calibration_weighting": (
            "equal_donor_then_equal_case_within_donor"
        ),
        "uncertainty_finite_sample_rule": (
            "max_donor_order_statistic_k_min_n_ceil_q_times_n_plus_1_v1"
        ),
        "selection_rule": (
            "admit_each_action_against_B_then_pairwise_rank_admissible_actions_"
            "with_exact_B_abstention"
        ),
        "alpha_grid": [0.01, 0.1, 1.0],
        "residual_quantile": 0.9,
        "opportunity_probability_threshold": 0.5,
        "policy": {
            "case_equal_bacc_contribution_gain_threshold": 0.0,
            "brier_noninferiority_margin": 0.0,
            "log_loss_noninferiority_margin": 0.0,
            "min_positive_opportunity_probability": 0.5,
            "min_compatibility_support_cases": 16,
            "min_donor_count": 4,
            "min_paired_case_count": 16,
        },
        "alpha_selected_inside_source_lodo": True,
        "policy_hyperparameters_frozen_preexecution": True,
        "learnability_admission_required": True,
        "learnability_null": "center_balanced_no_route_baseline",
        "exact_b_byte_identical_fallback": True,
    }
    if model != expected_model:
        raise ProtocolError("HARP v6 model contract drifted.")

    runtime = dict(_section(raw, "runtime"))
    expected_runtime = {
        "schema_version": "midogpp_harp_stage90_workstation_runtime_v6",
        "profile": "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        "gpu_devices": ["cuda:0", "cuda:1"],
        "persistent_gpu_workers": 2,
        "global_parent_blas_threads": 1,
        "classifier_workers": 4,
        "classifier_blas_threads_per_worker": 3,
        "science_workers": 4,
        "science_blas_threads_per_worker": 1,
        "multiprocessing_start_method": "spawn",
        "parent_cuda_context_created": False,
        "late_torch_interop_setter_used": False,
        "probability_transport_dtype": "float32",
        "scientific_reduction_dtype": "float64",
        "memory_mapped_surfaces": True,
        "bounded_inflight_batches_per_gpu": 2,
        "bounded_inflight_classifier_tasks_per_worker": 2,
        "bounded_inflight_science_tasks_per_worker": 1,
        "phase_disjoint_cpu_pools": True,
        "cuda_hidden_from_cpu_workers": True,
        "no_nested_process_pools": True,
        "scratch_root": "/data/local/fixed_bank_harp_router_v6",
    }
    if runtime != expected_runtime:
        raise ProtocolError("HARP v6 workstation runtime contract drifted.")
    boundary = dict(_section(raw, "claim_boundary"))
    if boundary != claim_boundary_payload(execution_authorized=authorized):
        raise ProtocolError("HARP v6 claim boundary drifted.")

    canonical = {
        "experiment": dict(experiment),
        "inputs": dict(inputs),
        "protocol": protocol,
        "model": model,
        "runtime": runtime,
        "claim_boundary": boundary,
    }
    return HarpStage90V6Config(
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


__all__ = ("HarpStage90V6Config", "INPUT_ARTIFACT_IDS", "load_config")
