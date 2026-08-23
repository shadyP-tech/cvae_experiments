"""Strict path-independent config loader for authorized P-DCAPS v4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...protocol import ProtocolError
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.config_payloads import (
    CLASSIFIER,
    canonical_action_library_payload as canonical_base_action_library_payload,
    canonical_evaluation_payload,
    canonical_policy_menu_payload as canonical_base_policy_menu_payload,
    canonical_runtime_payload as canonical_base_runtime_payload,
)
from .experiment_contracts import (
    AUTHORIZATION_DATE,
    CANONICAL_SCRATCH_ROOT,
    EXPECTED_COMBINED_THREE_SCOPE_SOURCE_SEAL_SHA256,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_V2_SOURCE_MANIFEST_SHA256,
    EXPECTED_V2_SOURCE_MEMBER_COUNT,
    EXPECTED_V2_SOURCE_TREE_SHA256,
    EXPECTED_V3_REPAIR_SOURCE_MANIFEST_SHA256,
    EXPECTED_V3_REPAIR_SOURCE_MEMBER_COUNT,
    EXPECTED_V3_REPAIR_SOURCE_TREE_SHA256,
    EXPECTED_V4_EXECUTION_SOURCE_MANIFEST_SHA256,
    EXPECTED_V4_EXECUTION_SOURCE_MEMBER_COUNT,
    EXPECTED_V4_EXECUTION_SOURCE_TREE_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    EXPERT_BANK_ARTIFACT_ID,
    FORBIDDEN_INPUT_FRAGMENTS,
    GENERATION_LOCK_ARTIFACT_ID,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    LEDGER_AMENDMENT_FILENAME,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    V2_SOURCE_SNAPSHOT_SCHEMA,
    V3_REPAIR_SOURCE_SNAPSHOT_SCHEMA,
    V4_EXECUTION_SOURCE_SNAPSHOT_SCHEMA,
)
from .identity import (
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    EXECUTION_REVISION,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    OUTPUT_ARTIFACT_ID,
    canonical_hash,
    require_sha256,
)
from .protocol import frozen_protocol_payload


CONFIG_TOP_LEVEL = frozenset(
    {
        "experiment",
        "inputs",
        "protocol",
        "action_library",
        "policy_menu",
        "classifier",
        "evaluation",
        "runtime",
        "claim_boundary",
    }
)


def canonical_action_library_payload() -> dict[str, object]:
    payload = dict(canonical_base_action_library_payload())
    payload.update(
        {
            "endpoint_donor_prior_policy": "ZERO_VECTOR_NO_FITTED_PRIOR",
            "minimum_effective_sample_size_per_class": 5.0,
        }
    )
    return payload


def canonical_policy_menu_payload() -> dict[str, object]:
    payload = dict(canonical_base_policy_menu_payload())
    payload["response_denominators"] = (
        "derived_inside_lifecycle_from_support_plus_held"
    )
    return payload


def canonical_runtime_payload() -> dict[str, object]:
    payload = dict(canonical_base_runtime_payload())
    payload.update(
        {
            "schema_version": "pdcaps_workstation_runtime_v4",
            "execution_authorized": True,
            "execution_authorization_basis": AUTHORIZATION_BASIS,
            "execution_revision": EXECUTION_REVISION,
            "authorization_date": AUTHORIZATION_DATE,
            "single_use_execution_identity": True,
            "authorization_exhausted": False,
            "consumed_test_reuse_authorized": True,
            "scratch_preference": [CANONICAL_SCRATCH_ROOT, "artifact_parent"],
            "v1_scratch_or_checkpoint_reuse_forbidden": True,
            "v2_scratch_or_checkpoint_reuse_forbidden": True,
            "v3_scratch_or_checkpoint_reuse_forbidden": True,
            "v1_v2_v3_output_or_run_state_reuse_forbidden": True,
            "cross_run_recovery_allowed": False,
            "worker_results_are_manifest_hashes_and_compact_offsets_only": False,
            "worker_results_are_plain_pickle_safe_science_DTOs": True,
            "outer_task_handles_both_posterior_controls_sequentially": True,
            "generation_device_ids": ["cuda:0", "cuda:1"],
            "persistent_generation_worker_count": 2,
            "outer_worker_start_method": "spawn",
            "outer_cpu_worker_count": 4,
            "blas_threads_per_outer_worker": 1,
            "outer_worker_task_unit": "one_complete_outer_H",
            "nested_process_pools_forbidden": True,
            "predecessor_source_seal_required": True,
            "repair_source_seal_required": True,
            "execution_source_seal_required": True,
        }
    )
    return payload


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
            "schema_version": "pdcaps_claim_boundary_v4",
            "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
            "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
            "claim_scope": "diagnostic_only",
            "execution_authorized": True,
            "implementation_authorizes_execution": False,
            "authorization_basis": AUTHORIZATION_BASIS,
            "authorization_scope": AUTHORIZATION_SCOPE,
            "authorization_date": AUTHORIZATION_DATE,
            "single_use_execution_identity": True,
            "authorization_exhausted": False,
            "consumed_test_reuse_authorized": True,
            "target_terminal_labels_may_open": True,
            "target_terminal_labels_open_only_after_durable_preterminal_attestation": True,
            "bounded_interpretation": (
                "one_shot_executable_nullable_admission_repair_on_consumed_"
                "MIDOGpp_test_only"
            ),
            "v1_output_used": False,
            "v1_amendment_used": False,
            "v1_label_capability_history_used": False,
            "v1_scratch_or_checkpoint_used": False,
            "prior_v1_execution_authorization_reused": False,
            "v2_output_used": False,
            "v2_amendment_used": False,
            "v2_label_capability_history_used": False,
            "v2_scratch_or_checkpoint_used": False,
            "prior_v2_execution_authorization_reused": False,
            "v2_execution_attempted": False,
            "v2_run_history_used": False,
            "v3_output_used": False,
            "v3_amendment_used": False,
            "v3_label_capability_history_used": False,
            "v3_scratch_or_checkpoint_used": False,
            "prior_v3_execution_authorization_reused": False,
            "cross_run_recovery_used": False,
            "v1_v2_v3_amendment_run_state_scratch_or_history_used": False,
            "scientific_protocol_unchanged_from_v3": True,
            "scientific_method_changed_from_v3": False,
            "mechanical_repair_only": True,
            "undefined_admission_statistic_representation": (
                "JSON_NULL_WITH_EXPLICIT_REASON"
            ),
            "undefined_statistic_gate_result": "FAIL_CLOSED_EXACT_P",
            "exact_p_fallback_required": True,
            "fresh_evidence": False,
            "routing_success_claimed": False,
            "downstream_utility_claimed": False,
            "nelbo_compatibility_claimed": False,
            "deployment_claimed": False,
            "promotion_eligible": False,
            "may_feed_stage50": False,
            "may_feed_stage60": False,
            "may_feed_stage70": False,
            "may_feed_another_stage90": False,
            "may_feed_another_experiment": False,
    }


@dataclass(frozen=True)
class PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV4Config:
    source_path: Path
    artifact_root: Path
    expert_bank_root: Path
    generation_lock_root: Path
    test_cache_root: Path
    test_manifest_path: Path
    test_consumption_ledger_path: Path
    ledger_amendment_path: Path
    protocol: Mapping[str, object]
    action_library: Mapping[str, object]
    policy_menu: Mapping[str, object]
    classifier: ClassifierSpec
    evaluation: Mapping[str, object]
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]
    contract_hash: str

    experiment_id: str = EXPERIMENT_ID
    output_artifact_id: str = OUTPUT_ARTIFACT_ID
    input_artifact_ids: tuple[str, ...] = INPUT_ARTIFACT_IDS
    expected_bank_lock_hash: str = EXPECTED_BANK_LOCK_HASH
    expected_generation_lock_hash: str = EXPECTED_GENERATION_LOCK_HASH
    expected_test_cache_semantic_id: str = EXPECTED_TEST_CACHE_SEMANTIC_ID
    expected_test_cache_representation_id: str = EXPECTED_TEST_CACHE_REPRESENTATION_ID
    expected_test_cache_content_hash: str = EXPECTED_TEST_CACHE_CONTENT_HASH
    expected_test_cache_row_order_hash: str = EXPECTED_TEST_CACHE_ROW_ORDER_HASH
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256
    expected_test_consumption_ledger_sha256: str = (
        EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
    )
    expected_ledger_amendment_sha256: str = EXPECTED_LEDGER_AMENDMENT_SHA256
    expected_source_snapshot_manifest_sha256: str = (
        EXPECTED_V4_EXECUTION_SOURCE_MANIFEST_SHA256
    )
    expected_source_snapshot_tree_sha256: str = (
        EXPECTED_V4_EXECUTION_SOURCE_TREE_SHA256
    )
    expected_source_snapshot_member_count: int = (
        EXPECTED_V4_EXECUTION_SOURCE_MEMBER_COUNT
    )
    expected_v2_source_snapshot_manifest_sha256: str = (
        EXPECTED_V2_SOURCE_MANIFEST_SHA256
    )
    expected_v2_source_snapshot_tree_sha256: str = EXPECTED_V2_SOURCE_TREE_SHA256
    expected_v2_source_snapshot_member_count: int = EXPECTED_V2_SOURCE_MEMBER_COUNT
    expected_v3_repair_source_manifest_sha256: str = (
        EXPECTED_V3_REPAIR_SOURCE_MANIFEST_SHA256
    )
    expected_v3_repair_source_tree_sha256: str = (
        EXPECTED_V3_REPAIR_SOURCE_TREE_SHA256
    )
    expected_v3_repair_source_member_count: int = EXPECTED_V3_REPAIR_SOURCE_MEMBER_COUNT
    expected_combined_source_seal_sha256: str = (
        EXPECTED_COMBINED_THREE_SCOPE_SOURCE_SEAL_SHA256
    )
    authorization_basis: str = AUTHORIZATION_BASIS
    authorization_scope: str = AUTHORIZATION_SCOPE

    @property
    def config_hash(self) -> str:
        return self.contract_hash

    @property
    def execution_authorized(self) -> bool:
        return True


PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterConfig = (
    PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV4Config
)


def load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4_config(
    path: str | Path,
) -> PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV4Config:
    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read P-DCAPS v4 config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != set(CONFIG_TOP_LEVEL):
        raise ProtocolError("P-DCAPS v4 top-level config drifted.")
    _reject_pending(raw)

    experiment = _section(raw, "experiment")
    expected_experiment = {
        "id": EXPERIMENT_ID,
        "name": EXPERIMENT_NAME,
        "claim_scope": "diagnostic_only",
        "status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
        "execution_authorized": True,
        "execution_authorization_basis": AUTHORIZATION_BASIS,
        "single_use_execution_identity": True,
    }
    if set(experiment) != {*expected_experiment, "artifact_root"} or any(
        experiment.get(key) != value for key, value in expected_experiment.items()
    ):
        raise ProtocolError("P-DCAPS v4 experiment identity drifted.")
    artifact_root_text = str(experiment["artifact_root"])
    if artifact_root_text.startswith("output://") and artifact_root_text != (
        f"output://{OUTPUT_ARTIFACT_ID}"
    ):
        raise ProtocolError("P-DCAPS v4 output identity drifted.")

    inputs = _section(raw, "inputs")
    fixed_inputs: dict[str, object] = {
        "expert_bank_artifact_id": EXPERT_BANK_ARTIFACT_ID,
        "generation_lock_artifact_id": GENERATION_LOCK_ARTIFACT_ID,
        "test_cache_artifact_id": TEST_CACHE_ARTIFACT_ID,
        "test_manifest_artifact_id": TEST_MANIFEST_ARTIFACT_ID,
        "test_consumption_ledger_artifact_id": TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
        "ledger_amendment_artifact_id": LEDGER_AMENDMENT_ARTIFACT_ID,
        "expected_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
        "expected_generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
        "expected_test_cache_semantic_id": EXPECTED_TEST_CACHE_SEMANTIC_ID,
        "expected_test_cache_representation_id": EXPECTED_TEST_CACHE_REPRESENTATION_ID,
        "expected_test_cache_content_hash": EXPECTED_TEST_CACHE_CONTENT_HASH,
        "expected_test_cache_row_order_hash": EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
        "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "expected_test_consumption_ledger_sha256": (
            EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        ),
        "expected_ledger_amendment_sha256": EXPECTED_LEDGER_AMENDMENT_SHA256,
        "expected_ledger_amendment_parent_sha256": (
            EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        ),
        "ledger_amendment_consumer_experiment_id": EXPERIMENT_ID,
        "ledger_amendment_execution_authorized": True,
        "ledger_amendment_authorization_basis": AUTHORIZATION_BASIS,
        "ledger_amendment_authorization_scope": AUTHORIZATION_SCOPE,
        "v2_source_snapshot_schema": V2_SOURCE_SNAPSHOT_SCHEMA,
        "expected_v2_source_snapshot_manifest_sha256": (
            EXPECTED_V2_SOURCE_MANIFEST_SHA256
        ),
        "expected_v2_source_snapshot_tree_sha256": EXPECTED_V2_SOURCE_TREE_SHA256,
        "expected_v2_source_snapshot_member_count": EXPECTED_V2_SOURCE_MEMBER_COUNT,
        "v3_repair_source_snapshot_schema": V3_REPAIR_SOURCE_SNAPSHOT_SCHEMA,
        "expected_v3_repair_source_manifest_sha256": (
            EXPECTED_V3_REPAIR_SOURCE_MANIFEST_SHA256
        ),
        "expected_v3_repair_source_tree_sha256": (
            EXPECTED_V3_REPAIR_SOURCE_TREE_SHA256
        ),
        "expected_v3_repair_source_member_count": (
            EXPECTED_V3_REPAIR_SOURCE_MEMBER_COUNT
        ),
        "source_snapshot_schema": V4_EXECUTION_SOURCE_SNAPSHOT_SCHEMA,
        "expected_source_snapshot_manifest_sha256": (
            EXPECTED_V4_EXECUTION_SOURCE_MANIFEST_SHA256
        ),
        "expected_source_snapshot_tree_sha256": (
            EXPECTED_V4_EXECUTION_SOURCE_TREE_SHA256
        ),
        "expected_source_snapshot_member_count": (
            EXPECTED_V4_EXECUTION_SOURCE_MEMBER_COUNT
        ),
        "expected_combined_source_seal_sha256": (
            EXPECTED_COMBINED_THREE_SCOPE_SOURCE_SEAL_SHA256
        ),
    }
    locations = {
        "expert_bank_root": (EXPERT_BANK_ARTIFACT_ID, ""),
        "generation_lock_root": (GENERATION_LOCK_ARTIFACT_ID, ""),
        "test_cache_root": (TEST_CACHE_ARTIFACT_ID, ""),
        "test_manifest_path": (TEST_MANIFEST_ARTIFACT_ID, "manifest.csv"),
        "test_consumption_ledger_path": (
            TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
            "reports/test_consumption_ledger.json",
        ),
        "ledger_amendment_path": (
            LEDGER_AMENDMENT_ARTIFACT_ID,
            LEDGER_AMENDMENT_FILENAME,
        ),
    }
    if set(inputs) != set(fixed_inputs) | set(locations) or any(
        inputs.get(key) != value for key, value in fixed_inputs.items()
    ):
        raise ProtocolError("P-DCAPS v4 exact-six input schema drifted.")
    require_sha256(inputs["expected_ledger_amendment_sha256"], "amendment hash")
    for key, (artifact_id, member) in locations.items():
        value = str(inputs[key])
        expected = f"artifact://{artifact_id}" + (f"/{member}" if member else "")
        if value.startswith("artifact://") and value != expected:
            raise ProtocolError(f"P-DCAPS v4 artifact URI drifted: {key}.")
        if any(fragment in value for fragment in FORBIDDEN_INPUT_FRAGMENTS):
            raise ProtocolError(f"P-DCAPS v4 forbidden predecessor input: {key}.")

    sections = {
        "protocol": frozen_protocol_payload(),
        "action_library": canonical_action_library_payload(),
        "policy_menu": canonical_policy_menu_payload(),
        "evaluation": canonical_evaluation_payload(),
        "runtime": canonical_runtime_payload(),
        "claim_boundary": canonical_claim_boundary_payload(),
    }
    for key, expected in sections.items():
        if dict(_section(raw, key)) != expected:
            raise ProtocolError(f"P-DCAPS v4 config section drifted: {key}.")
    classifier = _classifier(_section(raw, "classifier"))
    if classifier != CLASSIFIER:
        raise ProtocolError("P-DCAPS v4 classifier drifted.")

    scientific_contract = {
        "schema_version": "pdcaps_path_independent_config_v4",
        "experiment": expected_experiment,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "input_content_identities": {
            key: fixed_inputs[key]
            for key in fixed_inputs
            if key.startswith("expected_")
            or key.startswith("ledger_amendment_")
        },
        **sections,
        "classifier": classifier.to_payload(),
    }
    resolved = {
        key: _resolve(source.parent, str(inputs[key])) for key in locations
    }
    return PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV4Config(
        source_path=source,
        artifact_root=_resolve(source.parent, artifact_root_text),
        classifier=classifier,
        contract_hash=canonical_hash(scientific_contract),
        **resolved,
        **{key: dict(_section(raw, key)) for key in sections},
    )


load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_config = (
    load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4_config
)


def _section(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"P-DCAPS v4 config section absent: {key}.")
    return value


def _resolve(base: Path, value: str) -> Path:
    if value.startswith(("artifact://", "output://")):
        return Path(value)
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _classifier(raw: Mapping[str, object]) -> ClassifierSpec:
    try:
        if set(raw) != set(CLASSIFIER.to_payload()):
            raise KeyError("exact classifier schema")
        return ClassifierSpec(
            family=str(raw["family"]),
            C=float(raw["C"]),
            penalty=str(raw["penalty"]),
            solver=str(raw["solver"]),
            max_iter=int(raw["max_iter"]),
            class_weight=(
                None if raw["class_weight"] is None else str(raw["class_weight"])
            ),
            random_state=int(raw["random_state"]),
            l1_ratio=None if raw["l1_ratio"] is None else float(raw["l1_ratio"]),
            threshold_policy=str(raw["threshold_policy"]),
            scaler_fit=str(raw["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("P-DCAPS v4 classifier payload malformed.") from exc


def _reject_pending(value: object) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _reject_pending(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_pending(nested)
    elif isinstance(value, str) and any(
        token in value
        for token in ("pending://", "PENDING", "TO_BE_RECOMPUTED", "__PENDING_")
    ):
        raise ProtocolError("P-DCAPS v4 config contains a pending value.")


__all__ = (
    "CONFIG_TOP_LEVEL",
    "PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterConfig",
    "PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV4Config",
    "canonical_action_library_payload",
    "canonical_claim_boundary_payload",
    "canonical_policy_menu_payload",
    "canonical_runtime_payload",
    "load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_config",
    "load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4_config",
)
