"""Strict non-authorizing configuration for the P-DCAPS v3 repair plan.

The v3 configuration binds two immutable direct-original bank/generation
artifacts, four fresh v3 planning aliases, both source seals, the complete
unchanged v2 scientific-mechanics contract, and the workstation resource plan.
Filesystem locations are resolved for a future authorized identity, but do not
contribute to this planned contract hash.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ...protocol import ProtocolError
from .nullable_statistics import NULLABLE_STATISTIC_SCHEMA
from .identity import (
    EXPECTED_V2_SCIENTIFIC_MECHANICS_SHA256,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    V2_EXECUTION_STATUS,
    V2_EXPERIMENT_ID,
    V2_OUTPUT_ARTIFACT_ID,
    V2_PATH_INDEPENDENT_CONFIG_SHA256,
    V2_PROTOCOL_CONTRACT_SHA256,
    V2_SCIENTIFIC_MECHANICS_SCHEMA,
    canonical_hash,
    require_sha256,
)
from .protocol import frozen_protocol_payload, validate_protocol_payload
from .source_seal import (
    EXPECTED_V2_SOURCE_MANIFEST_SHA256,
    EXPECTED_V2_SOURCE_MEMBER_COUNT,
    EXPECTED_V2_SOURCE_TREE_SHA256,
    EXPECTED_V3_REPAIR_SOURCE_MANIFEST_SHA256,
    EXPECTED_V3_REPAIR_SOURCE_MEMBER_COUNT,
    EXPECTED_V3_REPAIR_SOURCE_TREE_SHA256,
    V2_SOURCE_SNAPSHOT_SCHEMA,
    V3_REPAIR_SOURCE_SNAPSHOT_SCHEMA,
)


EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
TEST_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_p_anchored_route_scoped_donor_crossfit_"
    "action_policy_surface_router_test_cache_v3"
)
TEST_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_p_anchored_route_scoped_donor_crossfit_"
    "action_policy_surface_router_test_manifest_v3"
)
TEST_CONSUMPTION_LEDGER_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router_parent_v3"
)
LEDGER_AMENDMENT_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router_amendment_v3"
)
LEDGER_AMENDMENT_FILENAME = (
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_"
    "donor_crossfit_action_policy_surface_router_ledger_amendment_v3.json"
)
INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    LEDGER_AMENDMENT_ARTIFACT_ID,
)

EXPECTED_BANK_LOCK_HASH = "9972a41dcd4814cd"
EXPECTED_GENERATION_LOCK_HASH = "34e551425710362e"
EXPECTED_TEST_CACHE_SEMANTIC_ID = "uniform_b_v2_descriptive_test_cache_v1"
EXPECTED_TEST_CACHE_REPRESENTATION_ID = "annotation_jpeg_fixed_center_b_v3"
EXPECTED_TEST_CACHE_CONTENT_HASH = (
    "df0bdbf64881ee000fe7c56bc486724313accf373ef8e90896344f8d03d187db"
)
EXPECTED_TEST_CACHE_ROW_ORDER_HASH = (
    "bd1a85b95496203500bfe2dc5232f8bfb383e73d222a8ba083e81b2c6b33c389"
)
EXPECTED_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256 = (
    "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
)
EXPECTED_LEDGER_AMENDMENT_SHA256 = (
    "6680969dac70c51a90083056ae8cf7baf4a6a2b83f2417effbf0248cb48b805c"
)

PREDECESSOR_SOURCE_SNAPSHOT_SCHEMA = V2_SOURCE_SNAPSHOT_SCHEMA
REPAIR_SOURCE_SNAPSHOT_SCHEMA = V3_REPAIR_SOURCE_SNAPSHOT_SCHEMA
NULLABLE_ADMISSION_STATISTICS_SCHEMA = NULLABLE_STATISTIC_SCHEMA

CONFIG_TOP_LEVEL = frozenset(
    {"experiment", "inputs", "protocol", "runtime", "claim_boundary"}
)


def canonical_runtime_payload() -> dict[str, object]:
    """Return the frozen non-executable workstation resource plan."""

    return {
        "schema_version": "pdcaps_v3_planned_workstation_runtime_v1",
        "workspace_status": "planned",
        "execution_authorized": False,
        "implementation_authorizes_execution": False,
        "consumed_test_reuse_authorized": False,
        "direct_runner_rejects_before_mutation": True,
        "output_root_creation_allowed": False,
        "scratch_root_creation_allowed": False,
        "cross_run_recovery_allowed": False,
        "v2_execution_status": V2_EXECUTION_STATUS,
        "v2_output_used": False,
        "v2_amendment_used": False,
        "v2_run_state_used": False,
        "v2_label_capability_history_used": False,
        "v2_scratch_or_checkpoint_used": False,
        "generation_device_ids": ["cuda:0", "cuda:1"],
        "persistent_generation_worker_count": 2,
        "outer_worker_start_method": "spawn",
        "outer_cpu_worker_count": 4,
        "blas_threads_per_outer_worker": 1,
        "outer_worker_task_unit": "one_complete_outer_H",
        "nested_process_pools_forbidden": True,
        "scratch_root": (
            "/data/local/fixed_bank_p_anchored_route_scoped_donor_crossfit_"
            "action_policy_surface_router_v3"
        ),
        "predecessor_source_seal_required": True,
        "repair_source_seal_required": True,
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    """Return the immutable terminal diagnostic claim boundary."""

    return {
        "schema_version": "pdcaps_v3_claim_boundary_v1",
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "claim_scope": "diagnostic_only",
        "bounded_interpretation": (
            "nullable_admission_statistics_mechanical_repair_plan_on_"
            "consumed_MIDOGpp_test_only"
        ),
        "execution_authorized": False,
        "implementation_authorizes_execution": False,
        "consumed_test_reuse_authorized": False,
        "target_terminal_labels_may_open": False,
        "fresh_evidence": False,
        "scientific_protocol_unchanged_from_v2": True,
        "scientific_method_changed_from_v2": False,
        "complete_v2_scientific_mechanics_payload_bound": True,
        "v2_scientific_mechanics_schema": V2_SCIENTIFIC_MECHANICS_SCHEMA,
        "v2_protocol_contract_sha256": V2_PROTOCOL_CONTRACT_SHA256,
        "v2_path_independent_config_sha256": (
            V2_PATH_INDEPENDENT_CONFIG_SHA256
        ),
        "v2_scientific_mechanics_sha256": (
            EXPECTED_V2_SCIENTIFIC_MECHANICS_SHA256
        ),
        "mechanical_repair_only": True,
        "nullable_admission_statistics_schema": (
            NULLABLE_ADMISSION_STATISTICS_SCHEMA
        ),
        "undefined_admission_statistic_representation": (
            "JSON_NULL_WITH_EXPLICIT_REASON"
        ),
        "undefined_statistic_gate_result": "FAIL_CLOSED_EXACT_P",
        "exact_p_fallback_required": True,
        "routing_success_claimed": False,
        "downstream_utility_claimed": False,
        "confidence_bound_claimed": False,
        "finite_sample_coverage_claimed": False,
        "promotion_eligible": False,
        "deployment_claimed": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
    }


@dataclass(frozen=True)
class PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV3Config:
    """Resolved locations plus the immutable v3 planning contract."""

    source_path: Path
    artifact_root: Path
    expert_bank_root: Path
    generation_lock_root: Path
    test_cache_root: Path
    test_manifest_path: Path
    test_consumption_ledger_path: Path
    ledger_amendment_path: Path
    protocol: Mapping[str, object]
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]
    contract_hash: str

    experiment_id: str = EXPERIMENT_ID
    output_artifact_id: str = OUTPUT_ARTIFACT_ID
    input_artifact_ids: tuple[str, ...] = INPUT_ARTIFACT_IDS
    expected_bank_lock_hash: str = EXPECTED_BANK_LOCK_HASH
    expected_generation_lock_hash: str = EXPECTED_GENERATION_LOCK_HASH
    expected_test_cache_semantic_id: str = EXPECTED_TEST_CACHE_SEMANTIC_ID
    expected_test_cache_representation_id: str = (
        EXPECTED_TEST_CACHE_REPRESENTATION_ID
    )
    expected_test_cache_content_hash: str = EXPECTED_TEST_CACHE_CONTENT_HASH
    expected_test_cache_row_order_hash: str = EXPECTED_TEST_CACHE_ROW_ORDER_HASH
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256
    expected_test_consumption_ledger_sha256: str = (
        EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
    )
    expected_ledger_amendment_sha256: str = EXPECTED_LEDGER_AMENDMENT_SHA256
    expected_v2_scientific_mechanics_sha256: str = (
        EXPECTED_V2_SCIENTIFIC_MECHANICS_SHA256
    )

    @property
    def config_hash(self) -> str:
        return self.contract_hash

    @property
    def execution_authorized(self) -> bool:
        return False


def load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3_config(
    path: str | Path,
) -> PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV3Config:
    """Load the exact planned v3 contract without granting execution."""

    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read P-DCAPS v3 config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != set(CONFIG_TOP_LEVEL):
        raise ProtocolError("P-DCAPS v3 top-level config drifted.")
    _reject_pending(raw)

    experiment = _section(raw, "experiment")
    expected_experiment = {
        "id": EXPERIMENT_ID,
        "name": EXPERIMENT_NAME,
        "claim_scope": "diagnostic_only",
        "publication_status": PUBLICATION_STATUS,
        "workspace_status": "planned",
        "execution_authorized": False,
        "implementation_authorizes_execution": False,
        "consumed_test_reuse_authorized": False,
        "v2_experiment_id": V2_EXPERIMENT_ID,
        "v2_output_artifact_id": V2_OUTPUT_ARTIFACT_ID,
        "v2_execution_status": V2_EXECUTION_STATUS,
    }
    if set(experiment) != {*expected_experiment, "artifact_root"} or any(
        experiment.get(key) != value
        for key, value in expected_experiment.items()
    ):
        raise ProtocolError("P-DCAPS v3 experiment identity drifted.")
    artifact_root_text = str(experiment["artifact_root"])
    if artifact_root_text.startswith("output://") and artifact_root_text != (
        f"output://{OUTPUT_ARTIFACT_ID}"
    ):
        raise ProtocolError("P-DCAPS v3 output identity drifted.")

    inputs = _section(raw, "inputs")
    fixed_inputs: dict[str, object] = {
        "expert_bank_artifact_id": EXPERT_BANK_ARTIFACT_ID,
        "generation_lock_artifact_id": GENERATION_LOCK_ARTIFACT_ID,
        "test_cache_artifact_id": TEST_CACHE_ARTIFACT_ID,
        "test_manifest_artifact_id": TEST_MANIFEST_ARTIFACT_ID,
        "test_consumption_ledger_artifact_id": (
            TEST_CONSUMPTION_LEDGER_ARTIFACT_ID
        ),
        "ledger_amendment_artifact_id": LEDGER_AMENDMENT_ARTIFACT_ID,
        "expected_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
        "expected_generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
        "expected_test_cache_semantic_id": EXPECTED_TEST_CACHE_SEMANTIC_ID,
        "expected_test_cache_representation_id": (
            EXPECTED_TEST_CACHE_REPRESENTATION_ID
        ),
        "expected_test_cache_content_hash": EXPECTED_TEST_CACHE_CONTENT_HASH,
        "expected_test_cache_row_order_hash": EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
        "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "expected_test_consumption_ledger_sha256": (
            EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        ),
        "expected_ledger_amendment_parent_sha256": (
            EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        ),
        "expected_ledger_amendment_sha256": EXPECTED_LEDGER_AMENDMENT_SHA256,
        "ledger_amendment_consumer_experiment_id": EXPERIMENT_ID,
        "ledger_amendment_execution_authorized": False,
        "predecessor_source_snapshot_schema": (
            PREDECESSOR_SOURCE_SNAPSHOT_SCHEMA
        ),
        "expected_predecessor_source_snapshot_manifest_sha256": (
            EXPECTED_V2_SOURCE_MANIFEST_SHA256
        ),
        "expected_predecessor_source_snapshot_tree_sha256": (
            EXPECTED_V2_SOURCE_TREE_SHA256
        ),
        "expected_predecessor_source_snapshot_member_count": (
            EXPECTED_V2_SOURCE_MEMBER_COUNT
        ),
        "repair_source_snapshot_schema": REPAIR_SOURCE_SNAPSHOT_SCHEMA,
        "expected_repair_source_snapshot_manifest_sha256": (
            EXPECTED_V3_REPAIR_SOURCE_MANIFEST_SHA256
        ),
        "expected_repair_source_snapshot_tree_sha256": (
            EXPECTED_V3_REPAIR_SOURCE_TREE_SHA256
        ),
        "expected_repair_source_snapshot_member_count": (
            EXPECTED_V3_REPAIR_SOURCE_MEMBER_COUNT
        ),
        "v2_output_used": False,
        "v2_amendment_used": False,
        "v2_run_state_used": False,
        "v2_label_capability_history_used": False,
        "v2_scratch_or_checkpoint_used": False,
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
        raise ProtocolError("P-DCAPS v3 exact-six input schema drifted.")
    for key in (
        "expected_test_cache_content_hash",
        "expected_test_cache_row_order_hash",
        "expected_manifest_sha256",
        "expected_test_consumption_ledger_sha256",
        "expected_ledger_amendment_parent_sha256",
        "expected_ledger_amendment_sha256",
        "expected_predecessor_source_snapshot_manifest_sha256",
        "expected_predecessor_source_snapshot_tree_sha256",
        "expected_repair_source_snapshot_manifest_sha256",
        "expected_repair_source_snapshot_tree_sha256",
    ):
        require_sha256(inputs[key], key)
    for key, (artifact_id, member) in locations.items():
        value = str(inputs[key])
        expected = f"artifact://{artifact_id}" + (
            f"/{member}" if member else ""
        )
        if value.startswith("artifact://") and value != expected:
            raise ProtocolError(f"P-DCAPS v3 artifact URI drifted: {key}.")

    protocol = dict(_section(raw, "protocol"))
    if protocol != frozen_protocol_payload():
        raise ProtocolError("P-DCAPS v3 config section drifted: protocol.")
    validate_protocol_payload(protocol)
    runtime = dict(_section(raw, "runtime"))
    if runtime != canonical_runtime_payload():
        raise ProtocolError("P-DCAPS v3 config section drifted: runtime.")
    claim_boundary = dict(_section(raw, "claim_boundary"))
    if claim_boundary != canonical_claim_boundary_payload():
        raise ProtocolError("P-DCAPS v3 config section drifted: claim_boundary.")

    scientific_contract = {
        "schema_version": "pdcaps_v3_path_independent_config_v1",
        "experiment": expected_experiment,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "input_content_identities": fixed_inputs,
        "protocol": protocol,
        "runtime": runtime,
        "claim_boundary": claim_boundary,
    }
    resolved = {
        key: _resolve(source.parent, str(inputs[key])) for key in locations
    }
    return PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV3Config(
        source_path=source,
        artifact_root=_resolve(source.parent, artifact_root_text),
        protocol=protocol,
        runtime=runtime,
        claim_boundary=claim_boundary,
        contract_hash=canonical_hash(scientific_contract),
        **resolved,
    )


def _section(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"P-DCAPS v3 config section absent: {key}.")
    return value


def _resolve(base: Path, value: str) -> Path:
    if value.startswith(("artifact://", "output://")):
        return Path(value)
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


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
        raise ProtocolError("P-DCAPS v3 config contains a pending value.")


__all__ = (
    "CONFIG_TOP_LEVEL",
    "EXPECTED_LEDGER_AMENDMENT_SHA256",
    "INPUT_ARTIFACT_IDS",
    "LEDGER_AMENDMENT_ARTIFACT_ID",
    "LEDGER_AMENDMENT_FILENAME",
    "NULLABLE_ADMISSION_STATISTICS_SCHEMA",
    "PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV3Config",
    "TEST_CACHE_ARTIFACT_ID",
    "TEST_CONSUMPTION_LEDGER_ARTIFACT_ID",
    "TEST_MANIFEST_ARTIFACT_ID",
    "canonical_claim_boundary_payload",
    "canonical_runtime_payload",
    "load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3_config",
)
