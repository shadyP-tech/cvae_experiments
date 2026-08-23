"""Strict path-independent config loader for authorized P-DCAPS v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from .....real_features.classifier_reference.classifiers import ClassifierSpec
from ....protocol import ProtocolError
from ..config_payloads import (
    CLASSIFIER,
    canonical_action_library_payload as canonical_v1_action_library_payload,
    canonical_claim_boundary_payload as canonical_v1_claim_boundary_payload,
    canonical_evaluation_payload,
    canonical_policy_menu_payload as canonical_v1_policy_menu_payload,
    canonical_runtime_payload as canonical_v1_runtime_payload,
)
from .experiment_contracts import (
    CANONICAL_SCRATCH_ROOT,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SOURCE_SNAPSHOT_MANIFEST_SHA256,
    EXPECTED_SOURCE_SNAPSHOT_MEMBER_COUNT,
    EXPECTED_SOURCE_SNAPSHOT_TREE_SHA256,
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
    SOURCE_SNAPSHOT_SCHEMA,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
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
from .protocol import (
    V2_METHODOLOGICAL_DELTA_ROLE,
    V2_METHODOLOGICAL_DELTAS,
    frozen_protocol_payload,
)


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
    payload = dict(canonical_v1_action_library_payload())
    payload.update(
        {
            "endpoint_donor_prior_policy": "ZERO_VECTOR_NO_FITTED_PRIOR",
            "minimum_effective_sample_size_per_class": 5.0,
        }
    )
    return payload


def canonical_policy_menu_payload() -> dict[str, object]:
    payload = dict(canonical_v1_policy_menu_payload())
    payload["response_denominators"] = (
        "derived_inside_lifecycle_from_support_plus_held"
    )
    return payload


def canonical_runtime_payload() -> dict[str, object]:
    payload = dict(canonical_v1_runtime_payload())
    payload.update(
        {
            "schema_version": "pdcaps_workstation_runtime_v2",
            "execution_authorized": True,
            "execution_authorization_basis": AUTHORIZATION_BASIS,
            "execution_revision": EXECUTION_REVISION,
            "single_use_execution_identity": True,
            "scratch_preference": [CANONICAL_SCRATCH_ROOT, "artifact_parent"],
            "v1_scratch_or_checkpoint_reuse_forbidden": True,
            "worker_results_are_manifest_hashes_and_compact_offsets_only": False,
            "worker_results_are_plain_pickle_safe_science_DTOs": True,
            "outer_task_handles_both_posterior_controls_sequentially": True,
        }
    )
    return payload


def canonical_claim_boundary_payload() -> dict[str, object]:
    payload = dict(canonical_v1_claim_boundary_payload())
    payload.update(
        {
            "schema_version": "pdcaps_claim_boundary_v2",
            "execution_authorized": True,
            "authorization_basis": AUTHORIZATION_BASIS,
            "authorization_scope": AUTHORIZATION_SCOPE,
            "single_use_execution_identity": True,
            "authorization_exhausted": False,
            "v1_output_used": False,
            "v1_amendment_used": False,
            "v1_label_capability_history_used": False,
            "v1_scratch_or_checkpoint_used": False,
            "prior_v1_execution_authorization_reused": False,
            "scientific_protocol_unchanged_from_v1": False,
            "scientific_method_changed_from_v1": True,
            "methodological_delta_role": V2_METHODOLOGICAL_DELTA_ROLE,
            "methodological_deltas": list(V2_METHODOLOGICAL_DELTAS),
            "methodological_deltas_are_terminal_consumed_test_only": True,
            "methodological_deltas_create_fresh_evidence": False,
            "methodological_deltas_are_promotable": False,
        }
    )
    return payload


@dataclass(frozen=True)
class PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV2Config:
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
        EXPECTED_SOURCE_SNAPSHOT_MANIFEST_SHA256
    )
    expected_source_snapshot_tree_sha256: str = EXPECTED_SOURCE_SNAPSHOT_TREE_SHA256
    expected_source_snapshot_member_count: int = EXPECTED_SOURCE_SNAPSHOT_MEMBER_COUNT
    authorization_basis: str = AUTHORIZATION_BASIS
    authorization_scope: str = AUTHORIZATION_SCOPE

    @property
    def config_hash(self) -> str:
        return self.contract_hash

    @property
    def execution_authorized(self) -> bool:
        return True


PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterConfig = (
    PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV2Config
)


def load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2_config(
    path: str | Path,
) -> PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV2Config:
    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read P-DCAPS v2 config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != set(CONFIG_TOP_LEVEL):
        raise ProtocolError("P-DCAPS v2 top-level config drifted.")
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
        raise ProtocolError("P-DCAPS v2 experiment identity drifted.")
    artifact_root_text = str(experiment["artifact_root"])
    if artifact_root_text.startswith("output://") and artifact_root_text != (
        f"output://{OUTPUT_ARTIFACT_ID}"
    ):
        raise ProtocolError("P-DCAPS v2 output identity drifted.")

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
        "source_snapshot_schema": SOURCE_SNAPSHOT_SCHEMA,
        "expected_source_snapshot_manifest_sha256": (
            EXPECTED_SOURCE_SNAPSHOT_MANIFEST_SHA256
        ),
        "expected_source_snapshot_tree_sha256": EXPECTED_SOURCE_SNAPSHOT_TREE_SHA256,
        "expected_source_snapshot_member_count": EXPECTED_SOURCE_SNAPSHOT_MEMBER_COUNT,
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
        raise ProtocolError("P-DCAPS v2 exact-six input schema drifted.")
    require_sha256(inputs["expected_ledger_amendment_sha256"], "amendment hash")
    for key, (artifact_id, member) in locations.items():
        value = str(inputs[key])
        expected = f"artifact://{artifact_id}" + (f"/{member}" if member else "")
        if value.startswith("artifact://") and value != expected:
            raise ProtocolError(f"P-DCAPS v2 artifact URI drifted: {key}.")
        if any(fragment in value for fragment in FORBIDDEN_INPUT_FRAGMENTS):
            raise ProtocolError(f"P-DCAPS v2 forbidden predecessor input: {key}.")

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
            raise ProtocolError(f"P-DCAPS v2 config section drifted: {key}.")
    classifier = _classifier(_section(raw, "classifier"))
    if classifier != CLASSIFIER:
        raise ProtocolError("P-DCAPS v2 classifier drifted.")

    scientific_contract = {
        "schema_version": "pdcaps_path_independent_config_v2",
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
    return PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV2Config(
        source_path=source,
        artifact_root=_resolve(source.parent, artifact_root_text),
        classifier=classifier,
        contract_hash=canonical_hash(scientific_contract),
        **resolved,
        **{key: dict(_section(raw, key)) for key in sections},
    )


load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_config = (
    load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2_config
)


def _section(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"P-DCAPS v2 config section absent: {key}.")
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
        raise ProtocolError("P-DCAPS v2 classifier payload malformed.") from exc


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
        raise ProtocolError("P-DCAPS v2 config contains a pending value.")


__all__ = (
    "CONFIG_TOP_LEVEL",
    "PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterConfig",
    "PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV2Config",
    "canonical_action_library_payload",
    "canonical_claim_boundary_payload",
    "canonical_policy_menu_payload",
    "canonical_runtime_payload",
    "load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_config",
    "load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2_config",
)
