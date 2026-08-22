"""Exact-schema config loader for the CBPUPR diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...protocol import ProtocolError
from .config_payloads import (
    CLASSIFIER,
    canonical_action_library_payload,
    canonical_claim_boundary_payload,
    canonical_evaluation_payload,
    canonical_policy_menu_payload,
    canonical_runtime_payload,
)
from .experiment_contracts import (
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    LEDGER_AMENDMENT_FILENAME,
    OUTPUT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from .hashing import require_sha256
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


@dataclass(frozen=True)
class PAnchoredRouteScopedCenterBalancedPosteriorUtilityPrefixRouterConfig:
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
    expected_ledger_amendment_sha256: str

    experiment_id: str = EXPERIMENT_ID
    output_artifact_id: str = OUTPUT_ARTIFACT_ID
    input_artifact_ids: tuple[str, ...] = INPUT_ARTIFACT_IDS
    expected_bank_lock_hash: str = EXPECTED_BANK_LOCK_HASH
    expected_generation_lock_hash: str = EXPECTED_GENERATION_LOCK_HASH
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256
    expected_test_consumption_ledger_sha256: str = (
        EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
    )
    expected_test_cache_semantic_id: str = EXPECTED_TEST_CACHE_SEMANTIC_ID
    expected_test_cache_representation_id: str = EXPECTED_TEST_CACHE_REPRESENTATION_ID
    expected_test_cache_content_hash: str = EXPECTED_TEST_CACHE_CONTENT_HASH
    expected_test_cache_row_order_hash: str = EXPECTED_TEST_CACHE_ROW_ORDER_HASH

    @property
    def config_hash(self) -> str:
        return self.contract_hash

    def to_payload(self) -> dict[str, object]:
        return {
            "experiment": {
                "id": EXPERIMENT_ID,
                "name": EXPERIMENT_NAME,
                "artifact_root": str(self.artifact_root),
                "claim_scope": "diagnostic_only",
                "status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
            },
            "inputs": {
                "expert_bank_root": str(self.expert_bank_root),
                "generation_lock_root": str(self.generation_lock_root),
                "test_cache_root": str(self.test_cache_root),
                "test_manifest_path": str(self.test_manifest_path),
                "test_consumption_ledger_path": str(
                    self.test_consumption_ledger_path
                ),
                "ledger_amendment_path": str(self.ledger_amendment_path),
            },
            "protocol": dict(self.protocol),
            "action_library": dict(self.action_library),
            "policy_menu": dict(self.policy_menu),
            "classifier": self.classifier.to_payload(),
            "evaluation": dict(self.evaluation),
            "runtime": dict(self.runtime),
            "claim_boundary": dict(self.claim_boundary),
            "contract_hash": self.contract_hash,
        }


def load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config(
    path: str | Path,
) -> PAnchoredRouteScopedCenterBalancedPosteriorUtilityPrefixRouterConfig:
    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read CBPUPR config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != set(CONFIG_TOP_LEVEL):
        raise ProtocolError("CBPUPR top-level config drifted.")
    _reject_pending(raw)
    experiment = _section(raw, "experiment")
    inputs = _section(raw, "inputs")
    if set(experiment) != {
        "id",
        "name",
        "artifact_root",
        "claim_scope",
        "status",
    } or any(
        (
            experiment.get("id") != EXPERIMENT_ID,
            experiment.get("name") != EXPERIMENT_NAME,
            experiment.get("claim_scope") != "diagnostic_only",
            experiment.get("status") != "POST_HOC_CONSUMED_TEST_SENSITIVITY",
        )
    ):
        raise ProtocolError("CBPUPR experiment identity drifted.")
    artifact_root_text = str(experiment["artifact_root"])
    if artifact_root_text.startswith("output://") and artifact_root_text != (
        f"output://{OUTPUT_ARTIFACT_ID}"
    ):
        raise ProtocolError("CBPUPR output identity drifted.")
    fixed_inputs = {
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
        "expected_test_consumption_ledger_sha256": EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
        "expected_ledger_amendment_parent_sha256": EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
        "ledger_amendment_authorized_experiment_id": EXPERIMENT_ID,
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
    dynamic_input_keys = {"expected_ledger_amendment_sha256"}
    if set(inputs) != set(fixed_inputs) | set(locations) | dynamic_input_keys or any(
        inputs.get(key) != value for key, value in fixed_inputs.items()
    ):
        raise ProtocolError("CBPUPR exact-six input schema drifted.")
    expected_ledger_amendment_sha256 = require_sha256(
        inputs.get("expected_ledger_amendment_sha256"),
        "CBPUPR expected ledger amendment hash",
    )
    for key, (artifact_id, member) in locations.items():
        value = str(inputs[key])
        expected = f"artifact://{artifact_id}" + (f"/{member}" if member else "")
        if value.startswith("artifact://") and value != expected:
            raise ProtocolError(f"CBPUPR artifact URI drifted: {key}.")
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
            raise ProtocolError(f"CBPUPR config section drifted: {key}.")
    classifier = _classifier(_section(raw, "classifier"))
    if classifier != CLASSIFIER:
        raise ProtocolError("CBPUPR classifier drifted.")
    scientific = {
        "experiment_id": EXPERIMENT_ID,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "expected_ledger_amendment_sha256": expected_ledger_amendment_sha256,
        **sections,
        "classifier": classifier.to_payload(),
    }
    resolved = {
        key: _resolve(source.parent, str(inputs[key])) for key in locations
    }
    return PAnchoredRouteScopedCenterBalancedPosteriorUtilityPrefixRouterConfig(
        source_path=source,
        artifact_root=_resolve(source.parent, artifact_root_text),
        classifier=classifier,
        contract_hash=stable_hash(scientific),
        expected_ledger_amendment_sha256=expected_ledger_amendment_sha256,
        **resolved,
        **{key: dict(_section(raw, key)) for key in sections},
    )


def _section(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"CBPUPR config section absent: {key}.")
    return value


def _resolve(base: Path, value: str) -> Path:
    if value.startswith(("artifact://", "output://")):
        return Path(value)
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _classifier(raw: Mapping[str, object]) -> ClassifierSpec:
    try:
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
            l1_ratio=(
                None if raw["l1_ratio"] is None else float(raw["l1_ratio"])
            ),
            threshold_policy=str(raw["threshold_policy"]),
            scaler_fit=str(raw["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("CBPUPR classifier payload malformed.") from exc


def _reject_pending(value: object) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _reject_pending(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_pending(nested)
    elif isinstance(value, str) and (
        "pending://" in value
        or "PENDING" in value
        or "TO_BE_RECOMPUTED" in value
        or value.startswith("__PENDING_")
    ):
        raise ProtocolError("CBPUPR config contains a pending value.")


__all__ = (
    "CONFIG_TOP_LEVEL",
    "PAnchoredRouteScopedCenterBalancedPosteriorUtilityPrefixRouterConfig",
    "load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config",
)


# Short aliases keep generic workspace glue simple without importing older diagnostics.
CBPUPRConfig = PAnchoredRouteScopedCenterBalancedPosteriorUtilityPrefixRouterConfig
load_cbpupr_config = load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config
