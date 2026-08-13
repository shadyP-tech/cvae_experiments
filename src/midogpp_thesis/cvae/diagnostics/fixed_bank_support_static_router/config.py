"""Typed exact-schema loader for the support-static S4 diagnostic."""

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
    canonical_controls_payload,
    canonical_evaluation_payload,
    canonical_protocol_payload,
    canonical_runtime_payload,
    canonical_support_router_payload,
)
from .experiment_contracts import (
    CLAIM_SCOPE,
    EVALUATION_SPLIT,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
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
    OOF_FOLD_COUNT,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from .protocol import SupportStaticRouterProtocol, canonical_consumed_test_protocol


CONFIG_TOP_LEVEL = frozenset(
    {
        "experiment",
        "inputs",
        "protocol",
        "action_library",
        "support_router",
        "controls",
        "classifier",
        "evaluation",
        "runtime",
        "claim_boundary",
    }
)


@dataclass(frozen=True)
class FixedBankSupportStaticRouterConfig:
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
    support_router: Mapping[str, object]
    controls: Mapping[str, object]
    classifier: ClassifierSpec
    evaluation: Mapping[str, object]
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]
    contract_hash: str

    experiment_id: str = EXPERIMENT_ID
    output_artifact_id: str = OUTPUT_ARTIFACT_ID
    input_artifact_ids: tuple[str, ...] = INPUT_ARTIFACT_IDS

    @property
    def expected_bank_lock_hash(self) -> str:
        return EXPECTED_BANK_LOCK_HASH

    @property
    def expected_generation_lock_hash(self) -> str:
        return EXPECTED_GENERATION_LOCK_HASH

    @property
    def expected_manifest_sha256(self) -> str:
        return EXPECTED_MANIFEST_SHA256

    @property
    def expected_test_consumption_ledger_sha256(self) -> str:
        return EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256

    @property
    def expected_ledger_amendment_sha256(self) -> str:
        return EXPECTED_LEDGER_AMENDMENT_SHA256

    @property
    def expected_test_cache_semantic_id(self) -> str:
        return EXPECTED_TEST_CACHE_SEMANTIC_ID

    @property
    def expected_test_cache_representation_id(self) -> str:
        return EXPECTED_TEST_CACHE_REPRESENTATION_ID

    @property
    def expected_test_cache_content_hash(self) -> str:
        return EXPECTED_TEST_CACHE_CONTENT_HASH

    @property
    def expected_test_cache_row_order_hash(self) -> str:
        return EXPECTED_TEST_CACHE_ROW_ORDER_HASH

    @property
    def evaluation_split(self) -> str:
        return EVALUATION_SPLIT

    @property
    def fold_count(self) -> int:
        return OOF_FOLD_COUNT

    @property
    def support_static_router_protocol(self) -> SupportStaticRouterProtocol:
        return canonical_consumed_test_protocol()


def load_fixed_bank_support_static_router_config(
    path: str | Path,
) -> FixedBankSupportStaticRouterConfig:
    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read support-static S4 config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != set(CONFIG_TOP_LEVEL):
        raise ProtocolError("Support-static S4 top-level config drifted.")
    _reject_pending(raw)

    experiment = _section(raw, "experiment")
    inputs = _section(raw, "inputs")
    sections = {
        "protocol": canonical_protocol_payload(),
        "action_library": canonical_action_library_payload(),
        "support_router": canonical_support_router_payload(),
        "controls": canonical_controls_payload(),
        "evaluation": canonical_evaluation_payload(),
        "runtime": canonical_runtime_payload(),
        "claim_boundary": canonical_claim_boundary_payload(),
    }
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
            experiment.get("claim_scope") != CLAIM_SCOPE,
            experiment.get("status") != PUBLICATION_STATUS,
        )
    ):
        raise ProtocolError("Support-static S4 experiment identity drifted.")

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
    fixed = {
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
        "expected_ledger_amendment_sha256": EXPECTED_LEDGER_AMENDMENT_SHA256,
        "expected_ledger_amendment_parent_sha256": (
            EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        ),
        "ledger_amendment_authorized_experiment_id": EXPERIMENT_ID,
    }
    if set(inputs) != set(fixed) | set(locations):
        raise ProtocolError("Support-static S4 exact-six input schema drifted.")
    if any(inputs.get(key) != value for key, value in fixed.items()):
        raise ProtocolError("Support-static S4 frozen input identity drifted.")
    for key, (artifact_id, member) in locations.items():
        value = str(inputs[key])
        expected = f"artifact://{artifact_id}" + (f"/{member}" if member else "")
        if value.startswith("artifact://") and value != expected:
            raise ProtocolError(f"Support-static S4 artifact URI drifted: {key}.")

    for key, expected in sections.items():
        if dict(_section(raw, key)) != expected:
            raise ProtocolError(f"Support-static S4 config section drifted: {key}.")
    classifier = _classifier(_section(raw, "classifier"))
    if classifier != CLASSIFIER:
        raise ProtocolError("Support-static S4 classifier drifted.")

    artifact_root_text = str(experiment["artifact_root"])
    if artifact_root_text.startswith("output://") and artifact_root_text != (
        f"output://{OUTPUT_ARTIFACT_ID}"
    ):
        raise ProtocolError("Support-static S4 output identity drifted.")
    resolved = {key: _resolve(source.parent, str(inputs[key])) for key in locations}
    scientific = {
        "experiment_id": EXPERIMENT_ID,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        **sections,
        "classifier": classifier.to_payload(),
    }
    return FixedBankSupportStaticRouterConfig(
        source_path=source,
        artifact_root=_resolve(source.parent, artifact_root_text),
        classifier=classifier,
        contract_hash=stable_hash(scientific),
        **resolved,
        **{key: dict(_section(raw, key)) for key in sections},
    )


def _section(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Support-static S4 config section absent: {key}.")
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
                None
                if raw["class_weight"] is None
                else str(raw["class_weight"])
            ),
            random_state=int(raw["random_state"]),
            l1_ratio=(
                None if raw["l1_ratio"] is None else float(raw["l1_ratio"])
            ),
            threshold_policy=str(raw["threshold_policy"]),
            scaler_fit=str(raw["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Support-static S4 classifier payload is malformed.") from exc


def _reject_pending(value: object) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _reject_pending(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_pending(nested)
    elif isinstance(value, str) and (
        "pending://" in value or "PENDING" in value or "TO_BE_RECOMPUTED" in value
    ):
        raise ProtocolError("Support-static S4 config contains a pending value.")


__all__ = (
    "CLASSIFIER",
    "CONFIG_TOP_LEVEL",
    "FixedBankSupportStaticRouterConfig",
    "canonical_action_library_payload",
    "canonical_claim_boundary_payload",
    "canonical_controls_payload",
    "canonical_evaluation_payload",
    "canonical_protocol_payload",
    "canonical_runtime_payload",
    "canonical_support_router_payload",
    "load_fixed_bank_support_static_router_config",
)
