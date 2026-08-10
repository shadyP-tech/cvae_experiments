"""Typed config loader for the actionability/recoverability diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...generation.contracts import EXPECTED_BANK_LOCK_HASH, EXPECTED_GENERATION_LOCK_HASH
from ...protocol import ProtocolError
from .config_payloads import (
    CLASSIFIER,
    canonical_action_library_payload,
    canonical_claim_boundary_payload,
    canonical_controls_payload,
    canonical_evaluation_payload,
    canonical_protocol_payload,
    canonical_recoverability_payload,
    canonical_runtime_payload,
)
from .experiment_contracts import (
    EVALUATION_SPLIT,
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
from .protocol import (
    ActionabilityRecoverabilityProtocol,
    canonical_consumed_test_protocol,
)


CONFIG_TOP_LEVEL = frozenset(
    {
        "experiment",
        "inputs",
        "protocol",
        "action_library",
        "recoverability",
        "controls",
        "classifier",
        "evaluation",
        "runtime",
        "claim_boundary",
    }
)


@dataclass(frozen=True)
class FixedBankActionabilityRecoverabilityConfig:
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
    recoverability: Mapping[str, object]
    controls: Mapping[str, object]
    classifier: ClassifierSpec
    evaluation: Mapping[str, object]
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]
    contract_hash: str

    @property
    def experiment_id(self) -> str:
        return EXPERIMENT_ID

    @property
    def output_artifact_id(self) -> str:
        return OUTPUT_ARTIFACT_ID

    @property
    def input_artifact_ids(self) -> tuple[str, ...]:
        return INPUT_ARTIFACT_IDS

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
    def actionability_recoverability_protocol(
        self,
    ) -> ActionabilityRecoverabilityProtocol:
        return canonical_consumed_test_protocol()


def load_fixed_bank_actionability_recoverability_config(
    path: str | Path,
) -> FixedBankActionabilityRecoverabilityConfig:
    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError(
            "Cannot read actionability/recoverability Stage-90 config."
        ) from exc
    if not isinstance(raw, Mapping) or set(raw) != set(CONFIG_TOP_LEVEL):
        raise ProtocolError(
            "Actionability/recoverability Stage-90 top-level config drifted."
        )
    _reject_pending(raw)

    experiment = _mapping_section(raw, "experiment")
    inputs = _mapping_section(raw, "inputs")
    protocol = _mapping_section(raw, "protocol")
    action_library = _mapping_section(raw, "action_library")
    recoverability = _mapping_section(raw, "recoverability")
    controls = _mapping_section(raw, "controls")
    classifier_raw = _mapping_section(raw, "classifier")
    evaluation = _mapping_section(raw, "evaluation")
    runtime = _mapping_section(raw, "runtime")
    claim = _mapping_section(raw, "claim_boundary")

    _require_exact(
        experiment,
        {
            "id": EXPERIMENT_ID,
            "name": EXPERIMENT_NAME,
            "artifact_root": experiment.get("artifact_root"),
            "claim_scope": "diagnostic_only",
            "status": PUBLICATION_STATUS,
        },
        "experiment",
    )
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
    if set(inputs) != set(fixed_inputs).union(locations):
        raise ProtocolError(
            "Actionability/recoverability Stage-90 input schema drifted."
        )
    for key, value in fixed_inputs.items():
        _require_exact(inputs.get(key), value, f"input {key}")
    for key, (artifact_id, member) in locations.items():
        _require_artifact_uri(inputs[key], artifact_id=artifact_id, member=member)

    canonical_sections = (
        (protocol, canonical_protocol_payload(), "protocol"),
        (action_library, canonical_action_library_payload(), "action library"),
        (recoverability, canonical_recoverability_payload(), "recoverability"),
        (controls, canonical_controls_payload(), "controls"),
        (evaluation, canonical_evaluation_payload(), "evaluation"),
        (runtime, canonical_runtime_payload(), "runtime"),
        (claim, canonical_claim_boundary_payload(), "claim boundary"),
    )
    for observed, expected, role in canonical_sections:
        _require_exact(observed, expected, role)
    classifier = _parse_classifier(classifier_raw)
    if classifier != CLASSIFIER:
        raise ProtocolError(
            "Actionability/recoverability Stage-90 classifier drifted."
        )

    artifact_root_text = _require_text(experiment["artifact_root"], "artifact root")
    if artifact_root_text.startswith("output://") and artifact_root_text != (
        f"output://{OUTPUT_ARTIFACT_ID}"
    ):
        raise ProtocolError(
            "Actionability/recoverability Stage-90 output identity drifted."
        )

    scientific = {
        "experiment_id": EXPERIMENT_ID,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "protocol": dict(protocol),
        "action_library": dict(action_library),
        "recoverability": dict(recoverability),
        "controls": dict(controls),
        "classifier": classifier.to_payload(),
        "evaluation": dict(evaluation),
        "runtime": dict(runtime),
        "claim_boundary": dict(claim),
    }
    return FixedBankActionabilityRecoverabilityConfig(
        source_path=source,
        artifact_root=_resolve_config_path(source.parent, artifact_root_text),
        expert_bank_root=_input_path(source, inputs, "expert_bank_root"),
        generation_lock_root=_input_path(source, inputs, "generation_lock_root"),
        test_cache_root=_input_path(source, inputs, "test_cache_root"),
        test_manifest_path=_input_path(source, inputs, "test_manifest_path"),
        test_consumption_ledger_path=_input_path(
            source, inputs, "test_consumption_ledger_path"
        ),
        ledger_amendment_path=_input_path(source, inputs, "ledger_amendment_path"),
        protocol=dict(protocol),
        action_library=dict(action_library),
        recoverability=dict(recoverability),
        controls=dict(controls),
        classifier=classifier,
        evaluation=dict(evaluation),
        runtime=dict(runtime),
        claim_boundary=dict(claim),
        contract_hash=stable_hash(scientific),
    )


def _mapping_section(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(
            f"Actionability/recoverability config section {key!r} is absent."
        )
    return value


def _require_exact(observed: object, expected: object, role: str) -> None:
    if observed != expected:
        raise ProtocolError(
            f"Actionability/recoverability config {role} drifted."
        )


def _require_text(value: object, role: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(
            f"Actionability/recoverability config {role} must be text."
        )
    return value


def _require_artifact_uri(value: object, *, artifact_id: str, member: str) -> None:
    expected = f"artifact://{artifact_id}" + (f"/{member}" if member else "")
    observed = _require_text(value, artifact_id)
    if observed.startswith("artifact://") and observed != expected:
        raise ProtocolError(
            "Actionability/recoverability artifact URI drifted: "
            f"{artifact_id}."
        )


def _resolve_config_path(base: Path, value: str) -> Path:
    if value.startswith(("artifact://", "output://")):
        return Path(value)
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _parse_classifier(raw: Mapping[str, object]) -> ClassifierSpec:
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
        raise ProtocolError(
            "Actionability/recoverability classifier is malformed."
        ) from exc


def _reject_pending(raw: object, trail: tuple[str, ...] = ()) -> None:
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            _reject_pending(value, (*trail, str(key)))
    elif isinstance(raw, list):
        for index, value in enumerate(raw):
            _reject_pending(value, (*trail, str(index)))
    elif isinstance(raw, str) and ("pending://" in raw or "PENDING" in raw):
        raise ProtocolError(
            "Actionability/recoverability config contains pending value at "
            f"{'.'.join(trail)}."
        )


def _input_path(source: Path, inputs: Mapping[str, object], key: str) -> Path:
    return _resolve_config_path(
        source.parent,
        _require_text(inputs[key], key.replace("_", " ")),
    )


__all__ = (
    "CLASSIFIER",
    "CONFIG_TOP_LEVEL",
    "FixedBankActionabilityRecoverabilityConfig",
    "canonical_action_library_payload",
    "canonical_claim_boundary_payload",
    "canonical_controls_payload",
    "canonical_evaluation_payload",
    "canonical_protocol_payload",
    "canonical_recoverability_payload",
    "canonical_runtime_payload",
    "load_fixed_bank_actionability_recoverability_config",
)
