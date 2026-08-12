"""Typed loader for the consumed-test target-static endpoint router."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...generation.contracts import EXPECTED_BANK_LOCK_HASH, EXPECTED_GENERATION_LOCK_HASH
from ...protocol import ProtocolError
from ...routing.metadata_compatibility.contracts import DOMAIN_MAPPING_SHA256
from .config_payloads import (
    CLASSIFIER,
    canonical_action_library_payload,
    canonical_claim_boundary_payload,
    canonical_evaluation_payload,
    canonical_model_payload,
    canonical_protocol_payload,
    canonical_runtime_payload,
)
from .config_validation import (
    input_path, mapping_section, parse_classifier, reject_pending,
    require_artifact_uri, require_exact, require_text, resolve_path,
)
from .experiment_contracts import (
    EXPECTED_LEDGER_AMENDMENT_SHA256, EXPECTED_MANIFEST_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH, EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH, EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256, EXPERIMENT_ID, EXPERIMENT_NAME,
    EXPERT_BANK_ARTIFACT_ID, GENERATION_LOCK_ARTIFACT_ID, INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID, LEDGER_AMENDMENT_FILENAME,
    OUTPUT_ARTIFACT_ID, PUBLICATION_STATUS, TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID, TEST_MANIFEST_ARTIFACT_ID,
)
from .protocol import canonical_consumed_test_protocol


CONFIG_TOP_LEVEL = frozenset({
    "experiment", "inputs", "protocol", "action_library", "model",
    "classifier", "evaluation", "runtime", "claim_boundary",
})


@dataclass(frozen=True)
class ConsumedTestEndpointRouterConfig:
    source_path: Path
    artifact_root: Path
    expert_bank_root: Path
    generation_lock_root: Path
    test_cache_root: Path
    test_manifest_path: Path
    domain_mapping_path: Path
    test_consumption_ledger_path: Path
    ledger_amendment_path: Path
    protocol: Mapping[str, object]
    action_library: Mapping[str, object]
    model: Mapping[str, object]
    classifier: ClassifierSpec
    evaluation: Mapping[str, object]
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]
    contract_hash: str

    experiment_id = EXPERIMENT_ID
    output_artifact_id = OUTPUT_ARTIFACT_ID
    input_artifact_ids = INPUT_ARTIFACT_IDS
    expected_bank_lock_hash = EXPECTED_BANK_LOCK_HASH
    expected_generation_lock_hash = EXPECTED_GENERATION_LOCK_HASH
    expected_manifest_sha256 = EXPECTED_MANIFEST_SHA256
    expected_domain_mapping_sha256 = DOMAIN_MAPPING_SHA256
    expected_test_consumption_ledger_sha256 = EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
    expected_ledger_amendment_sha256 = EXPECTED_LEDGER_AMENDMENT_SHA256
    expected_test_cache_semantic_id = EXPECTED_TEST_CACHE_SEMANTIC_ID
    expected_test_cache_representation_id = EXPECTED_TEST_CACHE_REPRESENTATION_ID
    expected_test_cache_content_hash = EXPECTED_TEST_CACHE_CONTENT_HASH
    expected_test_cache_row_order_hash = EXPECTED_TEST_CACHE_ROW_ORDER_HASH

    @property
    def endpoint_router_protocol(self) -> object:
        return canonical_consumed_test_protocol()


def load_utility_aligned_consumed_test_endpoint_router_config(
    path: str | Path,
) -> ConsumedTestEndpointRouterConfig:
    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read consumed-test endpoint-router config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != set(CONFIG_TOP_LEVEL):
        raise ProtocolError("Consumed-test endpoint-router top-level config drifted.")
    reject_pending(raw)
    experiment = mapping_section(raw, "experiment")
    inputs = mapping_section(raw, "inputs")
    sections = {
        "protocol": mapping_section(raw, "protocol"),
        "action_library": mapping_section(raw, "action_library"),
        "model": mapping_section(raw, "model"),
        "evaluation": mapping_section(raw, "evaluation"),
        "runtime": mapping_section(raw, "runtime"),
        "claim_boundary": mapping_section(raw, "claim_boundary"),
    }
    require_exact(experiment, {
        "id": EXPERIMENT_ID, "name": EXPERIMENT_NAME,
        "artifact_root": experiment.get("artifact_root"),
        "claim_scope": "diagnostic_only", "status": PUBLICATION_STATUS,
    }, "experiment")
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
        "expected_domain_mapping_sha256": DOMAIN_MAPPING_SHA256,
        "expected_test_consumption_ledger_sha256": EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
        "expected_ledger_amendment_sha256": EXPECTED_LEDGER_AMENDMENT_SHA256,
        "expected_ledger_amendment_parent_sha256": EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
        "ledger_amendment_authorized_experiment_id": EXPERIMENT_ID,
    }
    locations = {
        "expert_bank_root": (EXPERT_BANK_ARTIFACT_ID, ""),
        "generation_lock_root": (GENERATION_LOCK_ARTIFACT_ID, ""),
        "test_cache_root": (TEST_CACHE_ARTIFACT_ID, ""),
        "test_manifest_path": (TEST_MANIFEST_ARTIFACT_ID, "manifest.csv"),
        "domain_mapping_path": (TEST_MANIFEST_ARTIFACT_ID, "domain_mapping.json"),
        "test_consumption_ledger_path": (TEST_CONSUMPTION_LEDGER_ARTIFACT_ID, "reports/test_consumption_ledger.json"),
        "ledger_amendment_path": (LEDGER_AMENDMENT_ARTIFACT_ID, LEDGER_AMENDMENT_FILENAME),
    }
    if set(inputs) != set(fixed_inputs).union(locations):
        raise ProtocolError("Consumed-test endpoint-router input schema drifted.")
    for key, expected in fixed_inputs.items():
        require_exact(inputs.get(key), expected, f"input {key}")
    for key, (artifact_id, member) in locations.items():
        require_artifact_uri(inputs[key], artifact_id=artifact_id, member=member)
    expected_sections = {
        "protocol": canonical_protocol_payload(),
        "action_library": canonical_action_library_payload(),
        "model": canonical_model_payload(),
        "evaluation": canonical_evaluation_payload(),
        "runtime": canonical_runtime_payload(),
        "claim_boundary": canonical_claim_boundary_payload(),
    }
    for key, expected in expected_sections.items():
        require_exact(sections[key], expected, key.replace("_", " "))
    classifier = parse_classifier(mapping_section(raw, "classifier"))
    require_exact(classifier, CLASSIFIER, "classifier")
    artifact_root_text = require_text(experiment["artifact_root"], "artifact root")
    if artifact_root_text.startswith("output://"):
        valid_artifact_root = artifact_root_text == f"output://{OUTPUT_ARTIFACT_ID}"
    else:
        valid_artifact_root = Path(artifact_root_text).is_absolute()
    if not valid_artifact_root:
        raise ProtocolError("Consumed-test endpoint-router output identity drifted.")
    scientific = {
        "experiment_id": EXPERIMENT_ID, "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        **{key: dict(value) for key, value in sections.items()},
        "classifier": classifier.to_payload(),
    }
    return ConsumedTestEndpointRouterConfig(
        source_path=source, artifact_root=resolve_path(source.parent, artifact_root_text),
        expert_bank_root=input_path(source, inputs, "expert_bank_root"),
        generation_lock_root=input_path(source, inputs, "generation_lock_root"),
        test_cache_root=input_path(source, inputs, "test_cache_root"),
        test_manifest_path=input_path(source, inputs, "test_manifest_path"),
        domain_mapping_path=input_path(source, inputs, "domain_mapping_path"),
        test_consumption_ledger_path=input_path(source, inputs, "test_consumption_ledger_path"),
        ledger_amendment_path=input_path(source, inputs, "ledger_amendment_path"),
        protocol=dict(sections["protocol"]), action_library=dict(sections["action_library"]),
        model=dict(sections["model"]), classifier=classifier,
        evaluation=dict(sections["evaluation"]), runtime=dict(sections["runtime"]),
        claim_boundary=dict(sections["claim_boundary"]), contract_hash=stable_hash(scientific),
    )


__all__ = (
    "CLASSIFIER", "CONFIG_TOP_LEVEL", "ConsumedTestEndpointRouterConfig",
    "canonical_action_library_payload", "canonical_claim_boundary_payload",
    "canonical_evaluation_payload", "canonical_model_payload",
    "canonical_protocol_payload", "canonical_runtime_payload",
    "load_utility_aligned_consumed_test_endpoint_router_config",
)
