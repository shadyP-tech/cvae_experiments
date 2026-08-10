"""Typed config loader for the terminal hierarchical residual stacker."""

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
    canonical_claim_boundary_payload,
    canonical_controls_payload,
    canonical_evaluation_payload,
    canonical_features_payload,
    canonical_hierarchical_model_payload,
    canonical_probability_surface_payload,
    canonical_protocol_payload,
    canonical_runtime_payload,
    canonical_stacker_payload,
    canonical_target_support_payload,
)
from .config_validation import (
    CONFIG_TOP_LEVEL,
    mapping_section,
    parse_classifier,
    reject_pending,
    require_artifact_uri,
    require_exact,
    require_text,
    resolve_config_path,
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
    OOF_FOLD_COUNT,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)


@dataclass(frozen=True)
class FixedBankHierarchicalResidualStackerConfig:
    source_path: Path
    artifact_root: Path
    expert_bank_root: Path
    generation_lock_root: Path
    test_cache_root: Path
    test_manifest_path: Path
    test_consumption_ledger_path: Path
    ledger_amendment_path: Path
    protocol: Mapping[str, object]
    probability_surface: Mapping[str, object]
    features: Mapping[str, object]
    hierarchical_model: Mapping[str, object]
    target_support: Mapping[str, object]
    stacker: Mapping[str, object]
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
    def expected_manifest_sha256(self) -> str:
        return EXPECTED_MANIFEST_SHA256

    @property
    def evaluation_split(self) -> str:
        return EVALUATION_SPLIT

    @property
    def fold_count(self) -> int:
        return OOF_FOLD_COUNT

    @property
    def clip_epsilon(self) -> float:
        return float(self.probability_surface["logit_clip_epsilon"])

    @property
    def probability_threshold(self) -> float:
        return float(self.probability_surface["probability_threshold"])

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.features["model_feature_names"])

    @property
    def local_feature_names(self) -> tuple[str, ...]:
        return tuple(
            str(value) for value in self.features["local_residual_feature_names"]
        )

    @property
    def ridge_alpha_grid(self) -> tuple[float, ...]:
        return tuple(
            float(value) for value in self.hierarchical_model["ridge_alpha_grid"]
        )

    @property
    def intercept_grid(self) -> tuple[float, ...]:
        return tuple(
            float(value) for value in self.target_support["B_cal_intercept_grid"]
        )

    @property
    def lambda_grid(self) -> tuple[float, ...]:
        return tuple(
            float(value) for value in self.target_support["residual_lambda_grid"]
        )

    @property
    def variance_floor(self) -> float:
        return float(self.target_support["variance_floor"])

    @property
    def confidence_multiplier(self) -> float:
        return float(self.target_support["confidence_multiplier"])


def load_fixed_bank_hierarchical_residual_stacker_config(
    path: str | Path,
) -> FixedBankHierarchicalResidualStackerConfig:
    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read residual-stacker Stage-90 config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != set(CONFIG_TOP_LEVEL):
        raise ProtocolError("Residual-stacker Stage-90 top-level config drifted.")
    reject_pending(raw)

    experiment = mapping_section(raw, "experiment")
    inputs = mapping_section(raw, "inputs")
    protocol = mapping_section(raw, "protocol")
    probability_surface = mapping_section(raw, "probability_surface")
    features = mapping_section(raw, "features")
    hierarchical_model = mapping_section(raw, "hierarchical_model")
    target_support = mapping_section(raw, "target_support")
    stacker = mapping_section(raw, "stacker")
    controls = mapping_section(raw, "controls")
    classifier_raw = mapping_section(raw, "classifier")
    evaluation = mapping_section(raw, "evaluation")
    runtime = mapping_section(raw, "runtime")
    claim = mapping_section(raw, "claim_boundary")

    require_exact(
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
            "uniform_b_v2_consumed_test_fixed_bank_hierarchical_residual_"
            "stacker_ledger_amendment_v1.json",
        ),
    }
    if set(inputs) != set(fixed_inputs).union(locations):
        raise ProtocolError("Residual-stacker Stage-90 input schema drifted.")
    for key, value in fixed_inputs.items():
        require_exact(inputs.get(key), value, f"input {key}")
    for key, (artifact_id, member) in locations.items():
        require_artifact_uri(inputs[key], artifact_id=artifact_id, member=member)

    require_exact(protocol, canonical_protocol_payload(), "protocol")
    require_exact(
        probability_surface,
        canonical_probability_surface_payload(),
        "probability surface",
    )
    require_exact(features, canonical_features_payload(), "features")
    require_exact(
        hierarchical_model,
        canonical_hierarchical_model_payload(),
        "hierarchical model",
    )
    require_exact(target_support, canonical_target_support_payload(), "target support")
    require_exact(stacker, canonical_stacker_payload(), "stacker")
    require_exact(controls, canonical_controls_payload(), "controls")
    require_exact(evaluation, canonical_evaluation_payload(), "evaluation")
    require_exact(runtime, canonical_runtime_payload(), "runtime")
    require_exact(claim, canonical_claim_boundary_payload(), "claim boundary")
    classifier = parse_classifier(classifier_raw)
    if classifier != CLASSIFIER:
        raise ProtocolError("Residual-stacker Stage-90 classifier drifted.")

    artifact_root_text = require_text(experiment["artifact_root"], "artifact root")
    if artifact_root_text.startswith("output://") and artifact_root_text != (
        f"output://{OUTPUT_ARTIFACT_ID}"
    ):
        raise ProtocolError("Residual-stacker Stage-90 output identity drifted.")

    scientific = {
        "experiment_id": EXPERIMENT_ID,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "protocol": dict(protocol),
        "probability_surface": dict(probability_surface),
        "features": dict(features),
        "hierarchical_model": dict(hierarchical_model),
        "target_support": dict(target_support),
        "stacker": dict(stacker),
        "controls": dict(controls),
        "classifier": classifier.to_payload(),
        "evaluation": dict(evaluation),
        "runtime": dict(runtime),
        "claim_boundary": dict(claim),
    }
    return FixedBankHierarchicalResidualStackerConfig(
        source_path=source,
        artifact_root=resolve_config_path(source.parent, artifact_root_text),
        expert_bank_root=_input_path(source, inputs, "expert_bank_root"),
        generation_lock_root=_input_path(source, inputs, "generation_lock_root"),
        test_cache_root=_input_path(source, inputs, "test_cache_root"),
        test_manifest_path=_input_path(source, inputs, "test_manifest_path"),
        test_consumption_ledger_path=_input_path(
            source, inputs, "test_consumption_ledger_path"
        ),
        ledger_amendment_path=_input_path(source, inputs, "ledger_amendment_path"),
        protocol=dict(protocol),
        probability_surface=dict(probability_surface),
        features=dict(features),
        hierarchical_model=dict(hierarchical_model),
        target_support=dict(target_support),
        stacker=dict(stacker),
        controls=dict(controls),
        classifier=classifier,
        evaluation=dict(evaluation),
        runtime=dict(runtime),
        claim_boundary=dict(claim),
        contract_hash=stable_hash(scientific),
    )


def _input_path(source: Path, inputs: Mapping[str, object], key: str) -> Path:
    return resolve_config_path(
        source.parent,
        require_text(inputs[key], key.replace("_", " ")),
    )


__all__ = (
    "CLASSIFIER",
    "FixedBankHierarchicalResidualStackerConfig",
    "canonical_claim_boundary_payload",
    "canonical_controls_payload",
    "canonical_evaluation_payload",
    "canonical_features_payload",
    "canonical_hierarchical_model_payload",
    "canonical_probability_surface_payload",
    "canonical_protocol_payload",
    "canonical_runtime_payload",
    "canonical_stacker_payload",
    "canonical_target_support_payload",
    "load_fixed_bank_hierarchical_residual_stacker_config",
)
