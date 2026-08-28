"""Strict path-independent configuration for executable SCEPTRE v4."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...protocol import ProtocolError
from .experiment_contracts import (
    AUTHORIZED_INPUT_ROLES,
    EXECUTION_AMENDMENT_ARTIFACT_ID,
    EXECUTION_AMENDMENT_FILENAME,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SOURCE_INNER_AMENDMENT_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    EXPERT_BANK_ARTIFACT_ID,
    FORBIDDEN_INPUT_FRAGMENTS,
    GENERATION_LOCK_ARTIFACT_ID,
    INPUT_ARTIFACT_IDS,
    SOURCE_INNER_ALIAS_ARTIFACT_ID,
    SOURCE_INNER_AMENDMENT_ARTIFACT_ID,
    SOURCE_INNER_AMENDMENT_FILENAME,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from .identity import (
    AUTHORIZATION_BASIS,
    AUTHORIZATION_DATE,
    AUTHORIZATION_SCOPE,
    CLAIM_SCOPE,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    canonical_hash,
    require_sha256,
)
from .protocol import claim_boundary_payload, protocol_payload
from .source_seal import source_snapshot_identity
from .execution.workstation import validate_workstation_payload, workstation_payload


CONFIG_TOP_LEVEL = frozenset(
    {
        "experiment",
        "inputs",
        "protocol",
        "classifier",
        "source_provenance",
        "runtime",
        "claim_boundary",
    }
)

CLASSIFIER = ClassifierSpec(
    C=0.01,
    penalty="l2",
    solver="lbfgs",
    max_iter=3000,
    class_weight=None,
    random_state=23,
    l1_ratio=None,
    threshold_policy="predict",
    scaler_fit="synthetic_train_only",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def experiment_payload() -> dict[str, object]:
    return {
        "schema_version": "sceptre_v4_executable_config_v1",
        "id": EXPERIMENT_ID,
        "name": EXPERIMENT_NAME,
        "stage": "90_oracles_and_diagnostics",
        "status": "diagnostic",
        "claim_scope": CLAIM_SCOPE,
        "publication_status": PUBLICATION_STATUS,
        "fresh_evidence": False,
        "execution_authorized": True,
        "execution_authorization_basis": AUTHORIZATION_BASIS,
        "execution_authorization_scope": AUTHORIZATION_SCOPE,
        "authorization_date": AUTHORIZATION_DATE,
        "implementation_authorizes_execution": False,
        "single_use_execution_identity": True,
        "authorization_exhausted": False,
        "consumed_test_reuse_authorized": True,
    }


def input_policy_payload(
    *,
    execution_amendment_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": "sceptre_v4_exact_eight_input_policy_v1",
        "direct_input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "direct_input_count": 8,
        "authorized_input_roles": list(AUTHORIZED_INPUT_ROLES),
        "expected_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
        "expected_generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
        "expected_source_inner_amendment_sha256": (
            EXPECTED_SOURCE_INNER_AMENDMENT_SHA256
        ),
        "expected_test_cache_semantic_id": EXPECTED_TEST_CACHE_SEMANTIC_ID,
        "expected_test_cache_representation_id": (
            EXPECTED_TEST_CACHE_REPRESENTATION_ID
        ),
        "expected_test_cache_content_hash": EXPECTED_TEST_CACHE_CONTENT_HASH,
        "expected_test_cache_row_order_hash": EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
        "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "expected_parent_ledger_sha256": (
            EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        ),
        "expected_execution_amendment_sha256": execution_amendment_sha256,
        "execution_amendment_consumer_experiment_id": EXPERIMENT_ID,
        "previous_stage90_output_or_state_used": False,
        "cross_run_recovery_allowed": False,
    }


def source_provenance_payload() -> dict[str, object]:
    return {
        "schema_version": "sceptre_v4_source_provenance_v1",
        **dict(source_snapshot_identity()),
        "recompute_and_exact_match_on_load": True,
        "sceptre_owned_source_closure_sealed": True,
        "shared_runtime_dependencies_in_source_seal": True,
    }


@dataclass(frozen=True, slots=True)
class SceptreV4Config:
    source_path: Path | None
    artifact_root: Path
    expert_bank_root: Path
    generation_lock_root: Path
    source_inner_root: Path
    source_inner_amendment_path: Path
    test_cache_root: Path
    test_manifest_path: Path
    test_consumption_ledger_path: Path
    execution_amendment_path: Path
    classifier: ClassifierSpec
    protocol: Mapping[str, object]
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]
    source_provenance: Mapping[str, object]
    contract_hash: str
    expected_execution_amendment_sha256: str
    experiment_id: str = EXPERIMENT_ID
    output_artifact_id: str = OUTPUT_ARTIFACT_ID
    input_artifact_ids: tuple[str, ...] = INPUT_ARTIFACT_IDS
    expected_bank_lock_hash: str = EXPECTED_BANK_LOCK_HASH
    expected_generation_lock_hash: str = EXPECTED_GENERATION_LOCK_HASH
    expected_source_inner_amendment_sha256: str = (
        EXPECTED_SOURCE_INNER_AMENDMENT_SHA256
    )
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
    authorization_basis: str = AUTHORIZATION_BASIS
    authorization_scope: str = AUTHORIZATION_SCOPE

    @property
    def config_hash(self) -> str:
        return self.contract_hash

    @property
    def execution_authorized(self) -> bool:
        try:
            require_sha256(
                self.expected_execution_amendment_sha256,
                "execution amendment hash",
            )
        except ProtocolError:
            return False
        return True

    @property
    def expected_source_snapshot_manifest_sha256(self) -> str:
        return str(self.source_provenance["source_snapshot_manifest_sha256"])

    @property
    def expected_source_snapshot_tree_sha256(self) -> str:
        return str(self.source_provenance["source_snapshot_tree_sha256"])

    @property
    def expected_source_snapshot_member_count(self) -> int:
        return int(self.source_provenance["source_snapshot_member_count"])


SceptreConfig = SceptreV4Config


def load_config(path: str | Path) -> SceptreV4Config:
    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read SCEPTRE v4 config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != set(CONFIG_TOP_LEVEL):
        raise ProtocolError("SCEPTRE v4 top-level config drifted.")
    _reject_pending(raw)

    experiment = _section(raw, "experiment")
    expected_experiment = experiment_payload()
    if set(experiment) != {*expected_experiment, "artifact_root"} or any(
        experiment.get(key) != value for key, value in expected_experiment.items()
    ):
        raise ProtocolError("SCEPTRE v4 experiment identity drifted.")
    artifact_root_text = str(experiment["artifact_root"])
    if artifact_root_text.startswith("output://") and artifact_root_text != (
        f"output://{OUTPUT_ARTIFACT_ID}"
    ):
        raise ProtocolError("SCEPTRE v4 output identity drifted.")

    inputs = _section(raw, "inputs")
    amendment_hash = require_sha256(
        inputs.get("expected_execution_amendment_sha256"),
        "execution amendment hash",
    )
    fixed_inputs = input_policy_payload(execution_amendment_sha256=amendment_hash)
    locations = {
        "expert_bank_root": (EXPERT_BANK_ARTIFACT_ID, ""),
        "generation_lock_root": (GENERATION_LOCK_ARTIFACT_ID, ""),
        "source_inner_root": (SOURCE_INNER_ALIAS_ARTIFACT_ID, ""),
        "source_inner_amendment_path": (
            SOURCE_INNER_AMENDMENT_ARTIFACT_ID,
            SOURCE_INNER_AMENDMENT_FILENAME,
        ),
        "test_cache_root": (TEST_CACHE_ARTIFACT_ID, ""),
        "test_manifest_path": (TEST_MANIFEST_ARTIFACT_ID, "manifest.csv"),
        "test_consumption_ledger_path": (
            TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
            "reports/test_consumption_ledger.json",
        ),
        "execution_amendment_path": (
            EXECUTION_AMENDMENT_ARTIFACT_ID,
            EXECUTION_AMENDMENT_FILENAME,
        ),
    }
    if set(inputs) != set(fixed_inputs) | set(locations) or any(
        inputs.get(key) != value for key, value in fixed_inputs.items()
    ):
        raise ProtocolError("SCEPTRE v4 exact-eight input schema drifted.")
    for key, (artifact_id, member) in locations.items():
        value = str(inputs[key])
        expected = f"artifact://{artifact_id}" + (f"/{member}" if member else "")
        if value.startswith("artifact://") and value != expected:
            raise ProtocolError(f"SCEPTRE v4 artifact URI drifted: {key}.")
        if any(fragment.casefold() in value.casefold() for fragment in FORBIDDEN_INPUT_FRAGMENTS):
            raise ProtocolError(f"SCEPTRE v4 predecessor input detected: {key}.")

    expected_protocol = protocol_payload()
    expected_runtime = workstation_payload()
    expected_claim = claim_boundary_payload()
    expected_source = source_provenance_payload()
    if _section(raw, "protocol") != expected_protocol:
        raise ProtocolError("SCEPTRE v4 protocol drifted.")
    if _section(raw, "runtime") != expected_runtime:
        raise ProtocolError("SCEPTRE v4 runtime drifted.")
    if _section(raw, "claim_boundary") != expected_claim:
        raise ProtocolError("SCEPTRE v4 claim boundary drifted.")
    if _section(raw, "source_provenance") != expected_source:
        raise ProtocolError("SCEPTRE v4 scientific source seal drifted.")
    classifier = _classifier(_section(raw, "classifier"))
    if classifier != CLASSIFIER:
        raise ProtocolError("SCEPTRE v4 classifier drifted.")
    validate_workstation_payload(expected_runtime)

    scientific_contract = {
        "schema_version": "sceptre_v4_path_independent_config_v1",
        "experiment": expected_experiment,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "input_content_identities": fixed_inputs,
        "classifier": classifier.to_payload(),
        "protocol": expected_protocol,
        "runtime": expected_runtime,
        "claim_boundary": expected_claim,
        "source_provenance": expected_source,
    }
    resolved = {
        key: _resolve(source.parent, str(inputs[key])) for key in locations
    }
    return SceptreV4Config(
        source_path=source,
        artifact_root=_resolve(source.parent, artifact_root_text),
        classifier=classifier,
        protocol=expected_protocol,
        runtime=expected_runtime,
        claim_boundary=expected_claim,
        source_provenance=expected_source,
        contract_hash=canonical_hash(scientific_contract),
        expected_execution_amendment_sha256=amendment_hash,
        **resolved,
    )


def _section(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"SCEPTRE v4 {name} section is not a mapping.")
    return dict(value)


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
        raise ProtocolError("SCEPTRE v4 classifier payload malformed.") from exc


def _reject_pending(value: object) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _reject_pending(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_pending(nested)
    elif isinstance(value, str) and any(
        token in value
        for token in ("pending://", "PENDING", "TO_BE_RECOMPUTED", "__PENDING_")
    ):
        raise ProtocolError("SCEPTRE v4 config contains a pending value.")


__all__ = (
    "CLASSIFIER",
    "CONFIG_TOP_LEVEL",
    "SceptreConfig",
    "SceptreV4Config",
    "experiment_payload",
    "input_policy_payload",
    "load_config",
    "repository_root",
    "source_provenance_payload",
)
