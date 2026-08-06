"""Configuration for Stage-70 reservation and final prediction authorization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ....common.hashing import stable_hash
from ....workspace.runtime import MidogppWorkspace
from ...protocol import ProtocolError
from ..contracts import AUTHORIZED_CONSUMER_EXPERIMENT_ID
from .contracts import (
    CLAIM_SCOPE,
    EXPECTED_TEST_ROWS,
    FINAL_AUTHORIZATION_EXPERIMENT_ID,
    FINAL_AUTHORIZATION_EXPERIMENT_NAME,
    FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID,
    POLICY_ARMS,
    PURPOSE,
    RESERVATION_EXPERIMENT_ID,
    RESERVATION_EXPERIMENT_NAME,
    RESERVATION_OUTPUT_ARTIFACT_ID,
)


CACHE_EXPERIMENT_ID = (
    "midogpp.frozen_policy_downstream.uniform_b_v2_descriptive_test_cache.v1"
)
CACHE_ARTIFACT_ID = (
    "midogpp_virchow2_uniform_b_v2_descriptive_test_cache_seed42"
)
CANONICAL_CACHE_RELATIVE_ROOT = (
    "datasets/midogpp/derived/features/virchow2/"
    "uniform_b_v2_descriptive_test_cache_v1/seed42"
)


@dataclass(frozen=True)
class ReservationConfig:
    artifact_root: Path
    canonical_reference_root: Path
    bank_root: Path
    generation_lock_root: Path
    equal_union_policy_root: Path
    metadata_policy_root: Path
    utility_policy_root: Path
    scoring_manifest_path: Path
    test_consumption_ledger_path: Path
    prospective_cache_root: Path
    expected_scoring_manifest_sha256: str
    expected_cache_extractor_protocol_hash: str
    experiment_id: str = RESERVATION_EXPERIMENT_ID
    name: str = RESERVATION_EXPERIMENT_NAME
    output_artifact_id: str = RESERVATION_OUTPUT_ARTIFACT_ID
    cache_experiment_id: str = CACHE_EXPERIMENT_ID
    cache_artifact_id: str = CACHE_ARTIFACT_ID
    consumer_experiment_id: str = AUTHORIZED_CONSUMER_EXPERIMENT_ID
    purpose: str = PURPOSE
    claim_scope: str = CLAIM_SCOPE
    expected_test_rows: int = EXPECTED_TEST_ROWS
    production_workspace_binding: bool = True
    allow_test_validation_injection: bool = False

    @property
    def contract_hash(self) -> str:
        return stable_hash(
            {
                "experiment_id": self.experiment_id,
                "output_artifact_id": self.output_artifact_id,
                "cache_experiment_id": self.cache_experiment_id,
                "cache_artifact_id": self.cache_artifact_id,
                "consumer_experiment_id": self.consumer_experiment_id,
                "purpose": self.purpose,
                "claim_scope": self.claim_scope,
                "expected_test_rows": self.expected_test_rows,
                "expected_scoring_manifest_sha256": (
                    self.expected_scoring_manifest_sha256
                ),
                "expected_cache_extractor_protocol_hash": (
                    self.expected_cache_extractor_protocol_hash
                ),
                "policy_arms": list(POLICY_ARMS),
            }
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "experiment": {
                "id": self.experiment_id,
                "name": self.name,
                "artifact_root": str(self.artifact_root),
                "output_artifact_id": self.output_artifact_id,
            },
            "inputs": {
                "canonical_reference_root": str(self.canonical_reference_root),
                "bank_root": str(self.bank_root),
                "generation_lock_root": str(self.generation_lock_root),
                "equal_union_policy_root": str(self.equal_union_policy_root),
                "metadata_policy_root": str(self.metadata_policy_root),
                "utility_policy_root": str(self.utility_policy_root),
                "scoring_manifest_path": str(self.scoring_manifest_path),
                "test_consumption_ledger_path": str(
                    self.test_consumption_ledger_path
                ),
                "prospective_cache_root": (
                    CANONICAL_CACHE_RELATIVE_ROOT
                    if self.production_workspace_binding is True
                    else str(self.prospective_cache_root)
                ),
                "cache_experiment_id": self.cache_experiment_id,
                "cache_artifact_id": self.cache_artifact_id,
            },
            "protocol": {
                "consumer_experiment_id": self.consumer_experiment_id,
                "purpose": self.purpose,
                "claim_scope": self.claim_scope,
                "expected_test_rows": self.expected_test_rows,
                "expected_scoring_manifest_sha256": (
                    self.expected_scoring_manifest_sha256
                ),
                "expected_cache_extractor_protocol_hash": (
                    self.expected_cache_extractor_protocol_hash
                ),
                "policy_arms": list(POLICY_ARMS),
            },
            "execution": {
                "production_workspace_binding": self.production_workspace_binding,
                "allow_test_validation_injection": (
                    self.allow_test_validation_injection
                ),
                "generation_allowed": False,
                "prediction_allowed": False,
                "label_access_allowed": False,
                "metric_scoring_allowed": False,
            },
        }


@dataclass(frozen=True)
class FinalAuthorizationConfig:
    artifact_root: Path
    reservation_root: Path
    cache_root: Path
    canonical_reference_root: Path
    bank_root: Path
    generation_lock_root: Path
    equal_union_policy_root: Path
    metadata_policy_root: Path
    utility_policy_root: Path
    scoring_manifest_path: Path
    expected_scoring_manifest_sha256: str
    expected_cache_extractor_protocol_hash: str
    experiment_id: str = FINAL_AUTHORIZATION_EXPERIMENT_ID
    name: str = FINAL_AUTHORIZATION_EXPERIMENT_NAME
    output_artifact_id: str = FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID
    reservation_artifact_id: str = RESERVATION_OUTPUT_ARTIFACT_ID
    cache_experiment_id: str = CACHE_EXPERIMENT_ID
    cache_artifact_id: str = CACHE_ARTIFACT_ID
    consumer_experiment_id: str = AUTHORIZED_CONSUMER_EXPERIMENT_ID
    purpose: str = PURPOSE
    claim_scope: str = CLAIM_SCOPE
    expected_test_rows: int = EXPECTED_TEST_ROWS
    production_workspace_binding: bool = True
    allow_test_validation_injection: bool = False

    @property
    def contract_hash(self) -> str:
        return stable_hash(
            {
                "experiment_id": self.experiment_id,
                "output_artifact_id": self.output_artifact_id,
                "reservation_artifact_id": self.reservation_artifact_id,
                "cache_experiment_id": self.cache_experiment_id,
                "cache_artifact_id": self.cache_artifact_id,
                "consumer_experiment_id": self.consumer_experiment_id,
                "purpose": self.purpose,
                "claim_scope": self.claim_scope,
                "expected_test_rows": self.expected_test_rows,
                "expected_scoring_manifest_sha256": (
                    self.expected_scoring_manifest_sha256
                ),
                "expected_cache_extractor_protocol_hash": (
                    self.expected_cache_extractor_protocol_hash
                ),
                "policy_arms": list(POLICY_ARMS),
            }
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "experiment": {
                "id": self.experiment_id,
                "name": self.name,
                "artifact_root": str(self.artifact_root),
                "output_artifact_id": self.output_artifact_id,
            },
            "inputs": {
                "reservation_root": str(self.reservation_root),
                "reservation_artifact_id": self.reservation_artifact_id,
                "cache_root": str(self.cache_root),
                "cache_experiment_id": self.cache_experiment_id,
                "cache_artifact_id": self.cache_artifact_id,
                "canonical_reference_root": str(self.canonical_reference_root),
                "bank_root": str(self.bank_root),
                "generation_lock_root": str(self.generation_lock_root),
                "equal_union_policy_root": str(self.equal_union_policy_root),
                "metadata_policy_root": str(self.metadata_policy_root),
                "utility_policy_root": str(self.utility_policy_root),
                "scoring_manifest_path": str(self.scoring_manifest_path),
            },
            "protocol": {
                "consumer_experiment_id": self.consumer_experiment_id,
                "purpose": self.purpose,
                "claim_scope": self.claim_scope,
                "expected_test_rows": self.expected_test_rows,
                "expected_scoring_manifest_sha256": (
                    self.expected_scoring_manifest_sha256
                ),
                "expected_cache_extractor_protocol_hash": (
                    self.expected_cache_extractor_protocol_hash
                ),
                "policy_arms": list(POLICY_ARMS),
            },
            "execution": {
                "production_workspace_binding": self.production_workspace_binding,
                "allow_test_validation_injection": (
                    self.allow_test_validation_injection
                ),
                "generation_allowed": False,
                "prediction_allowed": False,
                "label_access_allowed": False,
                "metric_scoring_allowed": False,
                "authorization_output": "prediction_only_token",
            },
        }


def load_reservation_config(path: str | Path) -> ReservationConfig:
    payload, base = _load_yaml(path)
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    protocol = _mapping(payload, "protocol")
    execution = _mapping(payload, "execution")
    config = ReservationConfig(
        artifact_root=_path(base, experiment.get("artifact_root")),
        canonical_reference_root=_path(base, inputs.get("canonical_reference_root")),
        bank_root=_path(base, inputs.get("bank_root")),
        generation_lock_root=_path(base, inputs.get("generation_lock_root")),
        equal_union_policy_root=_path(base, inputs.get("equal_union_policy_root")),
        metadata_policy_root=_path(base, inputs.get("metadata_policy_root")),
        utility_policy_root=_path(base, inputs.get("utility_policy_root")),
        scoring_manifest_path=_path(base, inputs.get("scoring_manifest_path")),
        test_consumption_ledger_path=_path(
            base, inputs.get("test_consumption_ledger_path")
        ),
        prospective_cache_root=_canonical_cache_path(
            inputs.get("prospective_cache_root")
        ),
        expected_scoring_manifest_sha256=str(
            protocol.get("expected_scoring_manifest_sha256", "")
        ),
        expected_cache_extractor_protocol_hash=str(
            protocol.get("expected_cache_extractor_protocol_hash", "")
        ),
        experiment_id=str(experiment.get("id", "")),
        name=str(experiment.get("name", "")),
        output_artifact_id=str(experiment.get("output_artifact_id", "")),
        cache_experiment_id=str(inputs.get("cache_experiment_id", "")),
        cache_artifact_id=str(inputs.get("cache_artifact_id", "")),
        consumer_experiment_id=str(protocol.get("consumer_experiment_id", "")),
        purpose=str(protocol.get("purpose", "")),
        claim_scope=str(protocol.get("claim_scope", "")),
        expected_test_rows=int(protocol.get("expected_test_rows", -1)),
        production_workspace_binding=bool(
            execution.get("production_workspace_binding", False)
        ),
        allow_test_validation_injection=bool(
            execution.get("allow_test_validation_injection", False)
        ),
    )
    _validate_reservation_config(config, payload=payload)
    return config


def load_final_authorization_config(path: str | Path) -> FinalAuthorizationConfig:
    payload, base = _load_yaml(path)
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    protocol = _mapping(payload, "protocol")
    execution = _mapping(payload, "execution")
    config = FinalAuthorizationConfig(
        artifact_root=_path(base, experiment.get("artifact_root")),
        reservation_root=_path(base, inputs.get("reservation_root")),
        cache_root=_path(base, inputs.get("cache_root")),
        canonical_reference_root=_path(base, inputs.get("canonical_reference_root")),
        bank_root=_path(base, inputs.get("bank_root")),
        generation_lock_root=_path(base, inputs.get("generation_lock_root")),
        equal_union_policy_root=_path(base, inputs.get("equal_union_policy_root")),
        metadata_policy_root=_path(base, inputs.get("metadata_policy_root")),
        utility_policy_root=_path(base, inputs.get("utility_policy_root")),
        scoring_manifest_path=_path(base, inputs.get("scoring_manifest_path")),
        expected_scoring_manifest_sha256=str(
            protocol.get("expected_scoring_manifest_sha256", "")
        ),
        expected_cache_extractor_protocol_hash=str(
            protocol.get("expected_cache_extractor_protocol_hash", "")
        ),
        experiment_id=str(experiment.get("id", "")),
        name=str(experiment.get("name", "")),
        output_artifact_id=str(experiment.get("output_artifact_id", "")),
        reservation_artifact_id=str(inputs.get("reservation_artifact_id", "")),
        cache_experiment_id=str(inputs.get("cache_experiment_id", "")),
        cache_artifact_id=str(inputs.get("cache_artifact_id", "")),
        consumer_experiment_id=str(protocol.get("consumer_experiment_id", "")),
        purpose=str(protocol.get("purpose", "")),
        claim_scope=str(protocol.get("claim_scope", "")),
        expected_test_rows=int(protocol.get("expected_test_rows", -1)),
        production_workspace_binding=bool(
            execution.get("production_workspace_binding", False)
        ),
        allow_test_validation_injection=bool(
            execution.get("allow_test_validation_injection", False)
        ),
    )
    _validate_final_config(config, payload=payload)
    return config


def validate_reservation_config(config: ReservationConfig) -> None:
    _validate_reservation_config(config)


def validate_final_authorization_config(config: FinalAuthorizationConfig) -> None:
    _validate_final_config(config)


def _validate_reservation_config(
    config: ReservationConfig,
    *,
    payload: Mapping[str, object] | None = None,
) -> None:
    _validate_common(
        experiment_id=config.experiment_id,
        expected_experiment_id=RESERVATION_EXPERIMENT_ID,
        name=config.name,
        expected_name=RESERVATION_EXPERIMENT_NAME,
        output_artifact_id=config.output_artifact_id,
        expected_output_artifact_id=RESERVATION_OUTPUT_ARTIFACT_ID,
        cache_experiment_id=config.cache_experiment_id,
        cache_artifact_id=config.cache_artifact_id,
        consumer_experiment_id=config.consumer_experiment_id,
        purpose=config.purpose,
        claim_scope=config.claim_scope,
        expected_test_rows=config.expected_test_rows,
        manifest_sha256=config.expected_scoring_manifest_sha256,
        extractor_hash=config.expected_cache_extractor_protocol_hash,
    )
    if payload is not None:
        _validate_loaded_execution(config.production_workspace_binding, config.allow_test_validation_injection)
        _validate_payload_shape(payload, final=False)


def _validate_final_config(
    config: FinalAuthorizationConfig,
    *,
    payload: Mapping[str, object] | None = None,
) -> None:
    _validate_common(
        experiment_id=config.experiment_id,
        expected_experiment_id=FINAL_AUTHORIZATION_EXPERIMENT_ID,
        name=config.name,
        expected_name=FINAL_AUTHORIZATION_EXPERIMENT_NAME,
        output_artifact_id=config.output_artifact_id,
        expected_output_artifact_id=FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID,
        cache_experiment_id=config.cache_experiment_id,
        cache_artifact_id=config.cache_artifact_id,
        consumer_experiment_id=config.consumer_experiment_id,
        purpose=config.purpose,
        claim_scope=config.claim_scope,
        expected_test_rows=config.expected_test_rows,
        manifest_sha256=config.expected_scoring_manifest_sha256,
        extractor_hash=config.expected_cache_extractor_protocol_hash,
    )
    if config.reservation_artifact_id != RESERVATION_OUTPUT_ARTIFACT_ID:
        raise ProtocolError("Stage-70 final authorization reservation identity drifted.")
    if payload is not None:
        _validate_loaded_execution(config.production_workspace_binding, config.allow_test_validation_injection)
        _validate_payload_shape(payload, final=True)


def _validate_common(
    *,
    experiment_id: str,
    expected_experiment_id: str,
    name: str,
    expected_name: str,
    output_artifact_id: str,
    expected_output_artifact_id: str,
    cache_experiment_id: str,
    cache_artifact_id: str,
    consumer_experiment_id: str,
    purpose: str,
    claim_scope: str,
    expected_test_rows: int,
    manifest_sha256: str,
    extractor_hash: str,
) -> None:
    exact = {
        "experiment_id": (experiment_id, expected_experiment_id),
        "name": (name, expected_name),
        "output_artifact_id": (output_artifact_id, expected_output_artifact_id),
        "cache_experiment_id": (cache_experiment_id, CACHE_EXPERIMENT_ID),
        "cache_artifact_id": (cache_artifact_id, CACHE_ARTIFACT_ID),
        "consumer_experiment_id": (
            consumer_experiment_id,
            AUTHORIZED_CONSUMER_EXPERIMENT_ID,
        ),
        "purpose": (purpose, PURPOSE),
        "claim_scope": (claim_scope, CLAIM_SCOPE),
        "expected_test_rows": (expected_test_rows, EXPECTED_TEST_ROWS),
    }
    mismatch = [key for key, pair in exact.items() if pair[0] != pair[1]]
    if mismatch:
        raise ProtocolError(f"Stage-70 authorization identity drifted: {mismatch}.")
    if len(manifest_sha256) != 64 or not _lower_hex(manifest_sha256):
        raise ProtocolError("Stage-70 expected scoring-manifest SHA-256 is malformed.")
    if not _lower_hex(extractor_hash):
        raise ProtocolError("Stage-70 cache-extractor protocol hash is malformed.")


def _validate_loaded_execution(
    production_workspace_binding: bool,
    allow_test_validation_injection: bool,
) -> None:
    if production_workspace_binding is not True or allow_test_validation_injection is not False:
        raise ProtocolError(
            "Loaded Stage-70 configs must require production workspace binding and "
            "must disable test injection."
        )


def _validate_payload_shape(payload: Mapping[str, object], *, final: bool) -> None:
    if set(payload) != {"experiment", "inputs", "protocol", "execution"}:
        raise ProtocolError("Stage-70 authorization config top-level keys drifted.")
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    protocol = _mapping(payload, "protocol")
    execution = _mapping(payload, "execution")
    if set(experiment) != {"id", "name", "artifact_root", "output_artifact_id"}:
        raise ProtocolError("Stage-70 authorization experiment config keys drifted.")
    common_inputs = {
        "canonical_reference_root",
        "bank_root",
        "generation_lock_root",
        "equal_union_policy_root",
        "metadata_policy_root",
        "utility_policy_root",
        "scoring_manifest_path",
        "cache_experiment_id",
        "cache_artifact_id",
    }
    expected_inputs = (
        common_inputs | {"reservation_root", "reservation_artifact_id", "cache_root"}
        if final
        else common_inputs
        | {"prospective_cache_root", "test_consumption_ledger_path"}
    )
    if set(inputs) != expected_inputs:
        raise ProtocolError("Stage-70 authorization input config keys drifted.")
    if set(protocol) != {
        "consumer_experiment_id",
        "purpose",
        "claim_scope",
        "expected_test_rows",
        "expected_scoring_manifest_sha256",
        "expected_cache_extractor_protocol_hash",
        "policy_arms",
    } or protocol.get("policy_arms") != list(POLICY_ARMS):
        raise ProtocolError("Stage-70 authorization protocol config drifted.")
    expected_execution = {
        "production_workspace_binding",
        "allow_test_validation_injection",
        "generation_allowed",
        "prediction_allowed",
        "label_access_allowed",
        "metric_scoring_allowed",
    }
    if final:
        expected_execution.add("authorization_output")
    if set(execution) != expected_execution:
        raise ProtocolError("Stage-70 authorization execution config keys drifted.")
    required_execution: dict[str, object] = {
        "production_workspace_binding": True,
        "allow_test_validation_injection": False,
        "generation_allowed": False,
        "prediction_allowed": False,
        "label_access_allowed": False,
        "metric_scoring_allowed": False,
    }
    if final:
        required_execution["authorization_output"] = "prediction_only_token"
    if any(execution.get(key) != value for key, value in required_execution.items()):
        raise ProtocolError("Stage-70 authorization execution values drifted.")


def _load_yaml(path: str | Path) -> tuple[Mapping[str, object], Path]:
    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError(f"Cannot read Stage-70 authorization config: {path}.") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("Stage-70 authorization config must be a mapping.")
    return payload, config_path.parent


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Stage-70 config section {key!r} must be a mapping.")
    return value


def _path(base: Path, value: object) -> Path:
    rendered = str(value or "")
    if not rendered:
        raise ProtocolError("Stage-70 config path is empty.")
    if rendered.startswith(("artifact://", "output://")):
        return Path(rendered)
    path = Path(rendered)
    return path if path.is_absolute() else (base / path).resolve()


def _canonical_cache_path(value: object) -> Path:
    rendered = str(value or "")
    if rendered != CANONICAL_CACHE_RELATIVE_ROOT:
        raise ProtocolError(
            "Stage-70 reservation prospective cache root must be the canonical "
            "repository-relative derived-feature path."
        )
    return (MidogppWorkspace.load().repo_root / rendered).resolve()


def _lower_hex(value: str) -> bool:
    return bool(value) and value == value.lower() and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = (
    "CACHE_ARTIFACT_ID",
    "CACHE_EXPERIMENT_ID",
    "CANONICAL_CACHE_RELATIVE_ROOT",
    "FinalAuthorizationConfig",
    "ReservationConfig",
    "load_final_authorization_config",
    "load_reservation_config",
    "validate_final_authorization_config",
    "validate_reservation_config",
)
