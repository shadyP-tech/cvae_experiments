"""Strict non-authorizing SCALE-BP v1 configuration loader."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ...protocol import ProtocolError
from .config_payloads import (
    claim_boundary_payload,
    scientific_payloads,
    workstation_payload,
)
from .experiment_contracts import (
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from .hashing import canonical_hash, require_sha256
from .identity import (
    CLAIM_SCOPE,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    WORKSPACE_STATUS,
)
from .protocol import frozen_protocol_payload, validate_protocol_payload


LOCATION_KEYS = (
    "expert_bank_root",
    "generation_lock_root",
    "test_cache_root",
    "test_manifest_path",
    "test_consumption_ledger_path",
    "ledger_amendment_path",
)
CONFIG_TOP_LEVEL = frozenset(
    {
        "experiment",
        "inputs",
        "protocol",
        *scientific_payloads(),
        "runtime",
        "claim_boundary",
    }
)


def _fixed_experiment_payload() -> dict[str, object]:
    return {
        "id": EXPERIMENT_ID,
        "name": EXPERIMENT_NAME,
        "claim_scope": CLAIM_SCOPE,
        "publication_status": PUBLICATION_STATUS,
        "workspace_status": WORKSPACE_STATUS,
        "execution_authorized": False,
        "implementation_authorizes_execution": False,
        "consumed_test_reuse_authorized": False,
    }


def _fixed_inputs_payload() -> dict[str, object]:
    return {
        "expert_bank_artifact_id": EXPERT_BANK_ARTIFACT_ID,
        "generation_lock_artifact_id": GENERATION_LOCK_ARTIFACT_ID,
        "test_cache_artifact_id": TEST_CACHE_ARTIFACT_ID,
        "test_manifest_artifact_id": TEST_MANIFEST_ARTIFACT_ID,
        "test_consumption_ledger_artifact_id": TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
        "ledger_amendment_artifact_id": LEDGER_AMENDMENT_ARTIFACT_ID,
        "direct_input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "direct_input_count": 6,
        "all_direct_input_artifact_ids_unique": True,
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
        "expected_ledger_amendment_parent_sha256": (
            EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        ),
        "expected_ledger_amendment_sha256": EXPECTED_LEDGER_AMENDMENT_SHA256,
        "ledger_amendment_registered_consumer_experiment_id": EXPERIMENT_ID,
        "ledger_amendment_execution_authorized": False,
        "previous_stage90_outputs_used": False,
        "previous_stage90_amendments_used": False,
        "previous_stage90_scratch_or_checkpoints_used": False,
    }


def frozen_config_contract_payload() -> dict[str, object]:
    """Return the path-independent immutable v1 planning contract."""

    return {
        "experiment": _fixed_experiment_payload(),
        "inputs": _fixed_inputs_payload(),
        "protocol": frozen_protocol_payload(),
        **scientific_payloads(),
        "runtime": workstation_payload(),
        "claim_boundary": claim_boundary_payload(),
    }


@dataclass(frozen=True)
class ScaleBPConfig:
    """Resolved transport locations plus the frozen non-executable contract."""

    source_path: Path
    artifact_root: str
    expert_bank_root: str
    generation_lock_root: str
    test_cache_root: str
    test_manifest_path: str
    test_consumption_ledger_path: str
    ledger_amendment_path: str
    protocol: Mapping[str, object]
    action_geometry: Mapping[str, object]
    support_folds: Mapping[str, object]
    influence: Mapping[str, object]
    donor_prior: Mapping[str, object]
    local_residual: Mapping[str, object]
    empirical_bayes: Mapping[str, object]
    uncertainty: Mapping[str, object]
    selection: Mapping[str, object]
    admission: Mapping[str, object]
    controls: Mapping[str, object]
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]
    contract_hash: str

    experiment_id: str = EXPERIMENT_ID
    output_artifact_id: str = OUTPUT_ARTIFACT_ID
    input_artifact_ids: tuple[str, ...] = INPUT_ARTIFACT_IDS

    @property
    def config_hash(self) -> str:
        return self.contract_hash

    @property
    def execution_authorized(self) -> bool:
        return False


def load_support_calibrated_local_action_empirical_bayes_boundary_projected_router_config(
    path: str | Path,
) -> ScaleBPConfig:
    """Load exactly the registered plan without granting execution authority."""

    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read SCALE-BP v1 config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != set(CONFIG_TOP_LEVEL):
        raise ProtocolError("SCALE-BP top-level config drifted.")
    _reject_pending(raw)

    experiment = _section(raw, "experiment")
    expected_experiment = _fixed_experiment_payload()
    if set(experiment) != {*expected_experiment, "artifact_root"} or any(
        experiment.get(key) != value for key, value in expected_experiment.items()
    ):
        raise ProtocolError("SCALE-BP experiment identity drifted.")
    artifact_root = _nonempty_text(experiment.get("artifact_root"), "artifact root")
    if artifact_root.startswith("output://") and artifact_root != (
        f"output://{OUTPUT_ARTIFACT_ID}"
    ):
        raise ProtocolError("SCALE-BP output identity drifted.")

    inputs = _section(raw, "inputs")
    fixed_inputs = _fixed_inputs_payload()
    if set(inputs) != {*fixed_inputs, *LOCATION_KEYS} or any(
        inputs.get(key) != value for key, value in fixed_inputs.items()
    ):
        raise ProtocolError("SCALE-BP exact-six input contract drifted.")
    if len(INPUT_ARTIFACT_IDS) != 6 or len(set(INPUT_ARTIFACT_IDS)) != 6:
        raise ProtocolError("SCALE-BP exact-six input identities are not unique.")
    locations = {
        key: _nonempty_text(inputs.get(key), key) for key in LOCATION_KEYS
    }

    protocol = _section(raw, "protocol")
    validate_protocol_payload(protocol)
    for key, expected in scientific_payloads().items():
        if _section(raw, key) != expected:
            raise ProtocolError(f"SCALE-BP {key} contract drifted.")
    runtime = _section(raw, "runtime")
    claims = _section(raw, "claim_boundary")
    if runtime != workstation_payload():
        raise ProtocolError("SCALE-BP workstation contract drifted.")
    if claims != claim_boundary_payload():
        raise ProtocolError("SCALE-BP claim boundary drifted.")

    for role, digest in (
        ("test cache content hash", EXPECTED_TEST_CACHE_CONTENT_HASH),
        ("test cache row order hash", EXPECTED_TEST_CACHE_ROW_ORDER_HASH),
        ("manifest hash", EXPECTED_MANIFEST_SHA256),
        ("parent ledger hash", EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256),
        ("ledger amendment hash", EXPECTED_LEDGER_AMENDMENT_SHA256),
    ):
        require_sha256(digest, role)

    science = scientific_payloads()
    return ScaleBPConfig(
        source_path=source,
        artifact_root=artifact_root,
        expert_bank_root=locations["expert_bank_root"],
        generation_lock_root=locations["generation_lock_root"],
        test_cache_root=locations["test_cache_root"],
        test_manifest_path=locations["test_manifest_path"],
        test_consumption_ledger_path=locations["test_consumption_ledger_path"],
        ledger_amendment_path=locations["ledger_amendment_path"],
        protocol=protocol,
        action_geometry=science["action_geometry"],
        support_folds=science["support_folds"],
        influence=science["influence"],
        donor_prior=science["donor_prior"],
        local_residual=science["local_residual"],
        empirical_bayes=science["empirical_bayes"],
        uncertainty=science["uncertainty"],
        selection=science["selection"],
        admission=science["admission"],
        controls=science["controls"],
        runtime=runtime,
        claim_boundary=claims,
        contract_hash=canonical_hash(frozen_config_contract_payload()),
    )


def _section(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"SCALE-BP {name} section is not a mapping.")
    return dict(value)


def _nonempty_text(value: object, role: str) -> str:
    text = str(value) if value is not None else ""
    if not text:
        raise ProtocolError(f"SCALE-BP {role} is empty.")
    return text


def _reject_pending(value: object) -> None:
    if isinstance(value, str) and "__PENDING" in value:
        raise ProtocolError("SCALE-BP config contains an unresolved placeholder.")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_pending(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_pending(item)


__all__ = (
    "CONFIG_TOP_LEVEL",
    "ScaleBPConfig",
    "frozen_config_contract_payload",
    "load_support_calibrated_local_action_empirical_bayes_boundary_projected_router_config",
)
