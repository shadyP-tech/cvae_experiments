"""Strict, path-free planned configuration for SCEPTRE v1."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from midogpp_thesis.cvae.protocol import ProtocolError

from .experiment_contracts import claim_boundary_payload, direct_input_policy_payload
from .hashing import canonical_hash
from .identity import (
    AMENDMENT_RELATIVE_PATH,
    CLAIM_SCOPE,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    WORKSPACE_STATUS,
)
from .protocol import frozen_protocol_payload, validate_protocol_payload
from .source_fence import build_science_source_receipt
from .source_inner_authorization import load_reuse_amendment
from .workstation import validate_workstation_payload, workstation_payload


CONFIG_TOP_LEVEL = frozenset(
    {
        "experiment",
        "inputs",
        "protocol",
        "source_provenance",
        "runtime",
        "claim_boundary",
    }
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def experiment_payload() -> dict[str, object]:
    return {
        "schema_version": "sceptre_v1_planned_config_v1",
        "id": EXPERIMENT_ID,
        "name": EXPERIMENT_NAME,
        "stage": "90_oracles_and_diagnostics",
        "status": WORKSPACE_STATUS,
        "claim_scope": CLAIM_SCOPE,
        "publication_status": PUBLICATION_STATUS,
        "fresh_evidence": False,
        "adaptive_development_surface": True,
        "historical_source_inner_adaptive_reuse_authorized": True,
        "execution_authorized": False,
        "implementation_authorizes_execution": False,
        "consumed_test_reuse_authorized": False,
    }


def source_provenance_payload() -> dict[str, object]:
    receipt = build_science_source_receipt()
    return {
        "schema_version": "sceptre_v1_science_source_provenance_v1",
        "scientific_member_count": receipt.member_count,
        "label_free_core_member_count": receipt.core_member_count,
        "diagnostic_development_member_count": receipt.development_member_count,
        "scientific_source_tree_sha256": receipt.tree_sha256,
        "scientific_source_receipt_hash": receipt.receipt_hash,
        "label_free_core_import_fence_validated": True,
        "label_free_core_may_import_source_inner_utility": False,
        "development_adapter_is_separate": True,
        "recompute_and_exact_match_on_load": True,
    }


def _config_payload(source_provenance: Mapping[str, object]) -> dict[str, object]:
    return {
        "experiment": {
            **experiment_payload(),
            "artifact_root": f"output://{OUTPUT_ARTIFACT_ID}",
        },
        "inputs": direct_input_policy_payload(),
        "protocol": frozen_protocol_payload(),
        "source_provenance": dict(source_provenance),
        "runtime": workstation_payload(),
        "claim_boundary": claim_boundary_payload(),
    }


def frozen_config_contract_payload() -> dict[str, object]:
    return _config_payload(source_provenance_payload())


@dataclass(frozen=True, slots=True)
class SceptreConfig:
    source_path: Path | None
    artifact_root: str
    contract_hash: str
    source_provenance_items: tuple[tuple[str, object], ...]
    experiment_id: str = EXPERIMENT_ID
    output_artifact_id: str = OUTPUT_ARTIFACT_ID
    input_artifact_ids: tuple[str, ...] = INPUT_ARTIFACT_IDS
    execution_authorized: bool = False

    @property
    def protocol(self) -> dict[str, object]:
        return frozen_protocol_payload()

    @property
    def runtime(self) -> dict[str, object]:
        return workstation_payload()

    @property
    def claim_boundary(self) -> dict[str, object]:
        return claim_boundary_payload()

    @property
    def inputs(self) -> dict[str, object]:
        return direct_input_policy_payload()

    @property
    def source_provenance(self) -> dict[str, object]:
        return dict(self.source_provenance_items)


def build_planned_config() -> SceptreConfig:
    payload = frozen_config_contract_payload()
    return SceptreConfig(
        source_path=None,
        artifact_root=str(payload["experiment"]["artifact_root"]),
        contract_hash=canonical_hash(payload),
        source_provenance_items=tuple(payload["source_provenance"].items()),
    )


def load_config(path: str | Path) -> SceptreConfig:
    """Load the exact planned contract and verify source/amendment bytes."""

    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read SCEPTRE v1 config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != set(CONFIG_TOP_LEVEL):
        raise ProtocolError("SCEPTRE top-level config drifted.")
    _reject_pending(raw)
    # The amendment is repository-owned governance, not a runtime artifact
    # path supplied by the config. It is checked before any output resolution.
    load_reuse_amendment(repository_root() / AMENDMENT_RELATIVE_PATH)
    observed_provenance = source_provenance_payload()
    if _section(raw, "source_provenance") != observed_provenance:
        raise ProtocolError("SCEPTRE scientific source seal drifted.")
    expected = _config_payload(observed_provenance)
    if dict(raw) != expected:
        raise ProtocolError("SCEPTRE planned config contract drifted.")
    validate_protocol_payload(_section(raw, "protocol"))
    validate_workstation_payload(_section(raw, "runtime"))
    experiment = _section(raw, "experiment")
    return SceptreConfig(
        source_path=source,
        artifact_root=str(experiment["artifact_root"]),
        contract_hash=canonical_hash(expected),
        source_provenance_items=tuple(observed_provenance.items()),
    )


def _section(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"SCEPTRE {name} section is not a mapping.")
    return dict(value)


def _reject_pending(value: object) -> None:
    if isinstance(value, str) and "__PENDING" in value:
        raise ProtocolError("SCEPTRE config contains an unresolved placeholder.")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_pending(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_pending(item)


__all__ = (
    "CONFIG_TOP_LEVEL",
    "SceptreConfig",
    "build_planned_config",
    "experiment_payload",
    "frozen_config_contract_payload",
    "load_config",
    "repository_root",
    "source_provenance_payload",
)
