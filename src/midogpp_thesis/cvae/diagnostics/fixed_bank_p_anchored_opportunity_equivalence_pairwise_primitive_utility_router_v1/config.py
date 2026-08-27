"""Strict path-free configuration for planned OE-PPUR v1."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ...protocol import ProtocolError
from .contracts import claim_boundary_payload, direct_input_policy_payload
from .execution.workstation import workstation_payload
from .hashing import canonical_hash
from .identity import (
    CLAIM_SCOPE,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    WORKSPACE_STATUS,
)
from .protocol import frozen_protocol_payload, validate_protocol_payload
from .source_fence import build_source_fence_receipt


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


def experiment_payload() -> dict[str, object]:
    return {
        "schema_version": "oe_ppur_v1_planned_config_v1",
        "id": EXPERIMENT_ID,
        "name": EXPERIMENT_NAME,
        "stage": "90_oracles_and_diagnostics",
        "status": WORKSPACE_STATUS,
        "claim_scope": CLAIM_SCOPE,
        "publication_status": PUBLICATION_STATUS,
        "fresh_evidence": False,
        "execution_authorized": False,
        "implementation_authorizes_execution": False,
        "consumed_test_reuse_authorized": False,
    }


def source_provenance_payload() -> dict[str, object]:
    """Recompute the disjoint adapter/core source seal without path literals."""

    receipt = build_source_fence_receipt()
    return {
        "schema_version": "oe_ppur_v1_source_provenance_config_v1",
        "adapter_scope_role": receipt.adapter.role,
        "adapter_member_count": receipt.adapter.member_count,
        "adapter_tree_sha256": receipt.adapter_tree_sha256,
        "adapter_receipt_hash": receipt.adapter.receipt_hash,
        "core_scope_role": receipt.core.role,
        "core_member_count": receipt.core.member_count,
        "core_tree_sha256": receipt.core_tree_sha256,
        "core_receipt_hash": receipt.core.receipt_hash,
        "source_scopes_are_disjoint": True,
        "combined_member_count": receipt.member_count,
        "combined_source_seal_hash": receipt.combined_source_seal_hash,
        "combined_source_receipt_hash": receipt.receipt_hash,
        "recompute_and_exact_match_on_load": True,
    }


def _config_contract_payload(
    source_provenance: Mapping[str, object],
) -> dict[str, object]:
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
    return _config_contract_payload(source_provenance_payload())


@dataclass(frozen=True, slots=True)
class RouterConfig:
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


def build_planned_config() -> RouterConfig:
    payload = frozen_config_contract_payload()
    return RouterConfig(
        source_path=None,
        artifact_root=str(payload["experiment"]["artifact_root"]),
        contract_hash=canonical_hash(payload),
        source_provenance_items=tuple(payload["source_provenance"].items()),
    )


def load_config(path: str | Path) -> RouterConfig:
    """Load only the exact path-free planned contract."""

    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("Cannot read OE-PPUR v1 config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != set(CONFIG_TOP_LEVEL):
        raise ProtocolError("OE-PPUR top-level config drifted.")
    _reject_pending(raw)
    observed_provenance = source_provenance_payload()
    if _section(raw, "source_provenance") != observed_provenance:
        raise ProtocolError("OE-PPUR combined adapter/core source seal drifted.")
    expected = _config_contract_payload(observed_provenance)
    if dict(raw) != expected:
        raise ProtocolError("OE-PPUR planned config contract drifted.")
    validate_protocol_payload(_section(raw, "protocol"))
    experiment = _section(raw, "experiment")
    return RouterConfig(
        source_path=source,
        artifact_root=str(experiment["artifact_root"]),
        contract_hash=canonical_hash(expected),
        source_provenance_items=tuple(observed_provenance.items()),
    )


def _section(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"OE-PPUR {name} section is not a mapping.")
    return dict(value)


def _reject_pending(value: object) -> None:
    if isinstance(value, str) and "__PENDING" in value:
        raise ProtocolError("OE-PPUR config contains an unresolved placeholder.")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_pending(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_pending(item)


__all__ = (
    "CONFIG_TOP_LEVEL",
    "RouterConfig",
    "build_planned_config",
    "experiment_payload",
    "frozen_config_contract_payload",
    "load_config",
    "source_provenance_payload",
)
