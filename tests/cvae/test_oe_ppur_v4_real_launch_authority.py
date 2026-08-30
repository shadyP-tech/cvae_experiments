from __future__ import annotations

import json
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.execution.authority import (
    build_execution_launch_authority,
    load_execution_launch_authority,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.hashing import (
    canonical_bytes,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.identity import (
    LAUNCH_AUTHORIZATION_PHRASE,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _authority():
    return build_execution_launch_authority(
        authorization_phrase=LAUNCH_AUTHORIZATION_PHRASE,
        workspace_snapshot_sha256="1" * 64,
        workspace_plan_sha256="2" * 64,
        authorization_amendment_sha256="3" * 64,
        final_envelope_sha256="4" * 64,
        seven_input_inventory_sha256="5" * 64,
        topology_contract_sha256="6" * 64,
        scientific_seals_sha256="7" * 64,
        lifecycle_seal_sha256="8" * 64,
        workstation_topology_sha256="9" * 64,
        preflight_receipt_sha256="a" * 64,
        authorization_nonce="b" * 64,
    )


def test_execution_launch_authority_has_canonical_round_trip(tmp_path: Path) -> None:
    authority = _authority()
    path = tmp_path / "launch-authority.json"
    path.write_bytes(authority.canonical_file_bytes())

    loaded = load_execution_launch_authority(path)

    assert loaded.authority == authority
    assert loaded.authority.authority_hash == authority.authority_hash
    assert loaded.file_sha256 != authority.authority_hash
    assert path.read_bytes() == canonical_bytes(authority.to_payload()) + b"\n"


def test_execution_launch_authority_rejects_hash_bound_field_drift(
    tmp_path: Path,
) -> None:
    authority = _authority()
    payload = json.loads(authority.canonical_file_bytes())
    payload["workspace_snapshot_sha256"] = "c" * 64
    path = tmp_path / "launch-authority.json"
    path.write_bytes(canonical_bytes(payload) + b"\n")

    with pytest.raises(ProtocolError, match="semantics drifted"):
        load_execution_launch_authority(path)


def test_execution_launch_authority_rejects_noncanonical_bytes(
    tmp_path: Path,
) -> None:
    authority = _authority()
    path = tmp_path / "launch-authority.json"
    path.write_text(json.dumps(authority.to_payload(), indent=2), encoding="utf-8")

    with pytest.raises(ProtocolError, match="bytes are not canonical"):
        load_execution_launch_authority(path)
