from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.config import (
    build_workspace_sealed_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.execution.inputs import (
    ResolvedDirectInput,
    hash_resolved_input_locations,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.execution.replay_contract import (
    ReplayAdmissionContract,
    ReplayAuthorityBinding,
    build_replay_admission_contract,
    require_replay_admission_contract,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.identity import (
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPECTED_INPUT_KINDS,
)
from midogpp_thesis.cvae.protocol import ProtocolError


PACKAGE_ROOT = (
    Path(__file__).parents[2]
    / "src/midogpp_thesis/cvae/diagnostics"
    / "fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4"
)


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), path.as_posix())
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    return tuple(imports)


def _bindings(tmp_path: Path) -> tuple[ResolvedDirectInput, ...]:
    root = tmp_path / "direct-inputs"
    root.mkdir()
    rows: list[ResolvedDirectInput] = []
    for ordinal, (role, artifact_id, kind) in enumerate(
        zip(
            DIRECT_INPUT_ROLES,
            DIRECT_INPUT_ARTIFACT_IDS,
            EXPECTED_INPUT_KINDS,
            strict=True,
        ),
        start=1,
    ):
        path = root / f"input-{ordinal}"
        if kind == "directory":
            path.mkdir()
            relative = "member.bin"
        else:
            path.write_bytes(f"input-{ordinal}\n".encode())
            relative = path.name
        rows.append(
            ResolvedDirectInput(
                ordinal=ordinal,
                role=role,
                artifact_id=artifact_id,
                kind=kind,
                path=path,
                member_hashes=((relative, hashlib.sha256(relative.encode()).hexdigest()),),
            )
        )
    return tuple(rows)


def _contract(tmp_path: Path, **overrides: object) -> ReplayAdmissionContract:
    repository = tmp_path / "repository"
    repository.mkdir()
    preflight = tmp_path / "preflight.json"
    preflight.write_text("{}\n", encoding="utf-8")
    artifact = repository / "artifacts" / "router" / "v4"
    scratch = tmp_path / "scratch"
    bindings = _bindings(tmp_path)
    values: dict[str, object] = {
        "sealed_config": build_workspace_sealed_config(
            workspace_plan_sha256="1" * 64,
            authorization_amendment_sha256="2" * 64,
        ),
        "input_bindings": bindings,
        "repository_root": repository,
        "preflight_path": preflight,
        "artifact_root": artifact,
        "scratch_root": scratch,
        "lease_path": artifact.parent / ".lease",
        "amendment_parent": repository / "contracts/oe_ppur_v4",
        "resolved_config_path": artifact / "config.resolved.yaml",
        "input_manifest_path": artifact / "provenance/input_artifacts.json",
        "final_envelope_path": (
            artifact / "preparation/final_authorization_envelope.json"
        ),
        "workspace_snapshot_sha256": "3" * 64,
        "workspace_plan_sha256": "1" * 64,
        "authorization_amendment_sha256": "2" * 64,
        "final_envelope_sha256": "4" * 64,
        "seven_input_inventory_sha256": "5" * 64,
        "topology_contract_sha256": "6" * 64,
        "scientific_seals_sha256": "7" * 64,
        "scientific_source_seal_sha256": "8" * 64,
        "lifecycle_seal_sha256": "9" * 64,
        "workstation_topology_sha256": "a" * 64,
        "preflight_file_sha256": "b" * 64,
        "resolved_input_contract_sha256": hash_resolved_input_locations(bindings),
        "envelope_admission_sha256": "c" * 64,
        "input_manifest_file_sha256": "d" * 64,
        "sealed_replay_receipt_hash": "e" * 64,
    }
    values.update(overrides)
    return build_replay_admission_contract(**values)  # type: ignore[arg-type]


def test_replay_and_admission_depend_only_on_shared_contract_layer() -> None:
    run_admission_imports = _imports(PACKAGE_ROOT / "run_admission.py")
    replay_imports = _imports(PACKAGE_ROOT / "execution/sealed_replay.py")
    contract_imports = _imports(PACKAGE_ROOT / "execution/replay_contract.py")

    assert not any(name.endswith("execution.sealed_replay") for name in run_admission_imports)
    assert not any(name.endswith("run_admission") for name in replay_imports)
    assert not any(
        name.endswith(("execution.sealed_replay", "run_admission"))
        for name in contract_imports
    )
    assert any(name.endswith("execution.replay_contract") for name in run_admission_imports)
    assert any(name.endswith("replay_contract") for name in replay_imports)


def test_replay_admission_contract_rejects_lineage_drift(tmp_path: Path) -> None:
    bindings = _bindings(tmp_path)
    # Use a separate root because the helper deliberately creates all fixtures.
    drift_root = tmp_path / "drift"
    drift_root.mkdir()
    with pytest.raises(ProtocolError, match="contract drifted"):
        _contract(
            drift_root,
            resolved_input_contract_sha256=hash_resolved_input_locations(bindings),
        )


def test_replay_contract_is_factory_gated_and_exactly_typed(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    assert require_replay_admission_contract(contract) is contract
    carrier = SimpleNamespace(
        admission_contract=contract,
        receipt_hash=contract.authority.sealed_replay_receipt_hash,
    )
    assert require_replay_admission_contract(carrier) is contract

    carrier.receipt_hash = "f" * 64
    with pytest.raises(ProtocolError, match="run admission is untyped"):
        require_replay_admission_contract(carrier)

    with pytest.raises(ProtocolError, match="authority binding is untyped"):
        ReplayAuthorityBinding(
            workspace_snapshot_sha256="1" * 64,
            workspace_plan_sha256="1" * 64,
            authorization_amendment_sha256="1" * 64,
            final_envelope_sha256="1" * 64,
            seven_input_inventory_sha256="1" * 64,
            topology_contract_sha256="1" * 64,
            scientific_seals_sha256="1" * 64,
            scientific_source_seal_sha256="1" * 64,
            lifecycle_seal_sha256="1" * 64,
            workstation_topology_sha256="1" * 64,
            preflight_file_sha256="1" * 64,
            resolved_input_contract_sha256="1" * 64,
            envelope_admission_sha256="1" * 64,
            input_manifest_file_sha256="1" * 64,
            sealed_replay_receipt_hash="1" * 64,
        )

    with pytest.raises(ProtocolError, match="run admission is untyped"):
        require_replay_admission_contract(object())
