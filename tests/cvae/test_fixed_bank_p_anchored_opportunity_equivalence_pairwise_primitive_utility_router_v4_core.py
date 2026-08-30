from __future__ import annotations

import ast
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.admission import (
    LaunchAuthority,
    SealedEnvelopeAdmission,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.config import (
    PLANNED_STATE,
    SEALED_STATE,
    build_planned_config,
    build_workspace_sealed_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.identity import (
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPECTED_INPUT_KINDS,
    EXPERIMENT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.input_contract import (
    build_authorized_seven_input_contract,
    build_planned_seven_input_contract,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.runner import (
    run_oe_ppur_v4,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _sha(character: str) -> str:
    return character * 64


def _admission(
    tmp_path: Path,
) -> tuple[SealedEnvelopeAdmission, LaunchAuthority]:
    config = build_workspace_sealed_config(
        workspace_plan_sha256=_sha("1"),
        authorization_amendment_sha256=_sha("2"),
    )
    admission = SealedEnvelopeAdmission(
        config=config,
        workspace_snapshot_sha256=_sha("3"),
        workspace_plan_sha256=_sha("1"),
        authorization_amendment_sha256=_sha("2"),
        final_envelope_sha256=_sha("4"),
        direct_input_artifact_ids=DIRECT_INPUT_ARTIFACT_IDS,
        resolved_paths=tuple(
            tmp_path / f"direct-input-{ordinal}" for ordinal in range(1, 8)
        ),
        topology_contract_sha256=_sha("5"),
    )
    authority = LaunchAuthority(
        experiment_id=EXPERIMENT_ID,
        workspace_plan_sha256=_sha("1"),
        authorization_amendment_sha256=_sha("2"),
        final_envelope_sha256=_sha("4"),
        authorization_phrase_sha256=_sha("6"),
    )
    return admission, authority


def test_planned_and_sealed_configs_keep_launch_authority_absent() -> None:
    planned = build_planned_config()
    sealed = build_workspace_sealed_config(
        workspace_plan_sha256=_sha("1"),
        authorization_amendment_sha256=_sha("2"),
    )
    assert planned.authorization_state == PLANNED_STATE
    assert planned.execution_amendment_issued is False
    assert sealed.authorization_state == SEALED_STATE
    assert sealed.execution_amendment_issued is True
    assert planned.launch_authorized is sealed.launch_authorized is False


def test_exact_seven_input_order_and_kinds_are_frozen() -> None:
    planned = build_planned_seven_input_contract()
    sealed = build_authorized_seven_input_contract()
    assert tuple(row.ordinal for row in planned.ordered_inputs) == tuple(range(1, 8))
    assert tuple(row.role for row in planned.ordered_inputs) == DIRECT_INPUT_ROLES
    assert tuple(row.artifact_id for row in planned.ordered_inputs) == (
        DIRECT_INPUT_ARTIFACT_IDS
    )
    assert tuple(row.kind for row in planned.ordered_inputs) == EXPECTED_INPUT_KINDS
    assert planned.ordered_inputs[-1].issued is False
    assert sealed.ordered_inputs[-1].issued is True


def test_sealed_admission_rejects_v3_operational_paths(tmp_path: Path) -> None:
    config = build_workspace_sealed_config(
        workspace_plan_sha256=_sha("1"),
        authorization_amendment_sha256=_sha("2"),
    )
    paths = tuple(tmp_path / f"direct-input-{ordinal}" for ordinal in range(1, 8))
    paths = (*paths[:-1], tmp_path / "contracts/oe_ppur_v3/amendment.json")
    with pytest.raises(ProtocolError, match="predecessor operational path"):
        SealedEnvelopeAdmission(
            config=config,
            workspace_snapshot_sha256=_sha("3"),
            workspace_plan_sha256=_sha("1"),
            authorization_amendment_sha256=_sha("2"),
            final_envelope_sha256=_sha("4"),
            direct_input_artifact_ids=DIRECT_INPUT_ARTIFACT_IDS,
            resolved_paths=paths,
            topology_contract_sha256=_sha("5"),
        )


def test_runner_stays_closed_even_with_typed_preparation_authority(
    tmp_path: Path,
) -> None:
    admission, authority = _admission(tmp_path)
    with pytest.raises(ProtocolError, match="launch edge remains closed"):
        run_oe_ppur_v4(
            admission.config,
            admission=admission,
            launch_authority=authority,
        )


def test_v4_lifecycle_has_no_v3_package_imports() -> None:
    repository = Path(__file__).resolve().parents[2]
    package_roots = (
        repository
        / "src/midogpp_thesis/cvae/diagnostics/fixed_bank_p_anchored_"
        "opportunity_equivalence_pairwise_primitive_utility_router_v4",
        repository
        / "src/midogpp_thesis/cvae/diagnostics/oe_ppur_v4_preparation",
    )
    forbidden = (
        "fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_"
        "utility_router_v3"
    )
    imported_modules: list[str] = []
    for source in (path for root in package_roots for path in root.glob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)
    assert all(forbidden not in module for module in imported_modules)
