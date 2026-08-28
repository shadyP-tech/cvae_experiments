from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.config import (
    PLANNED_STATE,
    build_planned_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.execution.inputs import (
    build_planned_seven_input_contract,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.identity import (
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    SOURCE_SUPERVISION_ARTIFACT_ID,
    SOURCE_SUPERVISION_REQUIRED_MEMBERS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.runner import (
    inspect_planned_router,
    run_oe_ppur_v3,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def test_exact_seven_input_planned_contract_is_path_free() -> None:
    contract = build_planned_seven_input_contract()
    payload = contract.to_payload()

    assert contract.input_count == 7
    assert tuple(row.role for row in contract.ordered_inputs) == DIRECT_INPUT_ROLES
    assert tuple(row.artifact_id for row in contract.ordered_inputs) == DIRECT_INPUT_ARTIFACT_IDS
    assert contract.ordered_inputs[2].artifact_id == SOURCE_SUPERVISION_ARTIFACT_ID
    assert contract.ordered_inputs[2].required_members == SOURCE_SUPERVISION_REQUIRED_MEMBERS
    assert contract.ordered_inputs[2].issued is False
    assert contract.source_supervision_materialized is False
    assert contract.ordered_inputs[6].artifact_id == AUTHORIZATION_AMENDMENT_ARTIFACT_ID
    assert contract.ordered_inputs[6].issued is False
    assert contract.amendment_issued is False
    assert contract.execution_authorized is False
    assert payload["paths_present"] is False


def test_planned_config_has_no_source_hash_or_amendment() -> None:
    config = build_planned_config()
    payload = config.to_payload()

    assert config.authorization_state == PLANNED_STATE
    assert config.execution_authorized is False
    assert config.source_supervision_content_sha256 is None
    assert config.source_supervision_row_order_sha256 is None
    assert config.authorization_amendment_sha256 is None
    assert payload["paths_present"] is False

    with pytest.raises(ProtocolError, match="planned config identity drifted"):
        replace(config, execution_authorized=True)


def test_planned_runner_rejects_before_source_or_path_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.runner as runner

    monkeypatch.setattr(
        runner,
        "build_source_seal",
        lambda: pytest.fail("source seal must not be opened by planned run"),
    )
    with pytest.raises(ProtocolError, match="direct input #7 is absent"):
        run_oe_ppur_v3(
            build_planned_config(),
            artifact_root=Path("relative-output-must-not-be-inspected"),
            scratch_root=Path("relative-scratch-must-not-be-inspected"),
        )


def test_inspection_is_hardware_free_and_non_mutating() -> None:
    payload = inspect_planned_router(build_planned_config())
    assert payload["direct_input_count"] == 7
    assert payload["source_supervision_direct_input_ordinal"] == 3
    assert payload["authorization_amendment_input_ordinal"] == 7
    assert payload["authorization_amendment_issued"] is False
    assert payload["artifact_paths_present"] is False
    assert payload["hardware_probed"] is False
    assert payload["filesystem_mutation_performed"] is False
    assert payload["experiment_launched"] is False
    assert payload["end_to_end_scientific_execution_implemented"] is True
    assert payload["canonical_terminal_evaluator_implemented"] is True
