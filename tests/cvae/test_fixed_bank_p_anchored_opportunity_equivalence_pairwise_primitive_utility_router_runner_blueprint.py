from __future__ import annotations

from dataclasses import replace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.config import (
    build_planned_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.runner import (
    inspect_runner_blueprint,
    run_planned_router,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.runner_blueprint import (
    EXECUTABLE_SUCCESSOR_INPUT_ROLES,
    RunnerBlueprint,
    canonical_runner_phases,
)
from midogpp_thesis.cvae.protocol import ProtocolError


class _PoisonPath:
    def __str__(self) -> str:
        raise AssertionError("runner coerced an unauthorized path")

    def __fspath__(self) -> str:
        raise AssertionError("runner resolved an unauthorized path")


def test_runner_blueprint_is_complete_ordered_and_non_authorizing() -> None:
    blueprint = inspect_runner_blueprint(build_planned_config())

    assert isinstance(blueprint, RunnerBlueprint)
    assert blueprint.phases == canonical_runner_phases()
    assert blueprint.execution_authorized is False
    assert blueprint.phases[0].phase == "SOURCE_AND_CONFIG_PREFLIGHT"
    assert blueprint.phases[0].authorization_required is False
    assert all(row.authorization_required for row in blueprint.phases[1:])
    assert tuple(
        row.phase
        for row in blueprint.phases
        if row.process_topology == "TWO_PERSISTENT_GPU_WORKERS"
    ) == ("PHYSICAL_SURFACE_SEALED",)
    assert tuple(
        row.phase
        for row in blueprint.phases
        if row.process_topology == "FOUR_SPAWN_CPU_OUTER_WORKERS"
    ) == ("OUTER_FOLDS_COMPLETE",)
    terminal = tuple(
        row for row in blueprint.phases if row.label_access != "CLOSED"
    )
    assert tuple(row.phase for row in terminal) == ("TERMINAL_AGGREGATES_SCORED",)
    assert blueprint.to_payload()["failure_after_lease"] == "FAILED_EXHAUSTED"
    assert blueprint.to_payload()["cross_run_recovery_allowed"] is False
    assert tuple(blueprint.to_payload()["required_successor_input_roles"]) == (
        EXECUTABLE_SUCCESSOR_INPUT_ROLES
    )
    assert blueprint.to_payload()["required_successor_input_count"] == 6
    assert blueprint.to_payload()["preterminal_input_lineage_type"] == (
        "PreterminalInputLineage"
    )
    assert blueprint.to_payload()["raw_preterminal_hash_admission_allowed"] is False
    assert blueprint.to_payload()[
        "parsed_probability_matrix_science_receipt_required_by_successor"
    ] is True
    assert len(blueprint.blueprint_hash) == 64


def test_blueprint_cannot_be_changed_into_an_authorization() -> None:
    blueprint = inspect_runner_blueprint(build_planned_config())

    with pytest.raises(ProtocolError, match="blueprint topology drifted"):
        replace(blueprint, execution_authorized=True)


def test_public_v1_runner_still_rejects_before_path_capability() -> None:
    with pytest.raises(ProtocolError, match="execution is not authorized"):
        run_planned_router(
            build_planned_config(),
            artifact_root=_PoisonPath(),
            scratch_root=_PoisonPath(),
        )
