from __future__ import annotations

from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v16.config import (
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v16.runner import (
    HARP_V16_RUN_CONFIRMATION_TOKEN,
    dry_run_harp_stage90_v16,
    inspect_harp_stage90_v16,
    run_harp_stage90_v16,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v16.source_seal import (
    source_members,
    source_snapshot_identity,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.harp_v16_execution.physical import (
    build_physical_plan,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_harp_router_v16.yaml"
)


def test_execution_confirmation_precedes_config_and_path_access() -> None:
    with pytest.raises(ProtocolError, match="exact confirmation token"):
        run_harp_stage90_v16(
            object(),
            artifact_root="artifact://must-not-be-resolved",
            confirmation_token=None,
        )

    with pytest.raises(ProtocolError, match="typed configuration"):
        run_harp_stage90_v16(
            object(),
            artifact_root="artifact://must-not-be-resolved",
            confirmation_token=HARP_V16_RUN_CONFIRMATION_TOKEN,
        )


def test_planned_inspection_and_dry_run_are_path_free() -> None:
    config = load_config(CONFIG)
    impossible = ROOT / "does-not-exist" / "v16-output"

    inspection = inspect_harp_stage90_v16(config)
    dry_run = dry_run_harp_stage90_v16(config, artifact_root=impossible)

    assert inspection["status"] == "PLANNED_NEEDS_SEPARATE_EXECUTION_AMENDMENT"
    assert inspection["paths_resolved"] is False
    assert inspection["authorization_probed"] is False
    assert dry_run["status"] == "NEEDS_SEPARATE_EXECUTION_AMENDMENT"
    assert dry_run["paths_resolved"] is False
    assert dry_run["artifact_root_argument_recorded"] is False
    assert dry_run["filesystem_mutations"] == 0
    assert not impossible.exists()


def test_joint_physical_plan_reuses_each_fit_across_support_and_target() -> None:
    plan = build_physical_plan()

    assert plan["classifier_task_count"] == 81
    assert plan["classifier_fit_count"] == 810
    assert plan["support_context_count"] == 9
    assert plan["target_context_count"] == 9
    assert plan["joint_support_target_prediction"] is True
    assert plan["classifier_fit_reused_across_support_and_target"] is True


def test_source_snapshot_closure_is_predecessor_free() -> None:
    identity = source_snapshot_identity(ROOT)
    members = tuple(
        path.relative_to(ROOT / "src").as_posix() for path in source_members(ROOT)
    )

    assert identity["source_snapshot_member_count"] == len(members)
    assert any("harp_v16_execution/production.py" in member for member in members)
    assert any(
        "hierarchical_support_action_risk_router_v16/policy.py" in member
        for member in members
    )
    assert not any(
        f"fixed_bank_harp_router_v{version}/" in member
        or f"harp_v{version}_execution/" in member
        for version in range(1, 16)
        for member in members
    )
