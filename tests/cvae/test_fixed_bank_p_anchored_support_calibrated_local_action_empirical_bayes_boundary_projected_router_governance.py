from __future__ import annotations

from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.config import (
    load_support_calibrated_local_action_empirical_bayes_boundary_projected_router_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.execution_admission import (
    BLOCKED_MESSAGE,
    assert_execution_authorized,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.runner import (
    run_support_calibrated_local_action_empirical_bayes_boundary_projected_router,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.source_fence import (
    validate_source_fence,
)
from midogpp_thesis.cvae.protocol import ProtocolError


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_support_calibrated_"
    "local_action_empirical_bayes_boundary_projected_router_v1.yaml"
)


def test_scale_bp_source_fence_accepts_only_independent_package_source() -> None:
    receipt = validate_source_fence()
    assert receipt.member_count >= 10
    assert receipt.import_count > 0


def test_scale_bp_source_fence_rejects_sibling_diagnostic_import(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text(
        "from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_"
        "directional_signed_utility_router import routing\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="sibling diagnostics"):
        validate_source_fence(tmp_path)


def test_scale_bp_source_fence_allows_owned_subpackage_relative_import(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "influence"
    nested.mkdir()
    (tmp_path / "identity.py").write_text("VALUE = 1\n", encoding="utf-8")
    (nested / "contracts.py").write_text(
        "from ..identity import VALUE\n",
        encoding="utf-8",
    )
    receipt = validate_source_fence(tmp_path)
    assert receipt.member_count == 2


def test_scale_bp_source_fence_rejects_predecessor_artifact_path(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text(
        "ROOT = 'artifacts/midogpp/90_oracles_and_diagnostics/old/v1'\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="forbidden predecessor artifact path"):
        validate_source_fence(tmp_path)


@pytest.mark.parametrize("predecessor", ["pdcaps", "pdsur", "pcsi", "cbpupr"])
def test_scale_bp_source_fence_rejects_named_stage90_artifact_paths(
    tmp_path: Path,
    predecessor: str,
) -> None:
    (tmp_path / "bad.py").write_text(
        f"ROOT = 'artifact://archive/{predecessor}/v1'\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="forbidden predecessor artifact path"):
        validate_source_fence(tmp_path)


def test_scale_bp_direct_gate_refuses_before_path_mutation(tmp_path: Path) -> None:
    config = (
        load_support_calibrated_local_action_empirical_bayes_boundary_projected_router_config(
            CONFIG
        )
    )
    output = tmp_path / "must-not-exist-output"
    scratch = tmp_path / "must-not-exist-scratch"

    with pytest.raises(ProtocolError, match="planned-only"):
        assert_execution_authorized(
            config,
            artifact_root=output,
            scratch_root=scratch,
        )
    assert not output.exists()
    assert not scratch.exists()


def test_scale_bp_runner_refuses_without_output_scratch_or_lock(tmp_path: Path) -> None:
    config = (
        load_support_calibrated_local_action_empirical_bayes_boundary_projected_router_config(
            CONFIG
        )
    )
    output = tmp_path / "output"
    scratch = tmp_path / "scratch"

    with pytest.raises(ProtocolError, match="SCALE-BP v1 execution is not authorized"):
        run_support_calibrated_local_action_empirical_bayes_boundary_projected_router(
            config,
            artifact_root=output,
            scratch_root=scratch,
        )
    assert BLOCKED_MESSAGE.startswith("SCALE-BP v1 execution is not authorized")
    assert not output.exists()
    assert not scratch.exists()
    assert not (tmp_path / "lock").exists()
