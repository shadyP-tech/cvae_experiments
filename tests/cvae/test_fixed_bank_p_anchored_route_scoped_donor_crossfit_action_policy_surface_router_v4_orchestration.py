from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.contracts import (
    FavorableUtility,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.target_local_runtime import (
    POSTERIOR_CONTROL_IDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.admission import (
    PseudoPolicyEvidence,
    build_outer_admission,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4 import (
    runner,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.run_state import (
    write_run_state,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.validation_records import (
    _validate_outer_admissions,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
)
from midogpp_thesis.cvae.protocol import ProtocolError


_CONFIG_HASH = "a" * 64


def _admission(outer_center: str):
    donors = tuple(center for center in CENTERS if center != outer_center)
    evidence = tuple(
        PseudoPolicyEvidence(
            outer_center,
            donor,
            FavorableUtility(0.25, 0.01 * (index + 1), 0.02 * (index + 1)),
            FavorableUtility(
                0.01 * (index + 1),
                0.02 * (index + 1),
                0.03 * (index + 1),
            ),
            True,
            True,
            1.0,
            0.05,
            FavorableUtility(
                0.01 * (index + 1),
                0.02 * (index + 1),
                0.03 * (index + 1),
            ),
            True,
            True,
            0.10,
        )
        for index, donor in enumerate(donors)
    )
    admission = build_outer_admission(outer_center, evidence)
    assert admission.passed is False
    assert admission.statistics_by_name["bacc_spearman"].value is None
    return admission


def _persisted_admission_rows() -> list[dict[str, object]]:
    admissions = {center: _admission(center) for center in CENTERS}
    return [
        {
            "outer_center": center,
            "posterior_control_id": control,
            "admission": admissions[center].to_payload(),
        }
        for center in CENTERS
        for control in POSTERIOR_CONTROL_IDS
    ]


def test_v4_fresh_validator_replays_nullable_admissions_in_h_major_order() -> None:
    rows = _persisted_admission_rows()
    reconstructed = _validate_outer_admissions(tuple(rows), centers=CENTERS)

    assert tuple(reconstructed) == tuple(
        (control, center)
        for center in CENTERS
        for control in POSTERIOR_CONTROL_IDS
    )
    assert all(
        admission.statistics_by_name["bacc_spearman"].to_payload()
        == {
            "name": "bacc_spearman",
            "value": None,
            "defined": False,
            "undefined_reason": "CONSTANT_RANK_INPUT",
        }
        for admission in reconstructed.values()
    )

    poison = deepcopy(rows)
    statistics = poison[0]["admission"]["statistics"]  # type: ignore[index]
    bacc = next(row for row in statistics if row["name"] == "bacc_spearman")
    bacc["value"] = 0.0
    with pytest.raises(ProtocolError, match="undefined statistic"):
        _validate_outer_admissions(tuple(poison), centers=CENTERS)


def test_v4_failed_or_complete_state_exhausts_the_one_shot_identity(
    tmp_path: Path,
) -> None:
    (tmp_path / "reports").mkdir()
    write_run_state(
        tmp_path,
        config_hash=_CONFIG_HASH,
        status="RUNNING",
        phase="BEGIN",
    )
    failed = write_run_state(
        tmp_path,
        config_hash=_CONFIG_HASH,
        status="FAILED",
        phase="WORKSTATION_PREFLIGHT",
        error_class="ProtocolError",
        error="fixture failure",
    )
    assert failed["authorization_exhausted"] is True
    assert failed["cross_run_recovery_allowed"] is False
    with pytest.raises(ProtocolError, match="prior run state"):
        write_run_state(
            tmp_path,
            config_hash=_CONFIG_HASH,
            status="RUNNING",
            phase="WORKSTATION_PREFLIGHT",
        )


def test_v4_runner_keeps_terminal_labels_after_the_durable_barrier() -> None:
    source = inspect.getsource(
        runner.run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4
    )
    durable = source.index("persist_durable_attestation")
    revalidated = source.index("verify_durable_preterminal_attestation")
    terminal = source.index("begin_terminal_evaluation")
    labels = source.index("open_terminal_center_labels")
    assert durable < revalidated < terminal < labels
    assert "identity_admissions=outer.identity_admissions" in source
    assert "cyclic_admissions=outer.cyclic_admissions" in source

