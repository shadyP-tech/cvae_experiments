"""Complete-policy OOF seals cannot masquerade as primitive diagnostics."""
from dataclasses import replace

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.correction_mass_router_v21.candidate_prediction import POLICY_ARM_ID
from midogpp_thesis.cvae.routing.correction_mass_router_v21.composition import build_baseline_composite, build_exact_u_composite
from test_harp_v21_science_foundation import make_menu, seal


def test_nonbaseline_complete_policy_requires_its_winner_evidence():
    record = seal(build_exact_u_composite(make_menu()))
    with pytest.raises(ProtocolError, match="complete policy lacks winner evidence"):
        replace(record, requested_arm_id=POLICY_ARM_ID)


@pytest.mark.parametrize("enabled,reason", [
    (True, "NO_FEASIBLE_POSITIVE_GAIN_CANDIDATE"),
    (False, "NO_SAFE_INNER_OOF_POLICY"),
])
def test_explicit_empty_complete_policy_preserves_baseline(enabled, reason):
    record = replace(seal(build_baseline_composite(make_menu())),
        requested_arm_id=POLICY_ARM_ID, policy_enabled=enabled, fallback_reason=reason)
    assert not record.composite.route_selected
    assert record.public_payload()["winner_composite_hash"] is None


@pytest.mark.parametrize("reason", [None, "MISSING_COMPLETE_WINNER_GATE", "WINNER_GATE_BELOW_THRESHOLD"])
def test_missing_winner_cannot_be_hidden_behind_a_gate_fallback(reason):
    with pytest.raises(ProtocolError, match="complete policy lacks winner evidence"):
        replace(seal(build_baseline_composite(make_menu())),
            requested_arm_id=POLICY_ARM_ID, fallback_reason=reason)


def test_direct_primitive_diagnostics_do_not_claim_complete_policy_evidence():
    record = seal(build_exact_u_composite(make_menu()))
    assert record.requested_arm_id != POLICY_ARM_ID
    assert record.composite.route_selected
    assert record.public_payload()["winner_gate_prediction_payload"] is None


@pytest.mark.parametrize('admitted,reason', [
    (True,'NO_FEASIBLE_POSITIVE_GAIN_CANDIDATE'),
    (False,'SOURCE_OOF_ADMISSION_NO_NONZERO_SAFE_OOF_COVERAGE'),
    (False,'FINAL_REFIT_HAS_NO_NONZERO_POLICY'),
])
def test_empty_target_winner_retains_exact_baseline_without_source_only_fields(admitted,reason):
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.records import RouteDecision
    from midogpp_thesis.cvae.routing.correction_mass_router_v21 import SurfaceRole
    menu=make_menu('target',role=SurfaceRole.TARGET_EVALUATION)
    decision=RouteDecision(build_baseline_composite(menu),POLICY_ARM_ID,0.,.9,'a'*64,admitted,reason)
    assert decision.composite.probability_hex==menu.baseline_probability_hex
    assert decision.public_payload()['winner_composite_hash'] is None
    with pytest.raises(ProtocolError,match='complete policy lacks winner evidence'):
        replace(decision,fallback_reason='MISSING_COMPLETE_WINNER_GATE')


def test_nonadmitted_target_cannot_route_a_primitive_without_winner_fields():
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.records import RouteDecision
    with pytest.raises(ProtocolError,match='route decision is malformed'):
        RouteDecision(build_exact_u_composite(make_menu()),'U_FULL',1.,0.,'a'*64,False)
