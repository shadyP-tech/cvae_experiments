"""Complete-policy OOF seals cannot masquerade as primitive diagnostics."""
from dataclasses import replace

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.safe_winner_router_v19.candidate_prediction import POLICY_ARM_ID
from midogpp_thesis.cvae.routing.safe_winner_router_v19.composition import build_baseline_composite, build_exact_u_composite
from test_harp_v19_science_foundation import make_menu, seal


def test_nonbaseline_complete_policy_requires_its_winner_evidence():
    record = seal(build_exact_u_composite(make_menu()))
    with pytest.raises(ProtocolError, match="complete policy lacks winner evidence"):
        replace(record, requested_arm_id=POLICY_ARM_ID)


@pytest.mark.parametrize("enabled,reason", [
    (True, "NO_PREDICTION_CHANGING_CANDIDATE"),
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
